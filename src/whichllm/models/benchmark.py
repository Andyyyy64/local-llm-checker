"""Benchmark data fetcher: Chatbot Arena ELO + Open LLM Leaderboard."""

from __future__ import annotations

import io
import json
import logging
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "whichllm"
BENCHMARK_CACHE = CACHE_DIR / "benchmark.json"
DEFAULT_TTL_SECONDS = 24 * 3600  # 24 hours

# --- Data source URLs ---
ARENA_ROWS_URL = "https://datasets-server.huggingface.co/rows"
ARENA_DATASET = "mathewhe/chatbot-arena-elo"

LEADERBOARD_PARQUET_URL = (
    "https://huggingface.co/api/datasets/open-llm-leaderboard/contents"
    "/parquet/default/train/0.parquet"
)
LEADERBOARD_ROWS_URL = "https://datasets-server.huggingface.co/rows"
LEADERBOARD_DATASET = "open-llm-leaderboard/contents"

# --- Arena ELO normalization ---
# Open-source ELO range: ~1030 (worst) to ~1424 (best). Arena is frozen
# 2025-07-17 (no new models added) so the leaderboard cannot reflect any
# 2025-Q3+ release; we cap the normalized output at 82 so newer benchmark
# sources (AA Index / LiveBench, which can reach 95+) decisively win on
# conflict.
_ARENA_ELO_MIN = 1030
_ARENA_ELO_MAX = 1430
_ARENA_MAX_NORMALIZED = 82.0

# --- Leaderboard normalization ---
# OLLB v2 averages range ~5 to ~52. The leaderboard is archived 2025-06 with
# the top slot held by Qwen2.5-32B (47.6 raw = 91.5 if uncapped); capping at
# 78 prevents an older generation with a strong-but-frozen OLLB score from
# dominating rankings that now have AA Index / LiveBench coverage too.
_LB_AVG_MAX = 52
_OLLB_MAX_NORMALIZED = 78.0

# --- Arena display name -> HuggingFace org mapping ---
_ARENA_ORG_TO_HF: dict[str, list[str]] = {
    "Alibaba": ["Qwen"],
    "Meta": ["meta-llama"],
    "DeepSeek": ["deepseek-ai"],
    "DeepSeek AI": ["deepseek-ai"],
    "Google": ["google"],
    "Mistral": ["mistralai"],
    "Microsoft": ["microsoft"],
    "Nvidia": ["nvidia"],
    "01 AI": ["01-ai"],
    "Allen AI": ["allenai"],
    "Ai2": ["allenai"],
    "AllenAI/UW": ["allenai"],
    "Cohere": ["CohereForAI"],
    "HuggingFace": ["HuggingFaceH4", "huggingface"],
    "AI21 Labs": ["ai21labs"],
    "NousResearch": ["NousResearch"],
    "NexusFlow": ["Nexusflow"],
    "Princeton": ["princeton-nlp"],
    "IBM": ["ibm-granite"],
    "InternLM": ["internlm"],
    "Together AI": ["togethercomputer"],
    "TII": ["tiiuae"],
    "MiniMax": ["MiniMaxAI"],
    "MosaicML": ["mosaicml"],
    "Databricks": ["databricks"],
    "Moonshot": ["moonshotai"],
    "UC Berkeley": ["berkeley-nest"],
    "Cognitive Computations": ["cognitivecomputations"],
    "Upstage AI": ["upstage"],
    "UW": ["timdettmers"],
    "Snowflake": ["Snowflake"],
    "LMSYS": ["lmsys"],
    "OpenChat": ["openchat"],
}


@dataclass(frozen=True)
class BenchmarkEvidence:
    """Benchmark evidence with confidence.

    source values, ordered from most trusted to least:
      - "direct"        : independent leaderboard / Arena ELO hit on exact id
      - "variant"       : suffix-stripped derivative of a direct leaderboard hit
      - "base_model"    : cardData.base_model pointer to a direct hit
      - "line_interp"   : size-aware interpolation within the same model line
      - "self_reported" : evalResults reported by the uploader themselves
      - "none"          : no usable signal
    """

    score: float | None
    confidence: float
    source: str  # see above


def load_benchmark_cache() -> dict[str, float] | None:
    """Load cached benchmark scores. Returns None if expired or missing."""
    if not BENCHMARK_CACHE.exists():
        return None
    try:
        data = json.loads(BENCHMARK_CACHE.read_text())
        cached_at = data.get("cached_at", 0)
        if time.time() - cached_at > DEFAULT_TTL_SECONDS:
            logger.debug("Benchmark cache expired")
            return None
        return data.get("scores", {})
    except (json.JSONDecodeError, KeyError) as e:
        logger.debug(f"Benchmark cache corrupted: {e}")
        return None


def save_benchmark_cache(scores: dict[str, float]) -> None:
    """Save benchmark scores to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {"cached_at": time.time(), "scores": scores}
    BENCHMARK_CACHE.write_text(json.dumps(data, ensure_ascii=False))
    logger.debug(f"Saved {len(scores)} benchmark scores to cache")


def _normalize_arena_elo(elo: float) -> float:
    """Normalize Arena ELO to a frozen-source-aware 0-_ARENA_MAX_NORMALIZED scale."""
    score = (
        (elo - _ARENA_ELO_MIN)
        / (_ARENA_ELO_MAX - _ARENA_ELO_MIN)
        * _ARENA_MAX_NORMALIZED
    )
    return max(0.0, min(_ARENA_MAX_NORMALIZED, round(score, 1)))


def _normalize_leaderboard_avg(avg: float) -> float:
    """Normalize Open LLM Leaderboard average to 0-_OLLB_MAX_NORMALIZED scale."""
    score = avg / _LB_AVG_MAX * _OLLB_MAX_NORMALIZED
    return max(0.0, min(_OLLB_MAX_NORMALIZED, round(score, 1)))


def _arena_name_to_hf_ids(model_name: str, org: str) -> list[str]:
    """Convert Arena display name to potential HuggingFace model IDs."""
    hf_orgs = _ARENA_ORG_TO_HF.get(org, [])
    candidates = []

    # Clean the model name: remove date suffixes like "(03-2025)"
    clean_name = re.sub(r"\s*\([\d-]+\)\s*$", "", model_name).strip()
    # Remove -bf16, -fp8 suffixes for base matching
    base_name = re.sub(r"-(bf16|fp8|fp16)$", "", clean_name, flags=re.IGNORECASE)

    for hf_org in hf_orgs:
        candidates.append(f"{hf_org}/{clean_name}")
        if base_name != clean_name:
            candidates.append(f"{hf_org}/{base_name}")
        # Try with -Instruct suffix stripped for base model matching
        no_instruct = re.sub(r"-Instruct$", "", clean_name)
        if no_instruct != clean_name:
            candidates.append(f"{hf_org}/{no_instruct}")

    return candidates


def _fetch_arena_scores(client: httpx.Client) -> dict[str, float]:
    """Fetch Chatbot Arena ELO scores via rows API."""
    scores: dict[str, float] = {}
    offset = 0

    while True:
        resp = client.get(
            ARENA_ROWS_URL,
            params={
                "dataset": ARENA_DATASET,
                "config": "default",
                "split": "train",
                "offset": str(offset),
                "length": "100",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("rows", [])
        if not rows:
            break

        for r in rows:
            row = r.get("row", {})
            model_name = str(row.get("Model", ""))
            elo = row.get("Arena Score", 0)
            org = str(row.get("Organization", ""))
            lic = str(row.get("License", ""))

            if not model_name or not elo or elo <= 0:
                continue
            # Skip proprietary models (can't run locally)
            if "Proprietary" in lic or "Propretary" in lic:
                continue

            normalized = _normalize_arena_elo(elo)
            # Map to all potential HF IDs
            hf_ids = _arena_name_to_hf_ids(model_name, org)
            for hf_id in hf_ids:
                scores[hf_id] = normalized

        offset += len(rows)
        total = data.get("num_rows_total", 0)
        if total and offset >= total:
            break

    return scores


def _fetch_leaderboard_parquet(client: httpx.Client) -> dict[str, float]:
    """Download Open LLM Leaderboard parquet (requires pyarrow)."""
    import pyarrow.parquet as pq

    resp = client.get(LEADERBOARD_PARQUET_URL, follow_redirects=True)
    resp.raise_for_status()
    table = pq.read_table(
        io.BytesIO(resp.content),
        columns=["fullname", "Average ⬆️"],
    )
    d = table.to_pydict()
    scores: dict[str, float] = {}
    for i in range(len(d["fullname"])):
        name = d["fullname"][i]
        avg = d["Average ⬆️"][i]
        if name and avg and avg > 0:
            scores[name] = _normalize_leaderboard_avg(avg)
    return scores


def _fetch_leaderboard_api(client: httpx.Client) -> dict[str, float]:
    """Fetch Open LLM Leaderboard via rows API (no pyarrow needed)."""
    scores: dict[str, float] = {}
    offset = 0

    while True:
        resp = client.get(
            LEADERBOARD_ROWS_URL,
            params={
                "dataset": LEADERBOARD_DATASET,
                "config": "default",
                "split": "train",
                "offset": str(offset),
                "length": "100",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("rows", [])
        if not rows:
            break

        for r in rows:
            row = r.get("row", {})
            name = row.get("fullname")
            avg = row.get("Average ⬆️")
            if name and avg and avg > 0:
                scores[name] = _normalize_leaderboard_avg(avg)

        offset += len(rows)
        total = data.get("num_rows_total", 0)
        if total and offset >= total:
            break

    return scores


_LINEAGE_DEMOTION_REGEX = None


def _build_lineage_regex():
    """Compile MODEL_LINEAGE_VERSIONS once into (family, [(re, idx)]) form."""
    global _LINEAGE_DEMOTION_REGEX
    if _LINEAGE_DEMOTION_REGEX is not None:
        return _LINEAGE_DEMOTION_REGEX
    from whichllm.constants import MODEL_LINEAGE_VERSIONS

    out = {}
    for family, entries in MODEL_LINEAGE_VERSIONS.items():
        compiled = [(re.compile(pat), idx) for pat, idx in entries]
        max_idx = max(idx for _, idx in entries)
        out[family] = (compiled, max_idx)
    _LINEAGE_DEMOTION_REGEX = out
    return out


def _lineage_recency_factor(model_id: str) -> float:
    """Return a multiplicative recency factor for frozen-only scores.

    Newest generation in a known family → 1.0 (no demotion). Each generation
    older → another 12% off. Unknown families → 1.0.
    """
    if not model_id:
        return 1.0
    lower = model_id.lower()
    families = _build_lineage_regex()
    best_factor = 1.0
    for family, (patterns, max_idx) in families.items():
        for regex, idx in patterns:
            if regex.search(lower):
                gens_old = max(0, max_idx - idx)
                factor = max(0.55, 1.0 - 0.12 * gens_old)
                if factor < best_factor:
                    best_factor = factor
                break  # one family per id
    return best_factor


def _apply_lineage_recency_demotion(
    combined: dict[str, float],
    frozen: dict[str, float],
    current: dict[str, float],
) -> dict[str, float]:
    """Multiply frozen-only entries by a lineage-derived recency factor.

    A score is "frozen-only" when no current source (AA Index / LiveBench /
    Aider) provided a value for that id. Models with current coverage are
    left alone — their score already reflects 2026 evaluation methodology.
    """
    if not combined:
        return combined
    out: dict[str, float] = {}
    for k, v in combined.items():
        if k in current:
            out[k] = v
            continue
        factor = _lineage_recency_factor(k)
        out[k] = round(v * factor, 1)
    return out


def fetch_benchmark_scores() -> dict[str, float]:
    """Fetch and combine benchmark scores from multiple sources.

    Sources, merged in this order (later overwrites earlier on conflict):
      1. Open LLM Leaderboard v2 (archived 2025-06, broad legacy coverage)
      2. Chatbot Arena ELO (frozen 2025-07-17, but still useful older signal)
      3. LiveBench (monthly refresh, current generation)
      4. Aider polyglot (coding-specific, current generation)
      5. Artificial Analysis Intelligence Index (covers DeepSeek V4, GLM-5,
         Kimi K2.6, MiMo V2.5, Qwen3.6 — fills the Arena/Leaderboard gap)

    Returns dict mapping model_id -> normalized score (0-100). Any source
    that fails is logged and skipped; the function never raises.
    """
    from whichllm.models.benchmark_sources import (
        fetch_aa_index_scores,
        fetch_aider_polyglot_scores,
        fetch_livebench_scores,
        fetch_vision_scores,
    )

    # Layered merge: build a "current" dict from live sources (AA, LiveBench,
    # Aider) and a "frozen" dict from archived sources (OLLB v2, Arena). The
    # current dict OVERRIDES the frozen one per-model — so a 2024-era model
    # with a stale-but-high OLLB number cannot beat a 2026 model that AA or
    # LiveBench measure as merely mid-tier. Frozen scores still cover the
    # long tail of older models that no live source tracks.
    frozen: dict[str, float] = {}
    current: dict[str, float] = {}

    with httpx.Client(timeout=30.0) as client:
        # Frozen tier #1: Open LLM Leaderboard v2 (archived 2025-06)
        try:
            try:
                lb_scores = _fetch_leaderboard_parquet(client)
            except ImportError:
                lb_scores = _fetch_leaderboard_api(client)
            frozen.update(lb_scores)
            logger.debug(f"Leaderboard: {len(lb_scores)} scores (frozen)")
        except Exception as e:
            logger.warning(f"Leaderboard fetch failed: {e}")

        # Frozen tier #2: Chatbot Arena ELO (frozen 2025-07-17)
        try:
            arena_scores = _fetch_arena_scores(client)
            for k, v in arena_scores.items():
                if frozen.get(k, 0.0) < v:
                    frozen[k] = v
            logger.debug(f"Arena: {len(arena_scores)} scores (frozen)")
        except Exception as e:
            logger.warning(f"Arena fetch failed: {e}")

        # Current tier: LiveBench (monthly-refreshed)
        try:
            lb_scores = fetch_livebench_scores(client)
            for k, v in lb_scores.items():
                if current.get(k, 0.0) < v:
                    current[k] = v
            logger.debug(f"LiveBench: {len(lb_scores)} scores (current)")
        except Exception as e:
            logger.debug(f"LiveBench fetch failed: {e}")

        # Current tier: Artificial Analysis Intelligence Index (~weekly refresh)
        try:
            aa_scores = fetch_aa_index_scores(client)
            for k, v in aa_scores.items():
                if current.get(k, 0.0) < v:
                    current[k] = v
            logger.debug(f"AA Index: {len(aa_scores)} scores (current)")
        except Exception as e:
            logger.debug(f"AA Index fetch failed: {e}")

        # Current tier: Aider polyglot (coding-specific). Treat as a current
        # source but soft-merged — coding is one axis of capability, so a high
        # Aider score is informative but shouldn't unilaterally dethrone a
        # weaker-coding-but-strong-general AA result.
        try:
            aider_scores = fetch_aider_polyglot_scores(client)
            for k, v in aider_scores.items():
                if current.get(k, 0.0) < v * 0.85:
                    current[k] = v * 0.85
            logger.debug(f"Aider polyglot: {len(aider_scores)} scores (current, 0.85x)")
        except Exception as e:
            logger.debug(f"Aider fetch failed: {e}")

        # Current tier: vision-language capability index. Text leaderboards
        # don't score VLMs, so without this the only VLM with a direct hit
        # was a two-generations-old Qwen2-VL-7B. Profile filtering ensures
        # these scores only affect ``--profile vision`` rankings.
        try:
            vision_scores = fetch_vision_scores(client)
            for k, v in vision_scores.items():
                if current.get(k, 0.0) < v:
                    current[k] = v
            logger.debug(f"Vision: {len(vision_scores)} scores (current)")
        except Exception as e:
            logger.debug(f"Vision fetch failed: {e}")

    # Build combined: current overrides frozen entry-by-entry, but frozen still
    # contributes for any id no current source has tracked.
    combined: dict[str, float] = dict(frozen)
    combined.update(current)

    # Apply lineage-aware demotion to frozen-only scores. Without this, models
    # that have no live coverage (e.g. Qwen2.5-72B-Instruct, Llama-3.1-70B
    # — both 2024 releases) retain their generous frozen leaderboard score
    # while their *newer* siblings (Qwen3-32B, Llama-3.3-70B) get held to
    # the live AA/LiveBench numbers. The result was older-generation 70B+
    # models ranking *above* the current-gen frontier on H100 / M2 Ultra.
    # Demote frozen-only entries from non-newest generations of known
    # families so the staleness penalty is uniform.
    combined = _apply_lineage_recency_demotion(combined, frozen, current)

    logger.debug(f"Combined: {len(combined)} benchmark scores")
    return combined


def _extract_params_b_from_id(model_id: str) -> float | None:
    """Extract parameter size in billions from model ID text."""
    lower = model_id.lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)b(?:-a\d+(?:\.\d+)?b)?", lower)
    if not matches:
        return None
    try:
        return max(float(v) for v in matches)
    except ValueError:
        return None


def _extract_model_lines(model_id: str) -> list[str]:
    """Extract model line candidates from a model ID (most specific first).

    E.g.:
        Qwen/Qwen3.5-27B -> [qwen/qwen3.5, qwen/qwen3]
        Qwen/Qwen3-32B -> [qwen/qwen3]
        Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 -> [qwen/qwen3]
        meta-llama/Llama-3.3-70B-Instruct -> [meta-llama/llama-3.3, meta-llama/llama-3]
        google/gemma-3-27b-it -> [google/gemma-3]
        deepseek-ai/DeepSeek-V3.2 -> [deepseek-ai/deepseek-v3.2, deepseek-ai/deepseek-v3]
    """
    if "/" not in model_id:
        return []
    lower = model_id.lower()

    # Pre-strip repo/quant suffixes and date codes before line extraction
    stripped = re.sub(r"-(gguf|awq|gptq|fp8|fp16|bf16|nvfp4)$", "", lower)
    stripped = re.sub(r"-\d{4}(-hf)?$", "", stripped)  # date suffixes like -2507

    lines: list[str] = []

    # Remove size suffix: -32b, -70b, -0.6b, -235b-a22b, etc.
    # Allows trailing -instruct, -chat, -it, -base, -thinking, and arbitrary suffixes
    cleaned = re.sub(
        r"-\d+(\.\d+)?b(-a\d+b)?(-[a-z][-a-z0-9]*)*$",
        "",
        stripped,
    )
    if cleaned != stripped and "/" in cleaned:
        lines.append(cleaned)

    # Also strip minor version: qwen3.5 -> qwen3, llama-3.3 -> llama-3, v3.2 -> v3
    for line in list(lines) + ([stripped] if not lines else []):
        broader = re.sub(r"(\d+)\.\d+$", r"\1", line)
        if broader != line and broader not in lines:
            lines.append(broader)

    return lines


def _interpolate_line_score(
    bucket: list[tuple[float | None, float]],
    params_b: float | None,
) -> tuple[float, float]:
    """Interpolate score from same-model-line benchmarks with confidence."""
    if not bucket:
        return 0.0, 0.0

    valid = [(p, s) for p, s in bucket if p is not None]
    if not valid:
        vals = [s for _, s in bucket]
        return statistics.median(vals), 0.25

    if params_b is None or params_b <= 0:
        vals = [s for _, s in valid]
        return statistics.median(vals), 0.30

    weighted: list[tuple[float, float, float]] = []
    for p, s in valid:
        assert p is not None
        dist = abs(math.log2(max(params_b, 0.1) / max(p, 0.1)))
        w = 1.0 / (0.35 + dist)
        weighted.append((w, s, dist))

    score = sum(w * s for w, s, _ in weighted) / sum(w for w, _, _ in weighted)
    nearest = min(d for _, _, d in weighted)
    if nearest <= 0.15:
        conf = 0.45
    elif nearest <= 0.50:
        conf = 0.34
    else:
        conf = 0.26
    return score, conf


def build_score_index(
    scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Build lookup indices from benchmark scores.

    Returns:
        (case_insensitive_index, line_index)
        - case_insensitive_index: lowercased model_id -> best score
        - line_index: model_line -> best score among all models in that line
    """
    ci_index: dict[str, float] = {}
    line_index: dict[str, float] = {}

    for key, val in scores.items():
        lk = key.lower()
        if lk not in ci_index or val > ci_index[lk]:
            ci_index[lk] = val

        lines = _extract_model_lines(key)
        if not lines and "/" in key:
            # No size suffix (e.g., DeepSeek-V3, DeepSeek-R1) → use ID as its own line
            lines = [lk]
        for line in lines:
            if line not in line_index or val > line_index[line]:
                line_index[line] = val

    return ci_index, line_index


def build_line_bucket_index(
    scores: dict[str, float],
) -> dict[str, list[tuple[float | None, float]]]:
    """Build line -> [(params_b, score)] index for size-aware interpolation."""
    buckets: dict[str, list[tuple[float | None, float]]] = {}
    for key, val in scores.items():
        params_b = _extract_params_b_from_id(key)
        lines = _extract_model_lines(key)
        if not lines and "/" in key:
            lines = [key.lower()]
        for line in lines:
            buckets.setdefault(line, []).append((params_b, val))
    return buckets


def _try_lookup(
    candidate: str, scores: dict[str, float], ci_index: dict[str, float]
) -> float | None:
    """Try exact match, then case-insensitive match."""
    if candidate in scores:
        return scores[candidate]
    lc = candidate.lower()
    if lc in ci_index:
        return ci_index[lc]
    return None


_REPO_SUFFIXES = ("-GGUF", "-gguf", "-AWQ", "-GPTQ", "-FP8", "-fp8", "-BF16", "-bf16")


def _generate_candidates(model_id: str) -> list[str]:
    """Generate candidate IDs to look up for a model."""
    candidates = [model_id]

    # Strip common GGUF/quant repo suffixes
    for suffix in _REPO_SUFFIXES:
        if model_id.endswith(suffix):
            candidates.append(model_id[: -len(suffix)])
            break

    # Try adding/removing -Instruct suffix
    base = candidates[-1]  # use suffix-stripped version
    if base.endswith("-Instruct"):
        candidates.append(base[: -len("-Instruct")])
    else:
        candidates.append(base + "-Instruct")

    return candidates


def lookup_benchmark(
    model_id: str,
    base_model: str | None,
    scores: dict[str, float],
    ci_index: dict[str, float] | None = None,
    line_index: dict[str, float] | None = None,
) -> tuple[float, bool] | None:
    """Backward-compatible benchmark lookup helper."""
    evidence = lookup_benchmark_evidence(
        model_id,
        base_model,
        scores,
        ci_index=ci_index,
        line_index=line_index,
    )
    if evidence.score is None:
        return None
    return evidence.score, evidence.source == "direct"


def _params_compatible(actual_b: float | None, ref_id: str) -> bool:
    """Reject benchmark inheritance when the actual model size differs sharply
    from the size implied by a reference id. Catches cases like a 6.6B
    "imatrix-aligned" / draft / MTP head being matched to its 158B base model.

    Returns True if no actual size is provided (no check possible) or if
    ratio(actual, ref) stays inside [0.5, 2.0]. The window is wide enough
    that legitimate quantizations of the same model are unaffected.
    """
    if actual_b is None or actual_b <= 0:
        return True
    ref_b = _extract_params_b_from_id(ref_id)
    if ref_b is None or ref_b <= 0:
        return True
    ratio = actual_b / ref_b
    return 0.5 <= ratio <= 2.0


def lookup_benchmark_evidence(
    model_id: str,
    base_model: str | None,
    scores: dict[str, float],
    ci_index: dict[str, float] | None = None,
    line_index: dict[str, float] | None = None,
    line_bucket_index: dict[str, list[tuple[float | None, float]]] | None = None,
    self_reported_score: float | None = None,
    actual_params_b: float | None = None,
) -> BenchmarkEvidence:
    """Look up benchmark evidence with confidence.

    Resolution order:
      direct (leaderboard) → variant → base_model → line_interp → self_reported

    self_reported_score should be the uploader-provided evalResults score from
    the model card. It is the lowest-trust source and is only returned when
    every leaderboard/inheritance path fails.

    actual_params_b: actual parameter count in billions. When provided, the
    function refuses to inherit from base_model/variant ids whose implied
    size is more than 2x off from actual (e.g. a 6.6B "imatrix-aligned"
    inheriting from a 158B base would be rejected).
    """
    if ci_index is None or line_index is None:
        ci_index, line_index = build_score_index(scores)
    if line_bucket_index is None:
        line_bucket_index = build_line_bucket_index(scores)

    # Only exact model_id match in an independent leaderboard is considered
    # direct evidence. Self-reported evalResults are handled at the very end.
    direct_result = _try_lookup(model_id, scores, ci_index)
    if direct_result is not None:
        return BenchmarkEvidence(score=direct_result, confidence=1.0, source="direct")

    # Try model_id-derived variants (inherited)
    for candidate in _generate_candidates(model_id)[1:]:
        result = _try_lookup(candidate, scores, ci_index)
        if result is not None:
            if not _params_compatible(actual_params_b, candidate):
                continue
            return BenchmarkEvidence(score=result, confidence=0.55, source="variant")

    # Try base_model and its variants
    if base_model:
        for candidate in _generate_candidates(base_model):
            result = _try_lookup(candidate, scores, ci_index)
            if result is not None:
                if not _params_compatible(actual_params_b, candidate):
                    continue
                return BenchmarkEvidence(
                    score=result, confidence=0.60, source="base_model"
                )

    # Fallback: size-aware interpolation within model line.
    size_hint = (
        actual_params_b
        or _extract_params_b_from_id(model_id)
        or _extract_params_b_from_id(base_model or "")
    )
    for mid in (model_id, base_model):
        if mid:
            for line in _extract_model_lines(mid):
                if line in line_bucket_index:
                    score, conf = _interpolate_line_score(
                        line_bucket_index[line], size_hint
                    )
                    if score > 0:
                        return BenchmarkEvidence(
                            score=score, confidence=conf, source="line_interp"
                        )
                if line in line_index:
                    return BenchmarkEvidence(
                        score=line_index[line], confidence=0.22, source="line_interp"
                    )

    # Last resort: uploader-reported eval. Anyone can write any number here so
    # we keep confidence low and require downstream consumers to weight this
    # source separately.
    if (
        self_reported_score is not None
        and isinstance(self_reported_score, (int, float))
        and self_reported_score > 0
    ):
        return BenchmarkEvidence(
            score=float(self_reported_score),
            confidence=0.40,
            source="self_reported",
        )

    return BenchmarkEvidence(score=None, confidence=0.0, source="none")
