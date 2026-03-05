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
# Open-source ELO range: ~1030 (worst) to ~1424 (best)
# Normalize to 0-100 percentile
_ARENA_ELO_MIN = 1030
_ARENA_ELO_MAX = 1430

# --- Leaderboard normalization ---
# Average scores range: ~5 to ~52
# Normalize to 0-100 percentile
_LB_AVG_MAX = 52

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
    """Benchmark evidence with confidence."""

    score: float | None
    confidence: float
    source: str  # direct | variant | base_model | line_interp | none


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
    """Normalize Arena ELO to 0-100 scale."""
    score = (elo - _ARENA_ELO_MIN) / (_ARENA_ELO_MAX - _ARENA_ELO_MIN) * 100
    return max(0, min(100, round(score, 1)))


def _normalize_leaderboard_avg(avg: float) -> float:
    """Normalize Open LLM Leaderboard average to 0-100 scale."""
    score = avg / _LB_AVG_MAX * 100
    return max(0, min(100, round(score, 1)))


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


def fetch_benchmark_scores() -> dict[str, float]:
    """Fetch and combine benchmark scores from multiple sources.

    Sources (in priority order):
    1. Chatbot Arena ELO (most recent, covers latest models)
    2. Open LLM Leaderboard (broad coverage, older models)

    Returns dict mapping model_id -> normalized score (0-100).
    Arena scores take priority when both sources have data.
    """
    combined: dict[str, float] = {}

    with httpx.Client(timeout=30.0) as client:
        # 1. Open LLM Leaderboard (lower priority, loaded first)
        try:
            try:
                lb_scores = _fetch_leaderboard_parquet(client)
            except ImportError:
                lb_scores = _fetch_leaderboard_api(client)
            combined.update(lb_scores)
            logger.debug(f"Leaderboard: {len(lb_scores)} scores")
        except Exception as e:
            logger.warning(f"Leaderboard fetch failed: {e}")

        # 2. Chatbot Arena ELO (higher priority, overwrites leaderboard)
        try:
            arena_scores = _fetch_arena_scores(client)
            combined.update(arena_scores)
            logger.debug(f"Arena: {len(arena_scores)} scores")
        except Exception as e:
            logger.warning(f"Arena fetch failed: {e}")

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
        r"-\d+(\.\d+)?b(-a\d+b)?(-[a-z][-a-z0-9]*)*$", "", stripped,
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


def _try_lookup(candidate: str, scores: dict[str, float], ci_index: dict[str, float]) -> float | None:
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


def lookup_benchmark_evidence(
    model_id: str,
    base_model: str | None,
    scores: dict[str, float],
    ci_index: dict[str, float] | None = None,
    line_index: dict[str, float] | None = None,
    line_bucket_index: dict[str, list[tuple[float | None, float]]] | None = None,
) -> BenchmarkEvidence:
    """Look up benchmark evidence with confidence."""
    if ci_index is None or line_index is None:
        ci_index, line_index = build_score_index(scores)
    if line_bucket_index is None:
        line_bucket_index = build_line_bucket_index(scores)

    # Only exact model_id match is considered direct.
    # Derived candidates (suffix-stripped, instruct-toggled) are inherited.
    direct_result = _try_lookup(model_id, scores, ci_index)
    if direct_result is not None:
        return BenchmarkEvidence(score=direct_result, confidence=1.0, source="direct")

    # Try model_id-derived variants (inherited)
    for candidate in _generate_candidates(model_id)[1:]:
        result = _try_lookup(candidate, scores, ci_index)
        if result is not None:
            return BenchmarkEvidence(score=result, confidence=0.55, source="variant")

    # Try base_model and its variants
    if base_model:
        for candidate in _generate_candidates(base_model):
            result = _try_lookup(candidate, scores, ci_index)
            if result is not None:
                return BenchmarkEvidence(score=result, confidence=0.60, source="base_model")

    # Fallback: size-aware interpolation within model line.
    size_hint = _extract_params_b_from_id(model_id) or _extract_params_b_from_id(base_model or "")
    for mid in (model_id, base_model):
        if mid:
            for line in _extract_model_lines(mid):
                if line in line_bucket_index:
                    score, conf = _interpolate_line_score(line_bucket_index[line], size_hint)
                    if score > 0:
                        return BenchmarkEvidence(score=score, confidence=conf, source="line_interp")
                if line in line_index:
                    return BenchmarkEvidence(score=line_index[line], confidence=0.22, source="line_interp")

    return BenchmarkEvidence(score=None, confidence=0.0, source="none")
