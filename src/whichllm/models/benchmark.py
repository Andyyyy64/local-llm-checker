"""Benchmark data fetcher: Chatbot Arena ELO + Open LLM Leaderboard."""

from __future__ import annotations

import io
import json
import logging
import re
import time
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
    """Look up benchmark score for a model.

    Tries model_id first, then base_model, then common name variants.
    Falls back to model line (series) match for family-level data.
    Uses case-insensitive matching via index for Arena name mismatches.

    Returns (normalized_score, is_direct) or None.
    is_direct=True means the score was matched to this specific model ID
    (including common model_id variants).
    is_direct=False means the score was inherited from base_model or model line.
    """
    if ci_index is None or line_index is None:
        ci_index, line_index = build_score_index(scores)

    # Only exact model_id match is considered direct.
    # Derived candidates (suffix-stripped, instruct-toggled) are inherited.
    direct_result = _try_lookup(model_id, scores, ci_index)
    if direct_result is not None:
        return direct_result, True

    # Try model_id-derived variants (inherited)
    for candidate in _generate_candidates(model_id)[1:]:
        result = _try_lookup(candidate, scores, ci_index)
        if result is not None:
            return result, False

    # Try base_model and its variants
    if base_model:
        for candidate in _generate_candidates(base_model):
            result = _try_lookup(candidate, scores, ci_index)
            if result is not None:
                # base_model inheritance is useful but not a direct match
                return result, False

    # Fallback: model line (series) lookup
    # E.g., Qwen3-8B has no direct score, but Qwen3-32B does → use line score
    # Tries specific lines first (qwen/qwen3.5), then broader (qwen/qwen3)
    for mid in (model_id, base_model):
        if mid:
            for line in _extract_model_lines(mid):
                if line in line_index:
                    return line_index[line], False

    return None
