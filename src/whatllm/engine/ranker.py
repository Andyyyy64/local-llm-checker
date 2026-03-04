"""Model ranking: score and select the best models for the user's hardware."""

from __future__ import annotations

import math

from whatllm.constants import QUANT_PREFERENCE_ORDER, QUANT_QUALITY_PENALTY
from whatllm.engine.compatibility import check_compatibility
from whatllm.engine.performance import estimate_tok_per_sec
from whatllm.engine.types import CompatibilityResult
from whatllm.hardware.types import HardwareInfo
from whatllm.models.benchmark import build_score_index, lookup_benchmark
from whatllm.models.types import GGUFVariant, ModelInfo


def _pick_best_variant(
    model: ModelInfo,
    hardware: HardwareInfo,
    context_length: int,
    quant_filter: str | None = None,
) -> tuple[GGUFVariant | None, CompatibilityResult]:
    """Pick the best GGUF variant that fits in hardware, or None."""
    if not model.gguf_variants:
        # No GGUF: check if FP16 fits
        result = check_compatibility(model, None, hardware, context_length)
        return None, result

    # Filter by quant type if specified
    candidates = model.gguf_variants
    if quant_filter:
        filtered = [v for v in candidates if v.quant_type.upper() == quant_filter.upper()]
        if filtered:
            candidates = filtered
    else:
        # Exclude extreme quantizations unless explicitly requested
        _EXTREME_QUANTS = {"Q2_K", "IQ2_XXS", "IQ3_XXS"}
        filtered = [v for v in candidates if v.quant_type.upper() not in _EXTREME_QUANTS]
        if filtered:
            candidates = filtered

    # Sort by preference order
    def variant_sort_key(v: GGUFVariant) -> int:
        try:
            return QUANT_PREFERENCE_ORDER.index(v.quant_type.upper())
        except ValueError:
            return len(QUANT_PREFERENCE_ORDER)

    candidates = sorted(candidates, key=variant_sort_key)

    # Try each variant, pick first that fits
    best_result: CompatibilityResult | None = None
    best_variant: GGUFVariant | None = None

    for variant in candidates:
        result = check_compatibility(model, variant, hardware, context_length)
        if result.can_run and result.fit_type == "full_gpu":
            return variant, result
        if result.can_run and (best_result is None or best_result.fit_type != "full_gpu"):
            best_variant = variant
            best_result = result

    if best_result and best_variant:
        return best_variant, best_result

    # Nothing fits: return the smallest variant's result for reporting
    smallest = min(candidates, key=lambda v: v.file_size_bytes)
    result = check_compatibility(model, smallest, hardware, context_length)
    return smallest, result


_OFFICIAL_ORGS = frozenset({
    "Qwen", "meta-llama", "google", "mistralai", "deepseek-ai",
    "microsoft", "nvidia", "01-ai", "tiiuae", "apple",
    "CohereForAI", "bigcode",
})

# Trusted GGUF converters — format converters that don't change model quality
_TRUSTED_CONVERTERS = frozenset({
    "bartowski", "lmstudio-community", "QuantFactory", "unsloth",
    "ggml-org", "Mungert",
})

# Known repackagers — typically reupload others' models without added value
_REPACKAGER_ORGS = frozenset({
    "MaziyarPanahi", "TheBloke", "SanctumAI", "solidrust",
    "mradermacher",
})


def _compute_quality_score(
    model: ModelInfo,
    variant: GGUFVariant | None,
    tok_per_sec: float,
    fit_type: str,
    family_downloads: int = 0,
    family_likes: int = 0,
    benchmark_avg: float | None = None,
    benchmark_is_direct: bool = False,
) -> float:
    """Compute a quality score (0-100) for ranking.

    Factors:
    - Benchmark score (direct or line-estimated)
    - Model size (log scale base)
    - Quantization penalty
    - Fit type penalty (partial offload / CPU-only heavily penalized)
    - Speed bonus (practical usability)
    - Popularity (downloads/likes as tiebreaker)
    - Official repo bonus
    """
    params_b = model.parameter_count / 1e9
    if model.is_moe and model.parameter_count_active:
        effective_b = model.parameter_count_active / 1e9
    else:
        effective_b = params_b

    if effective_b <= 0:
        return 0.0

    # Base quality from parameter count (0-40)
    # 1B->12, 7B->31, 14B->37, 27B->40
    size_score = 6 * math.log2(max(effective_b, 0.5)) + 12
    size_score = min(size_score, 40)

    # Benchmark bonus (0-10): reward models with proven benchmark results
    # benchmark_avg is normalized to 0-100 scale (Arena ELO or Leaderboard)
    # Direct match gets full bonus; line-estimated gets 60% (less reliable)
    benchmark_bonus = 0.0
    has_benchmark = benchmark_avg is not None and benchmark_avg > 0
    if has_benchmark:
        raw_bonus = min(10.0, benchmark_avg / 100 * 10)
        if benchmark_is_direct:
            benchmark_bonus = raw_bonus
        else:
            benchmark_bonus = raw_bonus * 0.6  # line-estimated discount

    # Quantization penalty
    quant_penalty = 0.0
    if variant:
        quant_penalty = QUANT_QUALITY_PENALTY.get(variant.quant_type.upper(), 0.05)
    quality_after_quant = size_score * (1 - quant_penalty)

    # No-benchmark penalty: models without any benchmark data are unproven
    if not has_benchmark:
        quality_after_quant *= 0.85

    # Fit type penalty: partial offload and CPU-only are much worse in practice
    if fit_type == "partial_offload":
        quality_after_quant *= 0.60  # 40% penalty — offloading kills usability
    elif fit_type == "cpu_only":
        quality_after_quant *= 0.35  # 65% penalty — very slow

    # Speed bonus (0-20): practical usability matters a lot
    # 32 tok/s = 20, 16 tok/s = 16, 4 tok/s = 8, 1 tok/s = 0
    speed_bonus = 0.0
    if tok_per_sec > 0:
        speed_bonus = min(20.0, math.log2(max(tok_per_sec, 1)) * 4)

    # Popularity tiebreaker (0-3 points)
    # Use family-level stats so GGUF converters inherit base model popularity
    downloads = max(model.downloads, family_downloads)
    likes = max(model.likes, family_likes)
    pop_score = 0.0
    if downloads > 0:
        pop_score += min(1.5, math.log10(max(downloads, 1)) / 6 * 1.5)
    if likes > 0:
        pop_score += min(1.5, math.log10(max(likes, 1)) / 4 * 1.5)

    # Source trust adjustment: reward official, penalize repackagers
    source_bonus = 0.0
    org = model.id.split("/")[0] if "/" in model.id else ""
    if org in _OFFICIAL_ORGS:
        source_bonus = 5.0
    elif org in _REPACKAGER_ORGS:
        source_bonus = -5.0
    elif model.base_model:
        base_org = model.base_model.split("/")[0] if "/" in model.base_model else ""
        if base_org in _OFFICIAL_ORGS:
            if org in _TRUSTED_CONVERTERS:
                # Known GGUF converter — same quality as official
                source_bonus = 5.0
            else:
                # Unknown fine-tune/fork — no bonus (quality unproven)
                source_bonus = 0.0

    return min(100.0, quality_after_quant + benchmark_bonus + speed_bonus + pop_score + source_bonus)


def rank_models(
    models: list[ModelInfo],
    hardware: HardwareInfo,
    context_length: int = 4096,
    top_n: int = 10,
    quant_filter: str | None = None,
    min_speed: float | None = None,
    benchmark_scores: dict[str, float] | None = None,
) -> list[CompatibilityResult]:
    """Rank models by quality for the given hardware. Returns top N results."""
    results: list[CompatibilityResult] = []

    # Pre-compute max downloads/likes per family so GGUF converters
    # inherit popularity from the official base model
    family_max_downloads: dict[str, int] = {}
    family_max_likes: dict[str, int] = {}
    for m in models:
        fid = m.family_id
        family_max_downloads[fid] = max(family_max_downloads.get(fid, 0), m.downloads)
        family_max_likes[fid] = max(family_max_likes.get(fid, 0), m.likes)

    # Deduplicate by family: pick best variant per family
    seen_families: set[str] = set()

    # Sort models by downloads (popular first) to process best candidates first
    sorted_models = sorted(models, key=lambda m: m.downloads, reverse=True)

    # Build benchmark indices once (case-insensitive + model line)
    if benchmark_scores:
        bench_ci_index, bench_line_index = build_score_index(benchmark_scores)
    else:
        bench_ci_index, bench_line_index = {}, {}

    best_gpu = None
    for gpu in hardware.gpus:
        if best_gpu is None or gpu.vram_bytes > best_gpu.vram_bytes:
            best_gpu = gpu

    for model in sorted_models:
        variant, compat = _pick_best_variant(model, hardware, context_length, quant_filter)
        if not compat.can_run:
            continue

        # Estimate speed
        tok_per_sec = estimate_tok_per_sec(model, variant, best_gpu, compat.fit_type)
        compat.estimated_tok_per_sec = tok_per_sec

        # Apply min speed filter
        if min_speed is not None and tok_per_sec < min_speed:
            continue

        # Compute quality score
        fid = model.family_id
        bench_avg = None
        bench_is_direct = False
        if benchmark_scores:
            bench_result = lookup_benchmark(model.id, model.base_model, benchmark_scores, bench_ci_index, bench_line_index)
            if bench_result is not None:
                bench_avg, bench_is_direct = bench_result
        compat.quality_score = _compute_quality_score(
            model, variant, tok_per_sec, compat.fit_type,
            family_downloads=family_max_downloads.get(fid, 0),
            family_likes=family_max_likes.get(fid, 0),
            benchmark_avg=bench_avg,
            benchmark_is_direct=bench_is_direct,
        )
        if bench_avg is not None:
            compat.benchmark_status = "direct" if bench_is_direct else "estimated"
        else:
            compat.benchmark_status = "none"

        # Deduplicate by family: keep the one with highest quality score
        family_key = model.family_id
        if family_key in seen_families:
            # Check if this is better than existing
            existing = next(
                (r for r in results if r.model.family_id == family_key), None
            )
            if existing and compat.quality_score > existing.quality_score:
                results.remove(existing)
                results.append(compat)
            continue

        seen_families.add(family_key)
        results.append(compat)

    # Sort by quality score descending
    results.sort(key=lambda r: r.quality_score, reverse=True)

    return results[:top_n]
