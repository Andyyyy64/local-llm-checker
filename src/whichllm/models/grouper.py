"""Model family grouping logic."""

from __future__ import annotations

import re

from whichllm.models.types import ModelFamily, ModelInfo


def _normalize_name(model_id: str) -> str:
    """Normalize model ID for grouping by removing org prefix and GGUF/quant/chat suffixes."""
    name = model_id.lower()
    # Strip org prefix (e.g. "bartowski/Meta-Llama-3.1" -> "meta-llama-3.1")
    if "/" in name:
        name = name.split("/", 1)[1]
    # Strip common org prefixes in model names (e.g. "qwen_qwen3-8b" -> "qwen3-8b")
    name = re.sub(r"^(qwen_|meta-llama_|google_)", "", name)
    # Remove common suffixes (applied repeatedly to handle stacked suffixes)
    suffixes = [
        r"-gguf$",
        r"-gptq$",
        r"-awq$",
        r"-instruct$",
        r"-chat$",
        r"-it$",
        r"-hf$",
        r"-fp8$",
        r"-fp16$",
        r"-bf16$",
        r"-nvfp4$",
        r"-\d+bit$",
        r"-\d{4}$",  # date suffixes like -2507, -2503
    ]
    for _ in range(3):  # multiple passes to strip stacked suffixes
        prev = name
        for suffix in suffixes:
            name = re.sub(suffix, "", name)
        if name == prev:
            break

    # Strip version-before-size: mistral-small-3.2-24b -> mistral-small-24b
    # This catches patterns like MODEL-MAJOR.MINOR-SIZEb where the version
    # is a separate segment (preceded by '-') before the size suffix.
    # Does NOT match qwen3.5-27b because '3.5' is glued to 'qwen' without '-'.
    name = re.sub(r"-\d+\.\d+(-\d+(?:\.\d+)?b(?:-a\d+b)?)$", r"\1", name)

    # Split series name from size suffix, strip minor version from series only.
    # Merges qwen3.5-27b + qwen3-30b-a3b naming variants (different sizes stay separate).
    m = re.match(r"^(.+?)-(\d+(?:\.\d+)?b(?:-a\d+b)?)$", name)
    if m:
        series, size = m.group(1), m.group(2)
        series = re.sub(r"(\d+)\.\d+$", r"\1", series)
        name = f"{series}-{size}"
    else:
        # No size suffix (e.g. deepseek-v3.2) — strip minor version directly
        name = re.sub(r"(\d+)\.\d+$", r"\1", name)

    return name


def group_models(models: list[ModelInfo]) -> list[ModelFamily]:
    """Group models into families based on base_model and name similarity."""
    # Pass 1: Group by base_model
    base_model_groups: dict[str, list[ModelInfo]] = {}
    ungrouped: list[ModelInfo] = []

    for model in models:
        if model.base_model:
            key = model.base_model.lower()
            base_model_groups.setdefault(key, []).append(model)
        else:
            ungrouped.append(model)

    # Pass 2: Group ungrouped by normalized name
    name_groups: dict[str, list[ModelInfo]] = {}
    for model in ungrouped:
        key = _normalize_name(model.id)
        name_groups.setdefault(key, []).append(model)

    # Merge base_model groups that share the same normalized name
    merged_base: dict[str, list[ModelInfo]] = {}
    for key, group in base_model_groups.items():
        norm_key = _normalize_name(key)
        merged_base.setdefault(norm_key, []).extend(group)

    # Also merge with ungrouped via name matching
    for norm_key, group in list(merged_base.items()):
        if norm_key in name_groups:
            group.extend(name_groups.pop(norm_key))

    # Replace base_model_groups with merged version
    base_model_groups = merged_base

    # Build families
    families: list[ModelFamily] = []

    for group_key, group in list(base_model_groups.items()) + list(name_groups.items()):
        if not group:
            continue

        # Pick the base model: prefer the one with most downloads that has no GGUF suffix
        base_candidates = [
            m for m in group if not m.gguf_variants or m.base_model is None
        ]
        if not base_candidates:
            base_candidates = group

        base = max(base_candidates, key=lambda m: m.downloads)
        variants = [m for m in group if m.id != base.id]

        # Set family_id on all members
        family_id = _normalize_name(base.id)
        base.family_id = family_id
        for v in variants:
            v.family_id = family_id

        # Collect best benchmark scores across family
        best_bench: dict[str, float] = {}
        for m in group:
            for k, v in m.benchmark_scores.items():
                if k not in best_bench or v > best_bench[k]:
                    best_bench[k] = v

        families.append(
            ModelFamily(
                family_id=family_id,
                display_name=base.name,
                base_model=base,
                variants=variants,
                best_benchmark=best_bench,
            )
        )

    return families
