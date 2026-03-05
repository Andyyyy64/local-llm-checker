"""Tests for benchmark lookup direct/inherited semantics."""

from whichllm.models.benchmark import build_score_index, lookup_benchmark


def test_lookup_benchmark_model_id_match_is_direct():
    scores = {"Qwen/Qwen2.5-7B-Instruct": 70.0}
    ci, line = build_score_index(scores)
    result = lookup_benchmark(
        "Qwen/Qwen2.5-7B-Instruct",
        None,
        scores,
        ci,
        line,
    )
    assert result == (70.0, True)


def test_lookup_benchmark_base_model_match_is_inherited():
    scores = {"google/gemma-3-27b-it": 82.2}
    ci, line = build_score_index(scores)
    result = lookup_benchmark(
        "ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g",
        "google/gemma-3-27b-it",
        scores,
        ci,
        line,
    )
    assert result == (82.2, False)


def test_lookup_benchmark_gguf_suffix_match_is_inherited():
    scores = {"Qwen/Qwen2.5-7B-Instruct": 70.0}
    ci, line = build_score_index(scores)
    result = lookup_benchmark(
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        None,
        scores,
        ci,
        line,
    )
    assert result == (70.0, False)
