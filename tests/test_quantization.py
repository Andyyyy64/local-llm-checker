"""Tests for non-GGUF quantization inference."""

from whichllm.engine.quantization import (
    effective_quant_type,
    estimate_weight_bytes,
    infer_non_gguf_quant_type,
)
from whichllm.engine.vram import estimate_vram
from whichllm.models.types import ModelInfo


def _make_model(model_id: str, params: int = 14_000_000_000) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        family_id=model_id,
        name=model_id.split("/")[-1],
        parameter_count=params,
    )


def test_infer_non_gguf_awq():
    model = _make_model("Qwen/Qwen2.5-14B-Instruct-AWQ")
    assert infer_non_gguf_quant_type(model.id) == "AWQ"
    assert effective_quant_type(model, None) == "AWQ"


def test_estimate_weight_bytes_for_awq():
    model = _make_model("Qwen/Qwen2.5-14B-Instruct-AWQ", params=10_000_000_000)
    assert estimate_weight_bytes(model, None) == 5_000_000_000


def test_awq_vram_is_lower_than_fp16_fallback():
    awq = _make_model("Qwen/Qwen2.5-14B-Instruct-AWQ")
    fp16 = _make_model("Qwen/Qwen2.5-14B-Instruct")
    assert estimate_vram(awq, None, context_length=4096) < estimate_vram(
        fp16, None, context_length=4096
    )
