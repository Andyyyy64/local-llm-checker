import builtins
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

from whichllm.hardware.nvidia import detect_nvidia_gpus


def test_nvidia_smi_fallback_when_pynvml_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError
        return real_import(name, *args, **kwargs)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="NVIDIA GeForce RTX 5060 Ti, 16303\n")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(subprocess, "run", fake_run)

    gpus = detect_nvidia_gpus()

    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA GeForce RTX 5060 Ti"
    assert gpus[0].vendor == "nvidia"
    assert gpus[0].vram_bytes == 16303 * 1024**2


def test_nvidia_smi_fallback_when_nvml_init_fails(monkeypatch):
    class FakeNVMLError(Exception):
        pass

    fake_pynvml = SimpleNamespace(
        NVMLError=FakeNVMLError,
        nvmlInit=Mock(side_effect=FakeNVMLError("NVML unavailable")),
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            return fake_pynvml
        return real_import(name, *args, **kwargs)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="NVIDIA DGX Spark, 128000 MiB\n")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(subprocess, "run", fake_run)

    gpus = detect_nvidia_gpus()

    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA DGX Spark"
    assert gpus[0].vendor == "nvidia"
    assert gpus[0].vram_bytes == 128000 * 1024**2


def test_nvidia_smi_fallback_returns_empty_on_command_failure(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError
        return real_import(name, *args, **kwargs)

    def fake_run(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert detect_nvidia_gpus() == []
