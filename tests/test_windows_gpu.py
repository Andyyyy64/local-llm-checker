"""Tests for Windows GPU detection."""

from __future__ import annotations

import json
import subprocess

from whichllm.hardware import windows


def test_detect_windows_amd_gpu(monkeypatch):
    output = json.dumps(
        {
            "Name": "AMD Radeon RX 9060 XT",
            "AdapterRAM": 16 * 1024**3,
        }
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=output, stderr="")

    monkeypatch.setattr(windows.subprocess, "run", fake_run)

    gpus = windows.detect_windows_gpus()

    assert len(gpus) == 1
    assert gpus[0].name == "AMD Radeon RX 9060 XT"
    assert gpus[0].vendor == "amd"
    assert gpus[0].vram_bytes == 16 * 1024**3
    assert gpus[0].shared_memory is False
    assert gpus[0].memory_bandwidth_gbps == 320.0


def test_detect_windows_amd_gpu_prefers_64_bit_dedicated_memory(monkeypatch):
    output = json.dumps(
        {
            "Name": "AMD Radeon RX 9060 XT",
            "AdapterRAM": 4 * 1024**3,
            "DedicatedVideoMemory": 16 * 1024**3,
        }
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=output, stderr="")

    monkeypatch.setattr(windows.subprocess, "run", fake_run)

    gpus = windows.detect_windows_gpus()

    assert len(gpus) == 1
    assert gpus[0].vram_bytes == 16 * 1024**3
    assert gpus[0].shared_memory is False


def test_detect_windows_gpu_queries_control_video_qword_memory(monkeypatch):
    captured: dict[str, str] = {}

    def fake_run(*args, **kwargs):
        command = args[0]
        captured["script"] = command[-1]
        return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")

    monkeypatch.setattr(windows.subprocess, "run", fake_run)

    assert windows.detect_windows_gpus() == []
    assert "Control\\Video" in captured["script"]
    assert "VideoID" in captured["script"]
    assert "HardwareInformation.qwMemorySize" in captured["script"]


def test_detect_windows_amd_gpu_applies_known_floor_for_capped_adapter_ram(
    monkeypatch,
):
    output = json.dumps(
        {
            "Name": "AMD Radeon RX 9060 XT",
            "AdapterRAM": 4 * 1024**3,
            "DedicatedVideoMemory": None,
        }
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=output, stderr="")

    monkeypatch.setattr(windows.subprocess, "run", fake_run)

    gpus = windows.detect_windows_gpus()

    assert len(gpus) == 1
    assert gpus[0].vram_bytes == 8 * 1024**3
    assert gpus[0].shared_memory is False


def test_detect_windows_ryzen_ai_radeon_890m_as_shared_memory(monkeypatch):
    output = json.dumps(
        {
            "Name": "AMD Ryzen AI 9 HX 370 w/ Radeon 890M",
            "AdapterRAM": 512 * 1024**2,
            "DedicatedVideoMemory": None,
        }
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=output, stderr="")

    monkeypatch.setattr(windows.subprocess, "run", fake_run)

    gpus = windows.detect_windows_gpus()

    assert len(gpus) == 1
    assert gpus[0].name == "AMD Ryzen AI 9 HX 370 w/ Radeon 890M"
    assert gpus[0].vendor == "amd"
    assert gpus[0].vram_bytes == 0
    assert gpus[0].shared_memory is True
    assert gpus[0].memory_bandwidth_gbps == 120.0


def test_detect_windows_intel_gpu_list(monkeypatch):
    output = json.dumps(
        [
            {
                "Name": "Intel(R) Arc(TM) Graphics",
                "AdapterRAM": None,
            },
            {
                "Name": "Microsoft Basic Display Adapter",
                "AdapterRAM": 0,
            },
        ]
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=output, stderr="")

    monkeypatch.setattr(windows.subprocess, "run", fake_run)

    gpus = windows.detect_windows_gpus()

    assert len(gpus) == 1
    assert gpus[0].name == "Intel(R) Arc(TM) Graphics"
    assert gpus[0].vendor == "intel"
    assert gpus[0].vram_bytes == 0
    assert gpus[0].shared_memory is True


def test_detect_windows_intel_arc_discrete_is_not_shared_memory(monkeypatch):
    output = json.dumps(
        {
            "Name": "Intel(R) Arc(TM) B580 Graphics",
            "AdapterRAM": 4 * 1024**3,
            "DedicatedVideoMemory": 12 * 1024**3,
        }
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=output, stderr="")

    monkeypatch.setattr(windows.subprocess, "run", fake_run)

    gpus = windows.detect_windows_gpus()

    assert len(gpus) == 1
    assert gpus[0].name == "Intel(R) Arc(TM) B580 Graphics"
    assert gpus[0].vendor == "intel"
    assert gpus[0].vram_bytes == 12 * 1024**3
    assert gpus[0].shared_memory is False


def test_detect_windows_gpu_returns_empty_on_command_failure(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(windows.subprocess, "run", fake_run)

    assert windows.detect_windows_gpus() == []
