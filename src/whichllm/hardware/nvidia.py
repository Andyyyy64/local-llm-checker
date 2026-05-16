"""NVIDIA GPU detection via NVML with nvidia-smi fallback."""

from __future__ import annotations

import logging
import re
import subprocess

from whichllm.constants import GPU_BANDWIDTH, NVIDIA_COMPUTE_CAPABILITY
from whichllm.hardware.types import GPUInfo

logger = logging.getLogger(__name__)


def _lookup_compute_capability(name: str) -> tuple[int, int] | None:
    name_upper = name.upper()
    for key, cc in NVIDIA_COMPUTE_CAPABILITY.items():
        if key.upper() in name_upper:
            return cc
    return None


def _lookup_bandwidth(name: str) -> float | None:
    name_upper = name.upper()
    # Try longer matches first (e.g. "RTX 4080 SUPER" before "RTX 4080")
    for key in sorted(GPU_BANDWIDTH, key=len, reverse=True):
        if key.upper() in name_upper:
            return GPU_BANDWIDTH[key]
    return None


def _detect_nvidia_gpus_via_smi() -> list[GPUInfo]:
    """Detect NVIDIA GPUs using nvidia-smi when Python NVML cannot load."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
        logger.debug(f"nvidia-smi fallback failed: {e}")
        return []

    gpus: list[GPUInfo] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        if len(parts) != 2 or not parts[0]:
            continue

        name, memory_mib_text = parts
        match = re.search(r"\d+", memory_mib_text)
        if not match:
            logger.debug(f"Could not parse nvidia-smi memory value: {line!r}")
            continue

        memory_mib = int(match.group(0))
        gpus.append(
            GPUInfo(
                name=name,
                vendor="nvidia",
                vram_bytes=memory_mib * 1024**2,
                compute_capability=_lookup_compute_capability(name),
                memory_bandwidth_gbps=_lookup_bandwidth(name),
            )
        )

    return gpus


def detect_nvidia_gpus() -> list[GPUInfo]:
    """Detect NVIDIA GPUs. Returns empty list on failure."""
    try:
        import pynvml
    except ImportError:
        logger.debug("pynvml not installed, trying nvidia-smi fallback")
        return _detect_nvidia_gpus_via_smi()

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        logger.debug("NVML init failed, trying nvidia-smi fallback")
        return _detect_nvidia_gpus_via_smi()

    gpus: list[GPUInfo] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        # Get CUDA driver version
        try:
            pynvml.nvmlSystemGetDriverVersion()  # ensure driver is accessible
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda_str = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
        except Exception:
            cuda_str = None

        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            gpus.append(
                GPUInfo(
                    name=name,
                    vendor="nvidia",
                    vram_bytes=mem_info.total,
                    compute_capability=_lookup_compute_capability(name),
                    cuda_version=cuda_str,
                    memory_bandwidth_gbps=_lookup_bandwidth(name),
                )
            )
    except pynvml.NVMLError as e:
        logger.debug(f"Error enumerating NVIDIA GPUs: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    if gpus:
        return gpus

    logger.debug("NVML returned no NVIDIA GPUs, trying nvidia-smi fallback")
    return _detect_nvidia_gpus_via_smi()
