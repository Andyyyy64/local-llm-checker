"""Unified hardware detection orchestrator."""

from __future__ import annotations

import logging
import platform

from whatllm.hardware.amd import detect_amd_gpus
from whatllm.hardware.apple import detect_apple_gpu
from whatllm.hardware.cpu import detect_avx_support, detect_cpu_cores, detect_cpu_name
from whatllm.hardware.memory import detect_disk_free_bytes, detect_ram_bytes
from whatllm.hardware.nvidia import detect_nvidia_gpus
from whatllm.hardware.types import HardwareInfo

logger = logging.getLogger(__name__)


def detect_hardware() -> HardwareInfo:
    """Detect all hardware. Each detector is fail-safe (returns empty on error)."""
    os_name = platform.system().lower()
    if os_name not in ("linux", "darwin", "windows"):
        os_name = "linux"

    # GPU detection
    gpus = []
    gpus.extend(detect_nvidia_gpus())
    if os_name == "linux":
        gpus.extend(detect_amd_gpus())
    if os_name == "darwin":
        gpus.extend(detect_apple_gpu())

    # CPU
    cpu_name = detect_cpu_name()
    cpu_cores = detect_cpu_cores()
    has_avx2, has_avx512 = detect_avx_support()

    # Memory
    ram_bytes = detect_ram_bytes()
    disk_free = detect_disk_free_bytes()

    return HardwareInfo(
        gpus=gpus,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        has_avx2=has_avx2,
        has_avx512=has_avx512,
        ram_bytes=ram_bytes,
        disk_free_bytes=disk_free,
        os=os_name,
    )
