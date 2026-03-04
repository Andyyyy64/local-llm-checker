"""Apple Silicon detection via system_profiler."""

from __future__ import annotations

import json
import logging
import subprocess

from local_llm_checker.constants import GPU_BANDWIDTH
from local_llm_checker.hardware.types import GPUInfo

logger = logging.getLogger(__name__)


def _lookup_bandwidth(chip_name: str) -> float | None:
    chip_upper = chip_name.upper()
    for key in sorted(GPU_BANDWIDTH, key=len, reverse=True):
        if key.upper() in chip_upper:
            return GPU_BANDWIDTH[key]
    return None


def detect_apple_gpu() -> list[GPUInfo]:
    """Detect Apple Silicon GPU. Returns empty list on non-macOS or failure."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        logger.debug("system_profiler not available (not macOS)")
        return []

    try:
        hw_items = data["SPHardwareDataType"]
        hw = hw_items[0]
        chip_name = hw.get("chip_type", "")
        if not chip_name:
            return []

        # Apple Silicon uses unified memory - get total physical memory
        memory_str = hw.get("physical_memory", "0 GB")
        # Parse "32 GB" -> bytes
        parts = memory_str.split()
        mem_value = int(parts[0])
        mem_unit = parts[1].upper() if len(parts) > 1 else "GB"
        multiplier = {"GB": 1024**3, "TB": 1024**4, "MB": 1024**2}.get(
            mem_unit, 1024**3
        )
        unified_memory = mem_value * multiplier

        return [
            GPUInfo(
                name=chip_name,
                vendor="apple",
                vram_bytes=unified_memory,  # unified memory
                memory_bandwidth_gbps=_lookup_bandwidth(chip_name),
            )
        ]
    except (KeyError, IndexError, ValueError) as e:
        logger.debug(f"Failed to parse Apple hardware info: {e}")
        return []
