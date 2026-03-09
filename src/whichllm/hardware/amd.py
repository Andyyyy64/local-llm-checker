"""AMD GPU detection via rocm-smi."""

from __future__ import annotations

import json
import logging
import subprocess

from whichllm.constants import GPU_BANDWIDTH
from whichllm.hardware.types import GPUInfo

logger = logging.getLogger(__name__)


def _lookup_bandwidth(name: str) -> float | None:
    name_upper = name.upper()
    for key in sorted(GPU_BANDWIDTH, key=len, reverse=True):
        if key.upper() in name_upper:
            return GPU_BANDWIDTH[key]
    return None


def detect_amd_gpus() -> list[GPUInfo]:
    """Detect AMD GPUs using rocm-smi. Returns empty list on failure."""
    gpus: list[GPUInfo] = []

    # Get product names
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        product_data = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        logger.debug("rocm-smi not available or failed")
        return []

    # Get VRAM info
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        mem_data = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        logger.debug("Failed to get AMD VRAM info")
        return []

    # Get ROCm version
    rocm_version = None
    try:
        result = subprocess.run(
            ["rocm-smi", "--showdriverversion", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            driver_data = json.loads(result.stdout)
            # Extract version from first card entry
            for key, val in driver_data.items():
                if isinstance(val, dict) and "Driver version" in val:
                    rocm_version = val["Driver version"]
                    break
    except Exception:
        pass

    # Parse GPU info - rocm-smi JSON keys are like "card0", "card1"
    for key in sorted(product_data.keys()):
        if not key.startswith("card"):
            continue
        card_info = product_data[key]
        name = card_info.get(
            "Card SKU", card_info.get("Card series", "Unknown AMD GPU")
        )

        vram_total = 0
        if key in mem_data:
            vram_str = mem_data[key].get("VRAM Total Memory (B)", "0")
            try:
                vram_total = int(vram_str)
            except (ValueError, TypeError):
                pass

        gpus.append(
            GPUInfo(
                name=name,
                vendor="amd",
                vram_bytes=vram_total,
                rocm_version=rocm_version,
                memory_bandwidth_gbps=_lookup_bandwidth(name),
            )
        )

    return gpus
