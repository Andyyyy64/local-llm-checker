"""CPU detection: name, cores, AVX2/AVX512 support."""

from __future__ import annotations

import logging
import platform
import subprocess

logger = logging.getLogger(__name__)


def detect_cpu_name() -> str:
    """Get CPU model name."""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        elif system == "Windows":
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
                if len(lines) > 1:
                    return lines[1]
    except Exception as e:
        logger.debug(f"Failed to detect CPU name: {e}")
    return "Unknown CPU"


def detect_cpu_cores() -> int:
    """Get number of physical CPU cores."""
    import psutil

    return psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1


def _detect_avx_linux() -> tuple[bool, bool]:
    """Detect AVX2/AVX512 on Linux via /proc/cpuinfo."""
    has_avx2 = False
    has_avx512 = False
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read()
            flags_line = ""
            for line in content.split("\n"):
                if line.startswith("flags"):
                    flags_line = line
                    break
            has_avx2 = "avx2" in flags_line
            has_avx512 = "avx512f" in flags_line
    except Exception:
        pass
    return has_avx2, has_avx512


def _detect_avx_darwin() -> tuple[bool, bool]:
    """Detect AVX2/AVX512 on macOS via sysctl."""
    has_avx2 = False
    has_avx512 = False
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.avx2_0"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        has_avx2 = result.stdout.strip() == "1"
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.avx512f"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        has_avx512 = result.stdout.strip() == "1"
    except Exception:
        pass
    return has_avx2, has_avx512


def detect_avx_support() -> tuple[bool, bool]:
    """Detect AVX2 and AVX512 support. Returns (has_avx2, has_avx512)."""
    system = platform.system()
    if system == "Linux":
        return _detect_avx_linux()
    elif system == "Darwin":
        return _detect_avx_darwin()
    # Windows / fallback: assume AVX2 on modern CPUs
    return True, False
