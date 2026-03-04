"""RAM and disk space detection."""

from __future__ import annotations

import shutil

import psutil


def detect_ram_bytes() -> int:
    """Get total physical RAM in bytes."""
    return psutil.virtual_memory().total


def detect_disk_free_bytes(path: str = "/") -> int:
    """Get free disk space in bytes at the given path."""
    try:
        usage = shutil.disk_usage(path)
        return usage.free
    except OSError:
        return 0
