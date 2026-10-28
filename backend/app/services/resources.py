"""Monitoreo de CPU y RAM (host/contenedor)."""
from pathlib import Path

import psutil

from ..config import get_settings


def _read_int(path: str) -> int | None:
    try:
        raw = Path(path).read_text().strip()
        return None if raw == "max" else int(raw)
    except (OSError, ValueError):
        return None


def snapshot() -> dict:
    vm = psutil.virtual_memory()
    limit = _read_int("/sys/fs/cgroup/memory.max") or _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    used = _read_int("/sys/fs/cgroup/memory.current") or _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if limit and limit > vm.total * 2:  # "sin límite" en cgroup v1
        limit = None
    gb = 1024**3
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.2),
        "cpu_count": psutil.cpu_count(),
        "host_ram_total_gb": round(vm.total / gb, 2),
        "host_ram_used_gb": round((vm.total - vm.available) / gb, 2),
        "host_ram_percent": vm.percent,
        "container_ram_used_gb": round(used / gb, 2) if used else None,
        "container_ram_limit_gb": round(limit / gb, 2) if limit else None,
        "platform_ram_budget_gb": get_settings().total_ram_limit_gb,
    }
