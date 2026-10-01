"""Telemetry collection for host, WSL2, and GPU resources.

Rule:
- Record limited/unavailable WSL telemetry without fabricating memory readings.
- Inside WSL, report actual /proc/meminfo and psutil memory for the WSL instance.
- Explicitly mark host RAM as not directly measurable inside WSL rather than fabricating numbers.
- Query nvidia-smi for GPU memory and utilization; report unavailable if missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, Optional
import psutil


def get_wsl_memory_telemetry() -> Dict[str, Any]:
    """Read actual memory and swap from the running environment."""
    vmem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "wsl_ram_total_bytes": vmem.total,
        "wsl_ram_available_bytes": vmem.available,
        "wsl_ram_used_bytes": vmem.used,
        "wsl_ram_percent": vmem.percent,
        "wsl_swap_total_bytes": swap.total,
        "wsl_swap_free_bytes": swap.free,
        "wsl_swap_used_bytes": swap.used,
        "wsl_swap_percent": swap.percent,
    }


def get_host_telemetry() -> Dict[str, Any]:
    """Report host-level availability.

    Inside WSL2, Windows host RAM is isolated. We document this constraint
    honestly without fabricating memory readings.
    """
    is_wsl = os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop") or "microsoft" in os.uname().release.lower()
    if is_wsl:
        return {
            "available": False,
            "reason": "Host RAM metrics not exported inside WSL2 virtual machine boundary; fabrication prohibited",
        }
    else:
        # Running directly on host OS
        vmem = psutil.virtual_memory()
        return {
            "available": True,
            "host_ram_total_bytes": vmem.total,
            "host_ram_available_bytes": vmem.available,
        }


def get_gpu_telemetry() -> Dict[str, Any]:
    """Query live GPU telemetry via nvidia-smi without fabricating data."""
    smi_path = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    if not os.path.isfile(smi_path) and not shutil.which("nvidia-smi"):
        return {
            "gpu_available": False,
            "reason": "nvidia-smi not found in PATH or /usr/lib/wsl/lib/",
        }

    cmd = [
        smi_path,
        "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3.0)
        if res.returncode != 0:
            return {
                "gpu_available": False,
                "reason": f"nvidia-smi exited with code {res.returncode}: {res.stderr.strip()}",
            }
        line = res.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            return {
                "gpu_available": True,
                "gpu_name": parts[0],
                "driver_version": parts[1],
                "gpu_memory_total_mib": float(parts[2]),
                "gpu_memory_used_mib": float(parts[3]),
                "gpu_memory_free_mib": float(parts[4]),
                "gpu_utilization_percent": float(parts[5]),
            }
        return {"gpu_available": True, "raw_output": line}
    except Exception as exc:
        return {
            "gpu_available": False,
            "reason": f"Failed to query nvidia-smi: {exc}",
        }


def collect_full_telemetry() -> Dict[str, Any]:
    """Collect full telemetry snapshot."""
    return {
        "wsl_memory": get_wsl_memory_telemetry(),
        "host_memory": get_host_telemetry(),
        "gpu": get_gpu_telemetry(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_percent": psutil.cpu_percent(interval=None),
    }
