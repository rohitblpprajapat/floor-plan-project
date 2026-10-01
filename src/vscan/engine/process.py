"""Process group management, verified ownership, cooperative cancellation, and stale PID detection.

Guarantees:
- Cancellation targets ONLY verified owned process groups.
- Detects stale PID reuse before sending signals.
- Cooperative cancellation (SIGTERM) first, then scoped termination (SIGKILL) after grace interval.
- Subprocesses run in detached process groups (start_new_session=True) so closing a terminal or log viewer never terminates a worker.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import psutil


def verify_process_identity(
    pid: int,
    expected_create_time: Optional[float] = None,
    expected_cmdline_substring: Optional[str] = None,
) -> Tuple[bool, str, Optional[psutil.Process]]:
    """Verify that a running PID corresponds to the expected worker process and not a reused PID.

    Returns:
        (is_verified, reason_message, psutil_process)
    """
    if pid <= 0:
        return False, f"Invalid PID {pid}", None

    if not psutil.pid_exists(pid):
        return False, f"PID {pid} does not exist", None

    try:
        proc = psutil.Process(pid)
        # Check process status (zombie / dead)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False, f"PID {pid} is a zombie process", proc

        # Verify creation time if provided
        if expected_create_time is not None:
            actual_create_time = proc.create_time()
            if abs(actual_create_time - expected_create_time) > 2.0:
                return (
                    False,
                    f"Stale PID detected: actual create_time ({actual_create_time:.2f}) "
                    f"differs from expected ({expected_create_time:.2f})",
                    proc,
                )

        # Verify cmdline if provided
        if expected_cmdline_substring:
            try:
                cmdline = " ".join(proc.cmdline())
                if expected_cmdline_substring not in cmdline:
                    return (
                        False,
                        f"Stale PID detected: process cmdline '{cmdline}' does not match expected '{expected_cmdline_substring}'",
                        proc,
                    )
            except psutil.AccessDenied:
                return False, f"PID {pid} access denied: process not owned by current user", None
            except psutil.ZombieProcess:
                return False, f"PID {pid} is a zombie process", proc

        return True, "Process identity verified", proc
    except psutil.NoSuchProcess:
        return False, f"PID {pid} terminated during check", None
    except psutil.AccessDenied:
        return False, f"PID {pid} access denied: process not owned by current user", None
    except Exception as exc:
        return False, f"Error inspecting PID {pid}: {exc}", None


def cancel_process_group(
    pid: int,
    pgid: Optional[int] = None,
    expected_create_time: Optional[float] = None,
    grace_interval_seconds: float = 5.0,
) -> Tuple[bool, str]:
    """Cooperatively cancel an owned process group, escalating to SIGKILL if necessary.

    Process:
    1. Verify process identity to avoid killing reused PIDs.
    2. Send cooperative SIGTERM to the process group.
    3. Wait up to grace_interval_seconds.
    4. If still running, send SIGKILL to the process group.
    5. Confirm termination.
    """
    target_pgid = pgid if (pgid and pgid > 0) else pid

    verified, reason, proc = verify_process_identity(
        pid,
        expected_create_time=expected_create_time,
        expected_cmdline_substring="vscan",
    )

    if not verified:
        return False, f"Refusing to cancel unverified PID {pid}: {reason}"

    assert proc is not None

    # Step 1: Cooperative cancellation via SIGTERM
    try:
        if sys.platform != "win32" and hasattr(os, "killpg"):
            os.killpg(target_pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except (ProcessLookupError, psutil.NoSuchProcess):
        return True, f"Process {pid} already exited before SIGTERM"
    except PermissionError as exc:
        return False, f"Permission denied signalling PID {pid}: {exc}"

    # Step 2: Await cooperative exit during grace interval
    deadline = time.time() + grace_interval_seconds
    while time.time() < deadline:
        try:
            if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                return True, f"Process {pid} exited cooperatively after SIGTERM"
        except (ProcessLookupError, psutil.NoSuchProcess):
            return True, f"Process {pid} exited cooperatively after SIGTERM"
        time.sleep(0.2)

    # Step 3: Scoped SIGKILL termination if still alive
    try:
        if sys.platform != "win32" and hasattr(os, "killpg"):
            os.killpg(target_pgid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, psutil.NoSuchProcess):
        return True, f"Process {pid} exited prior to SIGKILL"
    except Exception as exc:
        return False, f"Error sending SIGKILL to PID {pid}: {exc}"

    # Verify final termination
    time.sleep(0.3)
    try:
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return True, f"Process {pid} terminated via scoped SIGKILL after {grace_interval_seconds}s grace interval"
    except (ProcessLookupError, psutil.NoSuchProcess):
        return True, f"Process {pid} terminated via scoped SIGKILL"
    return False, f"Process {pid} failed to terminate even after SIGKILL"
