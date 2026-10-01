"""Failure injection worker for rigorous error handling and lifecycle testing."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict

from vscan.workers.base import BaseWorker


class FailureInjectionWorker(BaseWorker):
    """Failure injection worker to test recovery, timeout, cancellation, and corruption."""

    stage_name: str = "fail_test"
    requires_gpu: bool = False

    def execute(self) -> Dict[str, str]:
        mode = self.config.get("fail_mode", "crash")
        print(f"[FailureInjectionWorker] Starting in mode='{mode}' for run_id={self.run_id}")

        if mode == "crash":
            self.report_progress(10.0, "Triggering deliberate failure injection", 1, 1)
            raise RuntimeError("Deliberate worker failure injected for test verification")

        elif mode == "exit_code":
            self.report_progress(10.0, "Deliberately exiting with non-zero code", 1, 1)
            sys.exit(42)

        elif mode == "hang":
            # Loop forever or until cancelled/timed out
            self.report_progress(20.0, "Entering infinite loop to test timeout/cancel", 1, 1)
            while True:
                time.sleep(0.5)
                if self.is_cancellation_requested():
                    raise InterruptedError("Deliberate hang cancelled by supervisor")

        elif mode == "corrupt_after_write":
            # Produce output, then corrupt it
            out_path = self.outputs_dir / "test_artifact.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("Original valid content before corruption\n")
            # We return output path so caller can write completion marker, but wait...
            # In runner, we write completion marker. If config says corrupt_post_marker,
            # runner or test will corrupt it.
            return {"artifact": "test_artifact.txt"}

        else:
            raise ValueError(f"Unknown fail_mode: {mode}")
