"""Deterministic short smoke worker."""

from __future__ import annotations

import json
import time
from typing import Any, Dict

from vscan.workers.base import BaseWorker


class SmokeWorker(BaseWorker):
    """Deterministic short smoke worker.

    Executes a bounded synthetic test:
    - Reports initial progress
    - Emits synthetic session manifest and frame count
    - Writes progress update
    - Produces manifest.json and summary.txt
    - Guarantees fast, reproducible execution
    """

    stage_name: str = "smoke"
    requires_gpu: bool = False

    def execute(self) -> Dict[str, str]:
        print(f"[SmokeWorker] Starting execution for session={self.session_id}, run_id={self.run_id}")
        self.report_progress(10.0, "Initializing smoke test", 1, 4)

        time.sleep(0.3)
        if self.is_cancellation_requested():
            raise InterruptedError("SmokeWorker cancelled during initialization")

        self.report_progress(50.0, "Processing synthetic fixtures", 2, 4)
        manifest_data = {
            "session_id": self.session_id,
            "stage": self.stage_name,
            "run_id": self.run_id,
            "synthetic_frames_count": 10,
            "clock_mapping": {"t0_raw_ns": 1000000000, "scale_factor": 1e-9},
            "status": "PROCESSED",
        }
        manifest_path = self.outputs_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        time.sleep(0.3)
        if self.is_cancellation_requested():
            raise InterruptedError("SmokeWorker cancelled during processing")

        self.report_progress(80.0, "Writing summary report", 3, 4)
        summary_path = self.outputs_dir / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Smoke worker completed successfully for session {self.session_id}\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Outputs generated: manifest.json, summary.txt\n")

        self.report_progress(100.0, "Completed", 4, 4)
        print(f"[SmokeWorker] Successfully completed smoke run {self.run_id}")

        return {
            "manifest": "manifest.json",
            "summary": "summary.txt",
        }
