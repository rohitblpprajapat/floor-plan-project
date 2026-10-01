"""GPU-labelled smoke worker to test exclusive GPU concurrency queue."""

from __future__ import annotations

import json
import time
from typing import Dict

from vscan.workers.base import BaseWorker


class GPUSmokeWorker(BaseWorker):
    """GPU-labelled synthetic smoke worker.

    Declares requires_gpu = True.
    Used to test that only one GPU job can execute concurrently across project workers.
    """

    stage_name: str = "gpu_smoke"
    requires_gpu: bool = True

    def execute(self) -> Dict[str, str]:
        print(f"[GPUSmokeWorker] Starting GPU stage execution for session={self.session_id}, run_id={self.run_id}")
        self.report_progress(10.0, "Allocating GPU queue slot", 1, 3)

        duration = float(self.config.get("gpu_smoke_duration_seconds", 1.5))
        steps = 5
        dt = duration / steps
        for step in range(1, steps + 1):
            time.sleep(dt)
            if self.is_cancellation_requested():
                raise InterruptedError("GPUSmokeWorker cancelled during execution")
            pct = 10.0 + (step / steps) * 80.0
            self.report_progress(pct, f"Simulated GPU computation step {step}/{steps}", step, steps)

        output_data = {
            "session_id": self.session_id,
            "stage": self.stage_name,
            "run_id": self.run_id,
            "gpu_slot_exclusive": True,
            "status": "SUCCEEDED",
        }
        gpu_meta_path = self.outputs_dir / "gpu_result.json"
        with open(gpu_meta_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        self.report_progress(100.0, "GPU execution completed", steps, steps)
        print(f"[GPUSmokeWorker] Finished GPU stage run {self.run_id}")

        return {
            "gpu_result": "gpu_result.json",
        }
