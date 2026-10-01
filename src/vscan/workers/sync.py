"""Worker implementation for Stage 07 IMU/camera synchronization and clock calibration.

Stage name: 'sync'
Registered in WORKER_REGISTRY.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from vscan.storage.artifacts import (
    compute_cache_key,
    compute_sha256_dict,
    compute_sha256_file,
)
from vscan.sync.config import SyncConfig
from vscan.sync.pipeline import SyncPipeline
from vscan.workers.base import BaseWorker


class SyncWorker(BaseWorker):
    """Pipeline worker executing the 'sync' stage."""

    stage_name: str = "sync"
    requires_gpu: bool = False

    def execute(self) -> Dict[str, str]:
        print(f"[SyncWorker] Starting synchronization for session={self.session_id}, run_id={self.run_id}", flush=True)
        if self.is_cancellation_requested():
            raise InterruptedError("Cancellation requested before Sync worker execution")

        self.report_progress(2.0, "Initializing Sync worker")

        sync_cfg = self.config.get("sync", {})
        storage_cfg = self.config.get("storage", {})

        sessions_dir = Path(storage_cfg.get("sessions_dir", "data/sessions")) / self.session_id
        sessions_dir.mkdir(parents=True, exist_ok=True)

        is_synthetic = bool(sync_cfg.get("synthetic_fixture", False))
        fixture_type = sync_cfg.get("synthetic_fixture_type", "multi_axis")

        # Resolve paths for real session
        input_manifest_path = None
        rel_map_manifest_path = None
        cam_csv_path = None
        phone_csv_path = None
        loc_json_path = None

        if not is_synthetic:
            # Check session directory
            cand_man = sessions_dir / "input_manifest.json"
            if cand_man.is_file():
                input_manifest_path = cand_man
            
            cand_rel = sessions_dir / "relative_map_manifest.json"
            if cand_rel.is_file():
                rel_map_manifest_path = cand_rel

            repo_root = Path(__file__).resolve().parents[3]
            cand_cam = sessions_dir / "inputs" / "insta360_export_1789041643971.csv"
            if not cand_cam.is_file():
                cand_cam = Path("data/2026-09-10/insta360_export_1789041643971.csv")
            if not cand_cam.is_file():
                cand_cam = repo_root / "data" / "2026-09-10" / "insta360_export_1789041643971.csv"
            cam_csv_path = cand_cam if cand_cam.is_file() else None

            cand_phone = sessions_dir / "inputs" / "20260910_172939_phone_sensors.csv"
            if not cand_phone.is_file():
                cand_phone = Path("data/2026-09-10/20260910_172939_phone_sensors.csv")
            if not cand_phone.is_file():
                cand_phone = repo_root / "data" / "2026-09-10" / "20260910_172939_phone_sensors.csv"
            phone_csv_path = cand_phone if cand_phone.is_file() else None

            cand_loc = sessions_dir / "inputs" / "20260910_172939_location.json"
            if not cand_loc.is_file():
                cand_loc = Path("data/2026-09-10/20260910_172939_location.json")
            if not cand_loc.is_file():
                cand_loc = repo_root / "data" / "2026-09-10" / "20260910_172939_location.json"
            loc_json_path = cand_loc if cand_loc.is_file() else None

        pipeline_cfg = SyncConfig(
            session_id=self.session_id,
            run_id=self.run_id,
            output_dir=self.outputs_dir,
            input_manifest_path=input_manifest_path,
            relative_map_manifest_path=rel_map_manifest_path,
            camera_imu_csv_path=cam_csv_path,
            phone_csv_path=phone_csv_path,
            location_json_path=loc_json_path,
            is_synthetic=is_synthetic,
            synthetic_fixture_type=fixture_type,
            search_window_s=float(sync_cfg.get("search_window_s", 2.0)),
            search_step_s=float(sync_cfg.get("search_step_s", 0.005)),
            fine_search_step_s=float(sync_cfg.get("fine_search_step_s", 0.0005)),
            enable_drift_estimation=bool(sync_cfg.get("enable_drift_estimation", False)),
            time_reference=str(sync_cfg.get("time_reference", "exposure_midpoint")),
        )

        def _progress_cb(pct: float, step: str) -> None:
            self.report_progress(pct, step)

        pipeline = SyncPipeline(pipeline_cfg)
        res = pipeline.run(
            progress_callback=_progress_cb,
            cancel_checker=self.is_cancellation_requested,
        )

        if self.is_cancellation_requested():
            raise InterruptedError("Cancellation requested during Sync worker execution")

        # Copy primary sync artifacts to session directory
        sync_man_src = self.outputs_dir / "sync_manifest.json"
        sync_man_dst = sessions_dir / "sync_manifest.json"
        if sync_man_src.is_file():
            shutil.copy2(sync_man_src, sync_man_dst)

        calib_src = self.outputs_dir / "extrinsic_calibration.json"
        if calib_src.is_file():
            shutil.copy2(calib_src, sessions_dir / "extrinsic_calibration.json")

        clock_src = self.outputs_dir / "clock_calibration.json"
        if clock_src.is_file():
            shutil.copy2(clock_src, sessions_dir / "clock_calibration.json")

        outputs = {
            "sync_manifest": "sync_manifest.json",
            "diagnostics_report": "diagnostics_report.json",
        }
        if (self.outputs_dir / "timing_diagnostics.json").is_file():
            outputs["timing_diagnostics"] = "timing_diagnostics.json"
        if (self.outputs_dir / "clock_calibration.json").is_file():
            outputs["clock_calibration"] = "clock_calibration.json"
        if (self.outputs_dir / "extrinsic_calibration.json").is_file():
            outputs["extrinsic_calibration"] = "extrinsic_calibration.json"

        return outputs
