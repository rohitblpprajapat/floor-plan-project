"""Stage worker for Structure from Motion (SfM) relative map reconstruction.

Stage: 'sfm'
Registered in WORKER_REGISTRY.
Adheres strictly to all build contracts:
- Wraps pinned PyCOLMAP 4.2.0 feature extraction, matching, geometric verification, and reconstruction.
- Uses laptop GPU (GTX 1650 4GB) with single GPU constraint and 6-thread budget.
- Strict device selection: no silent CPU fallbacks when CUDA is requested.
- Preserves raw and derived frame lineage; scale status strictly NOT_ESTIMATED.
- Configures rig membership before matching and mapping; virtual-view intrinsics and relative poses fixed.
- Evaluates temporal neighbors and bounded revisit candidates; rejects unverified loops.
- Exports explicit camera-to-world and source-view relationships; inverts COLMAP transforms.
- Retains all disconnected components and identifies primary pilot component.
- Supports cooperative cancellation and robust resumption.
- Produces immutable relative map manifest with SHA-256 artifact hashes.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Dict, Optional

from vscan.schemas.sfm import RelativeMapManifestSchema
from vscan.sfm.config import SfmConfig
from vscan.sfm.pipeline import SfmPipeline
from vscan.storage.artifacts import (
    compute_cache_key,
    compute_sha256_dict,
    compute_sha256_file,
)
from vscan.workers.base import BaseWorker


class SfmWorker(BaseWorker):
    """Pipeline worker executing the 'sfm' stage."""

    stage_name: str = "sfm"
    requires_gpu: bool = True

    def execute(self) -> Dict[str, str]:
        print(f"[SfmWorker] Starting SfM relative reconstruction for session={self.session_id}, run_id={self.run_id}", flush=True)
        self.report_progress(2.0, "Initializing SfM worker")

        sfm_cfg = self.config.get("sfm", {})
        storage_cfg = self.config.get("storage", {})
        concurrency_cfg = self.config.get("concurrency", {})

        sessions_dir = Path(storage_cfg.get("sessions_dir", "data/sessions")) / self.session_id
        sessions_dir.mkdir(parents=True, exist_ok=True)

        is_synthetic = bool(sfm_cfg.get("synthetic_fixture", False))

        # 1. Resolve inputs
        input_manifest_path = None
        rig_config_path = None
        images_dir = None
        masks_dir = None
        source_manifest_hash = "synthetic"
        rig_config_hash = "synthetic"

        if not is_synthetic:
            # Look up pilot manifest and rig config from session directory or config
            cand_manifest = sfm_cfg.get("pilot_manifest_path")
            if cand_manifest and Path(cand_manifest).is_file():
                input_manifest_path = Path(cand_manifest)
            else:
                input_manifest_path = sessions_dir / "pilot_manifest.json"

            cand_rig = sfm_cfg.get("rig_config_path")
            if cand_rig and Path(cand_rig).is_file():
                rig_config_path = Path(cand_rig)
            else:
                rig_config_path = sessions_dir / "rig_config.json"

            if not input_manifest_path.is_file() or not rig_config_path.is_file():
                # Check previous frames runs
                runs_dir = Path(storage_cfg.get("runs_dir", "runs"))
                frames_runs = sorted(runs_dir.glob("run_frames_*"), reverse=True)
                for fr in frames_runs:
                    m = fr / "outputs" / "pilot_manifest.json"
                    r = fr / "outputs" / "rig_config.json"
                    if m.is_file() and r.is_file():
                        input_manifest_path = m
                        rig_config_path = r
                        images_dir = fr / "outputs" / "pilot_frames"
                        masks_dir = fr / "outputs" / "masks"
                        break

            if not input_manifest_path or not input_manifest_path.is_file():
                raise FileNotFoundError(
                    f"Required pilot manifest not found for session '{self.session_id}'. "
                    "Ensure 'frames' stage has been executed and outputs are available."
                )

            source_manifest_hash = compute_sha256_file(input_manifest_path)
            rig_config_hash = compute_sha256_file(rig_config_path)

        # 2. Check Cache
        config_hash = compute_sha256_dict(sfm_cfg)
        cache_key = compute_cache_key(
            "sfm",
            config_hash,
            {"source_manifest": source_manifest_hash, "rig_config": rig_config_hash},
        )

        # 3. Setup SfmConfig
        thread_budget = int(sfm_cfg.get("threads", concurrency_cfg.get("worker_cpu_threads", 6)))
        device_req = str(sfm_cfg.get("device", "cuda" if not is_synthetic else "cpu")).lower()

        pipeline_cfg = SfmConfig(
            session_id=self.session_id,
            run_id=self.run_id,
            output_dir=self.outputs_dir,
            input_manifest_path=input_manifest_path,
            rig_config_path=rig_config_path,
            images_dir=images_dir,
            masks_dir=masks_dir,
            device_requested=device_req,
            num_threads=thread_budget,
            gpu_index=int(sfm_cfg.get("gpu_index", 0)),
            allow_cpu_fallback=bool(sfm_cfg.get("allow_cpu_fallback", False)),
            max_num_features=int(sfm_cfg.get("max_num_features", 4096)),
            temporal_overlap=int(sfm_cfg.get("temporal_overlap", 10)),
            quadratic_overlap=bool(sfm_cfg.get("quadratic_overlap", True)),
            enable_loop_detection=bool(sfm_cfg.get("enable_loop_detection", False)),
            revisit_candidate_budget=int(sfm_cfg.get("revisit_candidate_budget", 20)),
            fix_intrinsics=bool(sfm_cfg.get("fix_intrinsics", True)),
            fix_rig_extrinsics=bool(sfm_cfg.get("fix_rig_extrinsics", True)),
            is_synthetic=is_synthetic,
            synthetic_stations_count=int(sfm_cfg.get("synthetic_stations_count", 5)),
        )

        def _progress_cb(pct: float, step: str) -> None:
            self.report_progress(pct, step)

        pipeline = SfmPipeline(pipeline_cfg)
        res = pipeline.run(
            progress_callback=_progress_cb,
            cancel_checker=self.is_cancellation_requested,
        )

        # Copy accepted relative map manifest to session directory
        manifest_src = self.outputs_dir / "relative_map_manifest.json"
        session_manifest_dst = sessions_dir / "relative_map_manifest.json"
        if manifest_src.is_file():
            shutil.copy2(manifest_src, session_manifest_dst)

        return {
            "relative_map_manifest": "relative_map_manifest.json",
            "database": "database.db",
        }
