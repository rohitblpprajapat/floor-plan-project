"""Configuration dataclass and option builders for the SfM pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import pycolmap


@dataclass
class SfmConfig:
    """Configuration parameters for PyCOLMAP Structure from Motion (SfM) execution."""

    session_id: str
    run_id: str
    output_dir: Path

    # Input artifacts
    input_manifest_path: Optional[Path] = None
    rig_config_path: Optional[Path] = None
    images_dir: Optional[Path] = None
    masks_dir: Optional[Path] = None

    # Hardware & concurrency constraints (laptop GTX 1650 4GB budget)
    device_requested: str = "cuda"  # "cuda" or "cpu"
    num_threads: int = 6  # Worker CPU thread budget from laptop.yaml
    gpu_index: int = 0
    allow_cpu_fallback: bool = False  # Strict device enforcement; no silent downgrades

    # SIFT feature extraction
    max_num_features: int = 4096  # Bounded feature budget for 4GB VRAM
    peak_threshold: float = 0.006666666666666667
    edge_threshold: float = 10.0
    use_masks: bool = True

    # Feature matching & geometric verification
    temporal_overlap: int = 10  # Sequential temporal window (K stations)
    quadratic_overlap: bool = True
    enable_loop_detection: bool = False  # Bounded revisit candidates
    revisit_candidate_budget: int = 20
    min_inlier_matches: int = 15
    max_error_px: float = 4.0

    # Reconstruction & Bundle Adjustment
    fix_intrinsics: bool = True  # Strict: known virtual-view intrinsics kept fixed
    fix_rig_extrinsics: bool = True  # Strict: relative poses fixed in rig
    init_min_tri_angle: float = 4.0
    init_min_num_inliers: int = 15
    abs_pose_min_num_inliers: int = 12
    filter_max_reproj_error: float = 4.0
    min_model_size: int = 5
    multiple_models: bool = True  # Retain all disconnected components

    # Synthetic fixture execution
    is_synthetic: bool = False
    synthetic_stations_count: int = 5
    synthetic_pixel_residual_tolerance_px: float = 4.0
    synthetic_rmse_camera_threshold_pct: float = 2.0  # 2% of room diagonal

    @classmethod
    def from_dict(cls, session_id: str, run_id: str, output_dir: Path, cfg: Dict[str, Any]) -> "SfmConfig":
        sfm_dict = cfg.get("sfm", {})
        system_dict = cfg.get("concurrency", {})
        threads = int(sfm_dict.get("threads", system_dict.get("worker_cpu_threads", 6)))

        return cls(
            session_id=session_id,
            run_id=run_id,
            output_dir=output_dir,
            input_manifest_path=Path(sfm_dict["input_manifest_path"]) if "input_manifest_path" in sfm_dict else None,
            rig_config_path=Path(sfm_dict["rig_config_path"]) if "rig_config_path" in sfm_dict else None,
            images_dir=Path(sfm_dict["images_dir"]) if "images_dir" in sfm_dict else None,
            masks_dir=Path(sfm_dict["masks_dir"]) if "masks_dir" in sfm_dict else None,
            device_requested=str(sfm_dict.get("device", "cuda")).lower(),
            num_threads=threads,
            gpu_index=int(sfm_dict.get("gpu_index", 0)),
            allow_cpu_fallback=bool(sfm_dict.get("allow_cpu_fallback", False)),
            max_num_features=int(sfm_dict.get("max_num_features", 4096)),
            temporal_overlap=int(sfm_dict.get("temporal_overlap", 10)),
            quadratic_overlap=bool(sfm_dict.get("quadratic_overlap", True)),
            enable_loop_detection=bool(sfm_dict.get("enable_loop_detection", False)),
            revisit_candidate_budget=int(sfm_dict.get("revisit_candidate_budget", 20)),
            fix_intrinsics=bool(sfm_dict.get("fix_intrinsics", True)),
            fix_rig_extrinsics=bool(sfm_dict.get("fix_rig_extrinsics", True)),
            is_synthetic=bool(sfm_dict.get("synthetic_fixture", False)),
        )
