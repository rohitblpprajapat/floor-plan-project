"""Configuration dataclass for Stage 07 IMU/camera synchronization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SyncConfig:
    """Settings and hyper-parameters for temporal synchronization and spatial calibration."""

    session_id: str
    run_id: str
    output_dir: Path
    input_manifest_path: Optional[Path] = None
    relative_map_manifest_path: Optional[Path] = None
    camera_imu_csv_path: Optional[Path] = None
    phone_csv_path: Optional[Path] = None
    location_json_path: Optional[Path] = None
    insv_path: Optional[Path] = None

    is_synthetic: bool = False
    synthetic_fixture_type: str = "multi_axis"  # "multi_axis", "timelapse", "periodic_ambiguous", "wrong_units"

    # Search bounds
    search_window_s: float = 2.0
    search_step_s: float = 0.005  # 5 ms coarse search
    fine_search_step_s: float = 0.0005  # 0.5 ms fine optimization

    # Optimization iterations and drift
    max_iterations: int = 200
    enable_drift_estimation: bool = False
    time_reference: str = "exposure_midpoint"

    # Acceptance tolerances
    time_tolerance_s: float = 0.002  # 2 ms
    rotational_tolerance_deg: float = 3.0  # 3 degrees

    # Physical priors
    accel_norm_gravity_prior: float = 9.80665
    gyro_noise_density_prior: float = 0.001  # rad/s/sqrt(Hz) engineering prior
    accel_noise_density_prior: float = 0.01  # m/s^2/sqrt(Hz) engineering prior
