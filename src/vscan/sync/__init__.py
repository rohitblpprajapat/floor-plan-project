"""Stage 07 IMU/camera synchronization, clock calibration, and timing diagnostics."""

from vscan.sync.config import SyncConfig
from vscan.sync.pipeline import SyncPipeline
from vscan.sync.rotation import (
    evaluate_extrinsic_residual,
    integrate_imu_angular_velocity,
    so3_exp,
    so3_geodesic_distance,
    so3_hat,
    so3_log,
    so3_vee,
    solve_wahba_svd,
)
from vscan.sync.synchronizer import (
    SynchronizationResult,
    estimate_camera_imu_synchronization,
)
from vscan.sync.synthetic import (
    generate_periodic_ambiguous_fixture,
    generate_synthetic_multi_axis_sync_fixture,
    generate_timelapse_sync_fixture,
)
from vscan.sync.timing import (
    PlaybackToCaptureMapping,
    RawClockDomain,
    analyze_16_sample_boundary_pattern,
)
from vscan.sync.units import (
    build_full_unit_hypothesis,
    evaluate_accelerometer_units,
    evaluate_gyroscope_units,
    evaluate_timestamp_units,
)

__all__ = [
    "SyncConfig",
    "SyncPipeline",
    "SynchronizationResult",
    "estimate_camera_imu_synchronization",
    "integrate_imu_angular_velocity",
    "solve_wahba_svd",
    "evaluate_extrinsic_residual",
    "so3_exp",
    "so3_log",
    "so3_hat",
    "so3_vee",
    "so3_geodesic_distance",
    "analyze_16_sample_boundary_pattern",
    "RawClockDomain",
    "PlaybackToCaptureMapping",
    "evaluate_accelerometer_units",
    "evaluate_gyroscope_units",
    "evaluate_timestamp_units",
    "build_full_unit_hypothesis",
    "generate_synthetic_multi_axis_sync_fixture",
    "generate_timelapse_sync_fixture",
    "generate_periodic_ambiguous_fixture",
]
