"""Schemas for camera-IMU synchronization, clock calibration, and observability.

Contracts enforced:
1. Strict separation of software status, actual-data status, and scale status.
2. Scale status remains strictly NOT_ESTIMATED or UNRESOLVED.
3. Raw timestamps preserved as 64-bit integers; local relative seconds used for calculation.
4. Preserves 16-sample boundary timing pattern; replacing with uniform dt is strictly forbidden.
5. Digital stabilization transforms and timelapse compression must be modeled or marked UNRESOLVED.
6. Weakly observable translation baseline t_imu_cam must be annotated with uncertainty.
7. Phone sensors remain strictly DIAGNOSTIC_ONLY.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator
from vscan.schemas.common import SCHEMA_VERSION, CoordinateFrame, PoseMatrix, Units


class UnitHypothesisSchema(BaseModel):
    """Explicit unit hypothesis and physical verification for a sensor stream."""

    sensor_type: str = Field(..., description="Sensor descriptor (e.g. camera_imu, phone_sensors)")
    accel_norm_mean: float = Field(..., description="Mean magnitude of accelerometer measurement vector")
    accel_units_hypothesis: str = Field(..., description="Declared unit: 'g' (~1.0) or 'm/s^2' (~9.80665)")
    accel_hypothesis_status: str = Field(..., description="Validation status: ACCEPTED, REJECTED, PROVISIONAL")
    accel_conversion_to_m_s2: float = Field(default=9.80665, description="Scale factor to convert to m/s^2")

    gyro_norm_mean: float = Field(..., description="Mean magnitude of gyroscope measurement vector")
    gyro_units_hypothesis: str = Field(..., description="Declared unit: 'rad/s' or 'deg/s'")
    gyro_hypothesis_status: str = Field(..., description="Validation status: ACCEPTED, REJECTED, PROVISIONAL")
    gyro_conversion_to_rad_s: float = Field(default=1.0, description="Scale factor to convert to rad/s")

    timestamp_units_hypothesis: str = Field(..., description="Declared timestamp unit: microseconds, milliseconds, nanoseconds")
    timestamp_hypothesis_status: str = Field(..., description="Validation status: ACCEPTED, REJECTED, PROVISIONAL")
    timestamp_ticks_per_second: float = Field(..., description="Nominal integer tick rate per second (e.g. 1e6 for us, 1e3 for ms)")

    evidence: str = Field(..., description="Physical consistency evidence and rationale")
    is_inferred: bool = Field(default=True, description="True if inferred from empirical sensor norms")
    is_documented: bool = Field(default=False, description="True if documented in official manufacturer specifications")


class TimingBoundaryDiagnosticSchema(BaseModel):
    """Investigation and diagnostics of sensor packet boundary timing."""

    source_file: str = Field(..., description="Source telemetry file")
    total_samples: int = Field(..., description="Total IMU data samples parsed")
    raw_t_min: int = Field(..., description="Earliest integer raw timestamp")
    raw_t_max: int = Field(..., description="Latest integer raw timestamp")
    raw_t_span: int = Field(..., description="Total raw integer tick span")
    nominal_interval_ticks: float = Field(..., description="Nominal sampling interval in raw ticks")
    short_intervals_count: int = Field(..., description="Number of intervals significantly below nominal (< 950 us)")
    boundary_16_alignment_count: int = Field(..., description="Count of short intervals exactly aligning with sample_index % 16 == 0")
    boundary_16_alignment_percent: float = Field(..., description="Percentage of short intervals aligning with 16-sample boundary")
    dt_min_ticks: float = Field(..., description="Minimum inter-sample delta in ticks")
    dt_max_ticks: float = Field(..., description="Maximum inter-sample delta in ticks")
    dt_mean_ticks: float = Field(..., description="Mean inter-sample delta in ticks")
    dt_std_ticks: float = Field(..., description="Standard deviation of inter-sample delta in ticks")
    uniform_dt_substituted: bool = Field(default=False, description="Strictly False: actual non-uniform dt must be preserved")
    investigation_notes: str = Field(..., description="Physical explanation of FIFO packet transfer and boundary timing")


class ExtrinsicCalibrationSchema(BaseModel):
    """Spatial extrinsic calibration between camera optical center/rig and IMU body frame."""

    R_imu_cam: List[List[float]] = Field(
        ...,
        description="3x3 orthonormal rotation matrix transforming vectors from camera/rig to IMU: v_imu = R_imu_cam * v_cam",
    )
    t_imu_cam: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
        description="Translation vector [tx, ty, tz] from camera to IMU in IMU frame coordinates",
    )
    t_imu_cam_uncertainty_m: List[float] = Field(
        default_factory=lambda: [0.05, 0.05, 0.05],
        min_length=3,
        max_length=3,
        description="1-sigma uncertainty standard deviations in meters",
    )
    t_imu_cam_observability: str = Field(
        default="WEAKLY_OBSERVABLE",
        description="Observability status: UNOBSERVABLE, WEAKLY_OBSERVABLE, or OBSERVABLE",
    )
    rotational_residual_deg: float = Field(..., description="Root-mean-square angular residual error in degrees")
    provenance: str = Field(..., description="Calibration source (e.g. 'synthetic_ground_truth', 'so3_wahba_optimization', 'cad_prior')")
    notes: str = Field(default="", description="Observability constraints and metric scale impact notes")

    @field_validator("R_imu_cam")
    @classmethod
    def validate_rotation_so3(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("R_imu_cam must be a 3x3 matrix")
        r = np.array(v, dtype=np.float64)
        det = float(np.linalg.det(r))
        if abs(det - 1.0) > 1e-3:
            raise ValueError(f"Extrinsic rotation matrix is not proper SO(3): det(R) = {det:.6f} != 1.0")
        diff = r.T @ r - np.eye(3)
        if np.max(np.abs(diff)) > 1e-3:
            raise ValueError("Extrinsic rotation matrix is not orthogonal: R^T * R != I")
        return v


class ClockCalibrationSchema(BaseModel):
    """Temporal synchronization parameters mapping visual frame events to IMU clock."""

    time_reference: str = Field(
        default="exposure_midpoint",
        description="Camera timestamp physical meaning: 'exposure_start', 'exposure_midpoint', or 'exposure_end'",
    )
    time_offset_s: Optional[float] = Field(
        default=None,
        description="Estimated time offset tau = t_cam - t_imu in seconds such that t_imu = t_cam - tau",
    )
    time_offset_uncertainty_s: Optional[float] = Field(
        default=None,
        description="Estimated 1-sigma uncertainty of time offset in seconds",
    )
    clock_drift_ppm: float = Field(
        default=0.0,
        description="Estimated clock drift rate in parts per million (ppm)",
    )
    drift_fitted: bool = Field(
        default=False,
        description="True if clock drift was explicitly estimated from data",
    )
    playback_to_capture_mapping: str = Field(
        ...,
        description="Mapping status: 'REALTIME_1TO1', 'TIMELAPSE_PIECEWISE_SOLVED', or 'TIMELAPSE_UNRESOLVED'",
    )
    playback_speed_factor: Optional[float] = Field(
        default=None,
        description="Playback speedup factor relative to real-time (e.g. 10.0 for 10x timelapse)",
    )


class DiagnosticsSchema(BaseModel):
    """Multi-window validation, periodic ambiguity checks, rolling shutter and stabilization diagnostics."""

    competing_peaks: Dict[str, Any] = Field(
        default_factory=dict,
        description="Cross-correlation peak analysis (peak_1, peak_2, ratio, ambiguity_verdict)",
    )
    separated_windows: Dict[str, Any] = Field(
        default_factory=dict,
        description="Independent window validation results (window1_offset, window2_offset, consistency_check)",
    )
    rolling_shutter: Dict[str, Any] = Field(
        default_factory=dict,
        description="Rolling shutter readout duration and residual systematic distortion assessment",
    )
    stabilization: Dict[str, Any] = Field(
        default_factory=dict,
        description="Digital stabilization evaluation (FlowState status, rotation known/unknown, fitting policy)",
    )
    phone_sensors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Phone telemetry role: strictly diagnostic, non-rigid session evidence",
    )


class SyncManifestSchema(BaseModel):
    """Authoritative output manifest for Stage 07 IMU/Camera Synchronization and Clock Modeling."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    session_id: str = Field(..., description="Associated session identifier")
    run_id: str = Field(..., description="Unique job execution run ID")
    stage: str = Field(default="sync", description="Pipeline stage identifier")
    status: str = Field(..., description="Stage status: STAGE_07_COMPLETE, WAITING_INPUT, or UNRESOLVED")
    real_fusion_ready: bool = Field(..., description="True ONLY if real data prerequisites are fully satisfied")
    scale_status: str = Field(default="NOT_ESTIMATED", description="Authoritative metric scale status: strictly NOT_ESTIMATED")

    prompt_08_readiness: Dict[str, Any] = Field(
        ...,
        description="Operational handoff to Prompt 08 (actual_data_ready, fixtures_ready, rationale)",
    )
    accepted_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Verified accepted synchronization and calibration parameters",
    )
    unresolved_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters remaining unresolved due to missing prerequisites or unknown transforms",
    )
    unit_hypotheses: Dict[str, UnitHypothesisSchema] = Field(
        default_factory=dict,
        description="Validated unit hypotheses for camera and phone streams",
    )
    timing_diagnostics: Optional[TimingBoundaryDiagnosticSchema] = Field(
        default=None,
        description="16-sample boundary timing analysis",
    )
    clock_calibration: Optional[ClockCalibrationSchema] = Field(
        default=None,
        description="Clock mapping and time offset results",
    )
    extrinsic_calibration: Optional[ExtrinsicCalibrationSchema] = Field(
        default=None,
        description="Camera-to-IMU spatial rotation and translation calibration",
    )
    diagnostics: DiagnosticsSchema = Field(
        default_factory=DiagnosticsSchema,
        description="Validation diagnostics (competing peaks, rolling shutter, stabilization, windows)",
    )
    noise_priors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Explicitly labelled engineering noise priors for gyroscope and accelerometer",
    )
    parent_hashes: Dict[str, str] = Field(
        default_factory=dict,
        description="SHA-256 hashes of all input files and prior stage manifests",
    )
    created_at_utc: str = Field(..., description="UTC ISO-8601 generation timestamp")
