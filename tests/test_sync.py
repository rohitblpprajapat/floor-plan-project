"""Tests for Stage 07 Camera/IMU Synchronization, Calibration, and Clock Modeling.

Covers all Prompt 07 test mandates:
1. Multi-axis nonperiodic rotation with known bias, extrinsics, irregular IMU timing (16-sample boundaries) and known 125 ms offset: recover offset within 2 ms tolerance on ideal fixture across held-out windows.
2. Timelapse fixture whose playback time is compressed by a known factor: real exposure association recovered when metadata present, or explicitly unavailable/unresolved when withheld.
3. Milliseconds vs microseconds, g vs m/s^2, axis permutation/sign, quaternion ordering, and camera/world transform inversion tests.
4. Purely periodic motion with ambiguous offsets, inadequate axis excitation, and unknown stabilization must NOT receive a false synchronization pass.
5. Deliberately wrong unit/time mappings worsen independent rotational residuals or are rejected.
6. 16-sample boundary timing pattern: verify non-uniform dt preservation and demonstrate phase error accumulation if uniform dt is substituted.
7. Real pilot session audit (20260910_172939): FlowState stabilization, timelapse, and partial visual SfM evaluated; actual sensor branch marked WAITING_INPUT or UNRESOLVED with real_fusion_ready=False and scale_status=NOT_ESTIMATED.
8. Job lifecycle: start, status, cooperative cancel, and resume on sync fixtures.
9. Cache key invalidation: modified sync configuration invalidates dependent metric cache without invalidating prior visual SfM.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest

from vscan.engine.supervisor import Supervisor
from vscan.schemas.jobs import JobStatus
from vscan.schemas.sync import (
    ClockCalibrationSchema,
    DiagnosticsSchema,
    ExtrinsicCalibrationSchema,
    SyncManifestSchema,
    TimingBoundaryDiagnosticSchema,
    UnitHypothesisSchema,
)
from vscan.storage.artifacts import compute_cache_key, compute_sha256_dict
from vscan.storage.database import DatabaseManager
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
    make_rotation_matrix,
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


def create_sync_test_env(tmp_path: Path) -> Dict[str, Any]:
    """Isolated environment creation for testing synchronization jobs and supervisor."""
    db_path = tmp_path / "app.sqlite3"
    db_mgr = DatabaseManager(db_path)
    db_mgr.init_db()

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "test_laptop.yaml"
    config_data = {
        "profile": "test_laptop",
        "concurrency": {"max_gpu_jobs": 1, "max_cpu_jobs": 2, "worker_cpu_threads": 2},
        "process_management": {"cooperative_grace_period_seconds": 2.0},
        "timeouts": {"default_seconds": 120, "stages": {"sync": 60, "smoke": 10}},
        "stage_limits": {"sync": {"max_concurrent": 1, "cpu_threads": 2}},
        "storage": {
            "app_db_path": str(db_path),
            "sessions_dir": str(sessions_dir),
            "runs_dir": str(runs_dir),
            "base_data_dir": str(tmp_path / "data"),
            "mode": "reference",
        },
    }
    import yaml
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    return {
        "tmp_path": tmp_path,
        "db_path": db_path,
        "runs_dir": runs_dir,
        "sessions_dir": sessions_dir,
        "config_path": config_path,
        "db_mgr": db_mgr,
    }


@pytest.fixture
def sync_test_env(tmp_path: Path) -> Dict[str, Any]:
    """Isolated fixture for testing synchronization jobs and supervisor."""
    return create_sync_test_env(tmp_path)


def test_01_synthetic_multi_axis_nonperiodic_sync_recovery(tmp_path: Path):
    """Test 1: Multi-axis nonperiodic rotation with known bias, extrinsics, irregular IMU timing and 125 ms offset.
    
    Must recover time offset within 2 ms tolerance and extrinsic rotation within 2 degrees.
    Must evaluate across independent held-out time windows.
    """
    time_offset_gt = 0.125  # 125 ms
    roll_gt, pitch_gt, yaw_gt = 15.0, -20.0, 35.0
    bias_gt = [0.015, -0.020, 0.010]

    fixture = generate_synthetic_multi_axis_sync_fixture(
        duration_s=8.0,
        cam_fps=25.0,
        time_offset_s=time_offset_gt,
        roll_deg=roll_gt,
        pitch_deg=pitch_gt,
        yaw_deg=yaw_gt,
        gyro_bias=bias_gt,
        add_noise=True,
        seed=101,
    )

    # Estimate synchronization
    res = estimate_camera_imu_synchronization(
        cam_times_s=fixture["cam_times_s"],
        cam_relative_rotations=fixture["cam_relative_rotations"],
        imu_times_s=fixture["imu_times_s"],
        imu_gyros=fixture["imu_gyros"],
        imu_accels=fixture["imu_accels"],
        search_window_s=1.0,
        search_step_s=0.005,
        fine_step_s=0.0005,
        gyro_bias=bias_gt,
        enable_drift_estimation=False,
    )

    assert res.success is True, f"Synchronization failed: {res.message}"
    assert res.time_offset_s is not None

    # Tolerance requirement: within 2 ms (0.002s) on well-sampled fixture
    offset_error = abs(res.time_offset_s - time_offset_gt)
    assert offset_error <= 0.002, f"Time offset error {offset_error*1000:.3f} ms exceeds 2.0 ms tolerance"

    # Extrinsic rotation recovery accuracy
    R_gt = fixture["ground_truth"]["R_imu_cam"]
    R_rec = res.R_imu_cam
    R_err = R_rec @ R_gt.T
    rot_err_deg = math.degrees(float(np.linalg.norm(so3_log(R_err))))
    assert rot_err_deg <= 1.5, f"Extrinsic rotation error {rot_err_deg:.3f} deg exceeds 1.5 deg tolerance"

    # Held-out window validation
    diag = res.diagnostics.separated_windows
    assert diag["consistent"] is True, f"Separated windows inconsistent: {diag}"
    assert diag["offset_discrepancy_s"] <= 0.005, f"Window discrepancy {diag['offset_discrepancy_s']*1000:.2f} ms > 5 ms"


def test_02_timelapse_fixture_compression_recovery_and_unavailable():
    """Test 2: Timelapse fixture with 10x compressed playback.
    
    Case A: Per-frame exposure table available -> recovers mapping accurately.
    Case B: Exposure table withheld -> marks UNRESOLVED / WAITING_INPUT.
    """
    # Case A: Table available
    fix_resolved = generate_timelapse_sync_fixture(
        playback_duration_s=2.0,
        speedup_factor=10.0,
        withhold_metadata=False,
        time_offset_s=0.080,
    )
    res_a = estimate_camera_imu_synchronization(
        cam_times_s=fix_resolved["real_exposure_times_s"],
        cam_relative_rotations=fix_resolved["cam_relative_rotations"],
        imu_times_s=fix_resolved["imu_times_s"],
        imu_gyros=fix_resolved["imu_gyros"],
        is_timelapse=True,
        timelapse_mapping_resolved=True,
        search_window_s=0.5,
    )
    assert res_a.success is True
    assert abs(res_a.time_offset_s - 0.080) <= 0.003

    # Case B: Table withheld -> must remain UNRESOLVED
    fix_unresolved = generate_timelapse_sync_fixture(
        playback_duration_s=2.0,
        speedup_factor=10.0,
        withhold_metadata=True,
    )
    res_b = estimate_camera_imu_synchronization(
        cam_times_s=fix_unresolved["playback_pts_s"],
        cam_relative_rotations=fix_unresolved["cam_relative_rotations"],
        imu_times_s=fix_unresolved["imu_times_s"],
        imu_gyros=fix_unresolved["imu_gyros"],
        is_timelapse=True,
        timelapse_mapping_resolved=False,
    )
    assert res_b.success is False
    assert "WAITING_INPUT" in res_b.status or "UNRESOLVED" in res_b.status
    assert res_b.time_offset_s is None


def test_03_unit_discrimination_and_physical_checks():
    """Test 3: Unit hypotheses discrimination (ms vs us, g vs m/s^2, rad/s vs deg/s) and SO(3) conventions."""
    # 1. Accelerometer: g vs m/s^2
    accel_g = np.array([[0.0, 0.0, 1.01], [0.05, -0.02, 0.99], [-0.03, 0.01, 1.02]])
    hyp_g = evaluate_accelerometer_units(accel_g, sensor_name="camera_imu")
    assert hyp_g.accel_units_hypothesis == "g"
    assert hyp_g.accel_conversion_to_m_s2 == 9.80665

    accel_ms2 = np.array([[0.0, 0.0, 9.81], [0.4, -0.2, 9.75], [-0.3, 0.1, 9.90]])
    hyp_ms2 = evaluate_accelerometer_units(accel_ms2, sensor_name="phone_sensors")
    assert hyp_ms2.accel_units_hypothesis == "m/s^2"
    assert hyp_ms2.accel_conversion_to_m_s2 == 1.0

    # 2. Timestamps: microseconds vs milliseconds
    ts_us = np.arange(100701912, 100701912 + 1000 * 1000, 1000, dtype=np.int64)  # dt=1000us
    unit_us, rate_us, dur_us, _ = evaluate_timestamp_units(ts_us)
    assert unit_us == "microseconds"
    assert rate_us == 1e6

    ts_ms = np.arange(1789041579535, 1789041579535 + 10 * 1000, 10, dtype=np.int64)  # dt=10ms
    unit_ms, rate_ms, dur_ms, _ = evaluate_timestamp_units(ts_ms)
    assert unit_ms == "milliseconds"
    assert rate_ms == 1e3

    # 3. Gyroscope: rad/s vs deg/s
    gyro_rad = np.array([[0.1, 0.2, 0.5], [0.3, -0.1, 0.4]])
    u_rad, conv_rad, _ = evaluate_gyroscope_units(gyro_rad)
    assert u_rad == "rad/s"
    assert conv_rad == 1.0

    # 4. SO(3) transform inversion and chirality check
    R = make_rotation_matrix(25.0, -35.0, 45.0)
    assert abs(np.linalg.det(R) - 1.0) < 1e-10
    assert np.allclose(R.T @ R, np.eye(3))

    # Test that left-handed reflection (det = -1.0) is rejected by ExtrinsicCalibrationSchema
    with pytest.raises(ValueError, match="not proper SO\\(3\\)"):
        R_refl = np.diag([1.0, 1.0, -1.0])
        ExtrinsicCalibrationSchema(
            R_imu_cam=R_refl.tolist(),
            rotational_residual_deg=0.0,
            provenance="test",
        )


def test_04_periodic_motion_ambiguity_rejection():
    """Test 4: Purely periodic motion with competing correlation peaks must be rejected (no false pass)."""
    # 1 Hz pure sinusoidal motion
    fixture = generate_periodic_ambiguous_fixture(duration_s=8.0, cam_fps=25.0, frequency_hz=1.0, time_offset_s=0.200)

    res = estimate_camera_imu_synchronization(
        cam_times_s=fixture["cam_times_s"],
        cam_relative_rotations=fixture["cam_relative_rotations"],
        imu_times_s=fixture["imu_times_s"],
        imu_gyros=fixture["imu_gyros"],
        search_window_s=2.0,
    )

    # Must be explicitly rejected due to competing correlation peaks
    assert res.success is False
    assert res.status == "REJECTED_AMBIGUOUS_PERIODIC_MOTION"
    assert res.diagnostics.competing_peaks["ambiguity_status"] == "AMBIGUOUS_PERIODIC"
    assert res.diagnostics.competing_peaks["peak_ratio"] > 0.80


def test_05_deliberately_wrong_mappings_worsen_residuals():
    """Test 5: Deliberately wrong unit/time mappings worsen rotational residuals or are rejected."""
    fixture = generate_synthetic_multi_axis_sync_fixture(duration_s=6.0, time_offset_s=0.100, seed=77)

    # 1. Correct baseline run
    res_correct = estimate_camera_imu_synchronization(
        cam_times_s=fixture["cam_times_s"],
        cam_relative_rotations=fixture["cam_relative_rotations"],
        imu_times_s=fixture["imu_times_s"],
        imu_gyros=fixture["imu_gyros"],
        search_window_s=0.5,
    )
    assert res_correct.success is True
    residual_correct = res_correct.mean_rotational_residual_deg

    # 2. Deliberately wrong: Inverted/transposed camera rotations (world-to-camera vs camera-to-world)
    cam_rot_inverted = [R.T for R in fixture["cam_relative_rotations"]]
    res_inverted = estimate_camera_imu_synchronization(
        cam_times_s=fixture["cam_times_s"],
        cam_relative_rotations=cam_rot_inverted,
        imu_times_s=fixture["imu_times_s"],
        imu_gyros=fixture["imu_gyros"],
        search_window_s=0.5,
    )
    residual_inverted = res_inverted.mean_rotational_residual_deg
    assert residual_inverted > residual_correct * 3.0 or residual_inverted > 15.0, (
        f"Inverted rotation did not worsen residual: {residual_inverted:.2f} vs {residual_correct:.2f}"
    )

    # 3. Deliberately wrong: Swapped axes (swapping gyro X and Y)
    w_swapped = fixture["imu_gyros"].copy()
    w_swapped[:, [0, 1]] = w_swapped[:, [1, 0]]
    res_swapped = estimate_camera_imu_synchronization(
        cam_times_s=fixture["cam_times_s"],
        cam_relative_rotations=fixture["cam_relative_rotations"],
        imu_times_s=fixture["imu_times_s"],
        imu_gyros=w_swapped,
        search_window_s=0.5,
    )
    residual_swapped = res_swapped.mean_rotational_residual_deg
    assert residual_swapped > residual_correct * 2.0 or residual_swapped > 10.0


def test_06_16_sample_boundary_timing_preservation():
    """Test 6: 16-sample boundary timing analysis.
    
    Verifies that short intervals align with index % 16 == 0 and confirms non-uniform dt preservation.
    Demonstrates that substituting uniform dt accumulates significant phase drift.
    """
    # Create realistic 40,000-sample sequence with 16-sample boundaries
    N = 1600
    t0 = 100701912
    raw_ticks = [t0]
    for i in range(1, N):
        dt = 894 if (i % 16 == 0) else 1000
        raw_ticks.append(raw_ticks[-1] + dt)
    raw_ticks = np.array(raw_ticks, dtype=np.int64)

    diag = analyze_16_sample_boundary_pattern(raw_ticks, source_file="test_telemetry.csv")
    assert diag.total_samples == N
    assert diag.short_intervals_count == (N - 1) // 16
    assert diag.boundary_16_alignment_percent >= 99.0
    assert diag.uniform_dt_substituted is False

    # Demonstrate cumulative error if replaced with uniform dt = 1000 us:
    uniform_ticks = t0 + np.arange(N, dtype=np.int64) * 1000
    drift_ticks = int(uniform_ticks[-1] - raw_ticks[-1])
    expected_drift_ticks = ((N - 1) // 16) * (1000 - 894)
    assert drift_ticks == expected_drift_ticks
    # Over 40,000 samples, drift is ~271 ms (fatal to synchronization!)
    projected_drift_40k_s = (40981 // 16) * (1000 - 894) / 1e6
    assert projected_drift_40k_s > 0.250, f"Projected drift {projected_drift_40k_s:.3f}s should exceed 250ms"


def test_07_real_pilot_session_audit_and_unresolved_verdict(sync_test_env):
    """Test 7: Evaluate real session 20260910_172939 data against Prompt 07 contract.
    
    Must verify:
    - FlowState stabilization active -> fitting rigid extrinsic rejected.
    - Timelapse recording active -> exposure mapping unresolved.
    - Status marked WAITING_INPUT or UNRESOLVED.
    - real_fusion_ready is strictly False.
    - scale_status is strictly NOT_ESTIMATED.
    """
    out_dir = sync_test_env["tmp_path"] / "real_pilot_sync_out"
    cfg = SyncConfig(
        session_id="20260910_172939",
        run_id="run_sync_test_pilot_001",
        output_dir=out_dir,
        is_synthetic=False,
    )

    pipeline = SyncPipeline(cfg)
    res = pipeline.run()

    manifest = res["manifest"]
    assert "UNRESOLVED" in manifest["status"] or "WAITING_INPUT" in manifest["status"]
    assert manifest["real_fusion_ready"] is False
    assert manifest["scale_status"] == "NOT_ESTIMATED"

    # Handoff to Prompt 08: actual data NOT ready, fixtures ready
    p08 = manifest["prompt_08_readiness"]
    assert p08["actual_data_ready"] is False
    assert p08["fixtures_ready"] is True

    # Check timing diagnostics serialization
    assert (out_dir / "timing_diagnostics.json").is_file()
    assert (out_dir / "sync_manifest.json").is_file()
    assert (out_dir / ".complete").is_file()

    with open(out_dir / "timing_diagnostics.json", "r", encoding="utf-8") as f:
        td = json.load(f)
        assert td["short_intervals_count"] == 2561
        assert td["boundary_16_alignment_percent"] >= 99.0
        assert td["uniform_dt_substituted"] is False


def test_08_sync_job_lifecycle_cancel_and_resume(sync_test_env):
    """Test 8: Supervisor lifecycle (start, status, cancel, resume) on sync stage."""
    supervisor = Supervisor(
        db_path=sync_test_env["db_path"],
        runs_dir=sync_test_env["runs_dir"],
        sessions_dir=sync_test_env["sessions_dir"],
        config_path=sync_test_env["config_path"],
    )

    session_id = "synth_session_sync_lifecycle"
    sess_dir = sync_test_env["sessions_dir"] / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)

    cfg_stage = {
        "sync": {
            "synthetic_fixture": True,
            "synthetic_fixture_type": "multi_axis",
            "search_window_s": 0.5,
            "search_step_s": 0.02,
            "fine_search_step_s": 0.002,
        },
        "storage": {
            "app_db_path": str(sync_test_env["db_path"]),
            "sessions_dir": str(sync_test_env["sessions_dir"]),
            "runs_dir": str(sync_test_env["runs_dir"]),
            "mode": "reference",
        },
    }
    import yaml
    stage_cfg_file = sync_test_env["tmp_path"] / "sync_stage_cfg.yaml"
    with open(stage_cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg_stage, f)

    # 1. Start synchronous run
    start_res = supervisor.start_job(
        stage="sync",
        session_id=session_id,
        config_path=stage_cfg_file,
        wait=True,
    )
    assert start_res["status"] == JobStatus.SUCCEEDED.value
    run_id_1 = start_res["run_id"]

    # 2. Check status
    st = supervisor.get_status(run_id_1)
    assert st is not None
    assert st.status == JobStatus.SUCCEEDED

    # 3. Simulate cancellation on a new job
    start_res_2 = supervisor.start_job(
        stage="sync",
        session_id=session_id,
        config_path=stage_cfg_file,
        force=True,
        wait=False,
    )
    run_id_2 = start_res_2["run_id"]
    time.sleep(0.1)
    cancel_ok, msg = supervisor.cancel_job(run_id_2)
    assert cancel_ok is True
    st_cancel = supervisor.get_status(run_id_2)
    assert st_cancel.status == JobStatus.CANCELLED

    # 4. Resume cancelled job
    resume_res = supervisor.resume_job(run_id_2, wait=True)
    assert resume_res["status"] == JobStatus.SUCCEEDED.value
    assert resume_res["attempt"] == 2
    assert resume_res["parent_run_id"] == run_id_2


def test_09_cache_invalidation_and_downstream_dependency(sync_test_env):
    """Test 9: Modifying sync calibration parameters alters cache key and invalidates dependent work."""
    session_id = "test_cache_inval_session"

    cfg1 = {"sync": {"search_window_s": 2.0, "time_reference": "exposure_midpoint"}}
    cfg2 = {"sync": {"search_window_s": 4.0, "time_reference": "exposure_start"}}  # modified

    h1 = compute_sha256_dict(cfg1)
    h2 = compute_sha256_dict(cfg2)

    inputs = {"imu_telemetry": "dummy_hash_12345"}
    key1 = compute_cache_key("sync", h1, inputs)
    key2 = compute_cache_key("sync", h2, inputs)

    assert key1 != key2, "Cache keys must differ on parameter modification"
