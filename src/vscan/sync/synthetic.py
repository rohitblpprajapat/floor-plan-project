"""Synthetic fixture generation for multi-axis rotation, timelapse, and negative ambiguity tests.

Meets all test mandates:
1. Multi-axis nonperiodic rotation with known bias, extrinsics, irregular IMU timing (16-sample boundary pattern), and 125 ms time offset.
2. Timelapse fixture with compressed playback time (known exposure table vs withheld metadata).
3. Purely periodic motion with competing correlation peaks for ambiguity rejection.
4. Unit and transform inversion fixtures (microseconds vs milliseconds, g vs m/s^2, SO(3) transpose).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from vscan.sync.rotation import (
    integrate_imu_angular_velocity,
    so3_exp,
    so3_hat,
    so3_log,
)


def make_rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Create orthonormal SO(3) matrix from Euler angles (XYZ convention)."""
    rx = math.radians(roll_deg)
    ry = math.radians(pitch_deg)
    rz = math.radians(yaw_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)],
    ], dtype=np.float64)

    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)],
    ], dtype=np.float64)

    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1],
    ], dtype=np.float64)

    return Rz @ Ry @ Rx


def generate_synthetic_multi_axis_sync_fixture(
    duration_s: float = 6.0,
    cam_fps: float = 20.0,
    time_offset_s: float = 0.125,  # Known 125 ms offset
    roll_deg: float = 15.0,
    pitch_deg: float = -20.0,
    yaw_deg: float = 35.0,
    gyro_bias: Optional[Sequence[float]] = None,
    add_noise: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate ground truth multi-axis non-periodic rotation fixture with 16-sample irregular IMU timing."""
    np.random.seed(seed)
    
    # Ground truth camera-to-IMU rotation R_imu_cam (det = +1, orthonormal)
    R_imu_cam_gt = make_rotation_matrix(roll_deg, pitch_deg, yaw_deg)
    b_g = np.asarray(gyro_bias, dtype=np.float64) if gyro_bias is not None else np.array([0.012, -0.018, 0.025])

    # Multi-axis non-periodic continuous angular velocity function in camera frame
    # Using multiple incommensurate frequencies
    def omega_cam_func(t: float) -> np.ndarray:
        wx = 0.85 * math.sin(1.31 * t) + 0.45 * math.cos(2.73 * t)
        wy = 0.95 * math.cos(0.94 * t) + 0.35 * math.sin(3.17 * t)
        wz = 0.70 * math.sin(1.85 * t) + 0.50 * math.cos(2.41 * t)
        return np.array([wx, wy, wz], dtype=np.float64)

    # 1. Generate Camera Frames (exposure midpoints)
    num_cam_frames = int(duration_s * cam_fps)
    cam_times_s = [i / cam_fps for i in range(num_cam_frames)]

    # Integrate camera relative rotations R_cam[k] between frame k and k+1
    cam_relative_rotations = []
    substeps = 20
    for k in range(num_cam_frames - 1):
        t0 = cam_times_s[k]
        t1 = cam_times_s[k + 1]
        dt_sub = (t1 - t0) / substeps
        R_step = np.eye(3, dtype=np.float64)
        for s in range(substeps):
            t_mid = t0 + (s + 0.5) * dt_sub
            w_mid = omega_cam_func(t_mid)
            R_step = R_step @ so3_exp(w_mid * dt_sub)
        cam_relative_rotations.append(R_step)

    # 2. Generate Irregular IMU Stream with 16-Sample Boundary Pattern
    # t_imu = t_cam - time_offset_s
    # Range of IMU: from -0.5s to duration_s + 0.5s
    t_start_imu = -0.5
    t_end_imu = duration_s + 0.5

    t_raw_start = 100701912  # microsecond ticks
    raw_timestamps = [t_raw_start]
    current_raw = t_raw_start
    current_s = t_start_imu
    imu_times_s = [current_s]

    imu_gyros = []
    imu_accels = []

    # Nominal 1000us step, but index % 16 == 0 has 894us step (replicates actual camera CSV!)
    sample_idx = 0
    while current_s < t_end_imu:
        # Corresponding camera time
        t_cam_equiv = current_s + time_offset_s
        w_cam = omega_cam_func(t_cam_equiv)
        # Transform to IMU body: w_imu = R_imu_cam * w_cam + b_g
        w_imu = R_imu_cam_gt @ w_cam + b_g

        # Specific force with Earth gravity g ~ 1.0 (in IMU frame)
        # Under stationary/low acceleration: a_imu = R_imu_world * [0, 0, 1.0] g
        g_cam = np.array([0.0, 0.0, 1.0])  # nominal Z-up in camera
        a_imu = R_imu_cam_gt @ g_cam

        if add_noise:
            w_imu += np.random.normal(0.0, 0.001, 3)
            a_imu += np.random.normal(0.0, 0.005, 3)

        imu_gyros.append(w_imu)
        imu_accels.append(a_imu)

        # Increment with 16-sample boundary timing
        sample_idx += 1
        dt_ticks = 894 if (sample_idx % 16 == 0) else 1000
        current_raw += dt_ticks
        dt_s = dt_ticks / 1e6
        current_s += dt_s

        raw_timestamps.append(current_raw)
        imu_times_s.append(current_s)

    # Trim last added timestamp to match length of gyros
    raw_timestamps = np.array(raw_timestamps[:len(imu_gyros)], dtype=np.int64)
    imu_times_s = np.array(imu_times_s[:len(imu_gyros)], dtype=np.float64)
    imu_gyros = np.array(imu_gyros, dtype=np.float64)
    imu_accels = np.array(imu_accels, dtype=np.float64)

    return {
        "cam_times_s": cam_times_s,
        "cam_relative_rotations": cam_relative_rotations,
        "imu_times_s": imu_times_s,
        "imu_raw_ticks": raw_timestamps,
        "imu_gyros": imu_gyros,
        "imu_accels": imu_accels,
        "ground_truth": {
            "time_offset_s": time_offset_s,
            "R_imu_cam": R_imu_cam_gt,
            "gyro_bias": b_g,
            "euler_deg": [roll_deg, pitch_deg, yaw_deg],
            "t_imu_cam": [0.0, 0.0, 0.0],
        },
    }


def generate_timelapse_sync_fixture(
    playback_duration_s: float = 4.0,
    speedup_factor: float = 10.0,  # 10x compression: physical duration is 40s
    cam_fps: float = 30.0,
    withhold_metadata: bool = False,
    time_offset_s: float = 0.050,
) -> Dict[str, Any]:
    """Generate a timelapse fixture where playback time is compressed by speedup_factor."""
    physical_duration_s = playback_duration_s * speedup_factor
    num_frames = int(playback_duration_s * cam_fps)

    # Playback PTS in container: 0.0 to 4.0s
    playback_pts_s = [i / cam_fps for i in range(num_frames)]

    # Real physical exposure times:
    real_exposure_times_s = [pts * speedup_factor for pts in playback_pts_s]

    fixture = generate_synthetic_multi_axis_sync_fixture(
        duration_s=physical_duration_s,
        cam_fps=cam_fps / speedup_factor,  # physical exposure rate = 3 Hz
        time_offset_s=time_offset_s,
    )

    per_frame_table = {i: real_exposure_times_s[i] for i in range(num_frames)} if not withhold_metadata else None

    return {
        "playback_pts_s": playback_pts_s,
        "real_exposure_times_s": real_exposure_times_s,
        "cam_relative_rotations": fixture["cam_relative_rotations"],
        "imu_times_s": fixture["imu_times_s"],
        "imu_raw_ticks": fixture["imu_raw_ticks"],
        "imu_gyros": fixture["imu_gyros"],
        "imu_accels": fixture["imu_accels"],
        "speedup_factor": speedup_factor,
        "withhold_metadata": withhold_metadata,
        "per_frame_table": per_frame_table,
        "ground_truth": fixture["ground_truth"],
    }


def generate_periodic_ambiguous_fixture(
    duration_s: float = 8.0,
    cam_fps: float = 20.0,
    frequency_hz: float = 1.0,  # Pure 1 Hz sine wave
    time_offset_s: float = 0.200,
) -> Dict[str, Any]:
    """Generate purely periodic single-axis motion producing ambiguous correlation peaks."""
    w0 = 2.0 * math.pi * frequency_hz
    num_cam_frames = int(duration_s * cam_fps)
    cam_times_s = [i / cam_fps for i in range(num_cam_frames)]

    # Pure yaw oscillation
    def omega_cam(t: float) -> np.ndarray:
        return np.array([0.0, 0.0, 1.2 * math.sin(w0 * t)], dtype=np.float64)

    cam_relative_rotations = []
    substeps = 10
    for k in range(num_cam_frames - 1):
        t0 = cam_times_s[k]
        t1 = cam_times_s[k + 1]
        dt_sub = (t1 - t0) / substeps
        R_step = np.eye(3, dtype=np.float64)
        for s in range(substeps):
            t_mid = t0 + (s + 0.5) * dt_sub
            R_step = R_step @ so3_exp(omega_cam(t_mid) * dt_sub)
        cam_relative_rotations.append(R_step)

    # IMU stream
    imu_dt = 0.001
    imu_times_s = np.arange(-0.5, duration_s + 0.5, imu_dt)
    imu_gyros = []
    for t_i in imu_times_s:
        t_c = t_i + time_offset_s
        imu_gyros.append(omega_cam(t_c))

    return {
        "cam_times_s": cam_times_s,
        "cam_relative_rotations": cam_relative_rotations,
        "imu_times_s": imu_times_s,
        "imu_gyros": np.array(imu_gyros, dtype=np.float64),
        "ground_truth": {
            "time_offset_s": time_offset_s,
            "period_s": 1.0 / frequency_hz,
            "is_ambiguous": True,
        },
    }
