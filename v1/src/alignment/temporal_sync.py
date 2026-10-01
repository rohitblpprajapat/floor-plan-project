"""
Temporal Calibration and Camera-IMU Extrinsic Alignment.

Provides:
- Cross-correlation between visual angular velocity magnitude and IMU gyro magnitude
- Determination of optimal clock offset Delta_t: t_imu = t_camera + Delta_t
- Calibration of the spatial rotation extrinsic R_CI in SO(3) aligning IMU to Camera:
  omega_camera = R_CI * omega_imu
- Automated coordinate frame convention verification (24 orthogonal SO(3) permutations)
"""

import numpy as np
from scipy.signal import correlate
from scipy.interpolate import interp1d
from typing import Dict, List, Optional, Tuple
from src.inertial.imu_types import so3_log, so3_exp

# 24 proper orthogonal rotation matrices corresponding to 90-degree axis permutations (Cube rotation group O)
def generate_cube_rotations() -> List[np.ndarray]:
    rotations = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                # Candidate column vectors for permutation
                pass
    # Standard permutations
    axes = np.eye(3)
    from itertools import permutations, product
    for perm in permutations(range(3)):
        for signs in product([-1.0, 1.0], repeat=3):
            R = np.zeros((3, 3))
            for i, p in enumerate(perm):
                R[i, p] = signs[i]
            if np.isclose(np.linalg.det(R), 1.0):
                rotations.append(R)
    return rotations


def cross_correlate_signals(
    t_sig1: np.ndarray,
    sig1: np.ndarray,
    t_sig2: np.ndarray,
    sig2: np.ndarray,
    dt_resample: float = 0.02, # 50 Hz common sampling
    max_lag_sec: float = 20.0
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Find time offset Delta_t such that sig2(t + Delta_t) best aligns with sig1(t).
    Returns: (best_offset_sec, peak_correlation, lags_sec, correlation_values)
    """
    # Overlapping time span
    t_start = max(t_sig1[0], t_sig2[0] - max_lag_sec)
    t_end = min(t_sig1[-1], t_sig2[-1] + max_lag_sec)
    if t_end <= t_start:
        return 0.0, 0.0, np.array([]), np.array([])

    t_grid = np.arange(t_start, t_end, dt_resample)
    f1 = interp1d(t_sig1, sig1, kind='linear', fill_value=0.0, bounds_error=False)
    f2 = interp1d(t_sig2, sig2, kind='linear', fill_value=0.0, bounds_error=False)

    s1 = f1(t_grid)
    s2 = f2(t_grid)

    # Demean and normalize
    s1_norm = (s1 - np.mean(s1)) / (np.std(s1) * len(s1) + 1e-12)
    s2_norm = (s2 - np.mean(s2)) / (np.std(s2) + 1e-12)

    max_lags_samples = int(max_lag_sec / dt_resample)
    corr = correlate(s2_norm, s1_norm, mode='full')
    lags = np.arange(-len(s1_norm) + 1, len(s1_norm)) * dt_resample

    # Restrict to [-max_lag_sec, max_lag_sec]
    valid_mask = np.abs(lags) <= max_lag_sec
    valid_lags = lags[valid_mask]
    valid_corr = corr[valid_mask]

    peak_idx = np.argmax(valid_corr)
    best_lag = valid_lags[peak_idx]
    peak_val = valid_corr[peak_idx]

    # Parabolic sub-sample refinement
    if 0 < peak_idx < len(valid_corr) - 1:
        y0 = valid_corr[peak_idx - 1]
        y1 = valid_corr[peak_idx]
        y2 = valid_corr[peak_idx + 1]
        denom = 2.0 * (2.0 * y1 - y0 - y2)
        if denom > 1e-9:
            delta_p = (y0 - y2) / denom
            best_lag += delta_p * dt_resample

    return float(best_lag), float(peak_val), valid_lags, valid_corr


def solve_extrinsic_R_ci(
    omega_cam: np.ndarray,
    omega_imu: np.ndarray,
    find_nearest_discrete: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Solve for R_CI in SO(3) minimizing ||omega_cam - R_CI * omega_imu||^2 via Orthogonal Procrustes.
    Optionally snaps to the closest 90-degree cube symmetry (common in rigid camera bodies).
    Returns: (R_CI, rms_residual_rad_s)
    """
    if len(omega_cam) != len(omega_imu) or len(omega_cam) < 5:
        return np.eye(3), float('inf')

    # H = sum(omega_cam * omega_imu.T)
    H = omega_cam.T @ omega_imu
    U, S, Vt = np.linalg.svd(H)
    R_opt = U @ Vt
    if np.linalg.det(R_opt) < 0:
        U[:, 2] *= -1
        R_opt = U @ Vt

    res = omega_cam - (R_opt @ omega_imu.T).T
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))

    if find_nearest_discrete:
        best_R_cube = R_opt
        best_rms_cube = float('inf')
        for R_cube in generate_cube_rotations():
            res_cube = omega_cam - (R_cube @ omega_imu.T).T
            rms_cube = float(np.sqrt(np.mean(np.sum(res_cube ** 2, axis=1))))
            if rms_cube < best_rms_cube:
                best_rms_cube = rms_cube
                best_R_cube = R_cube
        # If discrete orientation explains the data closely, return discrete
        if best_rms_cube < rms * 1.3:
            return best_R_cube, best_rms_cube

    return R_opt, rms
