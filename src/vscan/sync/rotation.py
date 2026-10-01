"""SO(3) Lie group kinematics, angular velocity integration, and extrinsic rotation estimation.

Conventions:
- R in SO(3): det(R) = +1, R^T * R = I.
- Angular velocity integration preserves exact non-uniform dt (never uniform dt).
- Wahba/Kabsch SVD solver guarantees proper right-handed SO(3) rotations.
- Camera-to-IMU extrinsic rotation: v_imu = R_imu_cam * v_cam.
  Therefore: R_imu = R_imu_cam * R_cam * R_imu_cam^T.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple
import numpy as np


def so3_hat(w: np.ndarray) -> np.ndarray:
    """Compute 3x3 skew-symmetric matrix [w]_x from 3-vector w."""
    w = np.asarray(w, dtype=np.float64).ravel()
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ], dtype=np.float64)


def so3_vee(W: np.ndarray) -> np.ndarray:
    """Extract 3-vector w from 3x3 skew-symmetric matrix W."""
    W = np.asarray(W, dtype=np.float64)
    return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=np.float64)


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Compute matrix exponential on SO(3) using Rodrigues' formula with Taylor expansion."""
    w = np.asarray(w, dtype=np.float64).ravel()
    theta_sq = float(np.dot(w, w))
    theta = math.sqrt(theta_sq)
    I = np.eye(3, dtype=np.float64)

    if theta < 1e-7:
        # Taylor expansion around theta=0:
        # sin(theta)/theta ~ 1 - theta^2 / 6
        # (1 - cos(theta))/theta^2 ~ 1/2 - theta^2 / 24
        W = so3_hat(w)
        W2 = W @ W
        return I + W + 0.5 * W2

    W = so3_hat(w)
    W2 = W @ W
    sin_term = math.sin(theta) / theta
    cos_term = (1.0 - math.cos(theta)) / theta_sq
    return I + sin_term * W + cos_term * W2


def so3_log(R: np.ndarray) -> np.ndarray:
    """Compute matrix logarithm on SO(3) returning axis-angle 3-vector."""
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    cos_phi = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    phi = math.acos(cos_phi)

    if phi < 1e-7:
        # For small angles, (R - R^T) / 2 ~ [w]_x
        return np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ], dtype=np.float64) * 0.5

    if math.pi - phi < 1e-4:
        # Near pi singularity: use diagonal components
        # R + I = 2 * v * v^T (where v is unit rotation axis)
        diag = np.diag(R)
        idx = int(np.argmax(diag))
        col = R[:, idx].copy()
        col[idx] += 1.0
        norm_col = np.linalg.norm(col)
        if norm_col > 1e-8:
            axis = col / norm_col
        else:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return axis * phi

    scale = phi / (2.0 * math.sin(phi))
    return np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64) * scale


def so3_geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Calculate geodesic distance (rotation angle in radians) between R1 and R2 on SO(3)."""
    R_diff = np.asarray(R1, dtype=np.float64) @ np.asarray(R2, dtype=np.float64).T
    w = so3_log(R_diff)
    return float(np.linalg.norm(w))


def integrate_imu_angular_velocity(
    timestamps_s: Sequence[float],
    gyros: Sequence[Sequence[float]],
    t_start: float,
    t_end: float,
    bias: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Integrate angular velocity between t_start and t_end on SO(3).
    
    Preserves exact inter-sample deltas dt_i without replacing with uniform dt.
    Uses trapezoidal midpoint Lie algebra integration.
    """
    ts = np.asarray(timestamps_s, dtype=np.float64)
    w_arr = np.asarray(gyros, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64).ravel() if bias is not None else np.zeros(3, dtype=np.float64)

    if t_start >= t_end:
        return np.eye(3, dtype=np.float64)

    # Find sample indices spanning [t_start, t_end]
    idx_start = int(np.searchsorted(ts, t_start, side="right")) - 1
    idx_start = max(0, min(idx_start, len(ts) - 2))
    idx_end = int(np.searchsorted(ts, t_end, side="left"))
    idx_end = max(idx_start + 1, min(idx_end, len(ts) - 1))

    R_accum = np.eye(3, dtype=np.float64)

    for i in range(idx_start, idx_end):
        t0 = ts[i]
        t1 = ts[i + 1]

        # Clip interval to [t_start, t_end]
        seg_start = max(t0, t_start)
        seg_end = min(t1, t_end)
        if seg_end <= seg_start:
            continue

        dt_seg = seg_end - seg_start
        dt_full = t1 - t0

        if dt_full <= 1e-12:
            continue

        # Linear interpolation of gyros at seg_start and seg_end
        frac0 = (seg_start - t0) / dt_full
        frac1 = (seg_end - t0) / dt_full

        w0 = (1.0 - frac0) * w_arr[i] + frac0 * w_arr[i + 1] - b
        w1 = (1.0 - frac1) * w_arr[i] + frac1 * w_arr[i + 1] - b

        # Midpoint integration on SO(3) Lie algebra
        w_mid = 0.5 * (w0 + w1)
        dR = so3_exp(w_mid * dt_seg)
        R_accum = R_accum @ dR

    return R_accum


def solve_wahba_svd(
    cam_rot_vecs: Sequence[Sequence[float]],
    imu_rot_vecs: Sequence[Sequence[float]],
    weights: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, float]:
    """Solve for extrinsic rotation R_imu_cam in SO(3) aligning camera rotation vectors to IMU.
    
    Relation: r_imu = R_imu_cam * r_cam
    Returns:
        (R_imu_cam, mean_residual_deg)
    """
    v_cam = np.asarray(cam_rot_vecs, dtype=np.float64)
    v_imu = np.asarray(imu_rot_vecs, dtype=np.float64)
    N = len(v_cam)

    if N < 2:
        raise ValueError(f"At least 2 non-collinear rotation samples required for Wahba SVD, got {N}")

    w = np.asarray(weights, dtype=np.float64) if weights is not None else np.ones(N, dtype=np.float64)
    w = w / np.sum(w)

    # Cross-covariance matrix H = sum(w_k * v_imu_k * v_cam_k^T)
    H = np.zeros((3, 3), dtype=np.float64)
    for k in range(N):
        H += w[k] * np.outer(v_imu[k], v_cam[k])

    U, S, Vt = np.linalg.svd(H)
    det = np.linalg.det(U) * np.linalg.det(Vt)
    
    # Ensure proper SO(3) with det = +1 (prevent reflection)
    D = np.diag([1.0, 1.0, det])
    R_imu_cam = U @ D @ Vt

    # Compute residual angles
    residuals = []
    for k in range(N):
        pred_imu = R_imu_cam @ v_cam[k]
        norm_true = np.linalg.norm(v_imu[k])
        norm_pred = np.linalg.norm(pred_imu)
        if norm_true > 1e-6 and norm_pred > 1e-6:
            cos_ang = max(-1.0, min(1.0, float(np.dot(v_imu[k], pred_imu) / (norm_true * norm_pred))))
            residuals.append(math.degrees(math.acos(cos_ang)))
        else:
            residuals.append(0.0)

    mean_res_deg = float(np.mean(residuals)) if residuals else 0.0
    return R_imu_cam, mean_res_deg


def evaluate_extrinsic_residual(
    R_imu_cam: np.ndarray,
    cam_rotations: Sequence[np.ndarray],
    imu_rotations: Sequence[np.ndarray],
) -> Tuple[float, float, List[float]]:
    """Evaluate rotational residuals between R_imu and R_imu_cam * R_cam * R_imu_cam^T.
    
    Returns:
        (mean_res_deg, max_res_deg, list_of_residuals_deg)
    """
    R_ic = np.asarray(R_imu_cam, dtype=np.float64)
    residuals_deg = []

    for R_c, R_i in zip(cam_rotations, imu_rotations):
        # Predicted IMU relative rotation: R_pred = R_ic * R_c * R_ic^T
        R_pred = R_ic @ np.asarray(R_c, dtype=np.float64) @ R_ic.T
        err_rad = so3_geodesic_distance(R_i, R_pred)
        residuals_deg.append(math.degrees(err_rad))

    mean_deg = float(np.mean(residuals_deg)) if residuals_deg else 0.0
    max_deg = float(np.max(residuals_deg)) if residuals_deg else 0.0
    return mean_deg, max_deg, residuals_deg
