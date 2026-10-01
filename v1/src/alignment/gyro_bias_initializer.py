"""
Gyroscope Bias Initializer.
Estimates gyroscope bias b_g by minimizing rotation disagreement between visual relative rotations
and IMU preintegrated rotations (Section 19 of specification).
"""

import numpy as np
from typing import List, Tuple, Optional
from src.inertial.imu_types import ImuSample, ImuBias, so3_log, so3_exp, right_jacobian_so3
from src.inertial.preintegration import PreintegratedImu, preintegrate_segment
from src.stella.stella_io import VisualKeyframe

class GyroBiasInitializer:
    """Solves for initial gyroscope bias using visual rotation reference."""

    def __init__(self, R_ci: np.ndarray, max_iterations: int = 15, tol: float = 1e-6):
        self.R_ci = np.asarray(R_ci, dtype=np.float64).reshape(3, 3) # Camera <- IMU
        self.max_iterations = max_iterations
        self.tol = tol

    def solve(
        self,
        keyframes: List[VisualKeyframe],
        imu_samples: List[ImuSample],
        initial_bg: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Solves for b_g in rad/s.
        Returns: (b_g_optimal, final_rms_rot_error_rad, iteration_errors)
        """
        if len(keyframes) < 3:
            raise ValueError("Need at least 3 keyframes for gyro bias initialization.")

        bg = initial_bg.copy() if initial_bg is not None else np.zeros(3, dtype=np.float64)
        history_errors = []

        for iteration in range(self.max_iterations):
            # Accumulate normal equations: H * delta_bg = g
            H = np.zeros((3, 3), dtype=np.float64)
            g = np.zeros(3, dtype=np.float64)
            total_sq_err = 0.0
            count = 0

            current_bias = ImuBias(gyro_bias=bg, accel_bias=np.zeros(3))

            for k in range(len(keyframes) - 1):
                kf_i = keyframes[k]
                kf_j = keyframes[k + 1]

                # Visual relative rotation in Camera frame: Delta R_visual = R_i.T * R_j
                delta_R_vis = kf_i.R_wc.T @ kf_j.R_wc

                # Preintegrate IMU between keyframes
                preint = preintegrate_segment(imu_samples, kf_i.timestamp, kf_j.timestamp, current_bias)
                if preint is None or preint.sample_count < 2:
                    continue

                # IMU preintegrated rotation in IMU frame: delta_R_imu
                # Express in Camera frame: Delta R_imu_cam = R_ci * delta_R_imu * R_ci.T
                delta_R_imu_cam = self.R_ci @ preint.delta_R @ self.R_ci.T

                # Rotation error: r = Log((delta_R_vis)^T * delta_R_imu_cam)
                R_err = delta_R_vis.T @ delta_R_imu_cam
                r_vec = so3_log(R_err)
                err_norm = np.linalg.norm(r_vec)
                total_sq_err += err_norm ** 2
                count += 1

                # Jacobian of r_vec w.r.t b_g
                # d(R_err)/d(b_g) approx R_ci * J_R_bg
                J_k = self.R_ci @ preint.J_R_bg

                # Weight by inverse covariance if available
                cov_rot = preint.cov[0:3, 0:3]
                try:
                    W = np.linalg.inv(cov_rot + np.eye(3) * 1e-7)
                except np.linalg.LinAlgError:
                    W = np.eye(3)

                H += J_k.T @ W @ J_k
                g += J_k.T @ W @ r_vec

            if count == 0:
                break

            rms = np.sqrt(total_sq_err / count)
            history_errors.append(float(rms))

            # Add Levenberg-Marquardt damping
            H_damped = H + np.eye(3) * (1e-6 * np.trace(H) + 1e-9)
            try:
                delta_bg = np.linalg.solve(H_damped, -g)
            except np.linalg.LinAlgError:
                break

            bg += delta_bg

            if np.linalg.norm(delta_bg) < self.tol:
                break

        final_rms = history_errors[-1] if len(history_errors) > 0 else 0.0
        return bg, final_rms, history_errors
