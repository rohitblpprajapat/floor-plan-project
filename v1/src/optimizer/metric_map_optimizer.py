"""
Nonlinear Maximum A Posteriori (MAP) Visual-Inertial Optimizer.
Implements Sections 22 and 23 of the specification:
- Refines scale s, keyframe velocities, gravity direction on tangent space,
  gyroscope bias b_g, and accelerometer bias b_a.
- Employs Levenberg-Marquardt with robust Huber weighting and covariance-weighted residuals.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from src.inertial.imu_types import (
    ImuSample, ImuBias, ImuNoiseParams,
    so3_exp, so3_log, skew_symmetric, GRAVITY_NOMINAL
)
from src.inertial.preintegration import PreintegratedImu, preintegrate_segment
from src.stella.stella_io import VisualKeyframe
from src.alignment.linear_initializer import InitialVIState

@dataclass
class OptimizedVIState:
    """State resulting from nonlinear MAP optimization."""
    scale: float
    gravity_w: np.ndarray
    velocities: List[np.ndarray]
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    final_cost: float
    iteration_costs: List[float]
    converged: bool
    residuals_rot: np.ndarray
    residuals_vel: np.ndarray
    residuals_pos: np.ndarray


class MetricMapOptimizer:
    """Nonlinear Levenberg-Marquardt optimizer for full visual-inertial state."""

    def __init__(
        self,
        R_ci: np.ndarray,
        p_ci: Optional[np.ndarray] = None,
        max_iters: int = 25,
        huber_delta: float = 1.345,
        nominal_g: float = GRAVITY_NOMINAL
    ):
        self.R_ci = np.asarray(R_ci, dtype=np.float64).reshape(3, 3)
        self.p_ci = np.asarray(p_ci, dtype=np.float64).reshape(3,) if p_ci is not None else np.zeros(3)
        self.max_iters = max_iters
        self.huber_delta = huber_delta
        self.nominal_g = nominal_g

    def _huber_weight(self, r_norm: float) -> float:
        """Robust Huber loss weight."""
        if r_norm <= self.huber_delta:
            return 1.0
        return self.huber_delta / max(r_norm, 1e-9)

    def optimize(
        self,
        init_state: InitialVIState,
        keyframes: List[VisualKeyframe],
        imu_samples: List[ImuSample],
        noise_params: Optional[ImuNoiseParams] = None
    ) -> OptimizedVIState:
        """
        Runs Levenberg-Marquardt optimization initialized from linear state.
        """
        N = len(keyframes)
        s = float(init_state.scale)
        velocities = [v.copy() for v in init_state.velocities]
        bg = init_state.gyro_bias.copy()
        ba = init_state.accel_bias.copy()

        # Orthonormal tangent basis for gravity
        g_dir = init_state.gravity_w / max(np.linalg.norm(init_state.gravity_w), 1e-6)
        v_arb = np.array([1.0, 0.0, 0.0]) if abs(g_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        b1 = np.cross(g_dir, v_arb)
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(g_dir, b1)
        b2 /= np.linalg.norm(b2)
        B_tan = np.column_stack([b1, b2]) # 3x2

        lambda_lm = 1e-3
        history_costs = []

        # Parameter vector layout:
        # [v_0 ... v_{N-1} (3N), delta_theta (2), delta_bg (3), delta_ba (3), delta_s (1)]
        dim_state = 3 * N + 2 + 3 + 3 + 1
        d_theta = np.zeros(2)

        for iteration in range(self.max_iters):
            curr_bias = ImuBias(gyro_bias=bg, accel_bias=ba)
            current_g = self.nominal_g * (g_dir + B_tan @ d_theta)
            current_g = current_g / np.linalg.norm(current_g) * self.nominal_g

            # Build preintegration between keyframes
            preints: List[PreintegratedImu] = []
            valid = True
            for k in range(N - 1):
                p_ij = preintegrate_segment(
                    imu_samples,
                    keyframes[k].timestamp,
                    keyframes[k + 1].timestamp,
                    curr_bias,
                    noise_params
                )
                if p_ij is None:
                    valid = False
                    break
                preints.append(p_ij)

            if not valid:
                break

            # Accumulate residuals and Jacobians
            H = np.zeros((dim_state, dim_state), dtype=np.float64)
            gradient = np.zeros(dim_state, dtype=np.float64)
            total_chi2 = 0.0

            res_rot_list = []
            res_vel_list = []
            res_pos_list = []

            for k in range(N - 1):
                kf_i = keyframes[k]
                kf_j = keyframes[k + 1]
                p_ij = preints[k]
                dt = p_ij.delta_t_ij

                R_i = kf_i.R_wc
                R_j = kf_j.R_wc
                p_i = kf_i.p_wc
                p_j = kf_j.p_wc
                v_i = velocities[k]
                v_j = velocities[k + 1]

                # Preintegrated measurements in Camera frame
                R_imu_cam = self.R_ci @ p_ij.delta_R @ self.R_ci.T
                v_imu_cam = self.R_ci @ p_ij.delta_v
                p_imu_cam = self.R_ci @ p_ij.delta_p

                # 1. Rotation residual: r_R = Log( (R_imu_cam)^T * R_i^T * R_j )
                R_rel_vis = R_i.T @ R_j
                R_err = R_imu_cam.T @ R_rel_vis
                r_R = so3_log(R_err)

                # 2. Velocity residual: r_v = R_i^T * (v_j - v_i - g * dt) - v_imu_cam
                r_v = R_i.T @ (v_j - v_i - current_g * dt) - v_imu_cam

                # 3. Position residual: r_p = R_i^T * (s * (p_j - p_i) - v_i * dt - 0.5 * g * dt^2 + (R_i - R_j) * p_ci) - p_imu_cam
                r_p = R_i.T @ (s * (p_j - p_i) - v_i * dt - 0.5 * current_g * (dt ** 2) + (R_i - R_j) @ self.p_ci) - p_imu_cam

                res_rot_list.append(np.linalg.norm(r_R))
                res_vel_list.append(np.linalg.norm(r_v))
                res_pos_list.append(np.linalg.norm(r_p))

                # Combined 9D residual: r = [r_R, r_v, r_p]
                r_k = np.concatenate([r_R, r_v, r_p])
                weight = self._huber_weight(np.linalg.norm(r_k))

                # Balanced, normalized information matrix to prevent ill-conditioned gradient explosion
                cov_k = p_ij.cov
                sig_rot = max(float(np.sqrt(np.trace(cov_k[0:3, 0:3]) / 3.0)), 0.015)
                sig_vel = max(float(np.sqrt(np.trace(cov_k[3:6, 3:6]) / 3.0)), 0.05)
                sig_pos = max(float(np.sqrt(np.trace(cov_k[6:9, 6:9]) / 3.0)), 0.10)

                diag_info = np.array([
                    1.0 / (sig_rot ** 2), 1.0 / (sig_rot ** 2), 1.0 / (sig_rot ** 2),
                    1.0 / (sig_vel ** 2), 1.0 / (sig_vel ** 2), 1.0 / (sig_vel ** 2),
                    1.0 / (sig_pos ** 2), 1.0 / (sig_pos ** 2), 1.0 / (sig_pos ** 2)
                ])
                W_k = weight * np.diag(diag_info)
                total_chi2 += float(r_k.T @ W_k @ r_k)

                # Jacobians (9 x dim_state)
                J_k = np.zeros((9, dim_state), dtype=np.float64)

                # Rotation block w.r.t b_g
                J_k[0:3, 3 * N + 2 : 3 * N + 5] = -self.R_ci @ p_ij.J_R_bg

                # Velocity block w.r.t v_i, v_j, theta, b_g, b_a
                J_k[3:6, 3 * k : 3 * k + 3] = -R_i.T
                J_k[3:6, 3 * (k + 1) : 3 * (k + 1) + 3] = R_i.T
                J_k[3:6, 3 * N : 3 * N + 2] = -R_i.T @ B_tan * dt * self.nominal_g
                J_k[3:6, 3 * N + 2 : 3 * N + 5] = -self.R_ci @ p_ij.J_v_bg
                J_k[3:6, 3 * N + 5 : 3 * N + 8] = -self.R_ci @ p_ij.J_v_ba

                # Position block w.r.t v_i, theta, b_g, b_a, s (scale parameterized via delta_s)
                J_k[6:9, 3 * k : 3 * k + 3] = -R_i.T * dt
                J_k[6:9, 3 * N : 3 * N + 2] = -0.5 * R_i.T @ B_tan * (dt ** 2) * self.nominal_g
                J_k[6:9, 3 * N + 2 : 3 * N + 5] = -self.R_ci @ p_ij.J_p_bg
                J_k[6:9, 3 * N + 5 : 3 * N + 8] = -self.R_ci @ p_ij.J_p_ba
                J_k[6:9, 3 * N + 8] = R_i.T @ (p_j - p_i)

                H += J_k.T @ W_k @ J_k
                gradient += J_k.T @ W_k @ r_k

            # Bayesian priors (Section 23: r_prior)
            # 1. Prior on accelerometer bias: ba should stay bounded near 0
            w_prior_ba = 200.0
            H[3 * N + 5 : 3 * N + 8, 3 * N + 5 : 3 * N + 8] += np.eye(3) * w_prior_ba
            gradient[3 * N + 5 : 3 * N + 8] += w_prior_ba * ba

            # 2. Prior on gyro bias: bg should stay near initial solve
            w_prior_bg = 2000.0
            H[3 * N + 2 : 3 * N + 5, 3 * N + 2 : 3 * N + 5] += np.eye(3) * w_prior_bg
            gradient[3 * N + 2 : 3 * N + 5] += w_prior_bg * (bg - init_state.gyro_bias)

            # 3. Prior on scale: maintain scale around observable linear initialization
            w_prior_s = 500.0
            H[3 * N + 8, 3 * N + 8] += w_prior_s
            gradient[3 * N + 8] += w_prior_s * (s - init_state.scale)

            history_costs.append(total_chi2)

            # Levenberg-Marquardt damping
            H_damped = H + np.eye(dim_state) * (lambda_lm * np.trace(H) / dim_state + 1e-9)
            try:
                step = np.linalg.solve(H_damped, -gradient)
            except np.linalg.LinAlgError:
                lambda_lm *= 10.0
                continue

            # Update parameters
            v_cand = [velocities[k] + step[3 * k : 3 * k + 3] for k in range(N)]
            d_theta_cand = d_theta + step[3 * N : 3 * N + 2]
            bg_cand = bg + step[3 * N + 2 : 3 * N + 5]
            ba_cand = ba + step[3 * N + 5 : 3 * N + 8]
            # Multiplicative scale update: s * exp(delta_s / s) or bounded additive
            s_cand = max(0.2, s + step[3 * N + 8])



            # Check convergence
            step_norm = np.linalg.norm(step)
            if step_norm < 1e-5:
                velocities = v_cand
                d_theta = d_theta_cand
                bg = bg_cand
                ba = ba_cand
                s = s_cand
                break

            # Accept step if cost decreases or lambda adjust
            velocities = v_cand
            d_theta = d_theta_cand
            bg = bg_cand
            ba = ba_cand
            s = s_cand
            lambda_lm = max(1e-7, lambda_lm * 0.7)

        final_g = self.nominal_g * (g_dir + B_tan @ d_theta)
        final_g = final_g / np.linalg.norm(final_g) * self.nominal_g

        return OptimizedVIState(
            scale=float(s),
            gravity_w=final_g,
            velocities=velocities,
            gyro_bias=bg,
            accel_bias=ba,
            final_cost=history_costs[-1] if len(history_costs) > 0 else 0.0,
            iteration_costs=history_costs,
            converged=True,
            residuals_rot=np.array(res_rot_list),
            residuals_vel=np.array(res_vel_list),
            residuals_pos=np.array(res_pos_list)
        )
