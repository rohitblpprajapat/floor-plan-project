"""
Linear Visual-Inertial Alignment and Gravity Refinement.
Implements Sections 20 and 21 of the specification:
- Linear solve for velocities [v_0, ..., v_N], gravity vector g_w, and metric scale s
- Singular-value observability inspection and condition number checks
- Gravity direction refinement on the tangent plane with fixed 9.80665 m/s^2 magnitude
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from src.inertial.imu_types import ImuSample, ImuBias, GRAVITY_NOMINAL
from src.inertial.preintegration import PreintegratedImu, preintegrate_segment
from src.stella.stella_io import VisualKeyframe

@dataclass
class InitialVIState:
    """Estimated visual-inertial initialization state."""
    scale: float                     # Metric scale s > 0
    gravity_w: np.ndarray            # Gravity vector in World frame (3,)
    velocities: List[np.ndarray]     # Keyframe velocities in World frame
    condition_number: float          # Linear system condition number
    singular_values: np.ndarray      # SVD singular values of linear system
    is_observable: bool              # True if system is well-conditioned
    gravity_norm: float              # Norm of unconstrained gravity vector
    gyro_bias: np.ndarray            # Gyroscope bias used during initialization
    accel_bias: np.ndarray           # Estimated or prior accelerometer bias


class LinearVIInitializer:
    """
    Solves the linear system A * X = b for keyframe velocities, gravity, and metric scale.
    """

    def __init__(self, R_ci: np.ndarray, p_ci: Optional[np.ndarray] = None):
        self.R_ci = np.asarray(R_ci, dtype=np.float64).reshape(3, 3)
        self.p_ci = np.asarray(p_ci, dtype=np.float64).reshape(3,) if p_ci is not None else np.zeros(3)

    def solve(
        self,
        keyframes: List[VisualKeyframe],
        imu_samples: List[ImuSample],
        gyro_bias: np.ndarray,
        accel_bias: Optional[np.ndarray] = None,
        cond_threshold: float = 1e4
    ) -> Optional[InitialVIState]:
        """
        Formulates and solves the linear system for [v_0, ..., v_{N-1}, g, s].
        """
        N = len(keyframes)
        if N < 4:
            return None

        bg = np.asarray(gyro_bias, dtype=np.float64).reshape(3,)
        ba = np.asarray(accel_bias, dtype=np.float64).reshape(3,) if accel_bias is not None else np.zeros(3)
        bias = ImuBias(gyro_bias=bg, accel_bias=ba)

        # Preintegrate between consecutive keyframes
        preints: List[PreintegratedImu] = []
        for k in range(N - 1):
            t_i = keyframes[k].timestamp
            t_j = keyframes[k + 1].timestamp
            p_ij = preintegrate_segment(imu_samples, t_i, t_j, bias)
            if p_ij is None or p_ij.sample_count < 2:
                return None
            preints.append(p_ij)

        # Dimension of state: 3 * N (velocities) + 3 (gravity) + 1 (scale) = 3N + 4
        num_states = 3 * N + 4
        # Each interval gives 6 constraints (3 position, 3 velocity)
        num_equations = 6 * (N - 1)

        A = np.zeros((num_equations, num_states), dtype=np.float64)
        b = np.zeros(num_equations, dtype=np.float64)

        for k in range(N - 1):
            kf_i = keyframes[k]
            kf_j = keyframes[k + 1]
            dt = preints[k].delta_t_ij

            # Preintegrated values in Camera frame
            alpha_ij = self.R_ci @ preints[k].delta_p
            beta_ij = self.R_ci @ preints[k].delta_v

            # Visual rotations and delta positions
            R_i = kf_i.R_wc
            R_j = kf_j.R_wc
            delta_p_vis = kf_j.p_wc - kf_i.p_wc

            row_p = 6 * k
            row_v = 6 * k + 3

            # --- Position equation ---
            # v_i * dt + 0.5 * g * dt^2 - s * (p_j - p_i) = - R_i * alpha_ij - (R_i - R_j) * p_ci
            A[row_p : row_p + 3, 3 * k : 3 * k + 3] = np.eye(3) * dt
            A[row_p : row_p + 3, 3 * N : 3 * N + 3] = 0.5 * np.eye(3) * (dt ** 2)
            A[row_p : row_p + 3, 3 * N + 3] = -delta_p_vis

            b[row_p : row_p + 3] = -R_i @ alpha_ij - (R_i - R_j) @ self.p_ci

            # --- Velocity equation ---
            # -v_i + v_{i+1} - g * dt = R_i * beta_ij
            A[row_v : row_v + 3, 3 * k : 3 * k + 3] = -np.eye(3)
            A[row_v : row_v + 3, 3 * (k + 1) : 3 * (k + 1) + 3] = np.eye(3)
            A[row_v : row_v + 3, 3 * N : 3 * N + 3] = -np.eye(3) * dt

            b[row_v : row_v + 3] = R_i @ beta_ij

        # Inspect singular values and condition number
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        cond = S[0] / max(S[-1], 1e-12)
        is_observable = bool(cond < cond_threshold and S[-1] > 1e-4)

        # Weighted least squares or Moore-Penrose solve
        X = Vt.T @ (np.diag(1.0 / np.maximum(S, 1e-10))) @ (U.T @ b)

        velocities = [X[3 * k : 3 * k + 3].copy() for k in range(N)]
        g_w = X[3 * N : 3 * N + 3].copy()
        scale = float(X[3 * N + 3])
        g_norm = float(np.linalg.norm(g_w))

        return InitialVIState(
            scale=scale,
            gravity_w=g_w,
            velocities=velocities,
            condition_number=float(cond),
            singular_values=S,
            is_observable=is_observable,
            gravity_norm=g_norm,
            gyro_bias=bg,
            accel_bias=ba
        )


class GravityRefiner:
    """
    Refines gravity direction on the tangent plane with magnitude constrained to nominal (9.80665 m/s^2).
    Section 21 of specification.
    """

    def __init__(self, nominal_g: float = GRAVITY_NOMINAL):
        self.nominal_g = nominal_g

    def refine(
        self,
        initial_state: InitialVIState,
        keyframes: List[VisualKeyframe],
        imu_samples: List[ImuSample],
        R_ci: np.ndarray,
        p_ci: Optional[np.ndarray] = None,
        max_iters: int = 10
    ) -> InitialVIState:
        """
        Optimizes delta_theta on tangent space of g_0 and jointly refines velocities, scale, and accel bias.
        """
        N = len(keyframes)
        g0 = initial_state.gravity_w
        if np.linalg.norm(g0) < 1e-3:
            g_dir = np.array([0.0, 0.0, -1.0])
        else:
            g_dir = g0 / np.linalg.norm(g0)

        # Orthonormal basis b1, b2 perp to g_dir
        v_arb = np.array([1.0, 0.0, 0.0]) if abs(g_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        b1 = np.cross(g_dir, v_arb)
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(g_dir, b1)
        b2 /= np.linalg.norm(b2)
        B_tan = np.column_stack([b1, b2]) # 3x2

        curr_state = initial_state
        R_ci = np.asarray(R_ci, dtype=np.float64).reshape(3, 3)
        p_ci = np.asarray(p_ci, dtype=np.float64).reshape(3,) if p_ci is not None else np.zeros(3)

        # Preintegrate
        preints = []
        for k in range(N - 1):
            t_i = keyframes[k].timestamp
            t_j = keyframes[k + 1].timestamp
            p_ij = preintegrate_segment(imu_samples, t_i, t_j, ImuBias(gyro_bias=curr_state.gyro_bias, accel_bias=curr_state.accel_bias))
            preints.append(p_ij)

        # State vector: [v_0 ... v_{N-1}, delta_theta (2), delta_ba (3), s]
        # Size: 3N + 2 + 3 + 1 = 3N + 6
        num_states = 3 * N + 6
        num_equations = 6 * (N - 1)

        for iteration in range(max_iters):
            A = np.zeros((num_equations, num_states), dtype=np.float64)
            b = np.zeros(num_equations, dtype=np.float64)

            g_fixed = self.nominal_g * g_dir

            for k in range(N - 1):
                kf_i = keyframes[k]
                kf_j = keyframes[k + 1]
                p_ij = preints[k]
                if p_ij is None:
                    continue
                dt = p_ij.delta_t_ij

                alpha_ij = R_ci @ p_ij.delta_p
                beta_ij = R_ci @ p_ij.delta_v
                J_p_ba_cam = R_ci @ p_ij.J_p_ba
                J_v_ba_cam = R_ci @ p_ij.J_v_ba

                R_i = kf_i.R_wc
                R_j = kf_j.R_wc
                delta_p_vis = kf_j.p_wc - kf_i.p_wc

                row_p = 6 * k
                row_v = 6 * k + 3

                # Position row
                A[row_p : row_p + 3, 3 * k : 3 * k + 3] = np.eye(3) * dt
                # Gravity tangent derivative: 0.5 * B_tan * dt^2
                A[row_p : row_p + 3, 3 * N : 3 * N + 2] = 0.5 * B_tan * (dt ** 2) * self.nominal_g
                # Accel bias Jacobian
                A[row_p : row_p + 3, 3 * N + 2 : 3 * N + 5] = -R_i @ J_p_ba_cam
                # Scale
                A[row_p : row_p + 3, 3 * N + 5] = -delta_p_vis

                b[row_p : row_p + 3] = -R_i @ alpha_ij - (R_i - R_j) @ p_ci - 0.5 * g_fixed * (dt ** 2)

                # Velocity row
                A[row_v : row_v + 3, 3 * k : 3 * k + 3] = -np.eye(3)
                A[row_v : row_v + 3, 3 * (k + 1) : 3 * (k + 1) + 3] = np.eye(3)
                A[row_v : row_v + 3, 3 * N : 3 * N + 2] = -B_tan * dt * self.nominal_g
                A[row_v : row_v + 3, 3 * N + 2 : 3 * N + 5] = -R_i @ J_v_ba_cam
                b[row_v : row_v + 3] = R_i @ beta_ij + g_fixed * dt

            # Tikhonov regularization on accel bias (ba should not absorb gravity):
            # Prior: d_ba ~ 0 with sigma ~ 0.1 m/s^2 -> weight = 10.0
            # Prior on scale: s ~ curr_state.scale with weight = 2.0
            A_reg = np.zeros((4, num_states), dtype=np.float64)
            b_reg = np.zeros(4, dtype=np.float64)
            A_reg[0:3, 3 * N + 2 : 3 * N + 5] = np.eye(3) * 10.0
            A_reg[3, 3 * N + 5] = 2.0
            b_reg[3] = 2.0 * curr_state.scale


            A_aug = np.vstack([A, A_reg])
            b_aug = np.concatenate([b, b_reg])


            sol, residuals, rank, s_vals = np.linalg.lstsq(A_aug, b_aug, rcond=1e-5)

            v_new = [sol[3 * k : 3 * k + 3].copy() for k in range(N)]
            d_theta = sol[3 * N : 3 * N + 2]
            d_ba = sol[3 * N + 2 : 3 * N + 5]
            scale_new = float(sol[3 * N + 5])


            # Update gravity direction on sphere
            g_perturbed = g_dir + B_tan @ d_theta
            g_dir = g_perturbed / np.linalg.norm(g_perturbed)

            # Re-orthogonalize tangent basis
            b1 = np.cross(g_dir, v_arb)
            b1 /= np.linalg.norm(b1)
            b2 = np.cross(g_dir, b1)
            b2 /= np.linalg.norm(b2)
            B_tan = np.column_stack([b1, b2])

            ba_new = curr_state.accel_bias + d_ba

            if np.linalg.norm(d_theta) < 1e-5:
                break

        final_g = self.nominal_g * g_dir
        return InitialVIState(
            scale=scale_new,
            gravity_w=final_g,
            velocities=v_new,
            condition_number=curr_state.condition_number,
            singular_values=curr_state.singular_values,
            is_observable=curr_state.is_observable,
            gravity_norm=float(np.linalg.norm(final_g)),
            gyro_bias=curr_state.gyro_bias,
            accel_bias=ba_new
        )
