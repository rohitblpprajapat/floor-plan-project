"""
On-Manifold IMU Preintegration based on Forster et al. (RSS 2015, TRO 2017).

Preintegrates high-frequency IMU measurements between two keyframe timestamps:
- Delta R_ij in SO(3)
- Delta v_ij in R^3
- Delta p_ij in R^3
- First-order bias Jacobians: dR/dbg, dv/dbg, dv/dba, dp/dbg, dp/dba
- 9x9 or 15x15 covariance propagation
"""

import numpy as np
from typing import List, Optional, Tuple
from src.inertial.imu_types import (
    ImuSample, ImuBias, ImuNoiseParams,
    skew_symmetric, so3_exp, right_jacobian_so3
)

class PreintegratedImu:
    """Represents preintegrated inertial measurements between keyframe i and j."""

    def __init__(self, linearized_bias: ImuBias, noise_params: Optional[ImuNoiseParams] = None):
        self.bias = linearized_bias.copy()
        self.noise_params = noise_params if noise_params is not None else ImuNoiseParams()

        self.delta_t_ij: float = 0.0
        self.delta_R: np.ndarray = np.eye(3, dtype=np.float64)
        self.delta_v: np.ndarray = np.zeros(3, dtype=np.float64)
        self.delta_p: np.ndarray = np.zeros(3, dtype=np.float64)

        # Jacobians w.r.t. bias evaluated at self.bias
        # dR / dbg (3x3)
        self.J_R_bg: np.ndarray = np.zeros((3, 3), dtype=np.float64)
        # dv / dbg (3x3), dv / dba (3x3)
        self.J_v_bg: np.ndarray = np.zeros((3, 3), dtype=np.float64)
        self.J_v_ba: np.ndarray = np.zeros((3, 3), dtype=np.float64)
        # dp / dbg (3x3), dp / dba (3x3)
        self.J_p_bg: np.ndarray = np.zeros((3, 3), dtype=np.float64)
        self.J_p_ba: np.ndarray = np.zeros((3, 3), dtype=np.float64)

        # 9x9 covariance: [delta_theta, delta_v, delta_p]
        self.cov: np.ndarray = np.zeros((9, 9), dtype=np.float64)
        self.sample_count: int = 0

    def integrate_measurement(self, sample: ImuSample, dt: float):
        """Integrate a single IMU sample over time duration dt."""
        if dt <= 0.0 or dt > 0.5:
            # Reject invalid or non-physical delta t
            return

        self.sample_count += 1
        self.delta_t_ij += dt

        # Unbiased measurements
        omega_unbiased = sample.gyro - self.bias.gyro_bias
        accel_unbiased = sample.accel - self.bias.accel_bias

        # Rotation update
        step_rot = so3_exp(omega_unbiased * dt)
        Jr = right_jacobian_so3(omega_unbiased * dt)
        acc_skew = skew_symmetric(accel_unbiased)
        curr_R = self.delta_R

        # Update Jacobians (before updating delta_R, delta_v, delta_p)
        # dR_next/dbg = step_rot.T * dR_curr/dbg - Jr * dt
        next_J_R_bg = step_rot.T @ self.J_R_bg - Jr * dt

        # dv_next/dbg = dv_curr/dbg - curr_R * acc_skew * dR_curr/dbg * dt
        next_J_v_bg = self.J_v_bg - curr_R @ acc_skew @ self.J_R_bg * dt
        # dv_next/dba = dv_curr/dba - curr_R * dt
        next_J_v_ba = self.J_v_ba - curr_R * dt

        # dp_next/dbg = dp_curr/dbg + dv_curr/dbg * dt - 0.5 * curr_R * acc_skew * dR_curr/dbg * dt^2
        next_J_p_bg = self.J_p_bg + self.J_v_bg * dt - 0.5 * curr_R @ acc_skew @ self.J_R_bg * (dt ** 2)
        # dp_next/dba = dp_curr/dba + dv_curr/dba * dt - 0.5 * curr_R * dt^2
        next_J_p_ba = self.J_p_ba + self.J_v_ba * dt - 0.5 * curr_R * (dt ** 2)

        self.J_R_bg = next_J_R_bg
        self.J_v_bg = next_J_v_bg
        self.J_v_ba = next_J_v_ba
        self.J_p_bg = next_J_p_bg
        self.J_p_ba = next_J_p_ba

        # Covariance propagation: error state [delta_theta, delta_v, delta_p]
        A = np.eye(9, dtype=np.float64)
        A[0:3, 0:3] = step_rot.T
        A[3:6, 0:3] = -curr_R @ acc_skew * dt
        A[6:9, 0:3] = -0.5 * curr_R @ acc_skew * (dt ** 2)
        A[6:9, 3:6] = np.eye(3) * dt

        B = np.zeros((9, 6), dtype=np.float64)
        B[0:3, 0:3] = Jr * dt
        B[3:6, 3:6] = curr_R * dt
        B[6:9, 3:6] = 0.5 * curr_R * (dt ** 2)

        # Continuous noise covariance scaled by 1 / dt
        Q = np.zeros((6, 6), dtype=np.float64)
        Q[0:3, 0:3] = np.eye(3) * (self.noise_params.gyro_noise_density ** 2) / dt
        Q[3:6, 3:6] = np.eye(3) * (self.noise_params.accel_noise_density ** 2) / dt

        self.cov = A @ self.cov @ A.T + B @ Q @ B.T

        # Update states
        # Position: p_next = p + v * dt + 0.5 * R * a * dt^2
        self.delta_p = self.delta_p + self.delta_v * dt + 0.5 * curr_R @ accel_unbiased * (dt ** 2)
        # Velocity: v_next = v + R * a * dt
        self.delta_v = self.delta_v + curr_R @ accel_unbiased * dt
        # Rotation: R_next = R * Exp(w * dt)
        self.delta_R = self.delta_R @ step_rot

    def predict_delta_R(self, new_gyro_bias: np.ndarray) -> np.ndarray:
        """First-order correction of delta_R for a modified gyro bias."""
        d_bg = np.asarray(new_gyro_bias, dtype=np.float64).reshape(3,) - self.bias.gyro_bias
        return self.delta_R @ so3_exp(self.J_R_bg @ d_bg)

    def predict_delta_v(self, new_bias: ImuBias) -> np.ndarray:
        """First-order correction of delta_v for modified biases."""
        d_bg = new_bias.gyro_bias - self.bias.gyro_bias
        d_ba = new_bias.accel_bias - self.bias.accel_bias
        return self.delta_v + self.J_v_bg @ d_bg + self.J_v_ba @ d_ba

    def predict_delta_p(self, new_bias: ImuBias) -> np.ndarray:
        """First-order correction of delta_p for modified biases."""
        d_bg = new_bias.gyro_bias - self.bias.gyro_bias
        d_ba = new_bias.accel_bias - self.bias.accel_bias
        return self.delta_p + self.J_p_bg @ d_bg + self.J_p_ba @ d_ba


def preintegrate_segment(
    samples: List[ImuSample],
    t_start: float,
    t_end: float,
    linearized_bias: ImuBias,
    noise_params: Optional[ImuNoiseParams] = None
) -> Optional[PreintegratedImu]:
    """
    Preintegrate IMU measurements within [t_start, t_end] with high precision.
    Interpolates boundary samples to match exact timestamps t_start and t_end.
    """
    if t_end <= t_start or len(samples) < 2:
        return None

    preint = PreintegratedImu(linearized_bias, noise_params)

    # Find relevant samples
    sub = [s for s in samples if t_start - 0.1 <= s.timestamp <= t_end + 0.1]
    if len(sub) < 2:
        return None

    # Step through samples and integrate
    prev_t = t_start
    # Interpolate sample at t_start
    first_idx = 0
    while first_idx < len(sub) and sub[first_idx].timestamp < t_start:
        first_idx += 1

    curr_idx = max(0, first_idx - 1)
    t_curr = sub[curr_idx].timestamp

    for k in range(curr_idx, len(sub) - 1):
        s_k = sub[k]
        s_next = sub[k + 1]

        t0 = max(t_start, s_k.timestamp)
        t1 = min(t_end, s_next.timestamp)

        if t1 > t0:
            dt = t1 - t0
            # Interpolated sample midpoint
            alpha = (0.5 * (t0 + t1) - s_k.timestamp) / max(1e-9, (s_next.timestamp - s_k.timestamp))
            alpha = np.clip(alpha, 0.0, 1.0)
            interp_accel = (1.0 - alpha) * s_k.accel + alpha * s_next.accel
            interp_gyro = (1.0 - alpha) * s_k.gyro + alpha * s_next.gyro
            interp_sample = ImuSample(timestamp=0.5 * (t0 + t1), accel=interp_accel, gyro=interp_gyro)

            preint.integrate_measurement(interp_sample, dt)

        if s_next.timestamp >= t_end:
            break

    return preint if preint.sample_count > 0 else None
