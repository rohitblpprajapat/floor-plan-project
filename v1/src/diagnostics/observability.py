"""
Observability Diagnostics and Automated Quality Reporting.
Implements Sections 29, 37, and 38 of the specification:
- Multi-window scale stability check
- Gravity and bias physical bounds verification
- Velocity realism check
- Automated generation of required diagnostic plots
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from src.inertial.imu_types import ImuSample, ImuBias, GRAVITY_NOMINAL
from src.stella.stella_io import VisualKeyframe, VisualTrajectory
from src.alignment.linear_initializer import LinearVIInitializer, InitialVIState
from src.optimizer.metric_map_optimizer import OptimizedVIState

@dataclass
class ObservabilityReport:
    """Consolidated observability and quality check report."""
    passed_all: bool
    scale_stability_passed: bool
    scale_mean: float
    scale_std: float
    window_scales: List[float]
    gravity_norm_passed: bool
    gravity_norm: float
    bias_bounded_passed: bool
    gyro_bias_norm: float
    accel_bias_norm: float
    velocity_sanity_passed: bool
    max_speed_mps: float
    failure_reasons: List[str]


class ObservabilityChecker:
    """Verifies system observability and consistency before accepting metric result."""

    def __init__(
        self,
        R_ci: np.ndarray,
        p_ci: Optional[np.ndarray] = None,
        max_scale_cv: float = 0.20,         # Max coefficient of variation for scale (20%)
        gravity_tol_mps2: float = 0.6,      # Acceptable deviation from 9.81 m/s^2
        max_gyro_bias_rad_s: float = 0.25,   # Max physical gyro bias
        max_accel_bias_mps2: float = 1.8,   # Max physical accel bias
        max_velocity_mps: float = 15.0       # Max physical walking/hand speed
    ):
        self.R_ci = R_ci
        self.p_ci = p_ci
        self.max_scale_cv = max_scale_cv
        self.gravity_tol_mps2 = gravity_tol_mps2
        self.max_gyro_bias = max_gyro_bias_rad_s
        self.max_accel_bias = max_accel_bias_mps2
        self.max_velocity = max_velocity_mps

    def check_stability_across_windows(
        self,
        keyframes: List[VisualKeyframe],
        imu_samples: List[ImuSample],
        gyro_bias: np.ndarray,
        window_sec: float = 10.0,
        slide_sec: float = 5.0
    ) -> Tuple[bool, List[float], float, float]:
        """
        Estimates scale across sliding temporal windows (Section 29).
        Returns: (passed, window_scales, mean_scale, std_scale)
        """
        if len(keyframes) < 8:
            return False, [], 0.0, 0.0

        ts = np.array([kf.timestamp for kf in keyframes])
        t_min = ts[0]
        t_max = ts[-1]

        initializer = LinearVIInitializer(self.R_ci, self.p_ci)
        window_scales = []

        curr_t = t_min
        while curr_t + window_sec <= t_max + 1e-4:
            sub_kfs = [kf for kf in keyframes if curr_t <= kf.timestamp <= curr_t + window_sec]
            if len(sub_kfs) >= 5:
                sub_imu = [s for s in imu_samples if curr_t - 0.1 <= s.timestamp <= curr_t + window_sec + 0.1]
                res = initializer.solve(sub_kfs, sub_imu, gyro_bias=gyro_bias)
                if res is not None and res.is_observable and res.scale > 0.1 and res.condition_number < 250.0:
                    window_scales.append(res.scale)
            curr_t += slide_sec

        if len(window_scales) < 2:
            return True, window_scales, window_scales[0] if len(window_scales) > 0 else 0.0, 0.0

        # Filter extreme statistical outliers (> 2 IQR)
        arr = np.array(window_scales)
        q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q75 - q25
        valid_arr = arr[(arr >= q25 - 1.5 * iqr) & (arr <= q75 + 1.5 * iqr)]
        if len(valid_arr) >= 2:
            arr = valid_arr

        mean_s = float(np.mean(arr))
        std_s = float(np.std(arr))
        cv = std_s / max(mean_s, 1e-6)

        # Passing threshold: scale relative spread <= max_scale_cv (25%)
        passed = bool(cv <= self.max_scale_cv)
        return passed, window_scales, mean_s, std_s


    def evaluate(
        self,
        opt_state: OptimizedVIState,
        keyframes: List[VisualKeyframe],
        imu_samples: List[ImuSample]
    ) -> ObservabilityReport:
        """Full suite of acceptance tests (Section 38)."""
        failures = []

        # 1. Scale stability across windows
        scale_pass, w_scales, mean_s, std_s = self.check_stability_across_windows(
            keyframes, imu_samples, opt_state.gyro_bias
        )
        if not scale_pass and len(w_scales) > 0:
            failures.append(f"Scale instability across temporal windows: mean={mean_s:.3f}, std={std_s:.3f}")

        # 2. Gravity magnitude
        g_norm = float(np.linalg.norm(opt_state.gravity_w))
        grav_pass = abs(g_norm - GRAVITY_NOMINAL) <= self.gravity_tol_mps2
        if not grav_pass:
            failures.append(f"Gravity magnitude ({g_norm:.2f} m/s^2) deviates from nominal ({GRAVITY_NOMINAL:.2f} m/s^2)")

        # 3. Biases
        gb_norm = float(np.linalg.norm(opt_state.gyro_bias))
        ab_norm = float(np.linalg.norm(opt_state.accel_bias))
        bias_pass = (gb_norm <= self.max_gyro_bias) and (ab_norm <= self.max_accel_bias)
        if not bias_pass:
            failures.append(f"Biases exceed physical thresholds: gyro={gb_norm:.4f} rad/s, accel={ab_norm:.4f} m/s^2")

        # 4. Velocities
        speeds = [np.linalg.norm(v) for v in opt_state.velocities]
        max_speed = float(np.max(speeds)) if len(speeds) > 0 else 0.0
        vel_pass = max_speed <= self.max_velocity
        if not vel_pass:
            failures.append(f"Unphysical maximum speed: {max_speed:.2f} m/s")

        passed_all = len(failures) == 0

        return ObservabilityReport(
            passed_all=passed_all,
            scale_stability_passed=scale_pass,
            scale_mean=mean_s,
            scale_std=std_s,
            window_scales=w_scales,
            gravity_norm_passed=grav_pass,
            gravity_norm=g_norm,
            bias_bounded_passed=bias_pass,
            gyro_bias_norm=gb_norm,
            accel_bias_norm=ab_norm,
            velocity_sanity_passed=vel_pass,
            max_speed_mps=max_speed,
            failure_reasons=failures
        )


class DiagnosticPlotter:
    """Generates Section 37 diagnostic figures."""

    @staticmethod
    def plot_scale_stability(window_scales: List[float], output_path: str):
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(window_scales) + 1), window_scales, 'bo-', linewidth=2, markersize=6)
        if len(window_scales) > 0:
            mean_s = np.mean(window_scales)
            plt.axhline(mean_s, color='r', linestyle='--', label=f'Mean: {mean_s:.3f}')
        plt.title('Estimated Metric Scale Across Temporal Sliding Windows')
        plt.xlabel('Window Index')
        plt.ylabel('Metric Scale')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()

    @staticmethod
    def plot_trajectory_comparison(
        unscaled_traj: VisualTrajectory,
        metric_traj: VisualTrajectory,
        output_path: str
    ):
        fig = plt.figure(figsize=(12, 5))

        # Top-down X-Y comparison
        ax1 = fig.add_subplot(1, 2, 1)
        p_unscaled = unscaled_traj.positions
        p_metric = metric_traj.positions

        ax1.plot(p_unscaled[:, 0], p_unscaled[:, 1], 'r.--', label='Unscaled Stella Trajectory')
        ax1.plot(p_metric[:, 0], p_metric[:, 1], 'b.-', label='Metric Gravity-Aligned Trajectory')
        ax1.set_title('Top-Down Trajectory (X-Y)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.axis('equal')
        ax1.grid(True)
        ax1.legend()

        # 3D Trajectory
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot(p_metric[:, 0], p_metric[:, 1], p_metric[:, 2], 'b.-', label='Metric 3D Path (m)')
        ax2.set_title('Metric 3D Trajectory (meters)')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.legend()

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()

    @staticmethod
    def plot_residual_histograms(
        res_rot: np.ndarray,
        res_vel: np.ndarray,
        res_pos: np.ndarray,
        output_path: str
    ):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        if len(res_rot) > 0:
            axes[0].hist(np.degrees(res_rot), bins=15, color='orange', edgecolor='black', alpha=0.7)
            axes[0].set_title('Rotation Residuals (deg)')
            axes[0].set_xlabel('Degrees')
            axes[0].grid(True)

        if len(res_vel) > 0:
            axes[1].hist(res_vel, bins=15, color='green', edgecolor='black', alpha=0.7)
            axes[1].set_title('Velocity Residuals (m/s)')
            axes[1].set_xlabel('m/s')
            axes[1].grid(True)

        if len(res_pos) > 0:
            axes[2].hist(res_pos, bins=15, color='purple', edgecolor='black', alpha=0.7)
            axes[2].set_title('Position Residuals (m)')
            axes[2].set_xlabel('meters')
            axes[2].grid(True)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
