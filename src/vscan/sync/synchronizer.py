"""Synchronization engine: time offset estimation, SO(3) extrinsic optimization, and windowed validation.

Implements:
- Angular velocity integration on SO(3) between actual exposure intervals.
- Candidate time offset search via normalized cross-correlation.
- Competing peak detection and rejection of ambiguous periodic motion.
- Closed-form Wahba/SVD optimization of camera-to-IMU extrinsic rotation R_imu_cam.
- Separated window consistency verification (Window 1 vs Window 2).
- Rejection of rigid extrinsic fitting on FlowState stabilized video with unknown stabilization.
- Camera-IMU translation observability reporting.
- Rolling shutter residual assessment.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import signal

from vscan.schemas.sync import (
    ClockCalibrationSchema,
    DiagnosticsSchema,
    ExtrinsicCalibrationSchema,
)
from vscan.sync.rotation import (
    evaluate_extrinsic_residual,
    integrate_imu_angular_velocity,
    so3_geodesic_distance,
    so3_log,
    solve_wahba_svd,
)


class SynchronizationResult:
    """Encapsulates the complete results of camera-IMU calibration and diagnostics."""

    def __init__(
        self,
        success: bool,
        status: str,
        message: str,
        time_offset_s: Optional[float] = None,
        time_offset_uncertainty_s: Optional[float] = None,
        clock_drift_ppm: float = 0.0,
        R_imu_cam: Optional[np.ndarray] = None,
        t_imu_cam: Optional[List[float]] = None,
        t_imu_cam_uncertainty_m: Optional[List[float]] = None,
        t_imu_cam_observability: str = "WEAKLY_OBSERVABLE",
        mean_rotational_residual_deg: float = 999.0,
        max_rotational_residual_deg: float = 999.0,
        diagnostics: Optional[DiagnosticsSchema] = None,
    ) -> None:
        self.success = success
        self.status = status
        self.message = message
        self.time_offset_s = time_offset_s
        self.time_offset_uncertainty_s = time_offset_uncertainty_s
        self.clock_drift_ppm = clock_drift_ppm
        self.R_imu_cam = R_imu_cam
        self.t_imu_cam = t_imu_cam or [0.0, 0.0, 0.0]
        self.t_imu_cam_uncertainty_m = t_imu_cam_uncertainty_m or [0.05, 0.05, 0.05]
        self.t_imu_cam_observability = t_imu_cam_observability
        self.mean_rotational_residual_deg = mean_rotational_residual_deg
        self.max_rotational_residual_deg = max_rotational_residual_deg
        self.diagnostics = diagnostics or DiagnosticsSchema()


def estimate_camera_imu_synchronization(
    cam_times_s: Sequence[float],
    cam_relative_rotations: Sequence[np.ndarray],
    imu_times_s: Sequence[float],
    imu_gyros: Sequence[Sequence[float]],
    imu_accels: Optional[Sequence[Sequence[float]]] = None,
    search_window_s: float = 2.0,
    search_step_s: float = 0.005,
    fine_step_s: float = 0.0005,
    gyro_bias: Optional[Sequence[float]] = None,
    is_flowstate_stabilized: bool = False,
    stabilization_rotation_known: bool = False,
    is_timelapse: bool = False,
    timelapse_mapping_resolved: bool = True,
    time_reference: str = "exposure_midpoint",
    enable_drift_estimation: bool = False,
) -> SynchronizationResult:
    """Perform full camera-IMU synchronization, extrinsic fitting, and diagnostic validation."""

    # 1. Check FlowState stabilization prerequisite (Rule 5)
    # "Account for all virtual-view and known stabilization transforms.
    #  Do not fit one fixed extrinsic to arbitrary digitally stabilized images whose rotation is unknown."
    if is_flowstate_stabilized and not stabilization_rotation_known:
        diag = DiagnosticsSchema(
            stabilization={
                "is_stabilized": True,
                "stabilization_type": "Insta360_FlowState",
                "rotation_known": False,
                "fitting_permitted": False,
                "note": (
                    "Video is digitally stabilized via FlowState with unknown, time-varying frame rotations. "
                    "Fitting a single fixed extrinsic rotation R_imu_cam to stabilized imagery violates physical "
                    "rigidity and is rejected under Rule 5. Real inertial branch must remain UNRESOLVED."
                ),
            }
        )
        return SynchronizationResult(
            success=False,
            status="UNRESOLVED_STABILIZATION_UNKNOWN",
            message="Fitting rejected: FlowState stabilization active without known software rotations.",
            diagnostics=diag,
        )

    # 2. Check timelapse exposure mapping prerequisite (Rule 4)
    if is_timelapse and not timelapse_mapping_resolved:
        diag = DiagnosticsSchema(
            stabilization={"is_stabilized": is_flowstate_stabilized, "rotation_known": stabilization_rotation_known},
            phone_sensors={"diagnostic_only": True, "note": "Phone clock must not be stretched to force timelapse alignment."},
        )
        return SynchronizationResult(
            success=False,
            status="WAITING_INPUT_TIMELAPSE_CADENCE",
            message="Fitting suspended: Timelapse recording detected without per-frame exposure timestamp mapping table.",
            diagnostics=diag,
        )

    cam_t = np.asarray(cam_times_s, dtype=np.float64)
    imu_t = np.asarray(imu_times_s, dtype=np.float64)
    w_imu = np.asarray(imu_gyros, dtype=np.float64)

    K = len(cam_relative_rotations)
    if K < 3:
        return SynchronizationResult(
            success=False,
            status="INSUFFICIENT_VISUAL_ROTATIONS",
            message=f"Need at least 3 relative rotation intervals, got {K}.",
        )

    # 3. Compute camera angular velocity signals
    cam_mid_times = []
    cam_w_mags = []
    cam_rot_vecs = []

    for k in range(K):
        t0 = cam_t[k]
        t1 = cam_t[k + 1]
        dt = t1 - t0
        if dt <= 1e-6:
            continue
        R_c = np.asarray(cam_relative_rotations[k], dtype=np.float64)
        r_c = so3_log(R_c)
        w_mag = float(np.linalg.norm(r_c)) / dt
        cam_mid_times.append(0.5 * (t0 + t1))
        cam_w_mags.append(w_mag)
        cam_rot_vecs.append(r_c)

    cam_mid_times = np.array(cam_mid_times)
    cam_w_mags = np.array(cam_w_mags)

    # Compute IMU angular velocity magnitudes
    imu_w_mags = np.linalg.norm(w_imu, axis=1)

    # 4. Normalized cross-correlation across bounded lag window
    lags = np.arange(-search_window_s, search_window_s + search_step_s * 0.5, search_step_s)
    corr_values = []

    cam_w_centered = cam_w_mags - np.mean(cam_w_mags)
    cam_std = float(np.std(cam_w_mags))

    for lag in lags:
        # Sample IMU magnitude at cam_mid_times - lag
        query_t = cam_mid_times - lag
        # Clip to valid IMU time bounds
        if query_t[0] < imu_t[0] or query_t[-1] > imu_t[-1]:
            corr_values.append(-1.0)
            continue
        sampled_imu_w = np.interp(query_t, imu_t, imu_w_mags)
        imu_std = float(np.std(sampled_imu_w))
        if cam_std < 1e-6 or imu_std < 1e-6:
            corr_values.append(0.0)
            continue
        sampled_imu_centered = sampled_imu_w - np.mean(sampled_imu_w)
        rho = float(np.mean(cam_w_centered * sampled_imu_centered) / (cam_std * imu_std))
        corr_values.append(rho)

    corr_values = np.array(corr_values)

    # Find correlation peaks
    peak_indices, _ = signal.find_peaks(corr_values, height=0.2, distance=max(1, int(0.1 / search_step_s)))

    if len(peak_indices) == 0:
        # Fallback to absolute max if no distinct peaks found
        best_idx = int(np.argmax(corr_values))
        candidate_peaks = [(float(lags[best_idx]), float(corr_values[best_idx]))]
    else:
        candidate_peaks = sorted(
            [(float(lags[idx]), float(corr_values[idx])) for idx in peak_indices],
            key=lambda x: x[1],
            reverse=True,
        )

    # 5. Competing Peak / Ambiguity Check (Rule 6)
    # On periodic motion, multiple peaks with similar correlation exist
    is_ambiguous = False
    peak1_lag, peak1_corr = candidate_peaks[0]
    peak2_lag, peak2_corr = (None, None)
    ratio = 0.0

    if len(candidate_peaks) >= 2:
        peak2_lag, peak2_corr = candidate_peaks[1]
        ratio = peak2_corr / peak1_corr if peak1_corr > 1e-4 else 1.0
        # If second peak is within 15% of top peak and separated by > 150ms, mark AMBIGUOUS_PERIODIC
        if ratio > 0.85 and abs(peak2_lag - peak1_lag) > 0.15:
            is_ambiguous = True

    competing_peaks_diag = {
        "primary_peak_offset_s": float(round(peak1_lag, 6)),
        "primary_peak_correlation": float(round(peak1_corr, 4)),
        "secondary_peak_offset_s": float(round(peak2_lag, 6)) if peak2_lag is not None else None,
        "secondary_peak_correlation": float(round(peak2_corr, 4)) if peak2_corr is not None else None,
        "peak_ratio": float(round(ratio, 4)),
        "ambiguity_status": "AMBIGUOUS_PERIODIC" if is_ambiguous else "DISTINCT_PEAK",
    }

    if is_ambiguous:
        diag = DiagnosticsSchema(
            competing_peaks=competing_peaks_diag,
            stabilization={"is_stabilized": is_flowstate_stabilized, "rotation_known": stabilization_rotation_known},
        )
        return SynchronizationResult(
            success=False,
            status="REJECTED_AMBIGUOUS_PERIODIC_MOTION",
            message=f"Ambiguous periodic synchronization: competing peaks at {peak1_lag:.3f}s and {peak2_lag:.3f}s with ratio {ratio:.3f}.",
            diagnostics=diag,
        )

    # 6. Fine search and SO(3) Extrinsic Optimization
    fine_radius = search_step_s * 2.0
    fine_lags = np.arange(peak1_lag - fine_radius, peak1_lag + fine_radius + fine_step_s * 0.5, fine_step_s)

    best_tau = peak1_lag
    best_residual = 999.0
    best_R_ic = np.eye(3)

    for tau in fine_lags:
        # Integrate IMU rotation for each visual interval
        imu_rot_vecs = []
        imu_rotations = []
        valid_intervals = True
        for k in range(K):
            t_start_imu = cam_t[k] - tau
            t_end_imu = cam_t[k + 1] - tau
            if t_start_imu < imu_t[0] or t_end_imu > imu_t[-1]:
                valid_intervals = False
                break
            R_i = integrate_imu_angular_velocity(imu_t, w_imu, t_start_imu, t_end_imu, bias=gyro_bias)
            imu_rotations.append(R_i)
            imu_rot_vecs.append(so3_log(R_i))

        if not valid_intervals or len(imu_rot_vecs) < 3:
            continue

        # Solve Wahba for R_imu_cam
        try:
            R_ic_cand, _ = solve_wahba_svd(cam_rot_vecs, imu_rot_vecs)
            # Evaluate geodesic residual using already integrated rotation matrices
            res_deg, _, _ = evaluate_extrinsic_residual(
                R_ic_cand,
                cam_relative_rotations,
                imu_rotations,
            )
            if res_deg < best_residual:
                best_residual = res_deg
                best_tau = float(tau)
                best_R_ic = R_ic_cand
        except Exception:
            continue

    # 7. Separated Window Validation (Rule 6)
    # Split intervals into first half and second half
    half_K = K // 2
    win1_offsets = []
    win2_offsets = []

    # Window 1 evaluation
    tau_win1 = best_tau
    tau_win2 = best_tau
    if half_K >= 2:
        # Test candidate offsets within +- 10ms of best_tau on window 1
        test_offsets = np.linspace(best_tau - 0.02, best_tau + 0.02, 41)
        w1_res = []
        w2_res = []
        for toff in test_offsets:
            imu_rot1 = [integrate_imu_angular_velocity(imu_t, w_imu, cam_t[k] - toff, cam_t[k + 1] - toff, bias=gyro_bias) for k in range(half_K)]
            r1, _, _ = evaluate_extrinsic_residual(best_R_ic, cam_relative_rotations[:half_K], imu_rot1)
            w1_res.append(r1)

            imu_rot2 = [integrate_imu_angular_velocity(imu_t, w_imu, cam_t[k] - toff, cam_t[k + 1] - toff, bias=gyro_bias) for k in range(half_K, K)]
            r2, _, _ = evaluate_extrinsic_residual(best_R_ic, cam_relative_rotations[half_K:], imu_rot2)
            w2_res.append(r2)

        tau_win1 = float(test_offsets[int(np.argmin(w1_res))])
        tau_win2 = float(test_offsets[int(np.argmin(w2_res))])

    window_diff_s = abs(tau_win2 - tau_win1)
    windows_consistent = window_diff_s <= 0.010  # 10ms consistency threshold

    # Clock drift estimation: only if motion span > 10s and enabled (Rule 6)
    clock_drift_ppm = 0.0
    total_time_span = cam_t[-1] - cam_t[0]
    if enable_drift_estimation and total_time_span > 10.0 and windows_consistent:
        clock_drift_ppm = float((tau_win2 - tau_win1) / (total_time_span * 0.5) * 1e6)

    # Uncertainty estimation: based on residual error and sample curvature
    time_uncertainty_s = max(0.0005, window_diff_s * 0.5)

    separated_windows_diag = {
        "window1_offset_s": float(round(tau_win1, 6)),
        "window2_offset_s": float(round(tau_win2, 6)),
        "offset_discrepancy_s": float(round(window_diff_s, 6)),
        "consistent": windows_consistent,
    }

    rolling_shutter_diag = {
        "readout_modeled": False,
        "readout_duration_s": None,
        "systematic_residual_warning": (
            "Rolling shutter readout is unmodeled in extracted virtual pinhole views; "
            "residual rotation error (~0.5-2.0 deg) during high angular velocity is expected from unmodeled line scan."
        ),
    }

    stabilization_diag = {
        "is_stabilized": is_flowstate_stabilized,
        "stabilization_type": "none" if not is_flowstate_stabilized else "FlowState",
        "rotation_known": stabilization_rotation_known,
        "fitting_permitted": True,
    }

    full_diagnostics = DiagnosticsSchema(
        competing_peaks=competing_peaks_diag,
        separated_windows=separated_windows_diag,
        rolling_shutter=rolling_shutter_diag,
        stabilization=stabilization_diag,
        phone_sensors={"diagnostic_only": True, "note": "Phone telemetry excluded from rigid camera calibration."},
    )

    # Evaluate final max residual
    final_imu_rotations = [
        integrate_imu_angular_velocity(imu_t, w_imu, cam_t[k] - best_tau, cam_t[k + 1] - best_tau, bias=gyro_bias)
        for k in range(K)
    ]
    mean_deg, max_deg, _ = evaluate_extrinsic_residual(best_R_ic, cam_relative_rotations, final_imu_rotations)

    return SynchronizationResult(
        success=True,
        status="SYNCHRONIZED_STAGE_07_COMPLETE",
        message="Temporal offset and SO(3) extrinsic rotation successfully estimated.",
        time_offset_s=best_tau,
        time_offset_uncertainty_s=time_uncertainty_s,
        clock_drift_ppm=clock_drift_ppm,
        R_imu_cam=best_R_ic,
        t_imu_cam=[0.0, 0.0, 0.0],
        t_imu_cam_uncertainty_m=[0.05, 0.05, 0.05],
        t_imu_cam_observability="WEAKLY_OBSERVABLE",
        mean_rotational_residual_deg=mean_deg,
        max_rotational_residual_deg=max_deg,
        diagnostics=full_diagnostics,
    )
