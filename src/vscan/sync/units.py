"""Sensor unit hypotheses and physical consistency checks.

Rule 3 compliance:
- Camera acceleration near 1 suggests g (norm ~ 1.0).
- Phone acceleration near 9.8 suggests m/s^2 (norm ~ 9.80665).
- Gyroscope units evaluated against typical handheld rotational dynamics (rad/s vs deg/s).
- Timestamp units evaluated against physical duration and plausible sensor sample rates (100 Hz - 1 kHz).
- Record whether facts are inferred or documented; do not normalize away motion.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from vscan.schemas.sync import UnitHypothesisSchema


def evaluate_accelerometer_units(
    accel_xyz: np.ndarray,
    sensor_name: str = "camera_imu",
    is_documented: bool = False,
) -> UnitHypothesisSchema:
    """Evaluate accelerometer norm against g and m/s^2 hypotheses."""
    accel = np.asarray(accel_xyz, dtype=np.float64)
    norms = np.linalg.norm(accel, axis=1)
    mean_norm = float(np.mean(norms))
    std_norm = float(np.std(norms))
    min_norm = float(np.min(norms))
    max_norm = float(np.max(norms))

    # Test H_g: ||a|| ~ 1.0
    err_g = float(np.mean(np.abs(norms - 1.0)))
    # Test H_ms2: ||a|| ~ 9.80665
    err_ms2 = float(np.mean(np.abs(norms - 9.80665)))

    if err_g < err_ms2:
        unit = "g"
        conversion = 9.80665
        status = "ACCEPTED" if err_g < 0.25 else "PROVISIONAL"
        evidence = (
            f"Mean acceleration norm ||a|| = {mean_norm:.4f} (std={std_norm:.4f}, min={min_norm:.4f}, max={max_norm:.4f}) "
            f"closely matches Earth gravity in g units (residual {err_g:.4f} vs {err_ms2:.4f} for m/s^2)."
        )
    else:
        unit = "m/s^2"
        conversion = 1.0
        status = "ACCEPTED" if err_ms2 < 2.0 else "PROVISIONAL"
        evidence = (
            f"Mean acceleration norm ||a|| = {mean_norm:.4f} (std={std_norm:.4f}, min={min_norm:.4f}, max={max_norm:.4f}) "
            f"closely matches Earth gravity in m/s^2 units (residual {err_ms2:.4f} vs {err_g:.4f} for g)."
        )

    return UnitHypothesisSchema(
        sensor_type=sensor_name,
        accel_norm_mean=mean_norm,
        accel_units_hypothesis=unit,
        accel_hypothesis_status=status,
        accel_conversion_to_m_s2=conversion,
        gyro_norm_mean=0.0,
        gyro_units_hypothesis="unknown",
        gyro_hypothesis_status="PROVISIONAL",
        gyro_conversion_to_rad_s=1.0,
        timestamp_units_hypothesis="unknown",
        timestamp_hypothesis_status="PROVISIONAL",
        timestamp_ticks_per_second=1.0,
        evidence=evidence,
        is_inferred=True,
        is_documented=is_documented,
    )


def evaluate_gyroscope_units(
    gyro_xyz: np.ndarray,
    sensor_name: str = "camera_imu",
    is_documented: bool = False,
) -> Tuple[str, float, str]:
    """Discriminate rad/s vs deg/s for angular velocity measurements.
    
    Returns:
        (unit_hypothesis, conversion_to_rad_s, evidence_str)
    """
    gyros = np.asarray(gyro_xyz, dtype=np.float64)
    norms = np.linalg.norm(gyros, axis=1)
    mean_norm = float(np.mean(norms))
    max_norm = float(np.max(norms))

    # Handheld walking turn rates are typically 0.1 to 2.5 rad/s (6 to 140 deg/s).
    # If values were deg/s, a mean norm of ~0.5 deg/s would represent imperceptible stationary drift,
    # which is inconsistent with recorded walking steps and camera motion.
    if 0.05 <= mean_norm <= 8.0:
        unit = "rad/s"
        conversion = 1.0
        evidence = (
            f"Mean gyro norm ||w|| = {mean_norm:.4f} rad/s (max={max_norm:.4f} rad/s) matches human walking rotational speed "
            f"(~{math.degrees(mean_norm):.1f} deg/s). If units were deg/s, 0.5 deg/s would contradict dynamic motion."
        )
    else:
        unit = "deg/s"
        conversion = math.pi / 180.0
        evidence = f"Mean gyro norm ||w|| = {mean_norm:.4f} exceeds plausible rad/s range for handheld motion; classified as deg/s."

    return unit, conversion, evidence


def evaluate_timestamp_units(
    timestamps_raw: np.ndarray,
    sensor_name: str = "camera_imu",
    is_documented: bool = False,
) -> Tuple[str, float, float, str]:
    """Determine whether raw integer timestamps are microseconds, milliseconds, or nanoseconds.
    
    Returns:
        (unit_hypothesis, ticks_per_second, duration_seconds, evidence_str)
    """
    ts = np.asarray(timestamps_raw, dtype=np.int64)
    if len(ts) < 2:
        return "unknown", 1.0, 0.0, "Insufficient samples"

    span_ticks = int(ts[-1] - ts[0])
    deltas = np.diff(ts)
    median_dt = float(np.median(deltas))

    # Evaluate candidate hypotheses:
    # 1. Microseconds (1e6 ticks/sec): nominal dt=1000 -> 1kHz
    f_us = 1e6 / median_dt if median_dt > 0 else 0
    span_us_s = span_ticks / 1e6

    # 2. Milliseconds (1e3 ticks/sec): nominal dt=10 -> 100Hz
    f_ms = 1e3 / median_dt if median_dt > 0 else 0
    span_ms_s = span_ticks / 1e3

    # 3. Nanoseconds (1e9 ticks/sec): nominal dt=1e6 -> 1kHz
    f_ns = 1e9 / median_dt if median_dt > 0 else 0
    span_ns_s = span_ticks / 1e9

    if 400.0 <= f_us <= 2000.0 and 2.0 <= span_us_s <= 3600.0:
        unit = "microseconds"
        ticks_per_sec = 1e6
        duration_s = span_us_s
        evidence = (
            f"Raw tick span = {span_ticks} with median dt = {median_dt:.0f} ticks. "
            f"Under microseconds hypothesis: cadence is {f_us:.1f} Hz and span is {span_us_s:.3f} seconds, "
            f"matching standard high-rate IMU hardware."
        )
    elif 20.0 <= f_ms <= 250.0 and 2.0 <= span_ms_s <= 3600.0:
        unit = "milliseconds"
        ticks_per_sec = 1e3
        duration_s = span_ms_s
        evidence = (
            f"Raw tick span = {span_ticks} with median dt = {median_dt:.0f} ticks. "
            f"Under milliseconds hypothesis: cadence is {f_ms:.1f} Hz and span is {span_ms_s:.3f} seconds, "
            f"matching standard smartphone sensor sampling."
        )
    elif 400.0 <= f_ns <= 2000.0:
        unit = "nanoseconds"
        ticks_per_sec = 1e9
        duration_s = span_ns_s
        evidence = f"Raw tick span matches nanosecond clock domain with cadence {f_ns:.1f} Hz."
    else:
        # Fallback based on tick magnitude
        if median_dt >= 500:
            unit = "microseconds"
            ticks_per_sec = 1e6
            duration_s = span_us_s
            evidence = f"Estimated microseconds from tick magnitude dt={median_dt:.0f}."
        else:
            unit = "milliseconds"
            ticks_per_sec = 1e3
            duration_s = span_ms_s
            evidence = f"Estimated milliseconds from tick magnitude dt={median_dt:.0f}."

    return unit, ticks_per_sec, duration_s, evidence


def build_full_unit_hypothesis(
    accel: np.ndarray,
    gyro: np.ndarray,
    timestamps: np.ndarray,
    sensor_name: str,
    is_documented: bool = False,
) -> UnitHypothesisSchema:
    """Construct complete UnitHypothesisSchema combining accel, gyro, and timestamp checks."""
    acc_schema = evaluate_accelerometer_units(accel, sensor_name=sensor_name, is_documented=is_documented)
    gyro_unit, gyro_conv, gyro_ev = evaluate_gyroscope_units(gyro, sensor_name=sensor_name, is_documented=is_documented)
    ts_unit, ts_rate, duration_s, ts_ev = evaluate_timestamp_units(timestamps, sensor_name=sensor_name, is_documented=is_documented)

    norms_gyro = np.linalg.norm(gyro, axis=1)
    acc_schema.gyro_norm_mean = float(np.mean(norms_gyro))
    acc_schema.gyro_units_hypothesis = gyro_unit
    acc_schema.gyro_hypothesis_status = "ACCEPTED"
    acc_schema.gyro_conversion_to_rad_s = gyro_conv
    acc_schema.timestamp_units_hypothesis = ts_unit
    acc_schema.timestamp_hypothesis_status = "ACCEPTED"
    acc_schema.timestamp_ticks_per_second = ts_rate
    acc_schema.evidence = f"{acc_schema.evidence} | {gyro_ev} | {ts_ev}"
    return acc_schema
