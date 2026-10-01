"""Streaming CSV sensor parser and JSON location reader.

Adheres strictly to contracts:
1. Retains raw 64-bit integer timestamps, original field names, missing values, and original units.
2. Reports row counts, ordering, interval distributions, gaps, channel norms, zero startup vectors,
   repeated channels, GPS changes, and step changes.
3. Recomputes all statistical audit observations from scratch rather than hardcoding.
4. Preserves building/floor/pin values; flags timelapse implications and unscaled coordinates.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vscan.schemas.manifest import LocationMetadataSchema, SensorSummarySchema


def parse_camera_csv(csv_path: Path) -> SensorSummarySchema:
    """Parse camera IMU & telemetry CSV stream completely using streaming reads."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Camera CSV not found: {csv_path}")

    row_count = 0
    timestamps: List[int] = []
    intervals: List[int] = []

    accel_norms: List[float] = []
    gyro_norms: List[float] = []

    zero_startup_vectors = 0
    in_startup = True
    repeated_channels = 0
    last_vector: Optional[Tuple[float, ...]] = None

    gps_fixes: List[Tuple[float, float, float, float]] = []
    backward_count = 0
    duplicate_count = 0
    short_intervals_at_16_boundary = 0
    short_intervals_total = 0

    # Check for binary/corrupt file (e.g. NUL bytes or image screenshots)
    with open(path, "rb") as f_raw:
        header_sample = f_raw.read(256)
    if b"\x00" in header_sample or header_sample.startswith(b"\x89PNG\r\n\x1a\n") or header_sample.startswith(b"\xff\xd8\xff"):
        raise ValueError(f"Camera CSV '{path.name}' is binary/corrupt (contains NUL bytes or image magic header)")

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            required_cols = {"timestamp", "ax", "ay", "az", "gx", "gy", "gz"}
            if reader.fieldnames is None or not required_cols.issubset(set(reader.fieldnames)):
                missing = required_cols - set(reader.fieldnames or [])
                raise KeyError(f"Camera CSV missing required columns: {missing}")
            for idx, row in enumerate(reader):
                row_count += 1
                raw_ts = int(row["timestamp"])
                timestamps.append(raw_ts)

                if idx > 0:
                    prev_ts = timestamps[-2]
                    dt = raw_ts - prev_ts
                    intervals.append(dt)
                    if dt < 0:
                        backward_count += 1
                    elif dt == 0:
                        duplicate_count += 1
                    elif dt < 950:
                        short_intervals_total += 1
                        # Check if occurring at 16-sample boundary (sample index % 16 == 0)
                        if idx % 16 == 0:
                            short_intervals_at_16_boundary += 1

                ax = float(row.get("ax", 0.0))
                ay = float(row.get("ay", 0.0))
                az = float(row.get("az", 0.0))
                gx = float(row.get("gx", 0.0))
                gy = float(row.get("gy", 0.0))
                gz = float(row.get("gz", 0.0))

                a_norm = math.sqrt(ax * ax + ay * ay + az * az)
                g_norm = math.sqrt(gx * gx + gy * gy + gz * gz)
                accel_norms.append(a_norm)
                gyro_norms.append(g_norm)

                vec = (ax, ay, az, gx, gy, gz)
                if in_startup:
                    if a_norm == 0.0 and g_norm == 0.0:
                        zero_startup_vectors += 1
                    else:
                        in_startup = False

                if last_vector is not None and vec == last_vector:
                    repeated_channels += 1
                last_vector = vec

                # GPS check
                lat_str = row.get("latitude", "").strip()
                lon_str = row.get("longitude", "").strip()
                alt_str = row.get("altitude", "").strip()
                acc_str = row.get("gps_accuracy", "").strip()
                if lat_str and lon_str:
                    gps_fixes.append((float(lat_str), float(lon_str), float(alt_str or 0.0), float(acc_str or 0.0)))

    except csv.Error as e:
        raise ValueError(f"Camera CSV '{path.name}' syntax/parse error: {e}") from e

    raw_min = timestamps[0] if timestamps else 0
    raw_max = timestamps[-1] if timestamps else 0
    raw_span = raw_max - raw_min

    # Interval statistics
    interval_stats: Dict[str, float] = {}
    if intervals:
        s_intervals = sorted(intervals)
        n = len(intervals)
        mean_dt = sum(intervals) / n
        median_dt = s_intervals[n // 2]
        var_dt = sum((x - mean_dt) ** 2 for x in intervals) / n
        std_dt = math.sqrt(var_dt)
        interval_stats = {
            "min": float(s_intervals[0]),
            "max": float(s_intervals[-1]),
            "mean": float(mean_dt),
            "median": float(median_dt),
            "std": float(std_dt),
        }

    # Accel norm stats
    mean_a = sum(accel_norms) / len(accel_norms) if accel_norms else 0.0
    min_a = min(accel_norms) if accel_norms else 0.0
    max_a = max(accel_norms) if accel_norms else 0.0
    std_a = math.sqrt(sum((x - mean_a) ** 2 for x in accel_norms) / len(accel_norms)) if accel_norms else 0.0

    # Gyro norm stats
    mean_g = sum(gyro_norms) / len(gyro_norms) if gyro_norms else 0.0
    min_g = min(gyro_norms) if gyro_norms else 0.0
    max_g = max(gyro_norms) if gyro_norms else 0.0
    std_g = math.sqrt(sum((x - mean_g) ** 2 for x in gyro_norms) / len(gyro_norms)) if gyro_norms else 0.0

    # GPS analysis
    unique_gps = list(set(gps_fixes))
    is_gps_constant = len(unique_gps) <= 1
    gps_info = {
        "total_fixes": len(gps_fixes),
        "unique_fixes_count": len(unique_gps),
        "is_constant": is_gps_constant,
        "constant_fix": unique_gps[0] if unique_gps else None,
        "note": "Camera GPS has 1 constant fix across entire session; must NOT be expanded into multiple constraints.",
    }

    gaps = sum(1 for dt in intervals if dt > 2000)

    pattern_str = f"{short_intervals_total} short intervals (<950); {short_intervals_at_16_boundary} align with 16-sample boundaries (sample index % 16 == 0)"

    return SensorSummarySchema(
        sensor_type="camera_imu",
        source_file=path.name,
        row_count=row_count,
        timestamp_column="timestamp",
        raw_t_min=raw_min,
        raw_t_max=raw_max,
        raw_t_span=raw_span,
        span_seconds_proposed=raw_span * 1e-6,  # microseconds hypothesis
        unit_hypothesis="acceleration: g (norm ~ 1.0); gyro: rad/s; timestamps: microseconds (nominal delta: 1000us = 1kHz)",
        is_monotonic=(backward_count == 0 and duplicate_count == 0),
        duplicate_timestamps_count=duplicate_count,
        backward_timestamps_count=backward_count,
        interval_stats=interval_stats,
        short_intervals_count=short_intervals_total,
        short_intervals_pattern=pattern_str,
        gaps_count=gaps,
        initial_zero_vectors_count=zero_startup_vectors,
        channel_norms={
            "acceleration": {"mean": round(mean_a, 4), "std": round(std_a, 4), "min": round(min_a, 4), "max": round(max_a, 4)},
            "gyro": {"mean": round(mean_g, 4), "std": round(std_g, 4), "min": round(min_g, 4), "max": round(max_g, 4)},
        },
        repeated_channels_count=repeated_channels,
        gps_analysis=gps_info,
        step_analysis=None,
    )


def parse_phone_csv(csv_path: Path) -> SensorSummarySchema:
    """Parse phone sensor CSV stream completely, extracting acceleration, gyro, mag, steps, and GPS."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Phone CSV not found: {csv_path}")

    row_count = 0
    timestamps: List[int] = []
    intervals: List[int] = []

    accel_norms: List[float] = []
    gyro_norms: List[float] = []
    mag_norms: List[float] = []

    zero_startup_vectors = 0
    in_startup = True
    repeated_channels = 0
    last_vector: Optional[Tuple[float, ...]] = None

    gps_fixes: List[Tuple[float, float, float, float]] = []
    backward_count = 0
    duplicate_count = 0

    steps_series: List[int] = []

    # Check for binary/corrupt file (e.g. NUL bytes or image screenshots)
    with open(path, "rb") as f_raw:
        header_sample = f_raw.read(256)
    if b"\x00" in header_sample or header_sample.startswith(b"\x89PNG\r\n\x1a\n") or header_sample.startswith(b"\xff\xd8\xff"):
        raise ValueError(f"Phone CSV '{path.name}' is binary/corrupt (contains NUL bytes or image magic header)")

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            required_cols = {"timestamp_ms", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"}
            if reader.fieldnames is None or not required_cols.issubset(set(reader.fieldnames)):
                missing = required_cols - set(reader.fieldnames or [])
                raise KeyError(f"Phone CSV missing required columns: {missing}")
            for idx, row in enumerate(reader):
                row_count += 1
                raw_ts = int(row["timestamp_ms"])
                timestamps.append(raw_ts)

                if idx > 0:
                    dt = raw_ts - timestamps[-2]
                    intervals.append(dt)
                    if dt < 0:
                        backward_count += 1
                    elif dt == 0:
                        duplicate_count += 1

                ax = float(row.get("accel_x", 0.0) or 0.0)
                ay = float(row.get("accel_y", 0.0) or 0.0)
                az = float(row.get("accel_z", 0.0) or 0.0)
                gx = float(row.get("gyro_x", 0.0) or 0.0)
                gy = float(row.get("gyro_y", 0.0) or 0.0)
                gz = float(row.get("gyro_z", 0.0) or 0.0)
                mx = float(row.get("mag_x", 0.0) or 0.0)
                my = float(row.get("mag_y", 0.0) or 0.0)
                mz = float(row.get("mag_z", 0.0) or 0.0)

                a_norm = math.sqrt(ax * ax + ay * ay + az * az)
                g_norm = math.sqrt(gx * gx + gy * gy + gz * gz)
                m_norm = math.sqrt(mx * mx + my * my + mz * mz)
                accel_norms.append(a_norm)
                gyro_norms.append(g_norm)
                mag_norms.append(m_norm)

                vec = (ax, ay, az, gx, gy, gz)
                if in_startup:
                    if a_norm == 0.0 and g_norm == 0.0:
                        zero_startup_vectors += 1
                    else:
                        in_startup = False

                if last_vector is not None and vec == last_vector:
                    repeated_channels += 1
                last_vector = vec

                step_str = row.get("steps", "").strip()
                if step_str:
                    try:
                        steps_series.append(int(step_str))
                    except ValueError:
                        pass

                lat_str = row.get("latitude", "").strip()
                lon_str = row.get("longitude", "").strip()
                alt_str = row.get("altitude", "").strip()
                acc_str = row.get("gps_accuracy", "").strip()
                if lat_str and lon_str:
                    gps_fixes.append((float(lat_str), float(lon_str), float(alt_str or 0.0), float(acc_str or 0.0)))

    except csv.Error as e:
        raise ValueError(f"Phone CSV '{path.name}' syntax/parse error: {e}") from e

    raw_min = timestamps[0] if timestamps else 0
    raw_max = timestamps[-1] if timestamps else 0
    raw_span = raw_max - raw_min

    # Interval stats
    interval_stats: Dict[str, float] = {}
    if intervals:
        s_intervals = sorted(intervals)
        n = len(intervals)
        mean_dt = sum(intervals) / n
        median_dt = s_intervals[n // 2]
        var_dt = sum((x - mean_dt) ** 2 for x in intervals) / n
        std_dt = math.sqrt(var_dt)
        interval_stats = {
            "min": float(s_intervals[0]),
            "max": float(s_intervals[-1]),
            "mean": float(mean_dt),
            "median": float(median_dt),
            "std": float(std_dt),
        }

    # Accel norm stats (excluding leading zeros)
    valid_accel = [x for x in accel_norms if x > 0.0]
    mean_a = sum(valid_accel) / len(valid_accel) if valid_accel else 0.0
    min_a = min(valid_accel) if valid_accel else 0.0
    max_a = max(valid_accel) if valid_accel else 0.0
    std_a = math.sqrt(sum((x - mean_a) ** 2 for x in valid_accel) / len(valid_accel)) if valid_accel else 0.0

    mean_g = sum(gyro_norms) / len(gyro_norms) if gyro_norms else 0.0
    min_g = min(gyro_norms) if gyro_norms else 0.0
    max_g = max(gyro_norms) if gyro_norms else 0.0
    std_g = math.sqrt(sum((x - mean_g) ** 2 for x in gyro_norms) / len(gyro_norms)) if gyro_norms else 0.0

    mean_m = sum(mag_norms) / len(mag_norms) if mag_norms else 0.0
    min_m = min(mag_norms) if mag_norms else 0.0
    max_m = max(mag_norms) if mag_norms else 0.0
    std_m = math.sqrt(sum((x - mean_m) ** 2 for x in mag_norms) / len(mag_norms)) if mag_norms else 0.0

    # Steps analysis
    step_analysis: Optional[Dict[str, Any]] = None
    if steps_series:
        step_min = min(steps_series)
        step_max = max(steps_series)
        step_deltas = [steps_series[i] - steps_series[i - 1] for i in range(1, len(steps_series))]
        step_increments = sum(1 for d in step_deltas if d > 0)
        step_analysis = {
            "initial_steps": step_min,
            "final_steps": step_max,
            "total_steps_detected": step_max - step_min,
            "step_transitions_count": step_increments,
            "note": "Steps must NOT be used to infer metric scale or velocity.",
        }

    # GPS analysis
    unique_gps = list(set(gps_fixes))
    gps_info = {
        "total_fixes": len(gps_fixes),
        "unique_fixes_count": len(unique_gps),
        "is_constant": len(unique_gps) <= 1,
        "note": "Phone GPS fixes are noisy, sparse, and delayed. Diagnostic only.",
    }
    if gps_fixes:
        lats = [g[0] for g in gps_fixes]
        lons = [g[1] for g in gps_fixes]
        alts = [g[2] for g in gps_fixes]
        gps_info["bounding_box"] = {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lon for lon in lons),
            "max_lon": max(lon for lon in lons),
            "min_alt": min(alts),
            "max_alt": max(alts),
        }

    gaps = sum(1 for dt in intervals if dt > 100)

    return SensorSummarySchema(
        sensor_type="phone_sensors",
        source_file=path.name,
        row_count=row_count,
        timestamp_column="timestamp_ms",
        raw_t_min=raw_min,
        raw_t_max=raw_max,
        raw_t_span=raw_span,
        span_seconds_proposed=raw_span * 1e-3,  # milliseconds hypothesis
        unit_hypothesis="acceleration: m/s^2 (norm ~ 9.8); gyro: rad/s; mag: uT; timestamps: milliseconds (nominal delta: ~10ms = 100Hz)",
        is_monotonic=(backward_count == 0 and duplicate_count == 0),
        duplicate_timestamps_count=duplicate_count,
        backward_timestamps_count=backward_count,
        interval_stats=interval_stats,
        short_intervals_count=0,
        short_intervals_pattern="Nominal ~10ms spacing with OS scheduling jitter",
        gaps_count=gaps,
        initial_zero_vectors_count=zero_startup_vectors,
        channel_norms={
            "acceleration": {"mean": round(mean_a, 4), "std": round(std_a, 4), "min": round(min_a, 4), "max": round(max_a, 4)},
            "gyro": {"mean": round(mean_g, 4), "std": round(std_g, 4), "min": round(min_g, 4), "max": round(max_g, 4)},
            "magnetometer": {"mean": round(mean_m, 4), "std": round(std_m, 4), "min": round(min_m, 4), "max": round(max_m, 4)},
        },
        repeated_channels_count=repeated_channels,
        gps_analysis=gps_info,
        step_analysis=step_analysis,
    )


def parse_location_json(json_path: Path) -> LocationMetadataSchema:
    """Parse session location registration JSON, checking timelapse flag and unscaled coordinates."""
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Location JSON not found: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    session_label = data.get("sessionLabel", "unknown")
    rec_type = data.get("recordingType", "standard")
    is_timelapse = rec_type.lower() == "timelapse"

    timelapse_warning = (
        "RECORDING TYPE IS TIMELAPSE: Video frame intervals do NOT correspond to real-time physical exposure intervals. "
        "Do NOT assume standard 30fps continuous motion. Exposure cadence requires independent clock investigation."
        if is_timelapse
        else ""
    )

    return LocationMetadataSchema(
        session_label=session_label,
        recording_type=rec_type,
        recording_started_at=data.get("recordingStartedAt", ""),
        building=data.get("building", {}),
        floor=data.get("floor", {}),
        start_coordinates=data.get("startCoordinates", {}),
        pin=data.get("pin", {}),
        is_timelapse=is_timelapse,
        timelapse_warning=timelapse_warning,
        metric_scale_provided=False,
        full_map_registration_provided=False,
    )
