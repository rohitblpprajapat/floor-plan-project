"""
IMU CSV parser and data validator.
Supports:
- Insta360 exported IMU CSV (timestamp [us], ax, ay, az [g], gx, gy, gz [rad/s])
- Phone sensor CSV (timestamp_ms [ms], accel_x, accel_y, accel_z [m/s^2], gyro_x, gyro_y, gyro_z [rad/s])

Strictly converts all outputs to standardized SI units:
- Time: float64 seconds
- Accel: float64 m/s^2 (specific force)
- Gyro: float64 rad/s
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from src.inertial.imu_types import ImuSample, ImuBias, GRAVITY_NOMINAL

class ImuDataset:
    """Holds parsed and SI-standardized IMU samples with rich sensor metadata."""
    def __init__(self, samples: List[ImuSample], metadata: Dict):
        self.samples = samples
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([s.timestamp for s in self.samples], dtype=np.float64)

    @property
    def accels(self) -> np.ndarray:
        return np.array([s.accel for s in self.samples], dtype=np.float64)

    @property
    def gyros(self) -> np.ndarray:
        return np.array([s.gyro for s in self.samples], dtype=np.float64)

    def slice_time(self, t_start: float, t_end: float) -> 'ImuDataset':
        """Return a slice of samples within [t_start, t_end]."""
        sub_samples = [s for s in self.samples if t_start <= s.timestamp <= t_end]
        sub_meta = dict(self.metadata)
        sub_meta["t_start"] = t_start
        sub_meta["t_end"] = t_end
        sub_meta["num_samples"] = len(sub_samples)
        return ImuDataset(sub_samples, sub_meta)

    def estimate_static_bias(self, window_sec: float = 2.0, max_gyro_norm: float = 0.05) -> Tuple[Optional[ImuBias], float]:
        """
        Scan dataset for quiescent/static periods to estimate initial gyroscope bias
        and check stationary accelerometer gravity norm.
        """
        ts = self.timestamps
        if len(ts) < 100:
            return None, 0.0

        gyros = self.gyros
        accels = self.accels
        gyro_norms = np.linalg.norm(gyros, axis=1)

        # Sliding window variance of gyro
        dt = np.median(np.diff(ts))
        w_size = int(max(10, window_sec / dt))

        best_var = float('inf')
        best_idx = 0

        for i in range(0, len(ts) - w_size, w_size // 2):
            sub_g_norm = gyro_norms[i : i + w_size]
            sub_var = np.var(sub_g_norm)
            sub_mean = np.mean(sub_g_norm)
            if sub_mean < max_gyro_norm and sub_var < best_var:
                best_var = sub_var
                best_idx = i

        if best_var < float('inf'):
            static_gyro = gyros[best_idx : best_idx + w_size]
            static_accel = accels[best_idx : best_idx + w_size]
            b_g = np.mean(static_gyro, axis=0)
            a_mean = np.mean(static_accel, axis=0)
            accel_norm = np.linalg.norm(a_mean)
            return ImuBias(gyro_bias=b_g, accel_bias=np.zeros(3)), accel_norm
        else:
            # Fallback: whole-dataset minimum motion segment
            sorted_indices = np.argsort(gyro_norms)
            top_quiet = sorted_indices[: min(500, len(sorted_indices))]
            b_g = np.mean(gyros[top_quiet], axis=0)
            accel_norm = np.linalg.norm(np.mean(accels[top_quiet], axis=0))
            return ImuBias(gyro_bias=b_g, accel_bias=np.zeros(3)), accel_norm


def parse_insta360_csv(csv_path: str, normalize_t0: bool = False) -> ImuDataset:
    """
    Parses Insta360 export CSV file.
    Expected columns: timestamp, ax, ay, az, gx, gy, gz
    - timestamp: in microseconds (us)
    - ax, ay, az: in g units (1 g = 9.80665 m/s^2)
    - gx, gy, gz: in rad/s
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"IMU file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}. Found: {df.columns.tolist()}")

    # Sort by timestamp to enforce monotonicity
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Filter out exact duplicate timestamps
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

    raw_t_us = df['timestamp'].values.astype(np.float64)
    t_sec = raw_t_us * 1e-6

    t0_abs = t_sec[0]
    if normalize_t0:
        t_sec = t_sec - t0_abs

    # Convert acceleration from g to m/s^2
    accel_si = df[['ax', 'ay', 'az']].values.astype(np.float64) * GRAVITY_NOMINAL
    gyro_si = df[['gx', 'gy', 'gz']].values.astype(np.float64)

    dt_series = np.diff(t_sec)
    metadata = {
        "source": "insta360_export",
        "csv_path": os.path.abspath(csv_path),
        "num_samples": len(df),
        "t0_raw_us": raw_t_us[0],
        "t_start_sec": t_sec[0],
        "t_end_sec": t_sec[-1],
        "duration_sec": t_sec[-1] - t_sec[0],
        "dt_mean": float(np.mean(dt_series)) if len(dt_series) > 0 else 0.0,
        "dt_median": float(np.median(dt_series)) if len(dt_series) > 0 else 0.0,
        "dt_std": float(np.std(dt_series)) if len(dt_series) > 0 else 0.0,
        "approx_rate_hz": float(1.0 / np.median(dt_series)) if len(dt_series) > 0 else 0.0,
        "accel_units": "m/s^2",
        "gyro_units": "rad/s"
    }

    samples = [
        ImuSample(timestamp=t_sec[i], accel=accel_si[i], gyro=gyro_si[i])
        for i in range(len(df))
    ]

    return ImuDataset(samples, metadata)


def parse_phone_sensor_csv(csv_path: str, normalize_t0: bool = False) -> ImuDataset:
    """
    Parses mobile phone sensor CSV.
    Expected columns: timestamp_ms, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    - timestamp_ms: epoch milliseconds
    - accel_x, accel_y, accel_z: in m/s^2
    - gyro_x, gyro_y, gyro_z: in rad/s
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Phone sensor file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.sort_values(by='timestamp_ms').reset_index(drop=True)
    df = df.drop_duplicates(subset=['timestamp_ms']).reset_index(drop=True)

    t_sec = df['timestamp_ms'].values.astype(np.float64) * 1e-3
    t0_abs = t_sec[0]
    if normalize_t0:
        t_sec = t_sec - t0_abs

    accel_si = df[['accel_x', 'accel_y', 'accel_z']].values.astype(np.float64)
    gyro_si = df[['gyro_x', 'gyro_y', 'gyro_z']].values.astype(np.float64)

    dt_series = np.diff(t_sec)
    metadata = {
        "source": "phone_sensors",
        "csv_path": os.path.abspath(csv_path),
        "num_samples": len(df),
        "t_start_sec": t_sec[0],
        "t_end_sec": t_sec[-1],
        "duration_sec": t_sec[-1] - t_sec[0],
        "dt_mean": float(np.mean(dt_series)) if len(dt_series) > 0 else 0.0,
        "dt_median": float(np.median(dt_series)) if len(dt_series) > 0 else 0.0,
        "approx_rate_hz": float(1.0 / np.median(dt_series)) if len(dt_series) > 0 else 0.0,
        "accel_units": "m/s^2",
        "gyro_units": "rad/s"
    }

    samples = [
        ImuSample(timestamp=t_sec[i], accel=accel_si[i], gyro=gyro_si[i])
        for i in range(len(df))
    ]

    return ImuDataset(samples, metadata)

