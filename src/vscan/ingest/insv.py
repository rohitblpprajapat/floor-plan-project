"""Pure-Python parser for Insta360 .insv containers and proprietary binary trailers.

Extracts:
1. High-rate IMU telemetry (accelerometer in g, gyroscope in rad/s, timestamps in microseconds).
2. Per-frame physical optical exposure timestamps and shutter durations.
3. Camera identity and firmware maker notes (serial number, model, firmware revision).

No external dependencies outside standard library and numpy.
"""

from __future__ import annotations

import csv
import json
import math
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

INSTA360_MAGIC: bytes = b"8db42d694ccc418790edff439fe026bf"
GYRO_SCALE_RAD_S: float = math.pi / (180.0 * 16.384)
ACCEL_SCALE_G: float = 1.0 / 1024.0
GRAVITY_STANDARD: float = 9.80665


@dataclass(frozen=True)
class InsvExposureRecord:
    """Per-frame physical exposure record from INSV Tag 0x0004."""

    frame_index: int
    exposure_timestamp_us: int
    exposure_timestamp_s: float
    shutter_duration_s: float


@dataclass(frozen=True)
class InsvMakerNotes:
    """Hardware and firmware identity from INSV Tag 0x0101."""

    serial_number: str
    camera_model: str
    firmware_version: str
    raw_payload_length: int


class InsvParser:
    """Binary parser for Insta360 .insv raw capture container trailer."""

    def __init__(self, file_path: Path | str) -> None:
        self.file_path = Path(file_path).resolve()
        if not self.file_path.is_file():
            raise FileNotFoundError(f"INSV file not found: {self.file_path}")

        self.file_size = self.file_path.stat().st_size
        self.trailer_length: int = 0
        self.trailer_start: int = 0
        self.directory_entries: Dict[int, Tuple[int, int, int]] = {}  # tag -> (size, trailer_offset, file_offset)
        self._maker_notes: Optional[InsvMakerNotes] = None

        self._parse_trailer_directory()

    def _parse_trailer_directory(self) -> None:
        """Verify magic number and parse the 10-byte entry directory table."""
        with open(self.file_path, "rb") as f:
            # 1. Verify 32-byte magic footer at end of file
            f.seek(-32, os.SEEK_END)
            magic = f.read(32)
            if magic != INSTA360_MAGIC:
                raise ValueError(
                    f"File {self.file_path.name} is not a valid Insta360 container: "
                    f"magic mismatch (expected {INSTA360_MAGIC.decode('ascii')}, got {magic[:16]!r}...)"
                )

            # 2. Read trailer length from EOF - 78
            f.seek(-78, os.SEEK_END)
            buf = f.read(42)
            self.trailer_length = struct.unpack("<38xI", buf)[0]
            if self.trailer_length <= 0 or self.trailer_length > self.file_size:
                raise ValueError(f"Invalid trailer length: {self.trailer_length} bytes (file size {self.file_size})")

            self.trailer_start = self.file_size - self.trailer_length

            # 3. Read directory table at EOF - 78 - 310
            f.seek(-78 - 310, os.SEEK_END)
            dir_data = f.read(310)

            # Directory format: 31 entries of 10 bytes: <H (tag), <I (size), <I (offset rel to trailer_start)
            for i in range(31):
                chunk = dir_data[i * 10 : (i + 1) * 10]
                tag, size, rel_off = struct.unpack("<HII", chunk)
                if size > 0:
                    file_off = self.trailer_start + rel_off
                    self.directory_entries[tag] = (size, rel_off, file_off)

    def get_maker_notes(self) -> InsvMakerNotes:
        """Parse Tag 0x0101 for camera serial, model, and firmware version."""
        if self._maker_notes is not None:
            return self._maker_notes

        if 0x0101 not in self.directory_entries:
            raise KeyError("Tag 0x0101 (MakerNotes) not found in INSV trailer")

        size, _, file_off = self.directory_entries[0x0101]
        with open(self.file_path, "rb") as f:
            f.seek(file_off)
            payload = f.read(size)

        # Parse protobuf/TLV-like strings
        serial = ""
        model = ""
        firmware = ""

        # Search for ASCII runs or protobuf fields
        # Bytes structure in X3/X4/X5: \n <len> SERIAL \x12 <len> MODEL \x1a <len> FIRMWARE
        try:
            pos = 0
            if pos < len(payload) and payload[pos] == 0x0A:
                s_len = payload[pos + 1]
                serial = payload[pos + 2 : pos + 2 + s_len].decode("ascii", errors="replace")
                pos = pos + 2 + s_len
            if pos < len(payload) and payload[pos] == 0x12:
                m_len = payload[pos + 1]
                model = payload[pos + 2 : pos + 2 + m_len].decode("ascii", errors="replace")
                pos = pos + 2 + m_len
            if pos < len(payload) and payload[pos] == 0x1A:
                f_len = payload[pos + 1]
                firmware = payload[pos + 2 : pos + 2 + f_len].decode("ascii", errors="replace")
        except Exception:
            pass

        self._maker_notes = InsvMakerNotes(
            serial_number=serial,
            camera_model=model,
            firmware_version=firmware,
            raw_payload_length=size,
        )
        return self._maker_notes

    def get_exposure_records(self) -> List[InsvExposureRecord]:
        """Extract per-frame exposure table (Tag 0x0004).

        Each record is 16 bytes:
          timestamp_us: uint64 (<Q)
          shutter_duration_s: float64 (<d)
        """
        if 0x0004 not in self.directory_entries:
            return []

        size, _, file_off = self.directory_entries[0x0004]
        num_records = size // 16
        records: List[InsvExposureRecord] = []

        with open(self.file_path, "rb") as f:
            f.seek(file_off)
            raw_bytes = f.read(size)

        for i in range(num_records):
            chunk = raw_bytes[i * 16 : (i + 1) * 16]
            t_us, shutter_s = struct.unpack("<Qd", chunk)
            records.append(
                InsvExposureRecord(
                    frame_index=i,
                    exposure_timestamp_us=t_us,
                    exposure_timestamp_s=t_us * 1e-6,
                    shutter_duration_s=shutter_s,
                )
            )

        return records

    def get_imu_telemetry(self) -> Dict[str, np.ndarray]:
        """Extract high-rate IMU telemetry (Tag 0x0003).

        Each sample is 20 bytes:
          timestamp_us: uint64 (<Q)
          u_ax, u_ay, u_az: uint16 (<3H)
          u_gx, u_gy, u_gz: uint16 (<3H)

        Conversions:
          ax, ay, az = (uint16 - 32768) / 1024.0 in g
          gx, gy, gz = (uint16 - 32768) * (pi / (180.0 * 16.384)) in rad/s
        """
        if 0x0003 not in self.directory_entries:
            raise KeyError("Tag 0x0003 (IMU telemetry) not present in INSV trailer")

        size, _, file_off = self.directory_entries[0x0003]
        num_samples = size // 20

        with open(self.file_path, "rb") as f:
            f.seek(file_off)
            raw_bytes = f.read(size)

        timestamps_us = np.empty(num_samples, dtype=np.int64)
        accel_g = np.empty((num_samples, 3), dtype=np.float64)
        gyro_rad_s = np.empty((num_samples, 3), dtype=np.float64)

        for i in range(num_samples):
            chunk = raw_bytes[i * 20 : (i + 1) * 20]
            t, u_ax, u_ay, u_az, u_gx, u_gy, u_gz = struct.unpack("<Q6H", chunk)
            timestamps_us[i] = t
            accel_g[i, 0] = (u_ax - 32768) * ACCEL_SCALE_G
            accel_g[i, 1] = (u_ay - 32768) * ACCEL_SCALE_G
            accel_g[i, 2] = (u_az - 32768) * ACCEL_SCALE_G
            gyro_rad_s[i, 0] = (u_gx - 32768) * GYRO_SCALE_RAD_S
            gyro_rad_s[i, 1] = (u_gy - 32768) * GYRO_SCALE_RAD_S
            gyro_rad_s[i, 2] = (u_gz - 32768) * GYRO_SCALE_RAD_S

        accel_m_s2 = accel_g * GRAVITY_STANDARD

        return {
            "timestamps_raw_us": timestamps_us,
            "timestamps_s": timestamps_us * 1e-6,
            "accel_g": accel_g,
            "accel_m_s2": accel_m_s2,
            "gyro_rad_s": gyro_rad_s,
            "sample_count": num_samples,
        }

    def export_imu_csv(self, output_csv_path: Path | str) -> Path:
        """Export extracted IMU telemetry as CSV matching camera CSV format."""
        out_path = Path(output_csv_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        telemetry = self.get_imu_telemetry()
        ts = telemetry["timestamps_raw_us"]
        acc = telemetry["accel_g"]
        gyr = telemetry["gyro_rad_s"]
        n = len(ts)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "latitude", "longitude", "altitude", "gps_accuracy"])
            for i in range(n):
                writer.writerow([
                    int(ts[i]),
                    f"{acc[i, 0]:.8f}",
                    f"{acc[i, 1]:.8f}",
                    f"{acc[i, 2]:.8f}",
                    f"{gyr[i, 0]:.16f}",
                    f"{gyr[i, 1]:.16f}",
                    f"{gyr[i, 2]:.16f}",
                    "", "", "", "",
                ])

        return out_path

    def export_exposure_json(self, output_json_path: Path | str) -> Path:
        """Export per-frame exposure table as JSON."""
        out_path = Path(output_json_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        records = self.get_exposure_records()
        data = [asdict(r) for r in records]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return out_path

    def get_summary(self) -> Dict[str, Any]:
        """Return diagnostic summary of extracted trailer data."""
        notes = self.get_maker_notes()
        exposures = self.get_exposure_records()
        imu = self.get_imu_telemetry()

        exp_span_s = (exposures[-1].exposure_timestamp_s - exposures[0].exposure_timestamp_s) if exposures else 0.0
        imu_span_s = (float(imu["timestamps_raw_us"][-1] - imu["timestamps_raw_us"][0])) * 1e-6 if imu["sample_count"] > 0 else 0.0

        return {
            "file_name": self.file_path.name,
            "file_size_bytes": self.file_size,
            "trailer_length_bytes": self.trailer_length,
            "camera_model": notes.camera_model,
            "serial_number": notes.serial_number,
            "firmware_version": notes.firmware_version,
            "imu_sample_count": imu["sample_count"],
            "imu_span_seconds": round(imu_span_s, 4),
            "imu_first_timestamp_us": int(imu["timestamps_raw_us"][0]) if imu["sample_count"] > 0 else 0,
            "imu_last_timestamp_us": int(imu["timestamps_raw_us"][-1]) if imu["sample_count"] > 0 else 0,
            "exposure_frame_count": len(exposures),
            "exposure_span_seconds": round(exp_span_s, 4),
            "exposure_first_timestamp_us": exposures[0].exposure_timestamp_us if exposures else 0,
            "exposure_last_timestamp_us": exposures[-1].exposure_timestamp_us if exposures else 0,
            "available_tags": [f"0x{tag:04x}" for tag in sorted(self.directory_entries.keys())],
        }
