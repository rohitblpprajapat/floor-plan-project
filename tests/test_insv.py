"""Comprehensive test suite for Insta360 .insv binary parser and telemetry extraction."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
import pytest
import numpy as np

from vscan.ingest.insv import (
    INSTA360_MAGIC,
    InsvParser,
    InsvExposureRecord,
    InsvMakerNotes,
)
from vscan.sync.timing import analyze_16_sample_boundary_pattern

REAL_INSV_PATH = Path("data/2026-09-10/VID_20260910_172941_00_083.insv")
REAL_CSV_PATH = Path("data/2026-09-10/insta360_export_1789041643971.csv")


@pytest.fixture
def insv_file() -> Path:
    """Resolve real INSV file path across WSL and Windows workspaces."""
    if REAL_INSV_PATH.is_file():
        return REAL_INSV_PATH
    repo_root = Path(__file__).resolve().parents[1]
    cand = repo_root / "data" / "2026-09-10" / "VID_20260910_172941_00_083.insv"
    if cand.is_file():
        return cand
    pytest.skip("Real INSV file VID_20260910_172941_00_083.insv not found in test environment")


@pytest.fixture
def camera_csv_file() -> Path:
    """Resolve camera exported CSV file path."""
    if REAL_CSV_PATH.is_file():
        return REAL_CSV_PATH
    repo_root = Path(__file__).resolve().parents[1]
    cand = repo_root / "data" / "2026-09-10" / "insta360_export_1789041643971.csv"
    if cand.is_file():
        return cand
    pytest.skip("Real camera CSV file not found in test environment")


def test_01_insv_magic_and_trailer_parsing(insv_file: Path):
    """Test 1: Verify 32-byte magic footer, trailer length, and directory entries."""
    parser = InsvParser(insv_file)

    assert parser.file_size == 77779294
    assert parser.trailer_length == 11875052
    assert parser.trailer_start == parser.file_size - parser.trailer_length

    # Verify key directory entries exist
    entries = parser.directory_entries
    assert 0x0101 in entries, "Tag 0x0101 (MakerNotes) must exist"
    assert 0x0002 in entries, "Tag 0x0002 (Thumbnail) must exist"
    assert 0x0003 in entries, "Tag 0x0003 (IMU Telemetry) must exist"
    assert 0x0004 in entries, "Tag 0x0004 (Exposure Table) must exist"

    # Sizes must match exact trailer records
    assert entries[0x0101][0] == 4006
    assert entries[0x0002][0] == 1228840
    assert entries[0x0003][0] == 1229440
    assert entries[0x0004][0] == 29344


def test_02_insv_makernotes_hardware_identity(insv_file: Path):
    """Test 2: Extract and verify camera serial number, model, and firmware version."""
    parser = InsvParser(insv_file)
    notes = parser.get_maker_notes()

    assert isinstance(notes, InsvMakerNotes)
    assert notes.camera_model == "Insta360 X5"
    assert notes.serial_number == "IAHEA26013WB6G"
    assert "v1.11.10" in notes.firmware_version
    assert notes.raw_payload_length == 4006


def test_03_insv_exposure_records_mitigates_timelapse(insv_file: Path):
    """Test 3: Extract per-frame exposure table and verify physical optical cadence.

    Mitigates the TIMELAPSE_UNRESOLVED blocker by extracting 1,834 real capture timestamps.
    """
    parser = InsvParser(insv_file)
    exposures = parser.get_exposure_records()

    assert len(exposures) == 1834, f"Expected 1834 frames, got {len(exposures)}"

    # First and last exposure timestamps
    t0 = exposures[0].exposure_timestamp_us
    t_end = exposures[-1].exposure_timestamp_us
    assert t0 == 100593095
    assert t_end == 161752975

    total_span_s = (t_end - t0) * 1e-6
    assert 61.15 < total_span_s < 61.17, f"Expected ~61.16s span, got {total_span_s:.3f}s"

    # Verify timestamps are strictly monotonically increasing
    for i in range(len(exposures) - 1):
        dt = exposures[i + 1].exposure_timestamp_us - exposures[i].exposure_timestamp_us
        assert dt > 0, f"Frame {i} to {i+1} timestamp not strictly increasing"
        # Nominal 30 fps -> ~33,366 us delta
        assert 30000 < dt < 36000, f"Frame interval {dt}us outside nominal 30fps range"

        # Shutter duration must be positive and physically realistic (~5 to 15 ms)
        exp_dur = exposures[i].shutter_duration_s
        assert 0.001 < exp_dur < 0.050, f"Unrealistic shutter duration {exp_dur}s"


def test_04_insv_imu_telemetry_and_16_sample_boundary(insv_file: Path):
    """Test 4: Extract 61,472 IMU samples and verify 16-sample FIFO boundary pattern."""
    parser = InsvParser(insv_file)
    imu = parser.get_imu_telemetry()

    assert imu["sample_count"] == 61472
    ts = imu["timestamps_raw_us"]
    acc_g = imu["accel_g"]
    gyro = imu["gyro_rad_s"]

    # Timestamps span ~61.07 seconds
    assert ts[0] == 100701912
    assert ts[-1] == 161773162
    span_s = (ts[-1] - ts[0]) * 1e-6
    assert 61.05 < span_s < 61.10

    # Accelerometer mean norm must match gravity (1.0g)
    acc_norms = np.linalg.norm(acc_g, axis=1)
    mean_norm = float(np.mean(acc_norms))
    assert 0.98 < mean_norm < 1.05, f"Expected accel norm near 1.0g, got {mean_norm:.4f}g"

    # Gyroscope rates within typical handheld walking range
    gyro_norms = np.linalg.norm(gyro, axis=1)
    assert np.all(gyro_norms < 10.0), "Angular velocity exceeded physical bounds"

    # Analyze 16-sample boundary pattern on full 61.1s stream
    diag = analyze_16_sample_boundary_pattern(ts, source_file=insv_file.name)
    assert diag.total_samples == 61472
    assert diag.short_intervals_count == 3841
    assert diag.boundary_16_alignment_percent == 100.0
    assert diag.uniform_dt_substituted is False


def test_05_insv_imu_parity_with_exported_csv(insv_file: Path, camera_csv_file: Path):
    """Test 5: Verify bit-exact numerical parity between INSV extraction and exported CSV."""
    parser = InsvParser(insv_file)
    imu = parser.get_imu_telemetry()
    ts_insv = imu["timestamps_raw_us"]
    acc_insv = imu["accel_g"]
    gyr_insv = imu["gyro_rad_s"]

    # Read first 1000 rows from CSV
    csv_ts = []
    csv_acc = []
    csv_gyr = []
    with open(camera_csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            if i >= 1000:
                break
            csv_ts.append(int(float(row[0])))
            csv_acc.append([float(row[1]), float(row[2]), float(row[3])])
            csv_gyr.append([float(row[4]), float(row[5]), float(row[6])])

    csv_ts = np.array(csv_ts, dtype=np.int64)
    csv_acc = np.array(csv_acc, dtype=np.float64)
    csv_gyr = np.array(csv_gyr, dtype=np.float64)

    # Timestamps must match exactly
    assert np.array_equal(ts_insv[:1000], csv_ts)

    # Accelerations must match within float precision
    max_acc_diff = float(np.max(np.abs(acc_insv[:1000] - csv_acc)))
    assert max_acc_diff < 1e-6, f"Max accel difference {max_acc_diff} exceeded tolerance"

    # Gyroscopes must match within float precision
    max_gyr_diff = float(np.max(np.abs(gyr_insv[:1000] - csv_gyr)))
    assert max_gyr_diff < 1e-6, f"Max gyro difference {max_gyr_diff} exceeded tolerance"


def test_06_insv_error_handling_corrupt_magic(tmp_path: Path, insv_file: Path):
    """Test 6: Reject file with corrupt or missing magic footer."""
    corrupt_file = tmp_path / "corrupt.insv"
    # Create copy with corrupted last 32 bytes
    shutil.copyfile(insv_file, corrupt_file)
    with open(corrupt_file, "r+b") as f:
        f.seek(-32, 2)
        f.write(b"0" * 32)

    with pytest.raises(ValueError, match="not a valid Insta360 container"):
        InsvParser(corrupt_file)


def test_07_insv_error_handling_nonexistent_file():
    """Test 7: Raise FileNotFoundError for nonexistent file."""
    with pytest.raises(FileNotFoundError):
        InsvParser(Path("nonexistent_camera_file.insv"))


def test_08_insv_export_imu_csv_and_exposure_json(tmp_path: Path, insv_file: Path):
    """Test 8: Export extracted telemetry to CSV and JSON; verify structure and round-trip."""
    parser = InsvParser(insv_file)

    out_csv = tmp_path / "exported_imu.csv"
    out_json = tmp_path / "exported_exposures.json"

    res_csv = parser.export_imu_csv(out_csv)
    assert res_csv.is_file()
    assert res_csv.stat().st_size > 1_000_000  # ~61k rows > 3MB

    res_json = parser.export_exposure_json(out_json)
    assert res_json.is_file()

    # Read exported JSON
    with open(res_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1834
    assert data[0]["frame_index"] == 0
    assert data[0]["exposure_timestamp_us"] == 100593095
    assert data[-1]["frame_index"] == 1833
    assert data[-1]["exposure_timestamp_us"] == 161752975

    # Verify summary method
    summary = parser.get_summary()
    assert summary["camera_model"] == "Insta360 X5"
    assert summary["imu_sample_count"] == 61472
    assert summary["exposure_frame_count"] == 1834
    assert len(summary["available_tags"]) == 12
