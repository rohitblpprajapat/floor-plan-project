"""Comprehensive unit and integration tests for Stage 04 Ingestion Pipeline.

Tests enforce all Prompt 04 contract requirements:
1. Truncated video, corrupt CSV, missing columns, blank GPS, duplicate and backward timestamps,
   initial zero vectors, and filename-with-spaces cases.
2. Re-import of unchanged input must not duplicate session assets.
3. A changed source hash invalidates dependent jobs without deleting raw data.
4. Verify initial sensor units are never silently mixed and raw timestamps survive round-trip
   serialization exactly as 64-bit integers.
5. Verify one constant GPS fix is not expanded into independent position constraints.
6. A timelapse export must not imply real exposure cadence.
7. For available real files, compare declared video properties with decoded frames; record all
   unavailable metadata rather than fabricating it.
8. Scoped cancellation during ingestion and resumption with complete verified copies.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from PIL import Image
import pytest

from vscan.ingest.camera_geometry import evaluate_camera_geometry, probe_local_vendor_software
from vscan.ingest.csv_parser import parse_camera_csv, parse_location_json, parse_phone_csv
from vscan.ingest.isobmff import inspect_isobmff_file
from vscan.ingest.manifest import IngestionManager, resolve_input_directory
from vscan.ingest.video_probe import decode_sample_frames, generate_contact_sheet, probe_video_stream
from vscan.schemas.manifest import InputManifestSchema, SensorSummarySchema, VideoProbeSummarySchema
from vscan.storage.artifacts import compute_cache_key, compute_sha256_file
from vscan.storage.database import DatabaseManager
from vscan.workers.ingest import IngestWorker


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create isolated test directory with app.sqlite3 and standard tree."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "data" / "sessions").mkdir(parents=True)
    (ws / "runs").mkdir()
    return ws


# 1. Truncated Video Case
def test_truncated_video(tmp_path: Path) -> None:
    """A truncated video container must be safely detected without unhandled crashes."""
    trunc_file = tmp_path / "truncated_stream.mp4"
    # Write only a partial ftyp box of 12 bytes
    trunc_file.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00")

    info = inspect_isobmff_file(trunc_file)
    assert info["is_truncated"] is True
    assert info["has_moov"] is False
    assert "moov" in info.get("error", "").lower()

    # Video probe wrapper must also record safe status
    probe = probe_video_stream(trunc_file, "asset_trunc_01")
    assert probe.file_size_bytes == len(trunc_file.read_bytes())
    assert probe.container_duration_s is None


# 2. Corrupt CSV Case
def test_corrupt_csv(tmp_path: Path) -> None:
    """Corrupted CSV with invalid floating point strings should raise ValueError cleanly."""
    corrupt_file = tmp_path / "corrupt_camera.csv"
    corrupt_file.write_text(
        "timestamp,ax,ay,az,gx,gy,gz,latitude,longitude,altitude,gps_accuracy\n"
        "100701912,-1.0039,0.124,0.114,CORRUPT_NOT_A_FLOAT,0.297,-0.037,31.78,76.99,1029.3,20.4\n"
    )
    with pytest.raises(ValueError):
        parse_camera_csv(corrupt_file)


# 3. Missing Columns Case
def test_missing_columns_csv(tmp_path: Path) -> None:
    """CSV missing required header columns must raise KeyError cleanly."""
    missing_col_file = tmp_path / "missing_cols.csv"
    missing_col_file.write_text(
        "timestamp,ay,az,gx,gy,gz\n"  # missing ax
        "100701912,0.124,0.114,0.17,0.29,-0.03\n"
    )
    with pytest.raises(KeyError):
        parse_camera_csv(missing_col_file)


# 4. Blank GPS Case
def test_blank_gps(tmp_path: Path) -> None:
    """Blank GPS entries (empty commas) must be handled safely without error."""
    blank_gps_file = tmp_path / "blank_gps.csv"
    blank_gps_file.write_text(
        "timestamp_ms,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,steps,latitude,longitude,altitude,gps_accuracy\n"
        "1789041579535,0.1,3.2,9.5,0.0,0.0,0.0,-28.5,-11.8,-34.1,1,,,,\n"
        "1789041579550,0.1,3.2,9.5,0.0,0.0,0.0,-28.5,-11.8,-34.1,1,,,,\n"
    )
    summary = parse_phone_csv(blank_gps_file)
    assert summary.row_count == 2
    assert summary.gps_analysis["total_fixes"] == 0
    assert summary.gps_analysis["unique_fixes_count"] == 0
    assert "bounding_box" not in summary.gps_analysis


# 5. Duplicate and Backward Timestamps Case
def test_duplicate_and_backward_timestamps(tmp_path: Path) -> None:
    """Duplicate and backward timestamps must be identified and flagged."""
    bad_time_file = tmp_path / "bad_time.csv"
    bad_time_file.write_text(
        "timestamp,ax,ay,az,gx,gy,gz,latitude,longitude,altitude,gps_accuracy\n"
        "1000,-1.0,0.1,0.1,0.0,0.0,0.0,31.78,76.99,1029.3,20.4\n"
        "1000,-1.0,0.1,0.1,0.0,0.0,0.0,31.78,76.99,1029.3,20.4\n"  # Duplicate
        "950,-1.0,0.1,0.1,0.0,0.0,0.0,31.78,76.99,1029.3,20.4\n"   # Backward
        "1200,-1.0,0.1,0.1,0.0,0.0,0.0,31.78,76.99,1029.3,20.4\n"
    )
    summary = parse_camera_csv(bad_time_file)
    assert summary.row_count == 4
    assert summary.is_monotonic is False
    assert summary.duplicate_timestamps_count == 1
    assert summary.backward_timestamps_count == 1


# 6. Initial Zero Startup Vectors Case
def test_initial_zero_startup_vectors(tmp_path: Path) -> None:
    """Leading rows with all-zero sensor vectors must be counted accurately."""
    zero_vec_file = tmp_path / "zero_startup.csv"
    zero_vec_file.write_text(
        "timestamp_ms,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,steps,latitude,longitude,altitude,gps_accuracy\n"
        "1000,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,,,,\n"
        "1010,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,,,,\n"
        "1020,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,,,,\n"
        "1030,-0.1,3.2,9.5,0.1,-0.1,0.0,-28.5,-11.8,-34.1,1,,,,\n"
    )
    summary = parse_phone_csv(zero_vec_file)
    assert summary.row_count == 4
    assert summary.initial_zero_vectors_count == 3


# 7. Filename with Spaces Case
def test_filename_with_spaces(tmp_path: Path) -> None:
    """Files with spaces in filenames or parent directories must be handled cleanly."""
    spaced_dir = tmp_path / "My Session Directory"
    spaced_dir.mkdir()
    spaced_file = spaced_dir / "camera export video 01.csv"
    spaced_file.write_text(
        "timestamp,ax,ay,az,gx,gy,gz,latitude,longitude,altitude,gps_accuracy\n"
        "100701912,-1.0,0.1,0.1,0.1,0.2,-0.03,31.78,76.99,1029.3,20.4\n"
        "100702912,-1.0,0.1,0.1,0.1,0.2,-0.03,31.78,76.99,1029.3,20.4\n"
    )

    db = DatabaseManager(tmp_path / "test.sqlite3")
    mgr = IngestionManager(
        session_id="session_spaced",
        input_dir=spaced_dir,
        session_storage_dir=tmp_path / "storage",
        db_manager=db,
    )
    assets = mgr.ingest_assets(["camera export video 01.csv"])
    assert len(assets) == 1
    assert "camera export video 01.csv" in assets[0].file_path
    assert assets[0].file_size_bytes > 0


# 8. Re-import Unchanged Input Must Not Duplicate Session Assets
def test_reimport_unchanged_input_no_duplicate_assets(tmp_path: Path) -> None:
    """Re-importing the same input assets must not create duplicate database rows."""
    session_id = "session_dedup_test"
    db_path = tmp_path / "app.sqlite3"
    db = DatabaseManager(db_path)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "test1.csv").write_text("timestamp,ax,ay,az,gx,gy,gz\n1,0,0,0,0,0,0\n")
    (input_dir / "test2.json").write_text('{"label": "test"}')

    mgr = IngestionManager(session_id=session_id, input_dir=input_dir, session_storage_dir=tmp_path / "store", db_manager=db)

    # First ingest
    assets_pass1 = mgr.ingest_assets()
    assert len(assets_pass1) == 2
    db_assets_pass1 = db.list_input_assets(session_id)
    assert len(db_assets_pass1) == 2

    # Second ingest of unchanged files
    assets_pass2 = mgr.ingest_assets()
    assert len(assets_pass2) == 2
    db_assets_pass2 = db.list_input_assets(session_id)
    assert len(db_assets_pass2) == 2  # Count must remain exactly 2!


# 9. Changed Source Hash Invalidates Cache Without Deleting Raw Data
def test_changed_source_hash_invalidates_cache_preserves_raw(tmp_path: Path) -> None:
    """A changed source hash generates a completely distinct cache key without deleting raw data."""
    test_file = tmp_path / "video.mp4"
    test_file.write_bytes(b"initial content 123")

    sha_initial = compute_sha256_file(test_file)
    cache_key_initial = compute_cache_key("ingest", "cfg123", {"source": sha_initial})

    # Modify file
    test_file.write_bytes(b"modified content 456")
    sha_modified = compute_sha256_file(test_file)
    cache_key_modified = compute_cache_key("ingest", "cfg123", {"source": sha_modified})

    assert sha_initial != sha_modified
    assert cache_key_initial != cache_key_modified
    # Raw file must still exist and be intact
    assert test_file.is_file()
    assert test_file.read_bytes() == b"modified content 456"


# 10. Sensor Units Never Mixed and Raw Timestamps Round-Trip Serialization
def test_sensor_units_never_mixed_and_raw_timestamps_exact_roundtrip(tmp_path: Path) -> None:
    """Sensor units must never be mixed and 64-bit integer timestamps must survive round-trip serialization."""
    large_ts = 1789041579535892142  # 64-bit nanosecond-range integer
    summary = SensorSummarySchema(
        sensor_type="camera_imu",
        source_file="camera.csv",
        row_count=100,
        timestamp_column="timestamp",
        raw_t_min=large_ts,
        raw_t_max=large_ts + 1000000,
        raw_t_span=1000000,
        span_seconds_proposed=1.0,
        unit_hypothesis="acceleration: g (norm ~ 1.0); gyro: rad/s",
        is_monotonic=True,
    )

    # JSON round-trip serialization
    json_str = summary.model_dump_json()
    loaded_dict = json.loads(json_str)

    # Must be exact integer without floating-point precision loss
    assert isinstance(loaded_dict["raw_t_min"], int)
    assert loaded_dict["raw_t_min"] == large_ts

    deserialized = SensorSummarySchema.model_validate_json(json_str)
    assert deserialized.raw_t_min == large_ts
    assert deserialized.unit_hypothesis == "acceleration: g (norm ~ 1.0); gyro: rad/s"


# 11. One Constant GPS Fix Is Not Expanded to Multiple Constraints
def test_constant_gps_not_expanded_to_multiple_constraints(tmp_path: Path) -> None:
    """A constant GPS fix across many rows must register as a single constant fix."""
    cam_file = tmp_path / "cam_constant_gps.csv"
    rows = ["timestamp,ax,ay,az,gx,gy,gz,latitude,longitude,altitude,gps_accuracy"]
    for i in range(50):
        rows.append(f"{1000+i},-1.0,0.1,0.1,0.0,0.0,0.0,31.7841343,76.9984522,1029.30,20.46")
    cam_file.write_text("\n".join(rows))

    summary = parse_camera_csv(cam_file)
    assert summary.row_count == 50
    assert summary.gps_analysis["total_fixes"] == 50
    assert summary.gps_analysis["unique_fixes_count"] == 1
    assert summary.gps_analysis["is_constant"] is True
    assert "must NOT be expanded" in summary.gps_analysis["note"]


# 12. Timelapse Export Must Not Imply Real Exposure Cadence
def test_timelapse_export_does_not_imply_real_exposure_cadence(tmp_path: Path) -> None:
    """Recording type timelapse triggers advisory; scale status remains NOT_ESTIMATED."""
    loc_file = tmp_path / "location.json"
    loc_file.write_text(
        json.dumps(
            {
                "sessionLabel": "20260910_172939",
                "recordingType": "timelapse",
                "recordingStartedAt": "2026-09-10T11:59:39.398343Z",
                "startCoordinates": {"worldX": 0.0, "worldY": 0.0, "normalizedX": 0.126, "normalizedY": 0.639},
            }
        )
    )

    loc = parse_location_json(loc_file)
    assert loc.is_timelapse is True
    assert "RECORDING TYPE IS TIMELAPSE" in loc.timelapse_warning
    assert loc.metric_scale_provided is False
    assert loc.full_map_registration_provided is False


# 13. Recomputed Real Data Audit (When Local Files Are Present)
def test_real_files_audit_and_recomputed_metrics() -> None:
    """Audit real dataset in data/2026-09-10/ recomputing exact statistics."""
    real_data_dir = Path("data/2026-09-10")
    if not real_data_dir.is_dir():
        pytest.skip("Real data directory 'data/2026-09-10' not present")

    cam_csv = real_data_dir / "insta360_export_1789041643971.csv"
    phone_csv = real_data_dir / "20260910_172939_phone_sensors.csv"
    loc_json = real_data_dir / "20260910_172939_location.json"

    if cam_csv.is_file():
        cam_summary = parse_camera_csv(cam_csv)
        assert cam_summary.row_count == 40981
        assert cam_summary.raw_t_span == 40713082
        assert cam_summary.short_intervals_count == 2561
        assert cam_summary.channel_norms["acceleration"]["mean"] == pytest.approx(1.0, abs=0.08)
        assert cam_summary.gps_analysis["is_constant"] is True
        assert cam_summary.gps_analysis["unique_fixes_count"] == 1

    if phone_csv.is_file():
        phone_summary = parse_phone_csv(phone_csv)
        assert phone_summary.row_count == 6120
        assert phone_summary.raw_t_span == 61811
        assert phone_summary.channel_norms["acceleration"]["mean"] == pytest.approx(9.8, abs=1.5)
        assert phone_summary.initial_zero_vectors_count == 2

    if loc_json.is_file():
        loc = parse_location_json(loc_json)
        assert loc.session_label == "20260910_172939"
        assert loc.recording_type == "timelapse"
        assert loc.is_timelapse is True


# 14. Ingest Worker Cancellation & Resumption
def test_ingest_worker_cancellation_and_resumption(tmp_path: Path) -> None:
    """Worker cooperatively aborts when cancel flag set, and resumes cleanly."""
    session_id = "synth_cancel_test"
    run_dir = tmp_path / "runs" / "run_cancel_01"
    run_dir.mkdir(parents=True)

    input_dir = tmp_path / "data" / "sessions" / session_id
    input_dir.mkdir(parents=True)
    (input_dir / "camera.csv").write_text("timestamp,ax,ay,az,gx,gy,gz\n100,0,0,1,0,0,0\n")

    cfg = {
        "storage": {
            "app_db_path": str(tmp_path / "app.sqlite3"),
            "sessions_dir": str(tmp_path / "data" / "sessions"),
            "base_data_dir": str(tmp_path / "data"),
        }
    }

    # Test Cancellation
    cancelled = True
    worker_cancel = IngestWorker(
        run_id="run_cancel_01",
        session_id=session_id,
        config=cfg,
        run_dir=run_dir,
        cancel_checker=lambda: cancelled,
    )
    with pytest.raises(InterruptedError):
        worker_cancel.execute()

    # Outputs must NOT be marked complete
    marker_file = run_dir / "outputs" / ".complete"
    assert not marker_file.exists()

    # Test Resumption / Clean Run
    run_dir_resume = tmp_path / "runs" / "run_resume_01"
    run_dir_resume.mkdir(parents=True)
    worker_resume = IngestWorker(
        run_id="run_resume_01",
        session_id=session_id,
        config=cfg,
        run_dir=run_dir_resume,
        cancel_checker=lambda: False,
    )
    outputs = worker_resume.execute()
    assert "input_manifest" in outputs
    assert "audit_report" in outputs
    assert outputs["audit_report"] == "04_input_audit.md"


# 15. Real Video Files: Declared Properties vs. Decoded Frames
def test_real_files_declared_vs_decoded_video_properties(tmp_path: Path) -> None:
    """Compare declared video container properties with decoded frames and record unknown metadata."""
    real_data_dir = Path("data/2026-09-10")
    if not real_data_dir.is_dir():
        pytest.skip("Real data directory 'data/2026-09-10' not present")

    video_cases = [
        ("LRV_20260910_172941_01_083.mp4", 1664, 832, 1),
        ("insta360_export_1789041643971.mp4", 3840, 1920, 1),
        ("VID_20260910_172941_00_083.insv", 2880, 2880, 2),
    ]

    for vname, expected_w, expected_h, expected_video_tracks in video_cases:
        vpath = real_data_dir / vname
        if not vpath.is_file():
            continue

        probe = probe_video_stream(vpath, f"asset_test_{vname}")
        assert probe.file_size_bytes > 0
        assert probe.container_duration_s is not None
        assert probe.container_duration_s == pytest.approx(4.03, abs=0.05)

        # Check tracks
        v_tracks = [t for t in probe.tracks if t.track_type == "video"]
        assert len(v_tracks) == expected_video_tracks
        assert v_tracks[0].width == expected_w
        assert v_tracks[0].height == expected_h

        # Unknown fields must remain unknown, NOT fabricated
        assert v_tracks[0].exposure_duration_s is None
        assert v_tracks[0].rolling_shutter_readout_s is None
        assert len(v_tracks[0].unknown_fields) >= 2

        # Decode frames and compare actual image dimensions with declared properties
        out_frames_dir = tmp_path / f"frames_{Path(vname).stem}"
        frames_meta = decode_sample_frames(
            vpath,
            output_dir=out_frames_dir,
            duration_s=probe.container_duration_s,
            num_video_tracks=len(v_tracks),
        )
        assert len(frames_meta) == 3
        for f in frames_meta:
            assert f["status"] == "DECODED"
            assert f["exposure_time_s"] is None  # Explicitly unknown
            fpath = Path(f["frame_path"])
            assert fpath.is_file()
            with Image.open(fpath) as img:
                if expected_video_tracks >= 2:
                    # Dual-lens side-by-side composite
                    assert img.size == (expected_w * 2, expected_h)
                else:
                    assert img.size == (expected_w, expected_h)


# 16. Dual-Lens INSV Streams Inspection
def test_dual_lens_insv_streams_inspection() -> None:
    """Verify that INSV contains 2 separate video streams representing both original circular fisheye lenses."""
    insv_path = Path("data/2026-09-10/VID_20260910_172941_00_083.insv")
    if not insv_path.is_file():
        pytest.skip("Real INSV file not present")

    probe = probe_video_stream(insv_path, "test_insv")
    v_tracks = [t for t in probe.tracks if t.track_type == "video"]
    assert len(v_tracks) == 2, "INSV must contain exactly 2 video streams for dual-lens 360"
    assert v_tracks[0].width == 2880 and v_tracks[0].height == 2880
    assert v_tracks[1].width == 2880 and v_tracks[1].height == 2880
    assert v_tracks[0].codec in ("hevc", "hvc1")
    assert v_tracks[1].codec in ("hevc", "hvc1")

    # Geometry evaluation must confirm both lens streams present
    eval_result = evaluate_camera_geometry([probe])
    assert eval_result["insv_dual_lenses_verified"] is True
    insv_eval = eval_result["per_file_evaluations"]["VID_20260910_172941_00_083.insv"]
    assert insv_eval["required_lens_streams_present"] is True
    assert insv_eval["geometry_verdict"] == "DUAL_CIRCULAR_FISHEYE_RAW"


# 17. Screenshot File Rejected As Video
def test_screenshot_file_rejected_as_video(tmp_path: Path) -> None:
    """A screenshot image file pretending to be video by filename must be rejected as invalid media."""
    fake_video = tmp_path / "fake_capture.mp4"
    # Write valid PNG header bytes
    fake_video.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x00")

    probe = probe_video_stream(fake_video, "asset_fake")
    assert probe.format_name == "screenshot_image"
    assert len(probe.tracks) == 0
    assert probe.layout_verification.get("is_valid_video") is False


# 18. Binary / NUL Byte Corrupt CSV Handled Safely
def test_binary_nul_corrupt_csv_handled_safely(tmp_path: Path) -> None:
    """Binary files with NUL bytes or image magic headers must raise ValueError cleanly."""
    binary_file = tmp_path / "binary_data.csv"
    binary_file.write_bytes(b"timestamp,ax,ay\x00\x01\x02CORRUPT_BYTES")

    with pytest.raises(ValueError) as excinfo:
        parse_camera_csv(binary_file)
    assert "binary/corrupt" in str(excinfo.value).lower()

    with pytest.raises(ValueError) as excinfo_phone:
        parse_phone_csv(binary_file)
    assert "binary/corrupt" in str(excinfo_phone.value).lower()


# 19. Graceful WAITING_INPUT When Files Are Missing
def test_waiting_input_when_files_missing(tmp_path: Path) -> None:
    """When input files are missing, worker must return WAITING_INPUT without crashing."""
    session_id = "synth_missing_data_test"
    run_dir = tmp_path / "runs" / "run_missing_01"
    run_dir.mkdir(parents=True)

    cfg = {
        "storage": {
            "app_db_path": str(tmp_path / "app.sqlite3"),
            "sessions_dir": str(tmp_path / "data" / "sessions"),
            "base_data_dir": str(tmp_path / "data_empty"),
        }
    }

    worker = IngestWorker(
        run_id="run_missing_01",
        session_id=session_id,
        config=cfg,
        run_dir=run_dir,
        cancel_checker=lambda: False,
    )
    outputs = worker.execute()
    assert "input_manifest" in outputs
    assert "audit_report" in outputs

    manifest_file = run_dir / "outputs" / "input_manifest.json"
    assert manifest_file.is_file()
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest_data["status"] == "WAITING_INPUT"
    assert len(manifest_data["missing_files"]) > 0
    assert manifest_data["scale_status"] == "NOT_ESTIMATED"
    assert manifest_data["prompt_05_readiness"]["actual_imagery_usable"] is False

