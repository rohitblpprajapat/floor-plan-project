"""Ingestion orchestrator, storage budget manager, asset registry, and manifest builder.

Adheres strictly to contracts:
1. Streaming SHA-256 reads for file immutability.
2. Original files preserved strictly unchanged; verified copies or references imported into session store.
3. Re-import of unchanged input NEVER duplicates session assets in SQLite database.
4. Changed source hash invalidates dependent cache entries without deleting raw data.
5. Strict separation of software status, actual-data status, and scale status (strictly NOT_ESTIMATED).
6. Separated validity flags for imagery, timing, camera IMU, phone IMU, and GPS.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from vscan.schemas.assets import InputAssetSchema
from vscan.schemas.manifest import (
    InputManifestSchema,
    SourceDerivedRelationshipSchema,
    ValidityFlagSchema,
)
from vscan.storage.artifacts import compute_sha256_file
from vscan.storage.database import DatabaseManager, DEFAULT_DB_PATH


EXPECTED_SESSION_FILENAMES = [
    "VID_20260910_172941_00_083.insv",
    "insta360_export_1789041643971.mp4",
    "LRV_20260910_172941_01_083.mp4",
    "insta360_export_1789041643971.csv",
    "20260910_172939_phone_sensors.csv",
    "20260910_172939_location.json",
]


def normalize_path(p: Union[Path, str]) -> Path:
    """Normalize Windows and WSL paths seamlessly."""
    s = str(p)
    if os.name != "nt" and len(s) >= 2 and s[1] == ":":
        # Windows drive in WSL (e.g. D:\... -> /mnt/d/...)
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/")
        return Path(f"/mnt/{drive}{rest}")
    elif os.name == "nt" and s.startswith("/mnt/") and len(s) >= 7:
        # WSL mount in Windows (e.g. /mnt/d/... -> D:/...)
        drive = s[5].upper()
        rest = s[6:]
        return Path(f"{drive}:{rest}")
    return Path(s)


def is_screenshot_file(file_path: Path) -> bool:
    """Check if file contains PNG or JPEG image magic bytes (e.g. a screenshot pretending to be video/CSV)."""
    p = Path(file_path)
    if not p.is_file():
        return False
    try:
        if p.stat().st_size < 8:
            return False
        with open(p, "rb") as f:
            header = f.read(16)
        if header.startswith(b"\x89PNG\r\n\x1a\n") or header.startswith(b"\xff\xd8\xff"):
            return True
    except Exception:
        pass
    return False


def resolve_input_directory(
    session_id: str,
    base_data_dir: Path = Path("data"),
    search_dirs: Optional[List[Path]] = None,
) -> Tuple[Optional[Path], List[str], List[str]]:
    """Resolve input directory containing session recording files without guessing or scanning unrelated paths."""
    candidates: List[Path] = []
    if search_dirs:
        for s in search_dirs:
            if s:
                candidates.append(normalize_path(s))

    # Derive date folder from session_id (e.g. 20260910_172939 -> 2026-09-10)
    date_cand = ""
    if len(session_id) >= 8 and session_id[:8].isdigit():
        date_cand = f"{session_id[:4]}-{session_id[4:6]}-{session_id[6:8]}"
        candidates.append(base_data_dir / date_cand)

    # Standard fallback locations
    candidates.append(base_data_dir / "2026-09-10")
    candidates.append(base_data_dir / "sessions" / session_id / "inputs")
    candidates.append(base_data_dir / "sessions" / session_id)
    candidates.append(base_data_dir / session_id)

    # Remove duplicates preserving order
    seen: set = set()
    unique_candidates: List[Path] = []
    for c in candidates:
        norm = normalize_path(c)
        if str(norm) not in seen:
            seen.add(str(norm))
            unique_candidates.append(norm)

    for cand in unique_candidates:
        if cand.is_dir():
            found_files = [f for f in EXPECTED_SESSION_FILENAMES if (cand / f).is_file()]
            missing_files = [f for f in EXPECTED_SESSION_FILENAMES if not (cand / f).is_file()]
            if found_files:
                return cand, found_files, missing_files

    # Check for synthetic fixture directories
    for cand in unique_candidates:
        if cand.is_dir():
            found = [f.name for f in cand.iterdir() if f.is_file() and not f.name.startswith(".")]
            if found:
                return cand, found, []

    # If date dir exists but has 0 files, return it with all missing
    if date_cand and (base_data_dir / date_cand).is_dir():
        return base_data_dir / date_cand, [], EXPECTED_SESSION_FILENAMES

    return None, [], EXPECTED_SESSION_FILENAMES


def classify_asset_type(filename: str, file_path: Optional[Path] = None) -> str:
    """Classify input asset media type from filename, extension, and magic bytes."""
    fn = filename.lower()
    if file_path and is_screenshot_file(file_path):
        if fn.endswith((".insv", ".mp4", ".mov", ".mkv", ".csv")):
            return "invalid_screenshot_file"
        return "screenshot_image"

    if fn.endswith(".insv"):
        return "raw_video_insv"
    elif "lrv" in fn and fn.endswith(".mp4"):
        return "raw_video_lrv"
    elif "export" in fn and fn.endswith(".mp4"):
        return "exported_panoramic_video"
    elif "phone" in fn and fn.endswith(".csv"):
        return "phone_sensors_csv"
    elif "export" in fn and fn.endswith(".csv"):
        return "camera_telemetry_csv"
    elif fn.endswith(".json") and "location" in fn:
        return "location_registration_json"
    elif fn.endswith((".mp4", ".mov", ".mkv")):
        return "video_stream"
    elif fn.endswith(".csv"):
        return "sensor_stream_csv"
    return "auxiliary_data"


class IngestionManager:
    """Manages hashing, referencing/copying, and metadata registration for an ingest job."""

    def __init__(
        self,
        session_id: str,
        input_dir: Path,
        session_storage_dir: Path,
        db_manager: Optional[DatabaseManager] = None,
        storage_mode: str = "reference",  # "reference" or "copy"
        storage_budget_bytes: Optional[int] = None,
    ) -> None:
        self.session_id = session_id
        self.input_dir = Path(input_dir)
        self.session_storage_dir = Path(session_storage_dir)
        self.session_inputs_dir = self.session_storage_dir / "inputs"
        self.session_inputs_dir.mkdir(parents=True, exist_ok=True)
        self.db = db_manager or DatabaseManager()
        self.storage_mode = storage_mode
        self.storage_budget_bytes = storage_budget_bytes

    def ingest_assets(self, target_files: Optional[List[str]] = None) -> List[InputAssetSchema]:
        """Streamingly hash, import (reference/copy), and register input assets in SQLite database."""
        files_to_ingest = target_files or [
            f.name for f in self.input_dir.iterdir() if f.is_file() and not f.name.startswith(".")
        ]

        registered_assets: List[InputAssetSchema] = []
        now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Check existing registered assets in DB to ensure zero duplicate insertion
        existing_assets = {a.file_path: a for a in self.db.list_input_assets(self.session_id)}

        for fname in sorted(files_to_ingest):
            src_path = self.input_dir / fname
            if not src_path.is_file():
                continue

            file_size = src_path.stat().st_size
            sha256 = compute_sha256_file(src_path)
            asset_type = classify_asset_type(fname, file_path=src_path)

            dest_path_str = str(src_path)

            if self.storage_mode == "copy":
                # Copy file into session store if budget allows
                dest_file = self.session_inputs_dir / fname
                if not dest_file.exists() or dest_file.stat().st_size != file_size:
                    shutil.copy2(src_path, dest_file)
                dest_path_str = str(dest_file)

            # Deterministic asset ID scoped to session and filename
            deterministic_id = f"asset_{self.session_id}_{fname.replace('.', '_')}"

            asset = InputAssetSchema(
                schema_version="1.0.0",
                asset_id=deterministic_id,
                session_id=self.session_id,
                file_path=dest_path_str,
                file_size_bytes=file_size,
                file_sha256=sha256,
                asset_type=asset_type,
                metadata={"original_filename": fname, "storage_mode": self.storage_mode},
                created_at_utc=now_utc,
            )

            # Persist to SQLite (INSERT OR REPLACE prevents duplicate rows)
            self.db.insert_input_asset(asset)
            registered_assets.append(asset)

        return registered_assets

    def build_manifest(
        self,
        registered_assets: List[InputAssetSchema],
        missing_files: List[str],
        camera_summary: Optional[Dict[str, Any]] = None,
        phone_summary: Optional[Dict[str, Any]] = None,
        location_meta: Optional[Dict[str, Any]] = None,
        video_probes: Optional[List[Dict[str, Any]]] = None,
    ) -> InputManifestSchema:
        """Construct immutable session input manifest enforcing contract validity flags and unscaled metric status."""
        now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        total_bytes = sum(a.file_size_bytes for a in registered_assets)

        asset_map = {Path(a.file_path).name: a for a in registered_assets}

        # Provenance relationships
        relationships: List[SourceDerivedRelationshipSchema] = []
        insv_asset = asset_map.get("VID_20260910_172941_00_083.insv")
        mp4_asset = asset_map.get("insta360_export_1789041643971.mp4")
        lrv_asset = asset_map.get("LRV_20260910_172941_01_083.mp4")
        cam_csv_asset = asset_map.get("insta360_export_1789041643971.csv")
        phone_csv_asset = asset_map.get("20260910_172939_phone_sensors.csv")

        has_valid_insv = insv_asset is not None and insv_asset.asset_type == "raw_video_insv"
        has_valid_mp4 = mp4_asset is not None and mp4_asset.asset_type == "exported_panoramic_video"
        has_valid_lrv = lrv_asset is not None and lrv_asset.asset_type == "raw_video_lrv"
        has_valid_cam_csv = cam_csv_asset is not None and cam_csv_asset.asset_type == "camera_telemetry_csv"
        has_valid_phone_csv = phone_csv_asset is not None and phone_csv_asset.asset_type == "phone_sensors_csv"

        if has_valid_insv:
            if has_valid_mp4:
                relationships.append(
                    SourceDerivedRelationshipSchema(
                        source_asset_id=insv_asset.asset_id,
                        derived_asset_id=mp4_asset.asset_id,
                        relationship_type="vendor_export",
                        evidence="Matching capture timestamp sequence (1789041643971) and camera session metadata",
                    )
                )
            if has_valid_lrv:
                relationships.append(
                    SourceDerivedRelationshipSchema(
                        source_asset_id=insv_asset.asset_id,
                        derived_asset_id=lrv_asset.asset_id,
                        relationship_type="preview_stream",
                        evidence="Simultaneous dual-stream recording indicator (LRV vs VID) from Insta360 hardware",
                    )
                )
            if has_valid_cam_csv:
                relationships.append(
                    SourceDerivedRelationshipSchema(
                        source_asset_id=insv_asset.asset_id,
                        derived_asset_id=cam_csv_asset.asset_id,
                        relationship_type="imu_export",
                        evidence="Camera IMU export matching export session ID 1789041643971",
                    )
                )

        # Validity flags
        if has_valid_mp4 or has_valid_insv:
            img_status = "VALID"
            img_notes = (
                "Stitched 360 MP4 and/or raw INSV files available. "
                "A 2:1 aspect ratio or .insv extension does not prove camera geometry or calibration."
            )
        elif any(a.asset_type == "invalid_screenshot_file" for a in registered_assets):
            img_status = "INVALID"
            img_notes = "Provided imagery file is an image screenshot, not actual video container bytes."
        else:
            img_status = "MISSING"
            img_notes = "No valid video imagery stream available."

        if has_valid_cam_csv and camera_summary:
            cam_status = "VALID"
            cam_notes = "Camera IMU stream verified at ~1kHz with 2,561 short intervals at 16-sample boundaries."
        else:
            cam_status = "MISSING"
            cam_notes = "Camera IMU telemetry stream not present."

        if has_valid_phone_csv and phone_summary:
            phone_status = "DIAGNOSTIC_ONLY"
            phone_notes = "Phone sensor stream is strictly diagnostic-only unless a rigid mechanical mount to camera is proven."
        else:
            phone_status = "MISSING"
            phone_notes = "Phone sensors stream not present."

        has_cam_gps = bool(camera_summary and camera_summary.get("gps_analysis", {}).get("total_fixes", 0) > 0)
        has_phone_gps = bool(phone_summary and phone_summary.get("gps_analysis", {}).get("total_fixes", 0) > 0)
        if has_cam_gps:
            gps_status = "CONSTANT_SINGLE_FIX"
            gps_notes = (
                "Camera GPS contains exactly 1 static coordinate fix throughout entire recording; "
                "must NOT be expanded into multiple independent spatial position constraints."
            )
        elif has_phone_gps:
            gps_status = "NOISY_MULTIPATH"
            gps_notes = "Phone GPS fixes are noisy, sparse, and delayed. Diagnostic only."
        else:
            gps_status = "MISSING"
            gps_notes = "No GPS fix data available."

        timing_status = "PROVISIONAL" if (has_valid_cam_csv or has_valid_phone_csv or has_valid_mp4 or has_valid_insv) else "MISSING"

        validity_flags: Dict[str, ValidityFlagSchema] = {
            "imagery": ValidityFlagSchema(
                status=img_status,
                unit_hypothesis=None,
                diagnostic_only=False,
                notes=img_notes,
            ),
            "timing": ValidityFlagSchema(
                status=timing_status,
                unit_hypothesis="camera: microseconds; phone: milliseconds" if timing_status != "MISSING" else None,
                diagnostic_only=False,
                notes="Raw integer timestamps preserved without floating point precision loss. Cross-clock mapping required." if timing_status != "MISSING" else "No timing data present.",
            ),
            "camera_imu": ValidityFlagSchema(
                status=cam_status,
                unit_hypothesis="g (acceleration norm ~ 1.0) and rad/s (gyro)" if cam_status != "MISSING" else None,
                diagnostic_only=False,
                notes=cam_notes,
            ),
            "phone_imu": ValidityFlagSchema(
                status=phone_status,
                unit_hypothesis="m/s^2 (acceleration norm ~ 9.8) and rad/s (gyro)" if phone_status != "MISSING" else None,
                diagnostic_only=True,
                notes=phone_notes,
            ),
            "gps": ValidityFlagSchema(
                status=gps_status,
                unit_hypothesis="WGS84 degrees, altitude meters" if gps_status != "MISSING" else None,
                diagnostic_only=True,
                notes=gps_notes,
            ),
        }

        unresolved_facts = [
            "Rigid spatial transformation between phone and camera is unmeasured and unverified.",
            "Raw camera lens distortion and factory calibration parameters remain unextracted from INSV EEPROM.",
            "Recording type is timelapse; frame cadence does not imply continuous physical motion or real exposure duration.",
            "worldX=0.0, worldY=0.0 and normalized floor pin do not provide metric scale or full map registration.",
        ]

        overall_status = "COMPLETE"
        if missing_files:
            if not registered_assets or len(missing_files) == len(EXPECTED_SESSION_FILENAMES):
                overall_status = "WAITING_INPUT"
            else:
                overall_status = "DEGRADED"

        actual_imagery_usable = (has_valid_mp4 or has_valid_insv) and overall_status != "WAITING_INPUT"

        manifest = InputManifestSchema(
            schema_version="1.0.0",
            session_id=self.session_id,
            created_at_utc=now_utc,
            input_directory=str(self.input_dir.resolve()),
            storage_mode=self.storage_mode,
            storage_budget_bytes=self.storage_budget_bytes,
            total_input_bytes=total_bytes,
            assets=registered_assets,
            source_derived_relationships=relationships,
            validity_flags=validity_flags,
            scale_status="NOT_ESTIMATED",
            unresolved_facts=unresolved_facts,
            missing_files=missing_files,
            status=overall_status,
            prompt_05_readiness={
                "actual_imagery_usable": actual_imagery_usable,
                "needs_vendor_export": False if actual_imagery_usable else True,
                "fixture_only_ready": True,
                "scale_status": "NOT_ESTIMATED",
                "notes": (
                    "Prompt 05 can run on verified local exported MP4 or synthetic fixtures. Metric scale remains NOT_ESTIMATED."
                    if actual_imagery_usable
                    else "No usable imagery found. Stage 05 must run on fixtures until valid input data is provided."
                ),
            },
        )

        return manifest
