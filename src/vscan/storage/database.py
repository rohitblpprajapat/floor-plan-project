"""SQLite database manager for application state, sessions, and durable job records.

Guarantees:
- WAL mode enabled for safe concurrent reader/writer access.
- Busy timeout set to 5000ms.
- COLMAP feature databases remain strictly separate from app.sqlite3.
- Full JobRecordSchema round-trip persistence.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from vscan.schemas.assets import InputAssetSchema
from vscan.schemas.common import CoordinateFrame, Units
from vscan.schemas.jobs import ErrorClassification, JobRecordSchema, JobStatus
from vscan.schemas.poses import MapVersionSchema


DEFAULT_DB_PATH = Path("data/app.sqlite3")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    run_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    session_id TEXT NOT NULL,
    profile TEXT NOT NULL,
    status TEXT NOT NULL,
    parent_run_id TEXT,
    attempt INTEGER NOT NULL DEFAULT 1,
    created_at_utc TEXT NOT NULL,
    started_at_utc TEXT,
    finished_at_utc TEXT,
    duration_seconds REAL,
    pid INTEGER,
    pgid INTEGER,
    process_create_time REAL,
    hostname TEXT,
    cmd_args_json TEXT NOT NULL DEFAULT '[]',
    app_version TEXT NOT NULL,
    requires_gpu INTEGER NOT NULL DEFAULT 0,
    input_hashes_json TEXT NOT NULL DEFAULT '{}',
    config_hash TEXT NOT NULL DEFAULT '',
    calibration_hash TEXT NOT NULL DEFAULT '',
    pose_hash TEXT NOT NULL DEFAULT '',
    cache_key TEXT NOT NULL DEFAULT '',
    outputs_json TEXT NOT NULL DEFAULT '{}',
    error_classification TEXT NOT NULL DEFAULT 'NONE',
    error_message TEXT,
    telemetry_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_session ON jobs(session_id);
CREATE INDEX IF NOT EXISTS idx_jobs_cache_key ON jobs(cache_key);

CREATE TABLE IF NOT EXISTS input_assets (
    asset_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    file_sha256 TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS maps (
    map_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    parent_map_id TEXT,
    map_version INTEGER NOT NULL DEFAULT 1,
    map_type TEXT NOT NULL,
    gravity_aligned INTEGER NOT NULL DEFAULT 0,
    coordinate_frame TEXT NOT NULL,
    units TEXT NOT NULL,
    scale_factor_meters_per_unit REAL,
    provenance TEXT NOT NULL,
    is_immutable INTEGER NOT NULL DEFAULT 1,
    num_poses INTEGER NOT NULL DEFAULT 0,
    num_landmarks INTEGER NOT NULL DEFAULT 0,
    exported_at_utc TEXT NOT NULL
);
"""


class DatabaseManager:
    """Manages transactional interactions with data/app.sqlite3."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(SCHEMA_SQL)

    def init_db(self) -> None:
        """Public alias for database initialization."""
        self._init_db()

    def insert_job(self, record: JobRecordSchema) -> None:
        """Insert a newly queued or created job record."""
        sql = """
        INSERT OR REPLACE INTO jobs (
            run_id, stage, session_id, profile, status, parent_run_id, attempt,
            created_at_utc, started_at_utc, finished_at_utc, duration_seconds,
            pid, pgid, process_create_time, hostname, cmd_args_json, app_version,
            requires_gpu, input_hashes_json, config_hash, calibration_hash,
            pose_hash, cache_key, outputs_json, error_classification, error_message,
            telemetry_json
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?
        )
        """
        with self._get_connection() as conn:
            conn.execute(
                sql,
                (
                    record.run_id,
                    record.stage,
                    record.session_id,
                    record.profile,
                    record.status.value,
                    record.parent_run_id,
                    record.attempt,
                    record.created_at_utc,
                    record.started_at_utc,
                    record.finished_at_utc,
                    record.duration_seconds,
                    record.pid,
                    record.pgid,
                    record.process_create_time,
                    record.hostname,
                    json.dumps(record.cmd_args),
                    record.app_version,
                    1 if record.requires_gpu else 0,
                    json.dumps(record.input_hashes),
                    record.config_hash,
                    record.calibration_hash,
                    record.pose_hash,
                    record.cache_key,
                    json.dumps(record.outputs),
                    record.error_classification.value,
                    record.error_message,
                    json.dumps(record.telemetry),
                ),
            )

    def update_job(self, record: JobRecordSchema) -> None:
        """Update an existing job record."""
        self.insert_job(record)

    def get_job(self, run_id: str) -> Optional[JobRecordSchema]:
        """Fetch a single job record by run_id."""
        sql = "SELECT * FROM jobs WHERE run_id = ?"
        with self._get_connection() as conn:
            row = conn.execute(sql, (run_id,)).fetchone()
            if row is None:
                return None
            return self._row_to_job(row)

    def list_jobs(
        self,
        session_id: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> List[JobRecordSchema]:
        """List job records with optional filtering."""
        query = "SELECT * FROM jobs"
        params: List[Any] = []
        conditions = []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if status:
            conditions.append("status = ?")
            params.append(status.value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at_utc DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_job(r) for r in rows]

    def get_running_or_queued_jobs(self) -> List[JobRecordSchema]:
        """Get all active jobs currently RUNNING or QUEUED."""
        sql = "SELECT * FROM jobs WHERE status IN ('QUEUED', 'RUNNING') ORDER BY created_at_utc ASC"
        with self._get_connection() as conn:
            rows = conn.execute(sql).fetchall()
            return [self._row_to_job(r) for r in rows]

    def find_cached_successful_job(self, cache_key: str, session_id: Optional[str] = None) -> Optional[JobRecordSchema]:
        """Find the latest SUCCEEDED job with the exact matching cache key and optional session_id."""
        if not cache_key:
            return None
        if session_id:
            sql = "SELECT * FROM jobs WHERE cache_key = ? AND session_id = ? AND status = 'SUCCEEDED' ORDER BY created_at_utc DESC LIMIT 1"
            params = (cache_key, session_id)
        else:
            sql = "SELECT * FROM jobs WHERE cache_key = ? AND status = 'SUCCEEDED' ORDER BY created_at_utc DESC LIMIT 1"
            params = (cache_key,)
        with self._get_connection() as conn:
            row = conn.execute(sql, params).fetchone()
            if row is None:
                return None
            return self._row_to_job(row)

    def _row_to_job(self, row: sqlite3.Row) -> JobRecordSchema:
        return JobRecordSchema(
            run_id=row["run_id"],
            stage=row["stage"],
            session_id=row["session_id"],
            profile=row["profile"],
            status=JobStatus(row["status"]),
            parent_run_id=row["parent_run_id"],
            attempt=row["attempt"],
            created_at_utc=row["created_at_utc"],
            started_at_utc=row["started_at_utc"],
            finished_at_utc=row["finished_at_utc"],
            duration_seconds=row["duration_seconds"],
            pid=row["pid"],
            pgid=row["pgid"],
            process_create_time=row["process_create_time"],
            hostname=row["hostname"],
            cmd_args=json.loads(row["cmd_args_json"] or "[]"),
            app_version=row["app_version"],
            requires_gpu=bool(row["requires_gpu"]),
            input_hashes=json.loads(row["input_hashes_json"] or "{}"),
            config_hash=row["config_hash"] or "",
            calibration_hash=row["calibration_hash"] or "",
            pose_hash=row["pose_hash"] or "",
            cache_key=row["cache_key"] or "",
            outputs=json.loads(row["outputs_json"] or "{}"),
            error_classification=ErrorClassification(row["error_classification"] or "NONE"),
            error_message=row["error_message"],
            telemetry=json.loads(row["telemetry_json"] or "{}"),
        )

    def insert_input_asset(self, asset: InputAssetSchema) -> None:
        """Insert or replace an input asset manifest record."""
        sql = """
        INSERT OR REPLACE INTO input_assets (
            asset_id, session_id, file_path, file_size_bytes, file_sha256,
            asset_type, metadata_json, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            conn.execute(
                sql,
                (
                    asset.asset_id,
                    asset.session_id,
                    asset.file_path,
                    asset.file_size_bytes,
                    asset.file_sha256,
                    asset.asset_type,
                    json.dumps(asset.metadata),
                    asset.created_at_utc,
                ),
            )

    def get_input_asset(self, asset_id: str) -> Optional[InputAssetSchema]:
        """Fetch an input asset by asset_id."""
        sql = "SELECT * FROM input_assets WHERE asset_id = ?"
        with self._get_connection() as conn:
            row = conn.execute(sql, (asset_id,)).fetchone()
            if row is None:
                return None
            return InputAssetSchema(
                schema_version=row.get("schema_version", "1.0.0") if "schema_version" in row.keys() else "1.0.0",
                asset_id=row["asset_id"],
                session_id=row["session_id"],
                file_path=row["file_path"],
                file_size_bytes=row["file_size_bytes"],
                file_sha256=row["file_sha256"],
                asset_type=row["asset_type"],
                metadata=json.loads(row["metadata_json"] or "{}"),
                created_at_utc=row["created_at_utc"],
            )

    def list_input_assets(self, session_id: Optional[str] = None) -> List[InputAssetSchema]:
        """List input assets, optionally filtered by session_id."""
        query = "SELECT * FROM input_assets"
        params: List[Any] = []
        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at_utc ASC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                InputAssetSchema(
                    asset_id=r["asset_id"],
                    session_id=r["session_id"],
                    file_path=r["file_path"],
                    file_size_bytes=r["file_size_bytes"],
                    file_sha256=r["file_sha256"],
                    asset_type=r["asset_type"],
                    metadata=json.loads(r["metadata_json"] or "{}"),
                    created_at_utc=r["created_at_utc"],
                )
                for r in rows
            ]

    def insert_map(self, map_record: MapVersionSchema) -> None:
        """Insert or replace an immutable map version record."""
        sql = """
        INSERT OR REPLACE INTO maps (
            map_id, session_id, parent_map_id, map_version, map_type,
            gravity_aligned, coordinate_frame, units, scale_factor_meters_per_unit,
            provenance, is_immutable, num_poses, num_landmarks, exported_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            conn.execute(
                sql,
                (
                    map_record.map_id,
                    map_record.session_id,
                    map_record.parent_map_id,
                    map_record.map_version,
                    map_record.map_type,
                    1 if map_record.gravity_aligned else 0,
                    map_record.coordinate_frame.value,
                    map_record.units.value,
                    map_record.scale_factor_meters_per_unit,
                    map_record.provenance,
                    1 if map_record.is_immutable else 0,
                    map_record.num_poses,
                    map_record.num_landmarks,
                    map_record.exported_at_utc,
                ),
            )

    def get_map(self, map_id: str) -> Optional[MapVersionSchema]:
        """Fetch a map version record by map_id."""
        sql = "SELECT * FROM maps WHERE map_id = ?"
        with self._get_connection() as conn:
            row = conn.execute(sql, (map_id,)).fetchone()
            if row is None:
                return None
            return MapVersionSchema(
                map_id=row["map_id"],
                session_id=row["session_id"],
                parent_map_id=row["parent_map_id"],
                map_version=row["map_version"],
                map_type=row["map_type"],
                gravity_aligned=bool(row["gravity_aligned"]),
                coordinate_frame=CoordinateFrame(row["coordinate_frame"]),
                units=Units(row["units"]),
                scale_factor_meters_per_unit=row["scale_factor_meters_per_unit"],
                provenance=row["provenance"],
                is_immutable=bool(row["is_immutable"]),
                num_poses=row["num_poses"],
                num_landmarks=row["num_landmarks"],
                exported_at_utc=row["exported_at_utc"],
            )

    def list_maps(self, session_id: Optional[str] = None) -> List[MapVersionSchema]:
        """List map version records, optionally filtered by session_id."""
        query = "SELECT * FROM maps"
        params: List[Any] = []
        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY map_version ASC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                MapVersionSchema(
                    map_id=r["map_id"],
                    session_id=r["session_id"],
                    parent_map_id=r["parent_map_id"],
                    map_version=r["map_version"],
                    map_type=r["map_type"],
                    gravity_aligned=bool(r["gravity_aligned"]),
                    coordinate_frame=CoordinateFrame(r["coordinate_frame"]),
                    units=Units(r["units"]),
                    scale_factor_meters_per_unit=r["scale_factor_meters_per_unit"],
                    provenance=r["provenance"],
                    is_immutable=bool(r["is_immutable"]),
                    num_poses=r["num_poses"],
                    num_landmarks=r["num_landmarks"],
                    exported_at_utc=r["exported_at_utc"],
                )
                for r in rows
            ]
