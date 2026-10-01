"""Schemas for job lifecycle, execution records, completion markers, and progress telemetry.

States:
- QUEUED: Job submitted, waiting in concurrency queue
- RUNNING: Subprocess actively executing in its owned process group
- SUCCEEDED: Completed successfully with verified atomic completion marker and checksums
- FAILED: Worker exited with non-zero code or raised unhandled exception
- CANCELLED: Job cooperatively cancelled by user via SIGTERM/SIGKILL
- INTERRUPTED: Unexpected worker termination (supervisor crash, WSL shutdown, stale PID)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from vscan.schemas.common import SCHEMA_VERSION


class JobStatus(str, Enum):
    """Authoritative lifecycle states for all vscan jobs."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


class ErrorClassification(str, Enum):
    """Categorized root causes for non-successful jobs."""

    NONE = "NONE"
    USER_ERROR = "USER_ERROR"
    CONFIG_ERROR = "CONFIG_ERROR"
    TIMEOUT = "TIMEOUT"
    RESOURCE_EXHAUSTION = "RESOURCE_EXHAUSTION"
    SUBPROCESS_CRASH = "SUBPROCESS_CRASH"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"
    CORRUPT_OUTPUT = "CORRUPT_OUTPUT"
    STALE_PID_DETECTED = "STALE_PID_DETECTED"
    UNKNOWN = "UNKNOWN"


class CompletionMarkerSchema(BaseModel):
    """Atomic completion marker payload stored in outputs/.complete.

    Guarantees:
    - Written atomically via temporary file rename.
    - Contains sha256 checksums and byte sizes of all produced outputs.
    - An interrupted or failed output is never marked complete.
    """

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    run_id: str = Field(..., description="Unique job execution run ID")
    stage: str = Field(..., description="Pipeline stage name")
    session_id: str = Field(..., description="Associated session identifier")
    completed_at_utc: str = Field(..., description="UTC ISO-8601 completion timestamp")
    cache_key: str = Field(..., description="Deterministic cache key for outputs")
    output_checksums: Dict[str, str] = Field(
        ...,
        description="Map of relative output file path to sha256 checksum",
    )
    output_sizes_bytes: Dict[str, int] = Field(
        ...,
        description="Map of relative output file path to file size in bytes",
    )
    marker_sha256: Optional[str] = Field(default=None, description="Checksum of marker content itself")


class JobProgressSchema(BaseModel):
    """Heartbeat and progress status updated periodically (at least once per minute)."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    run_id: str = Field(..., description="Associated run ID")
    timestamp_utc: str = Field(..., description="Heartbeat timestamp")
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0, description="0.0 to 100.0 percent")
    current_step: str = Field(default="", description="Current operation description")
    items_completed: int = Field(default=0, ge=0)
    items_total: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    eta_seconds: Optional[float] = Field(default=None, ge=0.0)
    telemetry: Dict[str, Any] = Field(default_factory=dict, description="Live telemetry snapshot (RAM, GPU)")


class JobRecordSchema(BaseModel):
    """Full durable execution record stored in data/app.sqlite3 and runs/<run-id>/run.json."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    run_id: str = Field(..., description="Unique run identifier (e.g. run_smoke_20260916_123456_abc123)")
    stage: str = Field(..., description="Pipeline stage name")
    session_id: str = Field(..., description="Target session identifier")
    profile: str = Field(default="laptop", description="Execution profile name")
    status: JobStatus = Field(default=JobStatus.QUEUED, description="Current lifecycle state")
    parent_run_id: Optional[str] = Field(default=None, description="Predecessor run ID if resumed or retried")
    attempt: int = Field(default=1, ge=1, description="Attempt number for this stage/session")
    created_at_utc: str = Field(..., description="UTC ISO-8601 creation timestamp")
    started_at_utc: Optional[str] = Field(default=None, description="UTC ISO-8601 start timestamp")
    finished_at_utc: Optional[str] = Field(default=None, description="UTC ISO-8601 completion timestamp")
    duration_seconds: Optional[float] = Field(default=None, ge=0.0)
    pid: Optional[int] = Field(default=None, description="Worker OS Process ID")
    pgid: Optional[int] = Field(default=None, description="Worker OS Process Group ID")
    process_create_time: Optional[float] = Field(
        default=None,
        description="Worker OS process start time for stale PID detection",
    )
    hostname: Optional[str] = Field(default=None, description="Host/WSL instance name")
    cmd_args: List[str] = Field(default_factory=list, description="Exact command line arguments")
    app_version: str = Field(default="0.2.0", description="vscan package version")
    requires_gpu: bool = Field(default=False, description="True if stage requires exclusive project GPU lock")
    input_hashes: Dict[str, str] = Field(
        default_factory=dict,
        description="Frozen input artifact IDs / paths mapped to sha256",
    )
    config_hash: str = Field(default="", description="sha256 of effective frozen configuration")
    calibration_hash: str = Field(default="", description="sha256 of active calibration if applicable")
    pose_hash: str = Field(default="", description="sha256 of input poses if applicable")
    cache_key: str = Field(default="", description="Deterministic composite cache key")
    outputs: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of output logical names to relative paths",
    )
    error_classification: ErrorClassification = Field(
        default=ErrorClassification.NONE,
        description="Classification of failure if status != SUCCEEDED",
    )
    error_message: Optional[str] = Field(default=None, description="Detailed error summary or exception message")
    telemetry: Dict[str, Any] = Field(default_factory=dict, description="Captured resource telemetry")
