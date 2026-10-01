"""Schemas for input assets, source frames, and clock mappings.

Rule:
- Use integers for raw large timestamps (nanoseconds or microseconds).
- Use local relative seconds (float) for numerical optimization, interpolation, and SfM/VIO.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from vscan.schemas.common import SCHEMA_VERSION, Units


class InputAssetSchema(BaseModel):
    """Metadata and integrity manifest for an immutable input asset (video, LRV, IMU, etc.)."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    asset_id: str = Field(..., description="Unique identifier for asset (e.g. SHA256 or session-scoped UUID)")
    session_id: str = Field(..., description="Associated session identifier")
    file_path: str = Field(..., description="Relative or canonical path to input asset file")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    file_sha256: str = Field(..., description="SHA-256 hash of the asset file for immutability verification")
    asset_type: str = Field(
        ...,
        description="Type category: raw_video_insv, raw_video_lrv, imu_stream, calibration_file, audio, synthetic_fixture",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extracted container or stream metadata")
    created_at_utc: str = Field(..., description="UTC ISO-8601 timestamp when asset was registered")


class SourceFrameSchema(BaseModel):
    """Metadata for an individual extracted video frame.

    Conventions:
    - timestamp_raw_ns: 64-bit integer timestamp in device clock domain.
    - timestamp_relative_s: local floating-point seconds relative to session start t0.
    """

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    frame_id: str = Field(..., description="Unique identifier for the frame (e.g. frame_session_000123)")
    session_id: str = Field(..., description="Associated session identifier")
    index: int = Field(..., ge=0, description="Sequential frame index within the stream")
    camera_id: str = Field(..., description="Identifier of camera or rig sensor (e.g. 'cam0', 'cam1')")
    timestamp_raw_ns: int = Field(..., description="Integer raw device timestamp in nanoseconds")
    timestamp_relative_s: float = Field(
        ...,
        description="Local relative time in seconds from session start t0 (for numerical calculation)",
    )
    image_path: str = Field(..., description="Relative path to stored frame image artifact")
    width: int = Field(..., gt=0, description="Frame width in pixels")
    height: int = Field(..., gt=0, description="Frame height in pixels")
    asset_id: str = Field(..., description="Provenance reference to parent InputAsset")
    exposure_time_s: Optional[float] = Field(default=None, description="Exposure duration in seconds if known")


class ClockMappingSchema(BaseModel):
    """Clock mapping definition for mapping device raw integer clocks to relative seconds.

    Rule: Raw timestamps are large integers (nanoseconds); numerical work uses local relative seconds.
    """

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    session_id: str = Field(..., description="Associated session identifier")
    clock_source: str = Field(..., description="Source clock descriptor (e.g. 'insta360_sensor_ticks_ns')")
    t0_raw_ns: int = Field(..., description="Integer raw reference timestamp corresponding to relative t=0.0s")
    time_scale_factor: float = Field(
        default=1e-9,
        description="Conversion factor: seconds_per_tick (default 1e-9 for nanosecond ticks)",
    )
    sync_anchors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Checkpoints or anchors mapping raw timestamps to external time bases",
    )

    def raw_to_relative_s(self, raw_ns: int) -> float:
        """Convert a raw integer timestamp to local relative seconds."""
        return float(raw_ns - self.t0_raw_ns) * self.time_scale_factor

    def relative_to_raw_ns(self, rel_s: float) -> int:
        """Convert local relative seconds back to an integer raw timestamp."""
        return int(round(self.t0_raw_ns + (rel_s / self.time_scale_factor)))
