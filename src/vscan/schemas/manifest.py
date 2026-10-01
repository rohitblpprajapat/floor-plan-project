"""Schemas for input manifests, sensor summaries, video probes, and validity flags.

Contracts enforced:
1. Strict separation of software status, actual-data status, and scale status.
2. All scale status remains NOT_ESTIMATED until an explicit metric baseline is verified.
3. Raw timestamps stored strictly as 64-bit integers.
4. Sensor units never silently mixed; original units preserved separately from proposed hypotheses.
5. Phone data is diagnostic-only unless a rigid mount is established.
6. Constant GPS fixes must not be expanded into independent position constraints.
7. Timelapse exports do not imply real exposure cadence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from vscan.schemas.assets import InputAssetSchema
from vscan.schemas.common import SCHEMA_VERSION


class SourceDerivedRelationshipSchema(BaseModel):
    """Explicit provenance link between original-source input and derived export."""

    source_asset_id: str = Field(..., description="Unique ID of source asset")
    derived_asset_id: str = Field(..., description="Unique ID of derived asset")
    relationship_type: str = Field(..., description="Relationship classification (e.g. vendor_export, preview_stream, imu_export)")
    evidence: str = Field(..., description="Objective metadata or container evidence supporting the relationship")


class ValidityFlagSchema(BaseModel):
    """Granular validity and diagnostic status for an individual stream modality."""

    status: str = Field(..., description="Status flag: VALID, PROVISIONAL, DEGRADED, INVALID, DIAGNOSTIC_ONLY, UNVERIFIED_MOUNT, CONSTANT_SINGLE_FIX, NOISY_MULTIPATH")
    unit_hypothesis: Optional[str] = Field(default=None, description="Proposed unit hypothesis (e.g. 'g', 'm/s^2', 'rad/s', 'microseconds')")
    diagnostic_only: bool = Field(default=False, description="True if stream is strictly diagnostic and excluded from metric optimization")
    notes: str = Field(default="", description="Detailed factual notes regarding stream condition or anomalies")


class VideoTrackProbeSchema(BaseModel):
    """Detailed stream metadata for a single video/audio/metadata track."""

    track_index: int = Field(..., description="0-indexed track position in container")
    track_type: str = Field(..., description="Track media type: video, audio, metadata, hint")
    codec: Optional[str] = Field(default=None, description="Short codec FourCC / name (e.g. h264, hevc, aac)")
    codec_long_name: Optional[str] = Field(default=None, description="Descriptive codec identifier")
    width: Optional[int] = Field(default=None, description="Pixel width")
    height: Optional[int] = Field(default=None, description="Pixel height")
    aspect_ratio: Optional[str] = Field(default=None, description="Display or sample aspect ratio string (e.g. '2:1', '16:9')")
    projection_candidate: str = Field(..., description="Candidate projection: equirectangular_360, dual_fisheye, reframed_perspective, preview_equirectangular, unknown")
    time_base: Optional[str] = Field(default=None, description="Track time base string (e.g. '1/90000')")
    r_frame_rate: Optional[str] = Field(default=None, description="Real base frame rate")
    avg_frame_rate: Optional[str] = Field(default=None, description="Average computed frame rate")
    duration_seconds: Optional[float] = Field(default=None, description="Track duration in seconds")
    frame_count: Optional[int] = Field(default=None, description="Total packet/sample count")
    is_variable_cadence: bool = Field(default=False, description="True if packet timestamp intervals vary significantly")
    cadence_summary: Dict[str, Any] = Field(default_factory=dict, description="Distribution of timestamp intervals")
    tags: Dict[str, str] = Field(default_factory=dict, description="Metadata tags extracted from track or container")
    exposure_duration_s: Optional[float] = Field(default=None, description="Exposure duration in seconds if explicitly recorded")
    rolling_shutter_readout_s: Optional[float] = Field(default=None, description="Rolling shutter readout duration if recorded")
    unknown_fields: List[str] = Field(default_factory=list, description="Explicit list of unavailable/unknown hardware metadata fields")


class VideoProbeSummarySchema(BaseModel):
    """Container-level probe summary and frame inspection results."""

    asset_id: str = Field(..., description="Target asset ID")
    file_path: str = Field(..., description="Path to video file")
    file_size_bytes: int = Field(..., description="File size in bytes")
    format_name: Optional[str] = Field(default=None, description="Container format name (e.g. 'mov,mp4,m4a,3gp,3g2,mj2')")
    container_duration_s: Optional[float] = Field(default=None, description="Container duration in seconds")
    bit_rate: Optional[int] = Field(default=None, description="Bitrate in bits/second")
    tracks: List[VideoTrackProbeSchema] = Field(default_factory=list, description="Probed media tracks")
    layout_verification: Dict[str, Any] = Field(default_factory=dict, description="Physical image layout and geometry analysis")
    contact_sheet_path: Optional[str] = Field(default=None, description="Relative path to generated contact sheet image")
    decoded_frames: List[Dict[str, Any]] = Field(default_factory=list, description="Decoded start, middle, and end frame metadata in presentation order")


class SensorSummarySchema(BaseModel):
    """Statistical and structural summary of a parsed sensor CSV stream."""

    sensor_type: str = Field(..., description="Sensor descriptor (e.g. camera_imu, phone_sensors)")
    source_file: str = Field(..., description="Filename of source sensor file")
    row_count: int = Field(..., description="Total data rows parsed (excluding header)")
    timestamp_column: str = Field(..., description="Name of raw integer timestamp column")
    raw_t_min: int = Field(..., description="Earliest raw integer timestamp")
    raw_t_max: int = Field(..., description="Latest raw integer timestamp")
    raw_t_span: int = Field(..., description="Span of raw integer timestamps (raw_t_max - raw_t_min)")
    span_seconds_proposed: Optional[float] = Field(default=None, description="Span expressed in seconds under proposed hypothesis")
    unit_hypothesis: str = Field(..., description="Declared unit hypothesis (e.g. 'g / microseconds', 'm/s^2 / milliseconds')")
    is_monotonic: bool = Field(..., description="True if raw integer timestamps are strictly monotonically increasing")
    duplicate_timestamps_count: int = Field(default=0, description="Count of duplicate successive timestamps")
    backward_timestamps_count: int = Field(default=0, description="Count of backward timestamp steps")
    interval_stats: Dict[str, float] = Field(default_factory=dict, description="Timestamp interval statistics (min, max, mean, median, std)")
    short_intervals_count: int = Field(default=0, description="Count of intervals significantly shorter than nominal")
    short_intervals_pattern: str = Field(default="none", description="Analysis of short interval periodicity or alignment")
    gaps_count: int = Field(default=0, description="Count of intervals significantly larger than nominal")
    initial_zero_vectors_count: int = Field(default=0, description="Count of leading rows with all-zero sensor vectors")
    channel_norms: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Channel vector norms (mean, min, max, std)")
    repeated_channels_count: int = Field(default=0, description="Count of rows with repeated channel values")
    gps_analysis: Dict[str, Any] = Field(default_factory=dict, description="GPS fix analysis (fix count, constancy, bounding box)")
    step_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Pedometer step count analysis if present")


class LocationMetadataSchema(BaseModel):
    """Metadata preserved from session location registration JSON."""

    session_label: str = Field(..., description="Session label string")
    recording_type: str = Field(..., description="Declared recording type (e.g. 'timelapse')")
    recording_started_at: str = Field(..., description="ISO UTC start timestamp")
    building: Dict[str, Any] = Field(default_factory=dict, description="Preserved building metadata")
    floor: Dict[str, Any] = Field(default_factory=dict, description="Preserved floor metadata")
    start_coordinates: Dict[str, Any] = Field(default_factory=dict, description="Preserved start coordinates")
    pin: Dict[str, Any] = Field(default_factory=dict, description="Preserved start pin metadata")
    is_timelapse: bool = Field(default=False, description="True if recording type is timelapse")
    timelapse_warning: str = Field(default="", description="Factual note on timelapse implications for exposure/cadence")
    metric_scale_provided: bool = Field(default=False, description="False: worldX/Y=0 and normalized pins provide no metric scale")
    full_map_registration_provided: bool = Field(default=False, description="False: single pin provides no orientation or map registration")


class InputManifestSchema(BaseModel):
    """Top-level immutable session input manifest."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="Data contract schema version")
    session_id: str = Field(..., description="Target recording session ID")
    created_at_utc: str = Field(..., description="UTC ISO timestamp of manifest generation")
    input_directory: str = Field(..., description="Source path where inputs were resolved")
    storage_mode: str = Field(..., description="Storage policy: reference or copy")
    storage_budget_bytes: Optional[int] = Field(default=None, description="Configured storage budget limit")
    total_input_bytes: int = Field(..., description="Sum of all input file sizes in bytes")
    assets: List[InputAssetSchema] = Field(default_factory=list, description="Registered immutable input assets")
    source_derived_relationships: List[SourceDerivedRelationshipSchema] = Field(
        default_factory=list, description="Documented source-to-derived links"
    )
    validity_flags: Dict[str, ValidityFlagSchema] = Field(
        default_factory=dict, description="Separated validity flags for imagery, timing, camera_imu, phone_imu, gps"
    )
    scale_status: str = Field(default="NOT_ESTIMATED", description="Authoritative metric scale status: strictly NOT_ESTIMATED")
    unresolved_facts: List[str] = Field(default_factory=list, description="List of unresolved hardware or environmental facts")
    missing_files: List[str] = Field(default_factory=list, description="List of expected input files that were missing")
    status: str = Field(..., description="Overall ingestion status: COMPLETE, WAITING_INPUT, DEGRADED")
    prompt_05_readiness: Dict[str, Any] = Field(
        default_factory=dict, description="Evaluation of Prompt 05 readiness (actual imagery usable vs needs export vs fixtures)"
    )
