"""Schemas for perspective frames, virtual cameras, masks, quality scores, and pilot manifest.

Conventions enforced:
1. Virtual camera coordinate frame follows OpenCV: +X right, +Y down, +Z forward.
2. Pixel center convention is explicitly documented: half-integer offset (u + 0.5, v + 0.5).
3. All virtual views of a single panorama have identical camera centers and zero synthetic baseline.
4. Mask coordinates are stored in their original projection with full provenance.
5. All scale status remains strictly NOT_ESTIMATED.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import numpy as np

from vscan.schemas.common import SCHEMA_VERSION, PoseMatrix


class ReprojectionFace(str, Enum):
    """Names of the standard 6 cubemap perspective faces."""

    FRONT = "front"
    RIGHT = "right"
    BACK = "back"
    LEFT = "left"
    UP = "up"
    DOWN = "down"


class PinholeIntrinsics(BaseModel):
    """Pinhole camera intrinsic parameters with documented pixel-center convention."""

    camera_name: str = Field(..., description="Unique camera identifier (e.g. 'face_front')")
    width: int = Field(..., gt=0, description="Image sensor width in pixels")
    height: int = Field(..., gt=0, description="Image sensor height in pixels")
    fov_deg: float = Field(..., gt=0, lt=180, description="Horizontal and vertical field of view in degrees")
    fx: float = Field(..., gt=0, description="Focal length in X (pixels)")
    fy: float = Field(..., gt=0, description="Focal length in Y (pixels)")
    cx: float = Field(..., description="Principal point X coordinate in pixels")
    cy: float = Field(..., description="Principal point Y coordinate in pixels")
    pixel_center_convention: str = Field(
        default="half_integer_offset_center_at_x_plus_0.5_y_plus_0.5",
        description="Explicit pixel-center convention documentation",
    )

    @property
    def K(self) -> np.ndarray:
        """Return 3x3 intrinsic calibration matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)


class VirtualCameraPose(BaseModel):
    """Extrinsic pose and PyCOLMAP rig parameters for a virtual perspective camera."""

    camera_name: str = Field(..., description="Camera identifier (e.g. 'face_front')")
    face_name: ReprojectionFace = Field(..., description="Cubemap face name")
    face_index: int = Field(..., ge=0, le=5, description="Index 0 to 5 of the camera in rig")
    R_pano_cam: List[float] = Field(
        ...,
        min_length=9,
        max_length=9,
        description="3x3 rotation matrix from camera optical frame to panorama frame (row-major)",
    )
    t_pano_cam: List[float] = Field(
        default=[0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
        description="3D translation vector [tx, ty, tz] in panorama frame (strictly [0, 0, 0])",
    )
    cam_from_rig_rotation_wxyz: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Unit quaternion [w, x, y, z] for PyCOLMAP RigConfig cam_from_rig rotation",
    )
    cam_from_rig_translation: List[float] = Field(
        default=[0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
        description="Translation vector [tx, ty, tz] for PyCOLMAP RigConfig (strictly [0, 0, 0])",
    )

    @field_validator("R_pano_cam")
    @classmethod
    def validate_so3(cls, v: List[float]) -> List[float]:
        r = np.array(v, dtype=np.float64).reshape((3, 3))
        det = float(np.linalg.det(r))
        if abs(det - 1.0) > 1e-3:
            raise ValueError(f"R_pano_cam is not proper SO(3): det(R) = {det:.6f} != 1.0")
        diff = np.dot(r.T, r) - np.eye(3, dtype=np.float64)
        ortho_err = float(np.max(np.abs(diff)))
        if ortho_err > 1e-3:
            raise ValueError(f"R_pano_cam is not orthogonal: max |R^T * R - I| = {ortho_err:.6f} > 1e-3")
        return [float(x) for x in v]

    @field_validator("t_pano_cam", "cam_from_rig_translation")
    @classmethod
    def validate_zero_baseline(cls, v: List[float]) -> List[float]:
        if any(abs(x) > 1e-6 for x in v):
            raise ValueError(f"Virtual camera must have zero synthetic baseline, got {v}")
        return [0.0, 0.0, 0.0]


class MaskDefinition(BaseModel):
    """Specification of an explicit spatial mask with origin and provenance."""

    mask_id: str = Field(..., description="Unique mask identifier")
    mask_type: str = Field(..., description="Classification: hardware_mount, operator, moving_people, reflections_seams, synthetic_fixture")
    provenance: str = Field(..., description="Method or source: nadir_cone, inspected_polygon, supplied_polygon, synthetic_analytic")
    original_projection: str = Field(default="equirectangular_normalized", description="Projection system in which coordinates are defined")
    polygon_coordinates: List[List[float]] = Field(..., description="List of [x, y] coordinates in original projection (normalized [0, 1])")
    is_seam_crossing: bool = Field(default=False, description="True if polygon spans across the horizontal seam (x=0 / x=1)")
    description: str = Field(default="", description="Descriptive context for mask purpose")


class FrameQualityScore(BaseModel):
    """Quality and selection metrics for a panorama and its perspective faces."""

    sharpness_score: float = Field(..., description="Laplacian variance / Tenengrad sharpness metric (higher is sharper)")
    exposure_mean: float = Field(..., description="Mean pixel luminance [0..255]")
    clipped_dark_fraction: float = Field(..., description="Fraction of underexposed pixels (< 5)")
    clipped_bright_fraction: float = Field(..., description="Fraction of saturated/overexposed pixels (> 250)")
    exposure_score: float = Field(..., description="Normalized exposure quality score [0..1]")
    inter_frame_change: float = Field(default=0.0, description="Motion / difference from previous selected panorama")
    composite_quality: float = Field(..., description="Composite ranking score balancing sharpness, exposure, and distinctness")


class VirtualFrameView(BaseModel):
    """A single reprojected perspective view corresponding to one face of a panorama."""

    view_id: str = Field(..., description="Unique view ID (e.g. 'frame_000010_face_front')")
    panorama_index: int = Field(..., ge=0, description="Source panorama frame index")
    camera_name: str = Field(..., description="Camera identifier (e.g. 'face_front')")
    face_name: ReprojectionFace = Field(..., description="Face name")
    image_rel_path: str = Field(..., description="Relative path to perspective image file")
    mask_rel_path: Optional[str] = Field(default=None, description="Relative path to perspective mask file if generated")
    image_sha256: str = Field(..., description="SHA-256 hash of perspective image")
    mask_sha256: Optional[str] = Field(default=None, description="SHA-256 hash of perspective mask")
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    fov_deg: float = Field(..., gt=0)
    fx: float = Field(..., gt=0)
    fy: float = Field(..., gt=0)
    cx: float = Field(...)
    cy: float = Field(...)
    R_pano_cam: List[float] = Field(..., min_length=9, max_length=9)
    t_pano_cam: List[float] = Field(default=[0.0, 0.0, 0.0], min_length=3, max_length=3)


class PilotPanoramaFrame(BaseModel):
    """Metadata for a selected pilot panorama and its set of virtual perspective views."""

    panorama_index: int = Field(..., ge=0, description="Source decoded video frame index")
    source_file: str = Field(..., description="Filename of source video")
    source_file_sha256: str = Field(..., description="SHA-256 checksum of source video")
    source_stream_index: int = Field(default=0, ge=0, description="Video stream index within source container")
    source_pts: int = Field(..., description="Original presentation timestamp (PTS) from video stream")
    pts_seconds: float = Field(..., description="PTS converted to presentation seconds")
    capture_time_status: str = Field(
        ...,
        description="Timelapse or exposure status: 'timelapse_unverified_cadence', 'realtime_continuous', 'synthetic_analytic'",
    )
    export_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolution, codec, and vendor export properties",
    )
    stabilization_status: str = Field(
        ...,
        description="Camera stabilization status: 'STABILIZATION_UNKNOWN_TIME_DEPENDENT', 'STABILIZATION_VERIFIED_RIGID', 'SYNTHETIC_KNOWN_WORLD'",
    )
    image_orientation_relative_to_camera: str = Field(
        default="unknown_time_dependent_stabilization",
        description="Documentation of physical camera to image orientation transform",
    )
    quality_scores: FrameQualityScore = Field(..., description="Evaluated sharpness, exposure, and inter-frame motion scores")
    views: Dict[str, VirtualFrameView] = Field(
        default_factory=dict,
        description="Dictionary mapping camera_name to VirtualFrameView",
    )


class PilotManifestSchema(BaseModel):
    """Comprehensive manifest of selected pilot frames, virtual camera rig, and masks."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    session_id: str = Field(..., description="Target recording session identifier")
    created_at_utc: str = Field(..., description="Creation timestamp in UTC ISO-8601")
    config_hash: str = Field(..., description="Deterministic hash of configuration used to generate frames")
    source_asset_id: str = Field(..., description="Asset ID of source media")
    source_projection: str = Field(..., description="Source projection model: equirectangular_360, dual_circular_fisheye_raw, synthetic_analytic")
    projection_notes: str = Field(default="", description="Documentation of projection properties and approximations")
    total_source_frames_inspected: int = Field(..., ge=0, description="Total source video frames probed/inspected")
    selected_pilot_frames_count: int = Field(..., ge=0, description="Count of retained contiguous pilot panoramas (up to 40)")
    pilot_panoramas: List[PilotPanoramaFrame] = Field(default_factory=list, description="Ordered list of selected pilot frames")
    camera_intrinsics: Dict[str, PinholeIntrinsics] = Field(default_factory=dict, description="Intrinsics for each virtual camera in rig")
    virtual_camera_poses: Dict[str, VirtualCameraPose] = Field(default_factory=dict, description="Extrinsics and rig definitions")
    masks: List[MaskDefinition] = Field(default_factory=list, description="Active masks and polygon provenance")
    rig_config_path: str = Field(..., description="Relative path to PyCOLMAP rig_config.json")
    contact_sheet_path: Optional[str] = Field(default=None, description="Relative path to pilot frame contact sheet")
    perspective_contact_sheet_path: Optional[str] = Field(default=None, description="Relative path to perspective face contact sheet")
    scale_status: str = Field(default="NOT_ESTIMATED", description="Authoritative scale status: strictly NOT_ESTIMATED")
    actual_data_readiness: str = Field(
        ...,
        description="Status: 'PROVISIONAL_PANORAMIC_BASELINE', 'SYNTHETIC_ONLY', 'VERIFIED_CALIBRATED_BASELINE'",
    )
