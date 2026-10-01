"""Schemas for Structure from Motion (SfM) relative reconstruction maps, tracks, and poses.

Strict contracts enforced:
1. Metric scale is strictly scale='NOT_ESTIMATED'.
2. Camera-to-world (world_from_cam) and world-to-camera transforms are explicitly exported.
3. Observed 2D tracks, landmark IDs, visibility and pixel coordinates are fully preserved.
4. Physical panorama stations and virtual views are tracked and counted separately.
5. Multiple disconnected reconstruction components are retained and explicitly identified.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator
from vscan.schemas.common import SCHEMA_VERSION, Units


class SfmTrackObservationSchema(BaseModel):
    """2D image observation belonging to a 3D landmark track."""

    image_id: int = Field(..., description="PyCOLMAP integer image ID")
    point2d_idx: int = Field(..., description="Index of 2D point feature in image keypoints")
    xy_pixel: List[float] = Field(..., min_length=2, max_length=2, description="[x, y] subpixel measurement in image pixels")
    view_id: Optional[str] = Field(default=None, description="Descriptive view identifier (e.g. frame_000066_face_front)")
    camera_name: Optional[str] = Field(default=None, description="Rig camera name (e.g. face_front)")
    panorama_index: Optional[int] = Field(default=None, description="Physical panorama station index")


class SfmLandmarkSchema(BaseModel):
    """3D triangulated landmark point with track observations."""

    point3d_id: int = Field(..., description="PyCOLMAP 3D point landmark ID")
    xyz: List[float] = Field(..., min_length=3, max_length=3, description="[X, Y, Z] 3D coordinates in relative unscaled map frame")
    units: str = Field(default="ARBITRARY", description="Coordinate units: strictly ARBITRARY (relative unscaled)")
    color_rgb: List[int] = Field(default=[128, 128, 128], min_length=3, max_length=3, description="[R, G, B] color 0-255")
    reprojection_error_px: float = Field(..., description="Mean reprojection residual in pixels")
    track_length: int = Field(..., ge=2, description="Number of 2D image observations in this track")
    observations: List[SfmTrackObservationSchema] = Field(
        default_factory=list,
        description="List of 2D observations supporting this 3D landmark",
    )


class SfmCameraPoseSchema(BaseModel):
    """Estimated camera pose for a single registered virtual perspective image."""

    image_id: int = Field(..., description="PyCOLMAP image ID")
    image_name: str = Field(..., description="Image filename (e.g. face_front_frame_000066.jpg)")
    view_id: str = Field(..., description="Unique view identifier (e.g. frame_000066_face_front)")
    camera_name: str = Field(..., description="Virtual face camera name (e.g. face_front)")
    camera_id: int = Field(..., description="Associated camera model ID in database")
    frame_id: Optional[int] = Field(default=None, description="Rig station frame ID if configured in rig")
    panorama_index: Optional[int] = Field(default=None, description="Station panorama index")
    is_registered: bool = Field(default=True, description="Whether image was registered in this component")
    
    # World-to-camera transform (COLMAP native: X_c = R * X_w + T)
    cam_from_world_rotation_quaternion_wxyz: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="COLMAP native world-to-camera quaternion [w, x, y, z]",
    )
    cam_from_world_translation: List[float] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="COLMAP native translation vector T_w2c",
    )

    # Camera-to-world transform (Camera optical center and orientation in world)
    world_from_cam_rotation_matrix_3x3: List[List[float]] = Field(
        ...,
        description="Camera-to-world rotation matrix R_c2w = R_w2c^T (3x3 row-major)",
    )
    world_from_cam_translation: List[float] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Camera optical center in world coordinates C = -R_w2c^T * T_w2c",
    )
    units: str = Field(default="ARBITRARY", description="Coordinate units: strictly ARBITRARY")
    
    @field_validator("world_from_cam_rotation_matrix_3x3")
    @classmethod
    def validate_rotation_so3(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("world_from_cam_rotation_matrix_3x3 must be a 3x3 matrix")
        r = np.array(v, dtype=np.float64)
        det = float(np.linalg.det(r))
        if abs(det - 1.0) > 1e-3:
            raise ValueError(f"Rotation matrix is not proper SO(3): det(R) = {det:.6f} != 1.0")
        diff = r.T @ r - np.eye(3)
        if np.max(np.abs(diff)) > 1e-3:
            raise ValueError("Rotation matrix is not orthogonal: R^T * R != I")
        return v


class SfmStationPoseSchema(BaseModel):
    """Estimated physical panorama station pose (rig frame nodal center)."""

    station_index: int = Field(..., description="Physical panorama station index")
    frame_id: Optional[int] = Field(default=None, description="PyCOLMAP rig frame ID")
    ref_image_id: Optional[int] = Field(default=None, description="Reference sensor image ID (face_front)")
    is_registered: bool = Field(default=True, description="Whether station is registered in reconstruction")
    nodal_center_world: List[float] = Field(..., min_length=3, max_length=3, description="Nodal optical center [X, Y, Z] in world coordinates")
    registered_views: List[str] = Field(default_factory=list, description="List of registered view names for this station")


class SfmComponentSchema(BaseModel):
    """Summary of an independent connected reconstruction component."""

    component_id: int = Field(..., description="Integer component identifier (0, 1, ...)")
    is_primary: bool = Field(default=False, description="True if selected as the primary pilot component")
    num_registered_stations: int = Field(..., ge=0, description="Count of physical panorama stations registered")
    num_registered_views: int = Field(..., ge=0, description="Count of virtual perspective images registered")
    num_landmarks: int = Field(..., ge=0, description="Count of 3D point landmarks")
    num_observations: int = Field(..., ge=0, description="Total 2D point observations in tracks")
    mean_track_length: float = Field(..., ge=0.0, description="Mean track length across landmarks")
    median_track_length: float = Field(..., ge=0.0, description="Median track length")
    mean_triangulation_angle_deg: float = Field(default=0.0, ge=0.0, description="Mean two-view triangulation angle in degrees")
    mean_reprojection_error_px: float = Field(..., ge=0.0, description="Mean reprojection residual in pixels")
    median_reprojection_error_px: float = Field(..., ge=0.0, description="Median reprojection residual in pixels")
    rmse_reprojection_error_px: float = Field(..., ge=0.0, description="RMSE reprojection residual in pixels")
    reprojection_residual_percentiles: Dict[str, float] = Field(
        default_factory=dict,
        description="Residual error percentiles: p50, p90, p95, p99, max",
    )
    bounding_box_min: List[float] = Field(default_factory=list, description="[min_x, min_y, min_z]")
    bounding_box_max: List[float] = Field(default_factory=list, description="[max_x, max_y, max_z]")


class RelativeMapManifestSchema(BaseModel):
    """Manifest of an accepted relative SfM reconstruction map."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    session_id: str = Field(..., description="Associated session identifier")
    map_id: str = Field(..., description="Unique relative map identifier (e.g. map_20260910_172939_rel)")
    run_id: str = Field(..., description="SfM job run identifier")
    created_at_utc: str = Field(..., description="UTC ISO-8601 creation timestamp")
    status: str = Field(default="STAGE_06_COMPLETE", description="Software completion status")
    actual_data_status: str = Field(..., description="Status of actual data reconstruction")
    scale_status: str = Field(default="NOT_ESTIMATED", description="Authoritative: strictly NOT_ESTIMATED")

    # Device & options
    device_requested: str = Field(..., description="'cuda' or 'cpu'")
    device_used: str = Field(..., description="'cuda' or 'cpu'")
    gpu_info: Optional[Dict[str, Any]] = Field(default=None, description="GPU hardware details if CUDA was used")
    effective_options: Dict[str, Any] = Field(default_factory=dict, description="Complete logged options used by pipeline")

    # Inputs & provenance
    source_manifest_hash: Optional[str] = Field(default=None, description="SHA-256 of pilot_manifest.json")
    rig_config_hash: Optional[str] = Field(default=None, description="SHA-256 of rig_config.json")
    total_panoramas_input: int = Field(..., ge=0, description="Total input physical panorama frames")
    total_views_input: int = Field(..., ge=0, description="Total input virtual perspective views")

    # Components & registration
    primary_component_id: Optional[int] = Field(default=None, description="ID of primary pilot component")
    components: List[SfmComponentSchema] = Field(default_factory=list, description="List of all disconnected components")

    # Diagnostics
    loop_evidence: Dict[str, Any] = Field(default_factory=dict, description="Revisit candidate and loop verification evidence")
    weak_regions: List[Dict[str, Any]] = Field(default_factory=list, description="Identified weak regions with low parallax or sparse tracks")
    artifact_hashes: Dict[str, str] = Field(default_factory=dict, description="SHA-256 hashes of all exported relative map artifacts")
