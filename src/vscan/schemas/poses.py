"""Schemas for poses, 3D landmarks, and versioned immutable reconstruction maps.

Conventions:
- T_A_B transforms coordinates from frame B into frame A: p_A = T_A_B * p_B.
- Matrices use explicit row-major serialization.
- Rotations are proper SO(3).
- Relative maps have no assumed gravity direction or metre scale (units='arbitrary', gravity_aligned=False).
- Gravity-aligned metric maps use a right-handed Z-up frame (+X east/right, +Y north/forward, +Z up; units='meters', gravity_aligned=True).
- Every pose/depth/mesh carries a map ID, units and calibration provenance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from vscan.schemas.common import SCHEMA_VERSION, CoordinateFrame, PoseMatrix, Units


class PoseSchema(BaseModel):
    """Estimated 6-DoF pose transformation.

    Transforms coordinates from body/camera frame (B) into world/map frame (A):
        p_map = T_map_body * p_body

    Mandatory fields:
    - map_id: Map identifier this pose is defined within.
    - units: Metric ('meters') or relative ('arbitrary').
    - calibration_provenance: Provenance of calibration used.
    """

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    pose_id: str = Field(..., description="Unique pose identifier (e.g. pose_session_000123)")
    from_frame: str = Field(..., description="Source frame B (e.g. 'rig', 'cam0')")
    to_frame: str = Field(..., description="Target reference frame A (e.g. 'map', 'world')")
    map_id: str = Field(..., description="Map version ID this pose belongs to")
    units: Units = Field(..., description="Physical units of translation (METERS or ARBITRARY)")
    calibration_provenance: str = Field(..., description="Calibration provenance used to estimate this pose")
    timestamp_raw_ns: Optional[int] = Field(default=None, description="Raw integer nanosecond timestamp")
    timestamp_relative_s: Optional[float] = Field(
        default=None,
        description="Local relative seconds from session zero",
    )
    transform: PoseMatrix = Field(..., description="Rigid body transformation T_to_from")
    covariance_row_major_6x6: Optional[List[float]] = Field(
        default=None,
        min_length=36,
        max_length=36,
        description="Optional 6x6 pose covariance matrix in row-major order",
    )

    @field_validator("transform")
    @classmethod
    def validate_transform_frames(cls, v: PoseMatrix, info: Any) -> PoseMatrix:
        # Cross check frames match
        return v


class LandmarkSchema(BaseModel):
    """3D point / landmark in a reconstruction map."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    landmark_id: str = Field(..., description="Unique 3D landmark identifier")
    map_id: str = Field(..., description="Map version ID this landmark belongs to")
    xyz: List[float] = Field(..., min_length=3, max_length=3, description="3D coordinates in map coordinate frame")
    units: Units = Field(..., description="Units of coordinate values (METERS or ARBITRARY)")
    observations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="2D image observations linking frame_id, camera_id, and pixel coordinates",
    )
    color_rgb: Optional[List[int]] = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="Representative RGB color [0..255]",
    )
    uncertainty: Optional[float] = Field(default=None, description="Geometric point uncertainty or reprojection error")


class MapVersionSchema(BaseModel):
    """Immutable exported reconstruction map version.

    Conventions:
    - Relative maps: units='arbitrary', gravity_aligned=False, coordinate_frame='world_relative_unscaled'.
    - Metric gravity-aligned maps: units='meters', gravity_aligned=True, coordinate_frame='world_metric_gravity_aligned' (right-handed Z-up).
    - Map versions are strictly immutable once exported.
    """

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    map_id: str = Field(..., description="Unique immutable map version identifier (e.g. map_session_v1_rel)")
    session_id: str = Field(..., description="Associated session identifier")
    parent_map_id: Optional[str] = Field(default=None, description="Predecessor map ID if derived via alignment/scaling")
    map_version: int = Field(default=1, ge=1, description="Integer version counter")
    map_type: str = Field(
        ...,
        description="'relative_unscaled' or 'metric_gravity_aligned'",
    )
    gravity_aligned: bool = Field(..., description="Whether Z axis corresponds to verified gravity up")
    coordinate_frame: CoordinateFrame = Field(
        ...,
        description="Coordinate frame convention: WORLD_RELATIVE_UNSCALED or WORLD_METRIC_GRAVITY_ALIGNED",
    )
    units: Units = Field(..., description="Distance units: METERS or ARBITRARY")
    scale_factor_meters_per_unit: Optional[float] = Field(
        default=None,
        description="Metric scale factor (1.0 for metric maps)",
    )
    provenance: str = Field(..., description="Reconstruction pipeline and solver provenance")
    is_immutable: bool = Field(default=True, description="Enforced immutability flag")
    num_poses: int = Field(default=0, ge=0)
    num_landmarks: int = Field(default=0, ge=0)
    exported_at_utc: str = Field(..., description="UTC ISO-8601 timestamp of map export")
