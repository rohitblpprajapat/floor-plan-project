"""Schemas for surfaces, persistent image annotations, and spatial measurements.

Conventions:
- Every surface, depth, and mesh carries map_id, units, and calibration_provenance.
- Annotations maintain persistent source-image observations for 2D/3D ray re-projection.
- Measurements explicitly declare physical units (meters, degrees, square_meters).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from vscan.schemas.common import SCHEMA_VERSION, Units


class SurfaceSchema(BaseModel):
    """Fitted geometric surface (floor plane, ceiling, wall, bounding mesh)."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    surface_id: str = Field(..., description="Unique surface identifier")
    map_id: str = Field(..., description="Map ID this surface belongs to")
    units: Units = Field(..., description="Physical units (METERS or ARBITRARY)")
    calibration_provenance: str = Field(..., description="Calibration provenance used")
    surface_type: str = Field(..., description="'floor', 'ceiling', 'wall', 'generic_plane', 'mesh'")
    plane_equation_abcd: Optional[List[float]] = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Normalized plane equation [a, b, c, d] where ax + by + cz + d = 0",
    )
    polygon_vertices_xyz: Optional[List[List[float]]] = Field(
        default=None,
        description="Boundary vertices in map coordinate frame",
    )
    mesh_path: Optional[str] = Field(default=None, description="Path to mesh artifact (PLY/OBJ) if applicable")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Inlier ratio or fitting confidence")


class AnnotationObservation(BaseModel):
    """Observation of an annotation feature on an individual source image."""

    frame_id: str = Field(..., description="Source frame identifier")
    camera_id: str = Field(..., description="Camera identifier within the rig")
    pixel_coords: List[float] = Field(..., min_length=2, max_length=2, description="2D pixel location [u, v]")
    normalized_ray: Optional[List[float]] = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="Unit bearing vector in camera optical frame",
    )


class AnnotationSchema(BaseModel):
    """User or machine annotation tied to 3D map points and 2D source-image observations."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    annotation_id: str = Field(..., description="Unique annotation identifier")
    map_id: str = Field(..., description="Map version ID")
    session_id: str = Field(..., description="Session identifier")
    label: str = Field(..., description="Human-readable label (e.g. 'Front Door', 'Kitchen Outlet')")
    category: str = Field(..., description="Categorical tag: 'door', 'window', 'wall', 'furniture', 'outlet', 'custom'")
    source_observations: List[AnnotationObservation] = Field(
        default_factory=list,
        description="Persistent 2D source-image observations",
    )
    anchor_xyz_3d: Optional[List[float]] = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="3D anchored point in map coordinate frame",
    )
    units: Units = Field(default=Units.METERS, description="Units for 3D anchor position")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom properties or tags")
    created_at_utc: str = Field(..., description="UTC ISO-8601 timestamp")


class MeasurementSchema(BaseModel):
    """Derived physical measurement (distance, area, angle) derived from annotations or surfaces."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    measurement_id: str = Field(..., description="Unique measurement identifier")
    map_id: str = Field(..., description="Map version ID")
    session_id: str = Field(..., description="Session identifier")
    measurement_type: str = Field(
        ...,
        description="'point_to_point_distance', 'height_clearance', 'wall_to_wall', 'polygon_area', 'angle'",
    )
    source_annotation_ids: List[str] = Field(
        default_factory=list,
        description="IDs of annotations anchoring this measurement",
    )
    start_xyz: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    end_xyz: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    value: float = Field(..., description="Numerical measured value")
    units: Units = Field(..., description="Measurement units (METERS, DEGREES, SQUARE_METERS)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at_utc: str = Field(..., description="UTC ISO-8601 timestamp")


class DepthMapSchema(BaseModel):
    """Dense or sparse depth map representation.

    Conventions:
    - Every pose/depth/mesh carries map_id, units, and calibration_provenance.
    - Units must be explicit (METERS or ARBITRARY).
    """

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    depth_id: str = Field(..., description="Unique depth map identifier")
    map_id: str = Field(..., description="Reconstruction map ID this depth estimate belongs to")
    units: Units = Field(..., description="Physical units of depth values (METERS or ARBITRARY)")
    calibration_provenance: str = Field(..., description="Provenance of calibration used for depth estimation")
    frame_id: str = Field(..., description="Associated source frame identifier")
    camera_id: str = Field(..., description="Camera identifier within sensor rig")
    width: int = Field(..., gt=0, description="Depth map width in pixels")
    height: int = Field(..., gt=0, description="Depth map height in pixels")
    file_path: str = Field(..., description="Relative path to stored depth array artifact (.npy, .exr, .png)")
    min_depth: Optional[float] = Field(default=None, description="Minimum valid depth value in declared units")
    max_depth: Optional[float] = Field(default=None, description="Maximum valid depth value in declared units")
    scale_factor: float = Field(default=1.0, description="Multiplication factor to convert raw array values to declared units")
    timestamp_raw_ns: Optional[int] = Field(default=None, description="Raw integer nanosecond timestamp")
    timestamp_relative_s: Optional[float] = Field(default=None, description="Local relative seconds from t0")


class MeshSchema(BaseModel):
    """Reconstructed 3D surface mesh or geometric polygon representation.

    Conventions:
    - Every pose/depth/mesh carries map_id, units, and calibration_provenance.
    """

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    mesh_id: str = Field(..., description="Unique mesh artifact identifier")
    map_id: str = Field(..., description="Reconstruction map ID this mesh belongs to")
    units: Units = Field(..., description="Physical units of vertex coordinates (METERS or ARBITRARY)")
    calibration_provenance: str = Field(..., description="Provenance of sensor calibration and reconstruction pipeline")
    mesh_type: str = Field(default="triangular_mesh", description="'triangular_mesh', 'point_cloud', 'quad_mesh', 'voxel_grid'")
    file_path: str = Field(..., description="Relative path to mesh artifact file (.ply, .obj, .gltf)")
    vertex_count: int = Field(..., ge=0, description="Total number of 3D vertices")
    face_count: Optional[int] = Field(default=None, ge=0, description="Total number of polygon faces if applicable")
    format: str = Field(default="ply", description="File format encoding ('ply', 'obj', 'gltf')")
    bounding_box_min: Optional[List[float]] = Field(default=None, min_length=3, max_length=3, description="Axis-aligned bounding box min [x, y, z]")
    bounding_box_max: Optional[List[float]] = Field(default=None, min_length=3, max_length=3, description="Axis-aligned bounding box max [x, y, z]")
    exported_at_utc: str = Field(..., description="UTC ISO-8601 export timestamp")
