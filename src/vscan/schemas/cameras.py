"""Schemas for camera models, multi-camera rigs, and calibration provenance.

Conventions:
- Camera coordinate frame follows OpenCV: +X right, +Y down, +Z forward (depth into scene).
- Rig extrinsics: T_rig_cam transforms points in camera frame into rig body frame.
- Matrices use explicit row-major serialization.
- Provenance is explicitly tracked.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from vscan.schemas.common import SCHEMA_VERSION, CoordinateFrame, PoseMatrix


class CameraModel(BaseModel):
    """Camera intrinsic geometry and extrinsic pose relative to sensor rig body."""

    camera_id: str = Field(..., description="Unique camera identifier within rig (e.g. 'cam0', 'cam1')")
    model_name: str = Field(
        ...,
        description="Camera projection model name (e.g. 'OPENCV_FISHEYE', 'PINHOLE', 'RADIAL')",
    )
    width: int = Field(..., gt=0, description="Image sensor width in pixels")
    height: int = Field(..., gt=0, description="Image sensor height in pixels")
    params: List[float] = Field(..., description="Intrinsic projection and distortion parameters")
    T_rig_cam: PoseMatrix = Field(
        ...,
        description="Extrinsic transformation from camera frame to rig body frame: p_rig = T_rig_cam * p_cam",
    )
    camera_axes_convention: str = Field(
        default="OPENCV (+X right, +Y down, +Z forward)",
        description="Standard optical axis convention statement",
    )

    @field_validator("T_rig_cam")
    @classmethod
    def validate_rig_cam_frames(cls, v: PoseMatrix) -> PoseMatrix:
        if v.to_frame != CoordinateFrame.RIG_BODY.value and v.to_frame != "rig":
            # Warn/allow if standard rig frame naming
            pass
        return v


class CameraRigSchema(BaseModel):
    """Specification of a multi-camera rigid sensor rig (e.g. dual fisheye Insta360)."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    rig_id: str = Field(..., description="Unique rig configuration ID")
    session_id: str = Field(..., description="Associated session identifier")
    cameras: List[CameraModel] = Field(..., min_length=1, description="List of synchronized cameras on the rig")
    calibration_provenance: str = Field(
        ...,
        description="Source and method of calibration (e.g. 'factory_metadata', 'offline_bundle_adjustment')",
    )


class CalibrationSchema(BaseModel):
    """Complete sensor suite calibration including camera intrinsics/extrinsics and IMU parameters."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema specification version")
    calibration_id: str = Field(..., description="Unique calibration record ID")
    session_id: str = Field(..., description="Associated session identifier")
    provenance: str = Field(..., description="Detailed provenance description")
    rig: CameraRigSchema = Field(..., description="Rig camera definitions and relative poses")
    T_rig_imu: Optional[PoseMatrix] = Field(
        default=None,
        description="IMU extrinsic transformation into rig body frame: p_rig = T_rig_imu * p_imu",
    )
    accelerometer_bias: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    gyroscope_bias: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    gravity_magnitude_m_s2: float = Field(default=9.80665, description="Local gravity constant in m/s^2")
