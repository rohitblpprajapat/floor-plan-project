"""Common schema conventions, geometry, SO(3) validation, and coordinate frames.

Conventions enforced:
1. T_A_B transforms coordinates from frame B into frame A:
      p_A = T_A_B * p_B = R_A_B * p_B + t_A_B
2. Matrices use explicit row-major serialization.
3. Camera axes follow OpenCV: +X right, +Y down, +Z forward.
4. Rotations are proper SO(3): det(R) = +1, R^T * R = I.
5. Relative maps have no assumed gravity direction or metre scale.
6. Gravity-aligned metric maps use a right-handed Z-up frame (+X east/right, +Y north/forward, +Z up).
7. Every pose/depth/mesh carries a map ID, units and calibration provenance.
8. Use integers for raw large timestamps (nanoseconds) and local relative seconds (float) for numerical work.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Sequence
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


SCHEMA_VERSION = "1.0.0"


class Units(str, Enum):
    """Supported physical and mathematical units."""

    METERS = "meters"
    ARBITRARY = "arbitrary"
    PIXELS = "pixels"
    DEGREES = "degrees"
    RADIANS = "radians"
    SQUARE_METERS = "square_meters"
    SECONDS = "seconds"
    NANOSECONDS = "nanoseconds"


class CoordinateFrame(str, Enum):
    """Standardized coordinate frame conventions."""

    OPENCV_CAMERA = "opencv_camera"  # +X right, +Y down, +Z forward
    RIG_BODY = "rig_body"  # Sensor rig body coordinate frame
    IMU_BODY = "imu_body"  # IMU body coordinate frame
    WORLD_METRIC_GRAVITY_ALIGNED = "world_metric_gravity_aligned"  # Right-handed Z-up (+X east, +Y north, +Z up)
    WORLD_RELATIVE_UNSCALED = "world_relative_unscaled"  # Relative SfM frame (no gravity direction, arbitrary scale)


class PoseMatrix(BaseModel):
    """Rigid body transformation T_A_B that transforms coordinates from frame B to frame A.

    Mathematically:
        p_A = T_A_B * p_B = R_A_B * p_B + t_A_B

    Conventions:
        - rotation_row_major: 9 floats serialized in row-major order:
          [R00, R01, R02, R10, R11, R12, R20, R21, R22]
        - translation: 3 floats [tx, ty, tz]
        - Rotations are proper SO(3): det(R) == +1 within 1e-4 and R^T * R == I within 1e-4.
    """

    from_frame: str = Field(..., description="Source frame B (the frame points are currently expressed in)")
    to_frame: str = Field(..., description="Target frame A (the frame points are transformed into)")
    rotation_row_major: List[float] = Field(
        ...,
        description="Explicit 3x3 rotation matrix serialized in row-major order (9 elements)",
        min_length=9,
        max_length=9,
    )
    translation: List[float] = Field(
        ...,
        description="3D translation vector [tx, ty, tz] in target frame A coordinates",
        min_length=3,
        max_length=3,
    )

    @field_validator("rotation_row_major")
    @classmethod
    def validate_so3(cls, v: List[float]) -> List[float]:
        if len(v) != 9:
            raise ValueError(f"rotation_row_major must have 9 elements, got {len(v)}")
        r = np.array(v, dtype=np.float64).reshape((3, 3))
        det = float(np.linalg.det(r))
        if abs(det - 1.0) > 1e-3:
            raise ValueError(f"Rotation matrix is not proper SO(3): det(R) = {det:.6f} != 1.0")
        diff = np.dot(r.T, r) - np.eye(3, dtype=np.float64)
        ortho_err = float(np.max(np.abs(diff)))
        if ortho_err > 1e-3:
            raise ValueError(f"Rotation matrix is not orthogonal: max |R^T * R - I| = {ortho_err:.6f} > 1e-3")
        return [float(x) for x in v]

    @property
    def R(self) -> np.ndarray:
        """Return 3x3 rotation matrix as numpy array."""
        return np.array(self.rotation_row_major, dtype=np.float64).reshape((3, 3))

    @property
    def t(self) -> np.ndarray:
        """Return 3x1 translation vector as numpy array."""
        return np.array(self.translation, dtype=np.float64)

    @property
    def matrix_4x4_row_major(self) -> List[float]:
        """Return 4x4 homogeneous transformation matrix in row-major order."""
        R = self.R
        t = self.t
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return [float(x) for x in T.flatten()]

    def transform_point(self, point_in_b: Sequence[float]) -> List[float]:
        """Transform a 3D point from frame B to frame A: p_A = R * p_B + t."""
        if len(point_in_b) != 3:
            raise ValueError(f"Expected 3D point with 3 coordinates, got {len(point_in_b)}")
        p_b = np.array(point_in_b, dtype=np.float64)
        p_a = self.R @ p_b + self.t
        return [float(x) for x in p_a]

    def inverse(self) -> PoseMatrix:
        """Compute the inverse transformation T_B_A = (T_A_B)^(-1).

        p_B = R^T * (p_A - t) = R^T * p_A - R^T * t
        """
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return PoseMatrix(
            from_frame=self.to_frame,
            to_frame=self.from_frame,
            rotation_row_major=[float(x) for x in R_inv.flatten()],
            translation=[float(x) for x in t_inv.flatten()],
        )

    def compose(self, other: PoseMatrix) -> PoseMatrix:
        """Compose T_A_C = self(T_A_B) * other(T_B_C)."""
        if self.from_frame != other.to_frame:
            raise ValueError(
                f"Frame mismatch in composition: self.from_frame ('{self.from_frame}') "
                f"!= other.to_frame ('{other.to_frame}')"
            )
        R_composed = self.R @ other.R
        t_composed = self.R @ other.t + self.t
        return PoseMatrix(
            from_frame=other.from_frame,
            to_frame=self.to_frame,
            rotation_row_major=[float(x) for x in R_composed.flatten()],
            translation=[float(x) for x in t_composed.flatten()],
        )

    @classmethod
    def identity(cls, from_frame: str = "body", to_frame: str = "world") -> PoseMatrix:
        """Return an identity transformation."""
        return cls(
            from_frame=from_frame,
            to_frame=to_frame,
            rotation_row_major=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            translation=[0.0, 0.0, 0.0],
        )

    @classmethod
    def from_rot_trans(
        cls,
        R_3x3: Sequence[Sequence[float]],
        t_3: Sequence[float],
        from_frame: str,
        to_frame: str,
    ) -> PoseMatrix:
        """Create PoseMatrix from 3x3 array and 3-element translation."""
        R_flat = [float(x) for row in R_3x3 for x in row]
        t_flat = [float(x) for x in t_3]
        return cls(
            from_frame=from_frame,
            to_frame=to_frame,
            rotation_row_major=R_flat,
            translation=t_flat,
        )
