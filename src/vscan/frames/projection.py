"""Image projection models, ray transformations, and cubemap geometry.

Conventions:
- Camera optical frame follows OpenCV: +X right, +Y down, +Z forward.
- Panorama equirectangular frame:
  - Azimuth theta in [-pi, pi]: theta=0 is forward (+Z), +pi/2 is right (+X), -pi/2 is left (-X), +-pi is back (-Z).
  - Elevation phi in [-pi/2, pi/2]: phi=+pi/2 is zenith (-Y), phi=0 is equator, phi=-pi/2 is nadir (+Y).
  - Pixel centers are at half-integer coordinates: (x + 0.5, y + 0.5).
- All 6 virtual views of a single panorama share an identical center [0, 0, 0] with zero synthetic baseline.
- Rotation matrices are proper SO(3): det(R) = +1, R^T * R = I.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np

from vscan.schemas.frames import PinholeIntrinsics, ReprojectionFace, VirtualCameraPose


CUBEMAP_FACES = [
    ReprojectionFace.FRONT,
    ReprojectionFace.RIGHT,
    ReprojectionFace.BACK,
    ReprojectionFace.LEFT,
    ReprojectionFace.UP,
    ReprojectionFace.DOWN,
]

# R_pano_cam transforms rays from camera optical frame into panorama frame: d_pano = R_pano_cam @ d_cam
# In OpenCV convention: +X right, +Y down, +Z forward
CUBEMAP_ROTATIONS: Dict[ReprojectionFace, np.ndarray] = {
    ReprojectionFace.FRONT: np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64),
    ReprojectionFace.RIGHT: np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64),
    ReprojectionFace.BACK: np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64),
    ReprojectionFace.LEFT: np.array([
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64),
    ReprojectionFace.UP: np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64),
    ReprojectionFace.DOWN: np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64),
}


def matrix_to_quaternion_wxyz(R: np.ndarray) -> List[float]:
    """Convert a 3x3 rotation matrix to unit quaternion [w, x, y, z] with w >= 0."""
    r = np.asarray(R, dtype=np.float64)
    tr = r[0, 0] + r[1, 1] + r[2, 2]

    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0  # S = 4*w
        w = 0.25 * S
        x = (r[2, 1] - r[1, 2]) / S
        y = (r[0, 2] - r[2, 0]) / S
        z = (r[1, 0] - r[0, 1]) / S
    elif (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
        S = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0  # S = 4*x
        w = (r[2, 1] - r[1, 2]) / S
        x = 0.25 * S
        y = (r[0, 1] + r[1, 0]) / S
        z = (r[0, 2] + r[2, 0]) / S
    elif r[1, 1] > r[2, 2]:
        S = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0  # S = 4*y
        w = (r[0, 2] - r[2, 0]) / S
        x = (r[0, 1] + r[1, 0]) / S
        y = 0.25 * S
        z = (r[1, 2] + r[2, 1]) / S
    else:
        S = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0  # S = 4*z
        w = (r[1, 0] - r[0, 1]) / S
        x = (r[0, 2] + r[2, 0]) / S
        y = (r[1, 2] + r[2, 1]) / S
        z = 0.25 * S

    # Normalize
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    w /= norm
    x /= norm
    y /= norm
    z /= norm

    # Canonicalize w >= 0
    if w < 0.0:
        w = -w
        x = -x
        y = -y
        z = -z

    return [float(w), float(x), float(y), float(z)]


def quaternion_wxyz_to_matrix(q: Sequence[float]) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = [float(val) for val in q]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm > 1e-12:
        w /= norm
        x /= norm
        y /= norm
        z /= norm

    R = np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)
    return R


def get_cubemap_rotations() -> Dict[ReprojectionFace, np.ndarray]:
    """Return dictionary of 3x3 proper SO(3) rotations R_pano_cam for all 6 cubemap faces."""
    return {face: np.copy(rot) for face, rot in CUBEMAP_ROTATIONS.items()}


def get_default_pinhole_intrinsics(
    width: int = 960,
    height: int = 960,
    fov_deg: float = 90.0,
    camera_name: str = "virtual_pinhole",
) -> PinholeIntrinsics:
    """Compute known pinhole intrinsics for a given resolution and FOV."""
    fov_rad = math.radians(fov_deg)
    # Symmetrical pinhole: fx = (width / 2) / tan(fov_x / 2)
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    fy = (height / 2.0) / math.tan(fov_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0

    return PinholeIntrinsics(
        camera_name=camera_name,
        width=width,
        height=height,
        fov_deg=fov_deg,
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        pixel_center_convention="half_integer_offset_center_at_x_plus_0.5_y_plus_0.5",
    )


def scale_pinhole_intrinsics(
    intrinsics: PinholeIntrinsics,
    scale_x: float,
    scale_y: float,
    new_name: Optional[str] = None,
) -> PinholeIntrinsics:
    """Transform camera intrinsics under image resizing."""
    if scale_x <= 0.0 or scale_y <= 0.0:
        raise ValueError(f"Scale factors must be positive: got scale_x={scale_x}, scale_y={scale_y}")

    new_w = max(1, int(round(intrinsics.width * scale_x)))
    new_h = max(1, int(round(intrinsics.height * scale_y)))

    return PinholeIntrinsics(
        camera_name=new_name or f"{intrinsics.camera_name}_scaled_{scale_x:.2f}x{scale_y:.2f}",
        width=new_w,
        height=new_h,
        fov_deg=intrinsics.fov_deg,
        fx=intrinsics.fx * scale_x,
        fy=intrinsics.fy * scale_y,
        cx=intrinsics.cx * scale_x,
        cy=intrinsics.cy * scale_y,
        pixel_center_convention=intrinsics.pixel_center_convention,
    )


def crop_pinhole_intrinsics(
    intrinsics: PinholeIntrinsics,
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
    new_name: Optional[str] = None,
) -> PinholeIntrinsics:
    """Transform camera intrinsics under image cropping."""
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"Crop dimensions must be positive: got {crop_w}x{crop_h}")
    if crop_x0 < 0 or crop_y0 < 0 or crop_x0 + crop_w > intrinsics.width or crop_y0 + crop_h > intrinsics.height:
        raise ValueError(
            f"Crop window [{crop_x0}, {crop_y0}, {crop_w}, {crop_h}] exceeds image bounds [{intrinsics.width}x{intrinsics.height}]"
        )

    return PinholeIntrinsics(
        camera_name=new_name or f"{intrinsics.camera_name}_crop_{crop_x0}_{crop_y0}_{crop_w}x{crop_h}",
        width=crop_w,
        height=crop_h,
        fov_deg=intrinsics.fov_deg,  # Effective full FOV parameter remains documented
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx - float(crop_x0),
        cy=intrinsics.cy - float(crop_y0),
        pixel_center_convention=intrinsics.pixel_center_convention,
    )


def pixel_to_ray_equirectangular(
    u: Union[float, np.ndarray],
    v: Union[float, np.ndarray],
    width: int,
    height: int,
) -> Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Convert equirectangular pixel coordinates (u, v) to unit 3D bearing ray (x, y, z) in panorama frame.

    u in [0, width], v in [0, height].
    Half-integer center convention: pixel (i, j) center is at (i + 0.5, j + 0.5).
    Returns unit ray where:
    +X = right, +Y = down, +Z = forward.
    """
    u_c = np.asarray(u, dtype=np.float64) + 0.5
    v_c = np.asarray(v, dtype=np.float64) + 0.5

    # Azimuth theta in [-pi, pi]
    theta = (u_c / float(width) - 0.5) * (2.0 * np.pi)
    # Elevation phi in [-pi/2, pi/2]
    phi = (0.5 - v_c / float(height)) * np.pi

    cos_phi = np.cos(phi)
    x = cos_phi * np.sin(theta)
    y = -np.sin(phi)  # in OpenCV +Y is down, so phi > 0 (zenith) has y < 0
    z = cos_phi * np.cos(theta)

    if isinstance(u, (int, float)) and isinstance(v, (int, float)):
        return float(x), float(y), float(z)
    return x, y, z


def ray_to_pixel_equirectangular(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    width: int,
    height: int,
) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """Convert 3D bearing ray (x, y, z) in panorama frame to equirectangular pixel coordinates (u, v).

    Handles seam wrapping [-pi, pi] and polar clamping [-pi/2, pi/2] safely.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)

    norm = np.sqrt(x_arr * x_arr + y_arr * y_arr + z_arr * z_arr)
    norm = np.where(norm < 1e-12, 1e-12, norm)
    x_n = x_arr / norm
    y_n = y_arr / norm
    z_n = z_arr / norm

    phi = -np.arcsin(np.clip(y_n, -1.0, 1.0))
    theta = np.arctan2(x_n, z_n)

    u_c = (theta / (2.0 * np.pi) + 0.5) * float(width)
    v_c = (0.5 - phi / np.pi) * float(height)

    # Convert back from pixel center to pixel coordinate
    u_px = u_c - 0.5
    v_px = v_c - 0.5

    # Seam wrap in X
    u_px = np.mod(u_px, float(width))
    # Clamp in Y to valid sensor range [0, height - 1]
    v_px = np.clip(v_px, 0.0, float(height - 1))

    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float)):
        return float(u_px), float(v_px)
    return u_px, v_px


def pixel_to_ray_pinhole(
    u: Union[float, np.ndarray],
    v: Union[float, np.ndarray],
    intrinsics: PinholeIntrinsics,
) -> Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Convert pixel coordinates (u, v) on perspective image to 3D unit ray in camera frame.

    OpenCV optical frame: +X right, +Y down, +Z forward.
    """
    u_c = np.asarray(u, dtype=np.float64) + 0.5
    v_c = np.asarray(v, dtype=np.float64) + 0.5

    xc = (u_c - intrinsics.cx) / intrinsics.fx
    yc = (v_c - intrinsics.cy) / intrinsics.fy
    zc = np.ones_like(xc)

    norm = np.sqrt(xc * xc + yc * yc + zc * zc)
    xc /= norm
    yc /= norm
    zc /= norm

    if isinstance(u, (int, float)) and isinstance(v, (int, float)):
        return float(xc), float(yc), float(zc)
    return xc, yc, zc


def ray_to_pixel_pinhole(
    xc: Union[float, np.ndarray],
    yc: Union[float, np.ndarray],
    zc: Union[float, np.ndarray],
    intrinsics: PinholeIntrinsics,
) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """Project 3D bearing ray in camera frame to pixel coordinates (u, v) on image plane.

    Requires zc > 0 (points in front of camera).
    """
    x_arr = np.asarray(xc, dtype=np.float64)
    y_arr = np.asarray(yc, dtype=np.float64)
    z_arr = np.asarray(zc, dtype=np.float64)

    # Avoid division by zero
    z_safe = np.where(np.abs(z_arr) < 1e-12, 1e-12, z_arr)
    u_c = intrinsics.fx * (x_arr / z_safe) + intrinsics.cx
    v_c = intrinsics.fy * (y_arr / z_safe) + intrinsics.cy

    u_px = u_c - 0.5
    v_px = v_c - 0.5

    if isinstance(xc, (int, float)) and isinstance(yc, (int, float)) and isinstance(zc, (int, float)):
        return float(u_px), float(v_px)
    return u_px, v_px


def build_virtual_camera_poses(
    intrinsics: PinholeIntrinsics,
) -> Dict[str, VirtualCameraPose]:
    """Construct VirtualCameraPose specifications for all 6 cubemap faces with PyCOLMAP rig parameters."""
    poses: Dict[str, VirtualCameraPose] = {}

    for idx, face in enumerate(CUBEMAP_FACES):
        cam_name = f"face_{face.value}"
        R_pc = CUBEMAP_ROTATIONS[face]

        # PyCOLMAP RigConfig expects cam_from_rig = (R_pano_cam)^T
        R_cp = R_pc.T
        q_cp = matrix_to_quaternion_wxyz(R_cp)

        pose = VirtualCameraPose(
            camera_name=cam_name,
            face_name=face,
            face_index=idx,
            R_pano_cam=[float(x) for x in R_pc.flatten()],
            t_pano_cam=[0.0, 0.0, 0.0],
            cam_from_rig_rotation_wxyz=q_cp,
            cam_from_rig_translation=[0.0, 0.0, 0.0],
        )
        poses[cam_name] = pose

    return poses
