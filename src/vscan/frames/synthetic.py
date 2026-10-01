"""Deterministic synthetic textured room with multiple depth surfaces and analytic ground truth.

Features:
- Bounded 3D room with distinct textured planes: floor, ceiling, 4 walls, and internal depth obstacle.
- Closed-form analytic ray-surface intersection yielding subpixel ground truth 3D points and depths.
- Generates both analytic equirectangular panoramas and analytic pinhole perspective views.
- Generates a rotation-only sequence (zero translation baseline) as a negative fixture.
- Used strictly as test data, never as a substitute for real room data.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from vscan.frames.projection import (
    CUBEMAP_FACES,
    CUBEMAP_ROTATIONS,
    PinholeIntrinsics,
    ReprojectionFace,
    pixel_to_ray_equirectangular,
    pixel_to_ray_pinhole,
)


class PlanarSurface:
    """Bounded planar 3D surface with analytic texture function."""

    def __init__(
        self,
        surface_id: str,
        normal: Sequence[float],
        d: float,
        bounds_x: Tuple[float, float],
        bounds_y: Tuple[float, float],
        bounds_z: Tuple[float, float],
        texture_scale: float = 2.0,
        color1: Tuple[int, int, int] = (220, 220, 220),
        color2: Tuple[int, int, int] = (40, 40, 40),
    ) -> None:
        self.surface_id = surface_id
        n = np.asarray(normal, dtype=np.float64)
        n /= np.linalg.norm(n)
        self.normal = n
        self.d = float(d)  # Plane equation: n . P + d = 0
        self.bounds_x = (min(bounds_x), max(bounds_x))
        self.bounds_y = (min(bounds_y), max(bounds_y))
        self.bounds_z = (min(bounds_z), max(bounds_z))
        self.texture_scale = texture_scale
        self.color1 = color1
        self.color2 = color2

    def intersect(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> Tuple[bool, float, Optional[np.ndarray], Tuple[int, int, int]]:
        """Compute exact analytic intersection of ray with bounded plane."""
        denom = float(np.dot(self.normal, direction))
        if abs(denom) < 1e-6:
            return False, float("inf"), None, (0, 0, 0)

        t = -(float(np.dot(self.normal, origin)) + self.d) / denom
        if t <= 1e-4:
            return False, float("inf"), None, (0, 0, 0)

        p = origin + t * direction
        # Check bounding box
        eps = 1e-4
        if not (self.bounds_x[0] - eps <= p[0] <= self.bounds_x[1] + eps):
            return False, float("inf"), None, (0, 0, 0)
        if not (self.bounds_y[0] - eps <= p[1] <= self.bounds_y[1] + eps):
            return False, float("inf"), None, (0, 0, 0)
        if not (self.bounds_z[0] - eps <= p[2] <= self.bounds_z[1] + eps):
            return False, float("inf"), None, (0, 0, 0)

        # Procedural checkerboard texture
        # Find tangent axes perpendicular to normal
        if abs(self.normal[0]) > 0.8:
            u_coord = p[2] * self.texture_scale
            v_coord = p[1] * self.texture_scale
        elif abs(self.normal[1]) > 0.8:
            u_coord = p[0] * self.texture_scale
            v_coord = p[2] * self.texture_scale
        else:
            u_coord = p[0] * self.texture_scale
            v_coord = p[1] * self.texture_scale

        check = (int(math.floor(u_coord)) + int(math.floor(v_coord))) % 2
        rgb = self.color1 if check == 0 else self.color2

        return True, t, p, rgb


class SyntheticRoom:
    """Geometric 3D room containing walls, floor, ceiling, and interior obstacle surfaces."""

    def __init__(self) -> None:
        self.surfaces: List[PlanarSurface] = []
        self._build_surfaces()

    def _build_surfaces(self) -> None:
        # Dimensions: X in [-2, 2], Y in [-1.5, 1.5], Z in [-3, 3]
        # Front wall: Z = +3.0 (normal [0, 0, -1], d = 3.0)
        self.surfaces.append(
            PlanarSurface("front_wall", [0, 0, -1], 3.0, (-2, 2), (-1.5, 1.5), (3, 3), 2.0, (20, 180, 20), (220, 240, 220))
        )
        # Back wall: Z = -3.0 (normal [0, 0, 1], d = 3.0)
        self.surfaces.append(
            PlanarSurface("back_wall", [0, 0, 1], 3.0, (-2, 2), (-1.5, 1.5), (-3, -3), 2.0, (20, 180, 220), (220, 240, 240))
        )
        # Left wall: X = -2.0 (normal [1, 0, 0], d = 2.0)
        self.surfaces.append(
            PlanarSurface("left_wall", [1, 0, 0], 2.0, (-2, -2), (-1.5, 1.5), (-3, 3), 2.0, (180, 20, 20), (240, 220, 220))
        )
        # Right wall: X = +2.0 (normal [-1, 0, 0], d = 2.0)
        self.surfaces.append(
            PlanarSurface("right_wall", [-1, 0, 0], 2.0, (2, 2), (-1.5, 1.5), (-3, 3), 2.0, (180, 180, 20), (240, 240, 220))
        )
        # Ceiling: Y = -1.5 (normal [0, 1, 0], d = 1.5)
        self.surfaces.append(
            PlanarSurface("ceiling", [0, 1, 0], 1.5, (-2, 2), (-1.5, -1.5), (-3, 3), 4.0, (100, 100, 100), (200, 200, 200))
        )
        # Floor: Y = +1.5 (normal [0, -1, 0], d = 1.5)
        self.surfaces.append(
            PlanarSurface("floor", [0, -1, 0], 1.5, (-2, 2), (1.5, 1.5), (-3, 3), 3.0, (60, 40, 20), (160, 120, 80))
        )

        # Interior Obstacle: rectangular pillar with distinct depth
        # X in [0.4, 1.2], Y in [0.0, 1.5], Z in [1.0, 1.8]
        self.surfaces.append(
            PlanarSurface("obstacle_front", [0, 0, -1], 1.0, (0.4, 1.2), (0.0, 1.5), (1.0, 1.0), 5.0, (120, 30, 180), (240, 200, 255))
        )
        self.surfaces.append(
            PlanarSurface("obstacle_left", [1, 0, 0], -0.4, (0.4, 0.4), (0.0, 1.5), (1.0, 1.8), 5.0, (140, 40, 190), (250, 210, 255))
        )
        self.surfaces.append(
            PlanarSurface("obstacle_top", [0, 1, 0], 0.0, (0.4, 1.2), (0.0, 0.0), (1.0, 1.8), 5.0, (100, 20, 160), (230, 190, 245))
        )

    def trace_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> Tuple[bool, float, Optional[np.ndarray], Tuple[int, int, int], str]:
        """Intersect ray with all room surfaces and return closest analytic hit."""
        best_t = float("inf")
        best_p = None
        best_color = (0, 0, 0)
        best_id = "none"
        hit_any = False

        for surface in self.surfaces:
            hit, t, p, color = surface.intersect(origin, direction)
            if hit and t < best_t:
                best_t = t
                best_p = p
                best_color = color
                best_id = surface.surface_id
                hit_any = True

        return hit_any, best_t, best_p, best_color, best_id

    def render_analytic_equirectangular(
        self,
        camera_center: np.ndarray,
        width: int = 960,
        height: int = 480,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render exact equirectangular RGB image and analytic depth map from camera center."""
        rgb_img = np.zeros((height, width, 3), dtype=np.uint8)
        depth_img = np.zeros((height, width), dtype=np.float32)

        u_px, v_px = np.meshgrid(np.arange(width), np.arange(height))
        # Compute 3D rays for entire equirectangular grid
        rx, ry, rz = pixel_to_ray_equirectangular(u_px, v_px, width, height)

        origin = np.asarray(camera_center, dtype=np.float64)

        for v in range(height):
            for u in range(width):
                ray_dir = np.array([rx[v, u], ry[v, u], rz[v, u]], dtype=np.float64)
                hit, t, p, color, _ = self.trace_ray(origin, ray_dir)
                if hit:
                    rgb_img[v, u] = color  # BGR order
                    depth_img[v, u] = t

        return rgb_img, depth_img

    def render_analytic_pinhole(
        self,
        camera_center: np.ndarray,
        R_pano_cam: np.ndarray,
        intrinsics: PinholeIntrinsics,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render direct analytic pinhole perspective image and depth map."""
        w = intrinsics.width
        h = intrinsics.height
        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
        depth_img = np.zeros((h, w), dtype=np.float32)

        u_px, v_px = np.meshgrid(np.arange(w), np.arange(h))
        xc, yc, zc = pixel_to_ray_pinhole(u_px, v_px, intrinsics)

        origin = np.asarray(camera_center, dtype=np.float64)
        R = np.asarray(R_pano_cam, dtype=np.float64)

        for v in range(h):
            for u in range(w):
                d_c = np.array([xc[v, u], yc[v, u], zc[v, u]], dtype=np.float64)
                # Rotate to pano frame: d_pano = R @ d_c
                d_pano = R @ d_c
                hit, t, p, color, _ = self.trace_ray(origin, d_pano)
                if hit:
                    rgb_img[v, u] = color
                    depth_img[v, u] = t

        return rgb_img, depth_img


def generate_synthetic_translation_fixture(num_frames: int = 5) -> List[Dict[str, Any]]:
    """Generate known ground truth camera trajectory with pure/composite translations."""
    fixtures = []
    # Camera moving along Z axis from Z = -1.0 to Z = +0.6
    for i in range(num_frames):
        center = [0.0, 0.0, -1.0 + float(i) * 0.4]
        fixtures.append({
            "frame_index": i,
            "camera_center": center,
            "rotation_matrix": np.eye(3, dtype=np.float64).tolist(),
            "trajectory_type": "translation_baseline",
        })
    return fixtures


def generate_synthetic_rotation_only_fixture(num_frames: int = 5) -> List[Dict[str, Any]]:
    """Generate negative fixture: strictly ZERO translation baseline with rotating camera."""
    fixtures = []
    fixed_center = [0.0, 0.0, 0.0]

    for i in range(num_frames):
        yaw_deg = float(i) * 15.0
        yaw_rad = math.radians(yaw_deg)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        # Rotation around Y axis: OpenCV +Y down
        R_yaw = np.array([
            [cos_y, 0.0, sin_y],
            [0.0, 1.0, 0.0],
            [-sin_y, 0.0, cos_y],
        ], dtype=np.float64)

        fixtures.append({
            "frame_index": i,
            "camera_center": list(fixed_center),  # strictly zero baseline
            "rotation_matrix": R_yaw.tolist(),
            "yaw_deg": yaw_deg,
            "trajectory_type": "rotation_only_negative_fixture",
            "translation_baseline_norm": 0.0,
        })
    return fixtures
