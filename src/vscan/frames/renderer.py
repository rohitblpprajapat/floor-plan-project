"""High-performance vectorized equirectangular-to-perspective reprojection engine.

Features:
- Precomputes and caches coordinate maps (map_x, map_y) for all 6 cubemap faces.
- Enforces strict seam wrapping across longitude boundaries [-pi, pi] via modular arithmetic.
- Prevents polar leakage through elevation bounding.
- Supports RGB image reprojection (bilinear/bicubic) and binary mask reprojection (nearest-neighbor).
- Employs zero synthetic baseline across all faces.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np

from vscan.frames.projection import (
    CUBEMAP_FACES,
    CUBEMAP_ROTATIONS,
    PinholeIntrinsics,
    ReprojectionFace,
    get_default_pinhole_intrinsics,
)


class EquirectangularRenderer:
    """Deterministic perspective view renderer for equirectangular panoramas."""

    def __init__(
        self,
        source_width: int,
        source_height: int,
        intrinsics: Optional[PinholeIntrinsics] = None,
        target_width: int = 960,
        target_height: int = 960,
        fov_deg: float = 90.0,
    ) -> None:
        self.source_width = int(source_width)
        self.source_height = int(source_height)

        if intrinsics is None:
            self.intrinsics = get_default_pinhole_intrinsics(
                width=target_width,
                height=target_height,
                fov_deg=fov_deg,
            )
        else:
            self.intrinsics = intrinsics

        self._maps: Dict[ReprojectionFace, Tuple[np.ndarray, np.ndarray]] = {}
        self._precompute_maps()

    def _precompute_maps(self) -> None:
        """Precompute and cache cv2.remap coordinate tables for all 6 faces."""
        w = self.intrinsics.width
        h = self.intrinsics.height
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.cx
        cy = self.intrinsics.cy

        # 1. Pixel grid in target camera frame with half-integer pixel center convention
        u_coords = np.arange(w, dtype=np.float64) + 0.5
        v_coords = np.arange(h, dtype=np.float64) + 0.5
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)

        # 2. Normalized camera bearing rays
        xc = (u_grid - cx) / fx
        yc = (v_grid - cy) / fy
        zc = np.ones_like(xc)

        ray_norm = np.sqrt(xc * xc + yc * yc + zc * zc)
        xc /= ray_norm
        yc /= ray_norm
        zc /= ray_norm

        # 3. Compute (map_x, map_y) for each cubemap rotation
        for face, R_pc in CUBEMAP_ROTATIONS.items():
            # Rotate camera ray into panorama frame: d_pano = R_pc @ d_cam
            xp = R_pc[0, 0] * xc + R_pc[0, 1] * yc + R_pc[0, 2] * zc
            yp = R_pc[1, 0] * xc + R_pc[1, 1] * yc + R_pc[1, 2] * zc
            zp = R_pc[2, 0] * xc + R_pc[2, 1] * yc + R_pc[2, 2] * zc

            # Spherical coordinates
            phi = -np.arcsin(np.clip(yp, -1.0, 1.0))
            theta = np.arctan2(xp, zp)

            # Equirectangular coordinates (pixel center convention)
            # theta in [-pi, pi] -> u_center in [0, W_eq]
            u_center = (theta / (2.0 * np.pi) + 0.5) * float(self.source_width)
            # phi in [-pi/2, pi/2] -> v_center in [0, H_eq]
            v_center = (0.5 - phi / np.pi) * float(self.source_height)

            # Convert to cv2.remap coordinate convention (pixel indices)
            map_x = u_center - 0.5
            map_y = v_center - 0.5

            # Seam wrap in X (modulo source_width)
            map_x = np.mod(map_x, float(self.source_width)).astype(np.float32)
            # Clamp in Y to valid sensor boundaries
            map_y = np.clip(map_y, 0.0, float(self.source_height - 1)).astype(np.float32)

            self._maps[face] = (map_x, map_y)

    def get_map(self, face: Union[str, ReprojectionFace]) -> Tuple[np.ndarray, np.ndarray]:
        """Return cached (map_x, map_y) for specified face."""
        if isinstance(face, str):
            face = ReprojectionFace(face)
        return self._maps[face]

    def render_face(
        self,
        eq_image: np.ndarray,
        face: Union[str, ReprojectionFace],
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """Reproject equirectangular image into perspective view for single face."""
        if isinstance(face, str):
            face = ReprojectionFace(face)

        map_x, map_y = self._maps[face]
        # BORDER_WRAP provides mathematically correct spherical seam wrapping across longitude boundaries
        return cv2.remap(
            eq_image,
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=cv2.BORDER_WRAP,
        )

    def render_all_faces(
        self,
        eq_image: np.ndarray,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> Dict[str, np.ndarray]:
        """Reproject equirectangular image into all 6 perspective faces."""
        return {
            f"face_{face.value}": self.render_face(eq_image, face, interpolation=interpolation)
            for face in CUBEMAP_FACES
        }

    def render_mask(
        self,
        eq_mask: np.ndarray,
        face: Union[str, ReprojectionFace],
    ) -> np.ndarray:
        """Reproject equirectangular binary mask using nearest-neighbor interpolation."""
        if isinstance(face, str):
            face = ReprojectionFace(face)

        map_x, map_y = self._maps[face]
        return cv2.remap(
            eq_mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def render_all_masks(
        self,
        eq_mask: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Reproject equirectangular mask into all 6 perspective faces."""
        return {
            f"face_{face.value}": self.render_mask(eq_mask, face)
            for face in CUBEMAP_FACES
        }
