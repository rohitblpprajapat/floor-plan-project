"""
Vectorized Architectural Wall & Boundary Extractor.
Extracts clean CAD-style 2D line segments from 3D metric point clouds using:
1. Horizontal point cloud slicing around eye/camera level.
2. Iterative RANSAC line fitting with collinear clustering.
3. Manhattan World orthogonal snapping (enforcing strict 90-degree corners).
4. Exports clean SVG and GeoJSON/JSON vector floor plans.
"""

import os
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class WallSegment:
    """2D Line segment representing an architectural wall."""
    p1: np.ndarray # [x1, y1]
    p2: np.ndarray # [x2, y2]
    length: float
    angle_deg: float


class VectorWallExtractor:
    """Extracts rectilinear CAD wall boundaries from 3D point cloud slices."""

    def __init__(
        self,
        slice_z_half_width: float = 0.7,
        ransac_distance_threshold: float = 0.08,
        min_inliers_per_wall: int = 25,
        max_walls: int = 20
    ):
        self.slice_z = slice_z_half_width
        self.ransac_dist = ransac_distance_threshold
        self.min_inliers = min_inliers_per_wall
        self.max_walls = max_walls

    def extract_walls(
        self,
        landmarks: np.ndarray,
        camera_z: float = 0.0,
        enforce_manhattan: bool = True
    ) -> List[WallSegment]:
        """
        Extracts 2D wall segments from point cloud horizontal slice.
        """
        if len(landmarks) < self.min_inliers * 2:
            return []

        # 1. Horizontal slice around camera height
        z = landmarks[:, 2]
        mask_slice = (z >= camera_z - self.slice_z) & (z <= camera_z + self.slice_z)
        pts_2d = landmarks[mask_slice, :2].copy() # (M, 2)

        if len(pts_2d) < self.min_inliers:
            pts_2d = landmarks[:, :2].copy()

        remaining_pts = pts_2d.copy()
        raw_walls: List[WallSegment] = []

        # 2. Iterative RANSAC Line Extraction
        for _ in range(self.max_walls):
            if len(remaining_pts) < self.min_inliers:
                break

            best_inliers = None
            best_line = None
            max_cnt = 0

            # Random sample pairs
            n_samples = min(200, len(remaining_pts) * 2)
            idx1 = np.random.randint(0, len(remaining_pts), size=n_samples)
            idx2 = np.random.randint(0, len(remaining_pts), size=n_samples)

            for i in range(n_samples):
                if idx1[i] == idx2[i]:
                    continue
                pA = remaining_pts[idx1[i]]
                pB = remaining_pts[idx2[i]]
                v = pB - pA
                norm_v = np.linalg.norm(v)
                if norm_v < 0.2:
                    continue
                v_unit = v / norm_v
                n_unit = np.array([-v_unit[1], v_unit[0]]) # perpendicular normal

                # Distances of all points to line
                dists = np.abs((remaining_pts - pA) @ n_unit)
                inliers = np.where(dists < self.ransac_dist)[0]
                if len(inliers) > max_cnt:
                    max_cnt = len(inliers)
                    best_inliers = inliers
                    best_line = (pA, v_unit)

            if max_cnt < self.min_inliers or best_line is None:
                break

            pA, v_unit = best_line
            inlier_pts = remaining_pts[best_inliers]

            # Project inliers onto line to find extents
            projs = (inlier_pts - pA) @ v_unit
            min_proj, max_proj = np.min(projs), np.max(projs)
            p1 = pA + min_proj * v_unit
            p2 = pA + max_proj * v_unit
            length = float(np.linalg.norm(p2 - p1))

            if length >= 0.3:
                angle_deg = float(np.degrees(np.arctan2(v_unit[1], v_unit[0]))) % 180.0
                raw_walls.append(WallSegment(p1=p1, p2=p2, length=length, angle_deg=angle_deg))

            # Remove inliers for next iteration
            remaining_pts = np.delete(remaining_pts, best_inliers, axis=0)

        # 3. Manhattan World Orthogonal Snapping
        if enforce_manhattan and len(raw_walls) > 0:
            # Dominant orientation from longest wall
            longest = max(raw_walls, key=lambda w: w.length)
            dom_angle = longest.angle_deg % 90.0

            refined_walls = []
            for w in raw_walls:
                diff = (w.angle_deg - dom_angle) % 90.0
                # Snap to closest 0 or 90
                snap_to = 0.0 if diff < 45.0 else 90.0
                target_angle_rad = np.radians(dom_angle + (snap_to if diff >= 45.0 else 0.0))

                mid = (w.p1 + w.p2) / 2.0
                dir_vec = np.array([np.cos(target_angle_rad), np.sin(target_angle_rad)])
                half_len = w.length / 2.0
                p1_snap = mid - half_len * dir_vec
                p2_snap = mid + half_len * dir_vec

                refined_walls.append(WallSegment(
                    p1=p1_snap,
                    p2=p2_snap,
                    length=w.length,
                    angle_deg=float(np.degrees(target_angle_rad)) % 180.0
                ))
            return refined_walls

        return raw_walls

    def export_json(self, walls: List[WallSegment], output_path: str):
        """Exports wall line segments to JSON."""
        data = [
            {
                "id": i,
                "start": [float(w.p1[0]), float(w.p1[1])],
                "end": [float(w.p2[0]), float(w.p2[1])],
                "length_meters": float(w.length),
                "angle_degrees": float(w.angle_deg)
            }
            for i, w in enumerate(walls)
        ]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def export_svg(self, walls: List[WallSegment], output_path: str, width_px: int = 800, height_px: int = 800):
        """Exports clean architectural vector lines to SVG."""
        if len(walls) == 0:
            return

        all_pts = np.vstack([[w.p1, w.p2] for w in walls])
        min_x, min_y = np.min(all_pts, axis=0) - 0.5
        max_x, max_y = np.max(all_pts, axis=0) + 0.5
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        scale = min((width_px - 80) / span_x, (height_px - 80) / span_y)

        def to_svg(x, y):
            sx = 40 + (x - min_x) * scale
            sy = height_px - 40 - (y - min_y) * scale
            return sx, sy

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width_px} {height_px}" width="{width_px}" height="{height_px}">\n')
            f.write(f'  <rect width="100%" height="100%" fill="#121216"/>\n')
            f.write(f'  <g stroke="#00e5ff" stroke-width="3.5" stroke-linecap="round">\n')
            for w in walls:
                x1, y1 = to_svg(w.p1[0], w.p1[1])
                x2, y2 = to_svg(w.p2[0], w.p2[1])
                f.write(f'    <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>\n')
            f.write(f'  </g>\n')
            f.write(f'  <text x="30" y="40" fill="white" font-family="sans-serif" font-size="16" font-weight="bold">Architectural Vector Floor Plan</text>\n')
            f.write(f'</svg>\n')
