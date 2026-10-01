"""Mask generation, polygon seam-splitting, provenance tracking, and perspective propagation.

Features:
- Explicit mask categories: operator, support hardware (nadir mount), moving objects, seam boundaries.
- Stores polygon coordinates in their original equirectangular projection (normalized [0, 1] x [0, 1]).
- Robust handling of seam-crossing polygons (wrapping between x=0.0 and x=1.0).
- Does NOT blindly hide all floor/ceiling imagery; strictly bounds nadir hardware footprint.
- Propagates equirectangular masks into perspective views using nearest-neighbor sampling.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from vscan.frames.projection import PinholeIntrinsics, ReprojectionFace
from vscan.frames.renderer import EquirectangularRenderer
from vscan.schemas.frames import MaskDefinition


def _clip_polygon_halfplane(
    polygon: List[List[float]],
    is_inside: Any,
    line_intersect: Any,
) -> List[List[float]]:
    """Sutherland-Hodgman halfplane polygon clipping."""
    out: List[List[float]] = []
    if not polygon:
        return out
    prev = polygon[-1]
    prev_in = is_inside(prev)
    for curr in polygon:
        curr_in = is_inside(curr)
        if curr_in:
            if not prev_in:
                out.append(line_intersect(prev, curr))
            out.append(curr)
        elif prev_in:
            out.append(line_intersect(prev, curr))
        prev = curr
        prev_in = curr_in
    return out


def _clip_polygon_x_range(
    polygon: List[List[float]],
    x_min: float,
    x_max: float,
) -> List[List[float]]:
    """Clip polygon to vertical interval [x_min, x_max]."""
    p = _clip_polygon_halfplane(
        polygon,
        lambda pt: pt[0] >= x_min,
        lambda p1, p2: [x_min, p1[1] + (p2[1] - p1[1]) * (x_min - p1[0]) / (p2[0] - p1[0]) if abs(p2[0] - p1[0]) > 1e-12 else p1[1]],
    )
    p = _clip_polygon_halfplane(
        p,
        lambda pt: pt[0] <= x_max,
        lambda p1, p2: [x_max, p1[1] + (p2[1] - p1[1]) * (x_max - p1[0]) / (p2[0] - p1[0]) if abs(p2[0] - p1[0]) > 1e-12 else p1[1]],
    )
    return p


def split_seam_crossing_polygon(
    polygon_normalized: List[List[float]],
) -> Tuple[List[List[List[float]]], bool]:
    """Inspect polygon and split across the horizontal seam x=0.0 / x=1.0 if it spans across boundaries.

    Returns:
        (list_of_sub_polygons, is_seam_crossing)
    """
    pts = np.asarray(polygon_normalized, dtype=np.float64)
    if len(pts) < 3:
        return [polygon_normalized], False

    xs = pts[:, 0]

    # Check if edge connects across seam (|dx| > 0.5) or coordinates lie outside [0, 1]
    diffs = np.abs(np.diff(np.append(xs, xs[0])))
    has_seam_jump = bool(np.any(diffs > 0.5)) or bool(np.any(xs < 0.0)) or bool(np.any(xs > 1.0))

    if not has_seam_jump:
        return [polygon_normalized], False

    # Shift points into a continuous un-wrapped range
    unwrapped = [list(polygon_normalized[0])]
    for pt in polygon_normalized[1:]:
        dx = pt[0] - unwrapped[-1][0]
        if dx > 0.5:
            dx -= 1.0
        elif dx < -0.5:
            dx += 1.0
        unwrapped.append([unwrapped[-1][0] + dx, pt[1]])

    # Clip [0, 1]
    p_main = _clip_polygon_x_range(unwrapped, 0.0, 1.0)
    # Clip [-1, 0] and shift +1 to right boundary [0, 1]
    p_neg = _clip_polygon_x_range(unwrapped, -1.0, 0.0)
    p_neg_shifted = [[pt[0] + 1.0, pt[1]] for pt in p_neg]
    # Clip [1, 2] and shift -1 to left boundary [0, 1]
    p_pos = _clip_polygon_x_range(unwrapped, 1.0, 2.0)
    p_pos_shifted = [[pt[0] - 1.0, pt[1]] for pt in p_pos]

    results: List[List[List[float]]] = []
    for cand in [p_main, p_neg_shifted, p_pos_shifted]:
        if len(cand) >= 3:
            # Remove duplicate consecutive vertices
            cleaned: List[List[float]] = []
            for p in cand:
                if not cleaned or (abs(p[0] - cleaned[-1][0]) > 1e-5 or abs(p[1] - cleaned[-1][1]) > 1e-5):
                    cleaned.append([float(p[0]), float(p[1])])
            if len(cleaned) > 2 and abs(cleaned[0][0] - cleaned[-1][0]) < 1e-5 and abs(cleaned[0][1] - cleaned[-1][1]) < 1e-5:
                cleaned.pop()
            if len(cleaned) >= 3:
                # Remove collinear points
                final: List[List[float]] = []
                n = len(cleaned)
                for i in range(n):
                    p_prev = cleaned[i - 1]
                    p_curr = cleaned[i]
                    p_next = cleaned[(i + 1) % n]
                    cross = (p_curr[0] - p_prev[0]) * (p_next[1] - p_curr[1]) - (p_curr[1] - p_prev[1]) * (p_next[0] - p_curr[0])
                    if abs(cross) > 1e-6:
                        final.append(p_curr)
                if len(final) >= 3:
                    results.append(final)
                else:
                    results.append(cleaned)

    if not results:
        results = [polygon_normalized]

    return results, True


def rasterize_polygon_mask(
    width: int,
    height: int,
    polygon_normalized: List[List[float]],
) -> np.ndarray:
    """Rasterize a normalized polygon onto a uint8 binary mask (0 = valid, 255 = masked).

    Correctly handles seam-crossing polygons by rendering sub-components.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    sub_polys, is_crossing = split_seam_crossing_polygon(polygon_normalized)

    for poly in sub_polys:
        pts_px = np.array(
            [[int(round(p[0] * width)), int(round(p[1] * height))] for p in poly],
            dtype=np.int32,
        )
        # Clamp to bounds
        pts_px[:, 0] = np.clip(pts_px[:, 0], 0, width - 1)
        pts_px[:, 1] = np.clip(pts_px[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [pts_px], color=255)

    return mask


def create_hardware_nadir_mask(
    width: int,
    height: int,
    nadir_elevation_deg: float = -78.0,
) -> Tuple[np.ndarray, MaskDefinition]:
    """Generate a precise hardware mount / selfie stick mask around the nadir pole.

    Does NOT blindly mask all floor imagery: strictly limits masking to the physical
    mount footprint based on elevation angle.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Elevation phi = (0.5 - v_c / H) * pi
    # Nadir cutoff: phi < radians(nadir_elevation_deg)
    # (0.5 - v_c / H) * pi < rad -> v_c / H > 0.5 - rad / pi
    cutoff_ratio = 0.5 - (math.radians(nadir_elevation_deg) / math.pi)
    cutoff_y = max(0, min(height - 1, int(round(cutoff_ratio * height))))

    mask[cutoff_y:, :] = 255

    # Define polygon representation for provenance
    polygon = [
        [0.0, float(cutoff_ratio)],
        [1.0, float(cutoff_ratio)],
        [1.0, 1.0],
        [0.0, 1.0],
    ]

    definition = MaskDefinition(
        mask_id="mask_hardware_nadir_mount",
        mask_type="support_hardware",
        provenance=f"nadir_cone_elevation_cutoff_{nadir_elevation_deg:.1f}deg",
        original_projection="equirectangular_normalized",
        polygon_coordinates=polygon,
        is_seam_crossing=False,
        description=f"Physical mounting hardware mask at nadir (elevation < {nadir_elevation_deg:.1f} deg). Preserves surrounding floor.",
    )
    return mask, definition


class MaskManager:
    """Manages active masks, composite generation, and perspective propagation."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._definitions: List[MaskDefinition] = []
        self._masks: List[np.ndarray] = []

    def add_mask(self, mask: np.ndarray, definition: MaskDefinition) -> None:
        """Register a new mask layer with provenance."""
        if mask.shape != (self.height, self.width):
            raise ValueError(f"Mask shape {mask.shape} does not match manager dimensions ({self.height}, {self.width})")
        self._masks.append(mask)
        self._definitions.append(definition)

    def add_hardware_nadir(self, nadir_elevation_deg: float = -78.0) -> None:
        """Add standard hardware nadir mask."""
        mask, definition = create_hardware_nadir_mask(self.width, self.height, nadir_elevation_deg)
        self.add_mask(mask, definition)

    def add_polygon(
        self,
        polygon_normalized: List[List[float]],
        mask_id: str,
        mask_type: str = "operator",
        provenance: str = "supplied_polygon",
        description: str = "",
    ) -> None:
        """Add a polygon mask in normalized equirectangular coordinates."""
        _, is_crossing = split_seam_crossing_polygon(polygon_normalized)
        mask = rasterize_polygon_mask(self.width, self.height, polygon_normalized)
        definition = MaskDefinition(
            mask_id=mask_id,
            mask_type=mask_type,
            provenance=provenance,
            original_projection="equirectangular_normalized",
            polygon_coordinates=polygon_normalized,
            is_seam_crossing=is_crossing,
            description=description,
        )
        self.add_mask(mask, definition)

    @property
    def definitions(self) -> List[MaskDefinition]:
        """Return all registered mask definitions."""
        return list(self._definitions)

    def get_composite_equirectangular_mask(self) -> np.ndarray:
        """Combine all active masks using logical OR."""
        composite = np.zeros((self.height, self.width), dtype=np.uint8)
        for m in self._masks:
            composite = np.bitwise_or(composite, m)
        return composite

    def propagate_to_perspective(
        self,
        renderer: EquirectangularRenderer,
    ) -> Dict[str, np.ndarray]:
        """Reproject composite equirectangular mask to all 6 cubemap perspective views."""
        composite = self.get_composite_equirectangular_mask()
        return renderer.render_all_masks(composite)
