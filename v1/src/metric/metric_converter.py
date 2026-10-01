"""
Metric Converter.
Transforms StellaVSLAM up-to-scale camera trajectory and sparse point clouds into
gravity-aligned, metric-scaled physical coordinates in meters (Sections 25 and 26).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from src.inertial.imu_types import so3_exp, GRAVITY_NOMINAL
from src.stella.stella_io import VisualKeyframe, VisualTrajectory, save_point_cloud_ply

@dataclass
class MetricSummary:
    """Summary of metric physical dimensions and trajectory statistics."""
    scale: float
    trajectory_length_meters: float
    duration_seconds: float
    average_speed_mps: float
    height_range_meters: float
    bounding_box_meters: Dict[str, float]
    gravity_alignment_angle_deg: float


def compute_gravity_alignment_matrix(gravity_vector_w: np.ndarray) -> np.ndarray:
    """
    Computes R_g in SO(3) that rotates the estimated world gravity vector
    so that it points strictly along the negative Z-axis: R_g * (g_w / ||g_w||) = [0, 0, -1]^T.
    """
    g = np.asarray(gravity_vector_w, dtype=np.float64).reshape(3,)
    norm = np.linalg.norm(g)
    if norm < 1e-4:
        return np.eye(3, dtype=np.float64)

    u = g / norm
    target = np.array([0.0, 0.0, -1.0])

    dot = np.clip(np.dot(u, target), -1.0, 1.0)
    if np.isclose(dot, 1.0):
        return np.eye(3, dtype=np.float64)
    if np.isclose(dot, -1.0):
        # 180 degree rotation around X
        return np.diag([1.0, -1.0, -1.0])

    axis = np.cross(u, target)
    axis_norm = np.linalg.norm(axis)
    axis = axis / axis_norm
    angle = np.arccos(dot)

    return so3_exp(axis * angle)


class MetricConverter:
    """Applies metric scale and gravity alignment to trajectory and landmark map."""

    def __init__(self, scale: float, gravity_w: np.ndarray):
        if scale <= 0.0:
            raise ValueError(f"Metric scale must be strictly positive, got {scale}")
        self.scale = float(scale)
        self.gravity_w = np.asarray(gravity_w, dtype=np.float64).reshape(3,)
        self.R_g = compute_gravity_alignment_matrix(self.gravity_w)

    def convert_trajectory(self, trajectory: VisualTrajectory) -> Tuple[VisualTrajectory, MetricSummary]:
        """
        Transforms trajectory to metric coordinates and computes physical summary.
        """
        metric_keyframes = []
        for kf in trajectory.keyframes:
            p_metric = self.R_g @ (self.scale * kf.p_wc)
            R_metric = self.R_g @ kf.R_wc
            metric_keyframes.append(VisualKeyframe(
                keyframe_id=kf.keyframe_id,
                timestamp=kf.timestamp,
                R_wc=R_metric,
                p_wc=p_metric
            ))

        metric_traj = VisualTrajectory(metric_keyframes)

        # Compute physical summary statistics
        pts = metric_traj.positions
        diffs = np.diff(pts, axis=0)
        step_lengths = np.linalg.norm(diffs, axis=1)
        total_len = float(np.sum(step_lengths))

        ts = metric_traj.timestamps
        dur = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        avg_speed = total_len / max(dur, 1e-6)

        z_min, z_max = float(np.min(pts[:, 2])), float(np.max(pts[:, 2]))
        height_range = z_max - z_min

        bbox = {
            "dx_meters": float(np.max(pts[:, 0]) - np.min(pts[:, 0])),
            "dy_meters": float(np.max(pts[:, 1]) - np.min(pts[:, 1])),
            "dz_meters": height_range
        }

        # Gravity alignment angle
        u = self.gravity_w / np.linalg.norm(self.gravity_w)
        dot = np.clip(np.dot(u, [0.0, 0.0, -1.0]), -1.0, 1.0)
        align_angle_deg = float(np.degrees(np.arccos(dot)))

        summary = MetricSummary(
            scale=self.scale,
            trajectory_length_meters=total_len,
            duration_seconds=dur,
            average_speed_mps=avg_speed,
            height_range_meters=height_range,
            bounding_box_meters=bbox,
            gravity_alignment_angle_deg=align_angle_deg
        )

        return metric_traj, summary

    def convert_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Transforms (N, 3) visual landmarks to metric gravity-aligned coordinates.
        """
        landmarks = np.asarray(landmarks, dtype=np.float64)
        if len(landmarks) == 0:
            return landmarks
        # X_metric = R_g * (scale * X_stella)
        return (self.R_g @ (self.scale * landmarks.T)).T
