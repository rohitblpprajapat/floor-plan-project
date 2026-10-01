"""
I/O utilities for StellaVSLAM trajectories and sparse point clouds.
Supports:
- Keyframe trajectory CSV / TUM format
- PLY (Polygon File Format) point cloud export and import
- Pose convention normalization (World-to-Camera T_cw vs Camera-to-World T_wc)
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from src.inertial.imu_types import rot_to_quat, quat_to_rot

@dataclass
class VisualKeyframe:
    """Represents a visual SLAM keyframe pose."""
    keyframe_id: int
    timestamp: float          # in seconds
    R_wc: np.ndarray          # 3x3 rotation matrix from Camera to World (or virtual camera frame)
    p_wc: np.ndarray          # 3D position vector in World frame (up-to-scale)

    def __post_init__(self):
        self.R_wc = np.asarray(self.R_wc, dtype=np.float64).reshape(3, 3)
        self.p_wc = np.asarray(self.p_wc, dtype=np.float64).reshape(3,)
        self.timestamp = float(self.timestamp)
        self.keyframe_id = int(self.keyframe_id)

    @property
    def quat_wc(self) -> np.ndarray:
        """Return quaternion [qw, qx, qy, qz] representing R_wc."""
        return rot_to_quat(self.R_wc)

    @property
    def T_wc(self) -> np.ndarray:
        """4x4 homogeneous transformation matrix [R_wc | p_wc]."""
        T = np.eye(4, dtype=np.float64)
        T[0:3, 0:3] = self.R_wc
        T[0:3, 3] = self.p_wc
        return T

    @property
    def R_cw(self) -> np.ndarray:
        return self.R_wc.T

    @property
    def p_cw(self) -> np.ndarray:
        return -self.R_wc.T @ self.p_wc


class VisualTrajectory:
    """Holds an ordered sequence of visual keyframes."""
    def __init__(self, keyframes: List[VisualKeyframe]):
        self.keyframes = sorted(keyframes, key=lambda kf: kf.timestamp)

    def __len__(self) -> int:
        return len(self.keyframes)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([kf.timestamp for kf in self.keyframes], dtype=np.float64)

    @property
    def positions(self) -> np.ndarray:
        return np.array([kf.p_wc for kf in self.keyframes], dtype=np.float64)

    @property
    def rotations(self) -> List[np.ndarray]:
        return [kf.R_wc for kf in self.keyframes]

    def save_tum(self, filepath: str):
        """Save trajectory in standard TUM format: timestamp tx ty tz qx qy qz qw."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for kf in self.keyframes:
                q = kf.quat_wc # [qw, qx, qy, qz]
                # TUM uses [tx, ty, tz, qx, qy, qz, qw]
                f.write(f"{kf.timestamp:.6f} {kf.p_wc[0]:.6f} {kf.p_wc[1]:.6f} {kf.p_wc[2]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} {q[0]:.6f}\n")

    def save_csv(self, filepath: str):
        """Save trajectory in CSV format with explicit columns."""
        rows = []
        for kf in self.keyframes:
            q = kf.quat_wc
            rows.append({
                "timestamp": kf.timestamp,
                "keyframe_id": kf.keyframe_id,
                "tx": kf.p_wc[0],
                "ty": kf.p_wc[1],
                "tz": kf.p_wc[2],
                "qx": q[1],
                "qy": q[2],
                "qz": q[3],
                "qw": q[0]
            })
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)


def load_trajectory_csv(filepath: str, pose_convention: str = "T_wc") -> VisualTrajectory:
    """
    Load keyframe trajectory from CSV file.
    Supports pose_convention:
    - 'T_wc': position and quaternion represent camera pose in world coordinates.
    - 'T_cw': position and quaternion represent world in camera coordinates.
    """
    df = pd.read_csv(filepath)
    keyframes = []
    for _, row in df.iterrows():
        t = float(row['timestamp'])
        kf_id = int(row.get('keyframe_id', _))
        p = np.array([row['tx'], row['ty'], row['tz']], dtype=np.float64)
        q = np.array([row['qw'], row['qx'], row['qy'], row['qz']], dtype=np.float64)
        R = quat_to_rot(q)

        if pose_convention == "T_cw":
            # Invert to T_wc
            R_wc = R.T
            p_wc = -R.T @ p
        else:
            R_wc = R
            p_wc = p

        keyframes.append(VisualKeyframe(keyframe_id=kf_id, timestamp=t, R_wc=R_wc, p_wc=p_wc))

    return VisualTrajectory(keyframes)


def save_point_cloud_ply(points: np.ndarray, filepath: str, colors: Optional[np.ndarray] = None):
    """
    Save 3D point cloud to binary or ASCII PLY format.
    points: (N, 3) float64
    colors: optional (N, 3) uint8 [r, g, b]
    """
    points = np.asarray(points, dtype=np.float32)
    n_points = len(points)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None and len(colors) == n_points:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if colors is not None and len(colors) == n_points:
            colors = np.asarray(colors, dtype=np.uint8)
            for i in range(n_points):
                f.write(f"{points[i, 0]:.4f} {points[i, 1]:.4f} {points[i, 2]:.4f} {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")
        else:
            for i in range(n_points):
                f.write(f"{points[i, 0]:.4f} {points[i, 1]:.4f} {points[i, 2]:.4f}\n")


def load_point_cloud_ply(filepath: str) -> np.ndarray:
    """
    Loads 3D point cloud from an ASCII PLY file.
    Returns: (N, 3) float64 numpy array
    """
    points = []
    in_header = True
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_str = line.strip()
            if in_header:
                if line_str == "end_header":
                    in_header = False
                continue
            if not line_str:
                continue
            parts = line_str.split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])
                except ValueError:
                    continue

    return np.asarray(points, dtype=np.float64)

