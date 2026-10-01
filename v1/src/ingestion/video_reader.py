"""
Equirectangular 360 Video Reader and Spherical Feature Tracker.

Provides:
- Robust video frame extraction with precise frame timestamps
- Projection of equirectangular pixels (u, v) to 3D unit sphere bearing vectors
- Direct visual rotation rate (visual gyro) estimation via spherical feature matching
  and RANSAC rotation solve on the sphere.
"""

import os
import cv2
import numpy as np
from typing import Dict, Generator, List, Optional, Tuple
from src.inertial.imu_types import so3_exp, so3_log

def equirectangular_to_bearing(u: np.ndarray, v: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Project 2D equirectangular pixel coordinates to 3D unit sphere bearing vectors.
    Ray convention (Camera frame):
    - X: Right
    - Y: Down
    - Z: Forward (center of panorama)
    """
    theta = (u / float(width) - 0.5) * (2.0 * np.pi)       # Azimuth in [-pi, pi]
    phi = (0.5 - v / float(height)) * np.pi                # Elevation in [-pi/2, pi/2]

    # Bearing vector in camera coordinates
    x = np.cos(phi) * np.sin(theta)
    y = -np.sin(phi)
    z = np.cos(phi) * np.cos(theta)

    rays = np.stack([x, y, z], axis=-1)
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays / np.maximum(norms, 1e-12)


class VideoReader:
    """Reads 360 panoramic video frames and manages presentation timestamps."""

    def __init__(self, video_path: str, target_width: Optional[int] = None, t0_sec: float = 0.0):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = os.path.abspath(video_path)
        self.t0_sec = float(t0_sec)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {self.video_path}")

        self.orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.frame_count / self.fps if self.fps > 0 else 0.0
        cap.release()

        self.target_width = target_width if target_width is not None else self.orig_width
        self.target_height = int(self.target_width / 2) if target_width is not None else self.orig_height

    def frame_timestamp(self, frame_idx: int) -> float:
        """Exact presentation timestamp in seconds for a given frame index."""
        return self.t0_sec + (frame_idx / self.fps)

    def frames(self, start_frame: int = 0, max_frames: Optional[int] = None) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Generator yielding (frame_index, timestamp_sec, bgr_image)."""
        cap = cv2.VideoCapture(self.video_path)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        curr_idx = start_frame
        yielded = 0

        while cap.isOpened():
            if max_frames is not None and yielded >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            t = self.frame_timestamp(curr_idx)
            if self.target_width != self.orig_width or self.target_height != self.orig_height:
                frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)

            yield curr_idx, t, frame
            curr_idx += 1
            yielded += 1

        cap.release()


class SphericalVisualGyro:
    """
    Estimates relative 3D camera rotation and angular velocity directly from
    equirectangular frame pairs by tracking ORB features on the unit sphere.
    """

    def __init__(self, n_features: int = 1500):
        self.orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def estimate_relative_rotation(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Estimate rotation R in SO(3) such that ray2 approx R.T @ ray1.
        Returns: (R, num_inliers) or (None, 0) if insufficient matches.
        """
        h, w = img1.shape[:2]
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
            return None, 0

        matches = self.matcher.match(des1, des2)
        if len(matches) < 15:
            return None, 0

        # Sort matches by distance
        matches = sorted(matches, key=lambda m: m.distance)
        top_matches = matches[: min(300, len(matches))]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in top_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in top_matches])

        # Convert to bearing vectors on unit sphere
        rays1 = equirectangular_to_bearing(pts1[:, 0], pts1[:, 1], w, h)
        rays2 = equirectangular_to_bearing(pts2[:, 0], pts2[:, 1], w, h)

        # RANSAC Kabsch / Horn rotation alignment on bearing rays
        best_R = None
        best_inliers = 0
        inlier_thresh_rad = np.radians(2.0) # 2 degrees threshold

        np.random.seed(42)
        n_samples = len(rays1)
        if n_samples < 8:
            return None, 0

        for _ in range(80):
            sample_idx = np.random.choice(n_samples, size=4, replace=False)
            r1_sample = rays1[sample_idx]
            r2_sample = rays2[sample_idx]

            # Kabsch algorithm for pure rotation: R = V @ U.T where SVD(r2.T @ r1) = U @ S @ V.T
            H = r2_sample.T @ r1_sample
            U, S, Vt = np.linalg.svd(H)
            R_cand = Vt.T @ U.T
            if np.linalg.det(R_cand) < 0:
                Vt[2, :] *= -1
                R_cand = Vt.T @ U.T

            # Inlier check: angle between ray2 and R_cand.T @ ray1
            pred_rays2 = (R_cand.T @ rays1.T).T
            dot_prods = np.clip(np.sum(rays2 * pred_rays2, axis=1), -1.0, 1.0)
            angular_errors = np.arccos(dot_prods)

            inliers = np.sum(angular_errors < inlier_thresh_rad)
            if inliers > best_inliers:
                best_inliers = inliers
                best_R = R_cand

        if best_inliers < 12 or best_R is None:
            return None, 0

        # Refine with all inliers
        pred_rays2 = (best_R.T @ rays1.T).T
        dot_prods = np.clip(np.sum(rays2 * pred_rays2, axis=1), -1.0, 1.0)
        inlier_mask = np.arccos(dot_prods) < inlier_thresh_rad

        r1_in = rays1[inlier_mask]
        r2_in = rays2[inlier_mask]
        H = r2_in.T @ r1_in
        U, S, Vt = np.linalg.svd(H)
        R_refined = Vt.T @ U.T
        if np.linalg.det(R_refined) < 0:
            Vt[2, :] *= -1
            R_refined = Vt.T @ U.T

        return R_refined, int(best_inliers)
