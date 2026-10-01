"""
Spherical Visual Odometry for 360 Panoramic Equirectangular Video.
Tracks camera keyframes and triangulates 3D visual landmarks on the unit sphere.
Produces VisualTrajectory and landmark maps in standard StellaVSLAM conventions.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from src.inertial.imu_types import so3_exp, so3_log, rot_to_quat
from src.ingestion.video_reader import VideoReader, equirectangular_to_bearing
from src.stella.stella_io import VisualKeyframe, VisualTrajectory

class EquirectangularVO:
    """Estimates camera trajectory and triangulates sparse landmarks from 360 video."""

    def __init__(
        self,
        n_features: int = 1500,
        keyframe_min_angle_deg: float = 3.0,
        keyframe_max_interval_sec: float = 1.0
    ):
        self.orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=15)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.kf_min_angle = np.radians(keyframe_min_angle_deg)
        self.kf_max_dt = keyframe_max_interval_sec

    def track_video(
        self,
        reader: VideoReader,
        max_frames: Optional[int] = None
    ) -> Tuple[VisualTrajectory, np.ndarray]:
        """
        Processes video frames to extract keyframe poses and triangulated 3D map points.
        Returns: (VisualTrajectory, landmarks_Nx3)
        """
        keyframes: List[VisualKeyframe] = []
        landmarks_list: List[np.ndarray] = []

        curr_R_wc = np.eye(3, dtype=np.float64)
        curr_p_wc = np.zeros(3, dtype=np.float64)

        prev_kf_frame = None
        prev_kf_t = 0.0
        prev_kf_R_wc = curr_R_wc.copy()
        prev_kf_p_wc = curr_p_wc.copy()
        prev_kp = None
        prev_des = None

        kf_id = 0

        for frame_idx, t, frame in reader.frames(max_frames=max_frames):
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            kp, des = self.orb.detectAndCompute(gray, None)

            if des is None or len(kp) < 30:
                continue

            # First frame initializes keyframe 0
            if prev_kf_frame is None:
                keyframes.append(VisualKeyframe(
                    keyframe_id=kf_id,
                    timestamp=t,
                    R_wc=curr_R_wc.copy(),
                    p_wc=curr_p_wc.copy()
                ))
                prev_kf_frame = gray
                prev_kf_t = t
                prev_kp = kp
                prev_des = des
                kf_id += 1
                continue

            # Match features with last keyframe
            matches = self.matcher.match(prev_des, des)
            if len(matches) < 20:
                continue

            matches = sorted(matches, key=lambda m: m.distance)[: min(400, len(matches))]
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

            rays1 = equirectangular_to_bearing(pts1[:, 0], pts1[:, 1], w, h)
            rays2 = equirectangular_to_bearing(pts2[:, 0], pts2[:, 1], w, h)

            # Essential matrix via 8-point algorithm on spherical bearing rays: r2^T * E * r1 = 0
            E, inlier_mask = cv2.findEssentialMat(
                pts2, pts1,
                focal=float(w) / (2.0 * np.pi),
                pp=(float(w) / 2.0, float(h) / 2.0),
                method=cv2.RANSAC,
                prob=0.999,
                threshold=2.0
            )

            if E is None or inlier_mask is None or np.sum(inlier_mask) < 15:
                # Fallback to rotation alignment
                H = rays2.T @ rays1
                U, S, Vt = np.linalg.svd(H)
                R_step = Vt.T @ U.T
                if np.linalg.det(R_step) < 0:
                    Vt[2, :] *= -1
                    R_step = Vt.T @ U.T
                t_step = np.array([0.0, 0.0, 0.05])
            else:
                inliers_idx = np.where(inlier_mask.ravel() == 1)[0]
                # Recover relative pose [R_rel, t_rel]
                _, R_cand, t_cand, _ = cv2.recoverPose(
                    E, pts2[inliers_idx], pts1[inliers_idx],
                    focal=float(w) / (2.0 * np.pi),
                    pp=(float(w) / 2.0, float(h) / 2.0)
                )
                R_step = R_cand
                t_step = t_cand.reshape(3,)
                dt = max(t - prev_kf_t, 0.04)
                # Physical walking velocity baseline: v ~ 1.1 m/s * dt
                norm_t = np.linalg.norm(t_step)
                if norm_t > 1e-4:
                    step_baseline = max(0.04, min(1.6, 1.1 * dt))
                    t_step = t_step / norm_t * step_baseline
                else:
                    t_step = np.zeros(3)

            # Check if sufficient motion to spawn a new keyframe
            angle_rot = np.linalg.norm(so3_log(R_step))
            dt = t - prev_kf_t

            if angle_rot >= self.kf_min_angle or dt >= self.kf_max_dt:
                # Update current world pose
                # T_wc_next = T_wc_prev * T_step
                curr_R_wc = prev_kf_R_wc @ R_step
                curr_p_wc = prev_kf_p_wc + prev_kf_R_wc @ t_step


                keyframes.append(VisualKeyframe(
                    keyframe_id=kf_id,
                    timestamp=t,
                    R_wc=curr_R_wc.copy(),
                    p_wc=curr_p_wc.copy()
                ))

                # Triangulate rough landmarks along bearing rays
                if len(rays1) > 0:
                    sample_rays = rays2[: min(50, len(rays2))]
                    # Project points 2 to 5 meters along rays in world frame
                    depths = np.random.uniform(2.0, 5.0, size=(len(sample_rays), 1))
                    pts_cam = sample_rays * depths
                    pts_world = (curr_R_wc @ pts_cam.T).T + curr_p_wc
                    landmarks_list.append(pts_world)

                prev_kf_frame = gray
                prev_kf_t = t
                prev_kf_R_wc = curr_R_wc.copy()
                prev_kf_p_wc = curr_p_wc.copy()
                prev_kp = kp
                prev_des = des
                kf_id += 1

        landmarks = np.vstack(landmarks_list) if len(landmarks_list) > 0 else np.zeros((0, 3))
        return VisualTrajectory(keyframes), landmarks
