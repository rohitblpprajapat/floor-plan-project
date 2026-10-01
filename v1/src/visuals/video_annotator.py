"""
Video & Image Visual Annotator.
Overlays reconstructed metric telemetry, 3D landmark re-projections, and an animated
2D top-down mini-map HUD directly onto 360 panoramic equirectangular frames.
"""

import os
import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple
from src.stella.stella_io import VisualTrajectory, VisualKeyframe
from src.ingestion.video_reader import VideoReader

class VideoAnnotator:
    """Renders augmented reality overlays, telemetry HUDs, and mini-map PiPs."""

    def __init__(self, trajectory: VisualTrajectory, landmarks: np.ndarray):
        self.trajectory = trajectory
        self.landmarks = landmarks
        self.positions = trajectory.positions
        self.timestamps = trajectory.timestamps

        # Precompute bounding box of trajectory for mini-map
        self.min_x = np.min(self.positions[:, 0]) - 0.5
        self.max_x = np.max(self.positions[:, 0]) + 0.5
        self.min_y = np.min(self.positions[:, 1]) - 0.5
        self.max_y = np.max(self.positions[:, 1]) + 0.5

    def reproject_landmarks_to_equirectangular(
        self,
        R_wc: np.ndarray,
        p_wc: np.ndarray,
        width: int,
        height: int,
        max_points: int = 400
    ) -> List[Tuple[int, int, float]]:
        """
        Projects 3D world landmarks into 360 equirectangular pixel coordinates (u, v, depth).
        """
        if len(self.landmarks) == 0:
            return []

        # Transform to camera frame: X_c = R_wc^T * (X_w - p_wc)
        diff = self.landmarks - p_wc # (N, 3)
        pts_c = (R_wc.T @ diff.T).T # (N, 3)

        dists = np.linalg.norm(pts_c, axis=1)
        valid = (dists > 0.3) & (dists < 12.0)
        pts_c_valid = pts_c[valid]
        dists_valid = dists[valid]

        if len(pts_c_valid) == 0:
            return []

        # Subsample if too dense
        if len(pts_c_valid) > max_points:
            idx = np.random.choice(len(pts_c_valid), max_points, replace=False)
            pts_c_valid = pts_c_valid[idx]
            dists_valid = dists_valid[idx]

        # Bearing spherical angles
        # x: right, y: down/up, z: optical axis forward
        # In equirectangular:
        # phi = atan2(x, z) in [-pi, pi]
        # theta = asin(-y / dist) in [-pi/2, pi/2]
        x = pts_c_valid[:, 0]
        y = pts_c_valid[:, 1]
        z = pts_c_valid[:, 2]

        phi = np.arctan2(x, z)
        theta = np.arcsin(np.clip(-y / dists_valid, -1.0, 1.0))

        u = ((phi / (2.0 * np.pi) + 0.5) * width).astype(np.int32)
        v = (((0.5 - theta / np.pi)) * height).astype(np.int32)

        # Keep inside image bounds
        in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u_in = u[in_bounds]
        v_in = v[in_bounds]
        d_in = dists_valid[in_bounds]

        return list(zip(u_in, v_in, d_in))

    def render_minimap(self, curr_idx: int, map_size: int = 220) -> np.ndarray:
        """
        Renders a 2D floor plan mini-map with the current camera position and heading arrow.
        """
        minimap = np.full((map_size, map_size, 3), (24, 24, 30), dtype=np.uint8)

        # Coordinate transformation from metric meters to mini-map pixels
        span_x = max(self.max_x - self.min_x, 1e-3)
        span_y = max(self.max_y - self.min_y, 1e-3)
        scale = (map_size - 30) / max(span_x, span_y)

        def to_px(x, y):
            px = int(15 + (x - self.min_x) * scale)
            py = int(map_size - 15 - (y - self.min_y) * scale)
            return px, py

        # Draw full trajectory path
        for i in range(len(self.positions) - 1):
            p1 = to_px(self.positions[i, 0], self.positions[i, 1])
            p2 = to_px(self.positions[i + 1, 0], self.positions[i + 1, 1])
            cv2.line(minimap, p1, p2, (80, 80, 100), 1, cv2.LINE_AA)

        # Draw traversed path up to current keyframe
        for i in range(min(curr_idx, len(self.positions) - 1)):
            p1 = to_px(self.positions[i, 0], self.positions[i, 1])
            p2 = to_px(self.positions[i + 1, 0], self.positions[i + 1, 1])
            cv2.line(minimap, p1, p2, (255, 0, 128), 2, cv2.LINE_AA)

        # Draw start marker (green)
        p_start = to_px(self.positions[0, 0], self.positions[0, 1])
        cv2.circle(minimap, p_start, 4, (0, 255, 100), -1, cv2.LINE_AA)

        # Draw current camera position (bright yellow-cyan pulsating)
        curr_p = self.positions[min(curr_idx, len(self.positions) - 1)]
        p_curr = to_px(curr_p[0], curr_p[1])
        cv2.circle(minimap, p_curr, 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(minimap, p_curr, 8, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw heading arrow
        kf = self.trajectory.keyframes[min(curr_idx, len(self.trajectory) - 1)]
        fwd_2d = kf.R_wc @ np.array([0.0, 0.0, 0.4])
        p_fwd = to_px(curr_p[0] + fwd_2d[0], curr_p[1] + fwd_2d[1])
        cv2.arrowedLine(minimap, p_curr, p_fwd, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)

        # Outer border
        cv2.rectangle(minimap, (0, 0), (map_size - 1, map_size - 1), (60, 60, 80), 2)
        cv2.putText(minimap, "FLOOR PLAN HUD", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
        return minimap

    def annotate_frame(
        self,
        frame: np.ndarray,
        keyframe_idx: int,
        draw_landmarks: bool = True,
        draw_hud: bool = True,
        draw_minimap: bool = True
    ) -> np.ndarray:
        """
        Overlays HUD telemetry, 3D landmarks, and mini-map onto a single frame.
        """
        out = frame.copy()
        h, w = out.shape[:2]
        kf_idx = min(keyframe_idx, len(self.trajectory) - 1)
        kf = self.trajectory.keyframes[kf_idx]

        # 1. Overlay 3D Landmarks Re-projected
        if draw_landmarks:
            reproj_pts = self.reproject_landmarks_to_equirectangular(kf.R_wc, kf.p_wc, w, h)
            for u_px, v_px, depth in reproj_pts:
                # Color code by depth: close = green, mid = yellow, far = cyan
                if depth < 2.0:
                    color = (0, 255, 128)
                elif depth < 5.0:
                    color = (0, 230, 255)
                else:
                    color = (255, 200, 0)
                cv2.circle(out, (u_px, v_px), 3, color, -1, cv2.LINE_AA)

        # 2. Telemetry HUD Top Bar
        if draw_hud:
            # Semi-transparent top ribbon
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 20), -1)
            cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

            # Compute current metrics
            t_curr = kf.timestamp
            dist_curr = float(np.sum(np.linalg.norm(np.diff(self.positions[: kf_idx + 1], axis=0), axis=1))) if kf_idx > 0 else 0.0
            dt_step = self.timestamps[kf_idx] - self.timestamps[kf_idx - 1] if kf_idx > 0 else 1.0
            ds_step = np.linalg.norm(self.positions[kf_idx] - self.positions[kf_idx - 1]) if kf_idx > 0 else 0.0
            speed_curr = ds_step / max(dt_step, 1e-3)

            # Text labels
            cv2.putText(out, f"360 SURVEY RECONSTRUCTION", (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 1, cv2.LINE_AA)
            hud_info = (
                f"Time: {t_curr:.2f}s | KF: #{kf.keyframe_id:03d} | "
                f"Distance: {dist_curr:.2f}m | Speed: {speed_curr:.2f} m/s ({speed_curr * 3.6:.1f} km/h) | "
                f"Height Z: {kf.p_wc[2]:+.2f}m"
            )
            cv2.putText(out, hud_info, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

            # Artificial horizon indicator (pitch and roll)
            roll_deg = float(np.degrees(np.arctan2(kf.R_wc[2, 1], kf.R_wc[2, 2])))
            pitch_deg = float(np.degrees(-np.arcsin(np.clip(kf.R_wc[2, 0], -1.0, 1.0))))
            cv2.putText(out, f"Pitch: {pitch_deg:+.1f} | Roll: {roll_deg:+.1f}", (w - 240, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA)

        # 3. Picture-in-Picture Mini-Map
        if draw_minimap:
            m_size = min(220, int(h * 0.4))
            m_img = self.render_minimap(kf_idx, map_size=m_size)
            # Paste into bottom right with 20px padding
            x_offset = w - m_size - 20
            y_offset = h - m_size - 20
            out[y_offset : y_offset + m_size, x_offset : x_offset + m_size] = m_img

        return out

    def generate_annotated_video(
        self,
        reader: VideoReader,
        output_video_path: str,
        max_frames: Optional[int] = None,
        fps: Optional[float] = None
    ) -> str:
        """
        Produces an annotated replay MP4 video with real-time HUD and floor plan tracking.
        """
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fps = fps if fps is not None else reader.fps

        # Probe first frame for size
        first_frame = None
        for _, _, f in reader.frames(max_frames=1):
            first_frame = f
            break
        if first_frame is None:
            raise RuntimeError("Could not read any frames from video reader.")

        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

        # Build timestamp to keyframe index mapping
        kf_ts = np.array([kf.timestamp for kf in self.trajectory.keyframes])

        frame_count = 0
        for frame_idx, t, frame in reader.frames(max_frames=max_frames):
            # Find closest keyframe
            kf_idx = int(np.argmin(np.abs(kf_ts - t)))
            annotated = self.annotate_frame(frame, kf_idx)
            writer.write(annotated)
            frame_count += 1

        writer.release()
        return output_video_path
