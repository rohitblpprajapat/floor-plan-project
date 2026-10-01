"""Video frame decoding, timelapse inspection, and contiguous pilot selection.

Conventions:
- Preserves exact source frame indices and original presentation timestamps (PTS).
- Never manufactures or interpolates synthetic images.
- Does not use a default 1-frame-per-playback-second policy.
- Inspects all available frames before pilot selection.
- Selects a contiguous overlapping pilot up to 40 panoramas, preserving essential neighbors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple
import cv2
import numpy as np

from vscan.frames.renderer import EquirectangularRenderer
from vscan.frames.scoring import evaluate_panorama_quality
from vscan.schemas.frames import FrameQualityScore


class VideoFrameDecoder:
    """Sequential video reader that decodes frames and preserves exact presentation timing."""

    def __init__(self, video_path: Path) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video container: {self.video_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __enter__(self) -> VideoFrameDecoder:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def iter_frames(self) -> Iterator[Tuple[int, int, float, np.ndarray]]:
        """Yield (frame_index, pts_integer_ms, pts_seconds, frame_bgr) sequentially."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break

            # Presentation timestamp from container (in milliseconds)
            pts_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            if pts_ms < 0 or (pts_ms == 0.0 and frame_idx > 0):
                # Fallback to nominal frame cadence if container lacks per-frame millisecond tags
                pts_seconds = frame_idx / self.fps
                pts_int_ms = int(round(pts_seconds * 1000.0))
            else:
                pts_int_ms = int(round(pts_ms))
                pts_seconds = pts_int_ms / 1000.0

            yield frame_idx, pts_int_ms, pts_seconds, frame
            frame_idx += 1

    def get_frame(self, target_index: int) -> Optional[Tuple[int, int, float, np.ndarray]]:
        """Seek to specific frame index and return (frame_index, pts_ms, pts_seconds, frame_bgr)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        pts_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        if pts_ms < 0:
            pts_seconds = target_index / self.fps
            pts_int_ms = int(round(pts_seconds * 1000.0))
        else:
            pts_int_ms = int(round(pts_ms))
            pts_seconds = pts_int_ms / 1000.0

        return target_index, pts_int_ms, pts_seconds, frame


def inspect_and_score_video_frames(
    video_path: Path,
    renderer: EquirectangularRenderer,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[Tuple[int, int, float, FrameQualityScore]]:
    """Inspect all frames in video and compute perspective quality scores."""
    inspected: List[Tuple[int, int, float, FrameQualityScore]] = []
    prev_views: Optional[Dict[str, np.ndarray]] = None

    with VideoFrameDecoder(video_path) as decoder:
        total = decoder.frame_count or 1
        for f_idx, pts_ms, pts_s, frame in decoder.iter_frames():
            # Render horizontal cardinal faces for quality evaluation
            rendered = {
                "face_front": renderer.render_face(frame, "front"),
                "face_right": renderer.render_face(frame, "right"),
                "face_back": renderer.render_face(frame, "back"),
                "face_left": renderer.render_face(frame, "left"),
            }
            score = evaluate_panorama_quality(rendered, prev_views)
            inspected.append((f_idx, pts_ms, pts_s, score))
            prev_views = rendered

            if progress_callback and (f_idx % 10 == 0 or f_idx == total - 1):
                pct = (f_idx + 1) / total * 100.0
                progress_callback(pct, f"Inspecting source frame {f_idx + 1}/{total}")

    return inspected


def select_contiguous_pilot_frames(
    inspected_frames: List[Tuple[int, int, float, FrameQualityScore]],
    max_pilot_panoramas: int = 40,
    policy: str = "contiguous_window",
) -> List[int]:
    """Select a contiguous overlapping sequence of up to max_pilot_panoramas.

    Policies:
    - 'contiguous_window': Selects the best contiguous block of up to max_pilot_panoramas with highest cumulative quality.
    - 'contiguous_strided': If session is long, subsamples with a fixed small stride to cover the session while maintaining overlap.
    """
    total = len(inspected_frames)
    if total <= max_pilot_panoramas:
        # All available frames retained; never invent frames
        return [f[0] for f in inspected_frames]

    if policy == "contiguous_window":
        # Find contiguous window of length max_pilot_panoramas with highest sum of composite quality
        w_size = min(max_pilot_panoramas, total)
        scores = [f[3].composite_quality for f in inspected_frames]

        best_start = 0
        best_sum = sum(scores[:w_size])
        curr_sum = best_sum

        for i in range(1, total - w_size + 1):
            curr_sum = curr_sum - scores[i - 1] + scores[i + w_size - 1]
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_start = i

        selected_indices = [inspected_frames[best_start + k][0] for k in range(w_size)]
        return selected_indices

    elif policy == "contiguous_strided":
        # Evenly spaced contiguous-overlapping frames
        stride = max(1, int(round(total / float(max_pilot_panoramas))))
        selected_indices = [inspected_frames[i][0] for i in range(0, total, stride)][:max_pilot_panoramas]
        return selected_indices

    else:
        # Default fallback to first N contiguous frames
        return [f[0] for f in inspected_frames[:max_pilot_panoramas]]
