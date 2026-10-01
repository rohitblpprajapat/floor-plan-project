"""Quality evaluation for perspective regions: sharpness, exposure, and inter-frame redundancy.

Conventions:
- Evaluates metrics on undistorted perspective regions, not distorted equirectangular panoramas.
- Sharpness: Laplacian variance and Tenengrad gradient energy.
- Exposure: Mean luminance, dark/bright clipping fractions, and dynamic range penalty.
- Redundancy: Inter-frame photometric difference across perspective views.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from vscan.schemas.frames import FrameQualityScore


def score_sharpness(image_gray: np.ndarray) -> float:
    """Compute sharpness using variance of Laplacian augmented by Tenengrad gradient energy."""
    if image_gray.ndim != 2:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)

    # 1. Variance of Laplacian
    lap = cv2.Laplacian(image_gray, cv2.CV_64F, ksize=3)
    lap_var = float(np.var(lap))

    # 2. Tenengrad gradient magnitude
    gx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    tenengrad = float(np.mean(grad_mag))

    # Combined normalized sharpness metric
    return float(lap_var + 10.0 * tenengrad)


def score_exposure(image_gray: np.ndarray) -> Tuple[float, float, float, float]:
    """Evaluate exposure balance: mean luminance, dark clipping, bright clipping, and composite score.

    Returns:
        (mean_luminance, dark_fraction, bright_fraction, exposure_score_0_to_1)
    """
    if image_gray.ndim != 2:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)

    mean_lum = float(np.mean(image_gray))
    dark_frac = float(np.mean(image_gray < 5))
    bright_frac = float(np.mean(image_gray > 250))

    # Optimal indoor/outdoor midtone luminance is ~115-135
    target_lum = 125.0
    lum_dist = abs(mean_lum - target_lum) / target_lum  # 0 is optimal, 1 is bad
    lum_score = max(0.0, 1.0 - lum_dist)

    # Clipping penalty
    clipping_penalty = (dark_frac + bright_frac) * 2.0
    exposure_score = max(0.0, min(1.0, lum_score * (1.0 - min(1.0, clipping_penalty))))

    return mean_lum, dark_frac, bright_frac, float(exposure_score)


def score_inter_frame_difference(
    curr_image_gray: np.ndarray,
    prev_image_gray: np.ndarray,
) -> float:
    """Evaluate motion change between corresponding perspective views of successive frames."""
    if curr_image_gray.shape != prev_image_gray.shape:
        return 1.0

    if curr_image_gray.ndim != 2:
        curr_image_gray = cv2.cvtColor(curr_image_gray, cv2.COLOR_BGR2GRAY)
    if prev_image_gray.ndim != 2:
        prev_image_gray = cv2.cvtColor(prev_image_gray, cv2.COLOR_BGR2GRAY)

    diff = np.abs(curr_image_gray.astype(np.float32) - prev_image_gray.astype(np.float32))
    norm_diff = float(np.mean(diff) / 255.0)
    return norm_diff


def evaluate_panorama_quality(
    perspective_views: Dict[str, np.ndarray],
    prev_perspective_views: Optional[Dict[str, np.ndarray]] = None,
) -> FrameQualityScore:
    """Compute overall quality score across horizontal cardinal perspective faces."""
    cardinal_faces = ["face_front", "face_right", "face_back", "face_left"]

    sharpness_vals = []
    mean_lums = []
    dark_fracs = []
    bright_fracs = []
    exposure_scores = []
    motion_diffs = []

    for name in cardinal_faces:
        if name not in perspective_views:
            continue
        view_bgr = perspective_views[name]
        view_gray = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2GRAY) if view_bgr.ndim == 3 else view_bgr

        # Sharpness
        s = score_sharpness(view_gray)
        sharpness_vals.append(s)

        # Exposure
        ml, df, bf, es = score_exposure(view_gray)
        mean_lums.append(ml)
        dark_fracs.append(df)
        bright_fracs.append(bf)
        exposure_scores.append(es)

        # Motion difference
        if prev_perspective_views and name in prev_perspective_views:
            prev_bgr = prev_perspective_views[name]
            prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY) if prev_bgr.ndim == 3 else prev_bgr
            diff = score_inter_frame_difference(view_gray, prev_gray)
            motion_diffs.append(diff)

    avg_sharpness = float(np.mean(sharpness_vals)) if sharpness_vals else 0.0
    avg_mean_lum = float(np.mean(mean_lums)) if mean_lums else 128.0
    avg_dark_frac = float(np.mean(dark_fracs)) if dark_fracs else 0.0
    avg_bright_frac = float(np.mean(bright_fracs)) if bright_fracs else 0.0
    avg_exposure_score = float(np.mean(exposure_scores)) if exposure_scores else 0.5
    avg_motion_diff = float(np.mean(motion_diffs)) if motion_diffs else 0.0

    # Composite quality score
    # Balance sharpness (scaled logarithmically) and exposure
    sharp_norm = math.log1p(avg_sharpness) / 10.0
    composite = sharp_norm * 0.6 + avg_exposure_score * 0.4

    return FrameQualityScore(
        sharpness_score=avg_sharpness,
        exposure_mean=avg_mean_lum,
        clipped_dark_fraction=avg_dark_frac,
        clipped_bright_fraction=avg_bright_frac,
        exposure_score=avg_exposure_score,
        inter_frame_change=avg_motion_diff,
        composite_quality=composite,
    )
