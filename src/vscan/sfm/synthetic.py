"""Vectorized synthetic room renderer, test fixtures, and Sim(3) alignment evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from vscan.frames.projection import (
    CUBEMAP_FACES,
    CUBEMAP_ROTATIONS,
    PinholeIntrinsics,
    ReprojectionFace,
    build_virtual_camera_poses,
    get_default_pinhole_intrinsics,
    pixel_to_ray_pinhole,
)
from vscan.frames.rig import export_rig_config_json
from vscan.frames.synthetic import SyntheticRoom, generate_synthetic_rotation_only_fixture


def render_pinhole_vectorized(
    room: SyntheticRoom,
    camera_center: np.ndarray,
    R_pano_cam: np.ndarray,
    intrinsics: PinholeIntrinsics,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast vectorized analytic pinhole renderer for SyntheticRoom surfaces."""
    w = intrinsics.width
    h = intrinsics.height

    u_px, v_px = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    xc, yc, zc = pixel_to_ray_pinhole(u_px, v_px, intrinsics)
    d_c = np.stack([xc, yc, zc], axis=-1)  # (H, W, 3)

    R = np.asarray(R_pano_cam, dtype=np.float64)
    # Rotate ray directions into room panorama frame: d_pano = d_c @ R^T
    d_pano = d_c @ R.T
    origin = np.asarray(camera_center, dtype=np.float64)

    best_t = np.full((h, w), np.inf, dtype=np.float64)
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

    eps = 1e-4
    for surf in room.surfaces:
        n = surf.normal
        denom = np.dot(d_pano, n)
        valid_denom = np.abs(denom) > 1e-6
        # t = -(n . origin + d) / (n . d_pano)
        num = -(float(np.dot(n, origin)) + surf.d)
        t = np.where(valid_denom, num / np.where(valid_denom, denom, 1.0), np.inf)

        valid_t = (t > 1e-4) & (t < best_t)
        if not np.any(valid_t):
            continue

        p = origin + t[:, :, None] * d_pano
        in_bounds = (
            valid_t
            & (p[:, :, 0] >= surf.bounds_x[0] - eps)
            & (p[:, :, 0] <= surf.bounds_x[1] + eps)
            & (p[:, :, 1] >= surf.bounds_y[0] - eps)
            & (p[:, :, 1] <= surf.bounds_y[1] + eps)
            & (p[:, :, 2] >= surf.bounds_z[0] - eps)
            & (p[:, :, 2] <= surf.bounds_z[1] + eps)
        )
        if not np.any(in_bounds):
            continue

        best_t = np.where(in_bounds, t, best_t)

        # Procedural texture
        if abs(n[0]) > 0.8:
            u_c = p[:, :, 2] * surf.texture_scale
            v_c = p[:, :, 1] * surf.texture_scale
        elif abs(n[1]) > 0.8:
            u_c = p[:, :, 0] * surf.texture_scale
            v_c = p[:, :, 2] * surf.texture_scale
        else:
            u_c = p[:, :, 0] * surf.texture_scale
            v_c = p[:, :, 1] * surf.texture_scale

        u_c_safe = np.nan_to_num(u_c, nan=0.0, posinf=0.0, neginf=0.0)
        v_c_safe = np.nan_to_num(v_c, nan=0.0, posinf=0.0, neginf=0.0)
        check = (np.floor(u_c_safe).astype(int) + np.floor(v_c_safe).astype(int)) % 2
        color_mask = check == 0
        c1 = np.array(surf.color1, dtype=np.uint8)
        c2 = np.array(surf.color2, dtype=np.uint8)
        surf_rgb = np.where(color_mask[:, :, None], c1, c2)
        rgb_img = np.where(in_bounds[:, :, None], surf_rgb, rgb_img)

    depth_img = np.where(np.isfinite(best_t), best_t.astype(np.float32), 0.0)
    return rgb_img, depth_img


def generate_synthetic_sfm_dataset(
    target_dir: Path,
    num_stations: int = 5,
    resolution: int = 320,
    fov_deg: float = 90.0,
    trajectory_type: str = "translation",
) -> Dict[str, Any]:
    """Generate complete synthetic dataset with images, masks, rig config, and ground truth."""
    target_dir.mkdir(parents=True, exist_ok=True)
    images_dir = target_dir / "pilot_frames"
    masks_dir = target_dir / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    room = SyntheticRoom()
    intrinsics = get_default_pinhole_intrinsics(resolution, resolution, fov_deg)
    poses = build_virtual_camera_poses(intrinsics)

    rig_config_path = target_dir / "rig_config.json"
    export_rig_config_json(poses, rig_config_path)

    # Station positions
    gt_stations = []
    if trajectory_type == "rotation_only":
        # Pure rotation around Y, zero translation baseline
        for i in range(num_stations):
            yaw_deg = float(i) * 15.0
            yaw_rad = math.radians(yaw_deg)
            cos_y = math.cos(yaw_rad)
            sin_y = math.sin(yaw_rad)
            R_yaw = np.array([
                [cos_y, 0.0, sin_y],
                [0.0, 1.0, 0.0],
                [-sin_y, 0.0, cos_y],
            ], dtype=np.float64)
            gt_stations.append({
                "station_index": i,
                "camera_center": [0.0, 0.0, 0.0],
                "rotation_matrix": R_yaw,
            })
    else:
        # Non-collinear translation trajectory along Z and X inside room bounds
        for i in range(num_stations):
            frac = float(i) / max(1, num_stations - 1)
            # Center moves from Z = -0.8 to +0.8 with an X offset to break strict collinearity
            z = -0.8 + 1.6 * frac
            x = 0.25 * math.sin(frac * math.pi)
            y = 0.0
            gt_stations.append({
                "station_index": i,
                "camera_center": [x, y, z],
                "rotation_matrix": np.eye(3, dtype=np.float64),
            })

    # Render images and masks
    manifest_panos = []
    for st in gt_stations:
        s_idx = st["station_index"]
        c_world = np.array(st["camera_center"], dtype=np.float64)
        R_station = st["rotation_matrix"]

        station_views = {}
        for face in CUBEMAP_FACES:
            R_face_local = CUBEMAP_ROTATIONS[face]
            R_cam_world = R_station @ R_face_local

            rgb, depth = render_pinhole_vectorized(room, c_world, R_cam_world, intrinsics)

            # Name format: face_{face.value}_frame_{station:06d}.jpg
            img_filename = f"face_{face.value}_frame_{s_idx:06d}.jpg"
            img_path = images_dir / img_filename
            Image.fromarray(rgb).save(img_path)

            # Masks: face_down gets nadir mask
            mask_filename = f"{img_filename}.png"
            mask_path = masks_dir / mask_filename
            mask_arr = np.full((resolution, resolution), 255, dtype=np.uint8)
            if face == ReprojectionFace.DOWN:
                # Bottom nadir circle exclusion
                cy, cx = resolution // 2, resolution // 2
                rad = int(resolution * 0.22)
                y_idx, x_idx = np.ogrid[:resolution, :resolution]
                mask_arr[(x_idx - cx)**2 + (y_idx - cy)**2 <= rad**2] = 0
            Image.fromarray(mask_arr).save(mask_path)

            cam_key = f"face_{face.value}"
            station_views[cam_key] = {
                "view_id": f"frame_{s_idx:06d}_{face.value}",
                "camera_name": cam_key,
                "image_rel_path": f"pilot_frames/{img_filename}",
                "mask_rel_path": f"masks/{mask_filename}",
                "width": resolution,
                "height": resolution,
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy,
                "R_pano_cam": R_face_local.tolist(),
                "t_pano_cam": [0.0, 0.0, 0.0],
            }

        manifest_panos.append({
            "panorama_index": s_idx,
            "pts_seconds": float(s_idx) * 0.1,
            "ground_truth_center": list(st["camera_center"]),
            "views": station_views,
        })

    # Save manifest
    manifest_data = {
        "schema_version": "1.0.0",
        "session_id": f"synthetic_{trajectory_type}",
        "actual_data_readiness": "SYNTHETIC_ONLY",
        "scale_status": "NOT_ESTIMATED",
        "selected_pilot_frames_count": len(gt_stations),
        "pilot_panoramas": manifest_panos,
    }

    import json
    manifest_path = target_dir / "pilot_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return {
        "manifest_path": manifest_path,
        "rig_config_path": rig_config_path,
        "images_dir": images_dir,
        "masks_dir": masks_dir,
        "gt_stations": gt_stations,
        "intrinsics": intrinsics,
    }


def align_sim3_umeyama(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Compute optimal similarity transformation Sim(3): tgt = s * R @ src + t using Umeyama algorithm.

    Returns:
        (scale, rotation_matrix_3x3, translation_vector_3, rmse_error)
    """
    src = np.asarray(src_points, dtype=np.float64)
    tgt = np.asarray(tgt_points, dtype=np.float64)

    assert src.ndim == 2 and src.shape[1] == 3
    assert tgt.ndim == 2 and tgt.shape[1] == 3
    assert src.shape[0] == tgt.shape[0]
    n = src.shape[0]

    mu_src = np.mean(src, axis=0)
    mu_tgt = np.mean(tgt, axis=0)

    src_centered = src - mu_src
    tgt_centered = tgt - mu_tgt

    var_src = float(np.mean(np.sum(src_centered**2, axis=1)))
    if var_src < 1e-12:
        return 1.0, np.eye(3), mu_tgt - mu_src, 0.0

    # Covariance matrix H = (tgt_centered^T @ src_centered) / n
    H = (tgt_centered.T @ src_centered) / n

    U, D, Vt = np.linalg.svd(H)
    V = Vt.T

    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0.0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    # Scale s = tr(D @ S) / var_src
    s = float(np.trace(np.diag(D) @ S) / var_src)
    t = mu_tgt - s * (R @ mu_src)

    # Transformed source
    aligned_src = s * (src @ R.T) + t
    diff = tgt - aligned_src
    rmse = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

    return s, R, t, rmse
