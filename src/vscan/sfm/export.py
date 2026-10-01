"""Export reconstruction poses, 2D tracks, 3D landmarks, and diagnostic distributions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pycolmap

from vscan.frames.projection import matrix_to_quaternion_wxyz
from vscan.schemas.sfm import (
    SfmCameraPoseSchema,
    SfmComponentSchema,
    SfmLandmarkSchema,
    SfmStationPoseSchema,
    SfmTrackObservationSchema,
)


def parse_image_name(name: str) -> Tuple[str, str, Optional[int]]:
    """Parse face camera name, base filename, and station index from image filename.

    Example: 'face_front_frame_000066.jpg' -> ('face_front', 'frame_000066', 66)
    """
    clean_name = Path(name).stem
    # Standard prefix format: face_{face}_{suffix} or suffix_{face}
    for face in ["face_front", "face_right", "face_back", "face_left", "face_up", "face_down"]:
        if clean_name.startswith(f"{face}_"):
            suffix = clean_name[len(face) + 1 :]
            st_idx = None
            import re
            m = re.search(r"(\d+)", suffix)
            if m:
                st_idx = int(m.group(1))
            view_id = f"{suffix}_{face}"
            return face, view_id, st_idx
        elif clean_name.endswith(f"_{face}"):
            prefix = clean_name[: -(len(face) + 1)]
            st_idx = None
            import re
            m = re.search(r"(\d+)", prefix)
            if m:
                st_idx = int(m.group(1))
            view_id = f"{prefix}_{face}"
            return face, view_id, st_idx

    return "unknown", clean_name, None


def compute_point_triangulation_angle(
    point: np.ndarray,
    camera_centers: List[np.ndarray],
) -> float:
    """Compute maximum pairwise triangulation angle (degrees) between observing cameras."""
    if len(camera_centers) < 2:
        return 0.0

    max_angle_deg = 0.0
    for i in range(len(camera_centers)):
        for j in range(i + 1, len(camera_centers)):
            v1 = point - camera_centers[i]
            v2 = point - camera_centers[j]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angle_deg = math.degrees(math.acos(cos_val))
            if angle_deg > max_angle_deg:
                max_angle_deg = angle_deg

    return max_angle_deg


def export_reconstruction_component(
    recon: pycolmap.Reconstruction,
    component_id: int,
    output_dir: Path,
    is_primary: bool = False,
) -> Tuple[SfmComponentSchema, Dict[str, Any]]:
    """Export camera poses, 2D tracks, 3D landmarks, and COLMAP native models for a component."""
    comp_dir = output_dir / f"component_{component_id:02d}"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Native COLMAP export
    colmap_text_dir = comp_dir / "colmap_text"
    colmap_text_dir.mkdir(exist_ok=True)
    try:
        recon.write_text(str(colmap_text_dir))
    except Exception as exc:
        print(f"[Export] Warning: write_text failed: {exc}")

    # 2. Camera poses & source-view relationships
    camera_poses: List[Dict[str, Any]] = []
    station_map: Dict[int, Dict[str, Any]] = {}
    camera_centers_dict: Dict[int, np.ndarray] = {}

    for img_id in recon.reg_image_ids():
        im = recon.image(img_id)
        cam_from_world = im.cam_from_world()
        R_w2c = cam_from_world.rotation.matrix()
        q_w2c = matrix_to_quaternion_wxyz(R_w2c)
        t_w2c = [float(x) for x in cam_from_world.translation]

        # Invert to get camera-to-world (R_c2w = R_w2c^T, C = -R_w2c^T * T_w2c)
        world_from_cam = cam_from_world.inverse()
        R_c2w = world_from_cam.rotation.matrix().tolist()
        C_world = [float(x) for x in world_from_cam.translation]
        camera_centers_dict[img_id] = np.array(C_world, dtype=np.float64)

        cam_name, view_id, st_idx = parse_image_name(im.name)

        pose_entry = SfmCameraPoseSchema(
            image_id=img_id,
            image_name=im.name,
            view_id=view_id,
            camera_name=cam_name,
            camera_id=im.camera_id,
            frame_id=im.frame_id if im.has_frame_id() else None,
            panorama_index=st_idx,
            is_registered=True,
            cam_from_world_rotation_quaternion_wxyz=q_w2c,
            cam_from_world_translation=t_w2c,
            world_from_cam_rotation_matrix_3x3=R_c2w,
            world_from_cam_translation=C_world,
            units="ARBITRARY",
        )
        camera_poses.append(pose_entry.model_dump(mode="json"))

        # Track physical station centers
        if st_idx is not None:
            if st_idx not in station_map:
                station_map[st_idx] = {
                    "station_index": st_idx,
                    "frame_id": im.frame_id if im.has_frame_id() else None,
                    "registered_views": [],
                    "centers": [],
                }
            station_map[st_idx]["registered_views"].append(cam_name)
            station_map[st_idx]["centers"].append(C_world)

    # 3. Physical stations summary
    station_poses: List[Dict[str, Any]] = []
    for st_idx, st_data in sorted(station_map.items()):
        avg_center = np.mean(st_data["centers"], axis=0).tolist()
        st_entry = SfmStationPoseSchema(
            station_index=st_idx,
            frame_id=st_data["frame_id"],
            is_registered=True,
            nodal_center_world=avg_center,
            registered_views=st_data["registered_views"],
        )
        station_poses.append(st_entry.model_dump(mode="json"))

    # 4. Tracks, Landmarks, and Observations
    landmarks: List[Dict[str, Any]] = []
    track_lengths: List[int] = []
    residuals: List[float] = []
    triangulation_angles: List[float] = []

    for pt_id, pt3d in recon.points3D.items():
        p_xyz = np.array(pt3d.xyz, dtype=np.float64)
        err = float(pt3d.error)
        residuals.append(err)

        obs_list: List[SfmTrackObservationSchema] = []
        obs_cam_centers: List[np.ndarray] = []

        for elem in pt3d.track.elements:
            if elem.image_id in recon.images:
                obs_im = recon.image(elem.image_id)
                p2d = obs_im.points2D[elem.point2D_idx]
                cam_name, view_id, st_idx = parse_image_name(obs_im.name)
                obs_list.append(
                    SfmTrackObservationSchema(
                        image_id=elem.image_id,
                        point2d_idx=elem.point2D_idx,
                        xy_pixel=[float(p2d.xy[0]), float(p2d.xy[1])],
                        view_id=view_id,
                        camera_name=cam_name,
                        panorama_index=st_idx,
                    )
                )
                if elem.image_id in camera_centers_dict:
                    obs_cam_centers.append(camera_centers_dict[elem.image_id])

        t_len = len(obs_list)
        track_lengths.append(t_len)

        tri_angle = compute_point_triangulation_angle(p_xyz, obs_cam_centers)
        triangulation_angles.append(tri_angle)

        lm_entry = SfmLandmarkSchema(
            point3d_id=pt_id,
            xyz=[float(x) for x in pt3d.xyz],
            units="ARBITRARY",
            color_rgb=[int(c) for c in pt3d.color],
            reprojection_error_px=err,
            track_length=t_len,
            observations=obs_list,
        )
        landmarks.append(lm_entry.model_dump(mode="json"))

    # Compute statistics
    num_views = len(camera_poses)
    num_stations = len(station_poses)
    num_lm = len(landmarks)
    num_obs = sum(track_lengths)

    mean_track = float(np.mean(track_lengths)) if track_lengths else 0.0
    median_track = float(np.median(track_lengths)) if track_lengths else 0.0
    mean_angle = float(np.mean(triangulation_angles)) if triangulation_angles else 0.0

    mean_err = float(np.mean(residuals)) if residuals else 0.0
    median_err = float(np.median(residuals)) if residuals else 0.0
    rmse_err = float(np.sqrt(np.mean(np.array(residuals) ** 2))) if residuals else 0.0

    percentiles = {}
    if residuals:
        percentiles = {
            "p50": float(np.percentile(residuals, 50)),
            "p90": float(np.percentile(residuals, 90)),
            "p95": float(np.percentile(residuals, 95)),
            "p99": float(np.percentile(residuals, 99)),
            "max": float(np.max(residuals)),
        }

    # Bounding box
    bbox_min, bbox_max = [], []
    if landmarks:
        xyz_arr = np.array([lm["xyz"] for lm in landmarks])
        bbox_min = np.min(xyz_arr, axis=0).tolist()
        bbox_max = np.max(xyz_arr, axis=0).tolist()

    comp_summary = SfmComponentSchema(
        component_id=component_id,
        is_primary=is_primary,
        num_registered_stations=num_stations,
        num_registered_views=num_views,
        num_landmarks=num_lm,
        num_observations=num_obs,
        mean_track_length=mean_track,
        median_track_length=median_track,
        mean_triangulation_angle_deg=mean_angle,
        mean_reprojection_error_px=mean_err,
        median_reprojection_error_px=median_err,
        rmse_reprojection_error_px=rmse_err,
        reprojection_residual_percentiles=percentiles,
        bounding_box_min=bbox_min,
        bounding_box_max=bbox_max,
    )

    # Save JSON files
    with open(comp_dir / "camera_poses.json", "w", encoding="utf-8") as f:
        json.dump(camera_poses, f, indent=2)

    with open(comp_dir / "station_poses.json", "w", encoding="utf-8") as f:
        json.dump(station_poses, f, indent=2)

    with open(comp_dir / "tracks.json", "w", encoding="utf-8") as f:
        json.dump(landmarks, f, indent=2)

    with open(comp_dir / "component_summary.json", "w", encoding="utf-8") as f:
        json.dump(comp_summary.model_dump(mode="json"), f, indent=2)

    details = {
        "camera_poses": camera_poses,
        "station_poses": station_poses,
        "landmarks": landmarks,
        "track_lengths": track_lengths,
        "residuals": residuals,
        "triangulation_angles": triangulation_angles,
        "comp_dir": comp_dir,
    }

    return comp_summary, details
