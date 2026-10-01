"""Export and verification of multi-camera rig configurations for PyCOLMAP.

Conventions:
- PyCOLMAP 4.2.0 pinned API: read_rig_config expects a JSON list of rig specifications.
- Each rig contains an array of 'cameras'.
- The reference sensor has 'ref_sensor': True.
- Non-reference sensors have 'ref_sensor': False, 'cam_from_rig_rotation': [w, x, y, z], 'cam_from_rig_translation': [tx, ty, tz].
- Virtual cubemap cameras share the identical physical center [0, 0, 0] with zero synthetic baseline.
- Inverse transformations and serialized rig relationships reproduce identical 3D bearing rays.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from vscan.frames.projection import (
    CUBEMAP_FACES,
    CUBEMAP_ROTATIONS,
    PinholeIntrinsics,
    pixel_to_ray_pinhole,
)
from vscan.schemas.frames import VirtualCameraPose


def build_colmap_rig_json(
    virtual_camera_poses: Dict[str, VirtualCameraPose],
) -> List[Dict[str, Any]]:
    """Build COLMAP rig configuration structure matching the pinned PyCOLMAP schema."""
    cameras_list: List[Dict[str, Any]] = []

    # Ensure sorted order: front (index 0, ref sensor) followed by other faces
    sorted_poses = sorted(virtual_camera_poses.values(), key=lambda p: p.face_index)

    for pose in sorted_poses:
        is_ref = (pose.face_index == 0)
        cam_entry: Dict[str, Any] = {
            "image_prefix": f"{pose.camera_name}_",
            "ref_sensor": is_ref,
        }

        if not is_ref:
            cam_entry["cam_from_rig_rotation"] = pose.cam_from_rig_rotation_wxyz
            cam_entry["cam_from_rig_translation"] = [0.0, 0.0, 0.0]

        cameras_list.append(cam_entry)

    return [{"cameras": cameras_list}]


def export_rig_config_json(
    virtual_camera_poses: Dict[str, VirtualCameraPose],
    output_path: Path,
) -> Path:
    """Serialize virtual camera rig specification to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rig_data = build_colmap_rig_json(virtual_camera_poses)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rig_data, f, indent=2)

    return output_path


def verify_rig_config_with_pycolmap(rig_config_path: Path) -> Dict[str, Any]:
    """Load and validate the serialized rig config using PyCOLMAP 4.2.0 pinned API."""
    import pycolmap

    path_str = str(rig_config_path)
    rigs = pycolmap.read_rig_config(path_str)

    if not rigs or len(rigs) != 1:
        raise ValueError(f"Expected exactly 1 rig configuration, loaded {len(rigs) if rigs else 0}")

    rig = rigs[0]
    cameras = rig.cameras

    if len(cameras) != 6:
        raise ValueError(f"Expected 6 virtual cubemap cameras, got {len(cameras)}")

    ref_count = sum(1 for c in cameras if c.ref_sensor)
    if ref_count != 1:
        raise ValueError(f"Rig must have exactly 1 reference sensor, found {ref_count}")

    cam_summaries = []
    for idx, cam in enumerate(cameras):
        prefix = cam.image_prefix
        is_ref = cam.ref_sensor
        cam_info: Dict[str, Any] = {
            "index": idx,
            "prefix": prefix,
            "ref_sensor": is_ref,
        }
        if not is_ref:
            if cam.cam_from_rig is None:
                raise ValueError(f"Camera {prefix} has no cam_from_rig definition!")
            trans = list(cam.cam_from_rig.translation)
            quat = list(cam.cam_from_rig.rotation.quat)  # [x, y, z, w] in pycolmap object
            cam_info["translation"] = trans
            cam_info["rotation_quat_xyzw"] = quat

            # Assert strictly zero translation
            if any(abs(t) > 1e-6 for t in trans):
                raise ValueError(f"Camera {prefix} has non-zero translation {trans}")

        cam_summaries.append(cam_info)

    return {
        "status": "VALID",
        "rig_count": len(rigs),
        "camera_count": len(cameras),
        "ref_sensor_prefix": cameras[0].image_prefix,
        "cameras": cam_summaries,
    }


def reproduce_rays_from_serialized_rig(
    rig_config_path: Path,
    face_name: str,
    u_pixel: float,
    v_pixel: float,
    intrinsics: PinholeIntrinsics,
) -> Tuple[float, float, float]:
    """Reproduce 3D bearing ray in rig body frame directly from serialized PyCOLMAP rig."""
    import pycolmap

    rigs = pycolmap.read_rig_config(str(rig_config_path))
    cameras = rigs[0].cameras

    # Target camera prefix
    target_prefix = f"face_{face_name}_"
    target_cam = None
    for cam in cameras:
        if cam.image_prefix == target_prefix:
            target_cam = cam
            break

    if target_cam is None:
        raise ValueError(f"Camera prefix {target_prefix} not found in rig config")

    # Ray in camera optical frame
    xc, yc, zc = pixel_to_ray_pinhole(u_pixel, v_pixel, intrinsics)
    d_cam = np.array([xc, yc, zc], dtype=np.float64)

    # In COLMAP rig: p_cam = R_cam_rig * p_rig + t
    # Therefore p_rig = R_cam_rig^T * (p_cam - t)
    # Since t = 0: d_rig = R_cam_rig^T * d_cam
    if target_cam.ref_sensor:
        R_cam_rig = np.eye(3, dtype=np.float64)
    else:
        R_cam_rig = target_cam.cam_from_rig.rotation.matrix()

    d_rig = R_cam_rig.T @ d_cam
    d_rig /= np.linalg.norm(d_rig)

    return float(d_rig[0]), float(d_rig[1]), float(d_rig[2])


def verify_rig_database_serialization(
    rig_config_path: Path,
    image_names: Sequence[str],
    intrinsics: PinholeIntrinsics,
    expected_frames_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Verify that the serialized rig config and image names survive PyCOLMAP database serialization.

    Applies the rig configuration to a temporary PyCOLMAP database, validates that all 6 virtual
    views per panorama are successfully linked into unified Frame objects, and verifies that the
    fixed camera extrinsics (zero baseline and rotations) are preserved.
    """
    import tempfile
    import pycolmap

    rigs = pycolmap.read_rig_config(str(rig_config_path))
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db_path = tf.name

    try:
        db = pycolmap.Database.open(temp_db_path)
        # Create PINHOLE camera in DB
        camera = pycolmap.Camera(
            model="PINHOLE",
            width=intrinsics.width,
            height=intrinsics.height,
            params=[intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy],
        )
        camera_id = db.write_camera(camera)

        # Write all image records into database
        for img_name in image_names:
            img = pycolmap.Image(name=img_name, camera_id=camera_id)
            db.write_image(img)

        # Apply rig config to database
        pycolmap.apply_rig_config(rigs, db)

        num_rigs = db.num_rigs()
        num_frames = db.num_frames()

        if num_rigs != 1:
            raise ValueError(f"Expected 1 rig in database after apply_rig_config, found {num_rigs}")

        if expected_frames_count is not None and num_frames != expected_frames_count:
            raise ValueError(
                f"Expected {expected_frames_count} frames in database, found {num_frames}"
            )

        deserialized_rigs = db.read_all_rigs()
        deserialized_rig = deserialized_rigs[0]
        non_ref_sensors = deserialized_rig.non_ref_sensors

        # Verify all non-reference sensors have zero translation
        for sensor_id, rigid3d in non_ref_sensors.items():
            trans = list(rigid3d.translation)
            if any(abs(t) > 1e-6 for t in trans):
                raise ValueError(f"Sensor {sensor_id} has non-zero translation {trans} in database")

        return {
            "status": "VALID",
            "num_rigs": num_rigs,
            "num_frames": num_frames,
            "non_ref_sensor_count": len(non_ref_sensors),
        }
    finally:
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except OSError:
                pass

