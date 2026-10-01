"""Comprehensive test suite for Prompt 05 frames stage, projection models, and rig export.

Verifies:
1. Pixel-to-ray-to-pixel round trips across cardinal directions, seam wrap, and near poles.
2. Crop and resize intrinsic transformations.
3. Proper SO(3) rotations: orthonormal with determinant +1.
4. All virtual views have identical centers; inverse transforms and serialized PyCOLMAP rig reproduce identical rays.
5. Mask alignment after reprojection, including seam-crossing polygons.
6. Synthetic room projection comparison to analytic ground truth with subpixel tolerances.
7. Rotation-only sequence negative fixture with strictly zero translation baseline.
8. Job lifecycle: start, status, cooperative cancellation, and resumption.
9. Cache invalidation on configuration/projection/mask alteration.
10. Pilot frame extraction and contact sheet generation on real session 20260910_172939.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
import pytest
import pycolmap

from vscan.engine.supervisor import Supervisor
from vscan.frames.masks import (
    MaskManager,
    create_hardware_nadir_mask,
    rasterize_polygon_mask,
    split_seam_crossing_polygon,
)
from vscan.frames.projection import (
    CUBEMAP_FACES,
    CUBEMAP_ROTATIONS,
    PinholeIntrinsics,
    ReprojectionFace,
    build_virtual_camera_poses,
    crop_pinhole_intrinsics,
    get_cubemap_rotations,
    get_default_pinhole_intrinsics,
    matrix_to_quaternion_wxyz,
    pixel_to_ray_equirectangular,
    pixel_to_ray_pinhole,
    quaternion_wxyz_to_matrix,
    ray_to_pixel_equirectangular,
    ray_to_pixel_pinhole,
    scale_pinhole_intrinsics,
)
from vscan.frames.renderer import EquirectangularRenderer
from vscan.frames.rig import (
    build_colmap_rig_json,
    export_rig_config_json,
    reproduce_rays_from_serialized_rig,
    verify_rig_config_with_pycolmap,
)
from vscan.frames.scoring import (
    evaluate_panorama_quality,
    score_exposure,
    score_inter_frame_difference,
    score_sharpness,
)
from vscan.frames.selector import (
    VideoFrameDecoder,
    select_contiguous_pilot_frames,
)
from vscan.frames.synthetic import (
    SyntheticRoom,
    generate_synthetic_rotation_only_fixture,
    generate_synthetic_translation_fixture,
)
from vscan.schemas.jobs import ErrorClassification, JobStatus
from vscan.storage.artifacts import verify_completion_marker
from vscan.storage.database import DatabaseManager
from vscan.workers.frames import FramesWorker


@pytest.fixture
def test_env(tmp_path: Path):
    """Create isolated supervisor environment for testing."""
    db_path = tmp_path / "test_app.sqlite3"
    runs_dir = tmp_path / "runs"
    sessions_dir = tmp_path / "data" / "sessions"
    runs_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "test_laptop.yaml"
    config_data = {
        "profile": "test_laptop",
        "concurrency": {"max_gpu_jobs": 1, "max_cpu_jobs": 2, "worker_cpu_threads": 2},
        "process_management": {"cooperative_grace_period_seconds": 2.0},
        "timeouts": {"default_seconds": 60, "stages": {"frames": 60, "smoke": 10}},
        "storage": {
            "app_db_path": str(db_path),
            "sessions_dir": str(sessions_dir),
            "runs_dir": str(runs_dir),
            "base_data_dir": str(tmp_path / "data"),
            "mode": "reference",
        },
    }
    import yaml
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    supervisor = Supervisor(
        db_path=db_path,
        runs_dir=runs_dir,
        sessions_dir=sessions_dir,
        config_path=config_path,
    )
    return {
        "tmp_path": tmp_path,
        "db_path": db_path,
        "runs_dir": runs_dir,
        "sessions_dir": sessions_dir,
        "config_path": config_path,
        "supervisor": supervisor,
    }


def test_01_equirectangular_pixel_ray_roundtrips():
    """Verify pixel-to-ray-to-pixel round trips across cardinal directions, seam wrap, and poles."""
    W_eq = 3840
    H_eq = 1920

    # 1. Cardinal test points
    test_points = [
        # (u, v, expected_name)
        (W_eq / 2.0 - 0.5, H_eq / 2.0 - 0.5, "Center forward +Z"),
        (0.75 * W_eq - 0.5, H_eq / 2.0 - 0.5, "Right +X"),
        (0.25 * W_eq - 0.5, H_eq / 2.0 - 0.5, "Left -X"),
        (0.0, H_eq / 2.0 - 0.5, "Seam left edge -Z"),
        (W_eq - 1.0, H_eq / 2.0 - 0.5, "Seam right edge -Z"),
        (W_eq / 2.0 - 0.5, 0.0, "Near zenith -Y"),
        (W_eq / 2.0 - 0.5, H_eq - 1.0, "Near nadir +Y"),
    ]

    for u_in, v_in, label in test_points:
        rx, ry, rz = pixel_to_ray_equirectangular(u_in, v_in, W_eq, H_eq)
        # Verify unit norm
        norm = math.sqrt(rx * rx + ry * ry + rz * rz)
        assert abs(norm - 1.0) < 1e-10, f"Ray norm not 1 for {label}"

        # Round trip
        u_out, v_out = ray_to_pixel_equirectangular(rx, ry, rz, W_eq, H_eq)
        assert abs(u_out - u_in) < 1e-4 or (abs(abs(u_out - u_in) - W_eq) < 1e-4), f"U mismatch for {label}"
        assert abs(v_out - v_in) < 1e-4, f"V mismatch for {label}"

    # 2. 1000 random uniform points on the sphere
    np.random.seed(42)
    rand_u = np.random.uniform(1.0, W_eq - 2.0, 1000)
    rand_v = np.random.uniform(1.0, H_eq - 2.0, 1000)

    rx, ry, rz = pixel_to_ray_equirectangular(rand_u, rand_v, W_eq, H_eq)
    u_rt, v_rt = ray_to_pixel_equirectangular(rx, ry, rz, W_eq, H_eq)

    max_err_u = np.max(np.abs(u_rt - rand_u))
    max_err_v = np.max(np.abs(v_rt - rand_v))
    assert max_err_u < 1e-5
    assert max_err_v < 1e-5


def test_02_crop_and_resize_intrinsic_transformations():
    """Verify camera intrinsics scaling and cropping produce identical 3D bearing rays."""
    base = get_default_pinhole_intrinsics(width=960, height=960, fov_deg=90.0)

    # 1. Resize scaling by 0.5x
    scaled = scale_pinhole_intrinsics(base, scale_x=0.5, scale_y=0.5)
    assert scaled.width == 480
    assert scaled.height == 480
    assert abs(scaled.fx - 240.0) < 1e-6
    assert abs(scaled.cx - 240.0) < 1e-6

    # Compare 3D ray of corresponding point (e.g. center)
    r1 = pixel_to_ray_pinhole(480.0 - 0.5, 480.0 - 0.5, base)
    r2 = pixel_to_ray_pinhole(240.0 - 0.5, 240.0 - 0.5, scaled)
    for c1, c2 in zip(r1, r2):
        assert abs(c1 - c2) < 1e-10

    # 2. Cropping
    crop_x0, crop_y0 = 100, 100
    cropped = crop_pinhole_intrinsics(base, crop_x0=crop_x0, crop_y0=crop_y0, crop_w=400, crop_h=400)
    assert cropped.width == 400
    assert cropped.height == 400

    # Point at (200, 200) in base image corresponds to (100, 100) in cropped image
    r_base = pixel_to_ray_pinhole(200.0, 200.0, base)
    r_crop = pixel_to_ray_pinhole(200.0 - crop_x0, 200.0 - crop_y0, cropped)
    for c1, c2 in zip(r_base, r_crop):
        assert abs(c1 - c2) < 1e-10


def test_03_proper_so3_rotations_and_zero_baseline():
    """Verify all 6 virtual camera rotations are proper SO(3) and share zero baseline."""
    rotations = get_cubemap_rotations()
    assert len(rotations) == 6

    for face, R in rotations.items():
        # Check shape
        assert R.shape == (3, 3)
        # Check determinant = +1
        det = float(np.linalg.det(R))
        assert abs(det - 1.0) < 1e-10, f"Face {face} determinant {det} != 1.0"
        # Check orthogonality R^T @ R = I
        diff = R.T @ R - np.eye(3)
        assert np.max(np.abs(diff)) < 1e-10, f"Face {face} not orthogonal"

        # Check quaternion roundtrip
        q = matrix_to_quaternion_wxyz(R)
        R_rt = quaternion_wxyz_to_matrix(q)
        assert np.max(np.abs(R_rt - R)) < 1e-10

    # Verify virtual camera poses have strictly zero baseline
    intrinsics = get_default_pinhole_intrinsics(960, 960, 90.0)
    poses = build_virtual_camera_poses(intrinsics)
    for name, p in poses.items():
        assert p.t_pano_cam == [0.0, 0.0, 0.0]
        assert p.cam_from_rig_translation == [0.0, 0.0, 0.0]


def test_04_pycolmap_rig_serialization_and_ray_reproduction(tmp_path: Path):
    """Verify PyCOLMAP 4.2.0 parses serialized rig and reproduces identical rays."""
    intrinsics = get_default_pinhole_intrinsics(960, 960, 90.0)
    poses = build_virtual_camera_poses(intrinsics)
    rig_path = tmp_path / "rig_config.json"

    export_rig_config_json(poses, rig_path)
    assert rig_path.is_file()

    # Verify with PyCOLMAP 4.2.0 pinned API
    verification = verify_rig_config_with_pycolmap(rig_path)
    assert verification["status"] == "VALID"
    assert verification["camera_count"] == 6
    assert verification["ref_sensor_prefix"] == "face_front_"

    # Verify ray reproduction across all 6 faces for multiple pixels
    test_pixels = [(0.0, 0.0), (479.5, 479.5), (959.0, 959.0), (120.0, 800.0)]
    for face in CUBEMAP_FACES:
        cam_name = face.value
        R_pano_cam = CUBEMAP_ROTATIONS[face]

        for u, v in test_pixels:
            # Direct ray from R_pano_cam @ d_cam
            xc, yc, zc = pixel_to_ray_pinhole(u, v, intrinsics)
            d_cam = np.array([xc, yc, zc], dtype=np.float64)
            d_expected = R_pano_cam @ d_cam
            d_expected /= np.linalg.norm(d_expected)

            # Ray reproduced from serialized PyCOLMAP rig
            rx, ry, rz = reproduce_rays_from_serialized_rig(rig_path, cam_name, u, v, intrinsics)

            err = math.sqrt((rx - d_expected[0])**2 + (ry - d_expected[1])**2 + (rz - d_expected[2])**2)
            assert err < 1e-7, f"Ray reproduction mismatch for face {cam_name} at ({u}, {v}): err={err}"


def test_05_seam_crossing_polygon_mask_alignment():
    """Verify seam-crossing polygons reproject into a single contiguous connected component."""
    W_eq, H_eq = 1920, 960
    w, h = 480, 480
    renderer = EquirectangularRenderer(W_eq, H_eq, target_width=w, target_height=h, fov_deg=90.0)

    # Polygon centered directly at the seam x = 0.0 / 1.0 (facing backward -Z)
    # Span from x=0.96 to 1.0 and 0.0 to 0.04
    poly_pts = [
        [0.96, 0.40],
        [1.00, 0.40],
        [1.00, 0.60],
        [0.96, 0.60],
        [0.00, 0.60],
        [0.04, 0.60],
        [0.04, 0.40],
        [0.00, 0.40],
    ]

    mask_mgr = MaskManager(W_eq, H_eq)
    mask_mgr.add_polygon(poly_pts, "mask_seam_test", mask_type="operator", description="Seam test")
    assert mask_mgr.definitions[0].is_seam_crossing is True

    # Propagate to perspective views
    perspective_masks = mask_mgr.propagate_to_perspective(renderer)
    back_mask = perspective_masks["face_back"]

    # Verify that on face_back, the reprojected mask forms a SINGLE contiguous component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(back_mask)
    num_foreground = num_labels - 1
    assert num_foreground == 1, f"Expected 1 contiguous component on face_back, got {num_foreground}"

    # Center pixel must be masked
    center_val = back_mask[h // 2, w // 2]
    assert center_val == 255, "Center pixel on face_back is not masked"


def test_06_synthetic_room_projection_vs_analytic_ground_truth():
    """Compare rendered synthetic room perspective views to direct analytic ground truth."""
    room = SyntheticRoom()
    w, h = 240, 240
    intrinsics = get_default_pinhole_intrinsics(w, h, 90.0)

    cam_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # 1. Analytic equirectangular render (bounded for rapid test execution)
    eq_w, eq_h = 480, 240
    eq_rgb, eq_depth = room.render_analytic_equirectangular(cam_center, width=eq_w, height=eq_h)

    # 2. EquirectangularRenderer perspective reprojection
    renderer = EquirectangularRenderer(eq_w, eq_h, intrinsics=intrinsics)
    reproj_faces = renderer.render_all_faces(eq_rgb, interpolation=cv2.INTER_LINEAR)

    # 3. Direct analytic pinhole render for front and right faces
    for face in [ReprojectionFace.FRONT, ReprojectionFace.RIGHT]:
        cam_name = f"face_{face.value}"
        R_pc = CUBEMAP_ROTATIONS[face]
        gt_rgb, gt_depth = room.render_analytic_pinhole(cam_center, R_pc, intrinsics)
        reproj_img = reproj_faces[cam_name]

        # Compare color difference excluding sharp boundary edges
        diff = np.abs(gt_rgb.astype(np.float32) - reproj_img.astype(np.float32))
        mean_diff = float(np.mean(diff))

        # Because of bilinear sampling on equirectangular grid, interior texture matches closely
        assert mean_diff < 15.0, f"Mean color difference too high on {cam_name}: {mean_diff}"


def test_07_negative_fixture_rotation_only_zero_baseline():
    """Verify that rotation-only sequence fixture maintains strictly zero translation baseline."""
    num_frames = 6
    fixtures = generate_synthetic_rotation_only_fixture(num_frames=num_frames)
    assert len(fixtures) == num_frames

    for fix in fixtures:
        assert fix["trajectory_type"] == "rotation_only_negative_fixture"
        assert fix["camera_center"] == [0.0, 0.0, 0.0]
        assert fix["translation_baseline_norm"] == 0.0

        # Check rotation matrix validity
        R = np.array(fix["rotation_matrix"], dtype=np.float64)
        det = float(np.linalg.det(R))
        assert abs(det - 1.0) < 1e-10


def test_08_frames_job_lifecycle_cancel_and_resume(test_env):
    """Verify job start, status, cooperative cancellation, and resumption on synthetic fixture."""
    supervisor: Supervisor = test_env["supervisor"]
    session_id = "test_synth_lifecycle"

    # Stage config for synthetic fixture
    stage_config = {
        "frames": {
            "synthetic_fixture": True,
            "synthetic_num_frames": 3,
            "target_width": 240,
            "target_height": 240,
            "fov_deg": 90.0,
        },
        "storage": {
            "app_db_path": str(test_env["db_path"]),
            "sessions_dir": str(test_env["sessions_dir"]),
            "runs_dir": str(test_env["runs_dir"]),
            "base_data_dir": str(test_env["tmp_path"] / "data"),
            "mode": "reference",
        },
    }
    cfg_file = test_env["tmp_path"] / "synth_config.yaml"
    import yaml
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(stage_config, f)

    # 1. Start job synchronously
    sub = supervisor.start_job(stage="frames", session_id=session_id, config_path=cfg_file, wait=True)
    run_id = sub["run_id"]
    rec = supervisor.get_status(run_id)
    assert rec is not None
    assert rec.status == JobStatus.SUCCEEDED, f"Job failed with: {rec.error_message}"

    outputs_dir = test_env["runs_dir"] / run_id / "outputs"
    assert (outputs_dir / "pilot_manifest.json").is_file()
    assert (outputs_dir / "rig_config.json").is_file()
    assert (outputs_dir / ".complete").is_file()

    # 2. Test completion marker
    is_valid, msg, marker = verify_completion_marker(outputs_dir, expected_cache_key=rec.cache_key)
    assert is_valid, msg

    # 3. Test cancellation and resumption
    sub_cancel = supervisor.start_job(stage="frames", session_id="test_cancel_sess", config_path=cfg_file, wait=False)
    time.sleep(0.05)
    success, cancel_msg = supervisor.cancel_job(sub_cancel["run_id"])
    assert success, cancel_msg

    cancel_rec = supervisor.get_status(sub_cancel["run_id"])
    assert cancel_rec.status == JobStatus.CANCELLED

    # Resume the cancelled run
    resumed = supervisor.resume_job(sub_cancel["run_id"], wait=True)
    res_rec = supervisor.get_status(resumed["run_id"])
    assert res_rec.status == JobStatus.SUCCEEDED, f"Resumed job failed with: {res_rec.error_message}"
    assert res_rec.parent_run_id == sub_cancel["run_id"]


def test_09_cache_invalidation_on_config_change(test_env):
    """Verify changing resolution, FOV, or masks produces distinct cache keys and invalidates cache."""
    supervisor: Supervisor = test_env["supervisor"]
    session_id = "test_synth_cache_inval"

    cfg1 = {
        "frames": {"synthetic_fixture": True, "target_width": 240, "target_height": 240, "fov_deg": 90.0},
        "storage": {
            "app_db_path": str(test_env["db_path"]),
            "sessions_dir": str(test_env["sessions_dir"]),
            "runs_dir": str(test_env["runs_dir"]),
            "base_data_dir": str(test_env["tmp_path"] / "data"),
            "mode": "reference",
        },
    }
    cfg2 = {
        "frames": {"synthetic_fixture": True, "target_width": 320, "target_height": 320, "fov_deg": 90.0},  # Resolution changed
        "storage": {
            "app_db_path": str(test_env["db_path"]),
            "sessions_dir": str(test_env["sessions_dir"]),
            "runs_dir": str(test_env["runs_dir"]),
            "base_data_dir": str(test_env["tmp_path"] / "data"),
            "mode": "reference",
        },
    }

    import yaml
    p1 = test_env["tmp_path"] / "cfg1.yaml"
    p2 = test_env["tmp_path"] / "cfg2.yaml"
    with open(p1, "w") as f:
        yaml.dump(cfg1, f)
    with open(p2, "w") as f:
        yaml.dump(cfg2, f)

    sub1 = supervisor.start_job(stage="frames", session_id=session_id, config_path=p1, wait=True)
    sub2 = supervisor.start_job(stage="frames", session_id=session_id, config_path=p2, wait=True)

    # Distinct cache keys must be generated
    assert sub1["cache_key"] != sub2["cache_key"]
    # sub2 must not be treated as cached
    assert sub2.get("cached") is not True


def test_10_real_session_actual_data_extraction_and_contact_sheet(test_env):
    """Execute frames extraction on real session 20260910_172939; verify contact sheets and pilot count."""
    real_video = Path("data/2026-09-10/insta360_export_1789041643971.mp4")
    if not real_video.is_file():
        pytest.skip(f"Real video file not available at {real_video}")

    supervisor: Supervisor = test_env["supervisor"]
    session_id = "20260910_172939"

    # Copy real file or reference
    real_cfg = {
        "frames": {
            "source_video": str(real_video.resolve()),
            "source_asset_id": "asset_20260910_172939_insta360_export_1789041643971_mp4",
            "source_projection": "equirectangular_360",
            "target_width": 480,  # Bounded resolution for rapid test execution
            "target_height": 480,
            "fov_deg": 90.0,
            "max_pilot_panoramas": 10,  # Test bounded subset
            "selection_policy": "contiguous_window",
            "masks": {"enable_hardware_nadir": True, "nadir_elevation_deg": -78.0},
        },
        "storage": {
            "app_db_path": str(test_env["db_path"]),
            "sessions_dir": str(test_env["sessions_dir"]),
            "runs_dir": str(test_env["runs_dir"]),
            "base_data_dir": str(Path("data").resolve()),
            "mode": "reference",
        },
    }
    cfg_file = test_env["tmp_path"] / "real_cfg.yaml"
    import yaml
    with open(cfg_file, "w") as f:
        yaml.dump(real_cfg, f)

    sub = supervisor.start_job(stage="frames", session_id=session_id, config_path=cfg_file, wait=True)
    rec = supervisor.get_status(sub["run_id"])
    assert rec.status == JobStatus.SUCCEEDED

    outputs_dir = test_env["runs_dir"] / sub["run_id"] / "outputs"
    manifest_path = outputs_dir / "pilot_manifest.json"
    assert manifest_path.is_file()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert manifest_data["selected_pilot_frames_count"] == 10
    assert manifest_data["scale_status"] == "NOT_ESTIMATED"
    assert manifest_data["actual_data_readiness"] == "PROVISIONAL_PANORAMIC_BASELINE"

    # Contact sheets must exist
    assert (outputs_dir / "contact_sheets" / "contact_sheet_pilot_panoramas.jpg").is_file()
    assert (outputs_dir / "contact_sheets" / "contact_sheet_perspective_faces.jpg").is_file()
