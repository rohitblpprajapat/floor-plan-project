import os
import json
import numpy as np
import pytest
from src.visuals.wall_extractor import VectorWallExtractor, WallSegment
from src.stella.dense_stereo_360 import Dense360Reconstruction
from src.stella.stella_io import VisualKeyframe
from src.optimizer.metric_map_optimizer import MetricMapOptimizer
from src.alignment.linear_initializer import InitialVIState
from src.inertial.imu_types import ImuSample, GRAVITY_NOMINAL

def test_vector_wall_extractor(tmp_path):
    # Create synthetic rectangular room: 4m x 6m
    np.random.seed(42)
    pts = []
    # Wall 1: X in [0, 4], Y = 0
    for x in np.linspace(0, 4, 100):
        pts.append([x + np.random.normal(0, 0.02), 0.0 + np.random.normal(0, 0.02), np.random.uniform(-0.5, 0.5)])
    # Wall 2: X = 4, Y in [0, 6]
    for y in np.linspace(0, 6, 150):
        pts.append([4.0 + np.random.normal(0, 0.02), y + np.random.normal(0, 0.02), np.random.uniform(-0.5, 0.5)])
    # Wall 3: X in [0, 4], Y = 6
    for x in np.linspace(0, 4, 100):
        pts.append([x + np.random.normal(0, 0.02), 6.0 + np.random.normal(0, 0.02), np.random.uniform(-0.5, 0.5)])
    # Wall 4: X = 0, Y in [0, 6]
    for y in np.linspace(0, 6, 150):
        pts.append([0.0 + np.random.normal(0, 0.02), y + np.random.normal(0, 0.02), np.random.uniform(-0.5, 0.5)])

    pts_arr = np.array(pts)
    extractor = VectorWallExtractor(min_inliers_per_wall=30, max_walls=10)
    walls = extractor.extract_walls(pts_arr, camera_z=0.0, enforce_manhattan=True)

    assert len(walls) >= 2 # Successfully detected walls
    # Check that angles are snapped to orthogonal (differences are multiples of 90 deg)
    angles = [w.angle_deg % 90.0 for w in walls]
    for a in angles:
        assert np.isclose(a, angles[0], atol=1.0)

    # Test JSON and SVG export
    json_path = str(tmp_path / "walls.json")
    svg_path = str(tmp_path / "walls.svg")
    extractor.export_json(walls, json_path)
    extractor.export_svg(walls, svg_path)

    assert os.path.exists(json_path) and os.path.getsize(json_path) > 100
    assert os.path.exists(svg_path) and os.path.getsize(svg_path) > 200

def test_dense_stereo_reconstruction():
    densifier = Dense360Reconstruction(max_points_per_pair=200)

    # Synthetic checkerboard/edge images
    img1 = np.zeros((400, 800, 3), dtype=np.uint8)
    cv2_rect = [(100, 100, 300, 300), (450, 50, 650, 350)]
    for rx, ry, rw, rh in cv2_rect:
        img1[ry:rh, rx:rw] = 255
    img2 = img1.copy()

    kf1 = VisualKeyframe(keyframe_id=0, timestamp=0.0, R_wc=np.eye(3), p_wc=np.zeros(3))
    kf2 = VisualKeyframe(keyframe_id=1, timestamp=0.5, R_wc=np.eye(3), p_wc=np.array([0.5, 0.0, 0.0]))

    dense_pts = densifier.densify_keyframe_pair(img1, kf1, img2, kf2)
    assert len(dense_pts) > 0
    assert dense_pts.shape[1] == 3

def test_map_optimizer_normalization_stability():
    # Verify MAP optimizer maintains scale and does not blow up chi-square
    R_ci = np.eye(3)
    p_ci = np.zeros(3)
    opt = MetricMapOptimizer(R_ci, p_ci, max_iters=5)

    s_gt = 1.2
    keyframes = [
        VisualKeyframe(keyframe_id=i, timestamp=float(i)*0.4, R_wc=np.eye(3), p_wc=np.array([float(i)*0.4, 0.0, 0.0]))
        for i in range(6)
    ]
    imu_samples = [
        ImuSample(timestamp=float(k)*0.01, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
        for k in range(250)
    ]

    init_state = InitialVIState(
        scale=s_gt,
        gravity_w=np.array([0.0, 0.0, -9.81]),
        velocities=[np.array([1.0, 0.0, 0.0]) for _ in range(6)],
        condition_number=10.0,
        singular_values=np.ones(5),
        is_observable=True,
        gravity_norm=9.81,
        gyro_bias=np.zeros(3),
        accel_bias=np.zeros(3)
    )


    opt_state = opt.optimize(init_state, keyframes, imu_samples)
    assert opt_state.scale >= 0.2
    assert opt_state.final_cost < 1e7 # Well-bounded, no explosion
