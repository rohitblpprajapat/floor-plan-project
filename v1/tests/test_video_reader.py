import os
import numpy as np
import pytest
from src.ingestion.video_reader import equirectangular_to_bearing, VideoReader, SphericalVisualGyro

def test_equirectangular_bearing_geometry():
    w, h = 1000, 500
    # Center pixel (u=500, v=250) -> azimuth=0, elevation=0 -> ray=[0, 0, 1] (forward)
    u_c = np.array([500.0])
    v_c = np.array([250.0])
    ray_c = equirectangular_to_bearing(u_c, v_c, w, h)[0]
    np.testing.assert_allclose(ray_c, [0.0, 0.0, 1.0], atol=1e-5)

    # Top pole (u=500, v=0) -> elevation=pi/2 -> ray=[0, -1, 0] (up in image/camera coords)
    u_top = np.array([500.0])
    v_top = np.array([0.0])
    ray_top = equirectangular_to_bearing(u_top, v_top, w, h)[0]
    np.testing.assert_allclose(ray_top, [0.0, -1.0, 0.0], atol=1e-5)

    # Left equator (u=250, v=250) -> theta = -pi/2 -> ray=[-1, 0, 0] (left)
    u_left = np.array([250.0])
    v_left = np.array([250.0])
    ray_left = equirectangular_to_bearing(u_left, v_left, w, h)[0]
    np.testing.assert_allclose(ray_left, [-1.0, 0.0, 0.0], atol=1e-5)

def test_video_reader_metadata():
    try:
        from conftest import get_data_file
        video_file = get_data_file("LRV_20260831_163825_01_081.mp4")
    except ImportError:
        video_file = "LRV_20260831_163825_01_081.mp4"
    if not os.path.exists(video_file):
        pytest.skip("Video file not found")

    reader = VideoReader(video_file, target_width=800)
    assert reader.fps > 0
    assert reader.frame_count > 100
    assert reader.target_width == 800
    assert reader.target_height == 400

    frames_gen = reader.frames(start_frame=0, max_frames=5)
    count = 0
    for idx, t, frame in frames_gen:
        assert frame.shape == (400, 800, 3)
        assert np.isclose(t, idx / reader.fps, atol=1e-5)
        count += 1
    assert count == 5

def test_spherical_visual_gyro():
    try:
        from conftest import get_data_file
        video_file = get_data_file("LRV_20260831_163825_01_081.mp4")
    except ImportError:
        video_file = "LRV_20260831_163825_01_081.mp4"
    if not os.path.exists(video_file):
        pytest.skip("Video file not found")


    reader = VideoReader(video_file, target_width=800)
    frames = []
    for idx, t, frame in reader.frames(start_frame=50, max_frames=2):
        frames.append(frame)

    assert len(frames) == 2
    gyro = SphericalVisualGyro(n_features=1000)
    R, inliers = gyro.estimate_relative_rotation(frames[0], frames[1])

    assert R is not None
    assert inliers > 15
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-7)
