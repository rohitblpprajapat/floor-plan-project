import numpy as np
import pytest
from src.stella.stella_io import VisualKeyframe, VisualTrajectory
from src.metric.metric_converter import MetricConverter, compute_gravity_alignment_matrix

def test_gravity_alignment_matrix():
    # If gravity is already along -Z, R_g should be identity
    g_down = np.array([0.0, 0.0, -9.81])
    R_g = compute_gravity_alignment_matrix(g_down)
    np.testing.assert_allclose(R_g, np.eye(3), atol=1e-7)

    # If gravity is along +X, R_g should rotate +X into -Z
    g_x = np.array([9.81, 0.0, 0.0])
    R_g_x = compute_gravity_alignment_matrix(g_x)
    g_aligned = R_g_x @ (g_x / 9.81)
    np.testing.assert_allclose(g_aligned, [0.0, 0.0, -1.0], atol=1e-7)

    # If gravity is along arbitrary tilted direction
    g_tilt = np.array([3.0, -4.0, -8.0])
    g_tilt_unit = g_tilt / np.linalg.norm(g_tilt)
    R_g_tilt = compute_gravity_alignment_matrix(g_tilt)
    np.testing.assert_allclose(R_g_tilt @ g_tilt_unit, [0.0, 0.0, -1.0], atol=1e-7)

def test_metric_trajectory_conversion():
    # 5 keyframes walking along a line of 10 meters, but unscaled positions are [0...2]
    # So scale is 5.0
    s_true = 5.0
    g_w = np.array([0.0, 0.0, -9.81])

    keyframes = [
        VisualKeyframe(keyframe_id=i, timestamp=float(i), R_wc=np.eye(3), p_wc=np.array([float(i) * 0.5, 0.0, 0.0]))
        for i in range(5)
    ]
    traj = VisualTrajectory(keyframes)

    converter = MetricConverter(scale=s_true, gravity_w=g_w)
    metric_traj, summary = converter.convert_trajectory(traj)

    assert len(metric_traj) == 5
    # Total length: 4 steps of 0.5 * 5.0 = 2.5m each -> 10.0 meters total
    assert np.isclose(summary.trajectory_length_meters, 10.0, atol=1e-5)
    assert np.isclose(summary.average_speed_mps, 10.0 / 4.0, atol=1e-5)
    assert np.isclose(summary.scale, 5.0, atol=1e-5)

    # Landmarks conversion
    landmarks_unscaled = np.array([
        [1.0, 2.0, 3.0],
        [0.0, -1.0, 2.0]
    ])
    landmarks_metric = converter.convert_landmarks(landmarks_unscaled)
    np.testing.assert_allclose(landmarks_metric, landmarks_unscaled * s_true, atol=1e-5)
