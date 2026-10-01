import numpy as np
import pytest
from src.inertial.imu_types import so3_exp
from src.alignment.temporal_sync import (
    generate_cube_rotations, cross_correlate_signals, solve_extrinsic_R_ci
)

def test_cube_rotations():
    rotations = generate_cube_rotations()
    assert len(rotations) == 24
    for R in rotations:
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-8)

def test_cross_correlate_time_shift():
    t = np.linspace(0, 20.0, 1000)
    # Synthetic signal with rich features
    sig1 = np.sin(2.0 * np.pi * 0.5 * t) + 0.5 * np.cos(2.0 * np.pi * 1.3 * t) + np.sin(2.0 * np.pi * 2.1 * t)

    true_dt = 3.42 # Shift signal 2 by 3.42 seconds
    t2 = t + true_dt
    sig2 = sig1.copy()

    best_lag, peak_val, lags, corr = cross_correlate_signals(t, sig1, t2, sig2, dt_resample=0.01, max_lag_sec=10.0)

    # Cross-correlation should recover the true lag with high accuracy (< 0.02s)
    assert np.isclose(best_lag, true_dt, atol=0.02)
    assert peak_val > 0.9

def test_solve_extrinsic_R_ci():
    np.random.seed(42)
    # Ground truth 90-degree permutation rotation: e.g. x->y, y->-x, z->z
    R_gt = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ])

    omega_imu = np.random.uniform(-1.0, 1.0, size=(100, 3))
    omega_cam = (R_gt @ omega_imu.T).T + np.random.normal(0, 0.001, size=(100, 3))

    R_solved, rms = solve_extrinsic_R_ci(omega_cam, omega_imu)

    np.testing.assert_allclose(R_solved, R_gt, atol=1e-3)
    assert rms < 0.01
