import os
import numpy as np
import pytest
from src.inertial.imu_types import (
    skew_symmetric, so3_exp, so3_log, right_jacobian_so3,
    rot_to_quat, quat_to_rot, GRAVITY_NOMINAL
)
from src.ingestion.imu_parser import parse_insta360_csv, parse_phone_sensor_csv

def test_so3_exp_log_consistency():
    # Test identity
    zero_omega = np.zeros(3)
    R_id = so3_exp(zero_omega)
    np.testing.assert_allclose(R_id, np.eye(3), atol=1e-10)
    omega_rec = so3_log(R_id)
    np.testing.assert_allclose(omega_rec, zero_omega, atol=1e-10)

    # Test random rotation vectors
    np.random.seed(42)
    for _ in range(20):
        omega = np.random.uniform(-np.pi, np.pi, size=3)
        # Keep inside standard range (< pi)
        omega = omega / (np.linalg.norm(omega) + 1e-6) * np.random.uniform(0.01, 3.0)
        R = so3_exp(omega)
        # Check orthogonality
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)
        # Check log(exp(omega)) == omega
        omega_rec = so3_log(R)
        np.testing.assert_allclose(omega_rec, omega, atol=1e-7)

def test_quat_rot_conversion():
    np.random.seed(123)
    for _ in range(20):
        omega = np.random.uniform(-2.5, 2.5, size=3)
        R = so3_exp(omega)
        q = rot_to_quat(R)
        R_rec = quat_to_rot(q)
        np.testing.assert_allclose(R_rec, R, atol=1e-7)

def test_right_jacobian():
    omega = np.array([0.1, -0.2, 0.3])
    Jr = right_jacobian_so3(omega)
    delta = np.array([1e-5, -2e-5, 1e-5])
    R1 = so3_exp(omega + delta)
    R2 = so3_exp(omega) @ so3_exp(Jr @ delta)
    np.testing.assert_allclose(R1, R2, atol=1e-8)

def test_parse_insta360():
    try:
        from conftest import get_data_file
        csv_file = get_data_file("insta360_export_1788174571431.csv")
    except ImportError:
        csv_file = "insta360_export_1788174571431.csv"
    if not os.path.exists(csv_file):
        pytest.skip("Dataset file not found")
    ds = parse_insta360_csv(csv_file)

    assert len(ds) > 1000
    assert ds.metadata["num_samples"] == len(ds)
    assert ds.metadata["duration_sec"] > 40.0
    # Check that timestamps are strictly increasing
    ts = ds.timestamps
    assert np.all(np.diff(ts) > 0)
    # Check that acceleration norm is close to 1 g (approx 9.81 m/s^2)
    acc_norms = np.linalg.norm(ds.accels, axis=1)
    assert 9.0 < np.mean(acc_norms) < 11.0
    # Check bias estimation
    bias, a_norm = ds.estimate_static_bias()
    assert bias is not None
    assert 9.0 < a_norm < 11.0
