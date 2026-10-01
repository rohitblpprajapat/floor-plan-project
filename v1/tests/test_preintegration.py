import numpy as np
import pytest
from src.inertial.imu_types import ImuSample, ImuBias, so3_exp, so3_log
from src.inertial.preintegration import PreintegratedImu, preintegrate_segment
from src.ingestion.imu_parser import parse_insta360_csv

def test_constant_acceleration():
    # Zero gyro, constant accel = [0, 0, 9.81]
    bias = ImuBias()
    preint = PreintegratedImu(bias)

    dt = 0.01
    acc = np.array([0.0, 0.0, 9.81])
    gyro = np.zeros(3)

    for i in range(100): # 1 second total
        sample = ImuSample(timestamp=i * dt, accel=acc, gyro=gyro)
        preint.integrate_measurement(sample, dt)

    assert np.isclose(preint.delta_t_ij, 1.0, atol=1e-5)
    # Rotation should be identity
    np.testing.assert_allclose(preint.delta_R, np.eye(3), atol=1e-8)
    # Velocity should be a * t = [0, 0, 9.81]
    np.testing.assert_allclose(preint.delta_v, acc * 1.0, atol=1e-3)
    # Position should be 0.5 * a * t^2 = [0, 0, 0.5 * 9.81]
    np.testing.assert_allclose(preint.delta_p, 0.5 * acc * (1.0 ** 2), atol=1e-3)

def test_pure_rotation():
    bias = ImuBias()
    preint = PreintegratedImu(bias)

    dt = 0.005
    gyro = np.array([0.0, 0.0, 1.0]) # 1 rad/s around z
    acc = np.zeros(3)

    for i in range(200): # 1 second total -> 1 radian rotation
        sample = ImuSample(timestamp=i * dt, accel=acc, gyro=gyro)
        preint.integrate_measurement(sample, dt)

    assert np.isclose(preint.delta_t_ij, 1.0, atol=1e-5)
    expected_R = so3_exp(gyro * 1.0)
    np.testing.assert_allclose(preint.delta_R, expected_R, atol=1e-4)

def test_bias_jacobians_first_order():
    # Verify analytical Jacobians against re-integration with perturbed bias
    dt = 0.01
    b0 = ImuBias(gyro_bias=np.array([0.01, -0.02, 0.01]), accel_bias=np.array([0.05, -0.05, 0.02]))
    preint0 = PreintegratedImu(b0)

    np.random.seed(99)
    samples = []
    for i in range(50):
        # Varying motion
        g_val = np.array([np.sin(i * 0.1), np.cos(i * 0.1), 0.2])
        a_val = np.array([0.1, 0.5, 9.8 + 0.3 * np.sin(i * 0.2)])
        s = ImuSample(timestamp=i * dt, accel=a_val, gyro=g_val)
        samples.append(s)
        preint0.integrate_measurement(s, dt)

    # Perturb gyro bias
    delta_bg = np.array([1e-4, -2e-4, 1e-4])
    b_perturbed_g = ImuBias(gyro_bias=b0.gyro_bias + delta_bg, accel_bias=b0.accel_bias)
    preint_perturbed_g = PreintegratedImu(b_perturbed_g)
    for s in samples:
        preint_perturbed_g.integrate_measurement(s, dt)

    # Predict via Jacobian
    pred_R = preint0.predict_delta_R(b_perturbed_g.gyro_bias)
    rot_err = so3_log(pred_R.T @ preint_perturbed_g.delta_R)
    np.testing.assert_allclose(rot_err, np.zeros(3), atol=1e-4)

    pred_v = preint0.predict_delta_v(b_perturbed_g)
    np.testing.assert_allclose(pred_v, preint_perturbed_g.delta_v, atol=1e-4)

    pred_p = preint0.predict_delta_p(b_perturbed_g)
    np.testing.assert_allclose(pred_p, preint_perturbed_g.delta_p, atol=1e-4)

    # Perturb accel bias
    delta_ba = np.array([2e-4, 1e-4, -3e-4])
    b_perturbed_a = ImuBias(gyro_bias=b0.gyro_bias, accel_bias=b0.accel_bias + delta_ba)
    preint_perturbed_a = PreintegratedImu(b_perturbed_a)
    for s in samples:
        preint_perturbed_a.integrate_measurement(s, dt)

    pred_v_a = preint0.predict_delta_v(b_perturbed_a)
    np.testing.assert_allclose(pred_v_a, preint_perturbed_a.delta_v, atol=1e-4)

    pred_p_a = preint0.predict_delta_p(b_perturbed_a)
    np.testing.assert_allclose(pred_p_a, preint_perturbed_a.delta_p, atol=1e-4)

def test_preintegrate_segment_real_dataset():
    import os
    try:
        from conftest import get_data_file
        csv_file = get_data_file("insta360_export_1788174571431.csv")
    except ImportError:
        csv_file = "insta360_export_1788174571431.csv"
    if not os.path.exists(csv_file):
        pytest.skip("Dataset file not found")

    ds = parse_insta360_csv(csv_file)

    t0 = ds.samples[100].timestamp
    t1 = t0 + 1.0 # 1 second window

    bias = ImuBias()
    preint = preintegrate_segment(ds.samples, t0, t1, bias)

    assert preint is not None
    assert preint.sample_count > 500
    assert np.isclose(preint.delta_t_ij, 1.0, atol=0.01)
    # Check that covariance is positive semi-definite
    eigvals = np.linalg.eigvalsh(preint.cov)
    assert np.all(eigvals >= -1e-12)
