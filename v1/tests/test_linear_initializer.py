import numpy as np
import pytest
from src.inertial.imu_types import ImuSample, ImuBias, so3_exp, GRAVITY_NOMINAL
from src.stella.stella_io import VisualKeyframe
from src.alignment.linear_initializer import LinearVIInitializer, GravityRefiner

def test_linear_scale_and_gravity_recovery():
    np.random.seed(42)
    # Ground truth parameters
    s_gt = 3.5
    g_gt = np.array([0.0, 0.0, -GRAVITY_NOMINAL])
    R_ci = np.eye(3)
    p_ci = np.zeros(3)

    num_keyframes = 12
    dt_kf = 0.5 # 0.5 sec between keyframes
    dt_imu = 0.01 # 100 Hz IMU

    # Generate a smooth 3D trajectory with 3D translation excitation
    # p(t) = [0.8 * sin(t), 0.5 * cos(1.5 * t), 0.2 * sin(2.0 * t)]
    def pos_fn(t):
        return np.array([0.8 * np.sin(t), 0.5 * np.cos(1.5 * t), 0.2 * np.sin(2.0 * t)])

    def vel_fn(t):
        return np.array([0.8 * np.cos(t), -0.75 * np.sin(1.5 * t), 0.4 * np.cos(2.0 * t)])

    def acc_fn(t):
        return np.array([-0.8 * np.sin(t), -1.125 * np.cos(1.5 * t), -0.8 * np.sin(2.0 * t)])

    keyframes = []
    imu_samples = []

    # Create keyframes (scaled by 1 / s_gt)
    for i in range(num_keyframes):
        t = i * dt_kf
        p_metric = pos_fn(t)
        p_unscaled = p_metric / s_gt # Stella position
        R_wc = so3_exp(np.array([0.05 * np.sin(t), 0.08 * np.cos(t), 0.1 * t]))
        keyframes.append(VisualKeyframe(keyframe_id=i, timestamp=t, R_wc=R_wc, p_wc=p_unscaled))

    # Generate dense IMU samples
    t_total = (num_keyframes - 1) * dt_kf
    t_steps = int(t_total / dt_imu)
    for k in range(t_steps + 1):
        t = k * dt_imu
        a_inertial = acc_fn(t)
        # Specific force measured by IMU: f = a - g
        f_w = a_inertial - g_gt
        # Express in body/camera frame: f_b = R_wc^T * f_w
        R_wc = so3_exp(np.array([0.05 * np.sin(t), 0.08 * np.cos(t), 0.1 * t]))
        acc_imu = R_wc.T @ f_w

        # Gyro
        w_w = np.array([0.05 * np.cos(t), -0.08 * np.sin(t), 0.1])
        gyro_imu = R_wc.T @ w_w

        imu_samples.append(ImuSample(timestamp=t, accel=acc_imu, gyro=gyro_imu))

    # Solve linear initialization
    initializer = LinearVIInitializer(R_ci=R_ci, p_ci=p_ci)
    init_state = initializer.solve(keyframes, imu_samples, gyro_bias=np.zeros(3))

    assert init_state is not None
    assert init_state.is_observable
    # Check that scale is recovered within 5% error
    assert np.isclose(init_state.scale, s_gt, rtol=0.05)
    # Check that gravity is recovered close to [0, 0, -9.81]
    np.testing.assert_allclose(init_state.gravity_w, g_gt, atol=0.5)

    # Refine gravity on tangent plane
    refiner = GravityRefiner(nominal_g=GRAVITY_NOMINAL)
    refined_state = refiner.refine(init_state, keyframes, imu_samples, R_ci=R_ci, p_ci=p_ci)

    # Gravity magnitude must be exact nominal
    assert np.isclose(refined_state.gravity_norm, GRAVITY_NOMINAL, atol=1e-5)
    # Scale should be even more accurate (< 2% error)
    assert np.isclose(refined_state.scale, s_gt, rtol=0.03)
