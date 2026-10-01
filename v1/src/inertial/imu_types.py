"""
Data structures and types for Inertial Measurement Unit (IMU) processing.
Strictly SI units:
- Accelerations: m/s^2
- Angular velocities: rad/s
- Timestamps: seconds (float64)
- Orientations: SO(3) rotation matrices and unit quaternions [qw, qx, qy, qz]
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional, List, Tuple

# Standard gravity magnitude (SI: m/s^2)
GRAVITY_NOMINAL = 9.80665

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Compute the 3x3 skew-symmetric matrix [v]_x such that [v]_x * u = v x u."""
    v = np.asarray(v, dtype=np.float64).reshape(3,)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)

def so3_exp(omega: np.ndarray) -> np.ndarray:
    """
    Rodrigues' formula for SO(3) matrix exponential Exp(omega).
    Maps a 3D Lie algebra rotation vector omega into a 3x3 rotation matrix in SO(3).
    """
    omega = np.asarray(omega, dtype=np.float64).reshape(3,)
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        # Taylor expansion around zero: I + [omega]_x + 0.5 * [omega]_x^2
        return np.eye(3, dtype=np.float64) + skew_symmetric(omega)
    k = omega / theta
    K = skew_symmetric(k)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

def so3_log(R: np.ndarray) -> np.ndarray:
    """
    Logarithmic map from SO(3) matrix to 3D Lie algebra rotation vector.
    Inverse of so3_exp.
    """
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-10:
        return np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ], dtype=np.float64) * 0.5
    scale = theta / (2.0 * np.sin(theta))
    return np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ], dtype=np.float64) * scale

def right_jacobian_so3(omega: np.ndarray) -> np.ndarray:
    """
    Right Jacobian Jr(omega) of SO(3) for first-order Baker-Campbell-Hausdorff updates.
    Exp(omega + delta) approx Exp(omega) * Exp(Jr(omega) * delta)
    """
    omega = np.asarray(omega, dtype=np.float64).reshape(3,)
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64) - 0.5 * skew_symmetric(omega)
    k = omega / theta
    K = skew_symmetric(k)
    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / (theta ** 2)
    return A * np.eye(3, dtype=np.float64) - (1.0 - np.cos(theta)) / theta * K + (1.0 - A) * np.outer(k, k)

def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix into a unit quaternion [qw, qx, qy, qz]."""
    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion [qw, qx, qy, qz] into a 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64).reshape(4,)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
        [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
        [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)]
    ], dtype=np.float64)

@dataclass
class ImuSample:
    """A single IMU measurement in standard SI units."""
    timestamp: float          # In seconds (epoch or relative)
    accel: np.ndarray         # 3D acceleration [ax, ay, az] in m/s^2 (specific force)
    gyro: np.ndarray          # 3D angular velocity [gx, gy, gz] in rad/s

    def __post_init__(self):
        self.accel = np.asarray(self.accel, dtype=np.float64).reshape(3,)
        self.gyro = np.asarray(self.gyro, dtype=np.float64).reshape(3,)
        self.timestamp = float(self.timestamp)

@dataclass
class ImuBias:
    """IMU Gyroscope and Accelerometer biases."""
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self):
        self.gyro_bias = np.asarray(self.gyro_bias, dtype=np.float64).reshape(3,)
        self.accel_bias = np.asarray(self.accel_bias, dtype=np.float64).reshape(3,)

    def copy(self) -> 'ImuBias':
        return ImuBias(gyro_bias=self.gyro_bias.copy(), accel_bias=self.accel_bias.copy())

@dataclass
class ImuNoiseParams:
    """Continuous-time IMU noise parameters and random walk densities."""
    accel_noise_density: float = 2.0e-3       # m/s^2 / sqrt(Hz)
    gyro_noise_density: float = 1.6e-4        # rad/s / sqrt(Hz)
    accel_random_walk: float = 3.0e-4         # m/s^3 / sqrt(Hz)
    gyro_random_walk: float = 2.2e-5          # rad/s^2 / sqrt(Hz)

    @property
    def discrete_accel_cov(self) -> float:
        return self.accel_noise_density ** 2

    @property
    def discrete_gyro_cov(self) -> float:
        return self.gyro_noise_density ** 2