"""Shared quaternion and rotation math utilities.

Convention: scalar-first quaternions [w, x, y, z], matching DroneState.orientation.
"""

from __future__ import annotations

import numpy as np


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of quaternion [w, x, y, z] -> [w, -x, -y, -z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3D vector v by unit quaternion q: v' = q * [0,v] * q*."""
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    result = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))
    return result[1:]


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] -> Euler angles [roll, pitch, yaw]."""
    w, x, y, z = q
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def euler_to_quat(rpy: np.ndarray) -> np.ndarray:
    """Euler angles [roll, pitch, yaw] -> quaternion [w, x, y, z]."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,  # w
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
    ])


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def skew(v: np.ndarray) -> np.ndarray:
    """3D vector -> 3x3 skew-symmetric matrix for cross product."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def angular_vel_to_quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Compute quaternion time derivative: q_dot = 0.5 * q (x) [0, omega]."""
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    return 0.5 * quat_multiply(q, omega_quat)
