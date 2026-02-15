"""Tests for quaternion and rotation math utilities."""

from __future__ import annotations

import numpy as np
import pytest

from aigrandprix.utils.math_utils import (
    angular_vel_to_quat_derivative,
    euler_to_quat,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_to_euler,
    quat_to_rotation_matrix,
    skew,
)


class TestQuaternionBasics:
    def test_identity_multiply(self):
        """q * identity = q."""
        identity = np.array([1, 0, 0, 0], dtype=float)
        q = quat_normalize(np.array([0.5, 0.3, 0.1, 0.7]))
        result = quat_multiply(q, identity)
        np.testing.assert_allclose(result, q, atol=1e-12)

    def test_conjugate_multiply_gives_identity(self):
        """q * q* = identity (for unit quaternions)."""
        q = quat_normalize(np.array([0.5, 0.3, 0.1, 0.7]))
        result = quat_multiply(q, quat_conjugate(q))
        np.testing.assert_allclose(result, [1, 0, 0, 0], atol=1e-12)

    def test_normalize_unit(self):
        q = np.array([1, 2, 3, 4], dtype=float)
        qn = quat_normalize(q)
        assert abs(np.linalg.norm(qn) - 1.0) < 1e-12

    def test_normalize_zero_returns_identity(self):
        q = np.zeros(4)
        qn = quat_normalize(q)
        np.testing.assert_array_equal(qn, [1, 0, 0, 0])


class TestEulerQuatRoundtrip:
    @pytest.mark.parametrize("rpy", [
        [0, 0, 0],
        [0.3, 0, 0],
        [0, 0.4, 0],
        [0, 0, 1.2],
        [0.1, -0.2, 0.5],
    ])
    def test_roundtrip(self, rpy):
        rpy = np.array(rpy)
        q = euler_to_quat(rpy)
        rpy2 = quat_to_euler(q)
        np.testing.assert_allclose(rpy2, rpy, atol=1e-12)


class TestRotationMatrix:
    def test_identity_quat_gives_identity_matrix(self):
        R = quat_to_rotation_matrix(np.array([1, 0, 0, 0], dtype=float))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_orthogonality(self):
        q = quat_normalize(np.array([0.5, 0.3, 0.1, 0.7]))
        R = quat_to_rotation_matrix(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_quat_rotate_matches_matrix(self):
        q = quat_normalize(np.array([0.5, 0.3, 0.1, 0.7]))
        v = np.array([1.0, 2.0, 3.0])
        v_quat = quat_rotate(q, v)
        R = quat_to_rotation_matrix(q)
        v_mat = R @ v
        np.testing.assert_allclose(v_quat, v_mat, atol=1e-12)


class TestSkewMatrix:
    def test_skew_matrix(self):
        """skew(a) @ b should equal np.cross(a, b)."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            a = rng.standard_normal(3)
            b = rng.standard_normal(3)
            np.testing.assert_allclose(skew(a) @ b, np.cross(a, b), atol=1e-12)


class TestQuatDerivative:
    def test_zero_omega_gives_zero_derivative(self):
        q = np.array([1, 0, 0, 0], dtype=float)
        omega = np.zeros(3)
        qd = angular_vel_to_quat_derivative(q, omega)
        np.testing.assert_allclose(qd, np.zeros(4), atol=1e-12)
