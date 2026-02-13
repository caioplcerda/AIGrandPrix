"""Tests for drone controller."""

import numpy as np

from aigrandprix.control.drone_controller import DroneController, DroneState, PIDGains


def _make_state(pos=(0, 0, 1), vel=(0, 0, 0)):
    return DroneState(
        position=np.array(pos, dtype=float),
        velocity=np.array(vel, dtype=float),
        orientation=np.array([1, 0, 0, 0], dtype=float),
        angular_velocity=np.zeros(3),
    )


def test_controller_tracks_position():
    ctrl = DroneController()
    state = _make_state(pos=[0, 0, 1])
    target = np.array([5.0, 0.0, 1.0])
    cmd = ctrl.track_position(state, target)

    assert 0 <= cmd.thrust <= 1
    assert cmd.pitch_rate != 0  # should pitch forward to move in x


def test_controller_hover():
    ctrl = DroneController()
    state = _make_state(pos=[0, 0, 1])
    target = np.array([0.0, 0.0, 1.0])
    cmd = ctrl.track_position(state, target)

    # Should produce near-hover thrust
    assert cmd.thrust > 0.3


def test_controller_reset():
    ctrl = DroneController()
    state = _make_state()
    ctrl.track_position(state, np.array([1, 0, 0]))
    ctrl.reset()
    assert np.all(ctrl._integral_error == 0)
