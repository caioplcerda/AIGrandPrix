"""Tests for YAML config loading and wiring into modules."""

from __future__ import annotations

import numpy as np
import pytest

from aigrandprix.config import get_section, load_config


class TestLoadConfig:
    def test_loads_default(self):
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "race" in cfg
        assert "control" in cfg

    def test_missing_file_returns_empty(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg == {}

    def test_get_section_nested(self):
        cfg = load_config()
        pid = get_section(cfg, "control", "pid")
        assert pid is not None
        assert "kp" in pid

    def test_get_section_missing_returns_default(self):
        cfg = load_config()
        val = get_section(cfg, "no", "such", "key", default=42)
        assert val == 42


class TestModulesWithConfig:
    def test_controller_uses_config_gains(self):
        from aigrandprix.control.drone_controller import DroneController

        config = {
            "pid": {"kp": [1.0, 2.0, 3.0], "ki": [0.0, 0.0, 0.0], "kd": [0.0, 0.0, 0.0]},
            "max_rate": 4.0,
            "max_thrust": 0.8,
        }
        ctrl = DroneController(config=config)
        np.testing.assert_array_equal(ctrl.gains.kp, [1.0, 2.0, 3.0])
        assert ctrl.max_rate == 4.0
        assert ctrl.max_thrust == 0.8

    def test_planner_uses_config_distances(self):
        from aigrandprix.planning.path_planner import PathPlanner

        config = {"approach_distance": 2.5, "exit_distance": 1.8, "trajectory_dt": 0.01}
        planner = PathPlanner(config=config)
        assert planner._approach_distance == 2.5
        assert planner._exit_distance == 1.8

    def test_gate_detector_uses_config(self):
        from aigrandprix.perception.gate_detector import GateDetector

        config = {"area_filter": 100, "focal_length": 800}
        det = GateDetector(config=config)
        assert det._area_filter == 100
        assert det._focal_length == 800

    def test_race_config_from_config(self):
        from aigrandprix.main import RaceConfig

        cfg = load_config()
        rc = RaceConfig.from_config(cfg)
        assert rc.max_speed == 15.0
        assert rc.detection_method == "hybrid"
