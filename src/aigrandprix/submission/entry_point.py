"""Entry point for DCL competition submission.

This is the class DCL will instantiate and call. It wraps our full
autonomy stack (perception + planning + control) behind a clean API.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np

from aigrandprix.control.drone_controller import ControlCommand, DroneState
from aigrandprix.main import RaceConfig, RacingAgent
from aigrandprix.planning.path_planner import Gate
from aigrandprix.submission.safety import SafeActionWrapper
from aigrandprix.submission.type_converters import (
    command_to_array,
    gates_from_race_info,
    observation_to_internal,
)

logger = logging.getLogger(__name__)


class AutonomousRacer:
    """Main competition entry point.

    Lifecycle:
        racer = AutonomousRacer(config)
        racer.initialize(race_info)
        for each frame:
            action = racer.compute(observation)
        racer.shutdown()
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._agent: RacingAgent | None = None
        self._safety: SafeActionWrapper | None = None
        self._gates: list[Gate] = []
        self._initialized = False
        self._frame_count = 0
        self._elapsed = 0.0
        self._dt = self._config.get("race", {}).get("control_dt", 0.02)

    def initialize(self, race_info: dict) -> None:
        """Initialize for a race. Called once before compute() loop.

        Args:
            race_info: Dict with gate positions, course info, etc.
                Expected keys (adapt when DCL specs arrive):
                - "gates": list of gate dicts with "position" and optionally "normal"
                - "dt": simulation timestep (optional, default 0.02)
                - "num_gates": number of gates (optional)
        """
        self._gates = gates_from_race_info(race_info)
        self._dt = race_info.get("dt", self._dt)

        race_cfg = RaceConfig.from_config(self._config) if self._config else RaceConfig()
        self._agent = RacingAgent(config=race_cfg, full_config=self._config)
        self._safety = SafeActionWrapper(
            action_dim=4,
            max_latency_ms=self._config.get("safety", {}).get("max_latency_ms", 50.0),
        )

        self._frame_count = 0
        self._elapsed = 0.0
        self._initialized = True
        logger.info("AutonomousRacer initialized with %d gates", len(self._gates))

    def compute(self, observation: dict) -> np.ndarray:
        """Compute action for one frame.

        Args:
            observation: Dict from DCL platform (format TBD).
                Expected keys:
                - "image": camera image (H,W,3) BGR
                - "position": [x,y,z]
                - "velocity": [vx,vy,vz]
                - "orientation": quaternion [w,x,y,z]
                - "angular_velocity": [wx,wy,wz]
                - "elapsed_time": float
                - "current_gate_index": int (optional)

        Returns:
            np.ndarray of shape (4,) — [thrust, roll_rate, pitch_rate, yaw_rate]
        """
        if not self._initialized:
            raise RuntimeError("Must call initialize() before compute()")

        image, state, elapsed, gate_idx = observation_to_internal(observation)
        self._elapsed = elapsed if elapsed is not None else self._elapsed + self._dt
        self._frame_count += 1

        # External gate sync
        if gate_idx is not None and gate_idx > self._agent._gates_passed:
            for _ in range(gate_idx - self._agent._gates_passed):
                self._agent.on_gate_passed(self._agent._gates_passed)

        def _compute_inner() -> np.ndarray:
            cmd = self._agent.compute_action(
                image=image,
                state=state,
                elapsed_time=self._elapsed,
                known_gates=self._gates,
                current_gate_index=gate_idx,
            )
            return command_to_array(cmd)

        return self._safety.safe_compute(_compute_inner)

    def on_gate_passed(self, gate_id: int) -> None:
        """Explicit notification that a gate was passed."""
        if self._agent is not None:
            self._agent.on_gate_passed(gate_id)

    def shutdown(self) -> None:
        """Cleanup and log stats."""
        if self._safety is not None:
            stats = self._safety.get_stats()
            logger.info(
                "Shutdown after %d frames. Latency p50=%.1fms p95=%.1fms p99=%.1fms. "
                "Fallbacks=%d",
                self._frame_count,
                stats.get("p50_ms", 0),
                stats.get("p95_ms", 0),
                stats.get("p99_ms", 0),
                stats.get("fallback_count", 0),
            )
        self._initialized = False
        self._agent = None

    def run_loop(self, sim) -> dict:
        """Pull-based execution loop for internal testing.

        Args:
            sim: SimInterface instance

        Returns:
            dict with episode results
        """
        from aigrandprix.simulation.sim_interface import SimInterface

        if not isinstance(sim, SimInterface):
            raise TypeError(f"Expected SimInterface, got {type(sim)}")

        info = sim.reset()
        gates = sim.get_gate_poses()
        race_info = {
            "gates": [{"position": g.position.tolist(), "normal": g.normal.tolist()} for g in gates],
            "dt": sim.dt,
        }
        self.initialize(race_info)

        max_gates = 0
        steps = 0
        crashed = False
        completed = False

        while True:
            state = sim.get_state()
            image = sim.get_image()
            gate_idx = sim.get_next_gate_index()

            obs = {
                "image": image,
                "position": state.position.tolist(),
                "velocity": state.velocity.tolist(),
                "orientation": state.orientation.tolist(),
                "angular_velocity": state.angular_velocity.tolist(),
                "elapsed_time": steps * sim.dt,
                "current_gate_index": gate_idx,
            }

            action = self.compute(obs)
            terminated, truncated, step_info = sim.step(action)
            steps += 1

            if step_info.get("gates_passed", 0) > max_gates:
                self.on_gate_passed(max_gates)
                max_gates = step_info["gates_passed"]

            if terminated:
                crashed = step_info.get("crashed", False)
                completed = step_info.get("gates_passed", 0) >= len(gates)
                break
            if truncated:
                break

        self.shutdown()
        return {
            "gates_passed": max_gates,
            "total_gates": len(gates),
            "steps": steps,
            "crashed": crashed,
            "completed": completed,
        }
