"""Parametric sweep for PID + attitude gains under full dynamics.

Runs short episodes for each gain combination and reports the best config.

Usage:
    python scripts/tune_full_dynamics.py [--steps 200] [--seeds 3] [--gates 3]
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

from aigrandprix.control.drone_controller import DroneState
from aigrandprix.main import RaceConfig, RacingAgent
from aigrandprix.planning.path_planner import Gate
from aigrandprix.simulation.gym_env import DroneRacingEnv


def _env_state(env: DroneRacingEnv) -> DroneState:
    return DroneState(
        position=env._drone_pos.copy(),
        velocity=env._drone_vel.copy(),
        orientation=env._drone_orient.copy(),
        angular_velocity=env._drone_angular_vel.copy(),
    )


def _env_gates(env: DroneRacingEnv) -> list[Gate]:
    gates = []
    for i, pos in enumerate(env._gates):
        if i == 0:
            direction = pos - np.array([0, 0, 1])
        else:
            direction = pos - env._gates[i - 1]
        norm = np.linalg.norm(direction)
        normal = direction / norm if norm > 0.01 else np.array([1, 0, 0])
        normal[2] = 0
        ln = np.linalg.norm(normal)
        if ln > 0.01:
            normal /= ln
        else:
            normal = np.array([1, 0, 0])
        gates.append(Gate(position=pos.copy(), normal=normal, gate_id=i))
    return gates


def run_trial(
    pid_kp: list[float],
    pid_kd: list[float],
    pid_ki: list[float],
    att_kp: float,
    att_kd: float,
    max_speed: float,
    num_gates: int,
    max_steps: int,
    seed: int,
) -> dict:
    """Run a single trial and return metrics."""
    full_config = {
        "control": {
            "physics_mode": "full",
            "pid": {"kp": pid_kp, "kd": pid_kd, "ki": pid_ki},
            "attitude": {"kp": att_kp, "kd": att_kd},
        },
        "simulation": {"use_dynamics_model": True},
    }

    env = DroneRacingEnv(num_gates=num_gates, max_steps=max_steps, use_dynamics_model=True)
    env.reset(seed=seed)
    gates = _env_gates(env)

    agent = RacingAgent(
        RaceConfig(max_speed=max_speed, control_dt=0.02),
        full_config=full_config,
    )
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    gates_passed = 0
    crashed = False
    for step in range(max_steps):
        state = _env_state(env)
        cmd = agent.compute_action(
            image=dummy_image, state=state, elapsed_time=step * 0.02, known_gates=gates,
        )
        action = np.array(
            [cmd.thrust, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate], dtype=np.float32,
        )
        action = np.clip(action, env.action_space.low, env.action_space.high)
        _, _, terminated, truncated, info = env.step(action)
        if info["gates_passed"] > gates_passed:
            agent.on_gate_passed(gates_passed)
            gates_passed = info["gates_passed"]
        if terminated:
            crashed = env._drone_pos[2] < 0 or env._drone_pos[2] > 50
            break
        if truncated:
            break

    total_steps = min(step + 1, max_steps)
    env.close()

    return {
        "gates_passed": gates_passed,
        "total_steps": total_steps,
        "crashed": crashed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full dynamics gain tuning sweep")
    parser.add_argument("--steps", type=int, default=200, help="Max steps per trial")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--gates", type=int, default=3, help="Number of gates")
    parser.add_argument("--max-speed", type=float, default=8.0, help="Max speed")
    args = parser.parse_args()

    # Gain search space
    pid_kp_options = [
        [3.0, 3.0, 5.0],
        [4.0, 4.0, 6.0],
        [6.0, 6.0, 8.0],
    ]
    pid_kd_options = [
        [3.0, 3.0, 4.0],
        [4.0, 4.0, 5.0],
        [5.0, 5.0, 6.0],
    ]
    pid_ki_options = [
        [0.01, 0.01, 0.05],
        [0.05, 0.05, 0.1],
    ]
    att_kp_options = [6.0, 8.0, 10.0]
    att_kd_options = [1.5, 2.0, 3.0]

    combos = list(itertools.product(
        pid_kp_options, pid_kd_options, pid_ki_options, att_kp_options, att_kd_options,
    ))
    print(f"Testing {len(combos)} gain combinations x {args.seeds} seeds = {len(combos) * args.seeds} trials")
    print(f"Steps per trial: {args.steps}, Gates: {args.gates}")
    print("=" * 80)

    results = []
    t0 = time.time()
    for idx, (pid_kp, pid_kd, pid_ki, att_kp, att_kd) in enumerate(combos):
        total_gates = 0
        total_steps = 0
        total_crashed = 0

        for s in range(args.seeds):
            trial = run_trial(
                pid_kp=pid_kp,
                pid_kd=pid_kd,
                pid_ki=pid_ki,
                att_kp=att_kp,
                att_kd=att_kd,
                max_speed=args.max_speed,
                num_gates=args.gates,
                max_steps=args.steps,
                seed=42 + s,
            )
            total_gates += trial["gates_passed"]
            total_steps += trial["total_steps"]
            total_crashed += int(trial["crashed"])

        avg_gates = total_gates / args.seeds
        avg_steps = total_steps / args.seeds
        crash_rate = total_crashed / args.seeds

        # Score: gates_passed * 1000 - total_steps + (not crashed) * 500
        score = total_gates * 1000 - total_steps + (args.seeds - total_crashed) * 500

        results.append({
            "pid_kp": pid_kp,
            "pid_kd": pid_kd,
            "pid_ki": pid_ki,
            "att_kp": att_kp,
            "att_kd": att_kd,
            "avg_gates": avg_gates,
            "avg_steps": avg_steps,
            "crash_rate": crash_rate,
            "score": score,
        })

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx + 1}/{len(combos)}] elapsed: {elapsed:.1f}s")

    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)

    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 80)
    for i, r in enumerate(results[:10]):
        print(
            f"  #{i + 1:>2}  score={r['score']:>7.0f}  "
            f"gates={r['avg_gates']:.1f}  steps={r['avg_steps']:.0f}  "
            f"crash={r['crash_rate']:.0%}  "
            f"kp={r['pid_kp']}  kd={r['pid_kd']}  ki={r['pid_ki']}  "
            f"att_kp={r['att_kp']}  att_kd={r['att_kd']}"
        )

    print("\nBEST CONFIG:")
    best = results[0]
    print(f"  pid.kp: {best['pid_kp']}")
    print(f"  pid.kd: {best['pid_kd']}")
    print(f"  pid.ki: {best['pid_ki']}")
    print(f"  attitude.kp: {best['att_kp']}")
    print(f"  attitude.kd: {best['att_kd']}")
    print(f"  score: {best['score']:.0f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
