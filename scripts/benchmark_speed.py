#!/usr/bin/env python3
"""Benchmark speed performance across different config presets.

Compares gate completion rate AND average flight speed to find
the optimal speed/reliability trade-off.

Usage:
    python scripts/benchmark_speed.py [--episodes N] [--gates G]
"""

from __future__ import annotations

import argparse
import sys
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
            d = pos - np.array([0, 0, 1])
        else:
            d = pos - env._gates[i - 1]
        n = np.linalg.norm(d[:2])
        normal = np.array([d[0], d[1], 0.0]) / n if n > 0.01 else np.array([1, 0, 0])
        gates.append(Gate(position=pos.copy(), normal=normal, gate_id=i))
    return gates


# Preset configurations to benchmark
PRESETS = {
    "conservative": RaceConfig(
        max_speed=12.0,
        replan_interval=0.5,
        gate_closing_range=5.0,
        gate_closing_speed_frac=0.4,
        gate_closing_speed_cap=5.0,
        gate_closing_dist_scale=1.5,
        gate_closing_vz_clamp=-1.5,
        trajectory_lookahead=20,
        trajectory_speed_frac=0.6,
    ),
    "moderate": RaceConfig(
        max_speed=15.0,
        replan_interval=0.3,
        gate_closing_range=3.5,
        gate_closing_speed_frac=0.7,
        gate_closing_speed_cap=8.0,
        gate_closing_dist_scale=2.5,
        gate_closing_vz_clamp=-2.5,
        trajectory_lookahead=40,
        trajectory_speed_frac=0.85,
    ),
    "tuned": RaceConfig(
        max_speed=15.0,
        replan_interval=0.3,
        gate_closing_range=4.0,
        gate_closing_speed_frac=0.5,
        gate_closing_speed_cap=7.0,
        gate_closing_dist_scale=2.0,
        gate_closing_vz_clamp=-2.0,
        trajectory_lookahead=30,
        trajectory_speed_frac=0.7,
    ),
    "aggressive": RaceConfig(
        max_speed=15.0,
        replan_interval=0.2,
        gate_closing_range=2.5,
        gate_closing_speed_frac=0.9,
        gate_closing_speed_cap=12.0,
        gate_closing_dist_scale=3.5,
        gate_closing_vz_clamp=-3.0,
        trajectory_lookahead=60,
        trajectory_speed_frac=0.95,
    ),
}


def run_preset(
    name: str,
    config: RaceConfig,
    num_episodes: int,
    num_gates: int,
    base_seed: int,
    max_steps: int,
) -> dict:
    """Run benchmark episodes for a single preset."""
    results = []

    for ep in range(num_episodes):
        seed = base_seed + ep
        env = DroneRacingEnv(num_gates=num_gates, max_steps=max_steps)
        env.reset(seed=seed)
        gates = _env_gates(env)

        config_copy = RaceConfig(
            max_speed=config.max_speed,
            max_accel=config.max_accel,
            control_dt=0.02,
            replan_interval=config.replan_interval,
            gate_closing_range=config.gate_closing_range,
            gate_closing_speed_frac=config.gate_closing_speed_frac,
            gate_closing_speed_cap=config.gate_closing_speed_cap,
            gate_closing_dist_scale=config.gate_closing_dist_scale,
            gate_closing_vz_clamp=config.gate_closing_vz_clamp,
            trajectory_lookahead=config.trajectory_lookahead,
            trajectory_speed_frac=config.trajectory_speed_frac,
        )
        agent = RacingAgent(config_copy)
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        max_gates_passed = 0
        speeds = []

        terminated = False
        truncated = False
        info = {}

        for step in range(max_steps):
            state = _env_state(env)
            speeds.append(state.speed)
            cmd = agent.compute_action(
                image=dummy_image, state=state,
                elapsed_time=step * 0.02, known_gates=gates,
            )
            action = np.array(
                [cmd.thrust, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate],
                dtype=np.float32,
            )
            action = np.clip(action, env.action_space.low, env.action_space.high)
            _, _, terminated, truncated, info = env.step(action)

            if info["gates_passed"] > max_gates_passed:
                agent.on_gate_passed(max_gates_passed)
                max_gates_passed = info["gates_passed"]

            if terminated or truncated:
                break

        env.close()
        completed = max_gates_passed >= num_gates
        results.append({
            "seed": seed,
            "gates_passed": max_gates_passed,
            "total_gates": num_gates,
            "steps": step + 1,
            "completed": completed,
            "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
            "max_speed": float(np.max(speeds)) if speeds else 0.0,
            "sim_time": (step + 1) * 0.02,
        })

    return {"name": name, "episodes": results}


def print_report(all_results: list[dict], num_gates: int) -> None:
    """Print comparison report across presets."""
    print("=" * 90)
    print("  Speed Optimization Benchmark")
    print("=" * 90)

    header = f"  {'Preset':<14} | {'Gate %':>7} | {'Avg Spd':>8} | {'Max Spd':>8} | {'Avg Time':>9} | {'Crashes':>7}"
    print(header)
    print("-" * 90)

    for preset in all_results:
        name = preset["name"]
        eps = preset["episodes"]
        n = len(eps)
        total_passed = sum(e["gates_passed"] for e in eps)
        total_gates = sum(e["total_gates"] for e in eps)
        gate_pct = 100 * total_passed / total_gates if total_gates > 0 else 0
        avg_speed = float(np.mean([e["avg_speed"] for e in eps]))
        max_speed = float(np.max([e["max_speed"] for e in eps]))
        avg_time = float(np.mean([e["sim_time"] for e in eps]))
        crashes = sum(1 for e in eps if not e["completed"] and e["steps"] < 2000)

        print(
            f"  {name:<14} | {gate_pct:>6.1f}% | {avg_speed:>7.2f} | {max_speed:>7.2f} | "
            f"{avg_time:>8.2f}s | {crashes:>3}/{n}"
        )

    print("=" * 90)

    # Detailed per-episode for each preset
    for preset in all_results:
        name = preset["name"]
        eps = preset["episodes"]
        print(f"\n  --- {name} ---")
        for ep in eps:
            status = "OK" if ep["completed"] else "FAIL"
            print(
                f"    Seed {ep['seed']:>3}: {status:>4} | "
                f"Gates: {ep['gates_passed']}/{ep['total_gates']} | "
                f"Avg: {ep['avg_speed']:.2f} | "
                f"Max: {ep['max_speed']:.2f} | "
                f"Time: {ep['sim_time']:.2f}s"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Speed Benchmark")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per preset")
    parser.add_argument("--gates", type=int, default=5, help="Gates per course")
    parser.add_argument("--seed", type=int, default=0, help="Base seed")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps/episode")
    parser.add_argument(
        "--presets", nargs="*", default=list(PRESETS.keys()),
        help="Which presets to run (default: all)",
    )
    args = parser.parse_args()

    all_results = []
    for name in args.presets:
        if name not in PRESETS:
            print(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
            sys.exit(1)
        print(f"Running preset '{name}' ({args.episodes} episodes)...")
        t0 = time.time()
        result = run_preset(
            name=name,
            config=PRESETS[name],
            num_episodes=args.episodes,
            num_gates=args.gates,
            base_seed=args.seed,
            max_steps=args.max_steps,
        )
        print(f"  Done in {time.time() - t0:.1f}s")
        all_results.append(result)

    print_report(all_results, args.gates)


if __name__ == "__main__":
    main()
