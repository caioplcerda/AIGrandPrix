#!/usr/bin/env python3
"""Benchmark gate completion rate and classify failure modes.

Usage:
    python scripts/benchmark_gate_completion.py [--episodes N] [--gates G] [--seeds S]
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


def classify_failure(env: DroneRacingEnv, terminated: bool, truncated: bool, info: dict) -> str:
    """Classify the failure mode of an episode."""
    if info.get("gates_passed", 0) >= len(env._gates):
        return "completed"
    if terminated:
        z = env._drone_pos[2]
        if z < 0:
            return "ground_crash"
        elif z > 50:
            return "ceiling_crash"
        else:
            return "unknown_crash"
    if truncated:
        return "timeout"
    return "unknown"


def run_benchmark(
    num_episodes: int = 20,
    num_gates: int = 5,
    base_seed: int = 0,
    max_steps: int = 2000,
    max_speed: float = 12.0,
) -> dict:
    """Run benchmark episodes and collect stats."""
    results = []

    for ep in range(num_episodes):
        seed = base_seed + ep
        env = DroneRacingEnv(num_gates=num_gates, max_steps=max_steps)
        env.reset(seed=seed)
        gates = _env_gates(env)

        agent = RacingAgent(RaceConfig(max_speed=max_speed, control_dt=0.02))
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        max_gates_passed = 0

        t_start = time.time()
        terminated = False
        truncated = False
        info = {}

        for step in range(max_steps):
            state = _env_state(env)
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

        elapsed = time.time() - t_start
        failure_mode = classify_failure(env, terminated, truncated, info)
        env.close()

        results.append({
            "seed": seed,
            "gates_passed": max_gates_passed,
            "total_gates": num_gates,
            "steps": step + 1,
            "failure_mode": failure_mode,
            "wall_time": elapsed,
        })

    return {
        "episodes": results,
        "num_episodes": num_episodes,
        "num_gates": num_gates,
    }


def print_report(benchmark: dict) -> None:
    """Print formatted benchmark report."""
    episodes = benchmark["episodes"]
    n = benchmark["num_episodes"]
    ng = benchmark["num_gates"]

    print("=" * 70)
    print(f"  Gate Completion Benchmark: {n} episodes, {ng} gates each")
    print("=" * 70)

    # Per-episode results
    for ep in episodes:
        status = ep["failure_mode"].upper()
        print(
            f"  Seed {ep['seed']:>3}: {status:>14} | "
            f"Gates: {ep['gates_passed']}/{ep['total_gates']} | "
            f"Steps: {ep['steps']:>5} | "
            f"Time: {ep['wall_time']:.2f}s"
        )

    print("-" * 70)

    # Summary stats
    total_passed = sum(e["gates_passed"] for e in episodes)
    total_gates = sum(e["total_gates"] for e in episodes)
    completions = sum(1 for e in episodes if e["failure_mode"] == "completed")
    avg_gates = total_passed / n

    # Failure breakdown
    modes = {}
    for ep in episodes:
        m = ep["failure_mode"]
        modes[m] = modes.get(m, 0) + 1

    print(f"\n  Completion rate: {completions}/{n} ({100*completions/n:.0f}%)")
    print(f"  Gate pass rate:  {total_passed}/{total_gates} ({100*total_passed/total_gates:.0f}%)")
    print(f"  Avg gates/ep:    {avg_gates:.1f}/{ng}")
    print(f"\n  Failure breakdown:")
    for mode, count in sorted(modes.items()):
        print(f"    {mode:>14}: {count}/{n} ({100*count/n:.0f}%)")

    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate Completion Benchmark")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--gates", type=int, default=5, help="Gates per course")
    parser.add_argument("--seed", type=int, default=0, help="Base seed")
    parser.add_argument("--max-speed", type=float, default=12.0, help="Max speed")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps/episode")
    args = parser.parse_args()

    benchmark = run_benchmark(
        num_episodes=args.episodes,
        num_gates=args.gates,
        base_seed=args.seed,
        max_steps=args.max_steps,
        max_speed=args.max_speed,
    )
    print_report(benchmark)


if __name__ == "__main__":
    main()
