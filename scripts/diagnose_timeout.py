"""Diagnostic benchmark for seeds 9 and 12 to understand timeout behavior.

For each seed, runs an episode with 5 gates, max 2000 steps.
Every 50 steps, prints: step, gates_passed, drone position, velocity,
distance to next gate, next gate position.
Also prints when gates_passed changes.
"""

from __future__ import annotations

import sys
import time

import numpy as np

# Ensure src is on the path
sys.path.insert(0, "/Users/caiolacerda/TOTVS/Workspace/projects/AIGrandPrix/src")

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


def print_status(
    step, gates_passed, total_gates, pos, vel, next_gate_pos, label="",
):
    speed = np.linalg.norm(vel)
    if next_gate_pos is not None:
        dist = np.linalg.norm(pos - next_gate_pos)
        gate_str = (
            f"  next_gate=[{next_gate_pos[0]:7.2f}, {next_gate_pos[1]:7.2f}, {next_gate_pos[2]:7.2f}]"
            f"  dist_to_gate={dist:7.2f}"
        )
    else:
        gate_str = "  (all gates passed)"

    tag = f"  <<< {label}" if label else ""
    print(
        f"  step={step:5d}  gates={gates_passed}/{total_gates}"
        f"  pos=[{pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}]"
        f"  vel=[{vel[0]:7.2f}, {vel[1]:7.2f}, {vel[2]:7.2f}]"
        f"  speed={speed:5.2f}"
        f"{gate_str}"
        f"{tag}"
    )


def run_diagnostic(seed, num_gates=5, max_steps=2000):
    print("=" * 140)
    print(f"SEED {seed}: {num_gates} gates, max {max_steps} steps")
    print("=" * 140)

    env = DroneRacingEnv(num_gates=num_gates, max_steps=max_steps)
    env.reset(seed=seed)
    gates = _env_gates(env)

    # Print gate layout
    print("\nGate layout:")
    for i, g in enumerate(gates):
        p = g.position
        print(f"  Gate {i}: pos=[{p[0]:7.2f}, {p[1]:7.2f}, {p[2]:7.2f}]  normal=[{g.normal[0]:.2f}, {g.normal[1]:.2f}, {g.normal[2]:.2f}]")
    print()

    # Print inter-gate distances
    print("Inter-gate distances:")
    start = np.array([0.0, 0.0, 1.0])
    for i, g in enumerate(gates):
        prev = start if i == 0 else gates[i - 1].position
        d = np.linalg.norm(g.position - prev)
        dz = g.position[2] - prev[2]
        print(f"  {('start' if i == 0 else f'Gate {i-1}'):>7s} -> Gate {i}: dist={d:6.2f}  dz={dz:+5.2f}")
    print()

    agent = RacingAgent(RaceConfig(max_speed=10.0, control_dt=0.02))
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    max_gates_passed = 0
    prev_gates_passed = 0
    crashed = False
    completed = False
    truncated_flag = False

    # Track time
    t_start = time.perf_counter()

    # Track min distance to each gate
    min_dist_to_next_gate = float("inf")
    min_dist_history = {}  # gate_idx -> min_dist

    # Print initial state
    state0 = _env_state(env)
    next_gate_pos = gates[0].position if len(gates) > 0 else None
    print_status(0, 0, num_gates, state0.position, state0.velocity, next_gate_pos, label="INITIAL")

    step = 0
    for step in range(1, max_steps + 1):
        state = _env_state(env)
        cmd = agent.compute_action(
            image=dummy_image,
            state=state,
            elapsed_time=step * 0.02,
            known_gates=gates,
        )
        action = np.array(
            [cmd.thrust, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate],
            dtype=np.float32,
        )
        action = np.clip(action, env.action_space.low, env.action_space.high)
        _, _, terminated, truncated, info = env.step(action)

        current_gates_passed = info["gates_passed"]

        # Track min distance to next gate
        if current_gates_passed < num_gates:
            gate_pos = gates[current_gates_passed].position
            dist = np.linalg.norm(env._drone_pos - gate_pos)
            if dist < min_dist_to_next_gate:
                min_dist_to_next_gate = dist
            if current_gates_passed not in min_dist_history or dist < min_dist_history[current_gates_passed]:
                min_dist_history[current_gates_passed] = dist
        else:
            gate_pos = None

        # Gate passage event
        if current_gates_passed > prev_gates_passed:
            agent.on_gate_passed(prev_gates_passed)
            next_gp = gates[current_gates_passed].position if current_gates_passed < num_gates else None
            md = min_dist_history.get(prev_gates_passed, -1)
            print_status(
                step, current_gates_passed, num_gates,
                env._drone_pos.copy(), env._drone_vel.copy(), next_gp,
                label=f"GATE {prev_gates_passed} PASSED (min_dist was {md:.3f})",
            )
            min_dist_to_next_gate = float("inf")
            prev_gates_passed = current_gates_passed
            max_gates_passed = current_gates_passed

        # Periodic status every 50 steps
        if step % 50 == 0:
            next_gp = gates[current_gates_passed].position if current_gates_passed < num_gates else None
            min_d = min_dist_to_next_gate if min_dist_to_next_gate < float("inf") else -1
            print_status(
                step, current_gates_passed, num_gates,
                env._drone_pos.copy(), env._drone_vel.copy(), next_gp,
                label=f"min_dist_to_next={min_d:.3f}" if min_d >= 0 else "",
            )
            # Also print command info
            print(
                f"           cmd: thrust={cmd.thrust:.3f}  roll_r={cmd.roll_rate:+6.2f}"
                f"  pitch_r={cmd.pitch_rate:+6.2f}  yaw_r={cmd.yaw_rate:+6.2f}"
            )

        if terminated:
            crashed = env._drone_pos[2] < 0 or env._drone_pos[2] > 50
            completed = current_gates_passed >= num_gates
            break
        if truncated:
            truncated_flag = True
            break

    elapsed = time.perf_counter() - t_start

    # Final state
    final_pos = env._drone_pos.copy()
    final_vel = env._drone_vel.copy()
    final_speed = np.linalg.norm(final_vel)

    print()
    print("-" * 140)
    print(f"FINAL STATE (seed={seed}):")
    print(f"  steps={step}  gates_passed={max_gates_passed}/{num_gates}  wall_time={elapsed:.2f}s")
    print(f"  pos=[{final_pos[0]:7.2f}, {final_pos[1]:7.2f}, {final_pos[2]:7.2f}]"
          f"  vel=[{final_vel[0]:7.2f}, {final_vel[1]:7.2f}, {final_vel[2]:7.2f}]  speed={final_speed:.2f}")

    # Failure mode
    if completed:
        print(f"  RESULT: COMPLETED all {num_gates} gates")
    elif crashed:
        if final_pos[2] < 0:
            print(f"  RESULT: CRASHED (hit ground, z={final_pos[2]:.3f})")
        else:
            print(f"  RESULT: CRASHED (ceiling, z={final_pos[2]:.3f})")
    elif truncated_flag:
        print(f"  RESULT: TIMEOUT (max_steps={max_steps} reached)")
    else:
        print(f"  RESULT: TERMINATED (unknown)")

    # Min distance to each gate
    print("\n  Min distance achieved to each gate:")
    for gate_idx in range(num_gates):
        md = min_dist_history.get(gate_idx, float("inf"))
        passed = "PASSED" if gate_idx < max_gates_passed else "NOT PASSED"
        if md < float("inf"):
            print(f"    Gate {gate_idx}: min_dist={md:.3f}  [{passed}]")
        else:
            print(f"    Gate {gate_idx}: never approached  [{passed}]")

    # If stuck, check what gate we're stuck on
    if not completed and not crashed:
        stuck_gate = max_gates_passed
        if stuck_gate < num_gates:
            gp = gates[stuck_gate].position
            dist = np.linalg.norm(final_pos - gp)
            direction = gp - final_pos
            print(f"\n  STUCK on Gate {stuck_gate}: gate_pos=[{gp[0]:.2f}, {gp[1]:.2f}, {gp[2]:.2f}]"
                  f"  dist={dist:.2f}  direction=[{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]")

    print("-" * 140)
    print()
    env.close()
    return {
        "seed": seed,
        "gates_passed": max_gates_passed,
        "total_gates": num_gates,
        "steps": step,
        "crashed": crashed,
        "completed": completed,
        "timeout": truncated_flag,
        "min_dist_history": min_dist_history,
    }


if __name__ == "__main__":
    results = []
    for seed in [9, 12]:
        r = run_diagnostic(seed)
        results.append(r)
        print()

    # Summary
    print("=" * 140)
    print("SUMMARY")
    print("=" * 140)
    for r in results:
        status = "COMPLETED" if r["completed"] else ("CRASHED" if r["crashed"] else "TIMEOUT")
        min_dists = ", ".join(
            f"G{k}={v:.2f}" for k, v in sorted(r["min_dist_history"].items())
        )
        print(
            f"  Seed {r['seed']:2d}: gates={r['gates_passed']}/{r['total_gates']}"
            f"  steps={r['steps']:5d}  status={status}"
            f"  min_dists=[{min_dists}]"
        )
