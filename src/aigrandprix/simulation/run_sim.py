"""Simulation runner - runs episodes and visualizes drone trajectories.

Usage:
    python -m aigrandprix.simulation.run_sim [--episodes N] [--seed S] [--plot]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from aigrandprix.control.drone_controller import DroneState
from aigrandprix.main import RaceConfig, RacingAgent
from aigrandprix.planning.path_planner import Gate
from aigrandprix.simulation.gym_env import DroneRacingEnv


@dataclass
class EpisodeResult:
    gates_passed: int
    total_gates: int
    total_steps: int
    max_speed: float
    avg_speed: float
    crashed: bool
    completed: bool
    positions: np.ndarray  # (T, 3)
    velocities: np.ndarray  # (T, 3)
    rpy: np.ndarray  # (T, 3) roll, pitch, yaw in radians
    commands: np.ndarray  # (T, 4) thrust, roll_rate, pitch_rate, yaw_rate
    gate_positions: np.ndarray  # (N, 3)
    gate_pass_steps: list[int]  # step indices where gates were passed


def _env_state(env: DroneRacingEnv) -> DroneState:
    return DroneState(
        position=env._drone_pos.copy(),
        velocity=env._drone_vel.copy(),
        orientation=env._drone_orient.copy(),
        angular_velocity=env._drone_angular_vel.copy(),
    )


def _build_gates(env: DroneRacingEnv) -> list[Gate]:
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


def run_episode(
    seed: int = 42,
    num_gates: int = 5,
    max_steps: int = 2000,
    max_speed: float = 12.0,
    full_config: dict | None = None,
) -> EpisodeResult:
    """Run a single episode with the racing agent."""
    env = DroneRacingEnv(num_gates=num_gates, max_steps=max_steps)
    env.reset(seed=seed)
    gates = _build_gates(env)
    gate_positions = np.array([g.position for g in gates])

    race_cfg = RaceConfig(max_speed=max_speed, control_dt=0.02)
    agent = RacingAgent(race_cfg, full_config=full_config)
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    positions = [env._drone_pos.copy()]
    velocities = [env._drone_vel.copy()]
    rpy_log = [env._drone_rpy.copy()]
    commands_log: list[np.ndarray] = []
    speeds = []
    gate_pass_steps: list[int] = []
    elapsed = 0.0
    dt = 0.02
    prev_gates_passed = 0
    crashed = False
    completed = False

    for step in range(max_steps):
        state = _env_state(env)
        cmd = agent.compute_action(
            image=dummy_image,
            state=state,
            elapsed_time=elapsed,
            known_gates=gates,
        )
        action = np.array(
            [cmd.thrust, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate],
            dtype=np.float32,
        )
        action = np.clip(action, env.action_space.low, env.action_space.high)
        commands_log.append(action.copy())

        _, _, terminated, truncated, info = env.step(action)
        elapsed += dt

        positions.append(env._drone_pos.copy())
        velocities.append(env._drone_vel.copy())
        rpy_log.append(env._drone_rpy.copy())
        speeds.append(info["speed"])

        if info["gates_passed"] > prev_gates_passed:
            gate_pass_steps.append(step)
            agent.on_gate_passed(prev_gates_passed)
            prev_gates_passed = info["gates_passed"]

        if terminated:
            crashed = env._drone_pos[2] < 0 or env._drone_pos[2] > 50
            completed = info["gates_passed"] >= len(gates)
            break
        if truncated:
            break

    env.close()

    return EpisodeResult(
        gates_passed=prev_gates_passed,
        total_gates=len(gates),
        total_steps=len(positions) - 1,
        max_speed=max(speeds) if speeds else 0,
        avg_speed=float(np.mean(speeds)) if speeds else 0,
        crashed=crashed,
        completed=completed,
        positions=np.array(positions),
        velocities=np.array(velocities),
        rpy=np.array(rpy_log),
        commands=np.array(commands_log) if commands_log else np.zeros((0, 4)),
        gate_positions=gate_positions,
        gate_pass_steps=gate_pass_steps,
    )


def plot_trajectory(result: EpisodeResult, title: str = "Drone Trajectory") -> None:
    """Generate a multi-panel plot: 3D trajectory, top-down view, speed & altitude profiles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    pos = result.positions
    gp = result.gate_positions
    dt = 0.02
    times = np.arange(len(pos)) * dt
    speeds = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt

    # --- Panel 1: 3D trajectory ---
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], "b-", linewidth=0.8, alpha=0.8, label="Path")
    ax1.scatter(*pos[0], c="green", s=100, marker="^", label="Start", zorder=5)
    end_c = "red" if result.crashed else "green"
    end_m = "x" if result.crashed else "o"
    ax1.scatter(*pos[-1], c=end_c, s=100, marker=end_m, label="Crash" if result.crashed else "End", zorder=5)
    for i, g in enumerate(gp):
        c = "lime" if i < result.gates_passed else "orange"
        ax1.scatter(*g, c=c, s=200, marker="s", edgecolors="black", linewidths=1)
        ax1.text(g[0], g[1], g[2] + 0.5, f"G{i}", fontsize=8, ha="center")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D Trajectory")
    ax1.legend(fontsize=8)

    # --- Panel 2: Top-down (XY) view ---
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(pos[:, 0], pos[:, 1], "b-", linewidth=0.8, alpha=0.7)
    ax2.scatter(pos[0, 0], pos[0, 1], c="green", s=100, marker="^", zorder=5, label="Start")
    ax2.scatter(pos[-1, 0], pos[-1, 1], c=end_c, s=100, marker=end_m, zorder=5)
    for i, g in enumerate(gp):
        c = "lime" if i < result.gates_passed else "orange"
        ax2.scatter(g[0], g[1], c=c, s=150, marker="s", edgecolors="black", linewidths=1)
        ax2.annotate(f"G{i}", (g[0], g[1]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Top-Down View")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Speed profile ---
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(times[1:], speeds, "r-", linewidth=0.8, alpha=0.8)
    ax3.axhline(y=np.mean(speeds), color="gray", linestyle="--", alpha=0.5, label=f"Avg: {np.mean(speeds):.1f} m/s")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Speed (m/s)")
    ax3.set_title("Speed Profile")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Altitude profile ---
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(times, pos[:, 2], "g-", linewidth=0.8, alpha=0.8, label="Altitude")
    for i, g in enumerate(gp):
        c = "lime" if i < result.gates_passed else "orange"
        ax4.axhline(y=g[2], color=c, linestyle=":", alpha=0.4)
    ax4.axhline(y=0, color="red", linestyle="-", alpha=0.3, label="Ground")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Z (m)")
    ax4.set_title("Altitude Profile")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "trajectory.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Trajectory plot saved to {out_path}")
    plt.close(fig)


def plot_telemetry(result: EpisodeResult, title: str = "Flight Telemetry") -> None:
    """Time-series telemetry: position, roll/pitch/yaw, and motor commands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dt = 0.02
    n_pos = len(result.positions)
    n_cmd = len(result.commands)
    t_pos = np.arange(n_pos) * dt
    t_cmd = np.arange(n_cmd) * dt

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Vertical lines for gate passages
    def _draw_gate_lines(ax: plt.Axes) -> None:
        for i, gs in enumerate(result.gate_pass_steps):
            t_gate = gs * dt
            ax.axvline(x=t_gate, color="green", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.text(
                t_gate, ax.get_ylim()[1] * 0.95, f"G{i}",
                fontsize=7, color="green", ha="center", va="top",
            )

    # --- Panel 1: Position X, Y, Z ---
    ax = axes[0]
    ax.plot(t_pos, result.positions[:, 0], "r-", linewidth=0.9, label="X")
    ax.plot(t_pos, result.positions[:, 1], "g-", linewidth=0.9, label="Y")
    ax.plot(t_pos, result.positions[:, 2], "b-", linewidth=0.9, label="Z")
    ax.set_ylabel("Position (m)")
    ax.set_title("Position vs Time")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    for i, gs in enumerate(result.gate_pass_steps):
        ax.axvline(x=gs * dt, color="green", linestyle="--", alpha=0.5, linewidth=0.8)

    # --- Panel 2: Roll, Pitch, Yaw ---
    ax = axes[1]
    rpy_deg = np.degrees(result.rpy)
    ax.plot(t_pos, rpy_deg[:, 0], "r-", linewidth=0.9, label="Roll")
    ax.plot(t_pos, rpy_deg[:, 1], "g-", linewidth=0.9, label="Pitch")
    ax.plot(t_pos, rpy_deg[:, 2], "b-", linewidth=0.9, label="Yaw")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Orientation (Roll / Pitch / Yaw)")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    for i, gs in enumerate(result.gate_pass_steps):
        ax.axvline(x=gs * dt, color="green", linestyle="--", alpha=0.5, linewidth=0.8)

    # --- Panel 3: Motor thrust (normalized 0-1, mapped to approx RPM) ---
    ax = axes[2]
    if n_cmd > 0:
        thrust = result.commands[:, 0]
        # Map normalized thrust [0,1] to approximate motor RPM [0, ~25000]
        motor_rpm = thrust * 25000
        ax.plot(t_cmd, motor_rpm, color="purple", linewidth=0.9, label="Motor RPM (avg)")
        ax.fill_between(t_cmd, 0, motor_rpm, alpha=0.15, color="purple")
        ax.set_ylabel("Motor Speed (RPM)")
        ax.set_title("Motor Speed (estimated from thrust)")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    for i, gs in enumerate(result.gate_pass_steps):
        ax.axvline(x=gs * dt, color="green", linestyle="--", alpha=0.5, linewidth=0.8)

    # --- Panel 4: Angular rate commands ---
    ax = axes[3]
    if n_cmd > 0:
        ax.plot(t_cmd, result.commands[:, 1], "r-", linewidth=0.8, alpha=0.8, label="Roll rate cmd")
        ax.plot(t_cmd, result.commands[:, 2], "g-", linewidth=0.8, alpha=0.8, label="Pitch rate cmd")
        ax.plot(t_cmd, result.commands[:, 3], "b-", linewidth=0.8, alpha=0.8, label="Yaw rate cmd")
        ax.set_ylabel("Rate (rad/s)")
        ax.set_title("Angular Rate Commands")
        ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    for i, gs in enumerate(result.gate_pass_steps):
        ax.axvline(x=gs * dt, color="green", linestyle="--", alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    out_path = "telemetry.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Telemetry plot saved to {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Grand Prix Simulation Runner")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--gates", type=int, default=5, help="Number of gates")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--max-speed", type=float, default=12.0, help="Max drone speed (m/s)")
    parser.add_argument("--plot", action="store_true", help="Save trajectory plot")
    parser.add_argument("--telemetry", action="store_true", help="Save telemetry time-series plot")
    args = parser.parse_args()

    print("=" * 60)
    print("  AI Grand Prix - Simulation Runner")
    print("=" * 60)
    print(f"  Episodes: {args.episodes}  |  Gates: {args.gates}  |  Max speed: {args.max_speed} m/s")
    print("=" * 60)

    results: list[EpisodeResult] = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        result = run_episode(
            seed=seed,
            num_gates=args.gates,
            max_steps=args.max_steps,
            max_speed=args.max_speed,
        )
        results.append(result)

        status = "COMPLETED" if result.completed else ("CRASHED" if result.crashed else "TIMEOUT")
        print(
            f"  Episode {ep + 1:>2}: {status:>9} | "
            f"Gates: {result.gates_passed}/{result.total_gates} | "
            f"Steps: {result.total_steps:>5} | "
            f"Avg speed: {result.avg_speed:>5.1f} m/s | "
            f"Max speed: {result.max_speed:>5.1f} m/s"
        )

    print("-" * 60)
    total_passed = sum(r.gates_passed for r in results)
    total_gates = sum(r.total_gates for r in results)
    completions = sum(1 for r in results if r.completed)
    crashes = sum(1 for r in results if r.crashed)
    avg_speed_all = float(np.mean([r.avg_speed for r in results]))
    print(
        f"  Summary: {total_passed}/{total_gates} gates | "
        f"{completions} completed | {crashes} crashes | "
        f"Avg speed: {avg_speed_all:.1f} m/s"
    )
    print("=" * 60)

    if (args.plot or args.telemetry) and results:
        best = max(results, key=lambda r: (r.gates_passed, -r.total_steps))
        run_label = f"Best run: {best.gates_passed}/{best.total_gates} gates"
        if args.plot:
            plot_trajectory(best, title=run_label)
        if args.telemetry:
            plot_telemetry(best, title=f"Telemetry - {run_label}")


if __name__ == "__main__":
    main()
