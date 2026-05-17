#!/usr/bin/env python3
"""Visualize a drone racing episode: 3D static plot + animated MP4.

Runs an actual episode using the RacingAgent, collects the drone trajectory,
and produces:
  - flight_visualization.png  (3D static plot with gates and trajectory)
  - flight_visualization.mp4  (animated flythrough with rotating camera)

Usage:
    python scripts/visualize_flight.py [--seed SEED] [--gates N] [--max-speed S]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless rendering

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from aigrandprix.control.drone_controller import DroneState
from aigrandprix.main import RaceConfig, RacingAgent
from aigrandprix.planning.path_planner import Gate
from aigrandprix.simulation.gym_env import DroneRacingEnv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def _env_state(env: DroneRacingEnv) -> DroneState:
    """Extract DroneState from environment internals."""
    return DroneState(
        position=env._drone_pos.copy(),
        velocity=env._drone_vel.copy(),
        orientation=env._drone_orient.copy(),
        angular_velocity=env._drone_angular_vel.copy(),
    )


def _env_gates(env: DroneRacingEnv) -> list[Gate]:
    """Build Gate objects from environment gate positions."""
    gates: list[Gate] = []
    for i, pos in enumerate(env._gates):
        if i == 0:
            d = pos - np.array([0.0, 0.0, 1.0])
        else:
            d = pos - env._gates[i - 1]
        n = np.linalg.norm(d[:2])
        normal = (
            np.array([d[0], d[1], 0.0]) / n if n > 0.01 else np.array([1.0, 0.0, 0.0])
        )
        gates.append(Gate(position=pos.copy(), normal=normal, gate_id=i))
    return gates


def gate_corners(gate: Gate, width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """Compute 4 corners of a gate rectangle in 3D.

    Returns shape (4, 3) array of corners.
    """
    up = np.array([0.0, 0.0, 1.0])
    # Horizontal axis perpendicular to gate normal and up
    right = np.cross(gate.normal, up)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        right = np.array([0.0, 1.0, 0.0])
    else:
        right = right / rn

    hw = width / 2.0
    hh = height / 2.0
    center = gate.position

    corners = np.array([
        center - right * hw - up * hh,
        center + right * hw - up * hh,
        center + right * hw + up * hh,
        center - right * hw + up * hh,
    ])
    return corners


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(
    seed: int = 3,
    num_gates: int = 5,
    max_speed: float = 12.0,
    max_steps: int = 2000,
) -> dict:
    """Run a single racing episode and collect telemetry.

    Returns dict with keys: positions, velocities, speeds, gates, gate_pass_steps,
    completed, steps, seed.
    """
    env = DroneRacingEnv(num_gates=num_gates, max_steps=max_steps)
    env.reset(seed=seed)
    gates = _env_gates(env)

    agent = RacingAgent(RaceConfig(max_speed=max_speed, control_dt=0.02))
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    speeds: list[float] = []
    gate_pass_steps: list[int] = []  # step index when each gate is passed
    max_gates_passed = 0

    for step in range(max_steps):
        state = _env_state(env)
        positions.append(state.position.copy())
        velocities.append(state.velocity.copy())
        speeds.append(float(np.linalg.norm(state.velocity)))

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

        if info["gates_passed"] > max_gates_passed:
            agent.on_gate_passed(max_gates_passed)
            gate_pass_steps.append(step)
            max_gates_passed = info["gates_passed"]

        if terminated or truncated:
            break

    env.close()

    return {
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "speeds": np.array(speeds),
        "gates": gates,
        "gate_pass_steps": gate_pass_steps,
        "completed": max_gates_passed >= num_gates,
        "steps": step + 1,
        "seed": seed,
        "max_gates_passed": max_gates_passed,
    }


# ---------------------------------------------------------------------------
# Static 3D plot
# ---------------------------------------------------------------------------


def create_static_plot(data: dict, output_path: Path) -> None:
    """Create a 3D static visualization of the flight."""
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    positions = data["positions"]
    speeds = data["speeds"]
    gates = data["gates"]
    gate_pass_steps = data["gate_pass_steps"]

    # --- Trajectory colored by speed ---
    # Create line segments colored by speed
    n_pts = len(positions)
    max_speed = max(speeds) if max(speeds) > 0 else 1.0
    norm_speeds = np.array(speeds) / max_speed

    # Use scatter for color-mapped trajectory
    sc = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=speeds,
        cmap="plasma",
        s=1.5,
        alpha=0.8,
        zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.08, label="Speed (m/s)")
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    # Also draw a thin connecting line for continuity
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="white",
        alpha=0.15,
        linewidth=0.5,
        zorder=1,
    )

    # --- Start position ---
    ax.scatter(
        [positions[0, 0]],
        [positions[0, 1]],
        [positions[0, 2]],
        color="#00ff88",
        s=120,
        marker="^",
        edgecolors="white",
        linewidths=1.5,
        zorder=5,
        label="Start",
    )

    # --- End position ---
    ax.scatter(
        [positions[-1, 0]],
        [positions[-1, 1]],
        [positions[-1, 2]],
        color="#ff4444",
        s=120,
        marker="v",
        edgecolors="white",
        linewidths=1.5,
        zorder=5,
        label="End",
    )

    # --- Gate rectangles ---
    gate_colors_passed = "#00ff88"  # green for passed
    gate_colors_missed = "#ff6666"  # red for missed
    n_passed = data["max_gates_passed"]

    for i, gate in enumerate(gates):
        corners = gate_corners(gate, width=1.0, height=1.0)
        color = gate_colors_passed if i < n_passed else gate_colors_missed
        alpha = 0.5 if i < n_passed else 0.3

        # Draw filled rectangle
        verts = [corners.tolist()]
        poly = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor="white", linewidths=1.2)
        ax.add_collection3d(poly)

        # Gate number label
        ax.text(
            gate.position[0],
            gate.position[1],
            gate.position[2] + 0.8,
            f"G{i}",
            color="white",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=10,
        )

    # --- Axes and labels ---
    ax.set_xlabel("X (m)", fontsize=11, labelpad=10)
    ax.set_ylabel("Y (m)", fontsize=11, labelpad=10)
    ax.set_zlabel("Z (m)", fontsize=11, labelpad=10)

    status = "COMPLETED" if data["completed"] else f"{n_passed}/{len(gates)} gates"
    ax.set_title(
        f"Drone Racing Flight  |  Seed {data['seed']}  |  {status}  |  "
        f"{data['steps']} steps  |  Max speed: {max(speeds):.1f} m/s",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    # Good viewing angle
    ax.view_init(elev=25, azim=-50)
    ax.legend(loc="upper left", fontsize=10)

    # Equal aspect ratio for all axes
    all_pts = np.vstack([positions] + [g.position.reshape(1, 3) for g in gates])
    x_range = all_pts[:, 0].max() - all_pts[:, 0].min()
    y_range = all_pts[:, 1].max() - all_pts[:, 1].min()
    z_range = all_pts[:, 2].max() - all_pts[:, 2].min()
    max_range = max(x_range, y_range, z_range, 1.0) / 2.0
    mid_x = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2.0
    mid_y = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2.0
    mid_z = (all_pts[:, 2].max() + all_pts[:, 2].min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(max(mid_z - max_range, -0.5), mid_z + max_range)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Static plot saved: {output_path}")


# ---------------------------------------------------------------------------
# Animated video
# ---------------------------------------------------------------------------


def create_animation(data: dict, output_path: Path, fps: int = 30) -> None:
    """Create an animated MP4/GIF of the flight."""
    plt.style.use("dark_background")

    positions = data["positions"]
    speeds = data["speeds"]
    gates = data["gates"]
    gate_pass_steps = data["gate_pass_steps"]
    n_passed = data["max_gates_passed"]

    # Subsample trajectory (every 3rd point)
    subsample = 3
    sub_positions = positions[::subsample]
    sub_speeds = speeds[::subsample]
    n_frames = len(sub_positions)

    # Map gate pass steps to subsampled frame indices
    gate_pass_frames = [s // subsample for s in gate_pass_steps]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Compute axis limits once
    all_pts = np.vstack([positions] + [g.position.reshape(1, 3) for g in gates])
    x_range = all_pts[:, 0].max() - all_pts[:, 0].min()
    y_range = all_pts[:, 1].max() - all_pts[:, 1].min()
    z_range = all_pts[:, 2].max() - all_pts[:, 2].min()
    max_range = max(x_range, y_range, z_range, 1.0) / 2.0
    mid_x = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2.0
    mid_y = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2.0
    mid_z = (all_pts[:, 2].max() + all_pts[:, 2].min()) / 2.0
    xlim = (mid_x - max_range, mid_x + max_range)
    ylim = (mid_y - max_range, mid_y + max_range)
    zlim = (max(mid_z - max_range, -0.5), mid_z + max_range)

    # Pre-compute gate visuals
    gate_corners_list = [gate_corners(g, 1.0, 1.0) for g in gates]

    def _draw_frame(frame_idx: int) -> None:
        ax.cla()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel("X (m)", fontsize=10, labelpad=8)
        ax.set_ylabel("Y (m)", fontsize=10, labelpad=8)
        ax.set_zlabel("Z (m)", fontsize=10, labelpad=8)

        # Camera rotation for drama
        base_azim = -50
        azim = base_azim + frame_idx * 0.15  # slow rotation
        elev = 25 + 5 * np.sin(frame_idx * 0.02)  # gentle up/down
        ax.view_init(elev=elev, azim=azim)

        # Determine which gates have been passed by this frame
        gates_passed_now = 0
        for gf in gate_pass_frames:
            if frame_idx >= gf:
                gates_passed_now += 1

        # Draw gates
        for i, (gate, corners) in enumerate(zip(gates, gate_corners_list)):
            if i < gates_passed_now:
                color = "#00ff88"
                alpha = 0.5
            else:
                color = "#ffaa00"
                alpha = 0.35
            verts = [corners.tolist()]
            poly = Poly3DCollection(
                verts, alpha=alpha, facecolor=color, edgecolor="white", linewidths=1.0,
            )
            ax.add_collection3d(poly)
            ax.text(
                gate.position[0],
                gate.position[1],
                gate.position[2] + 0.8,
                f"G{i}",
                color="white" if i >= gates_passed_now else "#00ff88",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

        # Draw trajectory trail up to current frame
        trail = sub_positions[: frame_idx + 1]
        trail_speeds = sub_speeds[: frame_idx + 1]
        if len(trail) > 1:
            ax.scatter(
                trail[:, 0],
                trail[:, 1],
                trail[:, 2],
                c=trail_speeds,
                cmap="plasma",
                s=1.5,
                alpha=0.6,
                vmin=0,
                vmax=max(speeds) if max(speeds) > 0 else 1,
            )

        # Draw drone marker at current position
        current_pos = sub_positions[frame_idx]
        current_speed = sub_speeds[frame_idx]
        ax.scatter(
            [current_pos[0]],
            [current_pos[1]],
            [current_pos[2]],
            color="#00ccff",
            s=80,
            marker="o",
            edgecolors="white",
            linewidths=1.5,
            zorder=10,
        )

        # Start marker
        ax.scatter(
            [sub_positions[0, 0]],
            [sub_positions[0, 1]],
            [sub_positions[0, 2]],
            color="#00ff88",
            s=60,
            marker="^",
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )

        # Title with speed indicator
        sim_time = frame_idx * subsample * 0.02
        status = f"Gates: {gates_passed_now}/{len(gates)}"
        ax.set_title(
            f"Drone Racing  |  {status}  |  Speed: {current_speed:.1f} m/s  |  "
            f"t={sim_time:.1f}s",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )

    # Create animation
    anim = FuncAnimation(fig, _draw_frame, frames=n_frames, interval=1000 // fps)

    # Try FFMpegWriter, fall back to PillowWriter (GIF)
    saved = False
    mp4_path = output_path.with_suffix(".mp4")
    gif_path = output_path.with_suffix(".gif")

    try:
        writer = FFMpegWriter(fps=fps, metadata={"title": "Drone Racing Flight"}, bitrate=2000)
        anim.save(str(mp4_path), writer=writer, dpi=100)
        print(f"  Animation saved: {mp4_path}")
        saved = True
    except Exception as e:
        print(f"  FFMpeg failed ({e}), falling back to GIF...")

    if not saved:
        try:
            writer = PillowWriter(fps=fps)
            anim.save(str(gif_path), writer=writer, dpi=80)
            print(f"  Animation saved (GIF fallback): {gif_path}")
        except Exception as e2:
            print(f"  GIF also failed: {e2}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize drone racing flight")
    parser.add_argument("--seed", type=int, default=3, help="Random seed for course (default: 3)")
    parser.add_argument("--gates", type=int, default=5, help="Number of gates (default: 5)")
    parser.add_argument("--max-speed", type=float, default=12.0, help="Max speed (default: 12)")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps (default: 2000)")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30)")
    parser.add_argument("--output-dir", type=str, default=str(SCRIPT_DIR), help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Drone Racing Flight Visualization")
    print("=" * 60)
    print(f"  Seed:      {args.seed}")
    print(f"  Gates:     {args.gates}")
    print(f"  Max speed: {args.max_speed} m/s")
    print(f"  Max steps: {args.max_steps}")
    print()

    # --- Step 1: Run episode ---
    print("[1/3] Running episode...")
    data = run_episode(
        seed=args.seed,
        num_gates=args.gates,
        max_speed=args.max_speed,
        max_steps=args.max_steps,
    )
    n_passed = data["max_gates_passed"]
    status = "COMPLETED" if data["completed"] else f"{n_passed}/{len(data['gates'])} gates"
    print(f"  Episode finished: {status} in {data['steps']} steps")
    print(f"  Max speed: {max(data['speeds']):.1f} m/s, Avg speed: {np.mean(data['speeds']):.1f} m/s")
    print()

    # --- Step 2: Static plot ---
    print("[2/3] Creating static 3D plot...")
    png_path = output_dir / "flight_visualization.png"
    create_static_plot(data, png_path)
    print()

    # --- Step 3: Animation ---
    print("[3/3] Creating animated video...")
    mp4_path = output_dir / "flight_visualization.mp4"
    create_animation(data, mp4_path, fps=args.fps)
    print()

    print("=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
