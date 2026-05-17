#!/usr/bin/env python3
"""Run hard courses (20-40 gates) and create videos with telemetry overlay.

Generates 5 challenging course types with wide curves, altitude changes,
and sharp turns. Each video shows the drone trajectory alongside real-time
roll, pitch, yaw, speed, thrust, and altitude telemetry.

Usage:
    python scripts/hard_course_videos.py [--output-dir DIR] [--fps FPS]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from aigrandprix.control.drone_controller import DroneState
from aigrandprix.main import RaceConfig, RacingAgent
from aigrandprix.planning.path_planner import Gate
from aigrandprix.simulation.gym_env import DroneRacingEnv
from aigrandprix.utils.math_utils import quat_to_euler


# ---------------------------------------------------------------------------
# Course generators
# ---------------------------------------------------------------------------

def _make_serpentine(num_gates: int = 30) -> list[np.ndarray]:
    """Wide sinusoidal course with altitude variation."""
    gates = []
    for i in range(num_gates):
        x = 5.0 + i * 8.0
        y = 7.0 * np.sin(i * 0.5)
        z = 2.5 + 1.5 * np.sin(i * 0.3)
        z = np.clip(z, 1.0, 5.0)
        gates.append(np.array([x, y, z]))
    return gates


def _make_rollercoaster(num_gates: int = 25) -> list[np.ndarray]:
    """Big altitude changes with gentle lateral curves."""
    gates = []
    for i in range(num_gates):
        x = 5.0 + i * 10.0
        y = 5.0 * np.sin(i * 0.25)
        z = 3.0 + 2.0 * np.sin(i * 0.6)
        z = np.clip(z, 1.0, 5.0)
        gates.append(np.array([x, y, z]))
    return gates


def _make_hairpin(num_gates: int = 30) -> list[np.ndarray]:
    """Sharp switchback turns — forward then backward zigzag."""
    gates = []
    segment_len = 5
    for i in range(num_gates):
        seg = i // segment_len
        pos_in_seg = i % segment_len
        if seg % 2 == 0:
            x = 5.0 + seg * 12.0 + pos_in_seg * 6.0
            y = -8.0 + pos_in_seg * 4.0
        else:
            x = 5.0 + seg * 12.0 + pos_in_seg * 6.0
            y = 8.0 - pos_in_seg * 4.0
        z = 2.5 + 1.0 * np.sin(i * 0.4)
        z = np.clip(z, 1.0, 5.0)
        gates.append(np.array([x, y, z]))
    return gates


def _make_marathon(num_gates: int = 40) -> list[np.ndarray]:
    """Long mixed course — curves, straights, altitude changes."""
    gates = []
    rng = np.random.RandomState(42)
    pos = np.array([5.0, 0.0, 2.5])
    for i in range(num_gates):
        dx = rng.uniform(5.0, 12.0)
        dy = rng.uniform(-5.0, 5.0)
        dz = rng.uniform(-1.5, 1.5)
        pos = pos + np.array([dx, dy, dz])
        pos[2] = np.clip(pos[2], 1.0, 5.0)
        gates.append(pos.copy())
    return gates


def _make_spiral(num_gates: int = 25) -> list[np.ndarray]:
    """Expanding spiral with altitude ramp."""
    gates = []
    for i in range(num_gates):
        t = i * 0.4
        r = 3.0 + i * 0.8
        x = 5.0 + i * 5.0 + r * np.cos(t) * 0.3
        y = r * np.sin(t)
        z = 1.5 + (i / num_gates) * 3.0
        z = np.clip(z, 1.0, 5.0)
        gates.append(np.array([x, y, z]))
    return gates


COURSES = {
    "serpentine_30": ("Serpentine 30 gates", _make_serpentine, 30, 8000),
    "rollercoaster_25": ("Rollercoaster 25 gates", _make_rollercoaster, 25, 7000),
    "hairpin_30": ("Hairpin Turns 30 gates", _make_hairpin, 30, 10000),
    "marathon_40": ("Marathon 40 gates", _make_marathon, 40, 12000),
    "spiral_25": ("Spiral 25 gates", _make_spiral, 25, 7000),
}


# ---------------------------------------------------------------------------
# Episode runner with full telemetry
# ---------------------------------------------------------------------------

@dataclass
class Telemetry:
    positions: np.ndarray    # (T, 3)
    velocities: np.ndarray   # (T, 3)
    rpy: np.ndarray          # (T, 3) roll, pitch, yaw in degrees
    speeds: np.ndarray       # (T,)
    thrusts: np.ndarray      # (T,)
    roll_rates: np.ndarray   # (T,)
    pitch_rates: np.ndarray  # (T,)
    yaw_rates: np.ndarray    # (T,)
    times: np.ndarray        # (T,)
    gates: list[Gate]
    gate_positions: np.ndarray  # (G, 3)
    gate_pass_steps: list[int]
    gates_passed: int
    total_gates: int
    total_steps: int
    course_name: str


def run_hard_episode(
    course_name: str,
    gate_positions: list[np.ndarray],
    max_steps: int,
    max_speed: float = 15.0,
) -> Telemetry:
    """Run an episode with custom gate positions and collect full telemetry."""
    num_gates = len(gate_positions)
    env = DroneRacingEnv(num_gates=num_gates, max_steps=max_steps)
    env.reset(seed=0)

    # Override gate positions with our custom course
    env._gates = [g.copy() for g in gate_positions]
    env._current_gate_idx = 0

    # Build Gate objects
    gates: list[Gate] = []
    for i, pos in enumerate(gate_positions):
        if i == 0:
            d = pos - np.array([0.0, 0.0, 1.0])
        else:
            d = pos - gate_positions[i - 1]
        n = np.linalg.norm(d[:2])
        normal = np.array([d[0], d[1], 0.0]) / n if n > 0.01 else np.array([1, 0, 0])
        gates.append(Gate(position=pos.copy(), normal=normal, gate_id=i))

    agent = RacingAgent(RaceConfig(max_speed=max_speed, control_dt=0.02))
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Telemetry buffers
    positions_buf = []
    velocities_buf = []
    rpy_buf = []
    speeds_buf = []
    thrusts_buf = []
    roll_rates_buf = []
    pitch_rates_buf = []
    yaw_rates_buf = []
    gate_pass_steps = []
    max_gates_passed = 0

    for step in range(max_steps):
        state = DroneState(
            position=env._drone_pos.copy(),
            velocity=env._drone_vel.copy(),
            orientation=env._drone_orient.copy(),
            angular_velocity=env._drone_angular_vel.copy(),
        )

        # Collect telemetry
        positions_buf.append(state.position.copy())
        velocities_buf.append(state.velocity.copy())
        rpy_deg = np.degrees(quat_to_euler(state.orientation))
        rpy_buf.append(rpy_deg)
        speeds_buf.append(state.speed)

        cmd = agent.compute_action(
            image=dummy_image,
            state=state,
            elapsed_time=step * 0.02,
            known_gates=gates,
        )
        thrusts_buf.append(cmd.thrust)
        roll_rates_buf.append(cmd.roll_rate)
        pitch_rates_buf.append(cmd.pitch_rate)
        yaw_rates_buf.append(cmd.yaw_rate)

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
    total_steps = step + 1
    times = np.arange(total_steps) * 0.02

    return Telemetry(
        positions=np.array(positions_buf),
        velocities=np.array(velocities_buf),
        rpy=np.array(rpy_buf),
        speeds=np.array(speeds_buf),
        thrusts=np.array(thrusts_buf),
        roll_rates=np.array(roll_rates_buf),
        pitch_rates=np.array(pitch_rates_buf),
        yaw_rates=np.array(yaw_rates_buf),
        times=times,
        gates=gates,
        gate_positions=np.array([g.copy() for g in gate_positions]),
        gate_pass_steps=gate_pass_steps,
        gates_passed=max_gates_passed,
        total_gates=num_gates,
        total_steps=total_steps,
        course_name=course_name,
    )


# ---------------------------------------------------------------------------
# Video creation with telemetry panels
# ---------------------------------------------------------------------------

def _gate_corners_3d(gate: Gate, width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """Compute 4 corners of a gate rectangle in 3D."""
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(gate.normal, up)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        right = np.array([0.0, 1.0, 0.0])
    else:
        right = right / rn
    hw, hh = width / 2.0, height / 2.0
    c = gate.position
    return np.array([
        c - right * hw - up * hh,
        c + right * hw - up * hh,
        c + right * hw + up * hh,
        c - right * hw + up * hh,
    ])


def create_telemetry_video(
    telem: Telemetry,
    output_path: Path,
    fps: int = 30,
    subsample: int = 5,
) -> None:
    """Create animated video with 3D trajectory view + telemetry panels."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plt.style.use("dark_background")

    # Subsample for animation performance
    n_total = telem.total_steps
    frame_indices = list(range(0, n_total, subsample))
    n_frames = len(frame_indices)

    # Set up figure layout:
    # Left: 3D trajectory view (large)
    # Right: 6 stacked telemetry strips
    fig = plt.figure(figsize=(20, 11))
    gs = gridspec.GridSpec(
        6, 2,
        width_ratios=[1.5, 1],
        hspace=0.35,
        wspace=0.25,
        left=0.04, right=0.96, top=0.93, bottom=0.05,
    )

    ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_roll = fig.add_subplot(gs[0, 1])
    ax_pitch = fig.add_subplot(gs[1, 1])
    ax_yaw = fig.add_subplot(gs[2, 1])
    ax_speed = fig.add_subplot(gs[3, 1])
    ax_thrust = fig.add_subplot(gs[4, 1])
    ax_alt = fig.add_subplot(gs[5, 1])

    telem_axes = [ax_roll, ax_pitch, ax_yaw, ax_speed, ax_thrust, ax_alt]

    # Precompute 3D axis limits
    all_pts = np.vstack([telem.positions] + [g.reshape(1, 3) for g in telem.gate_positions])
    ranges = all_pts.max(axis=0) - all_pts.min(axis=0)
    max_range = max(ranges.max(), 1.0) / 2.0
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2.0
    xlim = (mid[0] - max_range, mid[0] + max_range)
    ylim = (mid[1] - max_range, mid[1] + max_range)
    zlim = (max(mid[2] - max_range, -0.5), mid[2] + max_range)

    # Precompute gate corners
    gate_corners_list = [_gate_corners_3d(g, 1.0, 1.0) for g in telem.gates]

    # Precompute telemetry y-limits
    roll_lim = (min(-15, telem.rpy[:, 0].min() - 2), max(15, telem.rpy[:, 0].max() + 2))
    pitch_lim = (min(-15, telem.rpy[:, 1].min() - 2), max(15, telem.rpy[:, 1].max() + 2))
    yaw_lim = (telem.rpy[:, 2].min() - 5, telem.rpy[:, 2].max() + 5)
    speed_lim = (0, max(telem.speeds.max() + 1, 8))
    thrust_lim = (0, 1.05)
    alt_lim = (0, max(telem.positions[:, 2].max() + 1, 6))

    max_speed_val = max(telem.speeds.max(), 1.0)
    telem_window = 15.0  # seconds of telemetry visible

    def _draw_frame(frame_num: int) -> None:
        step_idx = frame_indices[frame_num]
        cur_time = step_idx * 0.02

        # --- 3D trajectory view ---
        ax_3d.cla()
        ax_3d.set_xlim(*xlim)
        ax_3d.set_ylim(*ylim)
        ax_3d.set_zlim(*zlim)
        ax_3d.set_xlabel("X (m)", fontsize=9, labelpad=6)
        ax_3d.set_ylabel("Y (m)", fontsize=9, labelpad=6)
        ax_3d.set_zlabel("Z (m)", fontsize=9, labelpad=6)
        ax_3d.tick_params(labelsize=7)

        # Rotating camera that follows the drone loosely
        base_azim = -60
        azim = base_azim + frame_num * 0.12
        elev = 25 + 8 * np.sin(frame_num * 0.015)
        ax_3d.view_init(elev=elev, azim=azim)

        # Determine gates passed
        gates_passed_now = 0
        for gf in telem.gate_pass_steps:
            if step_idx >= gf:
                gates_passed_now += 1

        # Draw gate rectangles
        for i, (gate, corners) in enumerate(zip(telem.gates, gate_corners_list)):
            if i < gates_passed_now:
                color, alpha = "#00ff88", 0.5
            else:
                color, alpha = "#ffaa00", 0.35
            poly = Poly3DCollection(
                [corners.tolist()], alpha=alpha,
                facecolor=color, edgecolor="white", linewidths=0.8,
            )
            ax_3d.add_collection3d(poly)
            ax_3d.text(
                gate.position[0], gate.position[1], gate.position[2] + 0.7,
                f"{i}", color="white" if i >= gates_passed_now else "#00ff88",
                fontsize=7, fontweight="bold", ha="center", va="bottom",
            )

        # Draw trajectory trail colored by speed
        trail = telem.positions[:step_idx + 1]
        trail_speeds = telem.speeds[:step_idx + 1]
        if len(trail) > 1:
            ax_3d.scatter(
                trail[:, 0], trail[:, 1], trail[:, 2],
                c=trail_speeds, cmap="plasma", s=1.2, alpha=0.6,
                vmin=0, vmax=max_speed_val,
            )

        # Drone marker
        pos = telem.positions[step_idx]
        ax_3d.scatter(
            [pos[0]], [pos[1]], [pos[2]],
            color="#00ccff", s=100, marker="o",
            edgecolors="white", linewidths=1.5, zorder=10,
        )

        # Start marker
        ax_3d.scatter(
            [telem.positions[0, 0]], [telem.positions[0, 1]], [telem.positions[0, 2]],
            color="#00ff88", s=50, marker="^",
            edgecolors="white", linewidths=1.0, zorder=5,
        )

        status_str = f"Gates: {gates_passed_now}/{telem.total_gates}"
        speed_str = f"Speed: {telem.speeds[step_idx]:.1f} m/s"
        alt_str = f"Alt: {pos[2]:.1f} m"
        roll_str = f"R:{telem.rpy[step_idx, 0]:.0f}"
        pitch_str = f"P:{telem.rpy[step_idx, 1]:.0f}"
        ax_3d.set_title(
            f"{telem.course_name}  |  {status_str}  |  {speed_str}  |  "
            f"{alt_str}  |  {roll_str} {pitch_str}  |  t={cur_time:.1f}s",
            fontsize=10, fontweight="bold", pad=12,
        )

        # --- Telemetry panels ---
        t_start = max(0, cur_time - telem_window)
        t_end = cur_time + 1.0
        mask = (telem.times >= t_start) & (telem.times <= t_end)
        t_vis = telem.times[mask]

        panels = [
            (ax_roll, telem.rpy[mask, 0], "Roll", "deg", roll_lim, "#ff6b6b"),
            (ax_pitch, telem.rpy[mask, 1], "Pitch", "deg", pitch_lim, "#4ecdc4"),
            (ax_yaw, telem.rpy[mask, 2], "Yaw", "deg", yaw_lim, "#ffe66d"),
            (ax_speed, telem.speeds[mask], "Speed", "m/s", speed_lim, "#a29bfe"),
            (ax_thrust, telem.thrusts[mask], "Thrust", "", thrust_lim, "#fd79a8"),
            (ax_alt, telem.positions[mask, 2], "Altitude", "m", alt_lim, "#00b894"),
        ]

        for ax, data, label, unit, ylim_t, color in panels:
            ax.cla()
            ax.set_xlim(t_start, t_end)
            ax.set_ylim(*ylim_t)
            if len(t_vis) > 0:
                ax.plot(t_vis, data, color=color, linewidth=1.2, alpha=0.9)
            ax.axvline(cur_time, color="white", linewidth=0.5, alpha=0.5)

            cur_val = 0
            if label == "Roll":
                cur_val = telem.rpy[step_idx, 0]
            elif label == "Pitch":
                cur_val = telem.rpy[step_idx, 1]
            elif label == "Yaw":
                cur_val = telem.rpy[step_idx, 2]
            elif label == "Speed":
                cur_val = telem.speeds[step_idx]
            elif label == "Thrust":
                cur_val = telem.thrusts[step_idx]
            elif label == "Altitude":
                cur_val = telem.positions[step_idx, 2]

            unit_str = f" {unit}" if unit else ""
            ax.set_ylabel(
                f"{label}\n{cur_val:.1f}{unit_str}", fontsize=8,
                fontweight="bold", color=color, rotation=0, labelpad=45, va="center",
            )
            ax.tick_params(labelsize=7)
            if ax is not ax_alt:
                ax.set_xticklabels([])

        ax_alt.set_xlabel("Time (s)", fontsize=8)

        # Mark gate passes on telemetry panels
        for gstep in telem.gate_pass_steps:
            gt = gstep * 0.02
            if t_start <= gt <= t_end:
                for ax_t in telem_axes:
                    ax_t.axvline(gt, color="#00ff88", linewidth=0.5, alpha=0.3)

    print(f"    Rendering {n_frames} frames...")
    anim = FuncAnimation(fig, _draw_frame, frames=n_frames, interval=1000 // fps)

    # Save video
    saved = False
    mp4_path = output_path.with_suffix(".mp4")
    gif_path = output_path.with_suffix(".gif")

    try:
        writer = FFMpegWriter(fps=fps, metadata={"title": telem.course_name}, bitrate=3000)
        anim.save(str(mp4_path), writer=writer, dpi=100)
        print(f"    Video saved: {mp4_path}")
        saved = True
    except Exception as e:
        print(f"    FFMpeg failed ({e}), trying GIF...")

    if not saved:
        try:
            writer = PillowWriter(fps=min(fps, 15))
            anim.save(str(gif_path), writer=writer, dpi=80)
            print(f"    GIF saved: {gif_path}")
        except Exception as e2:
            print(f"    GIF also failed: {e2}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Hard course telemetry videos")
    parser.add_argument("--output-dir", type=str, default="scripts/hard_courses",
                        help="Output directory for videos")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--max-speed", type=float, default=15.0, help="Max speed")
    parser.add_argument("--courses", nargs="*", default=list(COURSES.keys()),
                        help="Which courses to run")
    parser.add_argument("--subsample", type=int, default=5,
                        help="Subsample factor for animation (higher=faster render)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Hard Course Telemetry Videos")
    print("=" * 70)

    for course_key in args.courses:
        if course_key not in COURSES:
            print(f"  Unknown course: {course_key}")
            print(f"  Available: {list(COURSES.keys())}")
            sys.exit(1)

        name, gen_fn, num_gates, max_steps = COURSES[course_key]
        print(f"\n--- {name} ---")

        # Generate course
        print(f"  Generating {num_gates} gates...")
        gate_positions = gen_fn(num_gates)

        # Run episode
        print(f"  Running episode (max {max_steps} steps)...")
        t0 = time.time()
        telem = run_hard_episode(
            course_name=name,
            gate_positions=gate_positions,
            max_steps=max_steps,
            max_speed=args.max_speed,
        )
        elapsed = time.time() - t0
        print(f"  Episode: {telem.gates_passed}/{telem.total_gates} gates in "
              f"{telem.total_steps} steps ({telem.total_steps * 0.02:.1f}s sim), "
              f"wall time: {elapsed:.1f}s")
        print(f"  Avg speed: {telem.speeds.mean():.2f} m/s, "
              f"Max speed: {telem.speeds.max():.2f} m/s, "
              f"Avg alt: {telem.positions[:, 2].mean():.2f} m")
        print(f"  Roll range: [{telem.rpy[:, 0].min():.1f}, {telem.rpy[:, 0].max():.1f}] deg")
        print(f"  Pitch range: [{telem.rpy[:, 1].min():.1f}, {telem.rpy[:, 1].max():.1f}] deg")

        # Create video
        print(f"  Creating video...")
        t0 = time.time()
        video_path = output_dir / f"{course_key}"
        create_telemetry_video(
            telem, video_path, fps=args.fps, subsample=args.subsample,
        )
        print(f"  Video render time: {time.time() - t0:.1f}s")

    print("\n" + "=" * 70)
    print(f"  All videos saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
