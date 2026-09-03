#!/usr/bin/env python3
"""Render a flight video from the CURRENT MAVLink/NED stack.

The older `hard_course_videos.py` renders from `simulation.gym_env` — the
legacy stack that was replaced. Its videos top out around 5 m/s and do not
represent what the current code does. This script drives the same loop the
course batteries use (mock server → MAVLink client → NED estimator →
planner → controller), records the trajectory, and renders it.

Usage:
    python scripts/render_run_video.py --course titan_triple_helix_24
    python scripts/render_run_video.py --course titan_chaos_30 --fps 30
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter, FuncAnimation

from test_titan_courses import (  # noqa: E402
    TITAN_COURSES, _MAX_SPEED, _LOOP_HZ, _MAX_DUR, _RP, _ADIST, _gate_crossed,
)


def run_and_record(name, gate_defs, mp, vp):
    """Run one course and record per-sample state. Mirrors test_titan_courses.run_course."""
    from aigrandprix.mock_sim.dcl_mock_server import DCLMockServer, MockGate
    from aigrandprix.comms.mavlink_client import MAVLinkClient, PositionTargetNED
    from aigrandprix.state.state_estimator import NEDStateEstimator
    from aigrandprix.planning.path_planner_ned import GateNED, PathPlannerNED, WaypointNED

    ng = len(gate_defs)
    gm = [MockGate(center_ned=p.copy(), normal_ned=n.copy(), gate_id=i) for i, (p, n) in enumerate(gate_defs)]
    gn = [GateNED(position=p.copy(), normal=n.copy(), gate_id=i) for i, (p, n) in enumerate(gate_defs)]
    res = {}; ev = threading.Event(); err = []

    def srv():
        try:
            s = DCLMockServer(gates=gm, mavlink_port=mp, vision_port=vp, host="127.0.0.1")
            res.update(s.run(max_duration_s=_MAX_DUR))
        except Exception:
            import traceback; err.append(traceback.format_exc())
        finally:
            ev.set()

    threading.Thread(target=srv, daemon=True).start()
    time.sleep(0.2)

    client = MAVLinkClient(host="127.0.0.1", mavlink_port=mp, heartbeat_rate_hz=5.)
    client.connect(); client.start()
    t0 = time.time()
    while not client.connected and time.time() - t0 < 15.:
        time.sleep(0.05)
    if not client.connected:
        client.stop()
        raise RuntimeError("no heartbeat from mock server")

    est = NEDStateEstimator()
    pl = PathPlannerNED(max_speed=_MAX_SPEED, approach_distance=_ADIST, exit_distance=1.)

    rec = {k: [] for k in ("t", "x", "y", "z", "speed", "roll", "pitch", "yaw", "gates")}
    dt = 1. / _LOOP_HZ; ni = 0; wps = []; lrp = -_RP; prev = None; ts = time.time()
    while True:
        tl = time.perf_counter(); el = time.time() - ts
        if el > _MAX_DUR:
            ev.wait(timeout=10.); break
        if ev.is_set(): break
        if ni >= ng:
            # Keep flying through to the exit point. Breaking here would end the
            # run before the server registers the final gate crossing.
            lg = gn[-1]; ep = lg.position + lg.normal * 6.
            client.send_position_target(PositionTargetNED(
                x=float(ep[0]), y=float(ep[1]), z=float(ep[2]),
                vx=float(lg.normal[0] * _MAX_SPEED), vy=float(lg.normal[1] * _MAX_SPEED),
                vz=0., yaw=0.))
            if (s := dt - (time.perf_counter() - tl)) > 0: time.sleep(s)
            continue
        tm = client.telemetry
        if not tm.connected: time.sleep(0.05); continue
        st = est.update(tm); cp = st.pos_ned
        if prev is not None and ni < ng:
            g = gn[ni]
            if _gate_crossed(g.position, g.normal, prev, cp):
                est.reset_position(g.position); ni += 1; wps = []
        prev = cp.copy()

        rec["t"].append(el)
        rec["x"].append(float(cp[0])); rec["y"].append(float(cp[1])); rec["z"].append(float(cp[2]))
        rec["speed"].append(float(np.linalg.norm(st.vel_ned)))
        rec["roll"].append(float(np.degrees(st.roll)))
        rec["pitch"].append(float(np.degrees(st.pitch)))
        rec["yaw"].append(float(np.degrees(st.yaw)))
        rec["gates"].append(ni)

        rem = gn[ni:]
        if rem and (not wps or el - lrp > _RP):
            wps = pl.plan(rem[:1], st.pos_ned, st.vel_ned); lrp = el
        wp = pl.next_position_target(wps, st.pos_ned, lookahead_m=5.) if wps else None
        if wp is None:
            wp = WaypointNED(pos=st.pos_ned.copy(), vel=np.zeros(3), yaw=st.yaw, time=el)
        if rem:
            apt = rem[0].position - rem[0].normal * _ADIST
            av = apt - st.pos_ned
            da = float(np.linalg.norm(av))
            ah = float(np.dot(av, rem[0].normal)) > 0.0
            if da > 0.1 and ah:
                bl = min(1.0, max(0.0, (da - 3.0) / 8.0))
                cv = bl * (av / da * _MAX_SPEED) + (1.0 - bl) * wp.vel
            else:
                cv = wp.vel
        else:
            cv = wp.vel
        client.send_position_target(PositionTargetNED(
            x=float(wp.pos[0]), y=float(wp.pos[1]), z=float(wp.pos[2]),
            vx=float(cv[0]), vy=float(cv[1]), vz=float(cv[2]), yaw=float(wp.yaw)))
        if (s := dt - (time.perf_counter() - tl)) > 0: time.sleep(s)

    client.stop()
    if err:
        raise RuntimeError(err[0][:300])
    rec = {k: np.asarray(v) for k, v in rec.items()}
    return rec, gn, int(res.get("gates_passed", ni)), ng


def _gate_patch(g, half=1.35):
    """Four corners of a gate square, perpendicular to its normal."""
    n = g.normal / (np.linalg.norm(g.normal) + 1e-9)
    up = np.array([0., 0., -1.])
    right = np.cross(up, n)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0., 1., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(n, right); up /= np.linalg.norm(up)
    c = g.position
    return np.array([c + right * half + up * half, c - right * half + up * half,
                     c - right * half - up * half, c + right * half - up * half])


def render(rec, gates, name, passed, total, out_path, fps=30, stride=2):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    idx = np.arange(0, len(rec["t"]), stride)
    n_frames = len(idx)
    x, y, z = rec["x"], rec["y"], -rec["z"]          # plot altitude up
    t, sp = rec["t"], rec["speed"]

    plt.rcParams.update({"font.size": 8, "text.color": "white",
                         "axes.labelcolor": "white", "xtick.color": "white",
                         "ytick.color": "white", "figure.facecolor": "black",
                         "axes.facecolor": "black", "savefig.facecolor": "black"})

    fig = plt.figure(figsize=(12, 6.75), dpi=100)
    gs = gridspec.GridSpec(4, 2, width_ratios=[1.55, 1], hspace=0.55, wspace=0.16,
                           left=0.03, right=0.97, top=0.90, bottom=0.07)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_sp = fig.add_subplot(gs[0, 1])
    ax_al = fig.add_subplot(gs[1, 1])
    ax_rp = fig.add_subplot(gs[2, 1])
    ax_yw = fig.add_subplot(gs[3, 1])

    ax3d.set_facecolor("black")
    ax3d.xaxis.set_pane_color((0.08, 0.08, 0.08, 1.0))
    ax3d.yaxis.set_pane_color((0.08, 0.08, 0.08, 1.0))
    ax3d.zaxis.set_pane_color((0.08, 0.08, 0.08, 1.0))
    ax3d.grid(True, color="0.25", linewidth=0.4)

    polys = [_gate_patch(g) for g in gates]
    gp = Poly3DCollection([np.column_stack([p[:, 0], p[:, 1], -p[:, 2]]) for p in polys],
                          facecolor="#00e5a0", edgecolor="#00ffc8", alpha=0.35, linewidths=0.7)
    ax3d.add_collection3d(gp)

    ax3d.set_xlim(x.min() - 8, x.max() + 8)
    ax3d.set_ylim(y.min() - 8, y.max() + 8)
    ax3d.set_zlim(min(0, z.min() - 3), z.max() + 6)
    ax3d.set_xlabel("North (m)"); ax3d.set_ylabel("East (m)"); ax3d.set_zlabel("Altitude (m)")

    (path_ln,) = ax3d.plot([], [], [], color="#ff8c1a", linewidth=1.9)
    (head_pt,) = ax3d.plot([], [], [], "o", color="white", markersize=5)

    panels = [
        (ax_sp, sp, "Speed (m/s)", "#7aa2ff"),
        (ax_al, z, "Altitude (m)", "#00e5a0"),
        (ax_rp, rec["roll"], "Roll (deg)", "#ff6b6b"),
        (ax_yw, rec["yaw"], "Yaw (deg)", "#ffd166"),
    ]
    lines = []
    for ax, series, label, color in panels:
        ax.set_xlim(t[0], t[-1])
        pad = 0.1 * (series.max() - series.min() + 1e-6)
        ax.set_ylim(series.min() - pad, series.max() + pad)
        ax.set_ylabel(label, fontsize=7.5, color=color)
        ax.grid(True, color="0.22", linewidth=0.4)
        for s in ax.spines.values(): s.set_color("0.35")
        ax.tick_params(labelsize=6.5)
        (ln,) = ax.plot([], [], color=color, linewidth=1.2)
        lines.append(ln)
    ax_yw.set_xlabel("Time (s)", fontsize=7.5)

    title = fig.text(0.5, 0.955, "", ha="center", fontsize=13, color="white", family="monospace")
    sub = fig.text(0.5, 0.925, "", ha="center", fontsize=8.5, color="0.65", family="monospace")

    def draw(fi):
        k = idx[fi]
        path_ln.set_data(x[:k + 1], y[:k + 1]); path_ln.set_3d_properties(z[:k + 1])
        head_pt.set_data([x[k]], [y[k]]); head_pt.set_3d_properties([z[k]])
        ax3d.view_init(elev=22, azim=-60 + 40 * (k / max(1, len(t) - 1)))
        for (ax, series, _, _), ln in zip(panels, lines):
            ln.set_data(t[:k + 1], series[:k + 1])
        title.set_text(f"{name}   |   gate {rec['gates'][k]}/{total}   |   {sp[k]:5.1f} m/s   |   t = {t[k]:5.1f} s")
        sub.set_text(f"max_speed {_MAX_SPEED:.0f} m/s   ·   MAVLink2/NED stack against mock_sim   ·   "
                     f"final {passed}/{total} gates")
        return [path_ln, head_pt, *lines]

    ani = FuncAnimation(fig, draw, frames=n_frames, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=1800,
                          metadata={"title": name, "artist": "aigrandprix"})
    ani.save(str(out_path), writer=writer)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", default="titan_triple_helix_24")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=3,
                    help="record every Nth sample; 2 gives ~2x playback at 50 Hz/30 fps")
    a = ap.parse_args()

    match = [c for c in TITAN_COURSES if c[0] == a.course]
    if not match:
        print(f"unknown course: {a.course}")
        print("available:", ", ".join(c[0] for c in TITAN_COURSES))
        return 2
    name, defs, mp, vp = match[0]
    defs_list = list(defs)

    print(f"running {name} ({len(defs_list)} gates) at {_MAX_SPEED} m/s ...", flush=True)
    rec, gates, passed, total = run_and_record(name, defs_list, mp + 700, vp + 700)
    print(f"  {passed}/{total} gates, {rec['t'][-1]:.1f}s, peak {rec['speed'].max():.1f} m/s, "
          f"{len(rec['t'])} samples", flush=True)

    out = Path(a.out) if a.out else Path(__file__).parent / "runs" / f"{name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"rendering -> {out}", flush=True)
    render(rec, gates, name, passed, total, out, fps=a.fps, stride=a.stride)
    print("done:", out, f"({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
