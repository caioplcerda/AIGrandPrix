"""Extreme course battery — maximum difficulty, maximum speed.

Tests: tight spirals, multi-level circuits, 12m+ gate circuits, hairpin turns.
All at max_speed=25 m/s for consistency.

Usage:
    python scripts/test_extreme_courses.py
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("extreme")

_MAX_SPEED = 25.0
_LOOP_HZ = 50.0
_MAX_DUR = 150.0
_RP = 0.25
_IH = 0.75


def _norm(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=float)
    return a / np.linalg.norm(a)


def _gate_crossed(gp, gn, op, np_, ih=_IH):
    n, c = gn, gp
    d1 = float(np.dot(op - c, n)); d2 = float(np.dot(np_ - c, n))
    if d1 * d2 > 0 or abs(d1 - d2) < 1e-9: return False
    t = d1 / (d1 - d2); x = op + t * (np_ - op); d = x - c
    up = np.array([0., 0., -1.]); r = np.cross(n, up); rn = float(np.linalg.norm(r))
    r = r / rn if rn > 1e-6 else np.array([0., 1., 0.])
    return abs(float(np.dot(d, r))) <= ih and abs(float(np.dot(d, up))) <= ih


EXTREME_COURSES = [
    # 8-gate figure-8 circuit
    (
        "figure_8",
        [
            (np.array([15.,  0., -2.5]), _norm([1., 0., 0.])),
            (np.array([25.,  8., -2.5]), _norm([0.707, 0.707, 0.])),
            (np.array([35.,  0., -2.5]), _norm([0.707, -0.707, 0.])),
            (np.array([25., -8., -2.5]), _norm([-0.707, -0.707, 0.])),
            (np.array([15.,  0., -2.5]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 5.,  8., -2.5]), _norm([-0.707, 0.707, 0.])),
            (np.array([-5.,  0., -2.5]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 5., -8., -2.5]), _norm([0.707, -0.707, 0.])),
        ],
        14600, 15800,
    ),
    # 10-gate circuit with altitude swings ±3m and diagonal gates
    (
        "altitude_slalom_10",
        [
            (np.array([ 10.,  0., -1.5]), _norm([1., 0., 0.])),
            (np.array([ 20.,  5., -4.0]), _norm([1., 0., 0.])),
            (np.array([ 30.,  0., -1.5]), _norm([1., 0., 0.])),
            (np.array([ 40., -5., -4.5]), _norm([1., 0., 0.])),
            (np.array([ 50.,  0., -2.0]), _norm([1., 0., 0.])),
            (np.array([ 60.,  5., -4.0]), _norm([1., 0., 0.])),
            (np.array([ 70.,  0., -1.5]), _norm([1., 0., 0.])),
            (np.array([ 80., -5., -4.5]), _norm([1., 0., 0.])),
            (np.array([ 90.,  0., -2.0]), _norm([1., 0., 0.])),
            (np.array([100.,  0., -2.5]), _norm([1., 0., 0.])),
        ],
        14601, 15801,
    ),
    # 12-gate omega circuit: long straight + hairpin + return
    (
        "omega_12gate",
        [
            (np.array([ 10.,  0., -2.5]), _norm([1., 0., 0.])),
            (np.array([ 25.,  0., -2.5]), _norm([1., 0., 0.])),
            (np.array([ 40.,  0., -2.5]), _norm([1., 0., 0.])),
            (np.array([ 55.,  0., -2.5]), _norm([1., 0., 0.])),
            (np.array([ 65.,  6., -2.5]), _norm([0.707, 0.707, 0.])),
            (np.array([ 65., 18., -2.5]), _norm([0., 1., 0.])),
            (np.array([ 55., 24., -2.5]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 40., 24., -2.5]), _norm([-1., 0., 0.])),
            (np.array([ 25., 24., -2.5]), _norm([-1., 0., 0.])),
            (np.array([ 10., 24., -2.5]), _norm([-1., 0., 0.])),
            (np.array([  5., 18., -2.5]), _norm([-0.707, -0.707, 0.])),
            (np.array([  5.,  6., -2.5]), _norm([0., -1., 0.])),
        ],
        14602, 15802,
    ),
    # Corkscrew: 8-gate vertical helix, +5m altitude, tight 8m spacing
    (
        "corkscrew_helix_8",
        [
            (np.array([ 8.,  0., -2.0]), _norm([1., 0., 0.])),
            (np.array([14.,  5., -3.5]), _norm([0.707, 0.707, 0.])),
            (np.array([ 8., 10., -5.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 2.,  8., -6.5]), _norm([-1., 0., 0.])),
            (np.array([ 4.,  2., -7.5]), _norm([0.707, -0.707, 0.])),
            (np.array([12.,  0., -8.5]), _norm([1., 0., 0.])),
            (np.array([18.,  6., -9.5]), _norm([0.707, 0.707, 0.])),
            (np.array([12., 12.,-10.5]), _norm([-0.707, 0.707, 0.])),
        ],
        14603, 15803,
    ),
    # Speed test: 15 gates, 15m spacing, straight — max speed benchmark
    (
        "speed_benchmark_15gate",
        [(np.array([float(i * 15), 0., -2.5]), np.array([1., 0., 0.])) for i in range(1, 16)],
        14604, 15804,
    ),
    # Ultra-tight: 6m spacing, 3D, diagonals
    (
        "ultra_tight_6m",
        [
            (np.array([ 6.,  0., -2.0]), _norm([1., 0., 0.])),
            (np.array([10.,  4., -3.5]), _norm([0.707, 0.707, 0.])),
            (np.array([ 6.,  8., -5.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 2.,  6., -6.5]), _norm([-1., 0., 0.])),
            (np.array([ 4.,  2., -5.0]), _norm([0.707, -0.707, 0.])),
            (np.array([10.,  0., -3.5]), _norm([1., 0., 0.])),
        ],
        14605, 15805,
    ),
    # Competition replica: realistic VQ1 layout, mixed everything
    (
        "vq1_replica_realistic",
        [
            (np.array([15.,   0., -2.5]), _norm([1., 0., 0.])),
            (np.array([30.,   5., -3.5]), _norm([1., 0., 0.])),
            (np.array([45.,  -5., -1.5]), _norm([0.707, -0.707, 0.])),
            (np.array([55., -18., -2.5]), _norm([0., -1., 0.])),
            (np.array([50., -30., -4.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([35., -35., -2.5]), _norm([-1., 0., 0.])),
            (np.array([20., -28., -1.5]), _norm([-0.707, 0.707, 0.])),
            (np.array([10., -18., -3.5]), _norm([0., 1., 0.])),
            (np.array([15.,  -5., -2.5]), _norm([1., 0., 0.])),
        ],
        14606, 15806,
    ),
    # 20-gate grand circuit — VQ2 scale
    (
        "grand_circuit_20gate",
        [
            (np.array([ 15.,   0., -2.5]), _norm([1., 0., 0.])),
            (np.array([ 30.,   0., -3.5]), _norm([1., 0., 0.])),
            (np.array([ 45.,   5., -2.0]), _norm([1., 0., 0.])),
            (np.array([ 60.,  10., -3.5]), _norm([0.707, 0.707, 0.])),
            (np.array([ 65.,  22., -2.5]), _norm([0., 1., 0.])),
            (np.array([ 55.,  32., -3.5]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 40.,  35., -2.0]), _norm([-1., 0., 0.])),
            (np.array([ 25.,  32., -3.5]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 15.,  22., -2.5]), _norm([0., -1., 0.])),
            (np.array([ 10.,  10., -3.5]), _norm([-0.707, -0.707, 0.])),
            (np.array([  5.,   0., -2.0]), _norm([-1., 0., 0.])),
            (np.array([ -5.,  -8., -3.5]), _norm([-0.707, -0.707, 0.])),
            (np.array([-10., -18., -2.5]), _norm([0., -1., 0.])),
            (np.array([  0., -28., -3.5]), _norm([0.707, -0.707, 0.])),
            (np.array([ 15., -32., -2.0]), _norm([1., 0., 0.])),
            (np.array([ 30., -28., -3.5]), _norm([0.707, 0.707, 0.])),
            (np.array([ 40., -18., -2.5]), _norm([0., 1., 0.])),
            (np.array([ 45.,  -8., -3.5]), _norm([0.707, 0.707, 0.])),
            (np.array([ 50.,   0., -2.0]), _norm([1., 0., 0.])),
            (np.array([ 15.,   0., -2.5]), _norm([1., 0., 0.])),
        ],
        14607, 15807,
    ),
    # Maximum speed straight: 20 gates, 20m spacing — absolute top-speed benchmark
    (
        "max_speed_20gate_20m",
        [(np.array([float(i * 20), 0., -2.5]), np.array([1., 0., 0.])) for i in range(1, 21)],
        14608, 15808,
    ),
    # Hairpin gauntlet: 4 back-to-back 180° U-turns
    (
        "hairpin_gauntlet",
        [
            (np.array([10.,  0., -2.5]), _norm([1., 0., 0.])),
            (np.array([25.,  0., -2.5]), _norm([1., 0., 0.])),
            (np.array([35.,  5., -2.5]), _norm([0.707, 0.707, 0.])),
            (np.array([35., 15., -2.5]), _norm([0., 1., 0.])),
            (np.array([25., 20., -2.5]), _norm([-0.707, 0.707, 0.])),
            (np.array([10., 20., -2.5]), _norm([-1., 0., 0.])),
            (np.array([ 0., 15., -2.5]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 0.,  5., -2.5]), _norm([0., -1., 0.])),
            (np.array([10.,  0., -2.5]), _norm([0.707, -0.707, 0.])),
            (np.array([25.,  0., -2.5]), _norm([1., 0., 0.])),
        ],
        14609, 15809,
    ),
]


def run_course(name, gate_defs, mp, vp):
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
        return {"course": name, "passed": False, "error": "no heartbeat", "gates_passed": 0, "total_gates": ng, "total_time_s": 0.}

    est = NEDStateEstimator()
    pl = PathPlannerNED(max_speed=_MAX_SPEED, approach_distance=6., exit_distance=1.)

    dt = 1. / _LOOP_HZ; ni = 0; wps = []; lrp = -_RP; prev = None; ts = time.time()
    while True:
        tl = time.perf_counter(); el = time.time() - ts
        if el > _MAX_DUR or ev.is_set(): break
        if ni >= ng:
            lg = gn[-1]; ep = lg.position + lg.normal * 6.
            client.send_position_target(PositionTargetNED(
                x=float(ep[0]), y=float(ep[1]), z=float(ep[2]),
                vx=float(lg.normal[0] * _MAX_SPEED), vy=float(lg.normal[1] * _MAX_SPEED), vz=0., yaw=0.))
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
        rem = gn[ni:]
        if rem and (not wps or el - lrp > _RP):
            wps = pl.plan(rem[:1], st.pos_ned, st.vel_ned); lrp = el
        wp = pl.next_position_target(wps, st.pos_ned, lookahead_m=5.) if wps else None
        if wp is None: wp = WaypointNED(pos=st.pos_ned.copy(), vel=np.zeros(3), yaw=st.yaw, time=el)
        client.send_position_target(PositionTargetNED(
            x=float(wp.pos[0]), y=float(wp.pos[1]), z=float(wp.pos[2]),
            vx=float(wp.vel[0]), vy=float(wp.vel[1]), vz=float(wp.vel[2]), yaw=float(wp.yaw)))
        if (s := dt - (time.perf_counter() - tl)) > 0: time.sleep(s)

    client.stop()
    gp = res.get("gates_passed", 0); tt = res.get("total_time_s", 0.)
    return {
        "course": name, "gates_passed": gp, "total_gates": ng, "total_time_s": tt,
        "completed": res.get("completed", False), "passed": gp >= ng,
        "error": err[0][:200] if err else None,
    }


def main() -> int:
    print(f"\nEXTREME COURSE BATTERY — max_speed={_MAX_SPEED} m/s")
    print("=" * 65)
    results = []
    for name, defs, mp, vp in EXTREME_COURSES:
        print(f"  {name} ({len(defs)} gates)...", flush=True, end=" ")
        r = run_course(name, defs, mp, vp)
        results.append(r)
        if r.get("error"):
            print(f"ERROR: {r['error'][:100]}")
        else:
            st = "PASS ✓" if r["passed"] else "FAIL ✗"
            print(f"{st}  {r['gates_passed']}/{r['total_gates']}  {r['total_time_s']:.2f}s")
        time.sleep(1.)

    print("=" * 65)
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    print(f"TOTAL: {passed}/{total} courses passed")
    print("=" * 65)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
