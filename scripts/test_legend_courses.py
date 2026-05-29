"""Legend course battery — physics-limit stress test.

Extreme conditions: max_speed=30 m/s, tight 6-8m gate spacing,
rapid direction reversals, deep 3D, very long 40+ gate courses.

Usage:
    python scripts/test_legend_courses.py
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
logger = logging.getLogger("legend")

_MAX_SPEED = 30.0
_LOOP_HZ = 50.0
_MAX_DUR = 240.0
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


LEGEND_COURSES = [
    # 1. Marathon 40-gate circuit — VQ2 double-scale, 30 m/s
    (
        "marathon_40gate",
        [
            (np.array([ 20.,   0., -2.5]), _norm([1., 0., 0.])),
            (np.array([ 40.,   5., -3.5]), _norm([1., 0., 0.])),
            (np.array([ 60.,  -5., -2.0]), _norm([1., 0., 0.])),
            (np.array([ 80.,   0., -4.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 95.,  15., -2.5]), _norm([0., 1., 0.])),
            (np.array([ 85.,  30., -4.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 70.,  40., -2.0]), _norm([-1., 0., 0.])),
            (np.array([ 50.,  38., -4.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 35.,  25., -2.5]), _norm([0., -1., 0.])),
            (np.array([ 25.,  10., -4.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 10.,   0., -2.0]), _norm([-1., 0., 0.])),
            (np.array([  0., -15., -4.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([-10., -30., -2.5]), _norm([0., -1., 0.])),
            (np.array([  0., -45., -4.0]), _norm([0.707, -0.707, 0.])),
            (np.array([ 20., -50., -2.0]), _norm([1., 0., 0.])),
            (np.array([ 40., -45., -4.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 55., -32., -2.5]), _norm([0., 1., 0.])),
            (np.array([ 65., -18., -4.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 80.,  -5., -2.0]), _norm([1., 0., 0.])),
            (np.array([ 95., -15., -4.0]), _norm([0.707, -0.707, 0.])),
            (np.array([105., -30., -2.5]), _norm([0., -1., 0.])),
            (np.array([ 95., -45., -4.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 80., -50., -2.0]), _norm([-1., 0., 0.])),
            (np.array([ 60., -45., -4.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 50., -30., -2.5]), _norm([0., 1., 0.])),
            (np.array([ 55., -15., -4.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 65.,  -5., -2.0]), _norm([1., 0., 0.])),
            (np.array([ 80.,   5., -4.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 90.,  18., -2.5]), _norm([0., 1., 0.])),
            (np.array([ 80.,  32., -4.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 65.,  38., -2.0]), _norm([-1., 0., 0.])),
            (np.array([ 45.,  35., -4.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 30.,  22., -2.5]), _norm([0., -1., 0.])),
            (np.array([ 25.,   8., -4.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 15.,  -2., -2.0]), _norm([-1., 0., 0.])),
            (np.array([  5., -12., -4.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([  5., -25., -2.5]), _norm([0., -1., 0.])),
            (np.array([ 15., -35., -4.0]), _norm([0.707, -0.707, 0.])),
            (np.array([ 30., -38., -2.0]), _norm([1., 0., 0.])),
            (np.array([ 50., -32., -3.5]), _norm([0.707, 0.707, 0.])),
        ],
        15300, 16300,
    ),
    # 2. Tight gates 8m: dense field, 8m spacing, mixed 3D
    (
        "tight_dense_20gate",
        [
            (np.array([ 8.,  0., -2.5]), _norm([1., 0., 0.])),
            (np.array([16.,  4., -4.0]), _norm([0.707, 0.707, 0.])),
            (np.array([16., 12., -2.5]), _norm([0., 1., 0.])),
            (np.array([ 8., 16., -4.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 0., 12., -2.5]), _norm([-1., 0., 0.])),
            (np.array([ 0.,  4., -4.0]), _norm([0., -1., 0.])),
            (np.array([ 8., -4., -2.5]), _norm([0.707, -0.707, 0.])),
            (np.array([16.,  0., -4.0]), _norm([1., 0., 0.])),
            (np.array([24.,  8., -2.5]), _norm([0.707, 0.707, 0.])),
            (np.array([24., 16., -4.0]), _norm([0., 1., 0.])),
            (np.array([16., 24., -2.5]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 8., 20., -4.0]), _norm([-1., 0., 0.])),
            (np.array([ 0., 16., -2.5]), _norm([0., -1., 0.])),
            (np.array([ 8.,  8., -4.0]), _norm([0.707, -0.707, 0.])),
            (np.array([16.,  4., -2.5]), _norm([1., 0., 0.])),
            (np.array([24., 12., -4.0]), _norm([0.707, 0.707, 0.])),
            (np.array([32., 20., -2.5]), _norm([0.707, 0.707, 0.])),
            (np.array([32., 28., -4.0]), _norm([0., 1., 0.])),
            (np.array([24., 32., -2.5]), _norm([-0.707, 0.707, 0.])),
            (np.array([16., 28., -4.0]), _norm([-1., 0., 0.])),
        ],
        15301, 16301,
    ),
    # 3. Hypersonic straight 30-gate: 25m spacing at 30 m/s
    (
        "hypersonic_30gate_25m",
        [(np.array([float(i * 25), 0., -2.5]), np.array([1., 0., 0.])) for i in range(1, 31)],
        15302, 16302,
    ),
    # 4. Alternating diagonal Z: every gate is 45° diagonal normal, alternating depth
    (
        "alternating_diagonal_18",
        [
            (np.array([12.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([20.,   8., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([28.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([36.,   8., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([44.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([52.,   8., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([60.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([68.,   8., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([76.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([84.,   8., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([92.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([100.,  8., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([108.,  0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([116.,  8., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([124.,  0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([132.,  8., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([140.,  0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([148.,  8., -5.5]), _norm([0.707, -0.707, 0.])),
        ],
        15303, 16303,
    ),
    # 5. 3D Labyrinth: 20-gate maze with back-and-forth + altitude
    (
        "labyrinth_3d_20",
        [
            (np.array([10.,  0., -2.0]), _norm([1., 0., 0.])),
            (np.array([25.,  0., -4.5]), _norm([1., 0., 0.])),
            (np.array([35.,  8., -2.0]), _norm([0.707, 0.707, 0.])),
            (np.array([35., 20., -4.5]), _norm([0., 1., 0.])),
            (np.array([25., 28., -2.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([10., 28., -4.5]), _norm([-1., 0., 0.])),
            (np.array([ 2., 20., -2.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 2.,  8., -4.5]), _norm([0., -1., 0.])),
            (np.array([10.,  0., -2.0]), _norm([0.707, -0.707, 0.])),
            (np.array([20., -8., -4.5]), _norm([0.707, -0.707, 0.])),
            (np.array([30., -4., -2.0]), _norm([1., 0., 0.])),
            (np.array([40.,  8., -4.5]), _norm([0.707, 0.707, 0.])),
            (np.array([45., 20., -2.0]), _norm([0., 1., 0.])),
            (np.array([38., 32., -4.5]), _norm([-0.707, 0.707, 0.])),
            (np.array([25., 38., -2.0]), _norm([-1., 0., 0.])),
            (np.array([ 8., 35., -4.5]), _norm([-0.707, -0.707, 0.])),
            (np.array([-5., 22., -2.0]), _norm([0., -1., 0.])),
            (np.array([ 0.,  8., -4.5]), _norm([0.707, -0.707, 0.])),
            (np.array([12., -5., -2.0]), _norm([0.707, -0.707, 0.])),
            (np.array([25., -2., -3.5]), _norm([1., 0., 0.])),
        ],
        15304, 16304,
    ),
    # 6. Inverted helix: ascend then descend in opposite corkscrew direction, 12 gates
    (
        "inverted_double_helix",
        [
            # Ascend phase (CW from above)
            (np.array([ 8.,  0.,  -4.0]), _norm([1., 0., 0.])),
            (np.array([14.,  6.,  -6.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 8., 12.,  -8.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 2.,  8., -10.0]), _norm([-1., 0., 0.])),
            (np.array([ 4.,  2., -12.0]), _norm([0.707, -0.707, 0.])),
            (np.array([12.,  0., -14.0]), _norm([1., 0., 0.])),
            # Descend phase (CCW from above, starting where ascend ended)
            (np.array([18.,  6., -12.0]), _norm([0.707, 0.707, 0.])),
            (np.array([12., 12., -10.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 4.,  8.,  -8.0]), _norm([-1., 0., 0.])),
            (np.array([ 6.,  2.,  -6.0]), _norm([0.707, -0.707, 0.])),
            (np.array([14.,  0.,  -4.0]), _norm([1., 0., 0.])),
            (np.array([22.,  6.,  -2.0]), _norm([0.707, 0.707, 0.])),
        ],
        15305, 16305,
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
    pl = PathPlannerNED(max_speed=_MAX_SPEED, approach_distance=5., exit_distance=1.)

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
        _ADIST = 5.0
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
    gp = res.get("gates_passed", 0); tt = res.get("total_time_s", 0.)
    return {
        "course": name, "gates_passed": gp, "total_gates": ng, "total_time_s": tt,
        "completed": res.get("completed", False), "passed": gp >= ng,
        "error": err[0][:200] if err else None,
    }


def main() -> int:
    print(f"\nLEGEND COURSE BATTERY — max_speed={_MAX_SPEED} m/s")
    print("=" * 65)
    results = []
    for name, defs, mp, vp in LEGEND_COURSES:
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
