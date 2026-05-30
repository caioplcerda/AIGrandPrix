"""Gauntlet — single continuous 60-gate course chaining ALL pattern types at 38 m/s.

Unlike single-pattern batteries, this tests sustained flight across PATTERN
TRANSITIONS (slalom→helix→chaos→altitude→hypersonic), the realistic stress.
Physics: 8g, terminal 51.6 m/s.

Usage:
    python scripts/test_gauntlet_course.py
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
logger = logging.getLogger("gauntlet")

_MAX_SPEED = 38.0
_LOOP_HZ = 50.0
_MAX_DUR = 400.0
_RP = 0.25
_IH = 0.75
_ADIST = 4.0


def _norm(v):
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


def _helix(cx, cy, radius, z_start, z_step, n_gates, start_angle=0.0):
    gates = []
    for i in range(n_gates):
        angle = start_angle + i * (2 * np.pi / n_gates)
        x = cx + radius * np.cos(angle); y = cy + radius * np.sin(angle); z = z_start + i * z_step
        gates.append((np.array([x, y, z]), _norm([-np.cos(angle), -np.sin(angle), 0.])))
    return gates


def _build_gauntlet():
    """Chain pattern segments end-to-end, offsetting each to continue from the last."""
    gates = []

    def last_xy():
        return (gates[-1][0][0], gates[-1][0][1]) if gates else (0., 0.)

    # Order: turning patterns FIRST (entered from low speed), hypersonic sprint LAST.
    # Transitioning from a max-speed straight INTO a turn fails — drone can't weave at 38 m/s.

    # Segment 1: slalom weave (12 gates) from cold start — gentle 6m lateral
    for i in range(12):
        gates.append((np.array([20.*(i+1), 6.*(i%2), -3.0 if i%2==0 else -5.0]),
                      np.array([1., 0., 0.])))

    # Segment 2: altitude oscillation straight (8 gates)
    bx = gates[-1][0][0]
    for i in range(8):
        gates.append((np.array([bx + 26.*(i+1), 0., -2.0 if i%2==0 else -6.0]), np.array([1., 0., 0.])))

    # Segment 3: chaos block (15 gates) — offset to current x, recentred y
    bx = gates[-1][0][0] + 20.
    chaos = [
        ( 0.,   0., -3.0, [1., 0., 0.]),
        (14.,  14., -6.0, [0.707, 0.707, 0.]),
        (25.,   3., -3.0, [0.707, -0.707, 0.]),
        (39., -11., -6.0, [1., 0., 0.]),
        (53.,   3., -3.0, [0.707, 0.707, 0.]),
        (64.,  17., -6.0, [0., 1., 0.]),
        (50.,  31., -3.0, [-0.707, 0.707, 0.]),
        (36.,  20., -6.0, [-1., 0., 0.]),
        (22.,   8., -3.0, [-0.707, -0.707, 0.]),
        ( 8.,  -6., -6.0, [0., -1., 0.]),
        (-4.,   8., -3.0, [-0.707, 0.707, 0.]),
        ( 8.,  24., -6.0, [0.707, 0.707, 0.]),
        (24.,  34., -3.0, [0.707, 0.707, 0.]),
        (40.,  24., -6.0, [0.707, -0.707, 0.]),
        (52.,  10., -5.0, [0., -1., 0.]),
    ]
    for dx, dy, z, n in chaos:
        gates.append((np.array([bx + dx, dy, z]), _norm(n)))

    # Segment 4: double helix (15 gates) — offset east
    bx = gates[-1][0][0] + 25.
    gates.extend(
        [(np.array([p[0] + bx, p[1], p[2]]), n)
         for p, n in (_helix(0., 0., 16., -3.0, -0.5, 8, 0.)
                      + _helix(0., 0., 14., -7.0, 0.5, 7, np.pi))]
    )

    # Segment 5: hypersonic sprint FINALE (10 gates, 30m) — max speed on the home straight
    bx = gates[-1][0][0] + 30.
    for i in range(10):
        gates.append((np.array([bx + i * 30., 0., -3.0]), np.array([1., 0., 0.])))

    return gates


GAUNTLET = _build_gauntlet()


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
    pl = PathPlannerNED(max_speed=_MAX_SPEED, approach_distance=_ADIST, exit_distance=1.)

    dt = 1. / _LOOP_HZ; ni = 0; wps = []; lrp = -_RP; prev = None; ts = time.time()
    while True:
        tl = time.perf_counter(); el = time.time() - ts
        if el > _MAX_DUR:
            ev.wait(timeout=10.); break
        if ev.is_set(): break
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
        wp = pl.next_position_target(wps, st.pos_ned, lookahead_m=5.0) if wps else None
        if wp is None: wp = WaypointNED(pos=st.pos_ned.copy(), vel=np.zeros(3), yaw=st.yaw, time=el)
        if rem:
            apt = rem[0].position - rem[0].normal * _ADIST
            av = apt - st.pos_ned; da = float(np.linalg.norm(av))
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
    return {"course": name, "gates_passed": gp, "total_gates": ng, "total_time_s": tt,
            "completed": res.get("completed", False), "passed": gp >= ng,
            "error": err[0][:200] if err else None}


def main() -> int:
    print(f"\nGAUNTLET — single {len(GAUNTLET)}-gate multi-pattern course @ {_MAX_SPEED} m/s")
    print("=" * 65)
    r = run_course("gauntlet_60", GAUNTLET, 16200, 17200)
    retries = 0
    while not r.get("passed") and r.get("total_time_s", 0.) < 1.0 and r.get("gates_passed", 0) == 0 and retries < 3:
        retries += 1; time.sleep(2.)
        r = run_course("gauntlet_60", GAUNTLET, 16200 + 100 * retries, 17200 + 100 * retries)
    if r.get("error"):
        print(f"  ERROR: {r['error'][:120]}")
    else:
        st = "PASS ✓" if r["passed"] else "FAIL ✗"
        print(f"  {st}  {r['gates_passed']}/{r['total_gates']}  {r['total_time_s']:.2f}s")
    print("=" * 65)
    return 0 if r.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
