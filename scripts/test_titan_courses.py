"""Titan course battery — 38 m/s (physics ceiling 44 m/s, MAX_ACCEL 40).

Highest reliable speed. Spacing scaled to ≥16m (38 m/s × 0.42s/gate margin).
Only physics-verified patterns: diagonal zigzag, straight altitude, wide helixes.

Usage:
    python scripts/test_titan_courses.py
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
logger = logging.getLogger("titan")

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


def _helix_outward(cx, cy, radius, z_start, z_step, n_gates, start_angle=0.0):
    gates = []
    for i in range(n_gates):
        angle = start_angle + i * (2 * np.pi / n_gates)
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        z = z_start + i * z_step
        nx, ny = -np.cos(angle), -np.sin(angle)
        gates.append((np.array([x, y, z]), _norm([nx, ny, 0.])))
    return gates


# Spacing scaled up for 38 m/s: diagonal 18m, helix radius 16-18m
TITAN_COURSES = [
    # 1. Diagonal sprint 50: 18m zigzag at 38 m/s
    (
        "titan_slalom_50",
        [(np.array([20.*(i+1), 10.*(i%2), -3.0 if i%2==0 else -5.0]),
          np.array([1., 0., 0.])) for i in range(50)],
        16050, 17050,
    ),
    # 2. Diagonal 30: 18m spacing
    (
        "titan_slalom_30",
        [(np.array([20.*(i+1), 10.*(i%2), -3.0 if i%2==0 else -5.0]),
          np.array([1., 0., 0.])) for i in range(30)],
        16051, 17051,
    ),
    # 3. Straight altitude 20: 26m spacing, 4m oscillations
    (
        "titan_altitude_straight_20",
        [(np.array([26.*(i+1), 0., -2.0 if i%2==0 else -6.0]), np.array([1., 0., 0.]))
         for i in range(20)],
        16052, 17052,
    ),
    # 4. Double helix 20: radius 18m/16m (arc 11.3m/10m at 38 m/s)
    (
        "titan_helix_20",
        (
            _helix_outward(cx=0., cy=0., radius=18., z_start=-3.0, z_step=-0.5, n_gates=10, start_angle=0.)
            + _helix_outward(cx=0., cy=0., radius=16., z_start=-8.0, z_step=0.5, n_gates=10, start_angle=np.pi)
        ),
        16053, 17053,
    ),
    # 5. Chaos 30: chaos pattern scaled ×1.4 for 38 m/s
    (
        "titan_chaos_30",
        [
            (np.array([  14.,   0., -3.0]), _norm([1., 0., 0.])),
            (np.array([  28.,  14., -6.0]), _norm([0.707, 0.707, 0.])),
            (np.array([  39.,   3., -3.0]), _norm([0.707, -0.707, 0.])),
            (np.array([  53., -11., -6.0]), _norm([1., 0., 0.])),
            (np.array([  67.,   3., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([  78.,  17., -6.0]), _norm([0., 1., 0.])),
            (np.array([  64.,  31., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([  50.,  20., -6.0]), _norm([-1., 0., 0.])),
            (np.array([  34.,   8., -3.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([  20.,  -6., -6.0]), _norm([0., -1., 0.])),
            (np.array([   8.,   8., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([  -6.,  22., -6.0]), _norm([0., 1., 0.])),
            (np.array([   8.,  36., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([  25.,  48., -6.0]), _norm([0.707, 0.707, 0.])),
            (np.array([  39.,  36., -3.0]), _norm([0.707, -0.707, 0.])),
            (np.array([  50.,  20., -6.0]), _norm([0., -1., 0.])),
            (np.array([  36.,   6., -3.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([  20.,  -8., -6.0]), _norm([-1., 0., 0.])),
            (np.array([   3.,   6., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([  -8.,  20., -6.0]), _norm([0., 1., 0.])),
            (np.array([   3.,  34., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([  20.,  45., -6.0]), _norm([0.707, 0.707, 0.])),
            (np.array([  34.,  34., -3.0]), _norm([0.707, -0.707, 0.])),
            (np.array([  45.,  17., -6.0]), _norm([0., -1., 0.])),
            (np.array([  31.,   3., -3.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([  14., -11., -6.0]), _norm([-1., 0., 0.])),
            (np.array([  -3.,   3., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ -11.,  17., -6.0]), _norm([0., 1., 0.])),
            (np.array([   0.,  31., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([  17.,  42., -6.0]), _norm([0.707, 0.707, 0.])),
        ],
        16054, 17054,
    ),
    # 6. Hypersonic 40: 30m spacing straight, 38 m/s
    (
        "titan_hypersonic_40",
        [(np.array([float(i * 30), 0., -3.0]), np.array([1., 0., 0.])) for i in range(1, 41)],
        16055, 17055,
    ),
    # 7. Diagonal 40: 18m zigzag
    (
        "titan_slalom_40",
        [(np.array([20.*(i+1), 10.*(i%2), -3.0 if i%2==0 else -5.0]),
          np.array([1., 0., 0.])) for i in range(40)],
        16056, 17056,
    ),
    # 8. Triple helix 24: radius 16m, _ADIST=4
    (
        "titan_triple_helix_24",
        (
            _helix_outward(cx=0., cy=0., radius=16., z_start=-3.0, z_step=-0.5, n_gates=8, start_angle=0.)
            + _helix_outward(cx=45., cy=0., radius=16., z_start=-7.0, z_step=0.5, n_gates=8, start_angle=np.pi)
            + _helix_outward(cx=90., cy=0., radius=16., z_start=-3.0, z_step=-0.5, n_gates=8, start_angle=0.)
        ),
        16057, 17057,
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
    pl = PathPlannerNED(max_speed=_MAX_SPEED, approach_distance=_ADIST, exit_distance=1.)

    dt = 1. / _LOOP_HZ; ni = 0; wps = []; lrp = -_RP; prev = None; ts = time.time()
    while True:
        tl = time.perf_counter(); el = time.time() - ts
        if el > _MAX_DUR:
            ev.wait(timeout=10.)
            break
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
        wp = pl.next_position_target(wps, st.pos_ned, lookahead_m=5.) if wps else None
        if wp is None:
            from aigrandprix.planning.path_planner_ned import WaypointNED
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
    gp = res.get("gates_passed", 0); tt = res.get("total_time_s", 0.)
    return {
        "course": name, "gates_passed": gp, "total_gates": ng, "total_time_s": tt,
        "completed": res.get("completed", False), "passed": gp >= ng,
        "error": err[0][:200] if err else None,
    }


def main() -> int:
    print(f"\nTITAN COURSE BATTERY — max_speed={_MAX_SPEED} m/s")
    print("=" * 65)
    results = []
    for name, defs, mp, vp in TITAN_COURSES:
        defs_list = list(defs)
        print(f"  {name} ({len(defs_list)} gates)...", flush=True, end=" ")
        r = run_course(name, defs_list, mp, vp)
        # Retry startup-race glitch (tt<1s + zero gates = server never populated res, not a real failure)
        retries = 0
        while not r.get("passed") and r.get("total_time_s", 0.) < 1.0 and r.get("gates_passed", 0) == 0 and retries < 3:
            retries += 1
            time.sleep(2.)
            r = run_course(name, defs_list, mp + 100 * retries, vp + 100 * retries)
        results.append(r)
        if r.get("error"):
            print(f"ERROR: {r['error'][:100]}")
        else:
            st = "PASS ✓" if r["passed"] else "FAIL ✗"
            print(f"{st}  {r['gates_passed']}/{r['total_gates']}  {r['total_time_s']:.2f}s")
        time.sleep(3.)

    print("=" * 65)
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    print(f"TOTAL: {passed}/{total} courses passed")
    print("=" * 65)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
