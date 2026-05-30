"""Omega course battery — 33 m/s (10% faster than godtier's 30 m/s).

Physics ceiling: MAX_SPEED=38, MAX_ACCEL=35 in physics_6dof.py.
All courses use _ADIST=4 (same as godtier) and physics-verified gate positions.

Usage:
    python scripts/test_omega_courses.py
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
logger = logging.getLogger("omega")

_MAX_SPEED = 33.0
_LOOP_HZ = 50.0
_MAX_DUR = 400.0
_RP = 0.25
_IH = 0.75
_ADIST = 4.0  # same as godtier — _ADIST=5 caused approach geometry failures


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


OMEGA_COURSES = [
    # 1. Diagonal sprint 50: 50 gates, 14m zigzag
    (
        "omega_slalom_50",
        [(np.array([14.*(i+1), 10.*(i%2), -3.0 if i%2==0 else -5.5]),
          np.array([1., 0., 0.])) for i in range(50)],
        15970, 16970,
    ),
    # 2. Diagonal sprint 28: proven reliable at 33+ m/s
    (
        "omega_slalom_28",
        [(np.array([14.*(i+1), 10.*(i%2), -3.0 if i%2==0 else -5.5]),
          np.array([1., 0., 0.])) for i in range(28)],
        15971, 16971,
    ),
    # 3. Straight altitude 20: 20m spacing, 3m oscillations, NO direction changes
    (
        "omega_altitude_straight_20",
        [(np.array([20.*(i+1), 0., -2.0 if i%2==0 else -5.0]), np.array([1., 0., 0.]))
         for i in range(20)],
        15972, 16972,
    ),
    # 4. Double helix 20: godtier-proven 14m/10m radii, _ADIST=4
    (
        "omega_helix_20",
        (
            _helix_outward(cx=0., cy=0., radius=14., z_start=-3.0, z_step=-0.5, n_gates=10, start_angle=0.)
            + _helix_outward(cx=0., cy=0., radius=10., z_start=-8.0, z_step=0.5, n_gates=10, start_angle=np.pi)
        ),
        15973, 16973,
    ),
    # 5. Chaos 30: godtier_chaos_3d pattern at 33 m/s — same geometry as godtier (10m spacing)
    (
        "omega_chaos_30",
        [
            (np.array([  10.,   0., -3.0]), _norm([1., 0., 0.])),
            (np.array([  20.,  10., -5.5]), _norm([0.707, 0.707, 0.])),
            (np.array([  28.,   2., -3.0]), _norm([0.707, -0.707, 0.])),
            (np.array([  38.,  -8., -5.5]), _norm([1., 0., 0.])),
            (np.array([  48.,   2., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([  56.,  12., -5.5]), _norm([0., 1., 0.])),
            (np.array([  46.,  22., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([  36.,  14., -5.5]), _norm([-1., 0., 0.])),
            (np.array([  24.,   6., -3.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([  14.,  -4., -5.5]), _norm([0., -1., 0.])),
            (np.array([   6.,   6., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([  -4.,  16., -5.5]), _norm([0., 1., 0.])),
            (np.array([   6.,  26., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([  18.,  34., -5.5]), _norm([0.707, 0.707, 0.])),
            (np.array([  28.,  26., -3.0]), _norm([0.707, -0.707, 0.])),
            (np.array([  36.,  14., -5.5]), _norm([0., -1., 0.])),
            (np.array([  26.,   4., -3.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([  14.,  -6., -5.5]), _norm([-1., 0., 0.])),
            (np.array([   2.,   4., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([  -6.,  14., -5.5]), _norm([0., 1., 0.])),
            (np.array([   2.,  24., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([  14.,  32., -5.5]), _norm([0.707, 0.707, 0.])),
            (np.array([  24.,  24., -3.0]), _norm([0.707, -0.707, 0.])),
            (np.array([  32.,  12., -5.5]), _norm([0., -1., 0.])),
            (np.array([  22.,   2., -3.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([  10.,  -8., -5.5]), _norm([-1., 0., 0.])),
            (np.array([  -2.,   2., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([  -8.,  12., -5.5]), _norm([0., 1., 0.])),
            (np.array([   0.,  22., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([  12.,  30., -5.5]), _norm([0.707, 0.707, 0.])),
        ],
        15974, 16974,
    ),
    # 6. Hypersonic 35: 25m spacing straight, 33 m/s
    (
        "omega_hypersonic_35",
        [(np.array([float(i * 25), 0., -3.0]), np.array([1., 0., 0.])) for i in range(1, 36)],
        15975, 16975,
    ),
    # 7. Diagonal 40: 40 gates, 14m zigzag
    (
        "omega_slalom_40",
        [(np.array([14.*(i+1), 10.*(i%2), -3.0 if i%2==0 else -5.5]),
          np.array([1., 0., 0.])) for i in range(40)],
        15976, 16976,
    ),
    # 8. Triple helix 24: godtier-proven 12m radii, _ADIST=4
    (
        "omega_triple_helix_24",
        (
            _helix_outward(cx=0., cy=0., radius=12., z_start=-3.0, z_step=-0.5, n_gates=8, start_angle=0.)
            + _helix_outward(cx=35., cy=0., radius=12., z_start=-7.0, z_step=0.5, n_gates=8, start_angle=np.pi)
            + _helix_outward(cx=70., cy=0., radius=12., z_start=-3.0, z_step=-0.5, n_gates=8, start_angle=0.)
        ),
        15977, 16977,
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
    print(f"\nOMEGA COURSE BATTERY — max_speed={_MAX_SPEED} m/s")
    print("=" * 65)
    results = []
    for name, defs, mp, vp in OMEGA_COURSES:
        defs_list = list(defs)
        print(f"  {name} ({len(defs_list)} gates)...", flush=True, end=" ")
        r = run_course(name, defs_list, mp, vp)
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
