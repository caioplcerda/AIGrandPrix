"""Ultra course battery — max difficulty, 30 m/s.

Harder than extreme: tighter spacing, deeper altitude swings, complex 3D paths.
All at max_speed=30 m/s.

Usage:
    python scripts/test_ultra_courses.py
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
logger = logging.getLogger("ultra")

_MAX_SPEED = 30.0
_LOOP_HZ = 50.0
_MAX_DUR = 180.0
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


ULTRA_COURSES = [
    # 1. Speed wall: 25-gate straight, 20m spacing — absolute speed benchmark at 30 m/s
    (
        "speed_wall_25gate",
        [(np.array([float(i * 20), 0., -2.5]), np.array([1., 0., 0.])) for i in range(1, 26)],
        15200, 16200,
    ),
    # 2. Deep corkscrew: 10-gate helix with 6m spacing, −12m altitude drop
    (
        "deep_corkscrew_10",
        [
            (np.array([ 8.,  0.,  -2.0]), _norm([1., 0., 0.])),
            (np.array([14.,  6.,  -3.5]), _norm([0.707, 0.707, 0.])),
            (np.array([ 8., 12.,  -5.5]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 2.,  8.,  -7.0]), _norm([-1., 0., 0.])),
            (np.array([ 4.,  2.,  -8.5]), _norm([0.707, -0.707, 0.])),
            (np.array([12.,  0., -10.0]), _norm([1., 0., 0.])),
            (np.array([18.,  8., -11.0]), _norm([0.707, 0.707, 0.])),
            (np.array([12., 16., -12.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 4., 12., -13.0]), _norm([-1., 0., 0.])),
            (np.array([ 8.,  4., -14.0]), _norm([0.707, -0.707, 0.])),
        ],
        15201, 16201,
    ),
    # 3. Roller coaster 18: massive altitude swings ±6m every 3 gates over 18 gates
    (
        "roller_coaster_18",
        [
            (np.array([ 15.,  0.,  -1.5]), _norm([1., 0., 0.])),
            (np.array([ 30.,  0.,  -7.5]), _norm([1., 0., 0.])),
            (np.array([ 45.,  0.,  -1.5]), _norm([1., 0., 0.])),
            (np.array([ 60.,  5.,  -8.0]), _norm([1., 0., 0.])),
            (np.array([ 75.,  0.,  -1.5]), _norm([1., 0., 0.])),
            (np.array([ 90., -5.,  -7.5]), _norm([1., 0., 0.])),
            (np.array([105.,  0.,  -2.0]), _norm([1., 0., 0.])),
            (np.array([120.,  5.,  -8.0]), _norm([1., 0., 0.])),
            (np.array([135.,  0.,  -1.5]), _norm([1., 0., 0.])),
            (np.array([150., -5.,  -7.5]), _norm([1., 0., 0.])),
            (np.array([165.,  0.,  -2.0]), _norm([1., 0., 0.])),
            (np.array([180.,  5.,  -8.0]), _norm([1., 0., 0.])),
            (np.array([195.,  0.,  -1.5]), _norm([1., 0., 0.])),
            (np.array([210., -5.,  -7.5]), _norm([1., 0., 0.])),
            (np.array([225.,  0.,  -2.0]), _norm([1., 0., 0.])),
            (np.array([240.,  5.,  -8.0]), _norm([1., 0., 0.])),
            (np.array([255.,  0.,  -1.5]), _norm([1., 0., 0.])),
            (np.array([270.,  0.,  -2.5]), _norm([1., 0., 0.])),
        ],
        15202, 16202,
    ),
    # 4. Infinity loop 3D: two crossing loops at different altitudes, 16 gates
    (
        "infinity_loop_3d",
        [
            # Left loop (lower, z=-3)
            (np.array([ 0.,  0., -3.0]), _norm([1., 0., 0.])),
            (np.array([12.,  8., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([20.,  0., -3.0]), _norm([0., -1., 0.])),
            (np.array([12., -8., -3.0]), _norm([-0.707, -0.707, 0.])),
            # Cross point (higher, z=-6)
            (np.array([ 0.,  0., -6.0]), _norm([1., 0., 0.])),
            # Right loop (lower, z=-3)
            (np.array([-12.,  8., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([-20.,  0., -3.0]), _norm([0., -1., 0.])),
            (np.array([-12., -8., -3.0]), _norm([0.707, -0.707, 0.])),
            # Return cross (higher)
            (np.array([ 0.,  0., -6.0]), _norm([1., 0., 0.])),
            # Second pass left loop
            (np.array([12.,  8., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([20.,  0., -3.0]), _norm([0., -1., 0.])),
            (np.array([12., -8., -3.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 0.,  0., -3.0]), _norm([-1., 0., 0.])),
            # Final cross at altitude
            (np.array([-12.,  8., -6.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([-20.,  0., -3.0]), _norm([0., 1., 0.])),
            (np.array([-12., -8., -3.0]), _norm([0.707, -0.707, 0.])),
        ],
        15203, 16203,
    ),
    # 5. Triple hairpin: 3 back-to-back 180° U-turns + altitude change between each
    (
        "triple_hairpin_3d",
        [
            # Pass 1 forward
            (np.array([10.,  0., -2.5]), _norm([1., 0., 0.])),
            (np.array([25.,  0., -2.5]), _norm([1., 0., 0.])),
            (np.array([35.,  5., -2.5]), _norm([0.707, 0.707, 0.])),
            (np.array([35., 15., -2.5]), _norm([0., 1., 0.])),
            # Hairpin 1 + altitude climb
            (np.array([25., 20., -5.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([10., 20., -5.0]), _norm([-1., 0., 0.])),
            (np.array([ 0., 15., -5.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 0.,  5., -5.0]), _norm([0., -1., 0.])),
            # Hairpin 2 + altitude drop
            (np.array([10.,  0., -2.5]), _norm([0.707, -0.707, 0.])),
            (np.array([25.,  0., -2.5]), _norm([1., 0., 0.])),
            (np.array([35.,  5., -2.5]), _norm([0.707, 0.707, 0.])),
            (np.array([35., 15., -2.5]), _norm([0., 1., 0.])),
            # Hairpin 3 + altitude climb again
            (np.array([25., 20., -7.5]), _norm([-0.707, 0.707, 0.])),
            (np.array([10., 20., -7.5]), _norm([-1., 0., 0.])),
            (np.array([ 0., 15., -7.5]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 0.,  5., -7.5]), _norm([0., -1., 0.])),
        ],
        15204, 16204,
    ),
    # 6. Grand prix 30-gate circuit at 30 m/s
    (
        "grand_prix_30gate",
        [
            (np.array([ 20.,   0., -2.5]), _norm([1., 0., 0.])),
            (np.array([ 40.,   0., -3.5]), _norm([1., 0., 0.])),
            (np.array([ 60.,   5., -2.0]), _norm([1., 0., 0.])),
            (np.array([ 80.,  10., -4.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 90.,  22., -2.5]), _norm([0., 1., 0.])),
            (np.array([ 80.,  35., -4.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 65.,  40., -2.0]), _norm([-1., 0., 0.])),
            (np.array([ 45.,  38., -4.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 30.,  25., -2.5]), _norm([0., -1., 0.])),
            (np.array([ 25.,  10., -4.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 15.,   0., -2.0]), _norm([-1., 0., 0.])),
            (np.array([  5., -10., -4.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ -5., -20., -2.5]), _norm([0., -1., 0.])),
            (np.array([  5., -35., -4.0]), _norm([0.707, -0.707, 0.])),
            (np.array([ 20., -40., -2.0]), _norm([1., 0., 0.])),
            (np.array([ 40., -38., -4.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 55., -25., -2.5]), _norm([0., 1., 0.])),
            (np.array([ 60., -10., -4.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 75.,   0., -2.0]), _norm([1., 0., 0.])),
            (np.array([ 90., -10., -4.0]), _norm([0.707, -0.707, 0.])),
            (np.array([100., -22., -2.5]), _norm([0., -1., 0.])),
            (np.array([ 90., -35., -4.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 75., -40., -2.0]), _norm([-1., 0., 0.])),
            (np.array([ 55., -35., -4.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 45., -22., -2.5]), _norm([0., 1., 0.])),
            (np.array([ 50., -10., -4.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 60.,   0., -2.0]), _norm([1., 0., 0.])),
            (np.array([ 75.,  10., -4.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 80.,  22., -2.5]), _norm([0., 1., 0.])),
            (np.array([ 70.,  35., -3.5]), _norm([-0.707, 0.707, 0.])),
        ],
        15205, 16205,
    ),
    # 7. Upside-down Z: steep diagonal descents + 180° turns, 12 gates
    (
        "diagonal_z_descent",
        [
            (np.array([10.,   0., -8.0]), _norm([1., 0., -0.3])),
            (np.array([25.,   0., -5.0]), _norm([1., 0., 0.3])),
            (np.array([40.,   0., -2.0]), _norm([1., 0., 0.])),
            (np.array([50.,   8., -2.0]), _norm([0.707, 0.707, 0.])),
            (np.array([50.,  20., -5.0]), _norm([0., 1., -0.3])),
            (np.array([40.,  28., -8.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([25.,  25., -8.0]), _norm([-1., 0., 0.])),
            (np.array([10.,  25., -5.0]), _norm([-1., 0., 0.3])),
            (np.array([ 5.,  15., -2.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 5.,   5., -5.0]), _norm([0., -1., -0.3])),
            (np.array([15.,  -2., -8.0]), _norm([0.707, -0.707, 0.])),
            (np.array([30.,  -5., -5.0]), _norm([1., 0., 0.3])),
        ],
        15206, 16206,
    ),
    # 8. Tight 8m spiral 15-gate: small loops, 8m spacing, altitude progression
    (
        "tight_spiral_15",
        [
            (np.array([10.,  0., -2.0]), _norm([1., 0., 0.])),
            (np.array([18.,  8., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([10., 16., -4.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 2., 12., -5.0]), _norm([-1., 0., 0.])),
            (np.array([ 4.,  4., -6.0]), _norm([0.707, -0.707, 0.])),
            (np.array([14.,  0., -7.0]), _norm([1., 0., 0.])),
            (np.array([22., 10., -8.0]), _norm([0.707, 0.707, 0.])),
            (np.array([14., 20., -9.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 4., 16.,-10.0]), _norm([-1., 0., 0.])),
            (np.array([ 6.,  6.,-11.0]), _norm([0.707, -0.707, 0.])),
            (np.array([16.,  0.,-12.0]), _norm([1., 0., 0.])),
            (np.array([26., 10.,-11.0]), _norm([0.707, 0.707, 0.])),
            (np.array([20., 20.,-10.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([10., 18., -9.0]), _norm([-1., 0., 0.])),
            (np.array([14.,  8., -8.0]), _norm([0.707, -0.707, 0.])),
        ],
        15207, 16207,
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
    print(f"\nULTRA COURSE BATTERY — max_speed={_MAX_SPEED} m/s")
    print("=" * 65)
    results = []
    for name, defs, mp, vp in ULTRA_COURSES:
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
