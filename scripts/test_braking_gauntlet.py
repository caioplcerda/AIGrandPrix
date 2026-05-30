"""Anticipatory-braking gauntlet — NEGATIVE RESULT (braking is unnecessary).

A/B test of anticipatory braking on the HARD gauntlet order (hypersonic sprint FIRST,
then slalom/chaos/helix) — the case originally thought to fail on high-speed
straight→turn transitions.

FINDING (do not add braking to run_vq1.py):
- WITHOUT braking: 60/60 at 38 m/s (94.6s) AND 60/60 at 48 m/s (130.7s) — passes clean.
- WITH braking:    11/60 (stalls — the speed cap fights the approach-blend and oscillates).

The original 11/60 "transition failure" was actually the speed-adaptive lookahead
corner-cutting on weave gates (since reverted). With fixed-5m lookahead, the existing
approach-waypoint velocity blend ALREADY decelerates correctly into turns, robustly up
to near-terminal 48 m/s. Anticipatory braking solves a non-problem and regresses.

Kept as a documented negative result. Braking impl (_turn_speed_cap) is the experiment
that proved unnecessary, not production code.

Usage:
    python scripts/test_braking_gauntlet.py
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
logger = logging.getLogger("braking")

_MAX_SPEED = 38.0
_V_TURN_MIN = 14.0   # speed floor entering a sharp (90°+) turn
_BRAKE_DECEL = 28.0  # m/s² assumed for braking-distance calc (conservative vs 80 limit)
_LOOP_HZ = 50.0
_MAX_DUR = 400.0
_RP = 0.25
_IH = 0.75
_ADIST = 4.0


def _norm(v):
    a = np.array(v, dtype=float)
    n = np.linalg.norm(a)
    return a / n if n > 1e-9 else a


def _gate_crossed(gp, gn, op, np_, ih=_IH):
    n, c = gn, gp
    d1 = float(np.dot(op - c, n)); d2 = float(np.dot(np_ - c, n))
    if d1 * d2 > 0 or abs(d1 - d2) < 1e-9: return False
    t = d1 / (d1 - d2); x = op + t * (np_ - op); d = x - c
    up = np.array([0., 0., -1.]); r = np.cross(n, up); rn = float(np.linalg.norm(r))
    r = r / rn if rn > 1e-6 else np.array([0., 1., 0.])
    return abs(float(np.dot(d, r))) <= ih and abs(float(np.dot(d, up))) <= ih


def _helix(cx, cy, radius, z_start, z_step, n_gates, start_angle=0.0):
    g = []
    for i in range(n_gates):
        a = start_angle + i * (2 * np.pi / n_gates)
        g.append((np.array([cx + radius*np.cos(a), cy + radius*np.sin(a), z_start + i*z_step]),
                  _norm([-np.cos(a), -np.sin(a), 0.])))
    return g


def _build_hard_gauntlet():
    """HARD order: hypersonic sprint FIRST (builds 38 m/s), then turns. The braking
    must save the sprint→slalom and slalom-exit→chaos transitions."""
    gates = []
    # Seg 1: hypersonic sprint (10 gates, 30m) — drone hits 38 m/s
    for i in range(1, 11):
        gates.append((np.array([float(i * 30), 0., -3.0]), np.array([1., 0., 0.])))
    # Seg 2: slalom weave (12 gates) — the killer transition from full speed
    bx = gates[-1][0][0]
    for i in range(12):
        gates.append((np.array([bx + 20.*(i+1), 10.*(i%2), -3.0 if i%2==0 else -5.0]),
                      np.array([1., 0., 0.])))
    # Seg 3: chaos (15 gates)
    bx = gates[-1][0][0] + 20.
    chaos = [
        (0.,0.,-3.0,[1.,0.,0.]),(14.,14.,-6.0,[0.707,0.707,0.]),(25.,3.,-3.0,[0.707,-0.707,0.]),
        (39.,-11.,-6.0,[1.,0.,0.]),(53.,3.,-3.0,[0.707,0.707,0.]),(64.,17.,-6.0,[0.,1.,0.]),
        (50.,31.,-3.0,[-0.707,0.707,0.]),(36.,20.,-6.0,[-1.,0.,0.]),(22.,8.,-3.0,[-0.707,-0.707,0.]),
        (8.,-6.,-6.0,[0.,-1.,0.]),(-4.,8.,-3.0,[-0.707,0.707,0.]),(8.,24.,-6.0,[0.707,0.707,0.]),
        (24.,34.,-3.0,[0.707,0.707,0.]),(40.,24.,-6.0,[0.707,-0.707,0.]),(52.,10.,-5.0,[0.,-1.,0.]),
    ]
    for dx, dy, z, n in chaos:
        gates.append((np.array([bx + dx, dy, z]), _norm(n)))
    # Seg 4: hypersonic sprint again (8 gates) — tests turn→straight (easy) + straight→? 
    bx = gates[-1][0][0] + 30.
    for i in range(8):
        gates.append((np.array([bx + i*30., 0., -3.0]), np.array([1., 0., 0.])))
    # Seg 5: helix finale (15 gates) — straight→helix transition (another braking case)
    bx = gates[-1][0][0] + 25.
    gates.extend([(np.array([p[0]+bx, p[1], p[2]]), n)
                  for p, n in (_helix(0.,0.,16.,-3.0,-0.5,8,0.) + _helix(0.,0.,14.,-7.0,0.5,7,np.pi))])
    return gates


HARD_GAUNTLET = _build_hard_gauntlet()


def _turn_speed_cap(rem, pos, v_max):
    """Anticipatory braking speed cap from upcoming gate geometry.

    Looks at heading into the next gate vs heading from it to the gate after. A sharp
    bend → low turn speed. Caps to that turn speed only within braking distance.
    """
    if len(rem) < 2:
        return v_max
    d_in = rem[0].position - pos
    dist = float(np.linalg.norm(d_in))
    if dist < 1e-6:
        return v_max
    d_in_n = d_in / dist
    d_out = rem[1].position - rem[0].position
    d_out_n = _norm(d_out)
    turn_cos = float(np.dot(d_in_n, d_out_n))   # 1 straight, 0 right-angle, <0 hairpin
    # turn speed: full when straight (cos≥0.9), floor at sharp (cos≤0)
    frac = max(0.0, min(1.0, (turn_cos - 0.0) / 0.9))
    v_turn = _V_TURN_MIN + (v_max - _V_TURN_MIN) * frac
    # braking distance to shed from v_max to v_turn
    brake_dist = max(0.0, (v_max**2 - v_turn**2) / (2.0 * _BRAKE_DECEL))
    if dist <= brake_dist + _ADIST:
        return v_turn
    return v_max


def run_course(name, gate_defs, mp, vp, braking=True):
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
                vx=float(lg.normal[0]*_MAX_SPEED), vy=float(lg.normal[1]*_MAX_SPEED), vz=0., yaw=0.))
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
        v_cap = _turn_speed_cap(rem, st.pos_ned, _MAX_SPEED) if braking else _MAX_SPEED
        if rem:
            apt = rem[0].position - rem[0].normal * _ADIST
            av = apt - st.pos_ned; da = float(np.linalg.norm(av))
            ah = float(np.dot(av, rem[0].normal)) > 0.0
            if da > 0.1 and ah:
                bl = min(1.0, max(0.0, (da - 3.0) / 8.0))
                cv = bl * (av / da * v_cap) + (1.0 - bl) * wp.vel
            else:
                cv = wp.vel
        else:
            cv = wp.vel
        # enforce speed cap on the final command vector
        cvs = float(np.linalg.norm(cv))
        if cvs > v_cap and cvs > 1e-6:
            cv = cv / cvs * v_cap
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
    print(f"\nANTICIPATORY-BRAKING GAUNTLET — HARD order (sprint→turn), {len(HARD_GAUNTLET)} gates @ {_MAX_SPEED} m/s")
    print("=" * 70)
    for label, braking in [("WITHOUT braking", False), ("WITH braking", True)]:
        r = run_course(f"hard_gauntlet_{braking}", HARD_GAUNTLET, 16300 if braking else 16310,
                       17300 if braking else 17310, braking=braking)
        retries = 0
        while not r.get("passed") and r.get("total_time_s", 0.) < 1.0 and r.get("gates_passed", 0) == 0 and retries < 3:
            retries += 1; time.sleep(2.)
            r = run_course(f"hard_gauntlet_{braking}", HARD_GAUNTLET,
                           (16300 if braking else 16310) + 100*retries,
                           (17300 if braking else 17310) + 100*retries, braking=braking)
        st = "PASS ✓" if r["passed"] else "FAIL ✗"
        print(f"  {label:18s}: {st}  {r['gates_passed']}/{r['total_gates']}  {r['total_time_s']:.2f}s")
        time.sleep(3.)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
