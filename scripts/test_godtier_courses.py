"""God-tier course battery — maximum difficulty stress test.

Rules learned from failures:
- Min gate spacing: 8m (5m impossible at 30 m/s)
- Max altitude delta between consecutive gates: 5m
- Helix radius: ≥10m (tangential normals need room for approach geometry)
- Approach dist: 4m (compromise between 3m tight and 5m ultra)

Harder than legend via: longer courses (30-50 gates), deeper 3D geometry,
more direction changes, back-and-forth traversals, double helixes.

Usage:
    python scripts/test_godtier_courses.py
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
logger = logging.getLogger("godtier")

_MAX_SPEED = 30.0
_LOOP_HZ = 50.0
_MAX_DUR = 400.0
_RP = 0.25
_IH = 0.75
_ADIST = 4.0  # slightly tighter than legend's 5m but safer than 3m


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


# ---------------------------------------------------------------------------
# Course definitions (all spacing ≥8m, altitude delta ≤5m)
# ---------------------------------------------------------------------------

def _helix_outward(cx, cy, radius, z_start, z_step, n_gates, start_angle=0.0):
    """Helix with inward-pointing normals — drone approaches from outside."""
    gates = []
    for i in range(n_gates):
        angle = start_angle + i * (2 * np.pi / n_gates)
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        z = z_start + i * z_step
        nx, ny = -np.cos(angle), -np.sin(angle)
        gates.append((np.array([x, y, z]), _norm([nx, ny, 0.])))
    return gates


def _helix_tangential(cx, cy, radius, z_start, z_step, n_gates, start_angle=0.0):
    """Helix with CCW tangential normals — drone orbits around center. Approach blend works correctly."""
    gates = []
    for i in range(n_gates):
        angle = start_angle + i * (2 * np.pi / n_gates)
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        z = z_start + i * z_step
        # CCW tangential: normal = (-sin(a), cos(a), 0)
        nx, ny = -np.sin(angle), np.cos(angle)
        gates.append((np.array([x, y, z]), _norm([nx, ny, 0.])))
    return gates


GODTIER_COURSES = [
    # 1. Marathon 50-gate — varied 3D, spacing 12-18m, alt delta ≤5m
    (
        "godtier_marathon_50gate",
        [
            (np.array([ 15.,   0., -3.0]), _norm([1., 0., 0.])),
            (np.array([ 30.,   5., -5.5]), _norm([1., 0., 0.])),
            (np.array([ 45.,  -5., -3.0]), _norm([0.707, -0.707, 0.])),
            (np.array([ 58., -18., -5.5]), _norm([0., -1., 0.])),
            (np.array([ 58., -32., -3.0]), _norm([0.707, -0.707, 0.])),
            (np.array([ 70., -42., -5.5]), _norm([1., 0., 0.])),
            (np.array([ 84., -36., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 94., -24., -5.5]), _norm([0., 1., 0.])),
            (np.array([ 98., -12., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([108.,   0., -5.5]), _norm([1., 0., 0.])),
            (np.array([122.,  -5., -3.0]), _norm([0.707, -0.707, 0.])),
            (np.array([132., -16., -5.5]), _norm([0., -1., 0.])),
            (np.array([126., -28., -3.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([116., -36., -5.5]), _norm([-1., 0., 0.])),
            (np.array([102., -30., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 92., -20., -5.5]), _norm([-1., 0., 0.])),
            (np.array([ 78., -14., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 66.,  -4., -5.5]), _norm([-1., 0., 0.])),
            (np.array([ 52.,   4., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 40.,  14., -5.5]), _norm([0., 1., 0.])),
            (np.array([ 30.,   6., -3.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 18.,  -6., -5.5]), _norm([-1., 0., 0.])),
            (np.array([  4.,   2., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ -8.,  12., -5.5]), _norm([0., 1., 0.])),
            (np.array([  2.,  26., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 14.,  38., -5.5]), _norm([0.707, 0.707, 0.])),
            (np.array([ 28.,  46., -3.0]), _norm([1., 0., 0.])),
            (np.array([ 42.,  40., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([ 52.,  28., -3.0]), _norm([0., -1., 0.])),
            (np.array([ 44.,  14., -5.5]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 30.,   8., -3.0]), _norm([-1., 0., 0.])),
            (np.array([ 16.,  16., -5.5]), _norm([-0.707, 0.707, 0.])),
            (np.array([  4.,  26., -3.0]), _norm([0., 1., 0.])),
            (np.array([ 14.,  40., -5.5]), _norm([0.707, 0.707, 0.])),
            (np.array([ 28.,  50., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 42.,  58., -5.5]), _norm([1., 0., 0.])),
            (np.array([ 56.,  52., -3.0]), _norm([0.707, -0.707, 0.])),
            (np.array([ 64.,  38., -5.5]), _norm([0., -1., 0.])),
            (np.array([ 56.,  24., -3.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 42.,  16., -5.5]), _norm([-1., 0., 0.])),
            (np.array([ 28.,  24., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 18.,  34., -5.5]), _norm([0., 1., 0.])),
            (np.array([ 28.,  48., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 42.,  56., -5.5]), _norm([0.707, 0.707, 0.])),
            (np.array([ 56.,  62., -3.0]), _norm([1., 0., 0.])),
            (np.array([ 70.,  56., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([ 78.,  42., -3.0]), _norm([0., -1., 0.])),
            (np.array([ 70.,  28., -5.5]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 56.,  22., -3.0]), _norm([-1., 0., 0.])),
            (np.array([ 42.,  30., -5.0]), _norm([-0.707, 0.707, 0.])),
        ],
        15700, 16700,
    ),
    # 2. Diagonal zigzag 24: alternating 45° normals, 12m spacing — proven navigable at 30m/s
    (
        "diagonal_zigzag_24",
        [
            (np.array([ 12.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 22.,  10., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([ 32.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 42.,  10., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([ 52.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 62.,  10., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([ 72.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 82.,  10., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([ 92.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([102.,  10., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([112.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([122.,  10., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([132.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([142.,  10., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([152.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([162.,  10., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([172.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([182.,  10., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([192.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([202.,  10., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([212.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([222.,  10., -5.5]), _norm([0.707, -0.707, 0.])),
            (np.array([232.,   0., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([242.,  10., -5.5]), _norm([0.707, -0.707, 0.])),
        ],
        15701, 16701,
    ),
    # 3. Altitude gauntlet: 20 gates, max 5m altitude drops/climbs, 14m spacing
    (
        "altitude_gauntlet_20",
        [
            (np.array([ 14.,  0., -2.0]), _norm([1., 0., 0.])),
            (np.array([ 28.,  0., -7.0]), _norm([1., 0., 0.])),  # -5m
            (np.array([ 42.,  0., -2.0]), _norm([1., 0., 0.])),  # +5m
            (np.array([ 56.,  0., -7.0]), _norm([1., 0., 0.])),  # -5m
            (np.array([ 70.,  0., -2.0]), _norm([1., 0., 0.])),  # +5m
            (np.array([ 80.,  8., -5.5]), _norm([0.707, 0.707, 0.])),
            (np.array([ 80., 18., -2.0]), _norm([0., 1., 0.])),
            (np.array([ 70., 26., -7.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([ 58., 20., -2.0]), _norm([-1., 0., 0.])),
            (np.array([ 46., 12., -7.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 38.,  2., -2.0]), _norm([0., -1., 0.])),
            (np.array([ 28., -8., -7.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([ 16.,  0., -2.0]), _norm([-1., 0., 0.])),
            (np.array([  4., -8., -7.0]), _norm([0., -1., 0.])),
            (np.array([ 12.,-16., -2.0]), _norm([0.707, -0.707, 0.])),
            (np.array([ 22., -8., -7.0]), _norm([1., 0., 0.])),
            (np.array([ 34.,  0., -2.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 42.,  8., -7.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 50., 14., -2.0]), _norm([0.707, 0.707, 0.])),
            (np.array([ 58., 20., -5.5]), _norm([1., 0., 0.])),
        ],
        15702, 16702,
    ),
    # 4. Double helix: two nested helices, radius 10m/14m, inward normals, 20 gates
    (
        "double_helix_nested_20",
        (
            _helix_outward(cx=0., cy=0., radius=14., z_start=-3.0, z_step=-0.5, n_gates=10, start_angle=0.)
            + _helix_outward(cx=0., cy=0., radius=10., z_start=-8.0, z_step=0.5, n_gates=10, start_angle=np.pi)
        ),
        15703, 16703,
    ),
    # 5. 3D chaos 30-gate: varied normals, 10m spacing min, alt delta ≤5m
    (
        "chaos_3d_30gate",
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
            (np.array([   4.,  24., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([  16.,  32., -5.5]), _norm([0.707, 0.707, 0.])),
            (np.array([  26.,  24., -3.0]), _norm([0.707, -0.707, 0.])),
            (np.array([  34.,  12., -5.5]), _norm([0., -1., 0.])),
            (np.array([  24.,   2., -3.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([  12.,  -8., -5.5]), _norm([-1., 0., 0.])),
            (np.array([   0.,   2., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([  -8.,  12., -5.5]), _norm([0., 1., 0.])),
            (np.array([   2.,  22., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([  14.,  30., -5.5]), _norm([0.707, 0.707, 0.])),
        ],
        15704, 16704,
    ),
    # 6. Tight 8m straight: 25 gates, minimum viable spacing, alternating alt
    (
        "tight_8m_25gate",
        [
            (np.array([ 8.*(i+1), 0., -3.0 if i % 2 == 0 else -5.5]), np.array([1., 0., 0.]))
            for i in range(25)
        ],
        15705, 16705,
    ),
    # 7. Grand circuit 36: extended chaos — 36 unique gates, 10m spacing, mixed 3D
    (
        "grand_circuit_36gate",
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
            (np.array([   4.,  24., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([  16.,  32., -5.5]), _norm([0.707, 0.707, 0.])),
            (np.array([  26.,  24., -3.0]), _norm([0.707, -0.707, 0.])),
            (np.array([  34.,  12., -5.5]), _norm([0., -1., 0.])),
            (np.array([  24.,   2., -3.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([  12.,  -8., -5.5]), _norm([-1., 0., 0.])),
            (np.array([   0.,   2., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([  -8.,  12., -5.5]), _norm([0., 1., 0.])),
            (np.array([   2.,  22., -3.0]), _norm([0.707, 0.707, 0.])),
            (np.array([  14.,  30., -5.5]), _norm([0.707, 0.707, 0.])),
            (np.array([  24.,  22., -3.0]), _norm([0.707, -0.707, 0.])),
            (np.array([  32.,  10., -5.5]), _norm([0., -1., 0.])),
            (np.array([  22.,   0., -3.0]), _norm([-0.707, -0.707, 0.])),
            (np.array([  10., -10., -5.5]), _norm([-1., 0., 0.])),
            (np.array([  -2.,   0., -3.0]), _norm([-0.707, 0.707, 0.])),
            (np.array([  -8.,  10., -5.5]), _norm([0., 1., 0.])),
        ],
        15706, 16706,
    ),
    # 8. Triple helix 24: 3×8-gate helixes at different centers — proven 8-gate pattern ×3
    (
        "triple_helix_24",
        (
            _helix_outward(cx=0., cy=0., radius=12., z_start=-3.0, z_step=-0.5, n_gates=8, start_angle=0.)
            + _helix_outward(cx=35., cy=0., radius=12., z_start=-7.0, z_step=0.5, n_gates=8, start_angle=np.pi)
            + _helix_outward(cx=70., cy=0., radius=12., z_start=-3.0, z_step=-0.5, n_gates=8, start_angle=0.)
        ),
        15807, 16807,
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
            ev.wait(timeout=10.)  # give server time to update res before reading it
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
        if wp is None: wp = WaypointNED(pos=st.pos_ned.copy(), vel=np.zeros(3), yaw=st.yaw, time=el)
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
    print(f"\nGODTIER COURSE BATTERY — max_speed={_MAX_SPEED} m/s")
    print("=" * 65)
    results = []
    for name, defs, mp, vp in GODTIER_COURSES:
        defs_list = list(defs)
        print(f"  {name} ({len(defs_list)} gates)...", flush=True, end=" ")
        r = run_course(name, defs_list, mp, vp)
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
