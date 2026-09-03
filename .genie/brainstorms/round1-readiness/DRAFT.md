# DRAFT — Round 1 Readiness (DCL Virtual Qualifier 1)

| Field | Value |
|-------|-------|
| **Slug** | `round1-readiness` |
| **Date** | 2026-05-17 |
| **Deadline** | 2026-05-23 (6 days) |
| **WRS** | 40/100 |
| **Status** | Simmering |
| **DCL sim** | NOT RELEASED — we have only spec VADR-TS-002. Forced strategy: build a MAVLink2/UDP mock faithful to the spec and run the autonomy against it. When the official sim ships, swapping the endpoint must be drop-in. |

## Newly incorporated context

### Official spec VADR-TS-002 Issue 00.02 (2026-05-08)

| Area | Actual specification | Our current code | Gap |
|------|----------------------|------------------|-----|
| **Protocol** | MAVLink2 over UDP, MAVSDK-compatible | `gym_env`, an internal Python interface | CRITICAL — no MAVLink layer |
| **Commands** | `SET_POSITION_TARGET_LOCAL_NED` or `SET_ATTITUDE_TARGET` | Thrust ∈ [0,1] + direct pitch/roll rates | CRITICAL — interface does not match |
| **Coordinates** | NED (X north, Y east, Z **down**) | Implicit ENU (Z up) | LARGE — broad refactor |
| **State** | No GPS, no global position. Only `ATTITUDE` + `HIGHRES_IMU` + vision | `obs["position"]` read straight from the sim | CRITICAL — no state estimation |
| **Telemetry** | attitude, orientation, linear velocities, status flags | full synthetic state | Medium |
| **Camera** | 640×360 @ 30 Hz, pinhole, cx,cy=320,180, fx,fy=320,320, VFoV 90°, tilt **+20° up** in body frame | Sim returns blank images | CRITICAL — no vision training |
| **Vision stream** | UDP port 5600, 24-byte header + JPEG chunks | N/A | Parser to implement |
| **Physics** | Rigid body 120 Hz, thrust + drag + gravity + collision | Simplified model at 50 Hz | Medium |
| **Timing** | Command rate < 100 Hz, heartbeat ≥ 2 Hz | Internal loop @ 50 Hz | OK |
| **Gates** | Outer 2.7×2.7×0.26 m, inner opening 1.5×1.5 m | Geometry assumed, no official dimensions | Small — calibrate |
| **Drone** | 280×280×160 mm chassis | Generic geometry | Small |
| **OS** | Windows 11 required (Linux **not** supported) | Developing on macOS | CRITICAL — need a test machine |
| **Python** | 3.14.2 validated (free choice) | 3.10+ | Small — validate 3.14 |
| **Internet** | Active connection required (anti-cheat) | N/A | Logistical |
| **VQ1** | < 10 gates, max 8 min, focus on **completion** | OK (prepared for 10) | OK |
| **VQ2** | < 20 gates, complex environment, fastest time wins | — | Phase 2 |

### Site updates (consolidated 2026-02-09 → 2026-05-08)
- IP retained by the teams. No entry fee.
- Multiple parallel instances allowed (scale testing).
- Full-time employees of partners (Anduril/DCL/Neros) ineligible.
- Roster updates permitted after VQ1 begins.
- Minimum hardware: i5-10400F, RTX 2060 Super, 16 GB RAM, 60 GB.

## Strategic implications

1. **The internal stack is structurally misaligned.** The simplified `gym_env` gave us 100% gate completion against a model that **is not** the official simulator. Those results do not transfer.
2. **The real interface is MAVLink2/UDP** — this needs a new bridge, not an adaptation of the existing one.
3. **There is no absolute position** — we have to do VIO/odometry from IMU + vision, or use the linear-velocity integration the sim provides (ATTITUDE + linear_velocities). Needs discussion.
4. **Vision is mandatory** — without a trained CNN we do not pass a single gate. The real protocol carries no ground-truth gate position.
5. **The +20° camera tilt** changes the gate projection; approach geometry has to account for it.
6. **NED**: axis refactor across planning, control and perception.
7. **Windows 11**: we need a machine or VM for real testing. macOS is development only.
8. **Six days for all of this** with a team — parallelizing is mandatory.

## WRS

```
WRS: ████░░░░░░ 20/100
 Problem ✅ | Scope ░ | Decisions ░ | Risks ░ | Criteria ░
```

- **Problem ✅** — Current stack misaligned with the newly released official spec; we need a refactor plus MAVLink integration, a trained vision model and a Windows environment by 2026-05-23 to submit to VQ1.
- **Scope ░** — awaiting a scope decision (full refactor vs. thin adapter vs. hybrid).
- **Decisions ░** — pending: interface strategy, state-estimation approach, vision priority, test environment.
- **Risks ░** — to list once decisions are made.
- **Criteria ░** — to define once scope is set (e.g. "pass 5 gates in the DCL sim in 60 s at 95% reliability").

## Decision history

(empty — awaiting the first Q&A round)

## Next choices to discuss (in order)

1. **Scope:** decompose into parallel sub-projects, or one integrated stack?
2. **MAVLink interface strategy:** native bridge vs. internal simulator + adapter?
3. **State estimation:** our own VIO vs. IMU integration + landmarks vs. relying on what the sim exposes?
4. **Vision:** train from scratch, transfer learning, or a classical detector (the gate's blue)?
5. **Windows environment:** Parallels/UTM VM, physical machine, or cloud (Paperspace)?
6. **Team composition:** how many people, which specialities, how to split the tracks?
