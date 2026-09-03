# Design: Round 1 Readiness — Rebuild for Official Spec VADR-TS-002

| Field | Value |
|-------|-------|
| **Slug** | `round1-readiness` |
| **Date** | 2026-05-17 |
| **Deadline** | 2026-05-23 |
| **WRS** | 100/100 |
| **Spec** | VADR-TS-002 Issue 00.02 (2026-05-08) |

## Problem

The current stack (internal gym_env, ENU, absolute position, direct thrust) is incompatible with the real DCL simulator interface. The simulator uses MAVLink2/UDP, NED without GPS, and mandatory real vision — and the official sim has not been released, so we have only the spec. We need to refactor the stack completely so it is plug-and-play when the binary ships, validating against a faithful mock we build ourselves, with a team of 4-8 people, by 2026-05-23.

## Scope

### IN
- Mock DCL sim: UDP server faithful to VADR-TS-002 (MAVLink2, 640×360 JPEG vision over UDP 5600)
- Python MAVLink2/UDP client (pymavlink): heartbeat, ATTITUDE, HIGHRES_IMU, SET_POSITION_TARGET_LOCAL_NED
- Full NED refactor across the stack (planning, control, perception)
- State estimator: quaternion attitude + linear-velocity integration (no GPS)
- Vision CNN: synthetic dataset of 2.7×2.7 m gates (640×360 camera, +20° tilt), trained YOLOv8n, output pixel→bearing→relative NED position
- Main control loop: 50 Hz, heartbeat 2 Hz, command rate <100 Hz
- Windows 11 environment (Python 3.14.2) validated, plus an end-to-end benchmark

### OUT
- SET_ATTITUDE_TARGET (reserved for Round 2 / aggressive turns)
- RL training pipeline
- VQ2 features (20 gates, complex environment)
- Fine-tuning on official sim data (post-release)
- Deployment on the real DCL sim (depends on the binary)

## Approach

**Mock-first, drop-in swap.** We build our own UDP server that emulates the VADR-TS-002 spec exactly. All development and validation happens against that mock. When the DCL sim is released, changing IP and port should be sufficient.

**6 parallel tracks** for 4-8 people:

| Track | Owner | Days | Deliverable |
|-------|-------|------|-------------|
| A — MAVLink Client | 1-2p | 18-20 | Python client connects, heartbeat loop, receives telemetry, sends POSITION_TARGET |
| B — Mock DCL Sim | 1-2p | 18-21 | UDP server: MAVLink2 pub/sub + 6-DOF at 120 Hz + JPEG vision stream on UDP 5600 |
| C — NED Refactor + State Estimation | 1p | 18-20 | ENU→NED across the stack; state estimator, quaternion + velocity integration, no GPS |
| D — Vision CNN | 1-2p | 18-22 | Synthetic dataset + trained YOLOv8n + pixel→NED gate bearing |
| E — Planner/Controller Adapter | 1p | 20-22 | state→planner→MAVLink POSITION_TARGET; autonomous gate sequencing |
| F — Integration + Windows CI | 1p | 21-23 | End-to-end on the mock (Windows 11, Python 3.14.2); benchmark replacing gym_env |

**Schedule:**

| Date | Milestone |
|------|-----------|
| 2026-05-17 | Kickoff: assign tracks, set up repos |
| 2026-05-18 | Tracks A, B, C, D start |
| 2026-05-20 | A+B integration point: client connects to the mock, heartbeat OK |
| 2026-05-21 | C+E integration: state estimator → planner → POSITION_TARGET on the mock |
| 2026-05-22 | D integration: vision detects gates in the mock's stream |
| 2026-05-23 | Full E2E: drone clears 5 gates on the mock under Windows; FREEZE |

## Decisions

| Decision | Rationale |
|----------|-----------|
| SET_POSITION_TARGET_LOCAL_NED as primary | The DCL sim has a native inner loop; the planner already produces pos+vel+yaw; removes gain tuning. ATTITUDE_TARGET reserved for Round 2 |
| Mock-first (not blind coding) | Without the official sim, all validation would be blind. A spec-derived mock gives real feedback and makes the swap trivial |
| YOLOv8n for vision | Speed/accuracy suitable for gate detection; TorchScript-exportable weights; inference <10 ms on a mid-tier GPU |
| pymavlink over MAVSDK-python | Lower overhead, fine control of MAVLink frames, easier to debug in pure Python |
| Python 3.14.2 target | Spec cites 3.14.2 as validated; develop locally on 3.10, validate 3.14 in track F |
| NED across the whole stack (not an adapter) | A conversion adapter accumulates silent bugs in numerical races; the one-off refactor cost is justified |
| gym_env kept as legacy | Do not delete — it keeps 243 tests passing as a regression suite; the new stack is separate code |

## Risks & Assumptions

| Risk | Severity | Mitigation |
|------|----------|------------|
| Mock diverges from the real DCL sim on release | High | Mock derived directly from the spec; keep a changelog of assumptions; re-test immediately on receiving the binary |
| Vision CNN does not generalize to DCL's rendering | High | Varied synthetic dataset (lighting, backgrounds, angles); classical blue-HSV fallback detector |
| Python 3.14.2 breaks dependencies (torch, opencv, pymavlink) | Medium | Test in track F on day 18; if broken, isolate in a 3.14 venv and pin versions with pip freeze |
| No physical Windows 11 machine available | Medium | Parallels/UTM on Apple Silicon works for development; for real performance, bare metal or a Windows cloud GPU (Paperspace) |
| Six days is tight for trained vision plus E2E | Medium | CNN fallback: blue-gate HSV detector (<30 min to implement) if training does not converge |
| The DCL sim may use a protocol different from the documented one | Low | The spec is official VADR-TS-002 — but record every UDP frame on the first real test for debugging |
| State estimate drifts without GPS | Medium | Use linear velocity from HIGHRES_IMU (integrated, not accumulated) + position reset on arming |

## Success Criteria

- [ ] MAVLink client connects to the mock, holds a 2 Hz heartbeat for 10 min with no drops
- [ ] Client receives ATTITUDE + HIGHRES_IMU at 120 Hz, latency <5 ms
- [ ] Client sends SET_POSITION_TARGET_LOCAL_NED, mock integrates position correctly
- [ ] Vision stream on UDP 5600 received and JPEG correctly reassembled from chunks
- [ ] State estimator produces pos_ned, vel_ned, yaw with <1 m drift over 30 s of hover
- [ ] CNN detects the central gate with IoU>0.7 on synthetic images (200-image hold-out set)
- [ ] Full stack clears 5/5 gates sequentially on the mock without crashing, on 3 distinct seeds
- [ ] Complete run in <8 min (VQ1 max run duration)
- [ ] Runs on Windows 11 Python 3.14.2 with no import errors
- [ ] Swapping endpoint (mock→real DCL) requires only an IP/port change in config
