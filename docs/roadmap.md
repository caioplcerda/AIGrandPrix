# AI Grand Prix — Roadmap to VQ1 (deadline: 2026-05-23)

> Updated 2026-05-17. Official spec: VADR-TS-002 Issue 00.02 (2026-05-08), not redistributed here.
> Full design: `.genie/brainstorms/round1-readiness/DESIGN.md`

## Status as of 2026-05-17

**Tracks A, B, C — COMPLETE.** MAVLink-native stack working:
- E2E mock: 5/5 gates, 10.79s, 243 tests passing
- Entry point: `python run_vq1.py --host <ip> --mavlink_port 14550`
- DCL sim binary: not yet released. The mock is a faithful drop-in for the documented interface.

**Remaining:** Tracks D (vision CNN), E (vision→loop integration), F (Windows 11).

---

## Remaining Tracks

### Track A — MAVLink Client ✅ DONE
`comms/mavlink_client.py` + `comms/vision_stream.py`. pymavlink 2.4.49. Heartbeat 2Hz, recv ATTITUDE+HIGHRES_IMU, send SET_POSITION_TARGET_LOCAL_NED, vision stream 640×360 reassembler.

---

### Track B — Mock DCL Sim ✅ DONE
`mock_sim/dcl_mock_server.py` + `physics_6dof.py` + `gate_renderer.py`. 120Hz physics, gate pass detection (plane crossing + inner opening check), vision stream UDP 5600 (24-byte header), camera tilt +20°.

**Bugs corrigidos:**
- `highres_imu_encode(id=0)` → fallback sem `id` (MAVLink1 dialect)
- `body_frame_accel()` → includes true vehicle acceleration, not gravity alone
- `parse_buffer()` → filtra `None` da lista

---

### Track C — NED Refactor + State Estimation ✅ DONE
`state/state_estimator.py` + `planning/path_planner_ned.py`. Native NED: X=north, Y=east, Z=down. IMU integration, yaw from ATTITUDE, altitude = -Z. gate_neds → waypoints → POSITION_TARGET.

---

---

### Track D — Vision CNN
**Owners:** 1-2 people | **Days:** 18-22

| # | Task | Deliverable |
|---|--------|---------|
| D1 | Synthetic dataset generator: gate 2.7 m outer / 1.5 m inner (dark blue, ~RGB 20,40,180) at 640×360, correct perspective, camera tilted +20°, varied backgrounds (textures, gradients, solid colour), distance range 2-15 m, angles ±45°, varied lighting | 5,000+ annotated images (YOLO bbox) |
| D2 | Classical HSV fallback detector: segment the gate's blue, bbox of the largest connected region | Detects gates in the mock in <1 ms |
| D3 | Train YOLOv8n: 80% train / 20% val, 50 epochs, batch 32, augmentation horizontal flip + brightness | mAP@0.5 >0.85 |
| D4 | Export TorchScript weights, test inference <15 ms on a mid-tier CPU | Measured time |
| D5 | Perception pipeline: take a 640×360 JPEG frame → detect gate → output (cx, cy) pixels + confidence + bbox | Clean interface |
| D6 | Convert pixel detection → relative NED bearing + distance estimate (using the known 2.7 m gate width and fx=320) | (bearing_h, bearing_v, dist_est) in metres |
| D7 | Integrate into `gate_detector.py`: fallback chain = CNN → HSV → None | Detector updated |
| D8 | Perception tests on a 200-image hold-out | IoU>0.7 on 85%+ |

**Distance formula:** `dist = (gate_width_m * fx) / bbox_width_px`

---

### Track E — Planner + Controller Adapter
**Owners:** 1 person | **Days:** 20-22

> Depends on C (NED, state estimator) and A (MAVLink client)

| # | Task | Deliverable |
|---|--------|---------|
| E1 | Main loop: 50 Hz, consumes the state estimator (pos_ned, vel_ned, yaw), sends POSITION_TARGET via the Track A client | Stable loop |
| E2 | Gate sequencing: advance to the next gate once the drone is within 1.5 m of centre | Automatic sequencing |
| E3 | NED waypoint → POSITION_TARGET: `(x,y,z)` + `(vx,vy,vz)` lookahead + `yaw` aligned with the gate | Correct message |
| E4 | Approach profile: decelerate to 3 m/s at 5 m before the gate, accelerate after passing | Configurable parameters |
| E5 | Heartbeat on its own thread at 2 Hz | Never drops below 2 Hz |
| E6 | Integrate vision detection (D6): bearing → correct the estimated gate position when the CNN confirms | Position correction |
| E7 | Hover fallback: if the state estimator loses confidence or the command loop stalls >500 ms, send a hover command | Safety wrapper |
| E8 | Benchmark: clear 5 gates in the mock, measure time, log gates passed | Time + gates measured |

---

### Track F — Integration + Windows CI
**Owners:** 1 person | **Days:** 21-23

| # | Task | Deliverable |
|---|--------|---------|
| F1 | Set up Windows 11 + Python 3.14.2 + venv + install deps (pymavlink, torch, ultralytics, opencv, numpy, scipy) | Imports without errors |
| F2 | Clone the repo, run the mock sim + autonomy client on the same Windows machine | Stack comes up |
| F3 | E2E test: autonomy clears 5 gates in the mock on Windows, Python 3.14.2 | Log of gates passed |
| F4 | requirements_vq1.txt: pinned versions for every dep | Reproducible |
| F5 | Single entry point: `python run_vq1.py --host <ip> --port <port>` (replaces the `dcl_adapter.py` stub) | Drop-in for the DCL sim |
| F6 | Final benchmark: 3 seeds × 5 gates, Windows, mock, mean time | Performance report |
| F7 | Update CLAUDE.md with Phase 6 results | CLAUDE.md updated |

---

## Checkpoint — 2026-05-22

**Freeze criteria:**
- [x] 5 gates in the mock on ≥1 seed (10.79 s) ✅
- [ ] Vision CNN mAP@0.5 > 0.80
- [ ] Vision → loop integration with no errors
- [ ] Windows 11 Python 3.14.2 imports and runs without error

If the CNN is not ready, fall back to the HSV detector already implemented in `gate_detector.py`. The minimum core already works without vision.

---

## Current file structure

```
src/aigrandprix/
├── comms/                    ✅ DONE
│   ├── mavlink_client.py
│   └── vision_stream.py
├── mock_sim/                 ✅ DONE
│   ├── dcl_mock_server.py
│   ├── physics_6dof.py
│   └── gate_renderer.py
├── state/                    ✅ DONE
│   └── state_estimator.py
├── planning/
│   ├── path_planner.py       (legacy, gym_env)
│   └── path_planner_ned.py   ✅ DONE — NED wrapper
├── perception/
│   ├── data_generator.py     ✅ DONE — 640×360, tilt +20°, COCO format
│   ├── gate_detector.py      ⚠️ YOLO backend, no trained weights yet
│   └── cnn_model.py          ⚠️ architecture OK, no weights
├── control/                  (legacy, gym_env)
├── submission/               (stub awaiting DCL API)
└── run_vq1.py                ✅ DONE — entry point completo
scripts/
└── test_e2e_mock.py          ✅ DONE — E2E: 5/5 gates 10.79s
```

---

## Do NOT touch (preserved legacy)

- `simulation/gym_env.py` — keeps the 243 tests as a regression suite
- `submission/entry_point.py` / `dcl_adapter.py` — stub awaiting the official API
- `control/mpc_controller.py` — preserved for Round 2
- `planning/trajectory_opt.py` — preserved for Round 2

---

## Quick wins if time runs short

| Situation | Fallback |
|----------|----------|
| CNN fails to converge | Blue HSV detector: `cv2.inRange(hsv, (100,80,40), (130,255,255))` |
| State estimate drifts badly | Integrate vel_ned for 10 s only, reset using a gate pass as the anchor |
| No Windows 11 machine | Paperspace A4000 Windows ($0.76/hr) or GeForce NOW Enterprise |
| JPEG stream hard to reassemble | Test with whole frames first (mock without chunking), then add chunking |

---

## Expected submission form

Once the DCL sim is released, submission should be:
```bash
python run_vq1.py --host <dcl_sim_ip> --mavlink_port 14550 --vision_port 5600
```
No extra code, no logic change — only IP/port.
