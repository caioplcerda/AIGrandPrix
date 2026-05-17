# AI Grand Prix — Próximos Passos até VQ1 (deadline: 2026-05-23)

> Atualizado em 2026-05-17. Spec oficial: VADR-TS-002 Issue 00.02 (2026-05-08).
> Design completo: `.genie/brainstorms/round1-readiness/DESIGN.md`

## Status atual (2026-05-17)

**Tracks A, B, C — COMPLETOS.** MAVLink-native stack funcionando:
- E2E mock: 5/5 gates, 10.79s, 243 testes passando
- Entry point: `python run_vq1.py --host <ip> --mavlink_port 14550`
- Sim DCL binary: ainda não liberado. Mock é drop-in fiel.

**Falta:** Tracks D (visão CNN), E (integração visão→loop), F (Windows 11).

---

## Tracks Restantes

### Track A — MAVLink Client ✅ DONE
`comms/mavlink_client.py` + `comms/vision_stream.py`. pymavlink 2.4.49. Heartbeat 2Hz, recv ATTITUDE+HIGHRES_IMU, send SET_POSITION_TARGET_LOCAL_NED, vision stream 640×360 reassembler.

---

### Track B — Mock DCL Sim ✅ DONE
`mock_sim/dcl_mock_server.py` + `physics_6dof.py` + `gate_renderer.py`. 120Hz physics, gate pass detection (plane crossing + inner opening check), vision stream UDP 5600 (24-byte header), camera tilt +20°.

**Bugs corrigidos:**
- `highres_imu_encode(id=0)` → fallback sem `id` (MAVLink1 dialect)
- `body_frame_accel()` → inclui aceleração real do veículo (não só gravidade)
- `parse_buffer()` → filtra `None` da lista

---

### Track C — NED Refactor + State Estimation ✅ DONE
`state/state_estimator.py` + `planning/path_planner_ned.py`. NED nativo: X=north, Y=east, Z=down. IMU integration, yaw from ATTITUDE, altitude = -Z. gate_neds → waypoints → POSITION_TARGET.

---

---

### Track D — Vision CNN
**Responsável:** 1-2 pessoas | **Dias:** 18-22

| # | Tarefa | Entrega |
|---|--------|---------|
| D1 | Gerador de dataset sintético: gate 2.7m outer / 1.5m inner (azul escuro, ~RGB 20,40,180) em 640×360, perspectiva correta, câmera tiltada +20°, fundo variado (texturas, gradientes, cor sólida), range de distância 2m-15m, ângulos ±45°, iluminação variada | 5.000+ imagens anotadas (bbox YOLO) |
| D2 | Fallback detector clássico HSV: segmentar azul do gate, bbox da região conectada maior | Detecta gates no mock em <1ms |
| D3 | Treinar YOLOv8n: 80% train / 20% val, 50 epochs, batch 32, aug horizontal flip + brightness | mAP@0.5 >0.85 |
| D4 | Exportar pesos TorchScript + testar inference <15ms em CPU mid-tier | Tempo medido |
| D5 | Pipeline de percepção: recebe frame JPEG 640×360 → detecta gate → output: (cx, cy) pixels + confidence + bbox | Interface limpa |
| D6 | Converter detecção pixel → bearing NED relativo + estimativa de distância (usando largura conhecida do gate 2.7m e fx=320) | (bearing_h, bearing_v, dist_est) em metros |
| D7 | Integrar em `gate_detector.py`: fallback chain = CNN → HSV → None | Detector atualizado |
| D8 | Testes de percepção com hold-out 200 imagens | IoU>0.7 em 85%+ |

**Formula distância:** `dist = (gate_width_m * fx) / bbox_width_px`

---

### Track E — Planner + Controller Adapter
**Responsável:** 1 pessoa | **Dias:** 20-22

> Depende de C (NED, state estimator) e A (cliente MAVLink)

| # | Tarefa | Entrega |
|---|--------|---------|
| E1 | Loop principal: 50Hz, consome state estimator (pos_ned, vel_ned, yaw), envia POSITION_TARGET via cliente A | Loop estável |
| E2 | Gate sequencing: avança para próximo gate quando drone entra em raio 1.5m do centro | Sequência automática |
| E3 | Waypoint NED para POSITION_TARGET: `(x,y,z)` + `(vx,vy,vz)` lookahead + `yaw` alinhado com gate | Mensagem correta |
| E4 | Approach profile: desacelera para 3 m/s a 5m antes do gate, acelera após passar | Parâmetros configuráveis |
| E5 | Heartbeat separado em thread a 2Hz | Nunca cai abaixo de 2Hz |
| E6 | Integrar detecção de visão (D6): bearing → corrigir posição estimada do gate quando CNN confirma | Correção de posição |
| E7 | Hover fallback: se state estimator perde confiança ou cmd loop trava >500ms, enviar hover cmd | Safety wrapper |
| E8 | Benchmark: fechar 5 gates no mock, medir tempo, log de gates passados | Tempo + gates medidos |

---

### Track F — Integração + Windows CI
**Responsável:** 1 pessoa | **Dias:** 21-23

| # | Tarefa | Entrega |
|---|--------|---------|
| F1 | Setup Windows 11 + Python 3.14.2 + venv + instalar deps (pymavlink, torch, ultralytics, opencv, numpy, scipy) | Import sem erros |
| F2 | Clonar repo, rodar mock sim + cliente autonomia no mesmo Windows | Stack sobe |
| F3 | E2E test: autonomia fecha 5 gates no mock, Windows, Python 3.14.2 | Log de gates passados |
| F4 | requirements_vq1.txt: versões fixadas de todas as deps | Reproducível |
| F5 | Script de entrada único: `python run_vq1.py --host <ip> --port <port>` (substitui `dcl_adapter.py` stub) | Drop-in para sim DCL |
| F6 | Benchmark final: 3 seeds × 5 gates, Windows, mock, medir tempo médio | Relatório de performance |
| F7 | Atualizar CLAUDE.md com Phase 6 results | CLAUDE.md atualizado |

---

## Checkpoint — 2026-05-22

**Critério de freeze:**
- [x] 5 gates no mock em ≥1 seed (10.79s) ✅
- [ ] Vision CNN mAP@0.5 > 0.80
- [ ] Integração visão → loop sem erro
- [ ] Windows 11 Python 3.14.2 importa e roda sem erro

Se CNN não ficar pronta: usar HSV fallback já implementado em `gate_detector.py`. Core mínimo já funciona sem visão.

---

## Estrutura atual de arquivos

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
│   ├── gate_detector.py      ⚠️ backend YOLO sem pesos treinados
│   └── cnn_model.py          ⚠️ arquitetura OK, sem pesos
├── control/                  (legacy, gym_env)
├── submission/               (stub aguarda API DCL)
└── run_vq1.py                ✅ DONE — entry point completo
scripts/
└── test_e2e_mock.py          ✅ DONE — E2E: 5/5 gates 10.79s
```

---

## O que NÃO tocar (legacy preservado)

- `simulation/gym_env.py` — mantém 243 testes como regressão
- `submission/entry_point.py` / `dcl_adapter.py` — stub aguarda API oficial
- `control/mpc_controller.py` — preservado para Round 2
- `planning/trajectory_opt.py` — preservado para Round 2

---

## Quick wins se o tempo apertar

| Situação | Fallback |
|----------|----------|
| CNN não converge | HSV detector azul: `cv2.inRange(hsv, (100,80,40), (130,255,255))` |
| State estimation deriva muito | Usar vel_ned integrada só por 10s e resetar com gate pass como âncora |
| Windows 11 sem máquina | Paperspace A4000 Windows ($0.76/hr) ou GeForce NOW Enterprise |
| JPEG stream difícil de reconstituir | Testar com frames inteiros primeiro (mock sem chunking), depois adicionar |

---

## Arquivo de submissão esperado

Quando sim DCL liberar, submissão deve ser:
```bash
python run_vq1.py --host <dcl_sim_ip> --mavlink_port 14550 --vision_port 5600
```
Sem código adicional, sem mudança de lógica — só IP/porta.
