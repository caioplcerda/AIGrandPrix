# AI Grand Prix — Próximos Passos até VQ1 (deadline: 2026-05-23)

> Gerado em 2026-05-17. Baseado na spec oficial VADR-TS-002 Issue 00.02 (2026-05-08).
> Design completo: `.genie/brainstorms/round1-readiness/DESIGN.md`

## Contexto crítico

O stack interno (gym_env, ENU, posição absoluta, thrust [0,1]) **não é** a interface real do simulador DCL.
A interface real é MAVLink2/UDP + NED sem GPS + visão obrigatória. O sim DCL ainda não foi liberado — temos só a spec.
Estratégia: construir mock fiel + stack plug-and-play. Quando binário liberar, troca de endpoint = drop-in.

---

## 6 Tracks Paralelos — Atribuir hoje (2026-05-17)

### Track A — MAVLink Client
**Responsável:** 1-2 pessoas | **Dias:** 18-20

| # | Tarefa | Entrega |
|---|--------|---------|
| A1 | Instalar `pymavlink` + script de conexão UDP básica | Conecta, imprime HEARTBEAT |
| A2 | Heartbeat loop (HEARTBEAT em 2Hz) | Loop estável por 10 min |
| A3 | Receber e parsear ATTITUDE (quat) + HIGHRES_IMU (accel, gyro, vel) | Log com timestamp |
| A4 | Receber TIMESYNC, responder para sync de clock | Clock sync OK |
| A5 | Enviar SET_POSITION_TARGET_LOCAL_NED (pos + vel + yaw NED) | Mock confirma recebimento |
| A6 | Receber vision stream UDP porta 5600, reconstituir JPEG de chunks (header 24B) | Frames salvos em disco |
| A7 | Testes unitários + integration test contra mock | pytest pass |

**Referências:**
- Transport: UDP. MAVLink2. Dialect: common.
- HEARTBEAT: type=6 (GCS), autopilot=8 (invalid)
- HIGHRES_IMU: xacc, yacc, zacc (m/s²), xgyro, ygyro, zgyro, xmag, ymag, zmag, abs_pressure, diff_pressure, pressure_alt, temperature, fields_updated, id
- Vision stream: port 5600, header (frame_id uint32, chunk_id uint16, total_chunks uint16, jpeg_size uint32, payload_size uint32, sim_time_ns uint64) = 24 bytes, little-endian

---

### Track B — Mock DCL Sim
**Responsável:** 1-2 pessoas | **Dias:** 18-21

| # | Tarefa | Entrega |
|---|--------|---------|
| B1 | Servidor UDP Python: publica HEARTBEAT a 1Hz | Cliente A1 conecta |
| B2 | Publica ATTITUDE (quaternion) + HIGHRES_IMU a 120Hz com física rígida simples (6DOF: pos, vel, quat, gyro) | A3 recebe corretamente |
| B3 | Aceita SET_POSITION_TARGET_LOCAL_NED, integra posição usando inner P-controller simples | Drone move para waypoint no log |
| B4 | Detecta colisão com gates (cilindro/caixa NED), emite evento gate_passed | Gate pass detectado |
| B5 | Gera frames JPEG 640×360 com gate renderizado via OpenCV (retângulo azul perspectiva simples baseada em pos relativa drone→gate) | Frame visível e correto |
| B6 | Publica vision stream no UDP 5600 com header VADR-TS-002, fatiado em chunks | Track A6 reconstitui |
| B7 | Tilt de câmera +20° aplicado na projeção de gate na imagem | Gate aparece corretamente posicionado |
| B8 | Configuração: arquivo YAML com lista de gates NED (pos + normal) | Qualquer curso carregável |

**Notas:**
- Não precisa ser física perfeita — precisa ser fiel ao protocolo (mensagens, timing, frames)
- inner-loop no mock: P controller simples (Kp=1.5 pos, Kd=1.0 vel), dt=1/120s
- Câmera: pinhole, fx=fy=320, cx=320, cy=180, resolução 640×360, tilt +20° = Ry(-20°) no body frame

---

### Track C — NED Refactor + State Estimation
**Responsável:** 1 pessoa | **Dias:** 18-20

| # | Tarefa | Entrega |
|---|--------|---------|
| C1 | Mapear todos os locais com eixos ENU em planning/, control/, perception/ | Lista de arquivos:linhas |
| C2 | Refatorar `path_planner.py`: eixos NED (X north, Y east, Z down) | Planner aceita e produz NED |
| C3 | Refatorar `drone_controller.py`: NED. Z positivo = down (ascender = Z negativo) | Controller NED |
| C4 | Refatorar `state_estimator.py`: consumir quat ATTITUDE + vel_ned de HIGHRES_IMU | Novo state estimator |
| C5 | Posição: integrar vel_ned (HIGHRES_IMU linear_velocity) desde arme. Origem = (0,0,0) NED | drift <1m em 30s hover |
| C6 | Yaw: extrair de quaternion ATTITUDE | yaw em rad, NED convention |
| C7 | Formato de gate: `GateNED(pos_ned: np.ndarray, normal_ned: np.ndarray, width: float=1.5, height: float=1.5)` | Dataclass NED |
| C8 | Atualizar configs/default.yaml com parâmetros NED | Config válida |
| C9 | Todos os testes existentes atualizados para NED | pytest pass |

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

## Dependências entre Tracks

```
A (MAVLink Client) ←────────────────── B (Mock Sim)
         ↓                                    ↓
C (NED + State) ──→ E (Planner Adapter) ──→ F (Integração)
         ↓                  ↑
D (Vision CNN) ─────────────┘
```

- A e B desenvolvem simultaneamente, integram no dia 20
- C e D independentes, rodam desde dia 18
- E começa quando C e A estão prontos (dia 20)
- F começa quando tudo está integrado (dia 21)

---

## Ponto de integração — 2026-05-20

**Critério de go/no-go:**
- [ ] A5: cliente envia POSITION_TARGET no mock
- [ ] B3: mock move drone para waypoint
- [ ] C4+C5: state estimator produz pos_ned válida

Se não OK em 20/05 AM: re-priorizar para apenas fechar 1 gate com fallback HSV (sem CNN) e sem state estimator (usar pos do mock via campo reservado se houver). Pragmatismo primeiro.

---

## Ponto de integração — 2026-05-22

**Critério de freeze:**
- [ ] 5 gates fechados no mock em ≥2 seeds
- [ ] Run completo <8 min
- [ ] Windows 11 Python 3.14.2 importa e roda sem erro

Se não OK: cortar visão CNN (usar HSV), cortar approach profile (ir direto ao gate), cortar gain scheduling. Core = MAVLink client + state NED + planner simples + detector HSV.

---

## Estrutura de arquivos novos

```
src/aigrandprix/
├── comms/                    # NOVO — MAVLink interface
│   ├── mavlink_client.py     # pymavlink UDP client, heartbeat, recv/send
│   └── vision_stream.py      # UDP 5600 JPEG chunk reassembler
├── mock_sim/                 # NOVO — Mock DCL Sim
│   ├── dcl_mock_server.py    # Servidor UDP MAVLink2 + vision
│   ├── physics_6dof.py       # 6DOF integrator 120Hz
│   └── gate_renderer.py      # OpenCV gate rendering com projeção perspectiva
├── perception/
│   ├── gate_detector.py      # ATUALIZAR — CNN + HSV fallback, output bearing NED
│   ├── cnn_model.py          # ATUALIZAR — YOLOv8n weights
│   └── data_generator.py     # ATUALIZAR — gates 2.7m, tilt +20°, NED
├── planning/
│   └── path_planner.py       # REFATORAR — NED
├── control/
│   └── drone_controller.py   # REFATORAR — NED + POSITION_TARGET output
├── state/                    # NOVO (ou mover de perception/)
│   └── state_estimator.py    # quat + vel_ned integração, sem GPS
└── run_vq1.py                # NOVO — entry point único para submissão
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
