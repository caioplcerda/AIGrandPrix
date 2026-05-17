# Design: Round 1 Readiness — Rebuild para Spec Oficial VADR-TS-002

| Field | Value |
|-------|-------|
| **Slug** | `round1-readiness` |
| **Date** | 2026-05-17 |
| **Deadline** | 2026-05-23 |
| **WRS** | 100/100 |
| **Spec** | VADR-TS-002 Issue 00.02 (2026-05-08) |

## Problem

Stack atual (gym_env interno, ENU, posição absoluta, thrust direto) é incompatível com a interface real do simulador DCL. O simulador usa MAVLink2/UDP, NED sem GPS, visão real obrigatória, e o sim oficial ainda não foi liberado — temos só a spec. Precisamos refatorar completamente o stack para ser plug-and-play quando o binário liberar, validando contra mock fiel que construímos, com equipe de 4-8 pessoas até 2026-05-23.

## Scope

### IN
- Mock DCL Sim: servidor UDP fiel à VADR-TS-002 (MAVLink2, visão JPEG 640×360 por UDP 5600)
- Cliente MAVLink2/UDP Python (pymavlink): heartbeat, ATTITUDE, HIGHRES_IMU, SET_POSITION_TARGET_LOCAL_NED
- Refator NED completo em todo o stack (planning, control, perception)
- State estimator: quaternion attitude + integração de velocidade linear (sem GPS)
- Vision CNN: dataset sintético gates 2.7×2.7m (câmera 640×360, tilt +20°), YOLOv8n treinado, output pixel→bearing→pos NED relativa
- Loop de controle principal: 50Hz, heartbeat 2Hz, cmd <100Hz
- Ambiente Windows 11 (Python 3.14.2) validado e benchmark e2e

### OUT
- SET_ATTITUDE_TARGET (reservado para Round 2 / curvas agressivas)
- RL training pipeline
- VQ2 features (20 gates, ambiente complexo)
- Fine-tune com dados do sim oficial (pós-liberação)
- Deploy no sim DCL real (depende do binário)

## Approach

**Mock-first, drop-in swap.** Construímos nosso próprio servidor UDP que emula exatamente a spec VADR-TS-002. Todo o desenvolvimento e validação acontece contra esse mock. Quando o sim DCL liberar, troca de IP/porta deve ser suficiente.

**6 tracks paralelos** para 4-8 pessoas:

| Track | Owner | Dias | Entrega |
|-------|-------|------|---------|
| A — MAVLink Client | 1-2p | 18-20 | Cliente Python conecta, heartbeat loop, recebe telemetria, envia POSITION_TARGET |
| B — Mock DCL Sim | 1-2p | 18-21 | Servidor UDP: MAVLink2 pub/sub + 6DOF 120Hz + vision stream JPEG UDP 5600 |
| C — NED Refactor + State Estimation | 1p | 18-20 | ENU→NED em todo o stack; state estimator quat+vel_int sem GPS |
| D — Vision CNN | 1-2p | 18-22 | Dataset sintético + YOLOv8n treinado + pixel→NED gate bearing |
| E — Planner/Controller Adapter | 1p | 20-22 | state→planner→MAVLink POSITION_TARGET; gate sequencing autônomo |
| F — Integração + Windows CI | 1p | 21-23 | E2E no mock (Windows 11, Python 3.14.2); benchmark substituindo gym_env |

**Cronograma:**

| Data | Marco |
|------|-------|
| 2026-05-17 | Kickoff: assign tracks, setup repos |
| 2026-05-18 | Tracks A, B, C, D start |
| 2026-05-20 | A+B integration point: cliente conecta no mock, heartbeat OK |
| 2026-05-21 | C+E integration: state estimator → planner → POSITION_TARGET no mock |
| 2026-05-22 | D integration: visão detecta gates no stream do mock |
| 2026-05-23 | Full E2E: drone fecha 5 gates no mock Windows; FREEZE |

## Decisions

| Decisão | Rationale |
|---------|-----------|
| SET_POSITION_TARGET_LOCAL_NED como primário | Sim DCL tem inner-loop nativo; planner já produz pos+vel+yaw; elimina sintonia de gains; ATTITUDE_TARGET reservado para Round 2 |
| Mock-first (não blind-code) | Sem sim oficial, toda validação seria cega. Mock baseado em spec dá feedback real e torna swap trivial |
| YOLOv8n para visão | Speed/accuracy adequado para gate detection; pesos TorchScript exportáveis; inference <10ms em GPU mid-tier |
| pymavlink sobre MAVSDK-python | Menor overhead, controle fino dos frames MAVLink, mais fácil de debugar em Python puro |
| Python 3.14.2 target | Spec cita 3.14.2 como validado; rodar em 3.10 localmente durante dev, validar 3.14 em track F |
| NED em todo o stack (não adapter) | Adapter de conversão acumula bugs silenciosos em races numéricas; custo de refator 1x justifica |
| gym_env mantido como legacy | Não deletar — mantém 243 testes passando como regressão; novo stack é código separado |

## Risks & Assumptions

| Risk | Severity | Mitigation |
|------|----------|------------|
| Mock diverge do sim DCL real quando liberar | High | Mock gerado diretamente da spec; manter changelog de premissas; re-testar imediatamente ao receber binário |
| Vision CNN não generaliza para rendering do DCL | High | Dataset sintético variado (iluminação, backgrounds, ângulos); fallback detector clássico por cor azul HSV |
| Python 3.14.2 quebra dependências (torch, opencv, pymavlink) | Medium | Testar em track F dia 18; se quebrar, isolar em venv 3.14 e fixar versões com pip freeze |
| Windows 11 sem máquina física disponível | Medium | Parallels/UTM em Apple Silicon funciona para dev; para performance real, baremental ou cloud GPU Windows (paperspace) |
| 6 dias é tight para visão treinada + E2E | Medium | CNN fallback: detector HSV de gate azul (<30 min de implementar) se treino não convergir |
| Sim DCL pode usar protocolo diferente do documentado | Low | Spec é oficial VADR-TS-002 — mas gravar todos os frames UDP no primeiro teste real para debug |
| State estimation deriva sem GPS | Medium | Usar velocidade linear do HIGHRES_IMU (integrada, não acumulada) + reset de posição no arme |

## Success Criteria

- [ ] MAVLink client conecta no mock, mantém heartbeat 2Hz por 10 min sem drop
- [ ] Cliente recebe ATTITUDE + HIGHRES_IMU a 120Hz, latência <5ms
- [ ] Cliente envia SET_POSITION_TARGET_LOCAL_NED, mock integra posição corretamente
- [ ] Vision stream UDP 5600 recebido e JPEG reconstituído corretamente de chunks
- [ ] State estimator produz pos_ned, vel_ned, yaw com drift <1m em 30s de voo hover
- [ ] CNN detecta gate central com IoU>0.7 em imagens sintéticas (hold-out set 200 imagens)
- [ ] Stack completo fecha 5/5 gates sequencialmente no mock sem crash em 3 seeds distintas
- [ ] Run completo em <8 min (max run duration do VQ1)
- [ ] Roda em Windows 11 Python 3.14.2 sem erros de import
- [ ] Swap de endpoint (mock→DCL real) requer apenas mudança de IP/porta em config
