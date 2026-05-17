# DRAFT — Round 1 Readiness (DCL Virtual Qualifier 1)

| Field | Value |
|-------|-------|
| **Slug** | `round1-readiness` |
| **Date** | 2026-05-17 |
| **Deadline** | 2026-05-23 (6 dias) |
| **WRS** | 40/100 |
| **Status** | Simmering |
| **Sim DCL** | NÃO LIBERADO — temos só a spec VADR-TS-002. Estratégia forçada: construir mock MAVLink2/UDP fiel à spec + autonomia rodando contra ele. Quando o sim oficial sair, troca de endpoint deve ser drop-in. |

## Contexto recém-incorporado

### Spec oficial VADR-TS-002 Issue 00.02 (2026-05-08)

| Área | Especificação real | Nosso código atual | Gap |
|------|--------------------|--------------------|-----|
| **Protocolo** | MAVLink2 sobre UDP, MAVSDK-compatible | `gym_env` interface Python interna | CRÍTICO — falta camada MAVLink |
| **Comandos** | `SET_POSITION_TARGET_LOCAL_NED` ou `SET_ATTITUDE_TARGET` | Thrust ∈ [0,1] + pitch/roll rate diretos | CRÍTICO — interface não bate |
| **Coordenadas** | NED (X north, Y east, Z **down**) | ENU implícito (Z up) | GRANDE — refator amplo |
| **Estado** | Sem GPS, sem posição global. Só `ATTITUDE` + `HIGHRES_IMU` + visão | `obs["position"]` lido direto do sim | CRÍTICO — falta state estimation |
| **Telemetria** | attitude, orientation, linear velocities, status flags | full state synthetic | Médio |
| **Câmera** | 640×360 @ 30Hz, pinhole, cx,cy=320,180, fx,fy=320,320, VFoV 90°, tilt **+20° up** body | Sim retorna imagens em branco | CRÍTICO — sem treino de visão |
| **Vision stream** | UDP porta 5600, header 24B + JPEG chunks | N/A | Implementar parser |
| **Physics** | Rigid-body 120Hz, thrust + drag + gravity + collision | Modelo simplificado a 50Hz | Médio |
| **Timing** | Command rate < 100Hz, heartbeat ≥ 2Hz | Loop interno @ 50Hz | OK |
| **Gates** | Outer 2.7×2.7×0.26m, inner opening 1.5×1.5m | Geometria assumida sem dimensões oficiais | Pequeno — calibrar |
| **Drone** | 280×280×160mm chassis | Geometria genérica | Pequeno |
| **OS** | Windows 11 obrigatório (Linux **não** suportado) | Dev em macOS | CRÍTICO — máquina de teste |
| **Python** | 3.14.2 validado (livre escolha) | 3.10+ | Pequeno — validar 3.14 |
| **Internet** | Conexão ativa exigida (anti-cheat) | N/A | Logístico |
| **VQ1** | < 10 gates, max 8 min, foco em **completion** | OK (preparado para 10) | OK |
| **VQ2** | < 20 gates, ambiente complexo, tempo mais rápido vence | — | Fase 2 |

### Updates do site (consolidado 2026-02-09 → 2026-05-08)
- IP retido pelos times. Sem taxa de entrada.
- Múltiplas instâncias paralelas permitidas (escalar testes).
- Funcionários FT de parceiros (Anduril/DCL/Neros) inelegíveis.
- Atualização de roster permitida após VQ1 iniciar.
- Hardware mínimo: i5-10400F, RTX 2060 Super, 16GB RAM, 60GB.

## Implicações estratégicas

1. **O stack interno está estruturalmente desalinhado.** O `gym_env` simplificado nos deu 100% de gate completion em um modelo que **não é** o simulador oficial. Esses resultados não transferem.
2. **A interface real é MAVLink2/UDP** — precisamos de um bridge novo, não adaptar o existente.
3. **Não há posição absoluta** — temos que fazer VIO/odometria de IMU+visão, ou usar a integração de velocidade linear que o sim entrega (ATTITUDE + linear_velocities). Crítico discutir.
4. **Visão é mandatória** — sem CNN treinada não passamos um gate. Não tem ground-truth de gate position no protocolo real.
5. **Tilt da câmera +20°** muda a projeção de gate — geometria de approach precisa considerar.
6. **NED**: refator de eixos no planning, control, perception.
7. **Windows 11**: precisamos de máquina/VM para teste real. macOS é só dev.
8. **6 dias para tudo isso** com equipe — paralelizar é mandatório.

## WRS

```
WRS: ████░░░░░░ 20/100
 Problem ✅ | Scope ░ | Decisions ░ | Risks ░ | Criteria ░
```

- **Problem ✅** — Stack atual desalinhada com spec oficial recém-lançada; precisamos refator + integração MAVLink + visão treinada + ambiente Windows até 2026-05-23 para submeter ao VQ1.
- **Scope ░** — aguardando decisão sobre escopo (refator total vs. adapter fino vs. estratégia híbrida).
- **Decisions ░** — pendente: estratégia de interface, abordagem de state estimation, prioridade visão, ambiente de teste.
- **Risks ░** — listar após decisões.
- **Criteria ░** — definir após escopo (ex: "passar 5 gates em sim DCL em 60s @ 95% reliability").

## Histórico de decisões

(vazio — aguardando primeira rodada de Q&A)

## Próximas escolhas a discutir (ordem)

1. **Escopo:** decompor em sub-projetos paralelos ou stack único integrado?
2. **Estratégia de interface MAVLink:** bridge nativo vs. simulador interno + adapter?
3. **State estimation:** VIO próprio vs. integração de IMU + landmarks vs. depender do que o sim expõe?
4. **Visão:** treino do zero, transfer learning, ou detector clássico (cor azul do gate)?
5. **Ambiente Windows:** VM Parallels/UTM, máquina física, ou cloud (paperspace)?
6. **Composição da equipe:** quantas pessoas, especialidades, como dividir tracks?
