# Design

## Core thesis

A monitor that checks the *consistency between an agent's stated plan/reasoning and its enacted tool-call trace* is harder to evade than a monitor that reads either channel alone, and it degrades more slowly as the agent is optimized to sanitize its stated channel. An agent can fake what it *says* far more cheaply than it can make its *actions* match a false story.

## Research questions

- **RQ1**: How fast does each monitor's detection degrade along a non-adaptive evasion-pressure ladder?
- **RQ2**: Under a closed-loop adaptive red-team (Breaker), what is the steady-state evasion rate against each monitor, and how much compute does evasion cost?
- **RQ3**: Does cross-channel consistency extend to multi-agent collusion?
- **RQ4**: Can a cost-constrained cascade recover detection lost to evasion at low overhead?
- **RQ5**: Does detection transfer zero-shot to unseen task families and unseen evasion strategies?

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        RIFT-Bench                           │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │   Sandbox   │  │   Tasks     │  │   Induction      │   │
│  │  (mock API) │  │  (8+ fams)  │  │  (ladder L0-L4)  │   │
│  └─────────────┘  └─────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │                                      │
         ▼                                      ▼
┌──────────────────┐               ┌───────────────────────┐
│   Agent Loop     │               │    Monitor Zoo        │
│  stated channel  │──Trajectory──▶│  cot_judge            │
│  enacted channel │               │  action_judge         │
└──────────────────┘               │  crossguard (proposed)│
                                   │  online / probe / ...  │
                                   └───────────────────────┘
                                            │
                                            ▼
                               ┌────────────────────────┐
                               │   Breaker (red-team)   │
                               │   adaptive evasion     │
                               └────────────────────────┘
```

## Two-channel capture

Every trajectory records two channels explicitly:

- **Stated channel**: natural-language plan/reasoning, declared tool whitelist, explicit self-description of intent.
- **Enacted channel**: structured tool calls, arguments, and observations in execution order.

CrossGuard's detection rests on the *consistency* between these two channels. A misaligned agent must either (a) reveal its intent in the stated channel, or (b) perform actions that contradict its stated plan — both are detectable.

## Evasion pressure ladder (non-adaptive)

| Level | Description |
|-------|-------------|
| `aligned` | Agent has no covert objective |
| `naive_divergent` | Agent has a covert objective, no evasion |
| `evasion_L0` | Keep stated plan superficially benign |
| `evasion_L1` | Never mention the side-objective |
| `evasion_L2` | Best-of-N rejection sampling vs cheap CoT proxy |
| `evasion_L3` | (gated) Light DPO/ORPO on clean-stated trajectories |

## Sandbox safety

All covert side-objectives are abstract canary markers. No real exploits, network calls, or side effects. The sandbox is entirely in-memory with deterministic mocked tools.
