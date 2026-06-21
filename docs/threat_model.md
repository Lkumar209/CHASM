# Threat Model

## Scope

CHASM is a **defensive alignment research** project in the model-organisms tradition. We study detection of intent-enactment divergence in synthetic, controlled settings. No real-world attack capability is implemented or documented.

## Threat scenario

A capable LLM agent is given a legitimate primary task. Unknown to the overseer, the agent also pursues a covert secondary objective: touching a canary file, emitting a canary token, deviating from a declared tool whitelist, or (in the multi-agent track) passing a covert bit to a colluding partner agent.

The overseer has access to the agent's stated reasoning and its enacted tool-call trace, but not to the agent's internal weights (for black-box monitors) or both (for white-box probes).

## Attacker model (Breaker)

- **Goal**: enact the covert objective while keeping monitor score low.
- **Capabilities**: can modify the agent's system prompt and few-shot context; can do best-of-N sampling over responses; can apply lightweight preference optimization on the stated channel only (gated).
- **Constraint**: covert action must still actually be enacted (verified by sandbox ground truth) — the benchmark discards trajectories where evasion was achieved by simply not performing the covert action.
- **Out of scope**: modifying model weights directly, exploiting monitor implementation bugs, or generating any real-world attack content.

## Monitor threat surface

Each monitor has a different threat surface:

| Monitor | What an attacker targets |
|---------|--------------------------|
| `cot_judge` | Sanitize the stated channel |
| `action_judge` | Blend covert actions among benign ones |
| `crossguard` | Make stated plan consistent with a false enacted story |
| `probe` | Internal activation patterns (hard without weight access) |
| `stego` | Statistical detectability of inter-agent covert channel |

## Situating against related work

- **AI control / trusted monitoring** [CITE: verify]: Frameworks for deploying AI systems with oversight monitors. CHASM contributes a detection benchmark and adaptive-robustness measurement.
- **CoT monitorability** [CITE: verify]: Work showing that chain-of-thought reasoning can be decoupled from actual computation. CHASM measures how fast this decoupling enables evasion.
- **Evaluation awareness** [CITE: verify]: Evidence that agents behave differently when told they are monitored. CHASM includes an eval-awareness toggle and reports the detection delta.
- **Activation probes for deception** [CITE: verify]: Linear probes for detecting deceptive intent in activations. CHASM includes a probe monitor and a layer sweep.

## Limitations

See [Limitations](limitations.md) for a full discussion. Key limitations: sandbox realism, small models, prompt-based vs trained evasion, synthetic covert objectives, single-GPU scale.
