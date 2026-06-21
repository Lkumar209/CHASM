# CHASM: Cross-cHannel Agent Safety Monitoring

_Paper-style writeup. All numeric placeholders (`{{...}}`) are filled from `results/metrics.json` by `scripts/fill_paper.py` at M11. Never hand-type a number._

## Abstract

We present CHASM, a framework for detecting stated-vs-enacted divergence in tool-using LLM agents. Our proposed monitor, CrossGuard, checks consistency between an agent's stated reasoning and its enacted tool-call trace. We benchmark CrossGuard and five baselines on RIFT-Bench, a new benchmark of {{N_TASK_FAMILIES}} task families spanning {{N_CONDITIONS}} induction conditions. CrossGuard achieves AUROC {{CROSSGUARD_AUROC_ALIGNED}} at zero evasion pressure and degrades to {{CROSSGUARD_AUROC_MAX_PRESSURE}} at maximum non-adaptive pressure, compared to {{COT_JUDGE_AUROC_MAX_PRESSURE}} for the CoT-only baseline (H3 verdict: {{H3_VERDICT}}). Under adaptive red-teaming (Breaker), CrossGuard's steady-state evasion rate is {{CROSSGUARD_EVASION_RATE}} vs {{BEST_BASELINE_EVASION_RATE}} for the best baseline (H4 verdict: {{H4_VERDICT}}). All results are reported with bootstrap 95% CIs.

## 1. Introduction

_To be written at M11._

## 2. Related Work

_All citations below are marked [CITE: verify] until confirmed against actual papers._

- AI control and trusted monitoring [CITE: verify]
- CoT monitorability and faithfulness [CITE: verify]
- Evaluation awareness in agentic settings [CITE: verify]
- Activation probes for deception detection [CITE: verify]
- Cost-constrained monitoring cascades [CITE: verify]
- Multi-agent collusion and steganographic channels [CITE: verify]

## 3. RIFT-Bench

_To be written at M1–M4._

## 4. CrossGuard

_To be written at M5._

## 5. Experiments

### 5.1 Non-adaptive robustness (RQ1, H2, H3)

_Figure 1: Robustness curve — AUROC vs evasion pressure level, per monitor, with bootstrap 95% CIs._

### 5.2 Adaptive robustness (RQ2, H4)

_Figure 2: Adaptive-robustness frontier — evasion success rate vs attacker compute, per monitor._

### 5.3 Cost-detection Pareto (RQ4, H5)

_Figure 3: Detection vs escalation fraction and vs compute/$._

### 5.4 Transfer (RQ5, H6)

_Figure 4: Zero-shot detection on transfer task families and held-out evasion strategies._

### 5.5 Multi-agent collusion (RQ3, H7)

_Table: judge vs consistency vs stego/probe detection of covert collusion._

### 5.6 Online detection (H8)

_Distribution of first-flag step / covert-action step; fraction caught before completion._

### 5.7 Ablations

_Remove each CrossGuard divergence type; rule-based vs LLM typing; probe layer sweep; ensemble dropout._

## 6. Hypothesis verdicts

| Hypothesis | Verdict | Key number |
|------------|---------|------------|
| H1 | {{H1_VERDICT}} | AUROC: {{H1_AUROC_MIN}} |
| H2 | {{H2_VERDICT}} | Drop: {{H2_AUROC_DROP}} |
| H3 | {{H3_VERDICT}} | Effect size: {{H3_EFFECT_SIZE}}, CI: {{H3_CI}} |
| H4 | {{H4_VERDICT}} | Evasion delta: {{H4_EVASION_DELTA}} |
| H5 | {{H5_VERDICT}} | Gap closed: {{H5_GAP_CLOSED}} at {{H5_ESCALATION_RATE}} escalation |
| H6 | {{H6_VERDICT}} | Transfer AUROC: {{H6_TRANSFER_AUROC}} |
| H7 | {{H7_VERDICT}} | Judge AUROC: {{H7_JUDGE_AUROC}} vs probe: {{H7_PROBE_AUROC}} |
| H8 | {{H8_VERDICT}} | Fraction caught early: {{H8_EARLY_FRACTION}} |

## 7. Limitations

See [docs/limitations.md](../docs/limitations.md). Key: sandbox realism, small models (4B–8B), prompt-based evasion only at most levels, synthetic canary objectives, single-GPU scale.

## 8. Conclusion

_To be written at M11._
