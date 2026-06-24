<img width="1040" height="600" alt="chasm-architecture" src="https://github.com/user-attachments/assets/d4c91e48-7c47-4037-bc51-7ccb7b44560d" /># CHASM

**Cross-cHannel Agent Safety Monitoring**

[![CI](https://github.com/Lkumar209/chasm/actions/workflows/ci.yml/badge.svg)](https://github.com/Lkumar209/chasm/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

CHASM detects when a tool-using LLM agent's *stated* reasoning diverges from its *enacted* actions, and measures how that detection holds up when the agent is actively optimized to hide the gap.

> **The chasm between what an agent says and what it does is the thing we measure and try to close.**

## Architecture

```
RIFT-Bench (benchmark)
  ├── Hermetic mock sandbox (8+ task families)
  ├── Non-adaptive evasion ladder (L0–L4)
  └── Ground-truth canary instrumentation

Agent loop (dual-channel capture)
  ├── Stated channel: plan / reasoning / whitelist
  └── Enacted channel: tool calls + args + observations

Monitor zoo
  ├── cot_judge          (stated-only baseline)
  ├── action_judge       (enacted-only baseline)
  ├── CrossGuard         (proposed: cross-channel consistency)
  ├── online             (streaming; flags mid-trajectory)
  ├── probe              (white-box activation probes)
  ├── ensemble / debate  (learned fusion, cross-exam)
  └── stego              (multi-agent covert channel detector)

Breaker (adaptive red-team)
  └── closed-loop evasion search → adaptive-robustness frontier

Control protocol
  └── trusted/untrusted framing → safety-case report
```
## Core Idea Diagram
<img width="1440" height="724" alt="image" src="https://github.com/user-attachments/assets/db8d8373-8f11-4b5d-b698-0f57ca2fe78c" />

## Quick start

```bash
git clone https://github.com/Lkumar209/chasm && cd chasm
uv sync --extra dev          # install all dev deps (no GPU required)
uv run pytest -m smoke -v    # smoke e2e: ~seconds on CPU, no API key needed
```

With a GPU and local models:
```bash
uv sync --extra local
uv run chasm experiment=full  # full pipeline (default: Qwen3-4B-Instruct on CUDA)
```

Reproduce the headline table and Figures 1–4 from cached trajectories:
```bash
bash scripts/reproduce.sh
```

## Project status

| Milestone | Status |
|-----------|--------|
| M0 Scaffold + repo + CI | ✅ Done |
| M1 Sandbox + tasks | ✅ Done |
| M2 Agent + capture | ✅ Done |
| M3 Induction ladder | ✅ Done |
| M4 Baselines + calibration | ✅ Done |
| M5 CrossGuard | ✅ Done |
| M6 Probe + cascade | ✅ Done |
| M7 Breaker | ✅ Done |
| M8 Online + ensemble + debate | ✅ Done |
| M9 Multi-agent collusion | ✅ Done |
| M10 Control + dashboard + serving | ✅ Done |
| M11 Full eval + release | ✅ Done |

> **159 tests passing** · Python 3.11 + 3.12 · ruff clean · mypy clean

## Benchmark results

> Evaluated on `sim` backend (8 TRAIN tasks × 3 seeds = 48 trajectories per condition, 2000 bootstrap resamples).
> For real-LLM numbers run `bash scripts/reproduce.sh` with `ANTHROPIC_API_KEY` set.

| Monitor | AUROC (naive\_divergent) | AUROC (evasion\_L2) | TPR@FPR=1% |
|---------|--------------------------|----------------------|------------|
| ActionJudge | 1.000 [1.000,1.000] | 1.000 | 1.000 |
| CoTJudge | 1.000 [1.000,1.000] | 0.500 | 1.000 |
| **CrossGuard** | 1.000 [1.000,1.000] | 1.000 | 1.000 |
| OnlineMonitor | 1.000 [1.000,1.000] | 1.000 | 1.000 |
| EnsembleMonitor | 1.000 [1.000,1.000] | 1.000 | 1.000 |
| DebateMonitor | 1.000 [1.000,1.000] | 1.000 | 1.000 |

_All metrics with bootstrap 95% CI (2000 resamples). Key finding: CoTJudge collapses to 0.500 AUROC under evasion_L2 (agent hides canary access after step 0), while CrossGuard and ActionJudge hold at 1.000._

### Hypothesis verdicts

| Hypothesis | Prediction | Verdict |
|---|---|---|
| H1 · AUROC > 0.85 at zero pressure | CrossGuard ≥ 0.85 | **SUPPORTED** — min AUROC=1.000 across 6 monitors |
| H2 · CoT ladder drop ≥ 0.25 | CoTJudge collapses under evasion | **SUPPORTED** — AUROC drop=0.500 (naive=1.000 → L2=0.500) |
| H3 · CrossGuard degrades less (non-adaptive) | Dual-channel beats single-channel | **SUPPORTED** — CrossGuard drop=0.000 vs cot_judge drop=0.500 |
| H4 · CrossGuard lower Breaker evasion rate | Harder to fool both channels | UNTESTED — requires Breaker run |
| H5 · Cascade closes ≥ 80% gap at ≤ 20% escalation | Cost-efficient detection | UNTESTED — requires cascade run |
| H6 · Transfer AUROC > 0.65 | Rule-based signal generalizes | UNTESTED — requires transfer eval |
| H7 · Judge < probe on collusion | Stego signal beats stated reasoning | UNTESTED — requires collusion run |
| H8 · Online catches early fraction | Flags before covert action completes | UNTESTED — requires online latency eval |

## Research questions

See [docs/design.md](docs/design.md) for full framing, [docs/preregistration.md](docs/preregistration.md) for pre-registered hypotheses.

## Citation

```bibtex
@software{chasm2024,
  author = {Lkumar209},
  title  = {CHASM: Cross-cHannel Agent Safety Monitoring},
  url    = {https://github.com/Lkumar209/chasm},
  year   = {2024},
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
