# CHASM

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
## Diagram
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
| M1 Sandbox + tasks | 🔲 Next |
| M2 Agent + capture | 🔲 Planned |
| M3 Induction ladder | 🔲 Planned |
| M4 Baselines + calibration | 🔲 Planned |
| M5 CrossGuard | 🔲 Planned |
| M6 Probe + cascade | 🔲 Planned |
| M7 Breaker | 🔲 Planned |
| M8 Online + ensemble + debate | 🔲 Planned |
| M9 Multi-agent collusion | 🔲 Gated |
| M10 Control + dashboard + serving | 🔲 Planned |
| M11 Full eval + release | 🔲 Planned |

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
