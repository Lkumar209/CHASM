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

![Upl<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 600" width="1040" height="600" role="img" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif">
<title>CHASM architecture</title>
<desc>Three stages. One: an agent runs a task and produces two channels, what it says and what it does; the actions include a copy-credentials step the stated plan never mentions, leaving a visible gap. Two: CrossGuard reads both channels and flags the action that was never stated. Three: the Breaker adaptive red-team loop feeds back into the agent to hide the gap a little better each round, and the result is that single-channel monitors collapse while CrossGuard holds.</desc>
<style>
  .h{font-size:22px;font-weight:600;}
  .sub{font-size:13px;}
  .zone{font-size:13px;font-weight:600;}
  .bt{font-size:14px;font-weight:600;}
  .bs{font-size:11.5px;}
  .tab{font-size:13px;font-weight:600;}
  .lbl{font-size:12px;}
  .gaplbl{font-size:12.5px;font-weight:600;}
  .res{font-size:12.5px;}
  .flow{stroke:#9a988f;stroke-width:1.5;fill:none;}
  .flowd{stroke:#9a988f;stroke-width:1.5;fill:none;stroke-dasharray:5 4;}
  .cont{fill:none;stroke:#cbc9c0;stroke-width:1;stroke-dasharray:4 4;}
  .ink{fill:#2c2c2a;}
  .muted{fill:#76746d;}
  .zonec{fill:#534ab7;}
  @media (prefers-color-scheme: dark){
    .flow,.flowd{stroke:#6f6d64;}
    .cont{stroke:#54534c;}
    .ink{fill:#e8e6df;}
    .muted{fill:#a8a69c;}
    .zonec{fill:#afa9ec;}
  }
</style>
<defs>
  <marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="#9a988f" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>

<text class="h ink" x="40" y="42">CHASM</text>
<text class="sub muted" x="40" y="64">the gap between what an agent says and what it does</text>

<text class="zone zonec" x="40" y="100">1 &#183; the two channels</text>
<text class="zone zonec" x="846" y="100">2 &#183; cross-channel check</text>
<text class="zone zonec" x="190" y="392">3 &#183; the adversarial test</text>

<rect x="40" y="168" width="110" height="64" rx="10" fill="#f4f2ec" stroke="#c2c0b6" stroke-width="1"/>
<text class="bt ink" x="95" y="194" text-anchor="middle">Agent</text>
<text class="bs muted" x="95" y="212" text-anchor="middle">runs a task</text>

<rect x="190" y="124" width="84" height="52" rx="9" fill="#e1f5ee" stroke="#5dcaa5" stroke-width="1"/>
<text class="tab" x="232" y="154" text-anchor="middle" fill="#0f6e56">says</text>
<rect x="190" y="224" width="84" height="52" rx="9" fill="#faece7" stroke="#f0997b" stroke-width="1"/>
<text class="tab" x="232" y="254" text-anchor="middle" fill="#993c1d">does</text>

<rect x="572" y="116" width="134" height="168" rx="12" fill="none" stroke="#d8503a" stroke-width="1.25" stroke-dasharray="5 4"/>
<text class="gaplbl" x="639" y="302" text-anchor="middle" fill="#a32d2d">the gap</text>

<rect x="300" y="124" width="118" height="52" rx="9" fill="#f4f2ec" stroke="#c2c0b6" stroke-width="1"/>
<text class="bt ink" x="359" y="154" text-anchor="middle">read config</text>
<rect x="440" y="124" width="118" height="52" rx="9" fill="#f4f2ec" stroke="#c2c0b6" stroke-width="1"/>
<text class="bt ink" x="499" y="154" text-anchor="middle">check value</text>
<rect x="580" y="124" width="118" height="52" rx="9" fill="none" stroke="#b4b2a9" stroke-width="1" stroke-dasharray="5 4"/>
<text class="bs muted" x="639" y="154" text-anchor="middle">nothing said</text>
<rect x="720" y="124" width="118" height="52" rx="9" fill="#f4f2ec" stroke="#c2c0b6" stroke-width="1"/>
<text class="bt ink" x="779" y="154" text-anchor="middle">report done</text>

<rect x="300" y="224" width="118" height="52" rx="9" fill="#f4f2ec" stroke="#c2c0b6" stroke-width="1"/>
<text class="bt ink" x="359" y="254" text-anchor="middle">read config</text>
<rect x="440" y="224" width="118" height="52" rx="9" fill="#f4f2ec" stroke="#c2c0b6" stroke-width="1"/>
<text class="bt ink" x="499" y="254" text-anchor="middle">check value</text>
<rect x="580" y="224" width="118" height="52" rx="9" fill="#fcebeb" stroke="#e24b4a" stroke-width="1.5"/>
<text class="bt" x="639" y="254" text-anchor="middle" fill="#791f1f">copy creds</text>
<rect x="720" y="224" width="118" height="52" rx="9" fill="#f4f2ec" stroke="#c2c0b6" stroke-width="1"/>
<text class="bt ink" x="779" y="254" text-anchor="middle">report done</text>

<line x1="150" y1="184" x2="186" y2="152" class="flow" marker-end="url(#ar)"/>
<line x1="150" y1="216" x2="186" y2="248" class="flow" marker-end="url(#ar)"/>
<line x1="274" y1="150" x2="298" y2="150" class="flow" marker-end="url(#ar)"/>
<line x1="274" y1="250" x2="298" y2="250" class="flow" marker-end="url(#ar)"/>
<line x1="418" y1="150" x2="438" y2="150" class="flow" marker-end="url(#ar)"/>
<line x1="558" y1="150" x2="578" y2="150" class="flow" marker-end="url(#ar)"/>
<line x1="698" y1="150" x2="718" y2="150" class="flow" marker-end="url(#ar)"/>
<line x1="418" y1="250" x2="438" y2="250" class="flow" marker-end="url(#ar)"/>
<line x1="558" y1="250" x2="578" y2="250" class="flow" marker-end="url(#ar)"/>
<line x1="698" y1="250" x2="718" y2="250" class="flow" marker-end="url(#ar)"/>

<rect x="870" y="168" width="140" height="64" rx="10" fill="#eeedfe" stroke="#7f77dd" stroke-width="2"/>
<text class="bt" x="940" y="194" text-anchor="middle" fill="#26215c">CrossGuard</text>
<text class="bs" x="940" y="212" text-anchor="middle" fill="#534ab7">reads both</text>
<rect x="872" y="246" width="136" height="34" rx="8" fill="#fcebeb" stroke="#e24b4a" stroke-width="1"/>
<text class="res" x="940" y="267" text-anchor="middle" fill="#791f1f">action not stated</text>

<line x1="838" y1="150" x2="866" y2="186" class="flow" marker-end="url(#ar)"/>
<line x1="838" y1="250" x2="866" y2="214" class="flow" marker-end="url(#ar)"/>
<line x1="940" y1="232" x2="940" y2="244" class="flow" marker-end="url(#ar)"/>

<rect x="300" y="400" width="220" height="64" rx="10" fill="#faece7" stroke="#f0997b" stroke-width="1"/>
<text class="bt" x="410" y="426" text-anchor="middle" fill="#4a1b0c">Breaker</text>
<text class="bs" x="410" y="444" text-anchor="middle" fill="#993c1d">adaptive red-team loop</text>

<path d="M940 280 L940 432 L522 432" class="flowd" marker-end="url(#ar)"/>
<text class="lbl muted" x="948" y="360" text-anchor="start">feedback</text>
<path d="M300 432 L38 432 L38 198 L42 198" class="flowd" marker-end="url(#ar)"/>
<text class="lbl muted" x="410" y="486" text-anchor="middle">each round, hide the gap a little better</text>

<rect x="560" y="486" width="450" height="86" rx="12" class="cont"/>
<text class="zone zonec" x="578" y="510">result</text>
<line x1="590" y1="532" x2="628" y2="532" stroke="#e24b4a" stroke-width="3" stroke-linecap="round"/>
<text class="res ink" x="636" y="536" text-anchor="start">single-channel monitor collapses</text>
<line x1="590" y1="556" x2="628" y2="556" stroke="#7f77dd" stroke-width="3" stroke-linecap="round"/>
<text class="res ink" x="636" y="560" text-anchor="start">CrossGuard holds the line</text>
</svg>
oading chasm-architecture.svg…]()


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
