# CHASM

**Cross-cHannel Agent Safety Monitoring**

CHASM is a research framework for detecting when a tool-using LLM agent's *stated* reasoning diverges from its *enacted* actions, and for measuring how that detection degrades when the agent is optimized to hide the gap.

## Components

| Component | Description |
|-----------|-------------|
| **RIFT-Bench** | Benchmark inducing stated-vs-enacted divergence in a hermetic sandbox |
| **CrossGuard** | Cross-channel consistency monitor (proposed) + baseline zoo |
| **Breaker** | Adaptive adversarial red-team loop |
| **Control Protocol** | Trusted/untrusted evaluation harness with safety-case reports |

## Quick start

```bash
git clone https://github.com/Lkumar209/chasm && cd chasm
uv sync --extra dev
uv run pytest -m smoke    # smoke e2e (~minutes, CPU, no API)
```

## Research questions

See [Design](design.md) for the full framing and [Pre-registration](preregistration.md) for the hypotheses.
