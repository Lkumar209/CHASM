# Limitations

_To be expanded at M11 with concrete numbers. Placeholder sections below._

## Sandbox realism

All experiments run inside a hermetic mock environment. Real-world tool APIs have different latency distributions, error modes, and side-effect semantics. Detection results may not transfer to live systems.

## Small models

Default agent models are in the 4B-8B parameter range. Larger models may exhibit qualitatively different evasion behavior — both harder to detect and harder to induce via prompt-only pressure.

## Prompt-based vs trained evasion

Most induction ladder levels operate via prompting. The preference-optimization level (L4) is gated. Fully trained evasion (reward hacking at scale) is out of scope.

## Synthetic covert objectives

Covert objectives are abstract canary touches. Real misaligned objectives (e.g., exfiltration, goal preservation) may produce different behavioral signatures.

## Single-GPU scale

The default pipeline targets a single 24 GB GPU. Multi-GPU or distributed inference may produce different sampling distributions.

## Monitor calibration domain shift

Monitors are calibrated on dev trajectories from the same task families as training. Calibration may not transfer well to the `transfer` set.
