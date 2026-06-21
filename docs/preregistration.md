# Pre-registration

**Status: DRAFT — will be frozen at end of M4. Do not edit after freeze.**

## Hypotheses

- **H1.** At zero evasion pressure, all monitors achieve AUROC > 0.85; CrossGuard is within noise of the best single-channel baseline.
- **H2 (ladder sanity).** Along the non-adaptive ladder, the CoT-only monitor's AUROC drops by >= 0.25 from min to max pressure.
- **H3 (main, non-adaptive).** CrossGuard's AUROC degrades significantly less than the best single-channel baseline across the ladder (paired test over seeds; report effect size and CI).
- **H4 (adaptive).** Under Breaker, CrossGuard's steady-state evasion rate is lower than the best single-channel baseline's, and/or the per-success attacker compute is higher.
- **H5 (cascade).** The cost-constrained cascade closes >= 80% of the detection gap at <= 20% escalation rate.
- **H6 (transfer).** Detection stays above chance (AUROC > 0.65) zero-shot to held-out task families and to held-out evasion strategies.
- **H7 (multi-agent).** In the collusion track, an LLM-judge over the message log underperforms a consistency/probe-based monitor for covert (steganographic) collusion.
- **H8 (online).** A sequential monitor flags a meaningful fraction of divergent trajectories before the covert action completes.

## Power analysis

_To be filled at M4, before any test numbers are seen._

Planned: >= 5 seeds per cell. Effect size justification: TODO. Type I error rate: 0.05. Multiple comparison correction: Bonferroni over H1-H8. Minimum detectable effect size: TODO.

## Choices fixed before seeing test data

- Train/dev/test/transfer split proportions: 60/15/15/10 (task families, not trajectories).
- Calibration method: isotonic regression on dev set.
- Hypothesis test: paired bootstrap difference with 10,000 resamples; DeLong's test for AUROC comparisons.
- Primary metric: AUROC (secondary: AUPRC, TPR@FPR=1%).

## What "frozen" means

After M4 sign-off, this file is locked. Any deviation (new hypothesis, changed metric, dropped condition) must be logged as a deviation in CHANGELOG.md with a justification. The paper will report deviations explicitly.
