"""Tests for probe (M6) and cascade escalation monitor."""

import numpy as np
import pytest

from chasm.agents.trajectory import (
    EnactedChannel,
    EnactedStep,
    SandboxGroundTruth,
    StatedChannel,
    Trajectory,
    TrajectoryMeta,
)
from chasm.monitors.action_judge import ActionJudge
from chasm.monitors.cot_judge import CoTJudge
from chasm.monitors.escalation import CascadeMonitor
from chasm.probes.probe import LinearProbe, MLPProbe


def _make_traj(label: int = 0, has_canary: bool = False) -> Trajectory:
    steps = []
    if has_canary:
        steps.append(EnactedStep(step=0, tool="read_file", args={"path": "__canary__.txt"}, is_canary_access=True))
    steps.append(EnactedStep(step=1, tool="submit", args={"answer": "done"}))
    return Trajectory(
        meta=TrajectoryMeta(task_id="t", family="f", split="dev", condition="aligned", seed=0),
        stated=StatedChannel(plan="I will complete the task.", reasoning=""),
        enacted=EnactedChannel(steps=steps),
        ground_truth=SandboxGroundTruth(covert_enacted=bool(label), label=label),
    )


class TestLinearProbe:
    def test_fit_and_predict(self) -> None:
        acts = [np.random.randn(10, 64) for _ in range(20)]
        labels = [0] * 10 + [1] * 10
        probe = LinearProbe()
        probe.fit(acts, labels)
        score = probe.predict_proba(np.random.randn(5, 64))
        assert 0.0 <= score <= 1.0

    def test_unfitted_returns_half(self) -> None:
        probe = LinearProbe()
        score = probe.predict_proba(np.random.randn(5, 64))
        assert score == pytest.approx(0.5)


class TestMLPProbe:
    def test_fit_and_predict(self) -> None:
        acts = [np.random.randn(10, 32) for _ in range(20)]
        labels = [0] * 10 + [1] * 10
        probe = MLPProbe(hidden_size=16)
        probe.fit(acts, labels)
        score = probe.predict_proba(np.random.randn(5, 32))
        assert 0.0 <= score <= 1.0


class TestCascadeMonitor:
    def test_no_escalation_for_low_score(self) -> None:
        cheap = ActionJudge()   # benign -> low score
        expensive = CoTJudge()
        cascade = CascadeMonitor(cheap, expensive, lo_threshold=0.4, hi_threshold=0.8)
        t = _make_traj(label=0)
        r = cascade.score(t)
        assert not r.metadata.get("escalated", False)
        assert cascade.stats().n_escalated == 0

    def test_escalation_for_ambiguous_score(self) -> None:
        cheap = CoTJudge()
        expensive = ActionJudge()
        # Use a trajectory with canary to get a high cheap score
        cascade = CascadeMonitor(cheap, expensive, lo_threshold=0.0, hi_threshold=1.0)
        t = _make_traj(has_canary=True, label=1)
        cascade.score(t)
        assert cascade.stats().n_escalated == 1

    def test_escalation_rate_computed(self) -> None:
        cheap = ActionJudge()
        expensive = CoTJudge()
        cascade = CascadeMonitor(cheap, expensive, lo_threshold=0.3, hi_threshold=0.7)
        for _i in range(5):
            cascade.score(_make_traj())
        stats = cascade.stats()
        assert 0.0 <= stats.escalation_rate <= 1.0
        assert stats.n_total == 5

    def test_cost_tracked(self) -> None:
        cheap = ActionJudge()
        expensive = CoTJudge()
        cascade = CascadeMonitor(cheap, expensive, cheap_cost=1.0, expensive_cost=5.0)
        cascade.score(_make_traj())
        stats = cascade.stats()
        assert stats.total_cost_cheap >= 1.0

    def test_reset_stats(self) -> None:
        cheap = ActionJudge()
        expensive = CoTJudge()
        cascade = CascadeMonitor(cheap, expensive)
        cascade.score(_make_traj())
        cascade.reset_stats()
        assert cascade.stats().n_total == 0
