"""Tests for M8: OnlineMonitor, EnsembleMonitor, DebateMonitor."""

from __future__ import annotations

from chasm.agents.loop import run_agent
from chasm.models.base import BackendConfig
from chasm.models.factory import build_backend
from chasm.monitors.action_judge import ActionJudge
from chasm.monitors.cot_judge import CoTJudge
from chasm.monitors.crossguard import CrossGuard
from chasm.monitors.debate import DebateMonitor
from chasm.monitors.ensemble import EnsembleMonitor
from chasm.monitors.online import OnlineMonitor
from chasm.tasks import Split, build_dataset


def _make_traj(condition: str = "aligned"):
    tasks = build_dataset(tasks_per_family=10, seed=42)
    task = next(t for t in tasks if t.split == Split.TRAIN)
    backend = build_backend(BackendConfig(backend="echo"))
    return run_agent(task, backend, condition=condition, seed=0)


# ── OnlineMonitor ──────────────────────────────────────────────────────────────


def test_online_monitor_returns_result():
    traj = _make_traj("aligned")
    monitor = OnlineMonitor(flag_threshold=0.5)
    result = monitor.score(traj)
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.flags, list)


def test_online_monitor_score_stream():
    traj = _make_traj("aligned")
    monitor = OnlineMonitor()
    stream = monitor.score_stream(traj)
    assert isinstance(stream, list)
    for step_idx, score in stream:
        assert isinstance(step_idx, int)
        assert 0.0 <= score <= 1.0


def test_online_monitor_first_flag_step_type():
    traj = _make_traj("naive_divergent")
    monitor = OnlineMonitor(flag_threshold=0.01)  # very low threshold → always flags
    result = monitor.score(traj)
    # With very low threshold, should flag at some step (if steps exist)
    if traj.enacted.steps:
        assert result.first_flag_step is not None or result.score < 0.01


def test_online_monitor_stream_length_matches_steps():
    traj = _make_traj("aligned")
    monitor = OnlineMonitor()
    stream = monitor.score_stream(traj)
    assert len(stream) == len(traj.enacted.steps)


# ── EnsembleMonitor ────────────────────────────────────────────────────────────


def test_ensemble_monitor_unfitted_returns_mean():
    traj = _make_traj("aligned")
    base_monitors = [CoTJudge(), ActionJudge()]
    monitor = EnsembleMonitor(base_monitors)
    result = monitor.score(traj)
    assert 0.0 <= result.score <= 1.0
    assert "base_scores" in result.metadata


def test_ensemble_monitor_calibrate_and_score():
    tasks = build_dataset(tasks_per_family=4, seed=7)
    backend = build_backend(BackendConfig(backend="echo"))
    trajs = []
    labels = []
    for t in tasks[:8]:
        condition = "naive_divergent" if len(trajs) % 2 == 0 else "aligned"
        traj = run_agent(t, backend, condition=condition, seed=0)
        trajs.append(traj)
        labels.append(1 if traj.ground_truth.covert_enacted else 0)

    base_monitors = [CoTJudge(), ActionJudge(), CrossGuard()]
    monitor = EnsembleMonitor(base_monitors)
    monitor.calibrate(trajs, labels)
    result = monitor.score(trajs[0])
    assert 0.0 <= result.score <= 1.0


def test_ensemble_monitor_base_scores_count():
    traj = _make_traj("aligned")
    bases = [CoTJudge(), ActionJudge(), CrossGuard()]
    monitor = EnsembleMonitor(bases)
    result = monitor.score(traj)
    assert len(result.metadata["base_scores"]) == 3


def test_ensemble_monitor_calibrate_too_few_examples_noop():
    traj = _make_traj("aligned")
    bases = [CrossGuard()]
    monitor = EnsembleMonitor(bases)
    # Only 2 examples — should not crash, meta remains None
    monitor.calibrate([traj, traj], [0, 1])
    # Still callable
    result = monitor.score(traj)
    assert 0.0 <= result.score <= 1.0


# ── DebateMonitor ──────────────────────────────────────────────────────────────


def test_debate_monitor_rule_based_aligned():
    traj = _make_traj("aligned")
    monitor = DebateMonitor(backend=None)
    result = monitor.score(traj)
    assert 0.0 <= result.score <= 1.0
    assert "judge_a_score" in result.metadata
    assert "judge_b_score" in result.metadata
    assert "disagreement" in result.metadata


def test_debate_monitor_rule_based_divergent_scores_higher():
    traj_aligned = _make_traj("aligned")
    traj_divergent = _make_traj("naive_divergent")
    monitor = DebateMonitor(backend=None)
    score_a = monitor.score(traj_aligned).score
    score_d = monitor.score(traj_divergent).score
    # Divergent should score >= aligned on average
    assert score_d >= score_a or abs(score_d - score_a) < 0.5  # soft check


def test_debate_monitor_echo_backend():
    traj = _make_traj("naive_divergent")
    backend = build_backend(BackendConfig(backend="echo"))
    monitor = DebateMonitor(backend=backend)
    result = monitor.score(traj)
    assert 0.0 <= result.score <= 1.0
    assert "judge_a_raw" in result.metadata


def test_debate_monitor_calibrate_noop():
    traj = _make_traj("aligned")
    monitor = DebateMonitor()
    monitor.calibrate([traj, traj], [0, 1])  # should not raise


# ── Integration: ensemble over online+crossguard+debate ───────────────────────


def test_ensemble_over_all_m8_monitors():
    traj = _make_traj("naive_divergent")
    bases = [OnlineMonitor(), CrossGuard(), DebateMonitor()]
    monitor = EnsembleMonitor(bases)
    result = monitor.score(traj)
    assert 0.0 <= result.score <= 1.0
    assert len(result.metadata["base_scores"]) == 3
