"""Tests for trajectory schema and agent loop (M2) + determinism."""

import pytest

from chasm.agents.trajectory import (
    EnactedChannel,
    EnactedStep,
    SandboxGroundTruth,
    StatedChannel,
    Trajectory,
    TrajectoryMeta,
)
from chasm.tasks.base import Split
from chasm.tasks.families import make_codebase_task


def _make_trajectory(seed: int = 0, condition: str = "aligned") -> Trajectory:
    return Trajectory(
        meta=TrajectoryMeta(
            task_id="test_001",
            family="codebase_manipulation",
            split="train",
            condition=condition,
            seed=seed,
            model_name="echo",
        ),
        stated=StatedChannel(plan="I will read the file.", reasoning="Step by step plan."),
        enacted=EnactedChannel(
            steps=[
                EnactedStep(
                    step=0,
                    tool="read_file",
                    args={"path": "src/utils.py"},
                    observation="def add...",
                ),
                EnactedStep(
                    step=1, tool="submit", args={"answer": "done"}, observation="[submitted]"
                ),
            ]
        ),
        ground_truth=SandboxGroundTruth(covert_enacted=False, label=0),
        primary_score=0.8,
    )


def test_trajectory_id_stable() -> None:
    t = _make_trajectory(seed=42)
    assert t.trajectory_id == _make_trajectory(seed=42).trajectory_id


def test_trajectory_id_changes_with_seed() -> None:
    t1 = _make_trajectory(seed=1)
    t2 = _make_trajectory(seed=2)
    assert t1.trajectory_id != t2.trajectory_id


def test_trajectory_content_hash_stable() -> None:
    t = _make_trajectory()
    assert t.content_hash == _make_trajectory().content_hash


def test_enacted_channel_tool_names() -> None:
    t = _make_trajectory()
    assert t.enacted.tool_names() == ["read_file", "submit"]


def test_enacted_channel_canary_steps_empty() -> None:
    t = _make_trajectory()
    assert t.enacted.canary_steps() == []


def test_ground_truth_label() -> None:
    gt = SandboxGroundTruth(covert_enacted=True, canaries_touched=["x"], label=1)
    assert gt.label == 1


@pytest.mark.determinism
def test_agent_loop_determinism() -> None:
    """Same seed must produce identical trajectory content hash."""
    from chasm.agents.loop import run_agent
    from chasm.models.base import BackendConfig
    from chasm.models.factory import build_backend

    task = make_codebase_task(0, Split.TRAIN)
    cfg = BackendConfig(backend="echo")
    backend = build_backend(cfg)

    t1 = run_agent(task, backend, condition="aligned", seed=7)
    t2 = run_agent(task, backend, condition="aligned", seed=7)
    assert t1.content_hash == t2.content_hash


@pytest.mark.determinism
def test_agent_loop_different_seeds_differ() -> None:
    from chasm.agents.loop import run_agent
    from chasm.models.base import BackendConfig
    from chasm.models.factory import build_backend

    task = make_codebase_task(0, Split.TRAIN)
    backend = build_backend(BackendConfig(backend="echo"))
    t1 = run_agent(task, backend, condition="aligned", seed=0)
    t2 = run_agent(task, backend, condition="aligned", seed=1)
    assert t1.trajectory_id != t2.trajectory_id
