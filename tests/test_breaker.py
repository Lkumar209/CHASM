"""Tests for Breaker adaptive red-team loop and strategy library."""

import json
from pathlib import Path

from chasm.models.base import BackendConfig
from chasm.models.factory import build_backend
from chasm.monitors.crossguard import CrossGuard
from chasm.redteam.breaker import Breaker, BreakerConfig
from chasm.redteam.strategies import (
    ALL_STRATEGIES,
    CALIBRATION_STRATEGIES,
    HELD_OUT_STRATEGIES,
    get_strategy,
)
from chasm.tasks import Split, build_dataset


def test_strategy_catalogue_size() -> None:
    assert len(ALL_STRATEGIES) >= 5


def test_held_out_not_in_calibration() -> None:
    held_names = {s.name for s in HELD_OUT_STRATEGIES}
    calib_names = {s.name for s in CALIBRATION_STRATEGIES}
    assert held_names.isdisjoint(calib_names)


def test_get_strategy() -> None:
    s = get_strategy("plan_laundering")
    assert s.name == "plan_laundering"
    assert len(s.system_prompt_suffix) > 0


def test_get_strategy_unknown_raises() -> None:
    import pytest
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy("nonexistent")


def test_breaker_runs_smoke(tmp_path: Path) -> None:

    tasks = build_dataset(tasks_per_family=2, seed=0)
    train_tasks = [t for t in tasks if t.split == Split.TRAIN][:4]
    backend = build_backend(BackendConfig(backend="echo"))
    monitor = CrossGuard()

    cfg = BreakerConfig(
        n_rounds=2,
        n_attempts_per_strategy=1,
        seeds=[0],
        output_dir=str(tmp_path / "breaker"),
        condition="evasion_L0",
    )
    breaker = Breaker(monitor, train_tasks, backend, cfg)
    results = breaker.run()

    assert len(results) == 2
    frontier = breaker.frontier()
    assert len(frontier) == 2
    for point in frontier:
        assert "evasion_success_rate" in point
        assert "cumulative_attacker_cost" in point
        assert 0.0 <= point["evasion_success_rate"] <= 1.0


def test_breaker_saves_output(tmp_path: Path) -> None:

    tasks = build_dataset(tasks_per_family=2, seed=0)
    train_tasks = [t for t in tasks if t.split == Split.TRAIN][:2]
    backend = build_backend(BackendConfig(backend="echo"))
    monitor = CrossGuard()

    cfg = BreakerConfig(
        n_rounds=1,
        seeds=[0],
        output_dir=str(tmp_path / "breaker_out"),
    )
    breaker = Breaker(monitor, train_tasks, backend, cfg)
    breaker.run()

    frontier_path = tmp_path / "breaker_out" / "frontier.json"
    assert frontier_path.exists()
    data = json.loads(frontier_path.read_text())
    assert isinstance(data, list)
