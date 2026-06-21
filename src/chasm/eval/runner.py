"""Parallelisable evaluation runner: generate trajectories + score monitors."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chasm.agents.loop import run_agent
from chasm.agents.store import TrajectoryStore
from chasm.eval.metrics import compute_all_metrics
from chasm.induction.conditions import InductionConfig, should_discard
from chasm.monitors.base import Monitor  # noqa: TCH001
from chasm.tasks.base import Split, TaskSpec
from chasm.utils.logging import get_logger

log = get_logger("runner")


@dataclass
class RunConfig:
    conditions: list[str] = field(default_factory=lambda: ["aligned", "naive_divergent"])
    seeds: list[int] = field(default_factory=lambda: [0])
    eval_aware_variants: bool = False
    max_steps: int = 12
    output_dir: str = "experiments/default"
    n_bootstrap: int = 500
    token_budget: int = 100_000


def run_ladder(
    tasks: list[TaskSpec],
    monitors: dict[str, Monitor],
    backend: Any,
    config: RunConfig,
    split: Split = Split.DEV,
) -> dict[str, Any]:
    """
    Generate trajectories for all (condition, seed) combos on the given split,
    score all monitors, and return a metrics dict keyed by monitor x condition.
    """
    split_tasks = [t for t in tasks if t.split == split]
    store = TrajectoryStore(store_dir=Path(config.output_dir) / "trajectories")

    all_trajectories: dict[str, list[Any]] = {}
    all_labels: dict[str, list[int]] = {}

    for condition in config.conditions:
        ic = InductionConfig(condition=condition)
        key = condition
        all_trajectories[key] = []
        all_labels[key] = []

        for seed in config.seeds:
            for task in split_tasks:
                traj = run_agent(
                    task,
                    backend,
                    condition=condition,
                    eval_aware=False,
                    seed=seed,
                    max_steps=config.max_steps,
                )
                if should_discard(traj, ic):
                    log.info(f"Discarded {traj.trajectory_id} (covert not enacted)")
                    continue
                store.save(traj)
                all_trajectories[key].append(traj)
                all_labels[key].append(traj.ground_truth.label)

    store.flush()

    results: dict[str, Any] = {}
    for monitor_name, monitor in monitors.items():
        results[monitor_name] = {}
        for condition_key, trajs in all_trajectories.items():
            if not trajs:
                continue
            scores = [monitor.score(t).score for t in trajs]
            labels = all_labels[condition_key]
            metrics = compute_all_metrics(labels, scores, n_bootstrap=config.n_bootstrap)
            results[monitor_name][condition_key] = metrics
            log.info(
                f"{monitor_name} / {condition_key}: "
                f"AUROC={metrics['auroc']:.3f} [{metrics['auroc_ci_lo']:.3f}, {metrics['auroc_ci_hi']:.3f}]"
            )

    output_path = Path(config.output_dir) / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results
