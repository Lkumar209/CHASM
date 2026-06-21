"""
Smoke robustness curve: verifies the eval runner produces metrics for each
condition without errors. Uses the echo backend so no GPU needed.
"""

from pathlib import Path

import pytest


@pytest.mark.smoke
def test_smoke_robustness_curve(tmp_path: Path) -> None:
    from chasm.eval.runner import RunConfig, run_ladder
    from chasm.models.base import BackendConfig
    from chasm.models.factory import build_backend
    from chasm.monitors.action_judge import ActionJudge
    from chasm.monitors.cot_judge import CoTJudge
    from chasm.tasks import Split, build_dataset

    tasks = build_dataset(tasks_per_family=2, seed=0)
    backend = build_backend(BackendConfig(backend="echo"))
    monitors = {"cot_judge": CoTJudge(), "action_judge": ActionJudge()}

    cfg = RunConfig(
        conditions=["aligned", "naive_divergent"],
        seeds=[0],
        output_dir=str(tmp_path / "smoke_run"),
        n_bootstrap=20,
    )

    results = run_ladder(tasks, monitors, backend, cfg, split=Split.DEV)

    assert "cot_judge" in results
    assert "action_judge" in results
    for _monitor_name, cond_metrics in results.items():
        for _condition, metrics in cond_metrics.items():
            assert "auroc" in metrics or metrics == {}
            assert metrics.get("n", 0) >= 0
