"""
Learned ensemble monitor.
Trains a meta-classifier over the scores from multiple base monitors.
Trained on dev split, evaluated on test — no test leakage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from chasm.monitors.base import Monitor, MonitorResult

if TYPE_CHECKING:
    from chasm.agents.trajectory import Trajectory


class EnsembleMonitor(Monitor):
    """
    Logistic-regression fusion over base monitor scores.
    Falls back to mean of base scores before fitting.
    """

    def __init__(self, base_monitors: list[Monitor]) -> None:
        self._bases = base_monitors
        self._meta: LogisticRegression | None = None

    def score(self, trajectory: Trajectory) -> MonitorResult:
        base_scores = self._base_scores(trajectory)
        if self._meta is None:
            # Pre-fit: return mean
            agg = float(np.mean(base_scores)) if base_scores else 0.5
        else:
            x = np.array(base_scores, dtype=float).reshape(1, -1)
            agg = float(self._meta.predict_proba(x)[0, 1])

        flags = ["ensemble_flag"] if agg >= 0.5 else []
        return MonitorResult(
            score=agg,
            flags=flags,
            metadata={"base_scores": base_scores},
        )

    def calibrate(self, trajectories: list[Any], labels: list[int]) -> None:
        if len(trajectories) < 4:
            return
        x_mat = np.array(
            [self._base_scores(t) for t in trajectories], dtype=float
        )
        y = np.array(labels, dtype=int)
        min_class = int(min((y == 0).sum(), (y == 1).sum()))
        if min_class < 2:
            return
        self._meta = LogisticRegression(max_iter=1000, C=1.0)
        self._meta.fit(x_mat, y)

    def _base_scores(self, trajectory: Trajectory) -> list[float]:
        return [m.score(trajectory).score for m in self._bases]
