"""Detection metrics: AUROC, AUPRC, TPR@FPR, ECE, bootstrap CIs."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


def auroc(labels: list[int], scores: list[float]) -> float:
    y = np.array(labels)
    s = np.array(scores)
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def auprc(labels: list[int], scores: list[float]) -> float:
    y = np.array(labels)
    s = np.array(scores)
    if len(set(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def tpr_at_fpr(labels: list[int], scores: list[float], target_fpr: float = 0.05) -> float:
    """TPR at a given FPR threshold, interpolated from the ROC curve."""
    from sklearn.metrics import roc_curve

    y = np.array(labels)
    s = np.array(scores)
    if len(set(y)) < 2:
        return float("nan")
    fpr_arr, tpr_arr, _ = roc_curve(y, s)
    idx = np.searchsorted(fpr_arr, target_fpr)
    if idx >= len(tpr_arr):
        return float(tpr_arr[-1])
    return float(tpr_arr[idx])


def ece(labels: list[int], scores: list[float], n_bins: int = 10) -> float:
    """Expected calibration error."""
    y = np.array(labels, dtype=float)
    s = np.array(scores)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n = len(y)
    for i in range(n_bins):
        mask = (s >= bins[i]) & (s < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = s[mask].mean()
        ece_val += mask.sum() / n * abs(acc - conf)
    return float(ece_val)


def accuracy_at_threshold(labels: list[int], scores: list[float], threshold: float = 0.5) -> float:
    y = np.array(labels)
    preds = (np.array(scores) >= threshold).astype(int)
    return float((y == preds).mean())


def bootstrap_ci(
    labels: list[int],
    scores: list[float],
    metric_fn: Any,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point_estimate, lower_ci, upper_ci) via bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    point = metric_fn(labels, scores)
    boot_vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        bl = [labels[i] for i in idx]
        bs = [scores[i] for i in idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = metric_fn(bl, bs)
        if not np.isnan(v):
            boot_vals.append(v)
    if not boot_vals:
        return point, float("nan"), float("nan")
    alpha = (1 - ci) / 2
    lo = float(np.quantile(boot_vals, alpha))
    hi = float(np.quantile(boot_vals, 1 - alpha))
    return point, lo, hi


def compute_all_metrics(
    labels: list[int],
    scores: list[float],
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute the full metric suite with bootstrap CIs."""
    auroc_pt, auroc_lo, auroc_hi = bootstrap_ci(labels, scores, auroc, n_bootstrap, seed=seed)
    auprc_pt, auprc_lo, auprc_hi = bootstrap_ci(labels, scores, auprc, n_bootstrap, seed=seed)
    return {
        "auroc": auroc_pt,
        "auroc_ci_lo": auroc_lo,
        "auroc_ci_hi": auroc_hi,
        "auprc": auprc_pt,
        "auprc_ci_lo": auprc_lo,
        "auprc_ci_hi": auprc_hi,
        "tpr_at_fpr1": tpr_at_fpr(labels, scores, target_fpr=0.01),
        "tpr_at_fpr5": tpr_at_fpr(labels, scores, target_fpr=0.05),
        "accuracy": accuracy_at_threshold(labels, scores, threshold),
        "ece": ece(labels, scores),
        "n": len(labels),
        "n_positive": int(sum(labels)),
    }
