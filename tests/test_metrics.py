"""Tests for eval metrics and bootstrap CIs."""

import pytest

from chasm.eval.metrics import (
    accuracy_at_threshold,
    auprc,
    auroc,
    bootstrap_ci,
    compute_all_metrics,
    ece,
    tpr_at_fpr,
)


def _perfect() -> tuple[list[int], list[float]]:
    labels = [1, 1, 1, 0, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
    return labels, scores


def _random() -> tuple[list[int], list[float]]:
    labels = [1, 0, 1, 0, 1, 0]
    scores = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    return labels, scores


def test_auroc_perfect() -> None:
    labels, scores = _perfect()
    assert auroc(labels, scores) == pytest.approx(1.0)


def test_auroc_random() -> None:
    labels, scores = _random()
    val = auroc(labels, scores)
    assert 0.4 <= val <= 0.6


def test_auroc_single_class_nan() -> None:
    import math

    val = auroc([1, 1, 1], [0.9, 0.8, 0.7])
    assert math.isnan(val)


def test_auprc_perfect() -> None:
    labels, scores = _perfect()
    val = auprc(labels, scores)
    assert val > 0.9


def test_tpr_at_fpr() -> None:
    labels, scores = _perfect()
    tpr = tpr_at_fpr(labels, scores, target_fpr=0.05)
    assert 0.0 <= tpr <= 1.0


def test_ece_perfect_calibration() -> None:
    labels = [1, 1, 0, 0]
    scores = [1.0, 1.0, 0.0, 0.0]
    assert ece(labels, scores) == pytest.approx(0.0, abs=0.05)


def test_accuracy_at_threshold() -> None:
    labels, scores = _perfect()
    acc = accuracy_at_threshold(labels, scores, threshold=0.5)
    assert acc == pytest.approx(1.0)


def test_bootstrap_ci_shape() -> None:
    labels, scores = _perfect()
    pt, lo, hi = bootstrap_ci(labels, scores, auroc, n_resamples=100, seed=0)
    assert lo <= pt <= hi


def test_compute_all_metrics_keys() -> None:
    labels, scores = _perfect()
    m = compute_all_metrics(labels, scores, n_bootstrap=50)
    required = {
        "auroc",
        "auprc",
        "tpr_at_fpr1",
        "tpr_at_fpr5",
        "accuracy",
        "ece",
        "n",
        "n_positive",
    }
    assert required.issubset(m.keys())


def test_compute_all_metrics_values_sane() -> None:
    labels, scores = _perfect()
    m = compute_all_metrics(labels, scores, n_bootstrap=50)
    assert m["auroc"] > 0.9
    assert m["n"] == 6
    assert m["n_positive"] == 3
