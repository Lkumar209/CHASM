"""Unit tests for monitor base interface."""

import pytest

from chasm.monitors.base import Monitor, MonitorResult


class DummyMonitor(Monitor):
    def score(self, trajectory: object) -> MonitorResult:
        return MonitorResult(score=0.42, flags=["test_flag"])


def test_monitor_result_defaults() -> None:
    r = MonitorResult(score=0.5)
    assert r.flags == []
    assert r.first_flag_step is None
    assert r.metadata == {}


def test_dummy_monitor_score() -> None:
    m = DummyMonitor()
    r = m.score(object())
    assert r.score == pytest.approx(0.42)
    assert "test_flag" in r.flags
