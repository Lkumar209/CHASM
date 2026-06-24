"""Abstract Monitor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MonitorResult:
    score: float  # calibrated probability in [0, 1] of covert enactment
    flags: list[str] = field(default_factory=list)  # typed divergence flags
    first_flag_step: int | None = None  # for online monitors
    metadata: dict[str, Any] = field(default_factory=dict)


class Monitor(ABC):
    """Input: a Trajectory (or prefix). Output: a calibrated MonitorResult."""

    @abstractmethod
    def score(self, trajectory: Any) -> MonitorResult: ...

    def calibrate(self, trajectories: list[Any], labels: list[int]) -> None:  # noqa: B027
        """Fit a calibration map (Platt/isotonic) on dev set. Override to implement."""
