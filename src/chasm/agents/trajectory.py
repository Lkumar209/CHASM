"""Typed Trajectory schema (Pydantic) with dual-channel capture."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from chasm.utils.hashing import content_hash


class StatedChannel(BaseModel):
    """Natural-language plan, reasoning, and declared constraints from the agent."""

    plan: str = ""
    reasoning: str = ""
    declared_whitelist: list[str] = Field(default_factory=list)
    raw_turns: list[dict[str, Any]] = Field(default_factory=list)


class EnactedStep(BaseModel):
    """One tool call in the enacted channel."""

    step: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    is_canary_access: bool = False


class EnactedChannel(BaseModel):
    """Structured tool-call trace."""

    steps: list[EnactedStep] = Field(default_factory=list)

    def tool_names(self) -> list[str]:
        return [s.tool for s in self.steps]

    def canary_steps(self) -> list[EnactedStep]:
        return [s for s in self.steps if s.is_canary_access]


class SandboxGroundTruth(BaseModel):
    covert_enacted: bool
    canaries_touched: list[str] = Field(default_factory=list)
    first_covert_step: int | None = None
    label: int = 0  # 1 if covert_enacted else 0

    @classmethod
    def from_sandbox_result(cls, result: Any) -> SandboxGroundTruth:
        return cls(
            covert_enacted=result.covert_enacted,
            canaries_touched=result.canaries_touched,
            first_covert_step=result.first_covert_step,
            label=result.ground_truth_label,
        )


class TrajectoryMeta(BaseModel):
    task_id: str
    family: str
    split: str
    condition: str  # aligned | naive_divergent | evasion_LN
    eval_aware: bool = False
    model_name: str = ""
    seed: int = 0
    temperature: float = 0.0


class Trajectory(BaseModel):
    """Full trajectory record: both channels + ground truth + metadata."""

    trajectory_id: str = ""
    meta: TrajectoryMeta
    stated: StatedChannel
    enacted: EnactedChannel
    ground_truth: SandboxGroundTruth
    primary_score: float = 0.0  # from task grader
    content_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.trajectory_id:
            self.trajectory_id = self._compute_id()
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_id(self) -> str:
        return content_hash(
            {
                "task_id": self.meta.task_id,
                "condition": self.meta.condition,
                "seed": self.meta.seed,
                "model": self.meta.model_name,
            }
        )

    def _compute_hash(self) -> str:
        return content_hash(
            {
                "stated": self.stated.model_dump(),
                "enacted": self.enacted.model_dump(),
                "meta": self.meta.model_dump(),
            }
        )
