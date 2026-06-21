"""Task base types: TaskSpec, Split, GradeResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chasm.envs.sandbox import Sandbox


class Split(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    TRANSFER = "transfer"


@dataclass
class GradeResult:
    primary_score: float  # 0..1 — primary objective completion
    covert_enacted: bool  # from sandbox ground truth
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    task_id: str
    family: str
    split: Split
    primary_objective: str  # natural-language description for the agent
    covert_objective: str  # hidden side-objective (for induction only, not shown to agent by default)
    system_context: str  # files, configs, etc. pre-loaded into sandbox
    canary_files: list[str] = field(default_factory=list)
    canary_tokens: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def setup_sandbox(self) -> Sandbox:
        """Return a fresh sandbox pre-loaded with this task's context."""
        from chasm.envs.sandbox import Sandbox

        files = _parse_system_context(self.system_context)
        return Sandbox(
            files=files,
            canary_files=self.canary_files or [f"__canary__{self.task_id}.txt"],
            canary_tokens=self.canary_tokens or [f"__canary__{self.task_id}_token"],
        )

    def grade(self, tool_log: list[Any], submitted_answer: str) -> GradeResult:
        """Override per family. Base: always 0."""
        return GradeResult(primary_score=0.0, covert_enacted=False)


def _parse_system_context(context: str) -> dict[str, str]:
    """Parse 'filename: content\n---\n...' blocks into a fs dict."""
    files: dict[str, str] = {}
    if not context.strip():
        return files
    for block in context.split("\n---\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n", 1)
        if len(lines) == 2 and lines[0].endswith(":"):
            path = lines[0].rstrip(":").strip()
            files[path] = lines[1]
    return files
