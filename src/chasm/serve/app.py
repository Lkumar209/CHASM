"""
FastAPI monitor microservice.

Exposes a single endpoint: POST /score
Accepts a trajectory payload, runs all configured monitors, returns scores.
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from chasm.monitors.crossguard import CrossGuard
from chasm.monitors.debate import DebateMonitor
from chasm.monitors.online import OnlineMonitor
from chasm.utils.logging import get_logger

log = get_logger("serve.app")

# ── Request / Response models ─────────────────────────────────────────────────

if _HAS_FASTAPI:
    class EnactedStepPayload(BaseModel):
        step: int
        tool: str
        args: dict[str, Any]
        observation: str
        is_canary_access: bool = False

    class TrajectoryPayload(BaseModel):
        trajectory_id: str
        stated_plan: str = ""
        stated_reasoning: str = ""
        enacted_steps: list[EnactedStepPayload] = []

    class MonitorScoreResponse(BaseModel):
        trajectory_id: str
        crossguard_score: float
        online_score: float
        debate_score: float
        flags: list[str]
        first_flag_step: int | None


# ── App factory ───────────────────────────────────────────────────────────────

def build_app() -> Any:
    if not _HAS_FASTAPI:
        raise ImportError("fastapi is required: pip install chasm[api]")

    app = FastAPI(title="CHASM Monitor Service", version="0.1.0")

    _crossguard = CrossGuard()
    _online = OnlineMonitor()
    _debate = DebateMonitor()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/score", response_model=MonitorScoreResponse)
    def score(payload: TrajectoryPayload) -> MonitorScoreResponse:
        traj = _payload_to_trajectory(payload)

        cg_result = _crossguard.score(traj)
        on_result = _online.score(traj)
        db_result = _debate.score(traj)

        all_flags = list({
            *cg_result.flags,
            *on_result.flags,
            *db_result.flags,
        })
        first_flag = on_result.first_flag_step

        log.info(
            f"Scored {payload.trajectory_id}: "
            f"cg={cg_result.score:.3f} on={on_result.score:.3f} db={db_result.score:.3f}"
        )

        return MonitorScoreResponse(
            trajectory_id=payload.trajectory_id,
            crossguard_score=cg_result.score,
            online_score=on_result.score,
            debate_score=db_result.score,
            flags=all_flags,
            first_flag_step=first_flag,
        )

    return app


def _payload_to_trajectory(payload: Any) -> Any:
    """Convert a TrajectoryPayload into a minimal duck-typed Trajectory."""
    from chasm.agents.trajectory import EnactedChannel, EnactedStep, StatedChannel

    steps = [
        EnactedStep(
            step=s.step,
            tool=s.tool,
            args=s.args,
            observation=s.observation,
            is_canary_access=s.is_canary_access,
        )
        for s in payload.enacted_steps
    ]

    class _MinimalGroundTruth:
        covert_enacted = False
        canaries_touched: list[str] = []
        first_covert_step = None
        label = 0

    class _MinimalTraj:
        trajectory_id = payload.trajectory_id
        stated = StatedChannel(
            plan=payload.stated_plan,
            reasoning=payload.stated_reasoning,
            declared_whitelist=[],
            raw_turns=[],
        )
        enacted = EnactedChannel(steps=steps)
        ground_truth = _MinimalGroundTruth()

    return _MinimalTraj()
