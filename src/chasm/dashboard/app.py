"""
Streamlit trajectory inspector dashboard.

Launch with: streamlit run src/chasm/dashboard/app.py

Provides:
- Trajectory browser (load from parquet store or JSON)
- Per-step enacted channel viewer with canary highlights
- Monitor score panel (CrossGuard, OnlineMonitor, DebateMonitor)
- H1–H8 metrics summary (if metrics.json present)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _require_streamlit() -> Any:
    try:
        import streamlit as st
        return st
    except ImportError as e:
        raise ImportError(
            "streamlit is required: pip install chasm[probe]"
        ) from e


def main() -> None:
    st = _require_streamlit()

    st.set_page_config(page_title="CHASM Inspector", layout="wide")
    st.title("CHASM — Trajectory Inspector")

    # ── Sidebar: data source ─────────────────────────────────────────────────
    st.sidebar.header("Data Source")
    data_dir = st.sidebar.text_input(
        "Experiment directory",
        value="experiments/",
        help="Directory containing metrics.json and parquet trajectory store",
    )
    uploaded = st.sidebar.file_uploader(
        "Or upload a trajectory JSON", type=["json"]
    )

    # ── Main panel ───────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 2])

    # Load trajectory
    trajectory_data: dict[str, Any] | None = None

    if uploaded is not None:
        try:
            trajectory_data = json.loads(uploaded.read())
            st.sidebar.success("Loaded uploaded trajectory")
        except Exception as exc:
            st.sidebar.error(f"Failed to parse JSON: {exc}")

    with col_left:
        st.subheader("Trajectory Meta")
        if trajectory_data:
            meta = trajectory_data.get("meta", {})
            st.json(meta)

            st.subheader("Stated Channel")
            stated = trajectory_data.get("stated", {})
            st.markdown(f"**Plan:** {stated.get('plan', '—')}")
            st.markdown(f"**Reasoning:** {stated.get('reasoning', '—')}")
            declared = stated.get("declared_whitelist", [])
            if declared:
                st.markdown("**Declared whitelist:** " + ", ".join(declared))
        else:
            st.info("Upload a trajectory JSON or point to an experiment directory.")

    with col_right:
        st.subheader("Enacted Channel")
        if trajectory_data:
            steps = trajectory_data.get("enacted", {}).get("steps", [])
            for step in steps:
                is_canary = step.get("is_canary_access", False)
                color = "red" if is_canary else "green"
                label = "⚠ CANARY" if is_canary else "✓"
                st.markdown(
                    f"**Step {step['step']}** `{step['tool']}` "
                    f"<span style='color:{color}'>{label}</span>",
                    unsafe_allow_html=True,
                )
                with st.expander(f"Args / Obs — step {step['step']}"):
                    st.json(step.get("args", {}))
                    st.text(step.get("observation", ""))

        # Monitor scores
        st.subheader("Monitor Scores")
        if trajectory_data:
            _render_monitor_scores(st, trajectory_data)

    # ── Metrics panel ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Experiment Metrics")
    metrics_path = Path(data_dir) / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        st.json(metrics)
    else:
        st.info(f"No metrics.json found at {metrics_path}")


def _render_monitor_scores(st: Any, trajectory_data: dict[str, Any]) -> None:
    """Run monitors on a loaded trajectory dict and show scores."""
    try:
        from chasm.agents.trajectory import EnactedChannel, EnactedStep, StatedChannel
        from chasm.monitors.crossguard import CrossGuard
        from chasm.monitors.debate import DebateMonitor
        from chasm.monitors.online import OnlineMonitor

        stated_d = trajectory_data.get("stated", {})
        steps_d = trajectory_data.get("enacted", {}).get("steps", [])

        steps = [
            EnactedStep(
                step=s["step"],
                tool=s["tool"],
                args=s.get("args", {}),
                observation=s.get("observation", ""),
                is_canary_access=s.get("is_canary_access", False),
            )
            for s in steps_d
        ]

        class _MinimalGroundTruth:
            covert_enacted = False
            canaries_touched: list[str] = []
            first_covert_step = None
            label = 0

        class _Traj:
            trajectory_id = trajectory_data.get("trajectory_id", "unknown")
            stated = StatedChannel(
                plan=stated_d.get("plan", ""),
                reasoning=stated_d.get("reasoning", ""),
                declared_whitelist=stated_d.get("declared_whitelist", []),
                raw_turns=stated_d.get("raw_turns", []),
            )
            enacted = EnactedChannel(steps=steps)
            ground_truth = _MinimalGroundTruth()

        traj = _Traj()
        cg = CrossGuard().score(traj)  # type: ignore[arg-type]
        on = OnlineMonitor().score(traj)  # type: ignore[arg-type]
        db = DebateMonitor().score(traj)  # type: ignore[arg-type]

        col1, col2, col3 = st.columns(3)
        col1.metric("CrossGuard", f"{cg.score:.3f}", help="Rule-based divergence")
        col2.metric("Online", f"{on.score:.3f}", help="Streaming prefix score")
        col3.metric("Debate", f"{db.score:.3f}", help="Two-judge cross-examination")

        all_flags = list({*cg.flags, *on.flags, *db.flags})
        if all_flags:
            st.warning("Flags: " + ", ".join(all_flags))

    except Exception as exc:
        st.error(f"Monitor scoring failed: {exc}")


if __name__ == "__main__":
    main()
