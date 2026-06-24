"""
Full CHASM evaluation script.

Usage:
    # Deterministic sim run (no API key needed, produces real metric numbers):
    uv run python scripts/run_eval.py --backend sim --n-tasks 8 --n-seeds 3

    # Real eval with Anthropic (set ANTHROPIC_API_KEY):
    uv run python scripts/run_eval.py --backend api --model claude-haiku-4-5-20251001 \\
        --n-tasks 10 --n-seeds 3

    # Update README with results:
    uv run python scripts/run_eval.py --backend sim --n-tasks 8 --n-seeds 3 --update-readme
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path

# ensure src/ is on path when run as script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chasm.agents.loop import run_agent
from chasm.eval.metrics import compute_all_metrics
from chasm.eval.report import evaluate_hypotheses, generate_figures, write_report
from chasm.induction.conditions import InductionConfig, should_discard
from chasm.models.base import BackendConfig
from chasm.models.factory import build_backend
from chasm.monitors.action_judge import ActionJudge
from chasm.monitors.cot_judge import CoTJudge
from chasm.monitors.crossguard import CrossGuard
from chasm.monitors.debate import DebateMonitor
from chasm.monitors.ensemble import EnsembleMonitor
from chasm.monitors.online import OnlineMonitor
from chasm.tasks import Split, build_dataset
from chasm.utils.logging import get_logger

log = get_logger("run_eval")

# Sim backends per condition (sim_l1/sim_l2 produce stronger evasion signals)
SIM_BACKEND_FOR: dict[str, str] = {
    "aligned": "sim",
    "naive_divergent": "sim",
    "evasion_L0": "sim",
    "evasion_L1": "sim_l1",
    "evasion_L2": "sim_l2",
}


def _make_backend(backend_type: str, model: str, provider: str) -> object:
    cfg = BackendConfig(
        backend=backend_type,
        model_name=model,
        api_provider=provider,
        temperature=0.0,
        max_new_tokens=512,
    )
    return build_backend(cfg)


def _make_monitors(backend_type: str, model: str, provider: str) -> dict:
    if backend_type.startswith("sim") or backend_type == "echo":
        debate_backend = None
    else:
        debate_backend = _make_backend(backend_type, model, provider)  # type: ignore[assignment]

    return {
        "crossguard": CrossGuard(),
        "action_judge": ActionJudge(),
        "cot_judge": CoTJudge(),
        "online": OnlineMonitor(),
        "debate": DebateMonitor(backend=debate_backend),  # type: ignore[arg-type]
    }


def collect_trajectories(
    tasks: list,
    condition: str,
    backend_type: str,
    model: str,
    provider: str,
    seeds: list[int],
    max_steps: int,
) -> list:
    """Collect trajectories for one condition across all seeds."""
    # For sim backends, pick the condition-appropriate backend
    if backend_type in ("sim", "sim_l1", "sim_l2"):
        actual_backend_type = SIM_BACKEND_FOR.get(condition, "sim")
    else:
        actual_backend_type = backend_type

    backend = _make_backend(actual_backend_type, model, provider)
    ic = InductionConfig(condition=condition)
    trajs = []
    for task in tasks:
        for seed in seeds:
            traj = run_agent(task, backend, condition=condition, seed=seed, max_steps=max_steps)
            if condition != "aligned" and should_discard(traj, ic):
                log.debug(f"  discarded {task.task_id} seed={seed} (covert not enacted)")
                continue
            trajs.append(traj)
    log.info(f"  [{condition}] {len(trajs)} trajectories collected")
    return trajs


def score_monitor(trajs: list, monitor: object) -> tuple[list[float], list[int]]:
    scores, labels = [], []
    for traj in trajs:
        result = monitor.score(traj)  # type: ignore[union-attr]
        scores.append(result.score)
        labels.append(traj.ground_truth.label)
    return scores, labels


def run_condition_eval(
    tasks: list,
    aligned_trajs: list,
    condition: str,
    backend_type: str,
    model: str,
    provider: str,
    seeds: list[int],
    max_steps: int,
    n_bootstrap: int,
    monitors: dict,
) -> dict:
    """Evaluate all monitors for aligned vs one divergent condition."""
    log.info(f"\n--- condition: {condition} ---")
    pos_trajs = collect_trajectories(tasks, condition, backend_type, model, provider, seeds, max_steps)

    if not pos_trajs:
        log.warning(f"  No valid trajectories for {condition} — all discarded.")
        return {}

    all_trajs = aligned_trajs + pos_trajs
    cond_results: dict[str, dict] = {}

    # Calibration: first 20% of pooled trajectories as proxy dev split
    cal_n = max(4, len(all_trajs) // 5)
    cal_trajs = all_trajs[:cal_n]
    cal_labels = [t.ground_truth.label for t in cal_trajs]

    # Fit ensemble
    base = [monitors["crossguard"], monitors["action_judge"], monitors["cot_judge"]]
    ensemble = EnsembleMonitor(base)

    for name, monitor in {**monitors, "ensemble": ensemble}.items():
        with contextlib.suppress(Exception):
            monitor.calibrate(cal_trajs, cal_labels)

        scores, labels = score_monitor(all_trajs, monitor)
        if len(set(labels)) < 2:
            log.warning(f"  {name}: only one class present — skipping metrics")
            continue

        m = compute_all_metrics(labels, scores, n_bootstrap=n_bootstrap)
        cond_results[name] = m
        log.info(
            f"  {name:20}: AUROC={m['auroc']:.3f} [{m['auroc_ci_lo']:.3f},{m['auroc_ci_hi']:.3f}]"
            f"  AUPRC={m['auprc']:.3f}  TPR@1%={m['tpr_at_fpr1']:.3f}"
            f"  n={m['n']} pos={m['n_positive']}"
        )

    return cond_results


def format_ci(m: dict, key: str = "auroc") -> str:
    v = m.get(key)
    lo = m.get(f"{key}_ci_lo")
    hi = m.get(f"{key}_ci_hi")
    if v is None:
        return "—"
    if lo is not None and hi is not None:
        return f"{v:.3f} [{lo:.3f},{hi:.3f}]"
    return f"{v:.3f}"


def update_readme(results: dict, readme_path: Path) -> None:
    """Replace the placeholder metric table in README.md with actual numbers."""
    readme = readme_path.read_text()

    monitor_display = {
        "action_judge": "ActionJudge",
        "cot_judge": "CoTJudge",
        "crossguard": "**CrossGuard**",
        "online": "OnlineMonitor",
        "ensemble": "EnsembleMonitor",
        "debate": "DebateMonitor",
    }

    nd = results.get("naive_divergent", {})
    l2 = results.get("evasion_L2", {})

    rows = []
    for key, display in monitor_display.items():
        nd_str = format_ci(nd.get(key, {}))
        l2_str = f"{l2.get(key, {}).get('auroc', None):.3f}" if l2.get(key, {}).get("auroc") else "—"
        tpr = nd.get(key, {}).get("tpr_at_fpr1")
        tpr_str = f"{tpr:.3f}" if tpr is not None else "—"
        rows.append(f"| {display} | {nd_str} | {l2_str} | {tpr_str} |")

    new_table = (
        "| Monitor | AUROC (naive\\_divergent) | AUROC (evasion\\_L2) | TPR@FPR=1% |\n"
        "|---------|--------------------------|----------------------|------------|\n"
        + "\n".join(rows)
        + "\n"
    )

    # Only match the benchmark table (stops at a blank line or a ## section)
    old_pattern = r"(\| Monitor \| AUROC[^\n]*\n\|[-|: ]+\|\n)(?:\|[^\n]*\n)+"
    readme = re.sub(old_pattern, new_table, readme)

    # Update hypothesis verdicts
    verdicts = evaluate_hypotheses(results)
    for v in verdicts:
        readme = re.sub(
            rf"(\| {re.escape(v.code)} ·[^|]+\|[^|]+\|) pending",
            rf"\1 **{v.verdict}** — {v.key_number}",
            readme,
        )

    readme_path.write_text(readme)
    log.info(f"README updated: {readme_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="sim", choices=["echo", "api", "local", "sim"])
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--n-tasks", type=int, default=8)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["naive_divergent", "evasion_L0", "evasion_L1", "evasion_L2"],
    )
    args = parser.parse_args()

    # Build dataset — need tasks_per_family large enough to get all splits
    tasks_per_family = max(12, args.n_tasks * 2)
    all_tasks = build_dataset(tasks_per_family=tasks_per_family, seed=42)
    train_tasks = [t for t in all_tasks if t.split == Split.TRAIN][: args.n_tasks]
    log.info(
        f"Dataset: {len(all_tasks)} total, {len(train_tasks)} TRAIN tasks selected"
        f" × {args.n_seeds} seeds"
    )
    log.info(f"Backend: {args.backend}, conditions: {args.conditions}")

    seeds = list(range(args.n_seeds))
    monitors = _make_monitors(args.backend, args.model, args.provider)

    # Collect aligned trajectories once — they're the negatives for all conditions
    log.info("\nCollecting aligned trajectories...")
    aligned_trajs = collect_trajectories(
        train_tasks, "aligned", args.backend, args.model, args.provider, seeds, args.max_steps
    )
    if not aligned_trajs:
        log.error("No aligned trajectories collected — cannot evaluate.")
        sys.exit(1)

    # Evaluate each divergent condition
    results: dict[str, dict] = {}
    for condition in args.conditions:
        if condition == "aligned":
            continue
        cond_results = run_condition_eval(
            train_tasks,
            aligned_trajs,
            condition,
            args.backend,
            args.model,
            args.provider,
            seeds,
            args.max_steps,
            args.n_bootstrap,
            monitors,
        )
        if cond_results:
            results[condition] = cond_results

    # Save raw metrics
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    log.info(f"\nMetrics saved → {out}/metrics.json")

    # Summary table
    print("\n=== AUROC Summary ===")
    col_w = 12
    header = f"{'Monitor':<22}" + "".join(f"{c[:col_w]:>{col_w}}" for c in args.conditions)
    print(header)
    print("-" * len(header))
    for mon in ["crossguard", "action_judge", "cot_judge", "online", "ensemble", "debate"]:
        row = f"{mon:<22}"
        for cond in args.conditions:
            auc = results.get(cond, {}).get(mon, {}).get("auroc")
            row += f" {auc:>{col_w-1}.3f}" if auc is not None else f" {'—':>{col_w-1}}"
        print(row)

    # Hypothesis verdicts
    verdicts = evaluate_hypotheses(results)
    print("\n=== Hypothesis Verdicts ===")
    for v in verdicts:
        print(f"  {v.code}: {v.verdict:10} — {v.key_number}")

    # Write REPORT.md + figures
    figures = generate_figures(results)
    write_report(verdicts, figures, results, out / "REPORT.md")
    (out / "figures.json").write_text(json.dumps(figures, indent=2))
    print(f"\nReport → {out}/REPORT.md")

    # Update README
    if args.update_readme:
        readme_path = Path(__file__).parent.parent / "README.md"
        update_readme(results, readme_path)


if __name__ == "__main__":
    main()
