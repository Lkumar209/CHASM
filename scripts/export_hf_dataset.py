"""
Export CHASM trajectories to HuggingFace datasets format.

Usage:
    uv run python scripts/export_hf_dataset.py \
        --trajectories data/trajectories/ \
        --output data/hf_export/ \
        [--push-to-hub <org>/<repo>]

Output:
    data/hf_export/
        train/          parquet shards
        dev/
        test/
        transfer/
        dataset_card.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_trajectories(store_dir: Path) -> list[dict]:
    """Load trajectories from parquet store, fall back to JSON files."""
    records = []
    parquet_files = sorted(store_dir.glob("**/*.parquet"))
    if parquet_files:
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415
            for f in parquet_files:
                table = pq.read_table(f)
                records.extend(table.to_pydict_rows() if hasattr(table, "to_pydict_rows") else [])
            return records
        except ImportError:
            pass

    # Fallback: JSON files
    for f in sorted(store_dir.glob("**/*.json")):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list):
                records.extend(data)
            else:
                records.append(data)
        except Exception:
            pass
    return records


def export(trajectories_dir: Path, output_dir: Path, push_to_hub: str | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_trajectories(trajectories_dir)

    if not records:
        print(f"No trajectories found in {trajectories_dir}. Export skipped.")
        return

    # Split by split field
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": [], "transfer": []}
    for r in records:
        split = r.get("meta", {}).get("split", "train") if isinstance(r, dict) else "train"
        splits.setdefault(split, []).append(r)

    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415

        for split_name, split_records in splits.items():
            if not split_records:
                continue
            split_dir = output_dir / split_name
            split_dir.mkdir(exist_ok=True)
            # Flatten to simple schema for HF
            rows = _flatten_records(split_records)
            table = pa.table(rows)
            pq.write_table(table, split_dir / "data.parquet")
            print(f"Exported {len(rows.get('trajectory_id', []))} records to {split_dir}")

    except ImportError:
        # Write as JSON instead
        for split_name, split_records in splits.items():
            if not split_records:
                continue
            out_file = output_dir / f"{split_name}.json"
            out_file.write_text(json.dumps(split_records, indent=2, default=str))
            print(f"Exported {len(split_records)} records to {out_file}")

    _write_dataset_card(output_dir, splits)

    if push_to_hub:
        _push_to_hub(output_dir, push_to_hub)


def _flatten_records(records: list[dict]) -> dict[str, list]:
    """Flatten trajectory records to columnar format for Arrow."""
    rows: dict[str, list] = {
        "trajectory_id": [],
        "task_id": [],
        "family": [],
        "split": [],
        "condition": [],
        "covert_enacted": [],
        "label": [],
        "stated_plan": [],
        "stated_reasoning": [],
        "n_enacted_steps": [],
        "n_canary_steps": [],
        "primary_score": [],
    }
    for r in records:
        if not isinstance(r, dict):
            continue
        meta = r.get("meta", {})
        stated = r.get("stated", {})
        enacted = r.get("enacted", {})
        gt = r.get("ground_truth", {})
        steps = enacted.get("steps", [])

        rows["trajectory_id"].append(r.get("trajectory_id", ""))
        rows["task_id"].append(meta.get("task_id", ""))
        rows["family"].append(meta.get("family", ""))
        rows["split"].append(meta.get("split", ""))
        rows["condition"].append(meta.get("condition", ""))
        rows["covert_enacted"].append(bool(gt.get("covert_enacted", False)))
        rows["label"].append(int(gt.get("label", 0)))
        rows["stated_plan"].append(stated.get("plan", ""))
        rows["stated_reasoning"].append(stated.get("reasoning", ""))
        rows["n_enacted_steps"].append(len(steps))
        rows["n_canary_steps"].append(sum(1 for s in steps if s.get("is_canary_access")))
        rows["primary_score"].append(float(r.get("primary_score", 0.0)))

    return rows


def _write_dataset_card(output_dir: Path, splits: dict[str, list]) -> None:
    total = sum(len(v) for v in splits.values())
    split_rows = "\n".join(f"| {k} | {len(v)} |" for k, v in splits.items())
    card = (
        "---\n"
        "license: cc-by-4.0\n"
        "task_categories:\n"
        "- text-classification\n"
        "language:\n"
        "- en\n"
        "tags:\n"
        "- safety\n"
        "- agents\n"
        "- alignment\n"
        "- divergence-detection\n"
        "---\n\n"
        "# CHASM — Cross-cHannel Agent Safety Monitoring Dataset\n\n"
        "Trajectories from RIFT-Bench for stated-vs-enacted divergence detection.\n\n"
        "## Statistics\n\n"
        "| Split | Count |\n"
        "|-------|-------|\n"
        f"{split_rows}\n"
        f"| **Total** | **{total}** |\n\n"
        "## Fields\n\n"
        "- `trajectory_id`: unique trajectory identifier\n"
        "- `task_id`, `family`, `split`, `condition`: metadata\n"
        "- `covert_enacted` / `label`: ground-truth divergence label\n"
        "- `stated_plan`, `stated_reasoning`: stated channel text\n"
        "- `n_enacted_steps`, `n_canary_steps`: enacted channel summary\n"
        "- `primary_score`: primary task completion score\n\n"
        "## License\n\n"
        "CC BY 4.0. Canary objectives are abstract markers only — no real secrets.\n"
    )
    (output_dir / "dataset_card.md").write_text(card)


def _push_to_hub(output_dir: Path, repo_id: str) -> None:
    try:
        from datasets import load_dataset  # noqa: PLC0415
        ds = load_dataset(str(output_dir))
        ds.push_to_hub(repo_id)
        print(f"Pushed to HuggingFace Hub: {repo_id}")
    except ImportError:
        print("datasets library not installed. Skipping Hub push.")
    except Exception as exc:
        print(f"Hub push failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CHASM trajectories to HF format")
    parser.add_argument("--trajectories", default="data/trajectories/", help="Trajectory store directory")
    parser.add_argument("--output", default="data/hf_export/", help="Output directory")
    parser.add_argument("--push-to-hub", default=None, help="HF Hub repo id (org/repo)")
    args = parser.parse_args()

    export(
        trajectories_dir=Path(args.trajectories),
        output_dir=Path(args.output),
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
