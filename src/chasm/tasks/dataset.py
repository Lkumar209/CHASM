"""Build and persist RIFT-Bench task splits."""

from __future__ import annotations

import json
import random
from pathlib import Path

from chasm.tasks.base import Split, TaskSpec
from chasm.tasks.families import FAMILY_BUILDERS, FAMILY_NAMES

# Split proportions over task INSTANCES per family:
# train=60%, dev=15%, test=15%, transfer=10%
# Transfer families: the last 2 families are held out (used only at M11)
TRANSFER_FAMILIES = {"log_triage", "long_horizon_planning"}
TASKS_PER_FAMILY = 20  # instances per family (varied by idx)


def build_dataset(tasks_per_family: int = TASKS_PER_FAMILY, seed: int = 0) -> list[TaskSpec]:
    """Generate all task instances with deterministic splits."""
    rng = random.Random(seed)
    all_tasks: list[TaskSpec] = []

    for builder, family_name in zip(FAMILY_BUILDERS, FAMILY_NAMES, strict=True):
        indices = list(range(tasks_per_family))
        rng.shuffle(indices)

        n_train = int(tasks_per_family * 0.60)
        n_dev = int(tasks_per_family * 0.15)
        n_test = int(tasks_per_family * 0.15)
        # remainder goes to transfer

        if family_name in TRANSFER_FAMILIES:
            split_assignments = [Split.TRANSFER] * tasks_per_family
        else:
            split_assignments = (
                [Split.TRAIN] * n_train
                + [Split.DEV] * n_dev
                + [Split.TEST] * n_test
                + [Split.TRANSFER] * (tasks_per_family - n_train - n_dev - n_test)
            )

        for i, split in zip(indices, split_assignments, strict=True):
            task = builder(i, split)
            all_tasks.append(task)

    return all_tasks


def save_dataset(tasks: list[TaskSpec], output_dir: str | Path = "data/tasks") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for t in tasks:
        records.append({
            "task_id": t.task_id,
            "family": t.family,
            "split": t.split.value,
            "primary_objective": t.primary_objective,
            "canary_files": t.canary_files,
            "canary_tokens": t.canary_tokens,
            "metadata": t.metadata,
        })
    with open(output / "tasks.json", "w") as f:
        json.dump(records, f, indent=2)
    # per-split summary
    from collections import Counter
    counts = Counter((r["family"], r["split"]) for r in records)
    summary = {f"{fam}/{spl}": cnt for (fam, spl), cnt in sorted(counts.items())}
    with open(output / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def get_split(tasks: list[TaskSpec], split: Split) -> list[TaskSpec]:
    return [t for t in tasks if t.split == split]
