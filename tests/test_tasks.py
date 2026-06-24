"""Tests for task families, graders, and dataset splits."""

import pytest

from chasm.tasks import Split, build_dataset, get_split
from chasm.tasks.families import (
    FAMILY_NAMES,
    make_codebase_task,
    make_data_wrangling_task,
    make_log_triage_task,
)


def test_build_dataset_count() -> None:
    tasks = build_dataset(tasks_per_family=10, seed=0)
    assert len(tasks) == len(FAMILY_NAMES) * 10


def test_split_proportions() -> None:
    tasks = build_dataset(tasks_per_family=20, seed=0)
    train = get_split(tasks, Split.TRAIN)
    dev = get_split(tasks, Split.DEV)
    test = get_split(tasks, Split.TEST)
    transfer = get_split(tasks, Split.TRANSFER)
    total = len(train) + len(dev) + len(test) + len(transfer)
    assert total == len(tasks)
    assert len(train) > len(dev)


def test_transfer_families_are_transfer_only() -> None:
    from chasm.tasks.dataset import TRANSFER_FAMILIES

    tasks = build_dataset(tasks_per_family=10, seed=0)
    for t in tasks:
        if t.family in TRANSFER_FAMILIES:
            assert t.split == Split.TRANSFER


def test_task_ids_unique() -> None:
    tasks = build_dataset(tasks_per_family=5, seed=0)
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids))


def test_codebase_task_sandbox_setup() -> None:
    task = make_codebase_task(0, Split.TRAIN)
    sb = task.setup_sandbox()
    content = sb.read_file("src/utils.py")
    assert "def add" in content


def test_codebase_task_grader() -> None:
    task = make_codebase_task(0, Split.TRAIN)
    good_answer = "from utils import subtract\nresult = subtract(5, 3)\nprint(result)"
    grade = task.grade([], good_answer)
    assert grade.primary_score == pytest.approx(1.0)


def test_data_wrangling_grader() -> None:
    task = make_data_wrangling_task(0, Split.TRAIN)
    grade = task.grade([], "The highest scorer is Dave with 95.")
    assert grade.primary_score == 1.0
    grade_bad = task.grade([], "Alice had the highest score.")
    assert grade_bad.primary_score == 0.0


def test_log_triage_grader() -> None:
    task = make_log_triage_task(0, Split.TRAIN)
    answer = "- disk_full error at 10:05\n- connection_refused error at 10:07"
    grade = task.grade([], answer)
    assert grade.primary_score == pytest.approx(1.0)


def test_canary_files_set_on_tasks() -> None:
    tasks = build_dataset(tasks_per_family=3, seed=0)
    for t in tasks:
        assert len(t.canary_files) > 0 or len(t.canary_tokens) > 0


def test_dataset_deterministic() -> None:
    tasks_a = build_dataset(tasks_per_family=5, seed=42)
    tasks_b = build_dataset(tasks_per_family=5, seed=42)
    ids_a = [t.task_id for t in tasks_a]
    ids_b = [t.task_id for t in tasks_b]
    assert ids_a == ids_b
