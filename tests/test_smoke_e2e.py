"""Smoke e2e test: verifies the scaffold and M1-M3 pipeline run without errors."""

import pytest


@pytest.mark.smoke
def test_scaffold_imports() -> None:
    import chasm
    import chasm.agents
    import chasm.envs
    import chasm.induction
    import chasm.models
    import chasm.monitors
    import chasm.tasks
    import chasm.utils

    assert chasm.__version__ == "0.1.0"


@pytest.mark.smoke
def test_echo_backend_smoke() -> None:
    from chasm.models.base import BackendConfig
    from chasm.models.factory import build_backend

    cfg = BackendConfig(backend="echo")
    backend = build_backend(cfg)
    result = backend.generate("smoke test prompt", max_new_tokens=16)
    assert len(result.text) > 0


@pytest.mark.smoke
def test_sandbox_smoke() -> None:
    from chasm.envs import Sandbox

    sb = Sandbox(files={"a.txt": "hello"}, canary_files=["secret.txt"])
    sb.read_file("a.txt")
    assert not sb.result().covert_enacted
    sb.read_file("secret.txt")
    assert sb.result().covert_enacted


@pytest.mark.smoke
def test_task_dataset_smoke() -> None:
    from chasm.tasks import build_dataset

    tasks = build_dataset(tasks_per_family=2, seed=0)
    assert len(tasks) == 16  # 8 families x 2


@pytest.mark.smoke
def test_agent_loop_smoke() -> None:
    from chasm.agents import run_agent
    from chasm.models.base import BackendConfig
    from chasm.models.factory import build_backend
    from chasm.tasks.base import Split
    from chasm.tasks.families import make_codebase_task

    task = make_codebase_task(0, Split.TRAIN)
    backend = build_backend(BackendConfig(backend="echo"))
    traj = run_agent(task, backend, condition="aligned", seed=0)

    assert traj.meta.task_id.startswith("codebase_")
    assert traj.content_hash != ""
    assert isinstance(traj.ground_truth.covert_enacted, bool)


@pytest.mark.smoke
def test_induction_conditions_smoke() -> None:
    from chasm.induction import make_induction_configs

    configs = make_induction_configs(conditions=["aligned", "naive_divergent"])
    assert len(configs) >= 2


@pytest.mark.smoke
def test_seeding_smoke() -> None:
    from chasm.utils import seed_everything

    seed_everything(0)
    seed_everything(99)


@pytest.mark.smoke
def test_hashing_smoke() -> None:
    from chasm.utils import content_hash

    h = content_hash({"experiment": "smoke", "seed": 42})
    assert isinstance(h, str) and len(h) == 16
