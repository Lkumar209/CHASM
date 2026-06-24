"""Tests for induction conditions and ladder."""

from chasm.agents.trajectory import (
    EnactedChannel,
    SandboxGroundTruth,
    StatedChannel,
    Trajectory,
    TrajectoryMeta,
)
from chasm.induction.conditions import (
    ALL_CONDITIONS,
    InductionConfig,
    make_induction_configs,
    should_discard,
)


def _dummy_trajectory(covert_enacted: bool) -> Trajectory:
    return Trajectory(
        meta=TrajectoryMeta(
            task_id="t", family="f", split="train", condition="naive_divergent", seed=0
        ),
        stated=StatedChannel(),
        enacted=EnactedChannel(),
        ground_truth=SandboxGroundTruth(covert_enacted=covert_enacted, label=int(covert_enacted)),
    )


def test_all_conditions_present() -> None:
    assert "aligned" in ALL_CONDITIONS
    assert "naive_divergent" in ALL_CONDITIONS
    assert "evasion_L0" in ALL_CONDITIONS
    assert "evasion_L2" in ALL_CONDITIONS


def test_make_induction_configs_count() -> None:
    configs = make_induction_configs(
        conditions=["aligned", "naive_divergent"], eval_aware_variants=True
    )
    assert len(configs) == 4  # 2 conditions x 2 (aware/unaware)


def test_aligned_has_no_covert_objective() -> None:
    cfg = InductionConfig(condition="aligned")
    assert not cfg.has_covert_objective()


def test_divergent_has_covert_objective() -> None:
    cfg = InductionConfig(condition="naive_divergent")
    assert cfg.has_covert_objective()


def test_should_discard_aligned_never() -> None:
    cfg = InductionConfig(condition="aligned")
    t = _dummy_trajectory(covert_enacted=False)
    assert not should_discard(t, cfg)


def test_should_discard_divergent_not_enacted() -> None:
    cfg = InductionConfig(condition="naive_divergent")
    t = _dummy_trajectory(covert_enacted=False)
    assert should_discard(t, cfg)


def test_should_keep_divergent_enacted() -> None:
    cfg = InductionConfig(condition="naive_divergent")
    t = _dummy_trajectory(covert_enacted=True)
    assert not should_discard(t, cfg)
