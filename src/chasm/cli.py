"""Main CLI entry point — wraps Hydra experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../configs", config_name="experiment/smoke")
def main(cfg: DictConfig) -> None:
    from chasm.utils import get_logger, seed_everything

    log = get_logger("cli")
    seed_everything(cfg.seed)
    log.info(f"Running experiment: {cfg.experiment_name}")
    log.info(f"Config: {cfg}")
    log.info("Scaffold OK — no pipeline stages implemented yet (M0).")


if __name__ == "__main__":
    main()
