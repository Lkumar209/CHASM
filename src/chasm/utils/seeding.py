"""Deterministic seeding for reproducibility."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch with the given seed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
