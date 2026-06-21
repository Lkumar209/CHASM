"""Structured logging via loguru."""

from __future__ import annotations

import os
import sys
from typing import Any

from loguru import logger


def get_logger(name: str) -> Any:
    """Return a loguru logger bound with the module name."""
    level = os.environ.get("CHASM_LOG_LEVEL", "INFO").upper()
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[name]}</cyan> | {message}",
    )
    return logger.bind(name=name)
