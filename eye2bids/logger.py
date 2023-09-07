"""General logger for the eye2bids package."""
from __future__ import annotations

import logging

from rich.logging import RichHandler


def eye2bids_logger(log_level: str = "INFO") -> logging.Logger:
    """Create a logger for the eye2bids package."""
    FORMAT = "%(message)s"

    logging.basicConfig(
        level=log_level,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler()],
    )

    return logging.getLogger("cohort_creator")
