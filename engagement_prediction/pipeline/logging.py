"""Logging helpers shared by pipeline stages."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from engagement_prediction.pipeline.core import Context


# Loggers are reused within a process so repeated stage setup does not add
# duplicate console or file handlers.
_stage_loggers: dict[str, logging.Logger] = {}


def get_stage_logger(
    stage_name: str,
    log_file: Path | None = None,
) -> logging.Logger:
    """Return the process-local logger for a pipeline stage."""
    if stage_name in _stage_loggers:
        return _stage_loggers[stage_name]

    logger = logging.getLogger(stage_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    _stage_loggers[stage_name] = logger
    return logger


def log_operation_start(
    operation_name: str,
    stage_name: str,
    logger: logging.Logger | None = None,
) -> logging.Logger:
    """Log a visible boundary before a major stage operation begins."""
    if logger is None:
        logger = get_stage_logger(stage_name)
    logger.info("=" * 60)
    logger.info("Starting: %s", operation_name)
    return logger


def log_prior_stage_inputs(
    context: Context,
    logger: logging.Logger,
    *,
    header: str = "Resolved prior inputs used",
) -> None:
    """Log the concrete prior-stage artifact directories used by a stage."""
    prior_inputs = context.get_active_stage_inputs()
    if not prior_inputs:
        logger.info("%s: none", header)
        return

    logger.info("%s:", header)
    for folder, path in sorted(prior_inputs.items()):
        logger.info("  %s: %s", folder, path)
