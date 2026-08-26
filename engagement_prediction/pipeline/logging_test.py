from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, call

import pytest

from engagement_prediction.pipeline import logging as pipeline_logging


@pytest.fixture(autouse=True)
def _close_test_loggers():
    yield
    for stage_name in list(pipeline_logging._stage_loggers):
        if not stage_name.startswith("TEST_"):
            continue
        logger = pipeline_logging._stage_loggers.pop(stage_name)
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()


def test_get_stage_logger_reuses_logger_and_writes_file(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "stage.log"

    logger = pipeline_logging.get_stage_logger("TEST_FILE_LOGGER", log_path)
    logger.info("hello")

    assert pipeline_logging.get_stage_logger("TEST_FILE_LOGGER") is logger
    assert "[TEST_FILE_LOGGER] hello" in log_path.read_text()


def test_log_operation_start_uses_supplied_logger() -> None:
    logger = Mock()

    returned = pipeline_logging.log_operation_start(
        "Build artifacts",
        "TEST_OPERATION",
        logger,
    )

    assert returned is logger
    assert logger.info.call_args_list == [
        call("=" * 60),
        call("Starting: %s", "Build artifacts"),
    ]


def test_log_prior_stage_inputs_reports_sorted_paths() -> None:
    context = Mock()
    context.get_active_stage_inputs.return_value = {
        "02_second": Path("/tmp/second"),
        "01_first": Path("/tmp/first"),
    }
    logger = Mock()

    pipeline_logging.log_prior_stage_inputs(context, logger)

    assert logger.info.call_args_list == [
        call("%s:", "Resolved prior inputs used"),
        call("  %s: %s", "01_first", Path("/tmp/first")),
        call("  %s: %s", "02_second", Path("/tmp/second")),
    ]


def test_log_prior_stage_inputs_reports_no_inputs() -> None:
    context = Mock()
    context.get_active_stage_inputs.return_value = {}
    logger = Mock()

    pipeline_logging.log_prior_stage_inputs(context, logger, header="Inputs")

    logger.info.assert_called_once_with("%s: none", "Inputs")
