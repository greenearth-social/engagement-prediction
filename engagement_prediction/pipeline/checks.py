"""Persist health-check results before stopping an unhealthy pipeline run."""

import json
import logging
from pathlib import Path
from typing import Any

from engagement_prediction.experiment_tracking import ExperimentTracker


def report_checks(
    report: dict[str, Any],
    *,
    path: Path,
    tracker: ExperimentTracker,
    logger: logging.Logger,
) -> None:
    """Keep a local report even when the stage fails or ClearML is unavailable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for check in report["checks"]:
        level = {"pass": logging.INFO, "warn": logging.WARNING, "fail": logging.ERROR}[check["status"]]
        logger.log(level, "%s", check["message"])
    logger.info("Check report (%s): %s", report["status"], path)
    try:
        uploaded = tracker.log_file_artifact(path.stem, path)
        if tracker.id and not uploaded:
            logger.warning("Could not upload check report %s; the local report is retained", path)
    except Exception as exc:
        # Reporting must not hide the original data or disk failure.
        logger.warning("Could not upload check report %s: %s", path, exc)
    failures = [check["message"] for check in report["checks"] if check["status"] == "fail"]
    if failures:
        reason = "Pipeline checks failed: " + "; ".join(failures) + f". Report: {path}"
        try:
            tracker.mark_failed(reason)
        except Exception as exc:
            logger.warning("Could not mark tracker task failed: %s", exc)
        raise ValueError(reason)
