"""Shared comparison orchestration, independent of CLI and experiment tracking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Sequence

from engagement_prediction.data.timestamps import parse_utc_datetime
from engagement_prediction.evaluation import artifacts, comparison
from engagement_prediction.evaluation.artifacts import ModelArtifact, Stage7Artifact
from engagement_prediction.evaluation.comparison import ComparisonSettings
from engagement_prediction.evaluation.workflow_reports import (
    ProductionReportContext,
    StandaloneReportContext,
    write_production_report,
    write_standalone_report,
)


HOLDOUT_SPLITS = ("holdout_seen_users", "holdout_unseen_users")
DATE_FIELDS = (
    "posts_start", "posts_end", "train_start", "val_start", "holdout_start", "holdout_end",
)


def _upstream_summary(dataset: Stage7Artifact, stage_folder: str, input_name: str) -> dict:
    """Follow this dataset's recorded lineage instead of the latest-stage links."""
    recorded = dataset.manifest.get("inputs", {}).get(stage_folder)
    if not recorded:
        recorded = dataset.summary.get("input", {}).get(input_name)
    if not recorded:
        raise ValueError(f"Stage 7 does not record its {stage_folder} input")
    root = Path(recorded)
    if not root.is_absolute():
        root = dataset.root / root
    path = root / "summary.json"
    summary = json.loads(path.read_text())
    if not isinstance(summary, dict):
        raise ValueError(f"Expected a summary object at {path}")
    return summary


def _holdout_dates(dataset: Stage7Artifact) -> dict[str, str | None]:
    summary = _upstream_summary(
        dataset, "01_query_selection", "query_selection_dir"
    )
    parameters = summary["parameters"]
    dates = {
        name: parse_utc_datetime(
            summary.get(name) if name in ("posts_start", "posts_end") else parameters.get(name),
            field_name=name,
        )
        for name in DATE_FIELDS
    }
    dates["holdout_end"] = dates["holdout_end"] or dates["posts_end"]
    start, end = dates["holdout_start"], dates["holdout_end"]
    if start is None or end is None or start >= end:
        raise ValueError("Production comparison requires a configured, nonempty holdout window")
    return {name: value.isoformat() if value else None for name, value in dates.items()}


def _history_extraction_limit(dataset: Stage7Artifact) -> int:
    limit = dataset.summary.get("parameters", {}).get("max_history_posts_per_query")
    if limit is None:
        limit = _upstream_summary(
            dataset, "02_user_history", "user_history_dir"
        )["parameters"].get("max_history_posts_per_query")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Dataset history extraction limit must be a positive integer")
    return limit


def validate_history_capacity(dataset: Stage7Artifact, required_history_len: int) -> int:
    """Reject source histories that would silently truncate either compared model."""
    extraction_limit = _history_extraction_limit(dataset)
    if extraction_limit < required_history_len:
        raise ValueError(
            f"Stage 7 histories were extracted with limit {extraction_limit}, but "
            f"the compared models require {required_history_len}; rebuild the dataset "
            "with a sufficient max_history_posts_per_query"
        )
    return extraction_limit


def validate_holdout_dataset(
    dataset: Stage7Artifact, required_history_len: int
) -> tuple[dict[str, str | None], int]:
    """Validate a reused or freshly prepared holdout before spending time training."""
    dates = _holdout_dates(dataset)
    counts = dataset.split_query_counts
    for split in HOLDOUT_SPLITS:
        if counts.get(split, 0) <= 0:
            raise ValueError(f"Production comparison requires a nonempty {split} split")
    return dates, validate_history_capacity(dataset, required_history_len)


def run_comparison_workflow(
    *,
    dataset: Stage7Artifact,
    models: Sequence[ModelArtifact],
    settings: ComparisonSettings,
    output_dir: Path,
    report_context: StandaloneReportContext | ProductionReportContext,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Validate, evaluate, and report without resolving models or uploading files.

    The explicit context preserves each entry point's existing validation and
    output policy. In particular, standalone comparisons do not acquire the
    automated run's holdout/history requirements or its full-precision JSON.
    Candidate construction, sampling, and scoring remain in the existing engine.
    """
    if isinstance(report_context, ProductionReportContext):
        if models[0].name != "production":
            raise ValueError("The pinned baseline must be named 'production'")
        if any(model.model_type != "bst-ranker" for model in models):
            raise ValueError("Production comparison supports BST rankers only")
        dates, extraction_limit = validate_holdout_dataset(
            dataset, max(model.max_history_len for model in models),
        )
    elif isinstance(report_context, StandaloneReportContext):
        artifacts.validate_comparison_contract(dataset, models, settings.max_history_len)
    else:
        raise TypeError("Unsupported comparison report context")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="author-mappings-", dir=output_dir) as temporary_dir:
        result = comparison.run_model_comparison(
            dataset=dataset, models=models, settings=settings,
            temporary_dir=Path(temporary_dir), logger=logger,
        )

    if isinstance(report_context, ProductionReportContext):
        return write_production_report(
            dataset=dataset, models=models, settings=settings, result=result,
            output_dir=output_dir, context=report_context,
            dates=dates, history_extraction_limit=extraction_limit,
        )
    return write_standalone_report(
        dataset=dataset, models=models, settings=settings, result=result,
        output_dir=output_dir, context=report_context,
    )
