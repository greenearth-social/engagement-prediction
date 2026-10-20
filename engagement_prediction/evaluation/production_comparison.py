"""Evaluate a candidate and a pinned production model on the candidate holdout."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import tempfile
from typing import Any

from engagement_prediction.data.timestamps import parse_utc_datetime
from engagement_prediction.evaluation.artifacts import (
    ModelArtifact,
    Stage7Artifact,
    resolve_model_artifact,
    resolve_stage7_artifact,
)
from engagement_prediction.evaluation.comparison import (
    ComparisonSettings,
    run_model_comparison,
)
from engagement_prediction.evaluation.reporting import (
    build_metric_deltas,
    write_metric_deltas_csv,
    write_metrics_csv,
)
from engagement_prediction.experiment_tracking import ExperimentTracker
from engagement_prediction.training.runtime import get_device


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


def _write_review(path: Path, report: dict[str, Any]) -> None:
    dates = report["dates"]
    lines = [
        "# Candidate versus production",
        "",
        f"Holdout: [{dates['holdout_start']}, {dates['holdout_end']}) (UTC, end exclusive).",
        f"Production model: {report['production'].get('ranker_clearml_model_id', 'unknown')}.",
        f"Production revision: {report['production'].get('cloud_run', {}).get('revision', 'unknown')}.",
        "Both models were evaluated on the same queries, labels and candidate slates.",
        "Each model uses its own author vocabulary and history limit.",
        "",
        "| Split | Queries | Production | Candidate | Candidate minus production |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, values in report["primary_comparison"].items():
        lines.append(
            f"| {split} | {report['split_row_counts'][split]} "
            f"| {values['production']:.6g} | {values['candidate']:.6g} "
            f"| {values['delta']:+.6g} ({values['direction']}) |"
        )
    lines.extend([
        "",
        f"Metric: {report['primary_metric']}. Full metrics and deltas are in metrics.json and the CSV files.",
        "",
        "| Model | History limit | Posts mapped to unknown author |",
        "| --- | ---: | ---: |",
    ])
    for name, coverage in report["mapping_coverage"].items():
        lines.append(
            f"| {name} | {report['history_lengths'][name]} "
            f"| {coverage['model_unknown_post_count']} / {coverage['stage7_post_count']} |"
        )
    lines.extend([
        "",
        "Author coverage counts all posts in the Stage 7 bundle, not only holdout uses.",
        "Seen/unseen cohort labels come from the candidate dataset; they do not establish whether production trained on those users.",
        "",
    ])
    baseline_window = report["production"].get("training_window")
    if baseline_window:
        lines.append("Recorded production training window: " + json.dumps(baseline_window, sort_keys=True) + ".")
        lines.append("Review that window for overlap with this holdout before interpreting the comparison.")
    else:
        lines.append("Production training dates are unknown; overlap with this holdout has not been ruled out.")
    lines.extend([
        "",
        "This is an offline ranking comparison. It does not establish live service compatibility or online performance.",
        "Model review and deployment remain manual; no model was deployed by this comparison.",
    ])
    path.write_text("\n".join(lines) + "\n")


def run_production_comparison(
    *,
    candidate_dir: Path,
    production_model: ModelArtifact,
    production_metadata: dict,
    args: argparse.Namespace,
    output_dir: Path,
    tracker: ExperimentTracker,
    logger: logging.Logger,
) -> dict:
    """Save a review of both models, retaining local reports if upload fails."""
    candidate = resolve_model_artifact("candidate", candidate_dir)
    if production_model.name != "production":
        raise ValueError("The pinned baseline must be named 'production'")
    if candidate.model_type != "bst-ranker" or production_model.model_type != "bst-ranker":
        raise ValueError("Production comparison supports BST rankers only")
    dataset = resolve_stage7_artifact(Path(candidate.training_config["stage7_dir"]))
    required_limit = max(candidate.max_history_len, production_model.max_history_len)
    dates, extraction_limit = validate_holdout_dataset(dataset, required_limit)

    training_config = candidate.training_config
    settings = ComparisonSettings(
        splits=HOLDOUT_SPLITS,
        batch_size=training_config["eval_batch_size"],
        additional_batch_negatives=training_config["bst_additional_batch_negatives"],
        metrics_top_ks=tuple(training_config.get("metrics_top_ks", args.metrics_top_ks)),
        bst_candidate_chunk_size=args.production_comparison_candidate_chunk_size,
        device=get_device(args.device),
        num_dataloader_workers=args.num_dataloader_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        random_seed=training_config["random_seed"],
        max_classification_metric_pairs=args.production_comparison_max_classification_metric_pairs,
        max_history_len=None,
        disable_progress=True,
    )
    models = (production_model, candidate)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="author-mappings-", dir=output_dir) as temporary_dir:
        result = run_model_comparison(
            dataset=dataset, models=models, settings=settings,
            temporary_dir=Path(temporary_dir), logger=logger,
        )
    primary_metric = f"ndcg@{settings.metrics_top_ks[0]}"
    primary_comparison = {}
    for split in HOLDOUT_SPLITS:
        values = [result.metrics_by_model[model.name][split][primary_metric] for model in models]
        if any(value is None or not math.isfinite(float(value)) for value in values):
            raise ValueError(f"Comparison returned no finite {primary_metric} for {split}")
        production_value, candidate_value = map(float, values)
        delta = candidate_value - production_value
        primary_comparison[split] = {
            "production": production_value, "candidate": candidate_value, "delta": delta,
            "direction": "higher" if delta > 0 else "lower" if delta < 0 else "equal",
        }

    # Keep full precision in JSON and comparisons; CSV helpers round for display.
    report = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir.resolve()),
        "production": production_metadata,
        "models": [model.to_dict() for model in models],
        "dataset": dataset.to_dict(),
        "dates": dates,
        "settings": asdict(settings),
        "history_extraction_limit": extraction_limit,
        "history_lengths": result.history_lengths,
        "split_row_counts": result.split_row_counts,
        "mapping_coverage": result.mapping_coverage_by_model,
        "metrics": result.metrics_by_model,
        "metric_deltas": build_metric_deltas(
            model_a_name="production", model_b_name="candidate",
            metrics_by_model=result.metrics_by_model,
        ),
        "primary_metric": primary_metric,
        "primary_comparison": primary_comparison,
    }
    paths = {name: output_dir / name for name in (
        "metrics.json", "metrics.csv", "metric_deltas.csv", "review.md",
    )}
    report["report_paths"] = {name: str(path.resolve()) for name, path in paths.items()}
    paths["metrics.json"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_metrics_csv(paths["metrics.csv"], models=models, metrics_by_model=result.metrics_by_model)
    write_metric_deltas_csv(
        paths["metric_deltas.csv"], model_a_name="production", model_b_name="candidate",
        metrics_by_model=result.metrics_by_model,
    )
    _write_review(paths["review.md"], report)

    if tracker.id:
        for filename, path in paths.items():
            artifact_name = "production_comparison_" + filename.replace(".", "_")
            if not tracker.log_file_artifact(artifact_name, path):
                raise RuntimeError(f"Failed to upload comparison report {filename}; local report retained at {path}")
        for split, values in primary_comparison.items():
            for series in ("production", "candidate", "delta"):
                tracker.log_scalar(
                    title=f"Production comparison/{split}/{primary_metric}",
                    series=series, value=values[series], iteration=0,
                )
    logger.info("Production comparison saved to %s", output_dir)
    return report
