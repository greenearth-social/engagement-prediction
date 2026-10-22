"""Training-pipeline adapter for the shared comparison workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from engagement_prediction.evaluation.artifacts import ModelArtifact, resolve_model_artifact, resolve_stage7_artifact
from engagement_prediction.evaluation.comparison import ComparisonSettings
from engagement_prediction.evaluation.workflow import (
    HOLDOUT_SPLITS,
    ProductionReportContext,
    run_comparison_workflow,
    validate_history_capacity,
    validate_holdout_dataset,
)
from engagement_prediction.experiment_tracking import ExperimentTracker
from engagement_prediction.training.runtime import get_device


def publish_production_comparison(report: dict, tracker: ExperimentTracker) -> None:
    """Publish completed local reports without coupling evaluation to ClearML."""
    if not tracker.id:
        return
    for filename, saved_path in report["report_paths"].items():
        path = Path(saved_path)
        artifact_name = "production_comparison_" + filename.replace(".", "_")
        if not tracker.log_file_artifact(artifact_name, path):
            raise RuntimeError(f"Failed to upload comparison report {filename}; local report retained at {path}")
    for split, values in report["primary_comparison"].items():
        for series in ("production", "candidate", "delta"):
            tracker.log_scalar(
                title=f"Production comparison/{split}/{report['primary_metric']}",
                series=series, value=values[series], iteration=0,
            )


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
    """Resolve training inputs, compare locally, then publish the saved results."""
    candidate = resolve_model_artifact("candidate", candidate_dir)
    dataset = resolve_stage7_artifact(Path(candidate.training_config["stage7_dir"]))
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
    report = run_comparison_workflow(
        dataset=dataset, models=(production_model, candidate), settings=settings,
        output_dir=output_dir,
        report_context=ProductionReportContext(metadata=production_metadata), logger=logger,
    )
    publish_production_comparison(report, tracker)
    logger.info("Production comparison saved to %s", output_dir)
    return report
