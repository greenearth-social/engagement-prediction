"""Shared evaluation and result serialization for canonical Stage 8 trainers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

from torch import nn

from engagement_prediction.training.listwise import EpochRunner
from engagement_prediction.training.model_artifacts import write_json_atomically


def evaluate_listwise_splits(
    *,
    model: nn.Module,
    epoch_runner: EpochRunner,
    device: str,
    loaders: Mapping[str, Any],
    disable_progress: bool,
    gradient_clip_max_norm: float,
    metrics_top_ks: list[int],
    max_batches_by_split: Mapping[str, Optional[int]],
) -> dict[str, dict[str, Any]]:
    """Evaluate a reloaded best model over deterministic split loaders."""

    unexpected_limits = set(max_batches_by_split).difference(loaders)
    if unexpected_limits:
        raise ValueError(
            "max_batches_by_split contains unknown splits: "
            f"{sorted(unexpected_limits)}"
        )
    results: dict[str, dict[str, Any]] = {}
    for split_name, loader in loaders.items():
        _, metrics, _ = epoch_runner(
            train=False,
            split_name=f"Final {split_name}",
            model=model,
            device=device,
            dataloader=loader,
            optimizer=None,
            disable_progress=disable_progress,
            gradient_clip_max_norm=gradient_clip_max_norm,
            metrics_top_ks=metrics_top_ks,
            calc_baseline_metrics=False,
            max_batches=max_batches_by_split.get(split_name),
        )
        results[split_name] = metrics
    return results


def build_training_result_payload(
    *,
    training_results: Mapping[str, Any],
    final_metrics: Mapping[str, Mapping[str, Any]],
    split_query_counts: Mapping[str, int],
    torchscript_export: Mapping[str, Any],
    author_map: Mapping[str, Any],
    clearml_publication: Mapping[str, Any],
    local_pipeline_runtime_seconds: float,
    runtime_seconds: float,
    extra_fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the stable result payload shared by canonical Stage 8 models."""

    payload = {
        "primary_metric_name": training_results["primary_metric_name"],
        "best_val_metric": training_results["best_val_metric"],
        "best_val_loss": training_results["best_val_loss"],
        "best_epoch": training_results["best_epoch"],
        "epochs_completed": training_results["epochs_completed"],
        "stopped_early": training_results["stopped_early"],
        "patience_counter": training_results["patience_counter"],
        "baseline_metrics": training_results["baseline_metrics"],
        "final_metrics": dict(final_metrics),
        "training_history": training_results["history"],
        "split_query_counts": {
            split: int(count) for split, count in split_query_counts.items()
        },
        "torchscript_export": dict(torchscript_export),
        "author_map": dict(author_map),
        "clearml_publication": dict(clearml_publication),
        "local_pipeline_runtime_seconds": float(local_pipeline_runtime_seconds),
        "runtime_seconds": float(runtime_seconds),
    }
    collisions = set(payload).intersection(extra_fields)
    if collisions:
        raise ValueError(
            "extra_fields cannot replace shared training-result keys: "
            f"{sorted(collisions)}"
        )
    return {**dict(extra_fields), **payload}


def build_training_summary(
    *,
    training_config: Mapping[str, Any],
    stage7_dir: Path,
    bundle_path: Path,
    model_config: Mapping[str, Any],
    result_payload: Mapping[str, Any],
    outputs: Mapping[str, Any],
    runtime_seconds: float,
    extra_sections: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the common Stage 8 summary while retaining model-specific sections."""

    summary = {
        "parameters": dict(training_config),
        "input": {
            "dataset_hydration_dir": str(stage7_dir),
            "hydrated_training_data_path": str(bundle_path),
        },
        "model": dict(model_config),
        "results": {
            key: value
            for key, value in result_payload.items()
            if key
            not in {
                "training_history",
                "final_metrics",
                "torchscript_export",
                "author_map",
                "clearml_publication",
            }
        },
        "final_metrics": result_payload["final_metrics"],
        "outputs": dict(outputs),
        "torchscript_export": result_payload["torchscript_export"],
        "author_map": result_payload["author_map"],
        "clearml_publication": result_payload["clearml_publication"],
        "runtime_seconds": float(runtime_seconds),
    }
    collisions = set(summary).intersection(extra_sections)
    if collisions:
        raise ValueError(
            "extra_sections cannot replace shared summary keys: "
            f"{sorted(collisions)}"
        )
    return {**summary, **dict(extra_sections)}


def write_training_result_files(
    *,
    training_results_path: Path,
    result_payload: Mapping[str, Any],
    summary_path: Path,
    summary: Mapping[str, Any],
) -> None:
    """Atomically publish the two common Stage 8 JSON result files."""

    write_json_atomically(training_results_path, result_payload)
    write_json_atomically(summary_path, summary)


def write_stage_info(
    *,
    stage_info_path: Path,
    lines: list[str],
    final_metrics: Mapping[str, Mapping[str, Any]],
    primary_metric_key: str,
) -> None:
    """Publish stage information with common final split metrics appended."""

    complete_lines = list(lines)
    for split, metrics in final_metrics.items():
        complete_lines.extend(
            [
                f"{split}_loss: {metrics['loss']:.6f}",
                f"{split}_{primary_metric_key}: "
                f"{metrics[primary_metric_key]:.6f}",
            ]
        )
    stage_info_path.write_text("\n".join(complete_lines) + "\n")


def upload_reproducibility_artifacts(
    *,
    tracker: Optional[Any],
    logger: logging.Logger,
    artifact_paths: Mapping[str, Path],
) -> None:
    """Attach completed local Stage 8 artifacts to a configured tracker."""

    if tracker is None or not str(getattr(tracker, "id", "") or ""):
        return
    for name, path in artifact_paths.items():
        if not tracker.log_file_artifact(name, path):
            logger.warning("Experiment tracker did not upload artifact '%s'", name)
