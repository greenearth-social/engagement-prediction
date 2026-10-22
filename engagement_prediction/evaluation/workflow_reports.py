"""Preserve the report formats used by the two comparison entry points."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

from engagement_prediction.evaluation.artifacts import ModelArtifact, Stage7Artifact
from engagement_prediction.evaluation.comparison import ComparisonResult, ComparisonSettings
from engagement_prediction.evaluation.reporting import (
    build_metric_deltas,
    round_output_floats,
    write_metric_deltas_csv,
    write_metrics_csv,
)


@dataclass(frozen=True)
class StandaloneReportContext:
    """Run identity and timing allocated by the standalone command."""

    run_id: str
    started_at: datetime
    started_monotonic: float


@dataclass(frozen=True)
class ProductionReportContext:
    """Provenance of the already resolved production snapshot."""

    metadata: dict


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def write_standalone_report(
    *,
    dataset: Stage7Artifact,
    models: Sequence[ModelArtifact],
    settings: ComparisonSettings,
    result: ComparisonResult,
    output_dir: Path,
    context: StandaloneReportContext,
) -> dict:
    """Write the standalone command's existing rounded metrics and metadata."""
    runtime_seconds = round_output_floats(time.monotonic() - context.started_monotonic)
    dataset_metadata = dataset.to_dict()
    model_spec_documents = [model.to_dict() for model in models]
    output_metrics = round_output_floats(result.metrics_by_model)
    output_mapping_coverage = round_output_floats(result.mapping_coverage_by_model)
    metrics_document = {
        "run_id": context.run_id,
        "started_at_utc": context.started_at.isoformat(),
        "runtime_seconds": runtime_seconds,
        "settings": settings,
        "dataset": dataset_metadata,
        "models": model_spec_documents,
        "history_lengths": result.history_lengths,
        "split_row_counts": result.split_row_counts,
        "skipped_splits": result.skipped_splits,
        "mapping_coverage": output_mapping_coverage,
        "metrics": output_metrics,
    }
    _write_json(output_dir / "metrics.json", metrics_document)
    _write_json(
        output_dir / "model_specs.json",
        {
            "models": model_spec_documents,
            "history_lengths": result.history_lengths,
            "mapping_coverage": output_mapping_coverage,
        },
    )
    write_metrics_csv(
        output_dir / "metrics.csv", models=models, metrics_by_model=output_metrics,
    )
    write_metric_deltas_csv(
        output_dir / "metric_deltas.csv",
        model_a_name=models[0].name,
        model_b_name=models[1].name,
        metrics_by_model=output_metrics,
    )
    stage_info_lines = [
        "tool: compare_model_performance",
        f"run_id: {context.run_id}",
        f"started_at_utc: {context.started_at.isoformat()}",
        f"runtime_seconds: {runtime_seconds:.5f}",
        f"dataset: {dataset.root}",
        f"dataset_bundle: {dataset.bundle_path}",
        f"model_a: {models[0].name}",
        f"model_a_type: {models[0].model_type}",
        f"model_a_artifact_format: {models[0].artifact_format}",
        f"model_a_path: {models[0].root}",
        f"model_a_history_len: {result.history_lengths[models[0].name]}",
        f"model_b: {models[1].name}",
        f"model_b_type: {models[1].model_type}",
        f"model_b_artifact_format: {models[1].artifact_format}",
        f"model_b_path: {models[1].root}",
        f"model_b_history_len: {result.history_lengths[models[1].name]}",
        f"device: {settings.device}",
        f"splits: {', '.join(settings.splits)}",
        f"metrics_top_ks: {', '.join(str(k) for k in settings.metrics_top_ks)}",
        f"batch_size: {settings.batch_size}",
        f"additional_batch_negatives: {settings.additional_batch_negatives}",
        (
            "skipped_splits: "
            + (", ".join(result.skipped_splits) if result.skipped_splits else "none")
        ),
    ]
    (output_dir / "stage_info.txt").write_text("\n".join(stage_info_lines) + "\n")
    return metrics_document


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


def write_production_report(
    *,
    dataset: Stage7Artifact,
    models: Sequence[ModelArtifact],
    settings: ComparisonSettings,
    result: ComparisonResult,
    output_dir: Path,
    context: ProductionReportContext,
    dates: dict,
    history_extraction_limit: int,
) -> dict:
    """Write the automated comparison's full-precision metrics and review."""
    primary_metric = f"ndcg@{settings.metrics_top_ks[0]}"
    primary_comparison = {}
    for split in settings.splits:
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
        "production": context.metadata,
        "models": [model.to_dict() for model in models],
        "dataset": dataset.to_dict(),
        "dates": dates,
        "settings": asdict(settings),
        "history_extraction_limit": history_extraction_limit,
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
    return report
