"""Tabular reporting helpers for aggregate model comparisons."""

from __future__ import annotations

import csv
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

from engagement_prediction.evaluation.artifacts import ModelArtifact


OUTPUT_DECIMAL_PLACES = 5


def round_output_floats(value: Any) -> Any:
    """Round result floats recursively without changing counts or booleans."""

    if isinstance(value, Mapping):
        return {
            key: round_output_floats(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [round_output_floats(item) for item in value]
    if isinstance(value, tuple):
        return tuple(round_output_floats(item) for item in value)
    if isinstance(value, Real) and not isinstance(value, (Integral, bool)):
        number = float(value)
        if not math.isfinite(number):
            return number
        rounded = round(number, OUTPUT_DECIMAL_PLACES)
        return 0.0 if rounded == 0.0 else rounded
    return value


def is_performance_metric(metric_name: str) -> bool:
    """Return whether a metric supports a meaningful model-B-minus-A delta."""

    return (
        metric_name.startswith("dcg@")
        or metric_name.startswith("ndcg@")
        or metric_name == "mean_average_precision"
        or metric_name.startswith("zero_history_dcg@")
        or metric_name.startswith("zero_history_ndcg@")
        or metric_name == "zero_history_mean_average_precision"
        or metric_name in {"auc_roc", "classification_average_precision"}
    )


def _optional_finite_float(value: Any) -> float | None:
    """Normalize optional numeric output while rejecting NaN and infinity."""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_metric_deltas(
    *,
    model_a_name: str,
    model_b_name: str,
    metrics_by_model: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Build long-form model-B-minus-model-A performance rows."""

    try:
        model_a_metrics = metrics_by_model[model_a_name]
        model_b_metrics = metrics_by_model[model_b_name]
    except KeyError as exc:
        raise ValueError(f"Missing metrics for comparison model {exc.args[0]!r}") from exc
    split_names = sorted(set(model_a_metrics) | set(model_b_metrics))
    rows: list[dict[str, Any]] = []
    for split_name in split_names:
        metrics_a = model_a_metrics.get(split_name, {})
        metrics_b = model_b_metrics.get(split_name, {})
        metric_names = sorted(
            metric_name
            for metric_name in set(metrics_a) | set(metrics_b)
            if is_performance_metric(metric_name)
        )
        for metric_name in metric_names:
            value_a = _optional_finite_float(metrics_a.get(metric_name))
            value_b = _optional_finite_float(metrics_b.get(metric_name))
            rows.append({
                "model_a_name": model_a_name,
                "model_b_name": model_b_name,
                "split": split_name,
                "metric": metric_name,
                "model_a_value": value_a,
                "model_b_value": value_b,
                "delta_model_b_minus_model_a": (
                    value_b - value_a
                    if value_a is not None and value_b is not None
                    else None
                ),
            })
    return rows


def _csv_value(value: Any) -> Any:
    """Format finite floats consistently while preserving nonnumeric values."""

    if value is None:
        return ""
    rounded = round_output_floats(value)
    if isinstance(rounded, float) and math.isfinite(rounded):
        return f"{rounded:.{OUTPUT_DECIMAL_PLACES}f}"
    return rounded


def write_metrics_csv(
    path: Path,
    *,
    models: Sequence[ModelArtifact],
    metrics_by_model: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    """Write every aggregate value in stable long-form order."""

    models_by_name = {model.name: model for model in models}
    if set(models_by_name) != set(metrics_by_model):
        raise ValueError("Resolved models and metrics_by_model names must match")
    with Path(path).open("w", newline="") as file_obj:
        fieldnames = [
            "model_name",
            "model_type",
            "model_path",
            "split",
            "metric",
            "value",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for model in models:
            for split_name in sorted(metrics_by_model[model.name]):
                split_metrics = metrics_by_model[model.name][split_name]
                for metric_name in sorted(split_metrics):
                    writer.writerow({
                        "model_name": model.name,
                        "model_type": model.model_type,
                        "model_path": str(model.root),
                        "split": split_name,
                        "metric": metric_name,
                        "value": _csv_value(split_metrics[metric_name]),
                    })


def write_metric_deltas_csv(
    path: Path,
    *,
    model_a_name: str,
    model_b_name: str,
    metrics_by_model: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    """Write only aggregate performance deltas, always as model B minus A."""

    rows = build_metric_deltas(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        metrics_by_model=round_output_floats(metrics_by_model),
    )
    fieldnames = [
        "model_a_name",
        "model_b_name",
        "split",
        "metric",
        "model_a_value",
        "model_b_value",
        "delta_model_b_minus_model_a",
    ]
    with Path(path).open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row[key]) for key in fieldnames})
