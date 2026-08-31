"""Experiment-tracker parameter normalization and construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from engagement_prediction.experiment_tracking.base import (
    ExperimentTracker,
    NoOpExperimentTracker,
)


def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert CLI parameters into tracker-safe primitive values."""

    def _normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _normalize(v) for k, v in value.items() if v is not None}
        if isinstance(value, (list, tuple)):
            return [_normalize(v) for v in value]
        return value

    return {k: _normalize(v) for k, v in params.items() if v is not None}


def build_experiment_tracker(
    kind: str,
    *,
    project_name: str,
    task_name: str,
    tags: Optional[Iterable[str]] = None,
    model_output_uri: Optional[str] = None,
) -> ExperimentTracker:
    """Construct the requested tracker, or a side-effect-free no-op tracker."""

    if kind == "clearml":
        from engagement_prediction.experiment_tracking.clearml import (
            ClearMLExperimentTracker,
        )

        return ClearMLExperimentTracker(
            project_name=project_name,
            task_name=task_name,
            tags=tags,
            model_output_uri=model_output_uri,
        )
    return NoOpExperimentTracker()
