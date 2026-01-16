#!/usr/bin/env python3

"""
Experiment tracking abstraction with a ClearML implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from clearml import Task


class ExperimentTracker(Protocol):
    def log_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        ...

    def log_artifact(self, name: str, path: Path) -> None:
        ...

    def log_params(self, params: Dict[str, Any]) -> None:
        ...

    def log_single_value(self, name: str, value: float) -> None:
        ...

    def close(self) -> None:
        ...


class NoOpExperimentTracker:
    def log_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        return None

    def log_artifact(self, name: str, path: Path) -> None:
        return None

    def log_params(self, params: Dict[str, Any]) -> None:
        return None

    def log_single_value(self, name: str, value: float) -> None:
        return None

    def close(self) -> None:
        return None


class ClearMLExperimentTracker:
    def __init__(
        self,
        project_name: str,
        task_name: str,
        tags: Optional[Iterable[str]] = None,
    ) -> None:
        from clearml import Task

        self._task: Task = Task.init(
            project_name=project_name,
            task_name=task_name,
            tags=list(tags) if tags else None,
            reuse_last_task_id=False,
            auto_connect_frameworks=True, # for auto-logging from PyTorch, matplotlib, etc
        )
        self._logger = self._task.get_logger()

    def log_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        self._logger.report_scalar(
            title=title,
            series=series,
            value=value,
            iteration=iteration,
        )

    def log_artifact(self, name: str, path: Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        self._task.upload_artifact(name=name, artifact_object=str(p))

    def log_params(self, params: Dict[str, Any]) -> None:
        self._task.connect(params)

    def log_single_value(self, name: str, value: float) -> None:
        self._logger.report_single_value(name=name, value=value)

    def close(self) -> None:
        self._task.close()


def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
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
) -> ExperimentTracker:
    if kind == "clearml":
        return ClearMLExperimentTracker(
            project_name=project_name,
            task_name=task_name,
            tags=tags,
        )
    return NoOpExperimentTracker()
