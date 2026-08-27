"""Experiment-tracker interface and no-op implementation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union


class ExperimentTracker(Protocol):
    id: str

    def log_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        ...

    def log_artifact(self, name: str, path: Path) -> dict[str, str]:
        ...

    def log_file_artifact(self, name: str, path: Path) -> bool:
        ...

    def log_params(self, params: Dict[str, Any], name: Optional[str] = None) -> None:
        ...

    def connect_args(self, args: argparse.Namespace, name: Optional[str] = None) -> argparse.Namespace:
        ...

    def log_single_value(self, name: str, value: float) -> None:
        ...

    def log_histogram(
        self,
        title: str,
        series: str,
        values: Union[
            List[Union[int, float]],
            List[List[Union[int, float]]],
        ],
        iteration: int = 0,
        xlabels: Optional[List[str]] = None,
        xaxis: Optional[str] = None,
        yaxis: Optional[str] = None,
        labels: Optional[List[str]] = None,
        mode: Optional[str] = None,
    ) -> None:
        ...

    def log_plot(
        self,
        title: str,
        series: str,
        figure: Any,
        iteration: int = 0,
    ) -> None:
        ...

    def close(self) -> None:
        ...


class NoOpExperimentTracker:
    id: str

    def log_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        return None

    def log_artifact(self, name: str, path: Path) -> dict[str, str]:
        return {}

    def log_file_artifact(self, name: str, path: Path) -> bool:
        return False

    def log_params(self, params: Dict[str, Any], name: Optional[str] = None) -> None:
        return None

    def connect_args(self, args: argparse.Namespace, name: Optional[str] = None) -> argparse.Namespace:
        return args

    def log_single_value(self, name: str, value: float) -> None:
        return None

    def log_histogram(
        self,
        title: str,
        series: str,
        values: Union[
            List[Union[int, float]],
            List[List[Union[int, float]]],
        ],
        iteration: int = 0,
        xlabels: Optional[List[str]] = None,
        xaxis: Optional[str] = None,
        yaxis: Optional[str] = None,
        labels: Optional[List[str]] = None,
        mode: Optional[str] = None,
    ) -> None:
        return None

    def log_plot(
        self,
        title: str,
        series: str,
        figure: Any,
        iteration: int = 0,
    ) -> None:
        return None

    def close(self) -> None:
        return None
