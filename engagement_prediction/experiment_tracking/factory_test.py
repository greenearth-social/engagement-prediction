from pathlib import Path

from engagement_prediction.experiment_tracking import (
    NoOpExperimentTracker,
    build_experiment_tracker,
    normalize_params,
)
from engagement_prediction.experiment_tracking import clearml as clearml_module


def test_normalize_params_converts_paths_and_removes_none_values(tmp_path):
    assert normalize_params({
        "path": tmp_path / "artifact",
        "nested": {"keep": 1, "drop": None},
        "items": [Path("relative"), None],
        "drop": None,
    }) == {
        "path": str(tmp_path / "artifact"),
        "nested": {"keep": 1},
        "items": ["relative", None],
    }


def test_build_experiment_tracker_returns_noop_for_disabled_tracking():
    tracker = build_experiment_tracker(
        "none",
        project_name="project",
        task_name="task",
    )

    assert isinstance(tracker, NoOpExperimentTracker)


def test_build_experiment_tracker_constructs_clearml_backend(monkeypatch):
    observed = {}

    def fake_tracker(**kwargs):
        observed.update(kwargs)
        return "tracker"

    monkeypatch.setattr(clearml_module, "ClearMLExperimentTracker", fake_tracker)

    tracker = build_experiment_tracker(
        "clearml",
        project_name="project",
        task_name="task",
        tags=["tag"],
        model_output_uri="gs://models",
    )

    assert tracker == "tracker"
    assert observed == {
        "project_name": "project",
        "task_name": "task",
        "tags": ["tag"],
        "model_output_uri": "gs://models",
    }
