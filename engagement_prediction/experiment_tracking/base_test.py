import argparse
from pathlib import Path

from engagement_prediction.experiment_tracking.base import NoOpExperimentTracker


def test_noop_tracker_preserves_arguments_and_reports_no_artifacts(tmp_path):
    tracker = NoOpExperimentTracker()
    args = argparse.Namespace(value=3)

    assert tracker.id == ""
    assert tracker.connect_args(args) is args
    assert tracker.log_artifact("model", tmp_path / "model.pt") == {}
    assert tracker.log_file_artifact("data", Path(tmp_path / "data.json")) is False
    assert tracker.log_scalar("metric", "series", 0.5, 1) is None
    assert tracker.close() is None
