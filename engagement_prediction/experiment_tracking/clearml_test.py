"""Tests for the ClearML experiment tracker."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from engagement_prediction.experiment_tracking import ClearMLExperimentTracker


class _FakeArtifact:
    def __init__(self, key, uri) -> None:
        self.key = key
        self.uri = uri


class _FakeTask:
    def __init__(self) -> None:
        self.params = None
        self.uploads = []
        self._artifacts = {}

    def set_parameters_as_dict(self, params):
        self.params = params

    def upload_artifact(self, name, artifact_object, wait_on_upload=False):
        self.uploads.append((name, artifact_object, wait_on_upload))
        self._artifacts[name] = _FakeArtifact(name, f"gs://bucket/{name}.parquet")
        return "artifact-id" if not wait_on_upload else True

    @property
    def artifacts(self):
        return self._artifacts


class _FakeOutputModel:
    instances = []

    def __init__(self, *, task, name, framework, tags) -> None:
        self.task = task
        self.name = name
        self.framework = framework
        self.tags = tags
        self.id = "model-123"
        self.updated_weights = []
        self.metadata = {}
        self.__class__.instances.append(self)

    def update_weights(self, path, auto_delete_file=False):
        self.updated_weights.append((path, auto_delete_file))
        return "gs://models/ranker.pt"

    def set_metadata(self, key, value):
        self.metadata[key] = value


class _FailingOutputModel(_FakeOutputModel):
    def update_weights(self, path, auto_delete_file=False):
        raise RuntimeError("upload failed")


class _FakeLogger:
    def __init__(self) -> None:
        self.histograms = []

    def report_histogram(self, **kwargs) -> None:
        self.histograms.append(kwargs)


def test_log_params_updates_clearml_parameters_with_section_prefix():
    tracker = ClearMLExperimentTracker.__new__(ClearMLExperimentTracker)
    tracker._task = _FakeTask()

    tracker.log_params(
        params={
            "run_dir": "/tmp/run",
            "run_name": "20260320_123456_all",
        },
        name="Directories",
    )

    assert tracker._task.params == {
        "Directories/run_dir": "/tmp/run",
        "Directories/run_name": "20260320_123456_all",
    }


def test_log_histogram_reports_grouped_categorical_values():
    tracker = ClearMLExperimentTracker.__new__(ClearMLExperimentTracker)
    tracker._logger = _FakeLogger()

    tracker.log_histogram(
        title="Random baseline NDCG@30",
        series="Baseline",
        values=[[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]],
        iteration=0,
        xlabels=["train", "val", "val_unseen_users"],
        labels=["All observations", "Zero history"],
        mode="group",
    )

    assert len(tracker._logger.histograms) == 1
    reported = tracker._logger.histograms[0]
    np.testing.assert_array_equal(
        reported["values"],
        np.asarray([[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]]),
    )
    assert reported["xlabels"] == ["train", "val", "val_unseen_users"]
    assert reported["labels"] == ["All observations", "Zero history"]
    assert reported["mode"] == "group"


def test_log_file_artifact_uploads_path_to_clearml_task(tmp_path):
    artifact_path = tmp_path / "author_idx.parquet"
    artifact_path.write_bytes(b"parquet")
    tracker = ClearMLExperimentTracker.__new__(ClearMLExperimentTracker)
    tracker._task = _FakeTask()

    result = tracker.log_file_artifact("author_idx_mapping", artifact_path)

    assert result is True
    assert tracker._task.uploads == [("author_idx_mapping", str(artifact_path), True)]


def test_log_artifact_registers_clearml_output_model(monkeypatch, tmp_path):
    model_path = tmp_path / "ranker.pt"
    model_path.write_bytes(b"torchscript")
    task = _FakeTask()
    task.data = SimpleNamespace(
        script=SimpleNamespace(
            repository="git@example.com:greenearth/engagement-prediction.git",
            branch="davidfreifeld/get-data-revamp",
            version_num="abc123",
        )
    )
    tracker = ClearMLExperimentTracker.__new__(ClearMLExperimentTracker)
    tracker._task = task
    _FakeOutputModel.instances = []
    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(OutputModel=_FakeOutputModel))

    result = tracker.log_artifact("ranker", model_path)

    assert result == {
        "model_id": "model-123",
        "uri": "gs://models/ranker.pt",
    }
    assert len(_FakeOutputModel.instances) == 1
    output_model = _FakeOutputModel.instances[0]
    assert output_model.task is task
    assert output_model.name == "ranker"
    assert output_model.framework == "pytorch"
    assert output_model.tags == ["candidate"]
    assert output_model.updated_weights == [(str(model_path), False)]
    assert output_model.metadata == {
        "git_repo": "git@example.com:greenearth/engagement-prediction.git",
        "git_branch": "davidfreifeld/get-data-revamp",
        "git_sha": "abc123",
    }


def test_log_artifact_returns_empty_metadata_for_missing_file(monkeypatch, tmp_path):
    tracker = ClearMLExperimentTracker.__new__(ClearMLExperimentTracker)
    tracker._task = _FakeTask()
    _FakeOutputModel.instances = []
    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(OutputModel=_FakeOutputModel))

    result = tracker.log_artifact("ranker", tmp_path / "missing.pt")

    assert result == {"model_id": "", "uri": ""}
    assert _FakeOutputModel.instances == []


def test_log_artifact_propagates_output_model_upload_failure(monkeypatch, tmp_path):
    model_path = tmp_path / "ranker.pt"
    model_path.write_bytes(b"torchscript")
    tracker = ClearMLExperimentTracker.__new__(ClearMLExperimentTracker)
    tracker._task = _FakeTask()
    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(OutputModel=_FailingOutputModel))

    with pytest.raises(RuntimeError, match="upload failed"):
        tracker.log_artifact("ranker", model_path)
