from utils.experiment_tracking import ClearMLExperimentTracker


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


def test_log_file_artifact_uploads_path_to_clearml_task(tmp_path):
    artifact_path = tmp_path / "author_idx.parquet"
    artifact_path.write_bytes(b"parquet")
    tracker = ClearMLExperimentTracker.__new__(ClearMLExperimentTracker)
    tracker._task = _FakeTask()

    metadata = tracker.log_file_artifact("author_idx_mapping", artifact_path)
    artifact_id = metadata["artifact_id"]

    # assert artifact_id == "artifact-id"
    assert tracker._task.uploads == [("author_idx_mapping", str(artifact_path), False)]


def test_log_file_artifact_with_metadata_returns_clearml_artifact_uri(tmp_path):
    artifact_path = tmp_path / "author_idx.parquet"
    artifact_path.write_bytes(b"parquet")
    tracker = ClearMLExperimentTracker.__new__(ClearMLExperimentTracker)
    tracker._task = _FakeTask()

    result = tracker.log_file_artifact_with_metadata("author_idx_mapping", artifact_path)

    assert result == {
        "artifact_id": "author_idx_mapping",
        "uri": "gs://bucket/author_idx_mapping.parquet",
    }
    assert tracker._task.uploads == [("author_idx_mapping", str(artifact_path), True)]
