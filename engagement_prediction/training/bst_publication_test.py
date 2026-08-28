"""Tests for canonical BST serving publication helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from engagement_prediction.training import bst_publication
from engagement_prediction.training.bst_publication import publish_ranker_to_tracker


class _Tracker:
    def __init__(
        self,
        *,
        task_id: str = "task-1",
        model_error: Exception | None = None,
        author_uploaded: bool = True,
        manifest_uploaded: bool = True,
        model_uri: str = "gs://models/task/models/ranker.pt",
    ) -> None:
        self.id = task_id
        self.model_error = model_error
        self.author_uploaded = author_uploaded
        self.manifest_uploaded = manifest_uploaded
        self.model_uri = model_uri
        self.model_calls = []
        self.file_calls = []

    def log_artifact(self, name, path):
        self.model_calls.append((name, Path(path)))
        if self.model_error is not None:
            raise self.model_error
        return {
            "model_id": "model-1",
            "uri": self.model_uri,
        }

    def log_file_artifact(self, name, path):
        self.file_calls.append((name, Path(path)))
        if name == "author_idx_mapping":
            return self.author_uploaded
        return self.manifest_uploaded


def test_publish_ranker_creates_exact_manifest_after_complete_upload(tmp_path):
    tracker = _Tracker()
    ranker_path = tmp_path / "ranker.pt"
    author_map_path = tmp_path / "ranker_author_idx.parquet"
    manifest_path = tmp_path / "ranker_serving_manifest.json"
    ranker_path.touch()
    author_map_path.touch()

    result = publish_ranker_to_tracker(
        tracker=tracker,
        logger=logging.getLogger("test"),
        torchscript_path=ranker_path,
        author_map_path=author_map_path,
        manifest_path=manifest_path,
    )

    assert result["status"] == "complete"
    assert tracker.model_calls == [("ranker", ranker_path)]
    assert tracker.file_calls == [
        ("author_idx_mapping", author_map_path),
        ("ranker_serving_manifest", manifest_path),
    ]
    assert json.loads(manifest_path.read_text()) == {
        "ranker_clearml_model_id": "model-1",
        "ranker_uri": "gs://models/task/models/ranker.pt",
        "clearml_task_id": "task-1",
    }


@pytest.mark.parametrize(
    ("tracker", "expected_model_registered", "expected_author_uploaded"),
    [
        (_Tracker(model_error=RuntimeError("offline")), False, True),
        (_Tracker(author_uploaded=False), True, False),
    ],
)
def test_publish_ranker_omits_manifest_when_required_upload_is_missing(
    tmp_path,
    tracker,
    expected_model_registered,
    expected_author_uploaded,
):
    ranker_path = tmp_path / "ranker.pt"
    author_map_path = tmp_path / "ranker_author_idx.parquet"
    manifest_path = tmp_path / "ranker_serving_manifest.json"
    ranker_path.touch()
    author_map_path.touch()

    result = publish_ranker_to_tracker(
        tracker=tracker,
        logger=logging.getLogger("test"),
        torchscript_path=ranker_path,
        author_map_path=author_map_path,
        manifest_path=manifest_path,
    )

    assert result["status"] == "incomplete"
    assert result["model_registered"] is expected_model_registered
    assert result["author_map_uploaded"] is expected_author_uploaded
    assert not manifest_path.exists()


def test_publish_ranker_keeps_local_manifest_when_manifest_upload_fails(tmp_path):
    tracker = _Tracker(manifest_uploaded=False)
    ranker_path = tmp_path / "ranker.pt"
    author_map_path = tmp_path / "ranker_author_idx.parquet"
    manifest_path = tmp_path / "ranker_serving_manifest.json"
    ranker_path.touch()
    author_map_path.touch()

    result = publish_ranker_to_tracker(
        tracker=tracker,
        logger=logging.getLogger("test"),
        torchscript_path=ranker_path,
        author_map_path=author_map_path,
        manifest_path=manifest_path,
    )

    assert result["status"] == "incomplete"
    assert result["manifest_created"] is True
    assert result["manifest_uploaded"] is False
    assert manifest_path.exists()


def test_publish_ranker_rejects_noncanonical_model_uri(tmp_path):
    tracker = _Tracker(model_uri="gs://models/task/ranker.pt")
    ranker_path = tmp_path / "ranker.pt"
    author_map_path = tmp_path / "ranker_author_idx.parquet"
    manifest_path = tmp_path / "ranker_serving_manifest.json"
    ranker_path.touch()
    author_map_path.touch()

    result = publish_ranker_to_tracker(
        tracker=tracker,
        logger=logging.getLogger("test"),
        torchscript_path=ranker_path,
        author_map_path=author_map_path,
        manifest_path=manifest_path,
    )

    assert result["status"] == "incomplete"
    assert result["model_registered"] is False
    assert not manifest_path.exists()


def test_publish_ranker_does_not_expose_manifest_after_local_write_failure(
    tmp_path,
    monkeypatch,
):
    tracker = _Tracker()
    ranker_path = tmp_path / "ranker.pt"
    author_map_path = tmp_path / "ranker_author_idx.parquet"
    manifest_path = tmp_path / "ranker_serving_manifest.json"
    ranker_path.touch()
    author_map_path.touch()

    def fail_manifest_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        bst_publication,
        "write_json_atomically",
        fail_manifest_write,
    )
    result = publish_ranker_to_tracker(
        tracker=tracker,
        logger=logging.getLogger("test"),
        torchscript_path=ranker_path,
        author_map_path=author_map_path,
        manifest_path=manifest_path,
    )

    assert result["status"] == "incomplete"
    assert result["manifest_created"] is False
    assert not manifest_path.exists()


def test_publish_ranker_without_tracker_keeps_only_local_files(tmp_path):
    tracker = _Tracker(task_id="")
    manifest_path = tmp_path / "ranker_serving_manifest.json"

    result = publish_ranker_to_tracker(
        tracker=tracker,
        logger=logging.getLogger("test"),
        torchscript_path=tmp_path / "ranker.pt",
        author_map_path=tmp_path / "ranker_author_idx.parquet",
        manifest_path=manifest_path,
    )

    assert result["status"] == "not_configured"
    assert tracker.model_calls == []
    assert tracker.file_calls == []
    assert not manifest_path.exists()
