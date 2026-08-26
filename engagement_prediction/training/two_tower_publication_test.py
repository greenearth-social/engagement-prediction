from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from engagement_prediction.training.two_tower_publication import (
    publish_two_tower_to_tracker,
)


class _Tracker:
    def __init__(
        self,
        *,
        fail_model=None,
        fail_file=None,
        manifest_uploaded=True,
        task_id="task-1",
    ):
        self.id = task_id
        self.fail_model = fail_model
        self.fail_file = fail_file
        self.manifest_uploaded = manifest_uploaded
        self.models = []
        self.files = []

    def log_artifact(self, name, path):
        self.models.append((name, Path(path)))
        if name == self.fail_model:
            raise RuntimeError("upload failed")
        return {
            "model_id": f"{name}-id",
            "uri": f"gs://models/task/models/{Path(path).name}",
        }

    def log_file_artifact(self, name, path):
        self.files.append((name, Path(path)))
        if name == self.fail_file:
            raise RuntimeError("upload failed")
        if name == "two_tower_serving_manifest":
            return self.manifest_uploaded
        return True


def _files(tmp_path):
    user_path = tmp_path / "engagement_user_tower.pt"
    post_path = tmp_path / "engagement_post_tower.pt"
    author_path = tmp_path / "two_tower_author_idx.parquet"
    for path in (user_path, post_path, author_path):
        path.write_bytes(b"artifact")
    return user_path, post_path, author_path


def test_publication_creates_exact_serving_manifest(tmp_path):
    user_path, post_path, author_path = _files(tmp_path)
    manifest_path = tmp_path / "two_tower_serving_manifest.json"
    tracker = _Tracker()

    result = publish_two_tower_to_tracker(
        tracker=tracker,
        logger=logging.getLogger("two-tower-publication-test"),
        user_tower_path=user_path,
        post_tower_path=post_path,
        author_map_path=author_path,
        manifest_path=manifest_path,
        output_embedding_dim=192,
    )

    assert result["status"] == "complete"
    assert json.loads(manifest_path.read_text()) == {
        "user_tower_clearml_model_id": "engagement_user_tower-id",
        "post_tower_clearml_model_id": "engagement_post_tower-id",
        "user_tower_uri": "gs://models/task/models/engagement_user_tower.pt",
        "post_tower_uri": "gs://models/task/models/engagement_post_tower.pt",
        "output_embedding_dim": 192,
        "clearml_task_id": "task-1",
        "embedding_space_id": "engagement_post_tower-id",
    }
    assert [name for name, _ in tracker.models] == [
        "engagement_user_tower",
        "engagement_post_tower",
    ]
    assert [name for name, _ in tracker.files] == [
        "author_idx_mapping",
        "two_tower_serving_manifest",
    ]


@pytest.mark.parametrize(
    "failed_model",
    ["engagement_user_tower", "engagement_post_tower"],
)
def test_failed_model_upload_does_not_create_manifest(tmp_path, failed_model):
    user_path, post_path, author_path = _files(tmp_path)
    manifest_path = tmp_path / "two_tower_serving_manifest.json"

    result = publish_two_tower_to_tracker(
        tracker=_Tracker(fail_model=failed_model),
        logger=logging.getLogger("two-tower-publication-failure-test"),
        user_tower_path=user_path,
        post_tower_path=post_path,
        author_map_path=author_path,
        manifest_path=manifest_path,
        output_embedding_dim=128,
    )

    assert result["status"] == "incomplete"
    failed_key = (
        "user_tower_registered"
        if failed_model == "engagement_user_tower"
        else "post_tower_registered"
    )
    assert result[failed_key] is False
    assert not manifest_path.exists()


def test_failed_author_map_upload_does_not_create_manifest(tmp_path):
    user_path, post_path, author_path = _files(tmp_path)
    manifest_path = tmp_path / "two_tower_serving_manifest.json"

    result = publish_two_tower_to_tracker(
        tracker=_Tracker(fail_file="author_idx_mapping"),
        logger=logging.getLogger("two-tower-publication-author-failure-test"),
        user_tower_path=user_path,
        post_tower_path=post_path,
        author_map_path=author_path,
        manifest_path=manifest_path,
        output_embedding_dim=128,
    )

    assert result["status"] == "incomplete"
    assert result["author_map_uploaded"] is False
    assert not manifest_path.exists()


def test_disabled_tracker_keeps_only_local_artifacts(tmp_path):
    user_path, post_path, author_path = _files(tmp_path)
    manifest_path = tmp_path / "two_tower_serving_manifest.json"

    result = publish_two_tower_to_tracker(
        tracker=_Tracker(task_id=""),
        logger=logging.getLogger("two-tower-publication-disabled-test"),
        user_tower_path=user_path,
        post_tower_path=post_path,
        author_map_path=author_path,
        manifest_path=manifest_path,
        output_embedding_dim=128,
    )

    assert result["status"] == "not_configured"
    assert not manifest_path.exists()


def test_manifest_upload_failure_keeps_valid_local_manifest(tmp_path):
    user_path, post_path, author_path = _files(tmp_path)
    manifest_path = tmp_path / "two_tower_serving_manifest.json"

    result = publish_two_tower_to_tracker(
        tracker=_Tracker(manifest_uploaded=False),
        logger=logging.getLogger("two-tower-publication-manifest-test"),
        user_tower_path=user_path,
        post_tower_path=post_path,
        author_map_path=author_path,
        manifest_path=manifest_path,
        output_embedding_dim=128,
    )

    assert result["status"] == "incomplete"
    assert result["manifest_created"] is True
    assert result["manifest_uploaded"] is False
    assert manifest_path.is_file()
