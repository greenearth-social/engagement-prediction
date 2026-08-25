from datetime import datetime, timezone
from pathlib import Path

import pytest

from engagement_prediction.data import ingex
from engagement_prediction.data.source_manifests import (
    load_source_snapshot,
    validate_aligned_source_snapshots,
)


UTC = timezone.utc


def _write_manifest(
    directory: Path,
    *,
    name: str,
    blob_prefix: str,
    bucket: str = "bucket",
    start: datetime = datetime(2026, 1, 1, tzinfo=UTC),
    end: datetime = datetime(2026, 1, 2, tzinfo=UTC),
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    ingex.write_source_manifest(
        path,
        ingex.build_source_manifest(
            gcs_bucket=bucket,
            blob_prefix=blob_prefix,
            start=start,
            end=end,
            paths=[f"gs://{bucket}/{blob_prefix}_20260101_000000.parquet"],
            timestamps=[start],
        ),
    )
    return path


def test_loads_typed_exact_source_snapshot(tmp_path):
    manifest_path = _write_manifest(
        tmp_path,
        name="like_sources_run.json",
        blob_prefix="bsky_likes",
    )

    snapshot = load_source_snapshot(
        tmp_path,
        manifest_prefix="like_sources_",
        expected_blob_prefix="bsky_likes",
    )

    assert snapshot.path == manifest_path
    assert snapshot.start == datetime(2026, 1, 1, tzinfo=UTC)
    assert snapshot.end == datetime(2026, 1, 2, tzinfo=UTC)
    assert snapshot.gcs_bucket == "bucket"
    assert snapshot.file_uris == (
        "gs://bucket/bsky_likes_20260101_000000.parquet",
    )


def test_rejects_ambiguous_or_wrong_source_manifest(tmp_path):
    _write_manifest(tmp_path, name="like_sources_one.json", blob_prefix="bsky_likes")
    _write_manifest(tmp_path, name="like_sources_two.json", blob_prefix="bsky_likes")
    with pytest.raises(FileNotFoundError, match="found 2"):
        load_source_snapshot(
            tmp_path,
            manifest_prefix="like_sources_",
            expected_blob_prefix="bsky_likes",
        )

    posts_dir = tmp_path / "posts"
    _write_manifest(posts_dir, name="post_sources_run.json", blob_prefix="bsky_posts")
    with pytest.raises(ValueError, match="must use bsky_replies"):
        load_source_snapshot(
            posts_dir,
            manifest_prefix="post_sources_",
            expected_blob_prefix="bsky_replies",
        )


def test_validates_common_source_bucket_and_window(tmp_path):
    post_dir = tmp_path / "posts"
    like_dir = tmp_path / "likes"
    _write_manifest(post_dir, name="post_sources_run.json", blob_prefix="bsky_posts")
    _write_manifest(like_dir, name="like_sources_run.json", blob_prefix="bsky_likes")
    posts = load_source_snapshot(
        post_dir,
        manifest_prefix="post_sources_",
        expected_blob_prefix="bsky_posts",
    )
    likes = load_source_snapshot(
        like_dir,
        manifest_prefix="like_sources_",
        expected_blob_prefix="bsky_likes",
    )
    assert validate_aligned_source_snapshots(
        (posts, likes),
        description="Test snapshots",
    ) == (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC))

    other_dir = tmp_path / "other"
    _write_manifest(
        other_dir,
        name="like_sources_run.json",
        blob_prefix="bsky_likes",
        bucket="other-bucket",
    )
    other = load_source_snapshot(
        other_dir,
        manifest_prefix="like_sources_",
        expected_blob_prefix="bsky_likes",
    )
    with pytest.raises(ValueError, match="do not share one source bucket and window"):
        validate_aligned_source_snapshots(
            (posts, other),
            description="Test snapshots",
        )
