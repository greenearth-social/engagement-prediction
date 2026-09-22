from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from engagement_prediction.data import ingex


class _FakeClient:
    def __init__(self, names):
        self._blobs = [SimpleNamespace(name=name) for name in names]

    def list_blobs(self, bucket, *, prefix):
        assert bucket == "test-bucket"
        assert prefix == "bsky_likes_"
        return self._blobs


def test_parse_ingex_blob_timestamp_requires_exact_prefix_and_name():
    assert ingex.parse_ingex_blob_timestamp(
        "bsky_likes_20260807_123456.parquet",
        "bsky_likes",
    ) == datetime(2026, 8, 7, 12, 34, 56, tzinfo=timezone.utc)
    assert ingex.parse_ingex_blob_timestamp(
        "other_20260807_123456.parquet",
        "bsky_likes",
    ) is None
    assert ingex.parse_ingex_blob_timestamp(
        "folder/bsky_likes_20260807_123456.parquet",
        "bsky_likes",
    ) is None


def test_list_ingex_parquet_files_filters_and_sorts():
    client = _FakeClient(
        [
            "bsky_likes_20260807_030000.parquet",
            "not_likes_20260807_020000.parquet",
            "bsky_likes_20260807_010000.parquet",
            "bsky_likes_20260807_020000.parquet",
            "bsky_likes_invalid.parquet",
        ]
    )

    uris, timestamps = ingex._list_ingex_parquet_files(
        client,
        gcs_bucket="test-bucket",
        blob_prefix="bsky_likes",
        start=datetime(2026, 8, 7, 2, tzinfo=timezone.utc),
        end=datetime(2026, 8, 7, 3, tzinfo=timezone.utc),
    )

    assert uris == ["gs://test-bucket/bsky_likes_20260807_020000.parquet"]
    assert timestamps == [datetime(2026, 8, 7, 2, tzinfo=timezone.utc)]


def test_source_manifest_round_trip_records_exact_files(tmp_path):
    start = datetime(2026, 8, 7, 1, tzinfo=timezone.utc)
    end = datetime(2026, 8, 7, 3, tzinfo=timezone.utc)
    manifest = ingex.build_source_manifest(
        gcs_bucket="bucket",
        blob_prefix="bsky_likes",
        start=start,
        end=end,
        paths=["gs://bucket/one.parquet", "gs://bucket/two.parquet"],
        timestamps=[start, datetime(2026, 8, 7, 2, tzinfo=timezone.utc)],
    )
    path = Path(tmp_path) / "sources.json"
    ingex.write_source_manifest(path, manifest)

    assert ingex.load_source_manifest(path) == manifest


@pytest.mark.parametrize("new_first", [False, True])
def test_scan_post_parquet_files_accepts_mixed_media_schemas(tmp_path, new_first):
    legacy_path = tmp_path / "legacy.parquet"
    current_path = tmp_path / "current.parquet"
    pl.DataFrame({"at_uri": ["legacy"]}).write_parquet(legacy_path)
    pl.DataFrame({
        "at_uri": ["images", "video", "both", "neither"],
        "contains_images": [True, False, True, False],
        "contains_video": [False, True, True, False],
    }).write_parquet(current_path)
    paths = [str(legacy_path), str(current_path)]
    if new_first:
        paths.reverse()

    result = ingex.scan_post_parquet_files(
        paths, include_file_paths="source_file"
    ).collect(engine="streaming").sort("at_uri")

    assert result.schema == pl.Schema({
        "at_uri": pl.String,
        "contains_images": pl.Boolean,
        "contains_video": pl.Boolean,
        "source_file": pl.String,
    })
    assert result.select("at_uri", "contains_images", "contains_video").to_dicts() == [
        {"at_uri": "both", "contains_images": True, "contains_video": True},
        {"at_uri": "images", "contains_images": True, "contains_video": False},
        {"at_uri": "legacy", "contains_images": None, "contains_video": None},
        {"at_uri": "neither", "contains_images": False, "contains_video": False},
        {"at_uri": "video", "contains_images": False, "contains_video": True},
    ]
    assert result.filter(pl.col("at_uri") == "legacy")["source_file"].item() == str(
        legacy_path
    )


def test_scan_post_parquet_files_expects_boolean_media_flags(tmp_path):
    path = tmp_path / "invalid.parquet"
    pl.DataFrame({
        "at_uri": ["post"],
        "contains_images": ["true"],
        "contains_video": [False],
    }).write_parquet(path)

    with pytest.raises(pl.exceptions.SchemaError, match="contains_images"):
        ingex.scan_post_parquet_files([str(path)]).collect(engine="streaming")


def test_scan_post_parquet_files_rejects_unexpected_extra_columns(tmp_path):
    legacy_path = tmp_path / "legacy.parquet"
    current_path = tmp_path / "unexpected.parquet"
    pl.DataFrame({"at_uri": ["legacy"]}).write_parquet(legacy_path)
    pl.DataFrame({"at_uri": ["new"], "unexpected": [True]}).write_parquet(current_path)

    with pytest.raises(pl.exceptions.SchemaError, match="unexpected"):
        ingex.scan_post_parquet_files(
            [str(legacy_path), str(current_path)]
        ).collect(engine="streaming")


def test_scan_post_parquet_files_rejects_empty_paths():
    with pytest.raises(ValueError, match="empty collection"):
        ingex.scan_post_parquet_files([])
