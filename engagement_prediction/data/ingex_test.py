from datetime import datetime, timezone
from types import SimpleNamespace

from engagement_prediction.data import ingex


class _FakeClient:
    def __init__(self, names):
        self._blobs = [SimpleNamespace(name=name) for name in names]

    def list_blobs(self, bucket):
        assert bucket == "test-bucket"
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
