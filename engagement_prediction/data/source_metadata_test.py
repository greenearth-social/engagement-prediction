from datetime import datetime, timezone

import polars as pl
import pytest

from engagement_prediction.data import source_metadata


UTC = timezone.utc


def test_normalizes_and_selects_latest_metadata_with_stable_tie_break():
    raw = pl.DataFrame({
        "at_uri": ["p", "p", "p", "", "outside"],
        "record_created_at": [
            "2026-01-01T01:00:00Z",
            "2026-01-01T02:00:00Z",
            "2026-01-01T02:00:00Z",
            "2026-01-01T03:00:00Z",
            "2026-01-02T00:00:00Z",
        ],
        "did": ["z", "z", "a", "a", "a"],
    })
    normalized = source_metadata.normalize_source_records(
        raw.lazy(),
        posts_start=datetime(2026, 1, 1, tzinfo=UTC),
        posts_end=datetime(2026, 1, 2, tzinfo=UTC),
        is_reply=False,
    ).collect()

    selected, stats = source_metadata.select_latest_metadata_rows(normalized)

    assert selected.to_dicts() == [{
        "subject_uri": "p",
        "post_created_at": datetime(2026, 1, 1, 2, tzinfo=UTC),
        "author_did": "a",
        "is_reply": False,
    }]
    assert stats == {
        "source_row_count": 5,
        "invalid_row_count": 2,
        "duplicate_row_count": 2,
        "duplicate_uri_count": 1,
        "unique_valid_count": 1,
    }


def test_applies_root_precedence():
    roots = pl.DataFrame({
        "subject_uri": ["both", "root"],
        "post_created_at": [datetime(2026, 1, 1, tzinfo=UTC)] * 2,
        "author_did": ["root-author", "root-author"],
        "is_reply": [False, False],
    }, schema=source_metadata.POST_METADATA_SCHEMA)
    replies = pl.DataFrame({
        "subject_uri": ["both", "reply"],
        "post_created_at": [datetime(2026, 1, 1, tzinfo=UTC)] * 2,
        "author_did": ["reply-author", "reply-author"],
        "is_reply": [True, True],
    }, schema=source_metadata.POST_METADATA_SCHEMA)

    resolved, overlap = source_metadata.apply_root_precedence(roots, replies)

    assert overlap == 1
    assert resolved["subject_uri"].to_list() == ["both", "reply", "root"]
    assert not resolved.filter(pl.col("subject_uri") == "both")["is_reply"].item()


def test_partition_validation_rejects_wrong_partition():
    frame = pl.DataFrame({
        "subject_uri": ["p"],
        "post_created_at": [datetime(2026, 1, 1, tzinfo=UTC)],
        "author_did": ["a"],
        "is_reply": [False],
    }, schema=source_metadata.POST_METADATA_SCHEMA)
    actual = frame.with_columns(source_metadata.uri_partition_expr(2))["_post_partition"].item()
    with pytest.raises(ValueError, match="contains rows assigned"):
        source_metadata.validate_metadata_partition(
            frame,
            partition_id=1 - actual,
            partition_count=2,
        )
