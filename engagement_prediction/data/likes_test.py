from datetime import datetime, timezone

import polars as pl
import pytest

from engagement_prediction.data import likes


UTC = timezone.utc


def test_prepare_likes_normalizes_utc_filters_window_and_invalid_rows():
    result = likes.prepare_likes(
        pl.DataFrame({
            "did": ["u1", "u2", None, "u4"],
            "subject_uri": ["p1", "p2", "p3", None],
            "record_created_at": [
                "2026-01-01T01:00:00",
                "2026-01-02T01:00:00+01:00",
                "2026-01-01T02:00:00Z",
                "2026-01-01T03:00:00Z",
            ],
        }).lazy(),
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
    ).collect()

    assert result["did"].to_list() == ["u1"]
    assert result["like_created_at"].to_list() == [datetime(2026, 1, 1, 1, tzinfo=UTC)]
    assert result.schema["like_created_at"] == pl.Datetime("us", "UTC")


def test_normalize_likes_rejects_missing_or_unsupported_timestamp():
    with pytest.raises(ValueError, match="missing required columns"):
        likes.normalize_likes(pl.DataFrame({"did": ["u"], "subject_uri": ["p"]}).lazy())

    with pytest.raises(ValueError, match="must be a string or datetime"):
        likes.normalize_likes(pl.DataFrame({
            "did": ["u"],
            "subject_uri": ["p"],
            "record_created_at": [1],
        }).lazy())
