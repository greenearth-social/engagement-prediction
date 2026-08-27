from datetime import datetime, timezone

import polars as pl
import pytest

from engagement_prediction.data import like_counts


UTC = timezone.utc


def test_prior_like_counts_are_strict_and_count_duplicate_rows():
    first_hour = datetime(2026, 1, 1, 1, tzinfo=UTC)
    second_hour = datetime(2026, 1, 1, 2, tzinfo=UTC)
    pairs = pl.DataFrame({
        "subject_uri": ["a", "a", "a", "b"],
        "query_hour": [first_hour, second_hour, second_hour, second_hour],
    }, schema=like_counts.POST_HOUR_SCHEMA)
    events = pl.DataFrame({
        "subject_uri": ["a", "a", "a", "a", "unrelated"],
        "like_created_at": [
            datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
            first_hour,
            first_hour,
            second_hour,
            datetime(2026, 1, 1, 0, tzinfo=UTC),
        ],
    }, schema=like_counts.LIKE_EVENT_SCHEMA)

    result = like_counts.calculate_prior_like_counts(pairs, events)

    assert result.to_dicts() == [
        {"subject_uri": "a", "query_hour": first_hour, "prior_like_count": 1},
        {"subject_uri": "a", "query_hour": second_hour, "prior_like_count": 3},
        {"subject_uri": "b", "query_hour": second_hour, "prior_like_count": 0},
    ]


def test_prior_like_counts_handle_empty_inputs():
    empty_pairs = pl.DataFrame(schema=like_counts.POST_HOUR_SCHEMA)
    empty_events = pl.DataFrame(schema=like_counts.LIKE_EVENT_SCHEMA)

    empty_result = like_counts.calculate_prior_like_counts(
        empty_pairs,
        empty_events,
    )
    assert empty_result.schema == pl.Schema(like_counts.POST_HOUR_COUNT_SCHEMA)
    assert empty_result.is_empty()

    hour = datetime(2026, 1, 1, tzinfo=UTC)
    zero_result = like_counts.calculate_prior_like_counts(
        pl.DataFrame({
            "subject_uri": ["post"],
            "query_hour": [hour],
        }, schema=like_counts.POST_HOUR_SCHEMA),
        empty_events,
    )
    assert zero_result.get_column("prior_like_count").to_list() == [0]


def test_prior_like_counts_require_hour_aligned_utc_query_times():
    events = pl.DataFrame(schema=like_counts.LIKE_EVENT_SCHEMA)
    with pytest.raises(ValueError, match="UTC"):
        like_counts.calculate_prior_like_counts(
            pl.DataFrame({
                "subject_uri": ["post"],
                "query_hour": [datetime(2026, 1, 1)],
            }),
            events,
        )
    with pytest.raises(ValueError, match="aligned"):
        like_counts.calculate_prior_like_counts(
            pl.DataFrame({
                "subject_uri": ["post"],
                "query_hour": [datetime(2026, 1, 1, 0, 1, tzinfo=UTC)],
            }, schema=like_counts.POST_HOUR_SCHEMA),
            events,
        )
