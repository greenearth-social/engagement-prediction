from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from engagement_prediction.data import candidate_popularity


UTC = timezone.utc


def _candidates(rows):
    return pl.DataFrame(rows, schema=candidate_popularity.CANDIDATE_SCHEMA)


def _likes(rows):
    return pl.DataFrame(rows, schema=candidate_popularity.NORMALIZED_LIKE_SCHEMA)


def _hours(*hours):
    return pl.DataFrame(
        {"query_hour": list(hours)},
        schema={"query_hour": candidate_popularity.UTC_DATETIME},
    )


def test_popularity_is_strictly_prior_and_counts_duplicate_rows():
    candidates = _candidates({
        "subject_uri": ["liked", "zero"],
        "post_created_at": [
            datetime(2026, 1, 1, 10, 30, tzinfo=UTC),
            datetime(2026, 1, 1, 10, 45, tzinfo=UTC),
        ],
    })
    likes = _likes({
        "subject_uri": ["liked", "liked", "liked", "liked", "other"],
        "like_created_at": [
            datetime(2026, 1, 1, 10, 5, tzinfo=UTC),
            datetime(2026, 1, 1, 10, 5, tzinfo=UTC),
            datetime(2026, 1, 1, 11, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
            datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
        ],
    })

    result = candidate_popularity.build_candidate_hour_popularity(
        candidates,
        likes,
        _hours(
            datetime(2026, 1, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 1, 11, tzinfo=UTC),
            datetime(2026, 1, 1, 12, tzinfo=UTC),
        ),
        max_candidate_age_hours=3,
    )

    assert result.filter(pl.col("subject_uri") == "liked").select(
        "query_hour", "prior_like_count"
    ).to_dicts() == [
        {
            "query_hour": datetime(2026, 1, 1, 10, tzinfo=UTC),
            "prior_like_count": 0,
        },
        {
            "query_hour": datetime(2026, 1, 1, 11, tzinfo=UTC),
            "prior_like_count": 2,
        },
        {
            "query_hour": datetime(2026, 1, 1, 12, tzinfo=UTC),
            "prior_like_count": 3,
        },
    ]
    assert result.filter(pl.col("subject_uri") == "zero")[
        "prior_like_count"
    ].to_list() == [0, 0, 0]


def test_popularity_filters_non_candidate_likes_before_aggregation(monkeypatch):
    created_at = datetime(2026, 1, 1, 10, 30, tzinfo=UTC)
    likes = _likes({
        "subject_uri": ["candidate", "candidate", "unrelated", "unrelated"],
        "like_created_at": [
            datetime(2026, 1, 1, 10, 40, tzinfo=UTC),
            datetime(2026, 1, 1, 10, 40, tzinfo=UTC),
            datetime(2026, 1, 1, 9, tzinfo=UTC),
            datetime(2026, 1, 1, 9, tzinfo=UTC),
        ],
    })
    aggregated_events = []
    original_builder = candidate_popularity.like_counts.build_cumulative_like_counts

    def record_aggregated_events(events_df):
        aggregated_events.append(events_df.clone())
        return original_builder(events_df)

    monkeypatch.setattr(
        candidate_popularity.like_counts,
        "build_cumulative_like_counts",
        record_aggregated_events,
    )

    result = candidate_popularity.build_candidate_hour_popularity(
        _candidates({"subject_uri": ["candidate"], "post_created_at": [created_at]}),
        likes,
        _hours(datetime(2026, 1, 1, 11, tzinfo=UTC)),
        max_candidate_age_hours=2,
    )

    assert len(aggregated_events) == 1
    assert aggregated_events[0]["subject_uri"].to_list() == [
        "candidate",
        "candidate",
    ]
    assert result["prior_like_count"].to_list() == [2]


def test_creation_hour_offsets_are_inclusive_then_exclusive():
    created_at = datetime(2026, 1, 1, 10, 59, tzinfo=UTC)
    creation_hour = created_at.replace(minute=0)
    query_hours = _hours(
        *[creation_hour + timedelta(hours=offset) for offset in range(25)]
    )

    result = candidate_popularity.build_candidate_hour_popularity(
        _candidates({"subject_uri": ["post"], "post_created_at": [created_at]}),
        _likes({"subject_uri": [], "like_created_at": []}),
        query_hours,
        max_candidate_age_hours=24,
    )

    assert result["query_hour"].to_list() == query_hours["query_hour"].to_list()[:24]


def test_build_candidate_reservoir_uses_sources_and_rejects_replies():
    posts = pl.DataFrame({
        "subject_uri": ["candidate", "required", "reply"],
        "post_created_at": [datetime(2026, 1, 1, tzinfo=UTC)] * 3,
        "author_did": ["a", "b", "c"],
        "is_reply": [False, False, True],
    })
    sources = pl.DataFrame({
        "subject_uri": ["candidate"],
        "candidate_source": ["random"],
    })

    assert candidate_popularity.build_candidate_reservoir(posts, sources)[
        "subject_uri"
    ].to_list() == ["candidate"]

    with pytest.raises(ValueError, match="contains a reply"):
        candidate_popularity.build_candidate_reservoir(
            posts,
            pl.DataFrame({
                "subject_uri": ["reply"],
                "candidate_source": ["future-source"],
            }),
        )


def test_query_hours_must_be_utc_aligned():
    with pytest.raises(ValueError, match="UTC datetime"):
        candidate_popularity.validate_query_hours(
            pl.DataFrame({"query_hour": [datetime(2026, 1, 1, 10)]})
        )
    with pytest.raises(ValueError, match="aligned"):
        candidate_popularity.validate_query_hours(
            _hours(datetime(2026, 1, 1, 10, 1, tzinfo=UTC))
        )
