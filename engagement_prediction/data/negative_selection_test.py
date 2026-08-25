from datetime import datetime, timezone

import polars as pl
import pytest

from engagement_prediction.data import candidate_popularity, negative_selection


UTC = timezone.utc


def _candidate_hours(rows):
    return pl.DataFrame(rows, schema=candidate_popularity.CANDIDATE_HOUR_SCHEMA)


def _select(rows, *, k=4, minimum=10, fraction=0.5, seed=42):
    local = negative_selection.select_local_finalists(
        _candidate_hours(rows),
        negative_candidates_per_hour=k,
        min_likes_for_popular_candidate=minimum,
        random_seed=seed,
    )
    return negative_selection.select_hourly_candidates(
        local,
        negative_candidates_per_hour=k,
        popular_candidate_fraction=fraction,
    )


def test_popular_first_then_random_fills_without_duplicates():
    hour = datetime(2026, 1, 1, 10, tzinfo=UTC)
    rows = {
        "query_hour": [hour] * 6,
        "subject_uri": [f"p{index}" for index in range(6)],
        "post_created_at": [datetime(2026, 1, 1, 9, tzinfo=UTC)] * 6,
        "prior_like_count": [20, 15, 12, 11, 2, 0],
    }

    result = _select(rows)

    assert result.height == 4
    assert result.filter(pl.col("selection_source") == "popular").height == 2
    assert result.filter(pl.col("selection_source") == "random").height == 2
    assert result.unique(subset=["query_hour", "subject_uri"]).height == 4
    # The random method is allowed to select a post that also meets N.
    assert result.filter(
        (pl.col("selection_source") == "random")
        & (pl.col("prior_like_count") >= 10)
    ).height == 1


def test_random_fills_a_popular_shortfall_and_total_shortage_is_reportable():
    hour = datetime(2026, 1, 1, 10, tzinfo=UTC)
    rows = {
        "query_hour": [hour] * 3,
        "subject_uri": ["popular", "low-one", "low-two"],
        "post_created_at": [datetime(2026, 1, 1, 9, tzinfo=UTC)] * 3,
        "prior_like_count": [10, 1, 0],
    }

    result = _select(rows, k=4, fraction=0.75)

    assert result.height == 3
    assert result.filter(pl.col("selection_source") == "popular").height == 1
    assert result.filter(pl.col("selection_source") == "random").height == 2


@pytest.mark.parametrize(
    ("fraction", "expected_popular"),
    [(0.0, 0), (1.0, 3)],
)
def test_fraction_extremes(fraction, expected_popular):
    hour = datetime(2026, 1, 1, 10, tzinfo=UTC)
    rows = {
        "query_hour": [hour] * 4,
        "subject_uri": ["a", "b", "c", "d"],
        "post_created_at": [datetime(2026, 1, 1, 9, tzinfo=UTC)] * 4,
        "prior_like_count": [10, 10, 10, 0],
    }

    result = _select(rows, k=3, fraction=fraction)

    assert result.height == 3
    assert result.filter(pl.col("selection_source") == "popular").height == expected_popular


def test_zero_k_and_zero_minimum_are_supported():
    hour = datetime(2026, 1, 1, 10, tzinfo=UTC)
    rows = {
        "query_hour": [hour],
        "subject_uri": ["zero"],
        "post_created_at": [datetime(2026, 1, 1, 9, tzinfo=UTC)],
        "prior_like_count": [0],
    }

    assert _select(rows, k=0).is_empty()
    result = _select(rows, k=1, minimum=0, fraction=1.0)
    assert result.to_dicts()[0]["selection_source"] == "popular"


def test_selection_is_stable_and_seed_sensitive():
    hour = datetime(2026, 1, 1, 10, tzinfo=UTC)
    rows = {
        "query_hour": [hour] * 100,
        "subject_uri": [f"p{index}" for index in range(100)],
        "post_created_at": [datetime(2026, 1, 1, 9, tzinfo=UTC)] * 100,
        "prior_like_count": list(range(100)),
    }

    first = _select(rows, k=20, seed=42)
    second = _select(rows, k=20, seed=42)
    changed = _select(rows, k=20, seed=43)

    assert first.equals(second)
    assert not first.equals(changed)


def test_local_finalist_merge_matches_single_partition_result():
    hour = datetime(2026, 1, 1, 10, tzinfo=UTC)
    rows = _candidate_hours({
        "query_hour": [hour] * 30,
        "subject_uri": [f"p{index}" for index in range(30)],
        "post_created_at": [datetime(2026, 1, 1, 9, tzinfo=UTC)] * 30,
        "prior_like_count": list(range(30)),
    })
    all_local = negative_selection.select_local_finalists(
        rows,
        negative_candidates_per_hour=8,
        min_likes_for_popular_candidate=10,
        random_seed=42,
    )
    partitioned_local = pl.concat([
        negative_selection.select_local_finalists(
            rows.filter(pl.col("subject_uri").str.slice(1).cast(pl.Int64) % 3 == partition),
            negative_candidates_per_hour=8,
            min_likes_for_popular_candidate=10,
            random_seed=42,
        )
        for partition in range(3)
    ])

    expected = negative_selection.select_hourly_candidates(
        all_local,
        negative_candidates_per_hour=8,
        popular_candidate_fraction=0.5,
    )
    actual = negative_selection.select_hourly_candidates(
        partitioned_local,
        negative_candidates_per_hour=8,
        popular_candidate_fraction=0.5,
    )

    assert actual.equals(expected)


@pytest.mark.parametrize(
    ("k", "fraction", "message"),
    [(-1, 0.5, "non-negative"), (10, -0.1, "between 0 and 1"), (10, 1.1, "between 0 and 1")],
)
def test_quota_validation(k, fraction, message):
    with pytest.raises(ValueError, match=message):
        negative_selection.calculate_popular_quota(k, fraction)


def test_round_half_up_quota():
    assert negative_selection.calculate_popular_quota(1, 0.5) == 1
    assert negative_selection.calculate_popular_quota(3, 0.5) == 2
