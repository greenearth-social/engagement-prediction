from datetime import datetime, timedelta, timezone
import json

import polars as pl
import pytest

from engagement_prediction.data import post_selection


UTC = timezone.utc


def _inference_json(news_score=None, politics_score=None):
    record = {"topic": {}}
    if news_score is not None:
        record["topic"]["News & Social Concern"] = news_score
    if politics_score is not None:
        record["text_arbitrary"] = {"Politics": politics_score}
    return json.dumps({"text": {"message.commit.record.text": record}})


def test_post_normalization_keeps_latest_created_row_with_stable_author_tie():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    normalized = post_selection.normalize_posts(
        pl.DataFrame({
            "at_uri": ["p1", "p1", "p1", "p2", None],
            "record_created_at": [
                "2026-01-01T01:00:00Z",
                "2026-01-01T02:00:00Z",
                "2026-01-01T02:00:00Z",
                "2026-01-02T01:00:00Z",
                "bad",
            ],
            "did": ["z", "z", "a", "outside", "invalid"],
        }).lazy(),
        posts_start=start,
        posts_end=end,
    ).collect()

    selected, stats = post_selection.select_latest_post_rows(normalized)

    assert selected.to_dicts() == [{
        "subject_uri": "p1",
        "post_created_at": datetime(2026, 1, 1, 2, tzinfo=UTC),
        "author_did": "a",
    }]
    assert stats == {
        "post_source_row_count": 5,
        "invalid_post_row_count": 2,
        "duplicate_post_row_count": 2,
        "duplicate_post_uri_count": 1,
        "unique_valid_post_count": 1,
    }


def test_random_candidate_hash_is_stable_and_supports_fraction_boundaries():
    posts = pl.DataFrame({"subject_uri": [f"p{idx}" for idx in range(100)]})
    first = posts.filter(post_selection.random_candidate_expr(0.1, 42))
    second = posts.filter(post_selection.random_candidate_expr(0.1, 42))
    other_seed = posts.filter(post_selection.random_candidate_expr(0.1, 43))

    assert first.equals(second)
    assert not first.equals(other_seed)
    assert posts.filter(post_selection.random_candidate_expr(0.0, 42)).is_empty()
    assert posts.filter(post_selection.random_candidate_expr(1.0, 42)).height == 100


def test_latest_news_social_concern_inference_controls_political_label():
    normalized = post_selection.normalize_inferences(
        pl.DataFrame({
            "at_uri": ["p1", "p1", "p2", "p3", "p4"],
            "indexed_at": [
                "2026-01-01T01:00:00Z",
                "2026-01-01T02:00:00Z",
                "2026-01-01T01:00:00Z",
                "2026-01-01T01:00:00Z",
                "2026-01-01T01:00:00Z",
            ],
            "inferences": [
                _inference_json(0.99, 0.0),
                _inference_json(0.94, 1.0),
                _inference_json(0.95, 0.0),
                _inference_json(None, 1.0),
                _inference_json(1.5, 1.0),
            ],
        }).lazy()
    ).collect()
    latest, stats = post_selection.select_latest_inferences(
        normalized,
        political_score_threshold=0.95,
    )

    assert latest.select("subject_uri", "news_social_concern_score", "is_political").to_dicts() == [
        {"subject_uri": "p1", "news_social_concern_score": 0.94, "is_political": False},
        {"subject_uri": "p2", "news_social_concern_score": 0.95, "is_political": True},
        {"subject_uri": "p3", "news_social_concern_score": None, "is_political": False},
        {"subject_uri": "p4", "news_social_concern_score": None, "is_political": False},
    ]
    assert stats["missing_or_invalid_inference_score_count"] == 2


def test_conflicting_equal_timestamp_inferences_fail():
    normalized = post_selection.normalize_inferences(
        pl.DataFrame({
            "at_uri": ["p1", "p1"],
            "indexed_at": ["2026-01-01T01:00:00Z"] * 2,
            "inferences": [_inference_json(0.9), _inference_json(0.99)],
        }).lazy()
    ).collect()

    with pytest.raises(ValueError, match="Conflicting political inference"):
        post_selection.select_latest_inferences(
            normalized,
            political_score_threshold=0.95,
        )


def test_conflicting_invalid_numeric_scores_at_equal_timestamps_fail():
    normalized = post_selection.normalize_inferences(
        pl.DataFrame({
            "at_uri": ["p1", "p1"],
            "indexed_at": ["2026-01-01T01:00:00Z"] * 2,
            "inferences": [_inference_json(1.5), _inference_json(-1.0)],
        }).lazy()
    ).collect()

    with pytest.raises(ValueError, match="Conflicting political inference"):
        post_selection.select_latest_inferences(
            normalized,
            political_score_threshold=0.95,
        )


def test_political_cap_is_independent_per_creation_hour_and_deterministic():
    first_hour = datetime(2026, 1, 1, 1, tzinfo=UTC)
    rows = [
        {
            "subject_uri": f"p{idx:04d}",
            "post_created_at": first_hour + timedelta(hours=idx // 1001),
            "_political_priority": idx,
        }
        for idx in range(2002)
    ]
    eligible = pl.DataFrame(rows)

    selected, stats = post_selection.select_political_candidates_for_day(
        eligible,
        max_candidates_per_creation_hour=1000,
    )

    assert selected.height == 2000
    assert [row["eligible_count"] for row in stats] == [1001, 1001]
    assert [row["selected_count"] for row in stats] == [1000, 1000]
    assert "p1000" not in selected["subject_uri"]
    assert "p2001" not in selected["subject_uri"]


def test_public_partition_validation_enforces_missing_and_candidate_contracts():
    created = datetime(2026, 1, 1, tzinfo=UTC)
    all_uris = pl.DataFrame({"subject_uri": ["found", "missing"]}).with_columns(
        post_selection.post_partition_expr(1)
    )
    assert all_uris["_post_partition"].to_list() == [0, 0]
    posts = pl.DataFrame({
        "subject_uri": ["found"],
        "post_created_at": [created],
        "author_did": ["author"],
        "news_social_concern_score": pl.Series([None], dtype=pl.Float64),
        "political_inference_indexed_at": pl.Series(
            [None], dtype=post_selection.UTC_DATETIME
        ),
        "is_political": pl.Series([None], dtype=pl.Boolean),
    })
    required = pl.DataFrame({
        "subject_uri": ["found", "missing"],
        "is_positive": [True, False],
        "is_history": [False, True],
    })
    candidates = pl.DataFrame({
        "subject_uri": ["found"],
        "candidate_source": ["random"],
    })
    missing = required.filter(pl.col("subject_uri") == "missing")

    post_selection.validate_public_partition(
        posts_df=posts,
        required_posts_df=required,
        candidate_sources_df=candidates,
        missing_required_posts_df=missing,
        partition_id=0,
        partition_count=1,
        political_score_threshold=0.95,
    )


def test_public_partition_validation_rejects_null_keys():
    required = pl.DataFrame({
        "subject_uri": pl.Series([None], dtype=pl.String),
        "is_positive": [True],
        "is_history": [False],
    })

    with pytest.raises(ValueError, match="null subject_uri"):
        post_selection.validate_public_partition(
            posts_df=post_selection.empty_frame(post_selection.POST_SCHEMA),
            required_posts_df=required,
            candidate_sources_df=post_selection.empty_frame(
                post_selection.CANDIDATE_SOURCE_SCHEMA
            ),
            missing_required_posts_df=required,
            partition_id=0,
            partition_count=1,
            political_score_threshold=0.95,
        )
