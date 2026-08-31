from datetime import datetime, timezone
import math

import polars as pl
import pytest

from engagement_prediction.training.popularity import fit_popularity_normalization


UTC = timezone.utc


def _frames():
    train_hour = datetime(2026, 1, 1, 10, tzinfo=UTC)
    val_hour = datetime(2026, 1, 1, 11, tzinfo=UTC)
    queries = pl.DataFrame({
        "did": ["u1", "u2", "u3"],
        "query_hour": [train_hour, train_hour, val_hour],
        "split": ["train", "train", "val"],
    }).lazy()
    positives = pl.DataFrame({
        "did": ["u1", "u2", "u3"],
        "query_hour": [train_hour, train_hour, val_hour],
        "subject_uri": ["shared", "shared", "validation"],
        "prior_like_count": [3, 3, 9999],
    }).lazy()
    histories = pl.DataFrame({
        "did": ["u1", "u2", "u3"],
        "query_hour": [train_hour, train_hour, val_hour],
        "history_prior_like_counts": [[0, 0], [8], [9999]],
    }).lazy()
    negatives = pl.DataFrame({
        "query_hour": [train_hour, train_hour, val_hour],
        "subject_uri": ["shared", "negative", "validation-negative"],
        "prior_like_count": [3, 15, 9999],
    }).lazy()
    return queries, positives, histories, negatives


def test_fit_popularity_uses_train_history_events_and_unique_candidate_hours():
    queries, positives, histories, negatives = _frames()

    stats = fit_popularity_normalization(
        queries_lf=queries,
        query_positives_lf=positives,
        query_histories_lf=histories,
        hourly_negative_candidates_lf=negatives,
        enabled=True,
    )

    expected = [math.log1p(value) for value in [0, 0, 8, 3, 15]]
    expected_mean = sum(expected) / len(expected)
    expected_std = math.sqrt(
        sum((value - expected_mean) ** 2 for value in expected) / len(expected)
    )
    assert stats.history_observation_count == 3
    assert stats.candidate_observation_count == 2
    assert stats.total_observation_count == 5
    assert stats.log_mean == pytest.approx(expected_mean)
    assert stats.log_std == pytest.approx(expected_std)


def test_fit_popularity_rejects_conflicting_candidate_counts():
    queries, positives, histories, negatives = _frames()
    conflicting = pl.concat([
        negatives.collect(),
        pl.DataFrame({
            "query_hour": [datetime(2026, 1, 1, 10, tzinfo=UTC)],
            "subject_uri": ["shared"],
            "prior_like_count": [4],
        }),
    ]).lazy()

    with pytest.raises(ValueError, match="conflicting prior_like_count"):
        fit_popularity_normalization(
            queries_lf=queries,
            query_positives_lf=positives,
            query_histories_lf=histories,
            hourly_negative_candidates_lf=conflicting,
            enabled=True,
        )


def test_fit_popularity_uses_identity_stats_for_zero_variance_and_empty_inputs():
    hour = datetime(2026, 1, 1, 10, tzinfo=UTC)
    stats = fit_popularity_normalization(
        queries_lf=pl.DataFrame({
            "did": ["u1"],
            "query_hour": [hour],
            "split": ["train"],
        }).lazy(),
        query_positives_lf=pl.DataFrame({
            "did": ["u1"],
            "query_hour": [hour],
            "subject_uri": ["p1"],
            "prior_like_count": [4],
        }).lazy(),
        query_histories_lf=pl.DataFrame({
            "did": ["u1"],
            "query_hour": [hour],
            "history_prior_like_counts": [[4, 4]],
        }).lazy(),
        hourly_negative_candidates_lf=pl.DataFrame({
            "query_hour": [],
            "subject_uri": [],
            "prior_like_count": [],
        }, schema={
            "query_hour": pl.Datetime(time_zone="UTC"),
            "subject_uri": pl.String,
            "prior_like_count": pl.UInt64,
        }).lazy(),
        enabled=True,
    )

    assert stats.log_mean == pytest.approx(math.log1p(4))
    assert stats.log_std == 1.0
    assert stats.total_observation_count == 3

    empty_stats = fit_popularity_normalization(
        queries_lf=pl.DataFrame({
            "did": [],
            "query_hour": [],
            "split": [],
        }, schema={
            "did": pl.String,
            "query_hour": pl.Datetime(time_zone="UTC"),
            "split": pl.String,
        }).lazy(),
        query_positives_lf=pl.DataFrame({
            "did": [],
            "query_hour": [],
            "subject_uri": [],
            "prior_like_count": [],
        }, schema={
            "did": pl.String,
            "query_hour": pl.Datetime(time_zone="UTC"),
            "subject_uri": pl.String,
            "prior_like_count": pl.UInt64,
        }).lazy(),
        query_histories_lf=pl.DataFrame({
            "did": [],
            "query_hour": [],
            "history_prior_like_counts": [],
        }, schema={
            "did": pl.String,
            "query_hour": pl.Datetime(time_zone="UTC"),
            "history_prior_like_counts": pl.List(pl.UInt64),
        }).lazy(),
        hourly_negative_candidates_lf=pl.DataFrame({
            "query_hour": [],
            "subject_uri": [],
            "prior_like_count": [],
        }, schema={
            "query_hour": pl.Datetime(time_zone="UTC"),
            "subject_uri": pl.String,
            "prior_like_count": pl.UInt64,
        }).lazy(),
        enabled=True,
    )
    assert empty_stats.log_mean == 0.0
    assert empty_stats.log_std == 1.0
    assert empty_stats.total_observation_count == 0


def test_disabled_popularity_does_not_fit_inputs():
    stats = fit_popularity_normalization(
        queries_lf=None,
        query_positives_lf=None,
        query_histories_lf=None,
        hourly_negative_candidates_lf=None,
        enabled=False,
    )

    assert stats.to_dict() == {
        "enabled": False,
        "log_mean": 0.0,
        "log_std": 1.0,
        "history_observation_count": 0,
        "candidate_observation_count": 0,
        "total_observation_count": 0,
    }
