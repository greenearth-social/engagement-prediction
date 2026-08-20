from datetime import datetime, timezone
import logging

import polars as pl

from engagement_prediction.data import query_selection_artifacts as artifacts


UTC = timezone.utc


def _run_membership_filter(tmp_path, partition_count):
    posts_path = tmp_path / f"posts-{partition_count}.parquet"
    pl.DataFrame({
        "at_uri": [
            "at://post/found",
            "at://post/found",
            None,
            "at://post/outside",
            "at://post/end",
        ],
        "record_created_at": [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:10:00Z",
            "2026-01-01T00:15:00Z",
            "2025-12-31T23:59:00Z",
            "2026-01-02T00:00:00Z",
        ],
        "did": ["did:author"] * 5,
    }).write_parquet(posts_path)

    query_hour = datetime(2026, 1, 1, 1, tzinfo=UTC)
    positive_rows_lf = pl.DataFrame({
        "did": ["did:one", "did:one", "did:one", "did:two"],
        "query_hour": [query_hour] * 4,
        "user_cohort": ["trainval"] * 4,
        "split": ["train"] * 4,
        "subject_uri": [
            "at://post/found",
            "at://post/found",
            "at://post/missing",
            "at://post/found",
        ],
        "like_created_at": [
            datetime(2026, 1, 1, 1, 5, tzinfo=UTC),
            datetime(2026, 1, 1, 1, 3, tzinfo=UTC),
            datetime(2026, 1, 1, 1, 7, tzinfo=UTC),
            datetime(2026, 1, 1, 1, 9, tzinfo=UTC),
        ],
    }).lazy()
    sampled_queries_lf = pl.DataFrame({
        "did": ["did:one", "did:two"],
        "query_hour": [query_hour, query_hour],
    }).lazy()
    post_rows_path = tmp_path / f"post-rows-{partition_count}"
    provisional_path = tmp_path / f"provisional-{partition_count}"
    eligible_path = tmp_path / f"eligible-{partition_count}"
    logger = logging.getLogger(__name__)

    artifacts.materialize_post_rows(
        post_paths=[str(posts_path)],
        posts_start=datetime(2026, 1, 1, tzinfo=UTC),
        posts_end=datetime(2026, 1, 2, tzinfo=UTC),
        partition_count=partition_count,
        output_path=post_rows_path,
        logger=logger,
    )
    artifacts.materialize_provisional_positive_rows(
        positive_rows_lf=positive_rows_lf,
        sampled_queries_lf=sampled_queries_lf,
        partition_count=partition_count,
        output_path=provisional_path,
        logger=logger,
    )
    stats = artifacts.filter_positive_partitions(
        provisional_positive_rows_path=provisional_path,
        post_rows_path=post_rows_path,
        eligible_positive_rows_path=eligible_path,
        partition_count=partition_count,
        splits=["train"],
        logger=logger,
    )
    eligible = (
        artifacts.scan_eligible_positive_rows(eligible_path)
        .sort(["query_hour", "did", "subject_uri"])
        .collect()
    )
    return eligible, stats


def test_partitioned_membership_deduplicates_before_join_and_preserves_earliest_like(tmp_path):
    eligible, stats = _run_membership_filter(tmp_path, partition_count=3)

    assert eligible.select("did", "subject_uri").rows() == [
        ("did:one", "at://post/found"),
        ("did:two", "at://post/found"),
    ]
    assert eligible.filter(pl.col("did") == "did:one")["like_created_at"].item() == datetime(
        2026, 1, 1, 1, 3, tzinfo=UTC
    )
    assert stats["positive_filter_stats_by_split"]["train"] == {
        "selected_like_row_count": 4,
        "provisional_positive_count": 3,
        "retained_positive_count": 2,
        "missing_post_positive_count": 1,
    }
    assert stats["post_source_stats"] == {
        "post_source_row_count": 5,
        "invalid_post_row_count": 3,
        "duplicate_post_row_count": 1,
        "duplicate_post_uri_count": 1,
        "unique_valid_post_count": 1,
    }


def test_membership_filter_is_logically_independent_of_partition_count(tmp_path):
    one_partition, _ = _run_membership_filter(tmp_path, partition_count=1)
    five_partitions, _ = _run_membership_filter(tmp_path, partition_count=5)

    assert one_partition.equals(five_partitions)


def test_membership_filter_writes_schema_correct_empty_dataset(tmp_path):
    posts_path = tmp_path / "invalid-posts.parquet"
    pl.DataFrame({
        "at_uri": ["at://post/found"],
        "record_created_at": ["2026-01-01T00:05:00Z"],
        "did": [None],
    }).write_parquet(posts_path)
    query_hour = datetime(2026, 1, 1, 1, tzinfo=UTC)
    positive_rows_lf = pl.DataFrame({
        "did": ["did:one"],
        "query_hour": [query_hour],
        "user_cohort": ["trainval"],
        "split": ["train"],
        "subject_uri": ["at://post/found"],
        "like_created_at": [datetime(2026, 1, 1, 1, 5, tzinfo=UTC)],
    }).lazy()
    post_rows_path = tmp_path / "post-rows"
    provisional_path = tmp_path / "provisional"
    eligible_path = tmp_path / "eligible"
    logger = logging.getLogger(__name__)

    artifacts.materialize_post_rows(
        post_paths=[str(posts_path)],
        posts_start=datetime(2026, 1, 1, tzinfo=UTC),
        posts_end=datetime(2026, 1, 2, tzinfo=UTC),
        partition_count=2,
        output_path=post_rows_path,
        logger=logger,
    )
    artifacts.materialize_provisional_positive_rows(
        positive_rows_lf=positive_rows_lf,
        sampled_queries_lf=pl.DataFrame({
            "did": ["did:one"],
            "query_hour": [query_hour],
        }).lazy(),
        partition_count=2,
        output_path=provisional_path,
        logger=logger,
    )
    artifacts.filter_positive_partitions(
        provisional_positive_rows_path=provisional_path,
        post_rows_path=post_rows_path,
        eligible_positive_rows_path=eligible_path,
        partition_count=2,
        splits=["train"],
        logger=logger,
    )

    eligible = artifacts.scan_eligible_positive_rows(eligible_path).collect()
    assert eligible.is_empty()
    assert eligible.schema == artifacts.INTERNAL_POSITIVE_SCHEMA
