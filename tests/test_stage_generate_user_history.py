"""
Tests for stage_generate_user_history.py (user-hour history directory).
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture(scope="session")
def stage_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "utils" / "02_user_history" / "stage_generate_user_history.py"
    spec = importlib.util.spec_from_file_location("stage_generate_user_history", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage_generate_user_history"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def build_history(stage_module):
    return stage_module._build_user_history_directory


@pytest.fixture
def build_post_liker_user_idx(stage_module):
    return stage_module._build_post_liker_user_idx


@pytest.fixture
def build_needed_post_hour_pairs(stage_module):
    return stage_module._build_needed_post_hour_pairs


@pytest.fixture
def build_post_recent_likers(stage_module):
    return stage_module._build_post_recent_likers


def _make_test_logger() -> logging.Logger:
    logger = logging.getLogger("test_user_history")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    return logger


def _hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)


def _make_likes(
    dids: list[str],
    timestamps: list[datetime],
    subject_uris: list[str],
    emb_idxs: list[int],
    prior_cumulative_likes: list[int | None] | None = None,
    author_idxs: list[int | None] | None = None,
    like_hour_buckets: list[datetime] | None = None,
    splits: list[str] | None = None,
) -> pl.LazyFrame:
    data = {
        "did": dids,
        "record_created_at": timestamps,
        "like_hour_bucket": like_hour_buckets or [_hour(ts) for ts in timestamps],
        "subject_uri": subject_uris,
        "emb_idx": emb_idxs,
        "prior_cumulative_likes": [0] * len(emb_idxs) if prior_cumulative_likes is None else prior_cumulative_likes,
    }
    if author_idxs is not None:
        data["author_idx"] = author_idxs
    if splits is not None:
        data["split"] = splits
    return pl.DataFrame(data).lazy()


def _make_posts(
    at_uris: list[str],
    in_random_sample: list[bool],
    negative_hour_buckets: list[datetime | None],
) -> pl.LazyFrame:
    return pl.DataFrame({
        "at_uri": pl.Series(at_uris, dtype=pl.String),
        "in_random_sample": pl.Series(in_random_sample, dtype=pl.Boolean),
        "negative_hour_bucket": pl.Series(negative_hour_buckets, dtype=pl.Datetime),
    }).lazy()


def _history_by_bucket(df: pl.DataFrame) -> dict[datetime, list[int]]:
    return {
        row["like_hour_bucket"]: list(row["prior_emb_indices"])
        for row in df.iter_rows(named=True)
    }


def _history_ages_by_bucket(df: pl.DataFrame) -> dict[datetime, list[float]]:
    return {
        row["like_hour_bucket"]: list(row["prior_like_age_hours_at_bucket_start"])
        for row in df.iter_rows(named=True)
    }


def _history_popularity_by_bucket(df: pl.DataFrame) -> dict[datetime, list[int]]:
    return {
        row["like_hour_bucket"]: list(row["prior_cumulative_likes"])
        for row in df.iter_rows(named=True)
    }


def test_user_hour_history_preserves_empty_first_bucket(build_history):
    logger = _make_test_logger()
    likes_lf = _make_likes(
        ["u1", "u1", "u1"],
        [
            datetime(2024, 1, 1, 10, 15),
            datetime(2024, 1, 1, 11, 20),
            datetime(2024, 1, 1, 12, 5),
        ],
        ["p1", "p2", "p3"],
        [100, 200, 300],
        [5, 15, 25],
    )

    result = build_history(
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect().sort("like_hour_bucket")

    assert result.height == 3
    histories = _history_by_bucket(result)
    assert histories[datetime(2024, 1, 1, 10)] == []
    assert histories[datetime(2024, 1, 1, 11)] == [100]
    assert histories[datetime(2024, 1, 1, 12)] == [200, 100]
    age_histories = _history_ages_by_bucket(result)
    assert age_histories[datetime(2024, 1, 1, 10)] == []
    assert age_histories[datetime(2024, 1, 1, 11)] == pytest.approx([0.75])
    assert age_histories[datetime(2024, 1, 1, 12)] == pytest.approx([2.0 / 3.0, 1.75])
    popularity_histories = _history_popularity_by_bucket(result)
    assert popularity_histories[datetime(2024, 1, 1, 10)] == []
    assert popularity_histories[datetime(2024, 1, 1, 11)] == [5]
    assert popularity_histories[datetime(2024, 1, 1, 12)] == [15, 5]
    assert result.filter(pl.col("like_hour_bucket") == datetime(2024, 1, 1, 10))["raw_prior_count"][0] == 0


def test_user_hour_history_recency_ordering_and_capping(build_history):
    logger = _make_test_logger()
    likes_lf = _make_likes(
        ["u1", "u1", "u1", "u1"],
        [
            datetime(2024, 1, 5, 0, 0),
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 10, 0, 0),
            datetime(2024, 1, 7, 0, 0),
        ],
        ["p1", "p2", "p3", "p4"],
        [10, 20, 30, 40],
        [50, 10, 100, 70],
    )

    result = build_history(
        likes_lf=likes_lf,
        max_prior_likes=2,
        logger=logger,
    ).collect()

    row = result.filter(pl.col("like_hour_bucket") == datetime(2024, 1, 10))
    assert row["prior_emb_indices"][0].to_list() == [40, 10]
    assert row["prior_like_age_hours_at_bucket_start"][0].to_list() == pytest.approx([72.0, 120.0])
    assert row["prior_cumulative_likes"][0].to_list() == [70, 50]
    assert row["raw_prior_count"][0] == 3


def test_user_hour_history_excludes_same_hour_likes(build_history):
    logger = _make_test_logger()
    likes_lf = _make_likes(
        ["u1", "u1", "u1"],
        [
            datetime(2024, 1, 1, 10, 5),
            datetime(2024, 1, 1, 11, 10),
            datetime(2024, 1, 1, 11, 50),
        ],
        ["p1", "p2", "p3"],
        [1, 2, 3],
        [7, 8, 9],
    )

    result = build_history(
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    row = result.filter(pl.col("like_hour_bucket") == datetime(2024, 1, 1, 11))
    assert row["prior_emb_indices"][0].to_list() == [1]
    assert row["prior_like_age_hours_at_bucket_start"][0].to_list() == pytest.approx([55.0 / 60.0])
    assert row["prior_cumulative_likes"][0].to_list() == [7]
    assert row["raw_prior_count"][0] == 1


def test_user_hour_history_multiple_users(build_history):
    logger = _make_test_logger()
    likes_lf = _make_likes(
        ["u1", "u1", "u2", "u2"],
        [
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 2, 0, 0),
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 3, 0, 0),
        ],
        ["a1", "a2", "b1", "b2"],
        [1, 2, 11, 12],
    )

    result = build_history(
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    histories = {
        (row["did"], row["like_hour_bucket"]): list(row["prior_emb_indices"])
        for row in result.iter_rows(named=True)
    }
    assert histories[("u1", datetime(2024, 1, 2))] == [1]
    assert histories[("u2", datetime(2024, 1, 3))] == [11]


def test_user_hour_history_output_schema(build_history):
    logger = _make_test_logger()
    likes_lf = _make_likes(
        ["u1"],
        [datetime(2024, 1, 1, 0, 0)],
        ["p1"],
        [100],
    )

    result = build_history(
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    assert result.columns == [
        "did",
        "like_hour_bucket",
        "prior_emb_indices",
        "raw_prior_count",
        "prior_like_age_hours_at_bucket_start",
        "prior_cumulative_likes",
    ]
    assert result.schema["prior_emb_indices"] == pl.List(pl.UInt32)
    assert result.schema["prior_like_age_hours_at_bucket_start"] == pl.List(pl.Float32)
    assert result.schema["prior_cumulative_likes"] == pl.List(pl.UInt64)


def test_user_hour_history_requires_prior_cumulative_likes(build_history):
    logger = _make_test_logger()
    likes_lf = pl.DataFrame({
        "did": ["u1"],
        "record_created_at": [datetime(2024, 1, 1, 0, 0)],
        "like_hour_bucket": [datetime(2024, 1, 1, 0, 0)],
        "subject_uri": ["p1"],
        "emb_idx": [100],
    }).lazy()

    with pytest.raises(ValueError, match="prior_cumulative_likes"):
        build_history(
            likes_lf=likes_lf,
            max_prior_likes=None,
            logger=logger,
        )


def test_user_hour_author_indices_preserve_order_and_unknowns(build_history):
    logger = _make_test_logger()
    likes_lf = _make_likes(
        ["u1", "u1", "u1", "u1"],
        [
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            datetime(2024, 1, 1, 12, 0),
            datetime(2024, 1, 1, 13, 0),
        ],
        ["p1", "p2", "p3", "p4"],
        [100, 200, 300, 400],
        [10, None, 30, 40],
        author_idxs=[2, None, 4, 9],
    )

    result = build_history(
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    row = result.filter(pl.col("like_hour_bucket") == datetime(2024, 1, 1, 13))
    assert row["prior_emb_indices"][0].to_list() == [300, 200, 100]
    assert row["prior_like_age_hours_at_bucket_start"][0].to_list() == pytest.approx([1.0, 2.0, 3.0])
    assert row["prior_cumulative_likes"][0].to_list() == [30, 0, 10]
    assert row["prior_author_indices"][0].to_list() == [4, None, 2]


def test_user_hour_without_author_idx_omits_author_history(build_history):
    logger = _make_test_logger()
    likes_lf = _make_likes(
        ["u1"],
        [datetime(2024, 1, 1, 10, 0)],
        ["p1"],
        [100],
    )

    result = build_history(
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    assert "prior_author_indices" not in result.columns


def test_post_liker_user_idx_uses_train_support_threshold_and_stable_offset(build_post_liker_user_idx):
    likes_lf = _make_likes(
        ["u2", "u1", "u1", "u2", "u3", "u3", "u3", "u4"],
        [
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 1, 1, 0),
            datetime(2024, 1, 1, 2, 0),
            datetime(2024, 1, 1, 3, 0),
            datetime(2024, 1, 1, 4, 0),
            datetime(2024, 1, 1, 5, 0),
            datetime(2024, 1, 1, 6, 0),
            datetime(2024, 1, 1, 7, 0),
        ],
        ["p1", "p1", "p2", "p2", "p3", "p4", "p5", "p6"],
        [1, 2, 3, 4, 5, 6, 7, 8],
        splits=["train", "train", "train", "train", "val", "val", "val", "train"],
    )

    result = build_post_liker_user_idx(likes_lf, min_user_support=2).collect()

    assert result.to_dict(as_series=False) == {
        "did": ["u1", "u2"],
        "user_train_like_count": [2, 2],
        "user_idx": [2, 3],
    }
    assert result.schema["user_train_like_count"] == pl.UInt64
    assert result.schema["user_idx"] == pl.UInt32


def test_needed_post_hour_pairs_include_positive_and_sampled_negative_pairs(
    build_needed_post_hour_pairs,
):
    likes_lf = _make_likes(
        ["u1", "u2"],
        [datetime(2024, 1, 1, 10, 5), datetime(2024, 1, 1, 11, 5)],
        ["p1", "p2"],
        [1, 2],
    )
    posts_lf = _make_posts(
        ["p2", "p3", "p4"],
        [True, True, False],
        [datetime(2024, 1, 1, 11), datetime(2024, 1, 1, 12), datetime(2024, 1, 1, 13)],
    )

    result = build_needed_post_hour_pairs(likes_lf, posts_lf).collect().sort(["subject_uri", "target_hour"])

    assert result.to_dict(as_series=False) == {
        "subject_uri": ["p1", "p2", "p3"],
        "target_hour": [
            datetime(2024, 1, 1, 10),
            datetime(2024, 1, 1, 11),
            datetime(2024, 1, 1, 12),
        ],
    }


def test_post_recent_likers_are_prior_recent_indexed_and_counted(build_post_recent_likers):
    likes_lf = _make_likes(
        ["u1", "u4", "u2", "u3", "u5"],
        [
            datetime(2024, 1, 1, 9, 10),
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 10, 30),
            datetime(2024, 1, 1, 11, 5),
            datetime(2024, 1, 1, 10, 15),
        ],
        ["p1", "p1", "p1", "p1", "p2"],
        [1, 2, 3, 4, 5],
        splits=["train", "train", "train", "train", "train"],
    )
    posts_lf = _make_posts(
        ["p2"],
        [True],
        [datetime(2024, 1, 1, 10)],
    )
    user_idx_lf = pl.DataFrame({
        "did": ["u1", "u2"],
        "user_train_like_count": [1, 1],
        "user_idx": [2, 3],
    }).lazy()

    result = build_post_recent_likers(
        likes_lf=likes_lf,
        posts_lf=posts_lf,
        user_idx_lf=user_idx_lf,
        max_recent_likers_per_post=100,
    ).collect().sort(["subject_uri", "target_hour"])

    p1 = result.filter(
        (pl.col("subject_uri") == "p1")
        & (pl.col("target_hour") == datetime(2024, 1, 1, 11))
    )
    assert p1["dataset_prior_liker_count"][0] == 3
    assert p1["indexed_prior_liker_count"][0] == 2
    assert p1["prior_recent_liker_user_indices"][0].to_list() == [3, 2]
    assert p1["prior_recent_liker_timestamps"][0].to_list() == [
        datetime(2024, 1, 1, 10, 30),
        datetime(2024, 1, 1, 9, 10),
    ]

    p2_negative = result.filter(
        (pl.col("subject_uri") == "p2")
        & (pl.col("target_hour") == datetime(2024, 1, 1, 10))
    )
    assert p2_negative["dataset_prior_liker_count"][0] == 0
    assert p2_negative["indexed_prior_liker_count"][0] == 0
    assert p2_negative["prior_recent_liker_user_indices"][0].to_list() == []
    assert p2_negative["prior_recent_liker_timestamps"][0].to_list() == []
    assert result.schema["prior_recent_liker_user_indices"] == pl.List(pl.UInt32)
    assert result.schema["prior_recent_liker_timestamps"] == pl.List(pl.Datetime(time_unit="us"))


def test_post_recent_likers_emit_empty_lists_when_no_users_are_indexed(build_post_recent_likers):
    likes_lf = _make_likes(
        ["u1", "u2"],
        [datetime(2024, 1, 1, 9, 0), datetime(2024, 1, 1, 10, 5)],
        ["p1", "p1"],
        [1, 2],
        splits=["train", "train"],
    )
    posts_lf = _make_posts([], [], [])
    user_idx_lf = pl.DataFrame({
        "did": pl.Series([], dtype=pl.String),
        "user_train_like_count": pl.Series([], dtype=pl.UInt64),
        "user_idx": pl.Series([], dtype=pl.UInt32),
    }).lazy()

    result = build_post_recent_likers(
        likes_lf=likes_lf,
        posts_lf=posts_lf,
        user_idx_lf=user_idx_lf,
        max_recent_likers_per_post=100,
    ).collect()

    row = result.filter(
        (pl.col("subject_uri") == "p1")
        & (pl.col("target_hour") == datetime(2024, 1, 1, 10))
    )
    assert row["dataset_prior_liker_count"][0] == 1
    assert row["indexed_prior_liker_count"][0] == 0
    assert row["prior_recent_liker_user_indices"][0].to_list() == []
    assert row["prior_recent_liker_timestamps"][0].to_list() == []


def test_post_recent_likers_cap_stored_rows_without_changing_pre_cap_count(build_post_recent_likers):
    timestamps = [
        datetime(2024, 1, 1, 8, 0),
        datetime(2024, 1, 1, 9, 0),
        datetime(2024, 1, 1, 10, 0),
        datetime(2024, 1, 1, 11, 0),
        datetime(2024, 1, 1, 12, 0),
        datetime(2024, 1, 1, 13, 0),
    ]
    likes_lf = _make_likes(
        ["u1", "u2", "u3", "u4", "u5", "target"],
        timestamps,
        ["p1", "p1", "p1", "p1", "p1", "p1"],
        [1, 2, 3, 4, 5, 6],
        splits=["train"] * 6,
    )
    posts_lf = _make_posts([], [], [])
    user_idx_lf = pl.DataFrame({
        "did": ["u1", "u2", "u3", "u4", "u5"],
        "user_train_like_count": [1, 1, 1, 1, 1],
        "user_idx": [2, 3, 4, 5, 6],
    }).lazy()

    result = build_post_recent_likers(
        likes_lf=likes_lf,
        posts_lf=posts_lf,
        user_idx_lf=user_idx_lf,
        max_recent_likers_per_post=3,
    ).collect()

    row = result.filter(
        (pl.col("subject_uri") == "p1")
        & (pl.col("target_hour") == datetime(2024, 1, 1, 13))
    )
    assert row["dataset_prior_liker_count"][0] == 5
    assert row["indexed_prior_liker_count"][0] == 5
    assert row["prior_recent_liker_user_indices"][0].to_list() == [6, 5, 4]
    assert row["prior_recent_liker_timestamps"][0].to_list() == [
        datetime(2024, 1, 1, 12),
        datetime(2024, 1, 1, 11),
        datetime(2024, 1, 1, 10),
    ]
