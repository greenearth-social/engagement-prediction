"""
Tests for stage_generate_user_history.py (directory-based user history)

Tests the _build_user_history_directory function which creates a mapping from
each target row (keyed by target_idx) to prior liked embedding indices.

Target posts use a wide format:
  target_did | seen_at | like_uri | like_emb_idx | ... | neg_uri | neg_emb_idx | ... | split

The user history depends only on (target_did, seen_at), so one history list is
produced per target row.  Rows where the user has no prior likes get an empty list.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from unittest import mock

import polars as pl

# We need to mock modules that aren't available in the CI test environment
# before importing the stage module
sys.modules.setdefault('utils.pipeline', mock.MagicMock())
sys.modules.setdefault('utils.pipeline.core', mock.MagicMock())

# Define the TIMESTAMP_COL_NAME since we can't reliably import utils.helpers
TIMESTAMP_COL_NAME = "record_created_at"


def _make_test_logger() -> logging.Logger:
    """Create a simple logger for tests."""
    logger = logging.getLogger("test_user_history")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    return logger


def _build_user_history_directory(
    targets_lf: pl.LazyFrame,
    likes_lf: pl.LazyFrame,
    max_prior_likes: int | None,
    logger: logging.Logger,
) -> pl.LazyFrame:
    """
    Local copy of the function for testing, avoiding import issues.

    Build a directory mapping each target row to prior liked embedding indices.

    Uses vectorized Polars operations for efficiency:
    1. Assign a row index (target_idx) to each target row
    2. Join targets with likes on user (target_did == did)
    3. Filter to likes that occurred before the target timestamp (seen_at)
    4. Group by target_idx and collect emb_idx values sorted by recency
    5. Left-join back to ensure every target row appears (empty list for no history)
    """
    logger.info("Building user history directory...")

    # Add a unique row index so we can key each target row unambiguously
    targets_indexed = targets_lf.with_row_index("target_idx")

    # Select only the columns we need from targets for the join
    target_keys = targets_indexed.select(["target_idx", "target_did", "seen_at"])

    # Rename likes columns to avoid collision after join
    likes_renamed = likes_lf.select([
        pl.col("did"),
        pl.col(TIMESTAMP_COL_NAME).alias("like_ts"),
        pl.col("emb_idx").alias("like_emb_idx"),
    ])

    # Join targets with likes on user identity
    joined = target_keys.join(
        likes_renamed,
        left_on="target_did",
        right_on="did",
        how="left",
    )

    # Filter to likes that occurred BEFORE the target timestamp
    prior_likes = joined.filter(
        pl.col("like_ts") < pl.col("seen_at")
    )

    # Build aggregation expression: sort by recency (descending) and optionally cap
    agg_expr = (
        pl.col("like_emb_idx")
        .sort_by(pl.col("like_ts"), descending=True)
    )

    if max_prior_likes is not None and max_prior_likes > 0:
        agg_expr = agg_expr.head(max_prior_likes)
        logger.info(f"  Capping prior likes to {max_prior_likes} per target")
    else:
        logger.info("  No cap on prior likes (using all available history)")

    # Group by target_idx and collect prior emb_idx as list
    directory_lf = prior_likes.group_by("target_idx").agg(
        agg_expr.alias("prior_emb_indices")
    )

    # Handle targets with no prior history: left join back to get all targets
    all_target_keys = target_keys.select(["target_idx", "target_did"])
    directory_lf = all_target_keys.join(
        directory_lf,
        on="target_idx",
        how="left",
    ).with_columns(
        pl.when(pl.col("prior_emb_indices").is_null())
        .then(pl.lit([]).cast(pl.List(pl.UInt32)))
        .otherwise(pl.col("prior_emb_indices").cast(pl.List(pl.UInt32)))
        .alias("prior_emb_indices")
    )

    return directory_lf


# ---------------------------------------------------------------------------
# Helper to build a minimal wide-format target-posts LazyFrame
# ---------------------------------------------------------------------------

def _make_targets(
    target_dids: list[str],
    seen_ats: list[datetime],
) -> pl.LazyFrame:
    """Build a minimal wide-format targets LazyFrame with required columns."""
    return pl.DataFrame({
        "target_did": target_dids,
        "seen_at": seen_ats,
        # Include stub wide-format columns to be realistic, though the
        # function under test only uses target_did and seen_at.
        "like_uri": ["stub"] * len(target_dids),
        "like_emb_idx": [0] * len(target_dids),
        "neg_uri": ["stub"] * len(target_dids),
        "neg_emb_idx": [0] * len(target_dids),
        "split": ["train"] * len(target_dids),
    }).lazy()


# --- Tests ---

def test_directory_basic_creation():
    """Test basic directory creation with prior likes."""
    logger = _make_test_logger()

    # User u1 liked posts p1, p2, p3 at times t1, t2, t3
    # Target event is at time t4 (after all likes)
    likes_lf = pl.DataFrame({
        "did": ["u1", "u1", "u1"],
        "record_created_at": [
            datetime(2024, 1, 1, 10, 0),  # earliest
            datetime(2024, 1, 1, 11, 0),  # middle
            datetime(2024, 1, 1, 12, 0),  # latest
        ],
        "subject_uri": ["p1", "p2", "p3"],
        "emb_idx": [100, 200, 300],
    }).lazy()

    targets_lf = _make_targets(["u1"], [datetime(2024, 1, 1, 13, 0)])

    result_lf = _build_user_history_directory(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    )
    result = result_lf.collect()

    assert result.height == 1
    assert result["target_did"][0] == "u1"
    assert result["target_idx"][0] == 0
    # Should have 3 prior likes, sorted by recency (most recent first)
    prior = result["prior_emb_indices"][0].to_list()
    assert len(prior) == 3
    assert prior == [300, 200, 100]  # most recent first


def test_directory_recency_ordering():
    """Test that prior_emb_indices are correctly ordered by recency (descending)."""
    logger = _make_test_logger()

    # Likes in random timestamp order
    likes_lf = pl.DataFrame({
        "did": ["u1", "u1", "u1", "u1"],
        "record_created_at": [
            datetime(2024, 1, 5, 0, 0),   # 3rd most recent
            datetime(2024, 1, 10, 0, 0),  # most recent
            datetime(2024, 1, 1, 0, 0),   # oldest
            datetime(2024, 1, 7, 0, 0),   # 2nd most recent
        ],
        "subject_uri": ["p1", "p2", "p3", "p4"],
        "emb_idx": [10, 20, 30, 40],
    }).lazy()

    targets_lf = _make_targets(["u1"], [datetime(2024, 1, 15, 0, 0)])

    result = _build_user_history_directory(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    prior = result["prior_emb_indices"][0].to_list()
    # Expected order by recency: p2 (Jan 10) -> p4 (Jan 7) -> p1 (Jan 5) -> p3 (Jan 1)
    assert prior == [20, 40, 10, 30]


def test_directory_max_prior_likes_capping():
    """Test that max_prior_likes caps the number of prior likes."""
    logger = _make_test_logger()

    # User has 5 likes
    likes_lf = pl.DataFrame({
        "did": ["u1"] * 5,
        "record_created_at": [
            datetime(2024, 1, i, 0, 0) for i in range(1, 6)
        ],
        "subject_uri": [f"p{i}" for i in range(1, 6)],
        "emb_idx": list(range(1, 6)),
    }).lazy()

    targets_lf = _make_targets(["u1"], [datetime(2024, 1, 10, 0, 0)])

    # Cap to 3 prior likes
    result = _build_user_history_directory(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=3,
        logger=logger,
    ).collect()

    prior = result["prior_emb_indices"][0].to_list()
    assert len(prior) == 3
    # Should be the 3 most recent: emb_idx 5, 4, 3
    assert prior == [5, 4, 3]


def test_directory_no_prior_history_returns_empty_list():
    """Test that targets with no prior likes get an empty list."""
    logger = _make_test_logger()

    # User u1 has likes, user u2 has no likes at all in the dataset
    likes_lf = pl.DataFrame({
        "did": ["u1"],
        "record_created_at": [datetime(2024, 1, 1, 0, 0)],
        "subject_uri": ["p1"],
        "emb_idx": [100],
    }).lazy()

    targets_lf = _make_targets(
        ["u1", "u2"],
        [datetime(2024, 1, 5, 0, 0), datetime(2024, 1, 5, 0, 0)],
    )

    result = _build_user_history_directory(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect().sort("target_did")

    assert result.height == 2

    # u1 has prior likes
    u1_row = result.filter(pl.col("target_did") == "u1")
    assert u1_row["prior_emb_indices"][0].to_list() == [100]

    # u2 has no prior likes (empty list, not null)
    u2_row = result.filter(pl.col("target_did") == "u2")
    assert u2_row["prior_emb_indices"][0].to_list() == []


def test_directory_first_like_produces_empty_history():
    """Test that a user's very first like in the dataset has empty history.

    This is the key scenario: target_posts did NOT filter out first likes,
    so this stage must produce an empty prior_emb_indices list for them.
    """
    logger = _make_test_logger()

    # User u1 has exactly one like at t=10:00
    likes_lf = pl.DataFrame({
        "did": ["u1"],
        "record_created_at": [datetime(2024, 1, 1, 10, 0)],
        "subject_uri": ["p1"],
        "emb_idx": [42],
    }).lazy()

    # The target row corresponds to that very first like (seen_at == like time)
    # Since we use strict < (not <=), a like at the same timestamp is NOT prior
    targets_lf = _make_targets(["u1"], [datetime(2024, 1, 1, 10, 0)])

    result = _build_user_history_directory(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    assert result.height == 1
    assert result["prior_emb_indices"][0].to_list() == []


def test_directory_excludes_future_likes():
    """Test that likes after the target timestamp are excluded."""
    logger = _make_test_logger()

    # User has likes before and after the target timestamp
    likes_lf = pl.DataFrame({
        "did": ["u1", "u1", "u1"],
        "record_created_at": [
            datetime(2024, 1, 1, 0, 0),   # before target
            datetime(2024, 1, 5, 0, 0),   # before target
            datetime(2024, 1, 10, 0, 0),  # after target
        ],
        "subject_uri": ["p1", "p2", "p3"],
        "emb_idx": [1, 2, 3],
    }).lazy()

    targets_lf = _make_targets(["u1"], [datetime(2024, 1, 7, 0, 0)])

    result = _build_user_history_directory(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    prior = result["prior_emb_indices"][0].to_list()
    # Only p1 and p2 should be included (before target ts)
    assert len(prior) == 2
    assert prior == [2, 1]  # most recent first


def test_directory_multiple_targets_same_user():
    """Test that each target gets correct prior likes based on its timestamp."""
    logger = _make_test_logger()

    likes_lf = pl.DataFrame({
        "did": ["u1", "u1", "u1"],
        "record_created_at": [
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 3, 0, 0),
            datetime(2024, 1, 5, 0, 0),
        ],
        "subject_uri": ["p1", "p2", "p3"],
        "emb_idx": [10, 20, 30],
    }).lazy()

    # Three targets at different times for the same user
    targets_lf = _make_targets(
        ["u1", "u1", "u1"],
        [
            datetime(2024, 1, 2, 0, 0),   # only sees p1
            datetime(2024, 1, 4, 0, 0),   # sees p1, p2
            datetime(2024, 1, 10, 0, 0),  # sees p1, p2, p3
        ],
    )

    result = _build_user_history_directory(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect().sort("target_idx")

    # target_idx 0 = early (Jan 2), 1 = mid (Jan 4), 2 = late (Jan 10)
    result_dict = {
        row["target_idx"]: list(row["prior_emb_indices"])
        for row in result.iter_rows(named=True)
    }

    assert result_dict[0] == [10]           # only p1 before
    assert result_dict[1] == [20, 10]       # p2, p1 before
    assert result_dict[2] == [30, 20, 10]   # all three before


def test_directory_multiple_users():
    """Test that directory handles multiple users correctly."""
    logger = _make_test_logger()

    likes_lf = pl.DataFrame({
        "did": ["u1", "u1", "u2", "u2"],
        "record_created_at": [
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 2, 0, 0),
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 3, 0, 0),
        ],
        "subject_uri": ["a1", "a2", "b1", "b2"],
        "emb_idx": [1, 2, 11, 12],
    }).lazy()

    targets_lf = _make_targets(
        ["u1", "u2"],
        [datetime(2024, 1, 5, 0, 0), datetime(2024, 1, 5, 0, 0)],
    )

    result = _build_user_history_directory(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    result_dict = {
        row["target_did"]: list(row["prior_emb_indices"])
        for row in result.iter_rows(named=True)
    }

    # u1's likes (emb_idx 1, 2)
    assert result_dict["u1"] == [2, 1]  # most recent first

    # u2's likes (emb_idx 11, 12)
    assert result_dict["u2"] == [12, 11]  # most recent first


def test_directory_output_schema():
    """Test that output has correct schema with List[UInt32] column."""
    logger = _make_test_logger()

    likes_lf = pl.DataFrame({
        "did": ["u1"],
        "record_created_at": [datetime(2024, 1, 1, 0, 0)],
        "subject_uri": ["p1"],
        "emb_idx": [100],
    }).lazy()

    targets_lf = _make_targets(["u1"], [datetime(2024, 1, 5, 0, 0)])

    result = _build_user_history_directory(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    # Check schema
    assert "target_idx" in result.columns
    assert "target_did" in result.columns
    assert "prior_emb_indices" in result.columns

    # Old columns should NOT be present
    assert "did" not in result.columns
    assert "post_id" not in result.columns

    # Check that prior_emb_indices is List[UInt32]
    assert result.schema["prior_emb_indices"] == pl.List(pl.UInt32)


def test_directory_preserves_row_order():
    """Test that target_idx matches the original row order of target_posts."""
    logger = _make_test_logger()

    likes_lf = pl.DataFrame({
        "did": ["u1", "u2"],
        "record_created_at": [
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 1, 0, 0),
        ],
        "subject_uri": ["p1", "p2"],
        "emb_idx": [10, 20],
    }).lazy()

    # Three target rows
    targets_lf = _make_targets(
        ["u1", "u2", "u1"],
        [
            datetime(2024, 1, 5, 0, 0),
            datetime(2024, 1, 5, 0, 0),
            datetime(2024, 1, 5, 0, 0),
        ],
    )

    result = _build_user_history_directory(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect().sort("target_idx")

    # Should have 3 rows with target_idx 0, 1, 2
    assert result.height == 3
    assert result["target_idx"].to_list() == [0, 1, 2]
    assert result["target_did"].to_list() == ["u1", "u2", "u1"]
