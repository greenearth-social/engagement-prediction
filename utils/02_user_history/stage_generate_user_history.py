#!/usr/bin/env python3

"""
Stage 2: Generate User-Hour History Directory

Creates a directory-style artifact that maps each (user, like-hour bucket) to a
list of prior liked post embedding indices and, when author metadata is
available, author indices. This enables efficient on-the-fly embedding retrieval
during training and stable author-history features.

Inputs:
- likes_core_*.parquet from 01_get_data: Contains
  {did, subject_uri, record_created_at, like_hour_bucket, emb_idx, prior_cumulative_likes, author_idx}
- liked_post_hour_cumulative_likes_*.parquet from 01_get_data: Contains
  {emb_idx, popularity_hour_bucket, prior_cumulative_likes}
- author_idx_*.parquet from 01_get_data, when available: Author index mapping with
  {author_did, author_train_count, author_idx}

Outputs under <run_dir>/02_user_history/<timestamp>/:
- history_posts_<timestamp>.parquet:
  {did, like_hour_bucket, prior_emb_indices, prior_like_age_hours_at_bucket_start, prior_cumulative_likes}
  and, when author_idx is available, {prior_author_indices}
  where prior_emb_indices is a List[UInt32] of embedding indices sorted by recency (most recent first),
  prior_like_age_hours_at_bucket_start is a List[Float32] aligned element-wise with
  prior_emb_indices and measured from the target like_hour_bucket,
  prior_cumulative_likes is a List[UInt64] aligned element-wise with
  prior_emb_indices and measured as of the target like_hour_bucket,
  prior_author_indices is a List[UInt32] aligned element-wise with
  prior_emb_indices, and user-hour rows where the user has no prior likes in
  the dataset get empty lists.
- user_idx_<timestamp>.parquet and post_recent_likers_<timestamp>.parquet when
  post-liker history generation is enabled.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import json
import logging
import polars as pl
import time

from utils.pipeline.core import Context
from utils.helpers import (
    get_stage_logger,
    log_operation_start,
    validate_dataframe_schema,
    load_parquet_from_prior,
    TIMESTAMP_COL_NAME,
    HISTORY_POPULARITY_SEMANTICS,
)
from utils.memory_helpers import MemoryTracker


def _build_user_history_directory(
    likes_lf: pl.LazyFrame,
    liked_post_hour_cumulative_likes_lf: pl.LazyFrame,
    max_prior_likes: Optional[int],
    logger: logging.Logger,
) -> pl.LazyFrame:
    """
    Build a directory mapping each (user, like-hour bucket) to prior liked embedding indices.

    Uses vectorized Polars operations for efficiency:
    1. Get distinct user-hour pairs from likes_core
    2. Join each pair to that user's likes
    3. Filter to likes that occurred before the hour bucket
    4. Group by (did, like_hour_bucket) and collect emb_idx values sorted by recency
    5. Join capped history posts to sparse target-hour popularity counts
    6. Left-join back to ensure every user-hour appears, including empty histories

    Args:
        likes_lf: LazyFrame with columns [did, like_hour_bucket, record_created_at, emb_idx]
        liked_post_hour_cumulative_likes_lf: LazyFrame with columns
            [emb_idx, popularity_hour_bucket, prior_cumulative_likes]
        max_prior_likes: Optional cap on prior likes per target (None = no cap)
        logger: Logger instance

    Returns:
        LazyFrame with columns [did, like_hour_bucket, prior_emb_indices, raw_prior_count,
        prior_like_age_hours_at_bucket_start, prior_cumulative_likes]
        where raw_prior_count is the uncapped number of prior likes (for distribution analysis).
    """
    logger.info("Building user history directory...")
    if max_prior_likes is None:
        logger.warning(
            "STRONG WARNING: max_prior_likes is not finite; target-time history popularity "
            "lookups will be uncapped and may use high memory. Set max_prior_likes to bound "
            "the Stage 2 rank/join/group step."
        )
    likes_schema = likes_lf.collect_schema()
    include_author_idx = "author_idx" in likes_schema
    popularity_schema = liked_post_hour_cumulative_likes_lf.collect_schema()
    missing_popularity_cols = [
        col_name
        for col_name in ("emb_idx", "popularity_hour_bucket", "prior_cumulative_likes")
        if col_name not in popularity_schema
    ]
    if missing_popularity_cols:
        raise ValueError(
            "liked_post_hour_cumulative_likes must contain "
            f"{', '.join(missing_popularity_cols)}"
        )

    user_bucket_pairs_lf = (
        likes_lf
        .select(['did', 'like_hour_bucket'])
        .unique()
    ) # [did, like_hour_bucket]

    # Join targets with likes on user identity
    # This creates one row per (target, like) pair for each user
    likes_cols = ['did', TIMESTAMP_COL_NAME, 'emb_idx']
    if include_author_idx:
        likes_cols.append('author_idx')
    pairs_with_prior_likes_lf = (
        user_bucket_pairs_lf
        .join(
            likes_lf.select(likes_cols),
            on="did",
            how="left"
        )
        .filter(pl.col(TIMESTAMP_COL_NAME) < pl.col("like_hour_bucket"))
        .with_columns(
            (
                (pl.col("like_hour_bucket") - pl.col(TIMESTAMP_COL_NAME))
                .dt.total_seconds()
                / 3600.0
            )
            .cast(pl.Float32)
            .alias("prior_like_age_hours_at_bucket_start"),
        )
        .select([
            "did",
            "like_hour_bucket",
            "emb_idx",
            "prior_like_age_hours_at_bucket_start",
            *(["author_idx"] if include_author_idx else []),
        ])
    ) # [did, like_hour_bucket, emb_idx, prior_like_age_hours_at_bucket_start, (author_idx)]

    ranked_prior_likes_lf = (
        pairs_with_prior_likes_lf
        .with_columns(
            pl.len().over(["did", "like_hour_bucket"]).alias("raw_prior_count"),
            pl.col("prior_like_age_hours_at_bucket_start")
            .rank(method="ordinal")
            .over(["did", "like_hour_bucket"])
            .alias("_prior_rank"),
            pl.col("emb_idx").cast(pl.UInt32).alias("emb_idx"),
        )
    ) # [did, like_hour_bucket, emb_idx, prior_like_age_hours_at_bucket_start, (author_idx), raw_prior_count, _prior_rank]
    if max_prior_likes is not None and max_prior_likes > 0:
        ranked_prior_likes_lf = ranked_prior_likes_lf.filter(pl.col("_prior_rank") <= max_prior_likes)

    logger.info("Adding target-hour popularity counts to capped history rows...")
    popularity_lf = (
        liked_post_hour_cumulative_likes_lf
        .select(["emb_idx", "popularity_hour_bucket", "prior_cumulative_likes"])
        .with_columns(
            pl.col("emb_idx").cast(pl.UInt32),
            pl.col("prior_cumulative_likes").fill_null(0).cast(pl.UInt64)
        )
    )
    history_with_popularity_lf = (
        ranked_prior_likes_lf
        .sort(["like_hour_bucket", "emb_idx"])
        .join_asof(
            popularity_lf.sort(["popularity_hour_bucket", "emb_idx"]),
            left_on="like_hour_bucket",
            right_on="popularity_hour_bucket",
            by="emb_idx",
            strategy="backward",
            check_sortedness=False,
        )
        .with_columns(
            pl.col("prior_cumulative_likes").fill_null(0).cast(pl.UInt64)
        )
    )
    agg_exprs = [
        pl.col("emb_idx").sort_by(pl.col("_prior_rank")).alias("prior_emb_indices"),
        pl.col("raw_prior_count").first().alias("raw_prior_count"),
        pl.col("prior_like_age_hours_at_bucket_start").sort_by(pl.col("_prior_rank")).alias("prior_like_age_hours_at_bucket_start"),
        pl.col("prior_cumulative_likes").sort_by(pl.col("_prior_rank")).alias("prior_cumulative_likes"),
    ]
    if include_author_idx:
        agg_exprs += [pl.col("author_idx").sort_by(pl.col("_prior_rank")).alias("prior_author_indices")]
    history_lists_with_counts_lf = (
        history_with_popularity_lf
        .group_by(["did", "like_hour_bucket"])
        .agg(agg_exprs)
    )
    pairs_with_history_list_lf = (
        user_bucket_pairs_lf
        .join(history_lists_with_counts_lf, on=["did", "like_hour_bucket"], how="left")
        .with_columns(
            pl.when(pl.col("prior_emb_indices").is_null())
            .then(pl.lit([]).cast(pl.List(pl.UInt32)))
            .otherwise(pl.col("prior_emb_indices").cast(pl.List(pl.UInt32)))
            .alias("prior_emb_indices"),
            pl.col("raw_prior_count").fill_null(0),
            pl.when(pl.col("prior_like_age_hours_at_bucket_start").is_null())
            .then(pl.lit([]).cast(pl.List(pl.Float32)))
            .otherwise(pl.col("prior_like_age_hours_at_bucket_start").cast(pl.List(pl.Float32)))
            .alias("prior_like_age_hours_at_bucket_start"),
            pl.when(pl.col("prior_cumulative_likes").is_null())
            .then(pl.lit([]).cast(pl.List(pl.UInt64)))
            .otherwise(pl.col("prior_cumulative_likes").cast(pl.List(pl.UInt64)))
            .alias("prior_cumulative_likes"),
        )
    ) # [did, like_hour_bucket, prior_emb_indices, raw_prior_count, prior_like_age_hours_at_bucket_start, prior_cumulative_likes]
    
    if include_author_idx:
        pairs_with_history_list_lf = (
            pairs_with_history_list_lf
            .with_columns(
                pl.when(pl.col("prior_author_indices").is_null())
                .then(pl.lit([]).cast(pl.List(pl.UInt32)))
                .otherwise(pl.col("prior_author_indices").cast(pl.List(pl.UInt32)))
                .alias("prior_author_indices"),
            )
        ) # [did, like_hour_bucket, prior_emb_indices, raw_prior_count, prior_like_age_hours_at_bucket_start, prior_cumulative_likes, (prior_author_indices)]

    return pairs_with_history_list_lf


def _build_post_liker_user_idx(
    likes_lf: pl.LazyFrame,
    min_user_support: int,
) -> pl.LazyFrame:
    if min_user_support <= 0:
        raise ValueError("min_user_support must be positive")
    return (
        likes_lf
        .select(["did", "split"])
        .filter(pl.col("split") == "train")
        .group_by("did")
        .len()
        .rename({"len": "user_train_like_count"})
        .filter(pl.col("user_train_like_count") >= min_user_support)
        .sort("did")
        .with_row_index(name="user_idx", offset=2)
        .with_columns(
            pl.col("user_train_like_count").cast(pl.UInt64),
            pl.col("user_idx").cast(pl.UInt32),
        )
        .select(["did", "user_train_like_count", "user_idx"])
    )


def _build_needed_post_hour_pairs(
    likes_lf: pl.LazyFrame,
    posts_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    positive_pairs_lf = (
        likes_lf
        .select(
            pl.col("subject_uri"),
            pl.col("like_hour_bucket").alias("target_hour"),
        )
        .unique()
    )
    negative_pairs_lf = (
        posts_lf
        .filter(pl.col("in_random_sample") & pl.col("negative_hour_bucket").is_not_null())
        .select(
            pl.col("at_uri").alias("subject_uri"),
            pl.col("negative_hour_bucket").alias("target_hour"),
        )
        .unique()
    )
    return (
        pl.concat([positive_pairs_lf, negative_pairs_lf], how="vertical_relaxed")
        .unique()
        .sort(["subject_uri", "target_hour"])
    )


def _source_likes_for_needed_posts(
    likes_lf: pl.LazyFrame,
    needed_pairs_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    needed_posts_lf = needed_pairs_lf.select("subject_uri").unique()
    return (
        likes_lf
        .select(["subject_uri", "did", TIMESTAMP_COL_NAME])
        .join(needed_posts_lf, on="subject_uri", how="semi")
    )


def _build_prior_liker_count_lf(
    needed_pairs_lf: pl.LazyFrame,
    source_likes_lf: pl.LazyFrame,
    count_col: str,
) -> pl.LazyFrame:
    hourly_counts_lf = (
        source_likes_lf
        .with_columns(
            pl.col(TIMESTAMP_COL_NAME).dt.truncate("1h").alias("_like_hour")
        )
        .group_by(["subject_uri", "_like_hour"])
        .len()
        .rename({"len": "_likes_this_hour"})
        .with_columns(pl.col("_likes_this_hour").cast(pl.UInt64))
        # subject_uri, _like_hour, _likes_this_hour
    )
    sparse_counts_lf = (
        hourly_counts_lf
        .sort(["subject_uri", "_like_hour"])
        .with_columns(
            pl.col("_likes_this_hour").cum_sum().over("subject_uri").alias(count_col)
        )
        .select(
            "subject_uri",
            (pl.col("_like_hour") + pl.duration(hours=1)).alias("_available_at_hour"),
            pl.col(count_col).cast(pl.UInt64),
        )
        # subject_uri, _available_at_hour, dataset_prior_liker_count
    )
    return (
        needed_pairs_lf
        .sort(["target_hour", "subject_uri"])
        .join_asof(
            sparse_counts_lf.sort(["_available_at_hour", "subject_uri"]),
            left_on="target_hour",
            right_on="_available_at_hour",
            by="subject_uri",
            strategy="backward",
            check_sortedness=False,
        )
        .drop("_available_at_hour")
        .with_columns(pl.col(count_col).fill_null(0).cast(pl.UInt64))
    )


def _empty_timestamp_list_expr(likes_lf: pl.LazyFrame) -> pl.Expr:
    timestamp_dtype = likes_lf.collect_schema()[TIMESTAMP_COL_NAME]
    return pl.lit([]).cast(pl.List(timestamp_dtype))


def _build_post_recent_likers(
    likes_lf: pl.LazyFrame,
    posts_lf: pl.LazyFrame,
    user_idx_lf: pl.LazyFrame,
    max_recent_likers_per_post: int,
) -> pl.LazyFrame:
    if max_recent_likers_per_post <= 0:
        raise ValueError("max_recent_likers_per_post must be positive")

    needed_pairs_lf = _build_needed_post_hour_pairs(likes_lf, posts_lf)
    source_likes_lf = _source_likes_for_needed_posts(likes_lf, needed_pairs_lf)
    dataset_counts_lf = _build_prior_liker_count_lf(
        needed_pairs_lf,
        source_likes_lf,
        "dataset_prior_liker_count",
    )
    indexed_likes_lf = (
        source_likes_lf
        .join(user_idx_lf.select(["did", "user_idx"]), on="did", how="inner")
        .select(["subject_uri", TIMESTAMP_COL_NAME, "user_idx"])
        .with_columns(pl.col("user_idx").cast(pl.UInt32))
    )
    indexed_counts_lf = _build_prior_liker_count_lf(
        needed_pairs_lf,
        indexed_likes_lf,
        "indexed_prior_liker_count",
    )
    pairs_with_counts_lf = (
        dataset_counts_lf
        .join(indexed_counts_lf, on=["subject_uri", "target_hour"], how="left")
        .with_columns(
            pl.col("indexed_prior_liker_count").fill_null(0).cast(pl.UInt64)
        )
    )

    indexed_event_lists_lf = (
        indexed_likes_lf
        .sort(["subject_uri", TIMESTAMP_COL_NAME, "user_idx"])
        .group_by("subject_uri")
        .agg(
            pl.col("user_idx").alias("_all_liker_user_indices"),
            pl.col(TIMESTAMP_COL_NAME).alias("_all_liker_timestamps"),
        )
    )
    pairs_with_slices_lf = (
        pairs_with_counts_lf
        .with_columns(
            pl.when(pl.col("indexed_prior_liker_count") > max_recent_likers_per_post)
            .then(pl.col("indexed_prior_liker_count") - max_recent_likers_per_post)
            .otherwise(0)
            .cast(pl.Int64)
            .alias("_slice_start")
        )
        .with_columns(
            (pl.col("indexed_prior_liker_count").cast(pl.Int64) - pl.col("_slice_start"))
            .cast(pl.Int64)
            .alias("_slice_len")
        )
    )
    return (
        pairs_with_slices_lf
        .join(indexed_event_lists_lf, on="subject_uri", how="left")
        .with_columns(
            pl.when(pl.col("_all_liker_user_indices").is_null())
            .then(pl.lit([]).cast(pl.List(pl.UInt32)))
            .otherwise(
                pl.col("_all_liker_user_indices")
                .list.slice(pl.col("_slice_start"), pl.col("_slice_len"))
                .list.reverse()
            )
            .alias("prior_recent_liker_user_indices"),
            pl.when(pl.col("_all_liker_timestamps").is_null())
            .then(_empty_timestamp_list_expr(likes_lf))
            .otherwise(
                pl.col("_all_liker_timestamps")
                .list.slice(pl.col("_slice_start"), pl.col("_slice_len"))
                .list.reverse()
            )
            .alias("prior_recent_liker_timestamps"),
        )
        .select([
            "subject_uri",
            "target_hour",
            "prior_recent_liker_user_indices",
            "prior_recent_liker_timestamps",
            "dataset_prior_liker_count",
            "indexed_prior_liker_count",
        ])
        .sort(["subject_uri", "target_hour"])
    )


def _log_and_plot_history_distribution(
    directory_df: pl.DataFrame,
    max_prior_likes: Optional[int],
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Log summary statistics and save a histogram of the per-user history length
    distribution (measured at each user's last target post) before and after
    the max_prior_likes cap.

    For each user we look at their chronologically last target post, which has
    the maximum available history.  The distribution of these counts across
    users reveals how many users are actually affected by the cap.
    """
    # For each user, find the raw prior count at their last target post
    last_target_per_user = directory_df.group_by("did").agg(
        pl.col("raw_prior_count").sort_by("like_hour_bucket").last().alias("history_len_before"),
    )

    before = last_target_per_user["history_len_before"]
    n_users = len(before)

    if n_users == 0:
        logger.warning("No users found for history distribution analysis")
        return

    if max_prior_likes is not None:
        last_target_per_user = last_target_per_user.with_columns(
            pl.col("history_len_before").clip(upper_bound=max_prior_likes).alias("history_len_after")
        )
        after = last_target_per_user["history_len_after"]
    else:
        after = before

    # --- Log summary statistics ---
    logger.info("=" * 60)
    logger.info("Per-user history distribution (at each user's last target post)")
    logger.info(f"  Number of unique users: {n_users:,}")

    for label, dist in [("Before capping", before), ("After capping", after)]:
        logger.info(f"  {label}:")
        logger.info(f"    mean={dist.mean():.1f}, median={dist.median():.1f}")
        logger.info(
            f"    p25={int(dist.quantile(0.25, 'nearest') or 0)}, "
            f"p75={int(dist.quantile(0.75, 'nearest') or 0)}, "
            f"p90={int(dist.quantile(0.90, 'nearest') or 0)}, "
            f"p95={int(dist.quantile(0.95, 'nearest') or 0)}, "
            f"p99={int(dist.quantile(0.99, 'nearest') or 0)}"
        )
        logger.info(f"    min={dist.min()}, max={dist.max()}")

    if max_prior_likes is not None:
        n_capped = int((before > max_prior_likes).sum())
        pct_capped = 100.0 * n_capped / n_users
        logger.info(
            f"  Users affected by cap ({max_prior_likes}): "
            f"{n_capped:,} ({pct_capped:.1f}%)"
        )
        total_before = int(before.sum())
        total_after = int(after.sum())
        dropped = total_before - total_after
        pct_dropped = 100.0 * dropped / max(total_before, 1)
        logger.info(
            f"  Total prior likes (last target): before={total_before:,}, "
            f"after={total_after:,}, dropped={dropped:,} ({pct_dropped:.1f}%)"
        )

    logger.info("=" * 60)

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available, skipping distribution plot")
        return

    before_np = before.to_numpy().astype(float)
    after_np = after.to_numpy().astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: before capping
    ax = axes[0]
    max_val = int(before_np.max()) if len(before_np) > 0 else 1
    n_bins = min(100, max(max_val + 1, 2))
    bins = np.linspace(-0.5, max_val + 0.5, n_bins)
    ax.hist(before_np, bins=bins, alpha=0.8, color="steelblue",
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Prior likes count")
    ax.set_ylabel("Number of users")
    ax.set_title("Before capping")
    if before_np.max() > 0:
        ax.set_yscale("log")
    if max_prior_likes is not None:
        ax.axvline(max_prior_likes, color="red", linestyle="--",
                    linewidth=1.5, label=f"cap = {max_prior_likes}")
        ax.legend()

    # Right panel: after capping
    ax = axes[1]
    max_val_after = int(after_np.max()) if len(after_np) > 0 else 1
    n_bins_after = min(100, max(max_val_after + 1, 2))
    bins_after = np.linspace(-0.5, max_val_after + 0.5, n_bins_after)
    ax.hist(after_np, bins=bins_after, alpha=0.8, color="darkorange",
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Prior likes count")
    ax.set_ylabel("Number of users")
    cap_label = (f" (max_prior_likes={max_prior_likes})"
                 if max_prior_likes is not None else " (no cap)")
    ax.set_title(f"After capping{cap_label}")
    if after_np.max() > 0:
        ax.set_yscale("log")

    fig.suptitle(
        "Distribution of history length per user (last target post)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    plot_path = out_dir / "history_distribution.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✓ Saved history distribution plot to {plot_path.name}")


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Stage 2: Generate user history directory.

    Creates a parquet file mapping each target row to a list of prior liked
    post embedding indices for efficient on-the-fly lookup during training.
    """
    out_dir = context.new_stage_dir('02_user_history')

    # Initialize logger and memory tracker
    logger = get_stage_logger('STAGE_02_USER_HISTORY', log_file=out_dir / 'stage.log')
    t0 = time.time()
    mem_tracker = MemoryTracker(logger=logger)
    mem_tracker.checkpoint("stage_start")

    # === Locate prior stage outputs ===

    # 1. Likes from get_data stage
    prior_get_data = context.resolve_prior_output(
        '01_get_data',
        prior_path=context.prior_outputs.get('01_get_data'),
    )

    # === Get CLI args ===
    max_prior_likes: Optional[int] = args.max_prior_likes
    if max_prior_likes is not None and max_prior_likes <= 0:
        max_prior_likes = None  # Treat 0 or negative as "no cap"
    generate_post_liker_history = bool(args.generate_post_liker_history)
    min_post_liker_user_support = int(args.min_post_liker_user_support)
    max_recent_likers_per_post = int(args.max_recent_likers_per_post)
    if min_post_liker_user_support <= 0:
        raise ValueError("min_post_liker_user_support must be positive")
    if max_recent_likers_per_post <= 0:
        raise ValueError("max_recent_likers_per_post must be positive")

    # === Load data ===
    log_operation_start('Load likes_core from prior stage', 'STAGE_02_USER_HISTORY', logger)
    likes_lf: pl.LazyFrame = load_parquet_from_prior(prior_get_data, "likes_core_")
    log_operation_start('Load liked-post popularity curve from prior stage', 'STAGE_02_USER_HISTORY', logger)
    try:
        liked_post_hour_cumulative_likes_lf: pl.LazyFrame = load_parquet_from_prior(
            prior_get_data,
            "liked_post_hour_cumulative_likes_",
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Stage 2 requires liked_post_hour_cumulative_likes_*.parquet from Stage 1 "
            "for target-hour history popularity counts. Rerun Stage 1 before generating "
            "user history."
        ) from exc

    # Validate likes schema
    likes_schema = {
        "did": str,
        TIMESTAMP_COL_NAME: pl.Datetime,
        "subject_uri": str,
        "emb_idx": int,
        "prior_cumulative_likes": int,
    }
    if generate_post_liker_history:
        likes_schema.update({
            "split": str,
            "like_hour_bucket": pl.Datetime,
        })
    validate_dataframe_schema(likes_lf, likes_schema)
    logger.info("✓ likes_core schema validated")
    liked_post_popularity_schema = {
        "emb_idx": int,
        "popularity_hour_bucket": pl.Datetime,
        "prior_cumulative_likes": int,
    }
    validate_dataframe_schema(liked_post_hour_cumulative_likes_lf, liked_post_popularity_schema)
    logger.info("✓ liked_post_hour_cumulative_likes schema validated")

    posts_lf: Optional[pl.LazyFrame] = None
    if generate_post_liker_history:
        log_operation_start('Load posts_core from prior stage', 'STAGE_02_USER_HISTORY', logger)
        posts_lf = load_parquet_from_prior(prior_get_data, "posts_core_")
        validate_dataframe_schema(
            posts_lf,
            {
                "at_uri": str,
                "in_random_sample": bool,
                "negative_hour_bucket": pl.Datetime,
            },
        )
        logger.info("✓ posts_core schema validated")

    mem_tracker.checkpoint("after_load_inputs", quiet=True)

    # Log input sizes (collect counts efficiently)
    n_likes = likes_lf.select(pl.len()).collect(engine="streaming").item()
    n_liked_post_popularity_rows = liked_post_hour_cumulative_likes_lf.select(pl.len()).collect().item()
    logger.info(f"Input: {n_likes:,} likes")
    logger.info(f"Input: {n_liked_post_popularity_rows:,} liked-post popularity rows")
    n_posts = None
    if posts_lf is not None:
        n_posts = posts_lf.select(pl.len()).collect(engine="streaming").item()
        logger.info(f"Input: {n_posts:,} post rows")

    # === Build user history directory ===
    log_operation_start('Build user history directory', 'STAGE_02_USER_HISTORY', logger)

    mem_tracker.checkpoint("before_history_rank_join_group", quiet=True)
    directory_lf = _build_user_history_directory(
        likes_lf=likes_lf,
        liked_post_hour_cumulative_likes_lf=liked_post_hour_cumulative_likes_lf,
        max_prior_likes=max_prior_likes,
        logger=logger,
    )

    mem_tracker.checkpoint("after_build_history", quiet=True)

    # === Write output ===
    log_operation_start('Write user history directory', 'STAGE_02_USER_HISTORY', logger)

    # Collect using the streaming engine so that the intermediate fan-out join
    # is processed in batches rather than fully materialised in memory.
    # Falls back to the default engine automatically if the plan can't be streamed.
    directory_df = directory_lf.collect(engine="streaming")
    mem_tracker.checkpoint("after_history_rank_join_group", quiet=True)

    # Log and plot the per-user history distribution before/after capping
    _log_and_plot_history_distribution(directory_df, max_prior_likes, out_dir, logger)

    # Select only the required persisted output columns. Author history is
    # optional so user-history outputs without author columns can still feed
    # training when the author embedding feature is not being used.
    user_history_output_path = out_dir / f"history_posts_{out_dir.name}.parquet"
    directory_df.write_parquet(user_history_output_path, compression="zstd")

    n_output = len(directory_df)

    n_with_history = directory_df.filter(pl.col("prior_emb_indices").list.len() > 0).height
    n_empty_history = n_output - n_with_history

    # Stats on prior likes counts
    prior_count_stats = (
        directory_df
        .select(
            pl.col("prior_emb_indices").list.len().mean().fill_null(0.0).alias("mean_prior"),
            pl.col("prior_emb_indices").list.len().max().fill_null(0).alias("max_prior"),
            pl.when(pl.col("prior_emb_indices").list.len() > 0)
            .then(pl.col("prior_emb_indices").list.len())
            .otherwise(None)
            .min()
            .fill_null(0)
            .alias("min_prior"),
        )
        .row(0)
    )
    mean_prior_value = float(prior_count_stats[0])
    max_prior_value = int(prior_count_stats[1])
    min_prior_value = int(prior_count_stats[2])

    summary = {
        "history_prior_cumulative_likes_semantics": HISTORY_POPULARITY_SEMANTICS,
        "history_prior_cumulative_likes_description": "count as of the target like_hour_bucket",
        "parameters": {
            "max_prior_likes": max_prior_likes,
        },
        "inputs": {
            "likes_core_rows": n_likes,
            "liked_post_hour_cumulative_likes_rows": n_liked_post_popularity_rows,
        },
        "outputs": {
            "user_history_directory_file": user_history_output_path.name,
            "user_history_directory_rows": n_output,
        },
        "stats": {
            "with_history": n_with_history,
            "empty_history": n_empty_history,
            "mean_prior": mean_prior_value,
            "max_prior": max_prior_value,
            "min_prior": min_prior_value,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    mem_tracker.checkpoint("after_write_output", quiet=True)

    logger.info(f"✓ Wrote {n_output:,} directory entries to {user_history_output_path.name}")
    logger.info(f"  With history: {n_with_history:,} ({100*n_with_history/n_output:.1f}%)")
    logger.info(f"  Empty history: {n_empty_history:,} ({100*n_empty_history/n_output:.1f}%)")
    logger.info(f"  Prior likes per target: mean={mean_prior_value:.1f}, min={min_prior_value}, max={max_prior_value}")

    artifacts = {
        'user_history_directory_path': str(user_history_output_path),
    }
    post_liker_info_lines = []
    if generate_post_liker_history:
        if posts_lf is None:
            raise RuntimeError("posts_core must be loaded when post-liker history generation is enabled")

        log_operation_start('Build post-liker user index', 'STAGE_02_USER_HISTORY', logger)
        user_idx_lf = _build_post_liker_user_idx(
            likes_lf=likes_lf,
            min_user_support=min_post_liker_user_support,
        )
        user_idx_df = user_idx_lf.collect(engine="streaming")
        user_idx_output_path = out_dir / f"user_idx_{out_dir.name}.parquet"
        user_idx_df.write_parquet(user_idx_output_path, compression="zstd")
        logger.info(f"✓ Wrote {len(user_idx_df):,} post-liker user index rows to {user_idx_output_path.name}")
        mem_tracker.checkpoint("after_post_liker_user_idx", quiet=True)

        log_operation_start('Build post recent-liker history', 'STAGE_02_USER_HISTORY', logger)
        post_recent_likers_lf = _build_post_recent_likers(
            likes_lf=likes_lf,
            posts_lf=posts_lf,
            user_idx_lf=user_idx_df.lazy(),
            max_recent_likers_per_post=max_recent_likers_per_post,
        )
        post_recent_likers_df = post_recent_likers_lf.collect(engine="streaming")
        post_recent_likers_output_path = out_dir / f"post_recent_likers_{out_dir.name}.parquet"
        post_recent_likers_df.write_parquet(post_recent_likers_output_path, compression="zstd")
        n_post_liker_rows = len(post_recent_likers_df)
        n_with_recent_likers = post_recent_likers_df.filter(
            pl.col("prior_recent_liker_user_indices").list.len() > 0
        ).height
        max_stored_likers = (
            int(post_recent_likers_df["prior_recent_liker_user_indices"].list.len().max())
            if n_post_liker_rows > 0 else 0
        )
        logger.info(f"✓ Wrote {n_post_liker_rows:,} post-liker rows to {post_recent_likers_output_path.name}")
        logger.info(
            f"  With stored indexed likers: {n_with_recent_likers:,} "
            f"({100*n_with_recent_likers/max(n_post_liker_rows, 1):.1f}%)"
        )
        logger.info(f"  Max stored recent likers per post-hour: {max_stored_likers:,}")
        mem_tracker.checkpoint("after_post_recent_likers", quiet=True)

        artifacts.update({
            'post_liker_user_idx_path': str(user_idx_output_path),
            'post_recent_likers_path': str(post_recent_likers_output_path),
        })
        post_liker_info_lines = [
            f"settings: generate_post_liker_history={generate_post_liker_history}",
            f"settings: min_post_liker_user_support={min_post_liker_user_support}",
            f"settings: max_recent_likers_per_post={max_recent_likers_per_post}",
            f"outputs: post_liker_user_idx ({len(user_idx_df):,} rows)",
            f"outputs: post_recent_likers ({n_post_liker_rows:,} rows)",
            f"stats: post_liker_needed_pairs={n_post_liker_rows:,}",
            f"stats: post_liker_rows_with_stored_likers={n_with_recent_likers:,}",
            f"stats: post_liker_max_stored_likers={max_stored_likers}",
        ]

    # Memory summary
    mem_tracker.summary()

    runtime = time.time() - t0

    # === Stage info ===
    info_lines = [
        f"stage: user_history",
        f"runtime_seconds: {runtime:.2f}",
        f"settings: max_prior_likes={max_prior_likes}",
        f"inputs: likes_core ({n_likes:,})",
        f"inputs: liked_post_hour_cumulative_likes ({n_liked_post_popularity_rows:,})",
        *( [f"inputs: posts_core ({n_posts:,})"] if n_posts is not None else [] ),
        f"outputs: user_history_directory ({n_output:,} entries)",
        f"history_prior_cumulative_likes_semantics: {HISTORY_POPULARITY_SEMANTICS}",
        f"stats: with_history={n_with_history:,}, empty_history={n_empty_history:,}",
        f"stats: mean_prior={mean_prior_value:.1f}, max_prior={max_prior_value}",
        *post_liker_info_lines,
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    logger.info(f"Stage 2 (user_history) completed in {runtime:.2f}s")

    return {
        'output_dir': out_dir,
        'artifacts': artifacts,
    }
