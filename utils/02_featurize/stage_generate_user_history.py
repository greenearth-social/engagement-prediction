#!/usr/bin/env python3

"""
Stage 2: Generate User History Directory

Creates a directory-style artifact that maps each target row to a list of prior
liked post embedding indices, enabling efficient on-the-fly embedding retrieval
during training.

Inputs:
- likes_core_*.parquet from 01_get_data: Contains {did, subject_uri, record_created_at, emb_idx}
- target_posts_*.parquet from 02_target_posts: Wide format with
  {target_did, seen_at, like_uri, like_emb_idx, ..., neg_uri, neg_emb_idx, ..., split}

Outputs under <run_dir>/02_featurize/<timestamp>/:
- user_history_directory_<timestamp>.parquet: {target_idx, target_did, prior_emb_indices}
  where prior_emb_indices is a List[UInt32] of embedding indices sorted by recency (most recent first)
  and target_idx corresponds to the row index in the target_posts file.
  Rows where the user has no prior likes in the dataset get an empty list.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
import logging
import polars as pl
import time

from utils.pipeline.core import new_stage_timestamp_dir, select_prior_output, Context
from utils.helpers import get_stage_logger, log_operation_start, validate_dataframe_schema, load_parquet_from_prior, TIMESTAMP_COL_NAME


def _build_user_history_directory(
    targets_lf: pl.LazyFrame,
    likes_lf: pl.LazyFrame,
    max_prior_likes: Optional[int],
    logger: logging.Logger,
) -> pl.LazyFrame:
    """
    Build a directory mapping each target row to prior liked embedding indices.

    The target posts use a wide format where each row represents a (user, like-event)
    training pair.  The user history depends only on the user (target_did) and the
    event timestamp (seen_at), so a single history list is produced per target row.

    Uses vectorized Polars operations for efficiency:
    1. Assign a row index (target_idx) to each target row
    2. Join targets with likes on user (target_did == did)
    3. Filter to likes that occurred before the target timestamp (seen_at)
    4. Group by target_idx and collect emb_idx values sorted by recency
    5. Left-join back to ensure every target row appears (empty list for no history)

    Args:
        targets_lf: LazyFrame with at least columns [target_did, seen_at, ...]
        likes_lf: LazyFrame with columns [did, subject_uri, record_created_at, emb_idx]
        max_prior_likes: Optional cap on prior likes per target (None = no cap)
        logger: Logger instance

    Returns:
        LazyFrame with columns [target_idx, target_did, prior_emb_indices]
    """
    logger.info("Building user history directory...")

    # Add a unique row index so we can key each target row unambiguously
    targets_indexed = targets_lf.with_row_index("target_idx")

    # Select only the columns we need from targets for the join
    target_keys = targets_indexed.select(["target_idx", "target_did", "seen_at"])

    # Rename likes columns to avoid collision after join
    # We need: did (join key), record_created_at (for filtering/sorting), emb_idx (the result)
    likes_renamed = likes_lf.select([
        pl.col("did"),
        pl.col(TIMESTAMP_COL_NAME).alias("like_ts"),
        pl.col("emb_idx").alias("like_emb_idx"),
    ])

    # Join targets with likes on user identity
    # This creates one row per (target, like) pair for each user
    joined = target_keys.join(
        likes_renamed,
        left_on="target_did",
        right_on="did",
        how="left",
    )

    # Filter to likes that occurred BEFORE the target timestamp
    # This ensures we only include prior history, not future likes
    prior_likes = joined.filter(
        pl.col("like_ts") < pl.col("seen_at")
    )

    # Build aggregation expression: sort by recency (descending) and optionally cap
    # The result is a list of emb_idx values, most recent first
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
    # Targets without prior likes (e.g. a user's first like in the dataset)
    # will have null prior_emb_indices, which we convert to an empty list.
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


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Stage 2: Generate user history directory.

    Creates a parquet file mapping each target row to a list of prior liked
    post embedding indices for efficient on-the-fly lookup during training.
    """
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '02_featurize')

    # Initialize logger
    logger = get_stage_logger('STAGE_02_FEATURIZE', log_file=out_dir / 'stage.log')
    t0 = time.time()

    # === Locate prior stage outputs ===

    # 1. Likes from get_data stage
    prior_get_data = select_prior_output(
        run_dir, '01_get_data',
        use_latest=context.use_latest,
        prior_path=context.prior_outputs.get('01_get_data'),
    )
    if prior_get_data is None:
        raise FileNotFoundError("Could not find 01_get_data output")

    # 2. Target posts from 02_target_posts stage
    prior_target_posts_dir = select_prior_output(
        run_dir, '02_target_posts',
        use_latest=context.use_latest,
        prior_path=context.prior_outputs.get('02_target_posts'),
    )
    if prior_target_posts_dir is None:
        raise FileNotFoundError(
            "Could not find 02_target_posts output. "
            "Run the target_posts stage first or provide --prior-output-target-posts."
        )
    # Find the parquet file inside the target posts output directory
    target_posts_candidates = sorted(
        prior_target_posts_dir.glob("target_posts_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not target_posts_candidates:
        raise FileNotFoundError(
            f"No target_posts_*.parquet found under {prior_target_posts_dir}"
        )
    target_posts_path = target_posts_candidates[0]

    logger.info(f"Using likes from: {prior_get_data}")
    logger.info(f"Using target posts from: {target_posts_path}")

    # === Get CLI args ===
    max_prior_likes: Optional[int] = getattr(args, 'max_prior_likes', None)
    if max_prior_likes is not None and max_prior_likes <= 0:
        max_prior_likes = None  # Treat 0 or negative as "no cap"

    # === Load data ===
    log_operation_start('Load likes_core from prior stage', 'STAGE_02_FEATURIZE', logger)
    likes_lf: pl.LazyFrame = load_parquet_from_prior(prior_get_data, "likes_core_")

    # Validate likes schema
    likes_schema = {
        "did": str,
        TIMESTAMP_COL_NAME: pl.Datetime,
        "subject_uri": str,
        "emb_idx": int,
    }
    validate_dataframe_schema(likes_lf, likes_schema)
    logger.info("✓ likes_core schema validated")

    log_operation_start('Load target_posts', 'STAGE_02_FEATURIZE', logger)
    targets_lf: pl.LazyFrame = pl.scan_parquet(target_posts_path)

    # Validate target posts schema (wide format)
    targets_schema = {
        "target_did": str,
        "seen_at": pl.Datetime,
    }
    validate_dataframe_schema(targets_lf, targets_schema)
    logger.info("✓ target_posts schema validated")

    # Log input sizes (collect counts efficiently)
    n_likes = likes_lf.select(pl.len()).collect().item()
    n_targets = targets_lf.select(pl.len()).collect().item()
    logger.info(f"Input sizes: {n_likes:,} likes, {n_targets:,} targets")

    # === Build user history directory ===
    log_operation_start('Build user history directory', 'STAGE_02_FEATURIZE', logger)

    directory_lf = _build_user_history_directory(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=max_prior_likes,
        logger=logger,
    )

    # === Write output ===
    log_operation_start('Write user history directory', 'STAGE_02_FEATURIZE', logger)
    output_path = out_dir / f"history_posts_{out_dir.name}.parquet"

    # Collect and ensure column order: target_idx, target_did, prior_emb_indices
    # target_idx is already assigned inside _build_user_history_directory
    directory_df = directory_lf.collect()
    directory_df = directory_df.select(["target_idx", "target_did", "prior_emb_indices"])

    directory_df.write_parquet(output_path, compression="zstd")

    n_output = len(directory_df)
    n_with_history = directory_df.filter(pl.col("prior_emb_indices").list.len() > 0).height
    n_empty_history = n_output - n_with_history

    # Stats on prior likes counts
    prior_counts = directory_df["prior_emb_indices"].list.len()
    mean_prior = prior_counts.mean()
    max_prior = prior_counts.max()
    min_prior = prior_counts.filter(prior_counts > 0).min() if n_with_history > 0 else 0

    logger.info(f"✓ Wrote {n_output:,} directory entries to {output_path.name}")
    logger.info(f"  With history: {n_with_history:,} ({100*n_with_history/n_output:.1f}%)")
    logger.info(f"  Empty history: {n_empty_history:,} ({100*n_empty_history/n_output:.1f}%)")
    logger.info(f"  Prior likes per target: mean={mean_prior:.1f}, min={min_prior}, max={max_prior}")

    runtime = time.time() - t0

    # === Stage info ===
    info_lines = [
        f"stage: featurize (user_history_directory)",
        f"runtime_seconds: {runtime:.2f}",
        f"settings: max_prior_likes={max_prior_likes}",
        f"inputs: likes_core ({n_likes:,}), target_posts ({n_targets:,})",
        f"outputs: user_history_directory ({n_output:,} entries)",
        f"stats: with_history={n_with_history:,}, empty_history={n_empty_history:,}",
        f"stats: mean_prior={mean_prior:.1f}, max_prior={max_prior}",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    logger.info(f"Stage 2 completed in {runtime:.2f}s")

    return {
        'output_dir': out_dir,
        'artifacts': {
            'user_history_directory_path': str(output_path),
        }
    }
