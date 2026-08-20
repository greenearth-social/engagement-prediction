"""Bounded Parquet processing for Stage 1 positive-post membership."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Any, Sequence

import polars as pl

from engagement_prediction.data import ingex
from engagement_prediction.data import post_selection as post_data


QUERY_KEY = ["did", "query_hour"]
POSITIVE_KEY = ["did", "query_hour", "subject_uri"]
INTERNAL_POSITIVE_COLUMNS = [
    "did",
    "query_hour",
    "user_cohort",
    "split",
    "subject_uri",
    "like_created_at",
]
INTERNAL_POSITIVE_SCHEMA = {
    "did": pl.String,
    "query_hour": post_data.UTC_DATETIME,
    "user_cohort": pl.String,
    "split": pl.String,
    "subject_uri": pl.String,
    "like_created_at": post_data.UTC_DATETIME,
}
MEMBERSHIP_POST_SCHEMA = {
    "subject_uri": pl.String,
    "_post_row_valid": pl.Boolean,
}


def _sink_partitioned(
    lf: pl.LazyFrame,
    *,
    output_path: Path,
    key: str,
) -> None:
    output_path.mkdir(parents=True, exist_ok=False)
    lf.sink_parquet(
        pl.PartitionBy(
            output_path,
            key=key,
            include_key=False,
            approximate_bytes_per_file="auto",
        ),
        compression="zstd",
        maintain_order=False,
        engine="streaming",
    )


def _load_paths(paths: list[Path], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame(schema=schema)
    return pl.read_parquet(paths)


def _merge_numeric_stats(stats: list[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for values in stats:
        for key, value in values.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def materialize_post_rows(
    *,
    post_paths: list[str],
    posts_start: datetime,
    posts_end: datetime,
    partition_count: int,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Normalize and route post rows by URI for bounded membership checks."""
    started = time.monotonic()
    logger.info(
        "Scanning and stream-sinking %s post source files into URI partitions",
        f"{len(post_paths):,}",
    )
    post_rows_lf = post_data.normalize_posts(
        ingex.scan_parquet_files(post_paths),
        posts_start=posts_start,
        posts_end=posts_end,
    ).select(list(MEMBERSHIP_POST_SCHEMA)).with_columns(
        post_data.post_partition_expr(partition_count)
    )
    _sink_partitioned(
        post_rows_lf,
        output_path=output_path,
        key="_post_partition",
    )
    logger.info("Finished partitioning post source rows in %.1fs", time.monotonic() - started)


def materialize_provisional_positive_rows(
    *,
    positive_rows_lf: pl.LazyFrame,
    sampled_queries_lf: pl.LazyFrame,
    partition_count: int,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Route likes for provisionally sampled query-hours by post URI."""
    started = time.monotonic()
    provisional_lf = (
        positive_rows_lf
        .join(sampled_queries_lf.select(QUERY_KEY), on=QUERY_KEY, how="inner")
        .select(INTERNAL_POSITIVE_COLUMNS)
        .with_columns(post_data.post_partition_expr(partition_count))
    )
    logger.info("Rescanning likes and routing provisional positives into URI partitions")
    _sink_partitioned(
        provisional_lf,
        output_path=output_path,
        key="_post_partition",
    )
    logger.info("Finished partitioning provisional positives in %.1fs", time.monotonic() - started)


def _deduplicate_positive_rows(positive_rows_df: pl.DataFrame) -> pl.DataFrame:
    if positive_rows_df.is_empty():
        return pl.DataFrame(schema=INTERNAL_POSITIVE_SCHEMA)
    return (
        positive_rows_df
        .group_by([*POSITIVE_KEY, "user_cohort", "split"])
        .agg(pl.col("like_created_at").min())
        .select(INTERNAL_POSITIVE_COLUMNS)
        .sort(["query_hour", "did", "subject_uri"])
    )


def _add_split_counts(
    stats: dict[str, dict[str, int]],
    df: pl.DataFrame,
    field_name: str,
) -> None:
    if df.is_empty():
        return
    for split, count in df.group_by("split").len().iter_rows():
        split_stats = stats.setdefault(split, {})
        split_stats[field_name] = split_stats.get(field_name, 0) + int(count)


def _select_valid_post_keys(post_rows_df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    invalid_row_count = post_rows_df.filter(~pl.col("_post_row_valid")).height
    valid_rows_df = post_rows_df.filter(pl.col("_post_row_valid")).select("subject_uri")
    duplicate_counts_df = valid_rows_df.group_by("subject_uri").len()
    duplicate_uri_count = duplicate_counts_df.filter(pl.col("len") > 1).height
    duplicate_row_count = int(
        duplicate_counts_df.select(
            (pl.col("len") - 1).clip(lower_bound=0).sum()
        ).item()
        or 0
    )
    unique_posts_df = valid_rows_df.unique().sort("subject_uri")
    return unique_posts_df, {
        "post_source_row_count": post_rows_df.height,
        "invalid_post_row_count": invalid_row_count,
        "duplicate_post_row_count": duplicate_row_count,
        "duplicate_post_uri_count": duplicate_uri_count,
        "unique_valid_post_count": unique_posts_df.height,
    }


def filter_positive_partitions(
    *,
    provisional_positive_rows_path: Path,
    post_rows_path: Path,
    eligible_positive_rows_path: Path,
    partition_count: int,
    splits: Sequence[str],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Semi-join deduplicated positives to valid posts one URI partition at a time."""
    eligible_positive_rows_path.mkdir(parents=True, exist_ok=False)
    positive_stats = {
        split: {
            "selected_like_row_count": 0,
            "provisional_positive_count": 0,
            "retained_positive_count": 0,
            "missing_post_positive_count": 0,
        }
        for split in splits
    }
    post_stats: list[dict[str, int]] = []
    started = time.monotonic()
    logger.info("Beginning bounded post-membership checks across %s partitions", partition_count)

    for partition_id in range(partition_count):
        partition_started = time.monotonic()
        positive_rows_df = _load_paths(
            post_data.partition_parquet_paths(
                provisional_positive_rows_path,
                partition_id,
            ),
            INTERNAL_POSITIVE_SCHEMA,
        )
        deduplicated_df = _deduplicate_positive_rows(positive_rows_df)
        normalized_posts_df = _load_paths(
            post_data.partition_parquet_paths(post_rows_path, partition_id),
            MEMBERSHIP_POST_SCHEMA,
        )
        unique_posts_df, partition_post_stats = _select_valid_post_keys(normalized_posts_df)
        post_stats.append(partition_post_stats)
        eligible_df = deduplicated_df.join(
            unique_posts_df.select("subject_uri"),
            on="subject_uri",
            how="semi",
        ).sort(["query_hour", "did", "subject_uri"])
        missing_df = deduplicated_df.join(
            unique_posts_df.select("subject_uri"),
            on="subject_uri",
            how="anti",
        )

        _add_split_counts(positive_stats, positive_rows_df, "selected_like_row_count")
        _add_split_counts(positive_stats, deduplicated_df, "provisional_positive_count")
        _add_split_counts(positive_stats, eligible_df, "retained_positive_count")
        _add_split_counts(positive_stats, missing_df, "missing_post_positive_count")

        if not eligible_df.is_empty():
            eligible_df.write_parquet(
                eligible_positive_rows_path / f"part-{partition_id:05d}.parquet",
                compression="zstd",
            )
        logger.info(
            "Checked positive URI partition %s/%s in %.1fs: selected_rows=%s "
            "deduplicated=%s retained=%s missing=%s valid_posts=%s",
            partition_id + 1,
            partition_count,
            time.monotonic() - partition_started,
            f"{positive_rows_df.height:,}",
            f"{deduplicated_df.height:,}",
            f"{eligible_df.height:,}",
            f"{missing_df.height:,}",
            f"{unique_posts_df.height:,}",
        )

    if not list(eligible_positive_rows_path.glob("*.parquet")):
        pl.DataFrame(schema=INTERNAL_POSITIVE_SCHEMA).write_parquet(
            eligible_positive_rows_path / "part-00000.parquet",
            compression="zstd",
        )
    logger.info("Finished post-membership checks in %.1fs", time.monotonic() - started)
    return {
        "post_source_stats": _merge_numeric_stats(post_stats),
        "positive_filter_stats_by_split": positive_stats,
    }


def scan_eligible_positive_rows(path: Path) -> pl.LazyFrame:
    paths = sorted(Path(path).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No eligible positive Parquet shards found in {path}")
    return pl.scan_parquet(paths)
