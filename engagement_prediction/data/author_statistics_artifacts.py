"""Bounded disk orchestration for Stage 6 author-statistics artifacts."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Protocol

import polars as pl

from engagement_prediction.data import author_statistics
from engagement_prediction.data import ingex
from engagement_prediction.data import likes
from engagement_prediction.data import source_metadata
from engagement_prediction.data.parquet import read_parquet_parts, sink_partitioned_parquet


class AuthorStatisticsConfig(Protocol):
    """Structural settings required by Stage 6 artifact helpers."""

    support_start: datetime
    support_end: datetime
    partition_count: int
    source_metadata_partition_count: int


def _count_rows(lf: pl.LazyFrame) -> int:
    """Count a lazy source partition without collecting its columns."""

    return int(lf.select(pl.len()).collect(engine="streaming").item())


def _write_if_not_empty(df: pl.DataFrame, path: Path) -> None:
    """Skip arbitrary empty shard files; public empty handling happens later."""

    if not df.is_empty():
        df.write_parquet(path, compression="zstd")


def _support_count_buckets(df: pl.DataFrame, column: str) -> dict[str, int]:
    """Build stable histogram buckets for author-support reporting."""

    value = pl.col(column)
    return {
        "0": df.filter(value == 0).height,
        "1_to_4": df.filter(value.is_between(1, 4)).height,
        "5_to_9": df.filter(value.is_between(5, 9)).height,
        "10_to_19": df.filter(value.is_between(10, 19)).height,
        "20_to_49": df.filter(value.is_between(20, 49)).height,
        "50_to_99": df.filter(value.is_between(50, 99)).height,
        "100_to_999": df.filter(value.is_between(100, 999)).height,
        "1000_plus": df.filter(value >= 1000).height,
    }


def materialize_like_routes(
    *,
    like_paths: list[str],
    normalized_likes_path: Path,
    config: AuthorStatisticsConfig,
    logger: logging.Logger,
) -> None:
    """Normalize the exact like snapshot and route rows to Stage 00 URI partitions."""
    started = time.monotonic()
    logger.info(
        "Scanning and stream-sinking all valid events from %s exact like files",
        f"{len(like_paths):,}",
    )
    normalized_likes_lf = (
        likes.prepare_likes(
            ingex.scan_parquet_files(like_paths),
            start=config.support_start,
            end=config.support_end,
        )
        .filter(
            (pl.col("did").str.len_chars() > 0)
            & (pl.col("subject_uri").str.len_chars() > 0)
        )
        .select("subject_uri", "like_created_at")
        .with_columns(
            source_metadata.uri_partition_expr(config.source_metadata_partition_count)
        )
    )
    sink_partitioned_parquet(
        normalized_likes_lf,
        output_path=normalized_likes_path,
        key="_post_partition",
    )
    logger.info(
        "Finished routing valid like rows in %.1fs",
        time.monotonic() - started,
    )


def process_uri_partitions(
    *,
    post_metadata_path: Path,
    normalized_likes_path: Path,
    per_post_shards_path: Path,
    config: AuthorStatisticsConfig,
    logger: logging.Logger,
) -> dict[str, object]:
    """Filter canonical metadata and collapse raw likes to one row per post URI."""
    per_post_shards_path.mkdir(parents=True, exist_ok=False)
    totals = {
        "root_reply_overlap_count": 0,
        "resolved_post_count": 0,
        "resolved_root_post_count": 0,
        "resolved_reply_post_count": 0,
        "valid_source_like_count": 0,
        "matched_like_event_count": 0,
        "unmatched_like_event_count": 0,
        "posts_with_likes_count": 0,
        "posts_without_likes_count": 0,
    }
    partition_stats: list[dict[str, object]] = []
    started = time.monotonic()

    for partition_id in range(config.source_metadata_partition_count):
        partition_started = time.monotonic()
        logger.info(
            "Processing author-statistics URI partition %s/%s",
            partition_id + 1,
            config.source_metadata_partition_count,
        )
        metadata_file = post_metadata_path / f"part-{partition_id:05d}.parquet"
        resolved_posts_df = read_parquet_parts(
            [metadata_file] if metadata_file.exists() else [],
            empty=source_metadata.empty_frame(source_metadata.POST_METADATA_SCHEMA),
        ).filter(
            (pl.col("post_created_at") >= config.support_start)
            & (pl.col("post_created_at") < config.support_end)
        )

        like_paths = source_metadata.partition_parquet_paths(
            normalized_likes_path,
            partition_id,
        )
        if like_paths:
            source_likes_lf = pl.scan_parquet(like_paths).select(
                "subject_uri",
                "like_created_at",
            )
        else:
            source_likes_lf = author_statistics.empty_frame({
                "subject_uri": pl.String,
                "like_created_at": author_statistics.UTC_DATETIME,
            }).lazy()
        valid_source_like_count = _count_rows(source_likes_lf)
        # Duplicate raw events intentionally contribute independently. Likes
        # whose URI has no resolved in-window record are counted as unmatched.
        matched_like_counts_df = (
            source_likes_lf.join(
                resolved_posts_df.select("subject_uri").lazy(),
                on="subject_uri",
                how="semi",
            )
            .group_by("subject_uri")
            .agg(pl.len().cast(pl.UInt64).alias("received_like_count"))
            .sort("subject_uri")
            .collect(engine="streaming")
        )
        matched_like_event_count = int(
            matched_like_counts_df.get_column("received_like_count").sum() or 0
        )
        per_post_df = author_statistics.build_per_post_statistics(
            resolved_posts_df,
            matched_like_counts_df,
        )
        author_statistics.validate_per_post_statistics(
            per_post_df,
            support_start=config.support_start,
            support_end=config.support_end,
        )
        _write_if_not_empty(
            per_post_df,
            per_post_shards_path / f"part-{partition_id:05d}.parquet",
        )

        posts_with_likes_count = per_post_df.filter(
            pl.col("received_like_count") > 0
        ).height
        totals["resolved_post_count"] += per_post_df.height
        totals["resolved_root_post_count"] += per_post_df.filter(
            ~pl.col("is_reply")
        ).height
        totals["resolved_reply_post_count"] += per_post_df.filter(
            pl.col("is_reply")
        ).height
        totals["valid_source_like_count"] += valid_source_like_count
        totals["matched_like_event_count"] += matched_like_event_count
        totals["unmatched_like_event_count"] += (
            valid_source_like_count - matched_like_event_count
        )
        totals["posts_with_likes_count"] += posts_with_likes_count
        totals["posts_without_likes_count"] += per_post_df.height - posts_with_likes_count

        elapsed = time.monotonic() - partition_started
        partition_stats.append({
            "partition_id": partition_id,
            "resolved_post_count": per_post_df.height,
            "valid_source_like_count": valid_source_like_count,
            "matched_like_event_count": matched_like_event_count,
            "runtime_seconds": elapsed,
        })
        logger.info(
            "Finished URI partition %s/%s in %.1fs: posts=%s source_likes=%s "
            "matched_likes=%s unmatched_likes=%s",
            partition_id + 1,
            config.source_metadata_partition_count,
            elapsed,
            f"{per_post_df.height:,}",
            f"{valid_source_like_count:,}",
            f"{matched_like_event_count:,}",
            f"{valid_source_like_count - matched_like_event_count:,}",
        )

    logger.info(
        "Finished URI-partition processing in %.1fs",
        time.monotonic() - started,
    )
    return {
        **totals,
        "uri_partition_stats": partition_stats,
    }


def route_per_post_rows_by_author(
    *,
    per_post_shards_path: Path,
    per_post_by_author_path: Path,
    partition_count: int,
) -> None:
    """Route already-collapsed post rows so every author has one local partition."""
    shards = sorted(per_post_shards_path.glob("*.parquet"))
    if not shards:
        per_post_by_author_path.mkdir(parents=True, exist_ok=False)
        return
    sink_partitioned_parquet(
        pl.scan_parquet(shards).with_columns(
            author_statistics.author_partition_expr(partition_count)
        ),
        output_path=per_post_by_author_path,
        key="_author_partition",
    )


def _author_partition_paths(dataset_path: Path, partition_id: int) -> list[Path]:
    """Return physical per-post files assigned to one author hash bucket."""

    partition_dir = Path(dataset_path) / f"_author_partition={partition_id}"
    return sorted(partition_dir.rglob("*.parquet")) if partition_dir.exists() else []


def process_author_partitions(
    *,
    per_post_by_author_path: Path,
    author_statistics_path: Path,
    config: AuthorStatisticsConfig,
    logger: logging.Logger,
) -> dict[str, object]:
    """Aggregate and publish every author one stable hash partition at a time."""
    author_statistics_path.mkdir(parents=True, exist_ok=False)
    totals = {
        "author_count": 0,
        "zero_like_author_count": 0,
        "max_posts_per_author": 0,
        "max_received_likes_per_author": 0,
    }
    post_count_distribution = {
        name: 0
        for name in ("0", "1_to_4", "5_to_9", "10_to_19", "20_to_49", "50_to_99", "100_to_999", "1000_plus")
    }
    received_like_count_distribution = dict(post_count_distribution)
    partition_stats: list[dict[str, object]] = []
    started = time.monotonic()

    for partition_id in range(config.partition_count):
        partition_started = time.monotonic()
        logger.info(
            "Processing author partition %s/%s",
            partition_id + 1,
            config.partition_count,
        )
        per_post_df = read_parquet_parts(
            _author_partition_paths(per_post_by_author_path, partition_id),
            empty=author_statistics.empty_frame(author_statistics.PER_POST_SCHEMA),
        ).select(author_statistics.PER_POST_COLUMNS)
        author_stats_df = author_statistics.aggregate_author_statistics(per_post_df)
        author_statistics.validate_author_partition(author_stats_df)
        misplaced_count = author_stats_df.filter(
            author_statistics.author_partition_expr(config.partition_count)
            != partition_id
        ).height
        if misplaced_count:
            raise ValueError(
                f"Author partition {partition_id} contains {misplaced_count} misplaced rows"
            )
        author_stats_df.write_parquet(
            author_statistics_path / f"part-{partition_id:05d}.parquet",
            compression="zstd",
        )
        totals["author_count"] += author_stats_df.height
        totals["zero_like_author_count"] += author_stats_df.filter(
            pl.col("received_like_count") == 0
        ).height
        totals["max_posts_per_author"] = max(
            totals["max_posts_per_author"],
            int(author_stats_df.get_column("post_count").max() or 0),
        )
        totals["max_received_likes_per_author"] = max(
            totals["max_received_likes_per_author"],
            int(author_stats_df.get_column("received_like_count").max() or 0),
        )
        for name, value in _support_count_buckets(author_stats_df, "post_count").items():
            post_count_distribution[name] += value
        for name, value in _support_count_buckets(
            author_stats_df,
            "received_like_count",
        ).items():
            received_like_count_distribution[name] += value

        elapsed = time.monotonic() - partition_started
        partition_stats.append({
            "partition_id": partition_id,
            "author_count": author_stats_df.height,
            "runtime_seconds": elapsed,
        })
        logger.info(
            "Finished author partition %s/%s in %.1fs: authors=%s",
            partition_id + 1,
            config.partition_count,
            elapsed,
            f"{author_stats_df.height:,}",
        )

    logger.info(
        "Finished author-partition processing in %.1fs",
        time.monotonic() - started,
    )
    return {
        **totals,
        "post_count_distribution": post_count_distribution,
        "received_like_count_distribution": received_like_count_distribution,
        "author_partition_stats": partition_stats,
    }
