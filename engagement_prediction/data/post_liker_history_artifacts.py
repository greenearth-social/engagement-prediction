"""Bounded disk orchestration for Stage 5 post-liker history artifacts."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import time

import polars as pl

from engagement_prediction.data import ingex
from engagement_prediction.data import likes
from engagement_prediction.data import post_liker_history
from engagement_prediction.data import post_selection
from engagement_prediction.data.parquet import read_parquet_parts, sink_partitioned_parquet


def _public_partition_path(dataset_path: Path, partition_id: int) -> list[Path]:
    """Return one Stage 3/4 public URI-partition file, if present."""

    path = Path(dataset_path) / f"part-{partition_id:05d}.parquet"
    return [path] if path.exists() else []


def materialize_selected_post_routes(
    *,
    required_posts_path: Path,
    missing_required_posts_path: Path,
    negative_post_uris_path: Path,
    stage3_partition_count: int,
    output_path: Path,
    shard_path: Path,
    partition_count: int,
) -> None:
    """Build selected roles in bounded Stage 3 shards, then route by Stage 5 hash."""
    shard_path.mkdir(parents=True, exist_ok=False)
    for partition_id in range(stage3_partition_count):
        required_posts_df = read_parquet_parts(
            _public_partition_path(required_posts_path, partition_id),
            empty=post_selection.empty_frame(post_selection.REQUIRED_POST_SCHEMA),
        )
        missing_required_posts_df = read_parquet_parts(
            _public_partition_path(missing_required_posts_path, partition_id),
            empty=post_selection.empty_frame(post_selection.REQUIRED_POST_SCHEMA),
        )
        negative_post_uris_df = read_parquet_parts(
            _public_partition_path(negative_post_uris_path, partition_id),
            empty=post_liker_history.empty_frame({"subject_uri": pl.String}),
        )
        selected_posts_df = post_liker_history.build_selected_posts(
            required_posts_df,
            missing_required_posts_df,
            negative_post_uris_df,
        )
        post_liker_history.validate_selected_posts(selected_posts_df)
        if not selected_posts_df.is_empty():
            selected_posts_df.write_parquet(
                shard_path / f"part-{partition_id:05d}.parquet",
                compression="zstd",
            )

    shards = sorted(shard_path.glob("*.parquet"))
    if not shards:
        output_path.mkdir(parents=True, exist_ok=False)
        return
    sink_partitioned_parquet(
        pl.scan_parquet(shards).with_columns(
            post_selection.post_partition_expr(partition_count)
        ),
        output_path=output_path,
        key="_post_partition",
    )


def materialize_normalized_likes(
    *,
    like_paths: list[str],
    source_start: datetime,
    source_end: datetime,
    output_path: Path,
    partition_count: int,
) -> None:
    """Normalize the exact source snapshot and route all valid likes by post URI."""
    normalized_lf = (
        likes.prepare_likes(
            ingex.scan_parquet_files(like_paths),
            start=source_start,
            end=source_end,
        )
        .select(
            "subject_uri",
            pl.col("did").alias("liker_did"),
            "like_created_at",
        )
    )
    sink_partitioned_parquet(
        normalized_lf.with_columns(
            post_selection.post_partition_expr(partition_count)
        ),
        output_path=output_path,
        key="_post_partition",
    )


def _count_rows(lf: pl.LazyFrame) -> int:
    """Count a lazy partition without collecting its event columns."""

    return int(lf.select(pl.len()).collect(engine="streaming").item())


def _event_count_buckets(post_liker_posts_df: pl.DataFrame) -> dict[str, int]:
    """Summarize per-post event counts into stable reporting buckets."""

    counts = pl.col("like_event_count")
    return {
        "0": post_liker_posts_df.filter(counts == 0).height,
        "1": post_liker_posts_df.filter(counts == 1).height,
        "2_to_4": post_liker_posts_df.filter(counts.is_between(2, 4)).height,
        "5_to_9": post_liker_posts_df.filter(counts.is_between(5, 9)).height,
        "10_to_49": post_liker_posts_df.filter(counts.is_between(10, 49)).height,
        "50_to_99": post_liker_posts_df.filter(counts.is_between(50, 99)).height,
        "100_to_999": post_liker_posts_df.filter(counts.is_between(100, 999)).height,
        "1000_plus": post_liker_posts_df.filter(counts >= 1000).height,
    }


def process_uri_partitions(
    *,
    selected_post_routes_path: Path,
    normalized_likes_path: Path,
    post_liker_events_path: Path,
    post_liker_posts_path: Path,
    source_start: datetime,
    source_end: datetime,
    partition_count: int,
    logger: logging.Logger,
) -> dict[str, object]:
    """Stream matched events and write summaries one URI partition at a time."""
    post_liker_events_path.mkdir(parents=True, exist_ok=False)
    post_liker_posts_path.mkdir(parents=True, exist_ok=False)
    totals = {
        "selected_post_count": 0,
        "positive_post_count": 0,
        "history_post_count": 0,
        "negative_post_count": 0,
        "positive_history_overlap_count": 0,
        "positive_negative_overlap_count": 0,
        "history_negative_overlap_count": 0,
        "all_role_overlap_count": 0,
        "valid_source_like_count": 0,
        "matched_like_event_count": 0,
        "posts_with_likes_count": 0,
        "posts_without_likes_count": 0,
        "max_like_events_per_post": 0,
    }
    event_count_distribution = {
        "0": 0,
        "1": 0,
        "2_to_4": 0,
        "5_to_9": 0,
        "10_to_49": 0,
        "50_to_99": 0,
        "100_to_999": 0,
        "1000_plus": 0,
    }
    partition_stats: list[dict[str, object]] = []
    started = time.monotonic()

    for partition_id in range(partition_count):
        partition_started = time.monotonic()
        logger.info(
            "Processing post-liker URI partition %s/%s",
            partition_id + 1,
            partition_count,
        )
        selected_posts_df = read_parquet_parts(
            post_selection.partition_parquet_paths(
                selected_post_routes_path,
                partition_id,
            ),
            empty=post_liker_history.empty_frame(
                post_liker_history.SELECTED_POST_SCHEMA
            ),
        ).select(post_liker_history.SELECTED_POST_COLUMNS).sort("subject_uri")
        post_liker_history.validate_selected_posts(selected_posts_df)

        like_paths = post_selection.partition_parquet_paths(
            normalized_likes_path,
            partition_id,
        )
        if like_paths:
            source_likes_lf = pl.scan_parquet(like_paths).select(
                post_liker_history.POST_LIKER_EVENT_COLUMNS
            )
        else:
            source_likes_lf = post_liker_history.empty_frame(
                post_liker_history.POST_LIKER_EVENT_SCHEMA
            ).lazy()
        valid_source_like_count = _count_rows(source_likes_lf)
        # The semi-join retains complete event multiplicity for selected posts;
        # it filters only by URI and performs no user or event deduplication.
        matched_events_lf = (
            source_likes_lf.join(
                selected_posts_df.select("subject_uri").lazy(),
                on="subject_uri",
                how="semi",
            )
            .select(post_liker_history.POST_LIKER_EVENT_COLUMNS)
            .sort(["subject_uri", "like_created_at", "liker_did"])
        )
        event_part_path = post_liker_events_path / f"part-{partition_id:05d}.parquet"
        matched_events_lf.sink_parquet(
            event_part_path,
            compression="zstd",
            maintain_order=True,
            engine="streaming",
        )
        written_events_lf = pl.scan_parquet(event_part_path)
        # Scan the persisted public events once. The narrow audit drives both
        # the public post summaries and every subsequent integrity check.
        event_audit_df = post_liker_history.audit_event_partition(written_events_lf)
        event_stats_df = post_liker_history.event_stats_from_audit(event_audit_df)
        post_liker_posts_df = post_liker_history.build_post_liker_posts(
            selected_posts_df,
            event_stats_df,
        )
        post_part_path = post_liker_posts_path / f"part-{partition_id:05d}.parquet"
        post_liker_posts_df.write_parquet(post_part_path, compression="zstd")
        post_liker_history.validate_public_partition(
            event_audit_df=event_audit_df,
            post_liker_posts_df=post_liker_posts_df,
            selected_posts_df=selected_posts_df,
            source_start=source_start,
            source_end=source_end,
            partition_id=partition_id,
            partition_count=partition_count,
        )

        matched_like_event_count = int(
            post_liker_posts_df.get_column("like_event_count").sum() or 0
        )
        posts_with_likes_count = post_liker_posts_df.filter(
            pl.col("like_event_count") > 0
        ).height
        max_like_events = int(
            post_liker_posts_df.get_column("like_event_count").max() or 0
        )
        totals["selected_post_count"] += selected_posts_df.height
        totals["positive_post_count"] += selected_posts_df.filter(
            pl.col("is_positive")
        ).height
        totals["history_post_count"] += selected_posts_df.filter(
            pl.col("is_history")
        ).height
        totals["negative_post_count"] += selected_posts_df.filter(
            pl.col("is_negative")
        ).height
        totals["positive_history_overlap_count"] += selected_posts_df.filter(
            pl.col("is_positive") & pl.col("is_history")
        ).height
        totals["positive_negative_overlap_count"] += selected_posts_df.filter(
            pl.col("is_positive") & pl.col("is_negative")
        ).height
        totals["history_negative_overlap_count"] += selected_posts_df.filter(
            pl.col("is_history") & pl.col("is_negative")
        ).height
        totals["all_role_overlap_count"] += selected_posts_df.filter(
            pl.col("is_positive")
            & pl.col("is_history")
            & pl.col("is_negative")
        ).height
        totals["valid_source_like_count"] += valid_source_like_count
        totals["matched_like_event_count"] += matched_like_event_count
        totals["posts_with_likes_count"] += posts_with_likes_count
        totals["posts_without_likes_count"] += (
            post_liker_posts_df.height - posts_with_likes_count
        )
        totals["max_like_events_per_post"] = max(
            totals["max_like_events_per_post"],
            max_like_events,
        )
        for name, value in _event_count_buckets(post_liker_posts_df).items():
            event_count_distribution[name] += value

        elapsed = time.monotonic() - partition_started
        partition_stats.append({
            "partition_id": partition_id,
            "selected_post_count": selected_posts_df.height,
            "valid_source_like_count": valid_source_like_count,
            "matched_like_event_count": matched_like_event_count,
            "runtime_seconds": elapsed,
        })
        logger.info(
            "Finished post-liker URI partition %s/%s in %.1fs: posts=%s "
            "source_likes=%s matched_events=%s zero_like_posts=%s",
            partition_id + 1,
            partition_count,
            elapsed,
            f"{selected_posts_df.height:,}",
            f"{valid_source_like_count:,}",
            f"{matched_like_event_count:,}",
            f"{post_liker_posts_df.height - posts_with_likes_count:,}",
        )

    logger.info(
        "Finished post-liker processing across URI partitions in %.1fs",
        time.monotonic() - started,
    )
    return {
        **totals,
        "event_count_distribution": event_count_distribution,
        "partition_stats": partition_stats,
    }
