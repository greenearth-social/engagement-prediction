"""Bounded disk orchestration for Stage 4 popularity and negative selection.

Popularity is first calculated in URI partitions, where all likes for a post are
local. Only bounded per-hour finalists leave those partitions. They are shuffled
by query hour to apply global quotas, then the selected URIs are shuffled back by
URI to publish a globally unique post list aligned with Stage 3.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Protocol

import polars as pl

from engagement_prediction.data import candidate_popularity
from engagement_prediction.data import ingex
from engagement_prediction.data import likes
from engagement_prediction.data import negative_selection
from engagement_prediction.data import post_selection
from engagement_prediction.data.parquet import (
    ensure_typed_parquet_dataset,
    read_parquet_parts,
    sink_partitioned_parquet,
    write_parquet_part_if_not_empty,
)


class NegativeSelectionConfigLike(Protocol):
    """Structural settings required by Stage 4 artifact construction."""

    @property
    def negative_candidates_per_hour(self) -> int: ...

    @property
    def min_likes_for_popular_candidate(self) -> int: ...

    @property
    def popular_candidate_fraction(self) -> float: ...

    @property
    def max_candidate_age_hours(self) -> int: ...

    @property
    def partition_count(self) -> int: ...

    @property
    def random_seed(self) -> int: ...


def _public_partition_path(dataset_path: Path, partition_id: int) -> list[Path]:
    """Return the Stage 3 public file for one URI partition, if present."""

    path = Path(dataset_path) / f"part-{partition_id:05d}.parquet"
    return [path] if path.exists() else []


def hour_partition_expr(partition_count: int) -> pl.Expr:
    """Assign query hours to stable partitions for global quota resolution."""
    if partition_count <= 0:
        raise ValueError("partition_count must be positive")
    return (
        pl.concat_str(
            [
                pl.lit("negative-selection-hour"),
                pl.col("query_hour").dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            ],
            separator="|",
        )
        .hash(seed=0)
        .mod(pl.lit(partition_count, dtype=pl.UInt64))
        .cast(pl.UInt32)
        .alias("_hour_partition")
    )


def materialize_normalized_likes(
    *,
    like_paths: list[str],
    source_start,
    source_end,
    output_path: Path,
    partition_count: int,
) -> None:
    """Normalize the exact Stage 1 like snapshot and route narrow rows by URI."""
    normalized_lf = likes.prepare_likes(
        ingex.scan_parquet_files(like_paths),
        start=source_start,
        end=source_end,
    ).select("subject_uri", "like_created_at")
    sink_partitioned_parquet(
        normalized_lf.with_columns(post_selection.post_partition_expr(partition_count)),
        output_path=output_path,
        key="_post_partition",
    )


def _merge_hour_counts(
    target: dict[str, dict[str, int]],
    frame: pl.DataFrame,
    *,
    minimum_likes: int,
) -> None:
    """Accumulate candidate eligibility counters keyed by query hour."""

    if frame.is_empty():
        return
    for row in frame.group_by("query_hour").agg(
        pl.len().alias("eligible_candidate_count"),
        (pl.col("prior_like_count") == 0).sum().alias("zero_like_candidate_count"),
        (pl.col("prior_like_count") >= minimum_likes)
        .sum()
        .alias("popular_eligible_candidate_count"),
    ).to_dicts():
        key = row.pop("query_hour").isoformat()
        values = target.setdefault(key, {})
        for name, value in row.items():
            values[name] = values.get(name, 0) + int(value)


def process_uri_partitions(
    *,
    posts_path: Path,
    candidate_sources_path: Path,
    normalized_likes_path: Path,
    query_hours_df: pl.DataFrame,
    local_finalists_path: Path,
    config: NegativeSelectionConfigLike,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Calculate popularity and write bounded method finalists per URI partition.

    The full candidate-by-query-hour relation is intentionally temporary. Only
    each partition's top K rows for each method cross into the hour shuffle, which
    bounds intermediate disk and memory while preserving the global top-K result.
    """
    local_finalists_path.mkdir(parents=True, exist_ok=False)
    totals = {
        "candidate_reservoir_count": 0,
        "valid_like_row_count": 0,
        "candidate_matched_like_row_count": 0,
        "eligible_candidate_hour_count": 0,
        "local_finalist_row_count": 0,
    }
    candidate_hour_stats: dict[str, dict[str, int]] = {}
    started = time.monotonic()
    for partition_id in range(config.partition_count):
        partition_started = time.monotonic()
        logger.info(
            "Calculating popularity in URI partition %s/%s",
            partition_id + 1,
            config.partition_count,
        )
        posts_df = read_parquet_parts(
            _public_partition_path(posts_path, partition_id),
            empty=post_selection.empty_frame(post_selection.POST_SCHEMA),
        )
        candidate_sources_df = read_parquet_parts(
            _public_partition_path(candidate_sources_path, partition_id),
            empty=post_selection.empty_frame(post_selection.CANDIDATE_SOURCE_SCHEMA),
        )
        candidates_df = candidate_popularity.build_candidate_reservoir(
            posts_df,
            candidate_sources_df,
        )
        like_rows_df = read_parquet_parts(
            post_selection.partition_parquet_paths(
                normalized_likes_path,
                partition_id,
            ),
            empty=candidate_popularity.empty_frame(
                candidate_popularity.NORMALIZED_LIKE_SCHEMA
            ),
        ).select(candidate_popularity.NORMALIZED_LIKE_COLUMNS)
        # Popularity needs only likes for the candidate reservoir. Duplicate
        # raw rows are preserved because the public count contract counts events.
        matched_likes_df = like_rows_df.join(
            candidates_df.select("subject_uri"),
            on="subject_uri",
            how="semi",
        )
        if config.negative_candidates_per_hour == 0 or query_hours_df.is_empty():
            candidate_hours_df = candidate_popularity.empty_frame(
                candidate_popularity.CANDIDATE_HOUR_SCHEMA
            )
        else:
            candidate_hours_df = candidate_popularity.build_candidate_hour_popularity(
                candidates_df,
                matched_likes_df,
                query_hours_df,
                max_candidate_age_hours=config.max_candidate_age_hours,
            )
        # Retaining at most K rows per method and URI partition bounds the
        # second pass while preserving the exact global selection result.
        local_finalists_df = negative_selection.select_local_finalists(
            candidate_hours_df,
            negative_candidates_per_hour=config.negative_candidates_per_hour,
            min_likes_for_popular_candidate=config.min_likes_for_popular_candidate,
            random_seed=config.random_seed,
        )
        write_parquet_part_if_not_empty(
            local_finalists_df,
            local_finalists_path / f"part-{partition_id:05d}.parquet",
        )

        totals["candidate_reservoir_count"] += candidates_df.height
        totals["valid_like_row_count"] += like_rows_df.height
        totals["candidate_matched_like_row_count"] += matched_likes_df.height
        totals["eligible_candidate_hour_count"] += candidate_hours_df.height
        totals["local_finalist_row_count"] += local_finalists_df.height
        _merge_hour_counts(
            candidate_hour_stats,
            candidate_hours_df,
            minimum_likes=config.min_likes_for_popular_candidate,
        )
        logger.info(
            "Finished URI partition %s/%s in %.1fs: candidates=%s likes=%s "
            "matched_likes=%s eligible_candidate_hours=%s finalists=%s",
            partition_id + 1,
            config.partition_count,
            time.monotonic() - partition_started,
            f"{candidates_df.height:,}",
            f"{like_rows_df.height:,}",
            f"{matched_likes_df.height:,}",
            f"{candidate_hours_df.height:,}",
            f"{local_finalists_df.height:,}",
        )
    logger.info(
        "Finished popularity calculation across URI partitions in %.1fs",
        time.monotonic() - started,
    )
    return {**totals, "candidate_hour_stats": candidate_hour_stats}


def route_local_finalists_by_hour(
    *,
    local_finalists_path: Path,
    output_path: Path,
    partition_count: int,
) -> None:
    """Repartition bounded URI-local finalists for exact per-hour selection."""
    parts = sorted(local_finalists_path.glob("*.parquet"))
    if not parts:
        output_path.mkdir(parents=True, exist_ok=False)
        return
    sink_partitioned_parquet(
        pl.scan_parquet(parts).with_columns(hour_partition_expr(partition_count)),
        output_path=output_path,
        key="_hour_partition",
    )


def process_hour_partitions(
    *,
    routed_finalists_path: Path,
    hourly_candidates_path: Path,
    query_hours_df: pl.DataFrame,
    config: NegativeSelectionConfigLike,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Apply global quotas and write deterministic public hour partitions.

    All finalists for one query hour share a hash partition at this point, so the
    popular-first quota and random shortfall fill are global rather than local to
    one URI partition.
    """
    hourly_candidates_path.mkdir(parents=True, exist_ok=False)
    hourly_stats = {
        row["query_hour"].isoformat(): {
            "selected_candidate_count": 0,
            "popular_selected_count": 0,
            "random_selected_count": 0,
            "shortfall_count": config.negative_candidates_per_hour,
        }
        for row in query_hours_df.to_dicts()
    }
    selected_candidate_row_count = 0
    started = time.monotonic()
    for partition_id in range(config.partition_count):
        partition_started = time.monotonic()
        local_finalists_df = read_parquet_parts(
            post_selection.partition_parquet_paths(
                routed_finalists_path,
                partition_id,
                key="_hour_partition",
            ),
            empty=negative_selection.empty_frame(
                negative_selection.LOCAL_FINALIST_SCHEMA
            ),
        )
        hourly_candidates_df = negative_selection.select_hourly_candidates(
            local_finalists_df,
            negative_candidates_per_hour=config.negative_candidates_per_hour,
            popular_candidate_fraction=config.popular_candidate_fraction,
        )
        negative_selection.validate_hourly_candidates(
            hourly_candidates_df,
            negative_candidates_per_hour=config.negative_candidates_per_hour,
        )
        if hourly_candidates_df.filter(
            (pl.col("selection_source") == "popular")
            & (
                pl.col("prior_like_count")
                < config.min_likes_for_popular_candidate
            )
        ).height:
            raise ValueError("Popular selections do not satisfy the minimum like count")
        if hourly_candidates_df.select("query_hour").unique().join(
            query_hours_df,
            on="query_hour",
            how="anti",
        ).height:
            raise ValueError("Hourly candidates contain an unselected query hour")
        if not hourly_candidates_df.is_empty():
            assigned = (
                hourly_candidates_df.select("query_hour")
                .with_columns(hour_partition_expr(config.partition_count))
                .get_column("_hour_partition")
                .unique()
                .to_list()
            )
            if assigned != [partition_id]:
                raise ValueError(
                    f"Hour partition {partition_id} contains rows assigned to {assigned}"
                )
        write_parquet_part_if_not_empty(
            hourly_candidates_df,
            hourly_candidates_path / f"part-{partition_id:05d}.parquet",
        )
        selected_candidate_row_count += hourly_candidates_df.height
        for row in hourly_candidates_df.group_by("query_hour").agg(
            pl.len().alias("selected_candidate_count"),
            (pl.col("selection_source") == "popular")
            .sum()
            .alias("popular_selected_count"),
            (pl.col("selection_source") == "random")
            .sum()
            .alias("random_selected_count"),
        ).to_dicts():
            key = row.pop("query_hour").isoformat()
            values = {name: int(value) for name, value in row.items()}
            values["shortfall_count"] = max(
                config.negative_candidates_per_hour
                - values["selected_candidate_count"],
                0,
            )
            hourly_stats[key] = values
        logger.info(
            "Resolved hour partition %s/%s in %.1fs: selected_rows=%s",
            partition_id + 1,
            config.partition_count,
            time.monotonic() - partition_started,
            f"{hourly_candidates_df.height:,}",
        )
    ensure_typed_parquet_dataset(
        hourly_candidates_path,
        negative_selection.HOURLY_CANDIDATE_SCHEMA,
    )
    logger.info(
        "Finished global hourly selection in %.1fs",
        time.monotonic() - started,
    )
    return {
        "selected_candidate_row_count": selected_candidate_row_count,
        "query_hour_count": query_hours_df.height,
        "short_query_hour_count": sum(
            values["shortfall_count"] > 0 for values in hourly_stats.values()
        ),
        "total_shortfall_count": sum(
            values["shortfall_count"] for values in hourly_stats.values()
        ),
        "popular_selected_count": sum(
            values["popular_selected_count"] for values in hourly_stats.values()
        ),
        "random_selected_count": sum(
            values["random_selected_count"] for values in hourly_stats.values()
        ),
        "hourly_selection_stats": hourly_stats,
    }


def build_negative_post_uris(
    *,
    hourly_candidates_path: Path,
    negative_post_uris_path: Path,
    uri_routes_path: Path,
    posts_path: Path,
    candidate_sources_path: Path,
    config: NegativeSelectionConfigLike,
) -> int:
    """Globally deduplicate selected URIs and validate them against Stage 3."""
    hourly_parts = sorted(hourly_candidates_path.glob("*.parquet"))
    uri_routes_path.mkdir(parents=True, exist_ok=False)
    if hourly_parts:
        # Hour-partitioned selections must be routed back by URI before a local
        # unique() operation can guarantee global post uniqueness.
        hourly_lf = pl.scan_parquet(hourly_parts)
        if hourly_lf.select(pl.len()).collect(engine="streaming").item() > 0:
            # Recreate the empty directory through the canonical partition sink.
            uri_routes_path.rmdir()
            sink_partitioned_parquet(
                hourly_lf.select("subject_uri").with_columns(
                    post_selection.post_partition_expr(config.partition_count)
                ),
                output_path=uri_routes_path,
                key="_post_partition",
            )
    negative_post_uris_path.mkdir(parents=True, exist_ok=False)
    unique_count = 0
    for partition_id in range(config.partition_count):
        routed_uris_df = read_parquet_parts(
            post_selection.partition_parquet_paths(uri_routes_path, partition_id),
            empty=negative_selection.empty_frame(
                negative_selection.NEGATIVE_POST_URI_SCHEMA
            ),
        )
        unique_uris_df = routed_uris_df.unique().sort("subject_uri")
        candidate_sources_df = read_parquet_parts(
            _public_partition_path(candidate_sources_path, partition_id),
            empty=post_selection.empty_frame(post_selection.CANDIDATE_SOURCE_SCHEMA),
        )
        posts_df = read_parquet_parts(
            _public_partition_path(posts_path, partition_id),
            empty=post_selection.empty_frame(post_selection.POST_SCHEMA),
        )
        if unique_uris_df.join(
            candidate_sources_df.select("subject_uri").unique(),
            on="subject_uri",
            how="anti",
        ).height:
            raise ValueError("Selected negatives contain a URI outside candidate_sources")
        selected_posts = unique_uris_df.join(posts_df, on="subject_uri", how="left")
        if selected_posts.get_column("post_created_at").null_count():
            raise ValueError("Selected negatives contain a URI missing from posts")
        if selected_posts.filter(pl.col("is_reply")).height:
            raise ValueError("Selected negatives contain a reply")
        write_parquet_part_if_not_empty(
            unique_uris_df,
            negative_post_uris_path / f"part-{partition_id:05d}.parquet",
        )
        unique_count += unique_uris_df.height
    ensure_typed_parquet_dataset(
        negative_post_uris_path,
        negative_selection.NEGATIVE_POST_URI_SCHEMA,
    )
    return unique_count
