"""Bounded disk-partition processing for the Stage 3 post universe."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Any, Protocol

import polars as pl

from engagement_prediction.data import ingex
from engagement_prediction.data import post_selection as post_data


class PostSelectionConfig(Protocol):
    posts_start: datetime
    posts_end: datetime
    random_candidate_sampling_fraction: float
    max_political_candidates_per_creation_hour: int
    political_score_threshold: float
    post_selection_partition_count: int
    random_seed: int


def sink_partitioned(
    lf: pl.LazyFrame,
    *,
    output_path: Path,
    key: str,
) -> None:
    """Stream a narrow lazy frame into hive-style partitions."""
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
        return post_data.empty_frame(schema)
    return pl.read_parquet(paths)


def _write_if_not_empty(df: pl.DataFrame, path: Path) -> None:
    if not df.is_empty():
        df.write_parquet(path, compression="zstd")


def _ensure_nonempty_dataset(path: Path, schema: dict[str, pl.DataType]) -> None:
    if not list(path.rglob("*.parquet")):
        post_data.empty_frame(schema).write_parquet(
            path / "part-00000.parquet",
            compression="zstd",
        )


def _public_part(
    path: Path,
    partition_id: int,
    schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    part_path = path / f"part-{partition_id:05d}.parquet"
    if not part_path.exists():
        return post_data.empty_frame(schema)
    return pl.read_parquet(part_path)


def _merge_numeric_stats(stats: list[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for values in stats:
        for key, value in values.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def materialize_required_rows(
    *,
    query_positives_lf: pl.LazyFrame,
    history_post_uris_lf: pl.LazyFrame,
    output_path: Path,
    partition_count: int,
) -> None:
    """Route positive/history requirement rows by URI without a global union."""
    positive_schema = query_positives_lf.collect_schema()
    history_schema = history_post_uris_lf.collect_schema()
    if "subject_uri" not in positive_schema or positive_schema["subject_uri"] != pl.String:
        raise ValueError("Stage 1 query_positives.subject_uri must be String")
    if history_schema.names() != ["subject_uri"] or history_schema["subject_uri"] != pl.String:
        raise ValueError("Stage 2 history_post_uris must contain only subject_uri: String")
    required_rows_lf = pl.concat(
        [
            query_positives_lf.select(
                pl.col("subject_uri"),
                pl.lit(True).alias("is_positive"),
                pl.lit(False).alias("is_history"),
            ),
            history_post_uris_lf.select(
                pl.col("subject_uri"),
                pl.lit(False).alias("is_positive"),
                pl.lit(True).alias("is_history"),
            ),
        ]
    ).filter(pl.col("subject_uri").is_not_null())
    sink_partitioned(
        required_rows_lf.with_columns(post_data.post_partition_expr(partition_count)),
        output_path=output_path,
        key="_post_partition",
    )


def materialize_source_rows(
    *,
    post_paths: list[str],
    inference_paths: list[str],
    config: PostSelectionConfig,
    normalized_posts_path: Path,
    normalized_inferences_path: Path | None,
    logger: logging.Logger,
) -> None:
    """Normalize and route raw post and inference rows by stable URI hash."""
    posts_started = time.monotonic()
    logger.info(
        "Scanning and stream-sinking %s post source files into URI partitions",
        f"{len(post_paths):,}",
    )
    normalized_posts_lf = post_data.normalize_posts(
        ingex.scan_parquet_files(post_paths),
        posts_start=config.posts_start,
        posts_end=config.posts_end,
    ).with_columns(post_data.post_partition_expr(config.post_selection_partition_count))
    sink_partitioned(
        normalized_posts_lf,
        output_path=normalized_posts_path,
        key="_post_partition",
    )
    logger.info(
        "Finished partitioning post source rows in %.1fs",
        time.monotonic() - posts_started,
    )
    if normalized_inferences_path is not None:
        inferences_started = time.monotonic()
        logger.info(
            "Scanning and stream-sinking %s inference source files into URI partitions",
            f"{len(inference_paths):,}",
        )
        normalized_inferences_lf = post_data.normalize_inferences(
            ingex.scan_parquet_files(inference_paths)
        ).with_columns(post_data.post_partition_expr(config.post_selection_partition_count))
        sink_partitioned(
            normalized_inferences_lf,
            output_path=normalized_inferences_path,
            key="_post_partition",
        )
        logger.info(
            "Finished partitioning inference source rows in %.1fs",
            time.monotonic() - inferences_started,
        )


def process_uri_partitions(
    *,
    required_rows_path: Path,
    normalized_posts_path: Path,
    normalized_inferences_path: Path | None,
    required_posts_path: Path,
    missing_required_posts_path: Path,
    base_posts_shards_path: Path,
    random_candidate_shards_path: Path,
    political_eligible_shards_path: Path,
    config: PostSelectionConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Resolve metadata and candidate membership one bounded URI partition at a time."""
    for path in (
        required_posts_path,
        missing_required_posts_path,
        base_posts_shards_path,
        random_candidate_shards_path,
        political_eligible_shards_path,
    ):
        path.mkdir(parents=True, exist_ok=False)

    post_stats: list[dict[str, int]] = []
    inference_stats: list[dict[str, int]] = []
    required_counts = {
        "required_post_count": 0,
        "positive_required_post_count": 0,
        "history_required_post_count": 0,
        "positive_and_history_required_post_count": 0,
        "found_required_post_count": 0,
        "missing_required_post_count": 0,
        "found_positive_required_post_count": 0,
        "found_history_required_post_count": 0,
        "missing_positive_required_post_count": 0,
        "missing_history_required_post_count": 0,
    }
    processing_started = time.monotonic()
    logger.info(
        "Beginning bounded processing of %s URI partitions",
        config.post_selection_partition_count,
    )

    for partition_id in range(config.post_selection_partition_count):
        partition_started = time.monotonic()
        logger.info(
            "Processing URI partition %s/%s",
            partition_id + 1,
            config.post_selection_partition_count,
        )
        required_rows_df = _load_paths(
            post_data.partition_parquet_paths(required_rows_path, partition_id),
            post_data.REQUIRED_POST_SCHEMA,
        )
        required_posts_df = post_data.build_required_posts(required_rows_df)
        normalized_posts_df = _load_paths(
            post_data.partition_parquet_paths(normalized_posts_path, partition_id),
            post_data.NORMALIZED_POST_SCHEMA,
        )
        unique_posts_df, partition_post_stats = post_data.select_latest_post_rows(
            normalized_posts_df
        )
        post_stats.append(partition_post_stats)

        if normalized_inferences_path is None:
            latest_inferences_df = post_data.empty_frame(post_data.LATEST_INFERENCE_SCHEMA)
        else:
            normalized_inferences_df = _load_paths(
                post_data.partition_parquet_paths(
                    normalized_inferences_path,
                    partition_id,
                ),
                post_data.NORMALIZED_INFERENCE_SCHEMA,
            )
            latest_inferences_df, partition_inference_stats = (
                post_data.select_latest_inferences(
                    normalized_inferences_df,
                    political_score_threshold=config.political_score_threshold,
                )
            )
            inference_stats.append(partition_inference_stats)
        labeled_posts_df = post_data.label_posts(unique_posts_df, latest_inferences_df)

        required_found_with_metadata_df = labeled_posts_df.join(
            required_posts_df,
            on="subject_uri",
            how="inner",
        )
        found_required_df = required_found_with_metadata_df.select(
            post_data.REQUIRED_POST_COLUMNS
        )
        missing_required_df = required_posts_df.join(
            found_required_df.select("subject_uri"), on="subject_uri", how="anti"
        ).sort("subject_uri")
        _write_if_not_empty(
            required_posts_df,
            required_posts_path / f"part-{partition_id:05d}.parquet",
        )
        _write_if_not_empty(
            missing_required_df,
            missing_required_posts_path / f"part-{partition_id:05d}.parquet",
        )

        random_posts_df = labeled_posts_df.filter(
            post_data.random_candidate_expr(
                config.random_candidate_sampling_fraction,
                config.random_seed,
            )
        )
        required_found_posts_df = required_found_with_metadata_df.select(
            post_data.POST_COLUMNS
        )
        base_posts_df = (
            pl.concat([required_found_posts_df, random_posts_df])
            .unique(subset="subject_uri")
            .sort("subject_uri")
        )
        random_candidate_sources_df = random_posts_df.select(
            "subject_uri",
            pl.lit("random").alias("candidate_source"),
        ).sort(["subject_uri", "candidate_source"])
        political_eligible_df = labeled_posts_df.filter(pl.col("is_political")).with_columns(
            post_data.political_priority_expr(config.random_seed)
        )
        _write_if_not_empty(
            base_posts_df,
            base_posts_shards_path / f"part-{partition_id:05d}.parquet",
        )
        _write_if_not_empty(
            random_candidate_sources_df,
            random_candidate_shards_path / f"part-{partition_id:05d}.parquet",
        )
        _write_if_not_empty(
            political_eligible_df,
            political_eligible_shards_path / f"part-{partition_id:05d}.parquet",
        )

        required_counts["required_post_count"] += required_posts_df.height
        required_counts["positive_required_post_count"] += required_posts_df.filter(
            pl.col("is_positive")
        ).height
        required_counts["history_required_post_count"] += required_posts_df.filter(
            pl.col("is_history")
        ).height
        required_counts["positive_and_history_required_post_count"] += required_posts_df.filter(
            pl.col("is_positive") & pl.col("is_history")
        ).height
        required_counts["found_required_post_count"] += found_required_df.height
        required_counts["missing_required_post_count"] += missing_required_df.height
        required_counts["found_positive_required_post_count"] += found_required_df.filter(
            pl.col("is_positive")
        ).height
        required_counts["found_history_required_post_count"] += found_required_df.filter(
            pl.col("is_history")
        ).height
        required_counts["missing_positive_required_post_count"] += missing_required_df.filter(
            pl.col("is_positive")
        ).height
        required_counts["missing_history_required_post_count"] += missing_required_df.filter(
            pl.col("is_history")
        ).height
        logger.info(
            "Resolved URI partition %s/%s in %.1fs: source_rows=%s posts=%s "
            "required=%s missing=%s random=%s political_eligible=%s",
            partition_id + 1,
            config.post_selection_partition_count,
            time.monotonic() - partition_started,
            f"{normalized_posts_df.height:,}",
            f"{unique_posts_df.height:,}",
            f"{required_posts_df.height:,}",
            f"{missing_required_df.height:,}",
            f"{random_posts_df.height:,}",
            f"{political_eligible_df.height:,}",
        )

    logger.info(
        "Finished all URI partitions in %.1fs",
        time.monotonic() - processing_started,
    )
    return {
        "post_source_stats": _merge_numeric_stats(post_stats),
        "inference_source_stats": _merge_numeric_stats(inference_stats),
        "required_post_stats": required_counts,
    }


def select_political_candidates(
    *,
    political_eligible_shards_path: Path,
    political_by_date_path: Path,
    selected_political_shards_path: Path,
    config: PostSelectionConfig,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """Repartition eligible posts by day, then apply each hourly cap in memory."""
    selected_political_shards_path.mkdir(parents=True, exist_ok=False)
    eligible_paths = sorted(political_eligible_shards_path.glob("*.parquet"))
    if not eligible_paths or config.max_political_candidates_per_creation_hour == 0:
        logger.info("No political-candidate capping work is required")
        return []
    repartition_started = time.monotonic()
    logger.info(
        "Repartitioning %s political-eligible URI shards by creation date",
        f"{len(eligible_paths):,}",
    )
    sink_partitioned(
        pl.scan_parquet(eligible_paths).with_columns(
            pl.col("post_created_at").dt.strftime("%Y-%m-%d").alias("_creation_date")
        ),
        output_path=political_by_date_path,
        key="_creation_date",
    )
    logger.info(
        "Finished political creation-date repartitioning in %.1fs",
        time.monotonic() - repartition_started,
    )
    hour_stats: list[dict[str, Any]] = []
    date_dirs = sorted(
        path for path in political_by_date_path.iterdir() if path.is_dir()
    )
    logger.info("Applying hourly political caps across %s creation dates", len(date_dirs))
    for date_index, date_dir in enumerate(date_dirs):
        date_started = time.monotonic()
        logger.info(
            "Processing political creation date %s/%s (%s)",
            date_index + 1,
            len(date_dirs),
            date_dir.name,
        )
        eligible_df = pl.read_parquet(sorted(date_dir.rglob("*.parquet")))
        selected_df, date_hour_stats = post_data.select_political_candidates_for_day(
            eligible_df,
            max_candidates_per_creation_hour=(
                config.max_political_candidates_per_creation_hour
            ),
        )
        _write_if_not_empty(
            selected_df,
            selected_political_shards_path / f"part-{date_index:05d}.parquet",
        )
        hour_stats.extend(date_hour_stats)
        logger.info(
            "Capped political candidates for date %s/%s in %.1fs: eligible=%s selected=%s",
            date_index + 1,
            len(date_dirs),
            time.monotonic() - date_started,
            f"{eligible_df.height:,}",
            f"{selected_df.height:,}",
        )
    return sorted(hour_stats, key=lambda row: row["post_created_hour"])


def materialize_final_routes(
    *,
    base_posts_shards_path: Path,
    random_candidate_shards_path: Path,
    selected_political_shards_path: Path,
    final_posts_routed_path: Path,
    final_candidates_routed_path: Path,
    partition_count: int,
    logger: logging.Logger,
) -> None:
    """Route the final union and candidate-source rows back to URI partitions."""
    base_post_paths = sorted(base_posts_shards_path.glob("*.parquet"))
    political_post_paths = sorted(selected_political_shards_path.glob("*.parquet"))
    post_lfs = [
        pl.scan_parquet(paths).select(post_data.POST_COLUMNS)
        for paths in (base_post_paths, political_post_paths)
        if paths
    ]
    if post_lfs:
        posts_started = time.monotonic()
        logger.info(
            "Routing final post rows from %s base and %s political shards",
            f"{len(base_post_paths):,}",
            f"{len(political_post_paths):,}",
        )
        sink_partitioned(
            pl.concat(post_lfs).with_columns(post_data.post_partition_expr(partition_count)),
            output_path=final_posts_routed_path,
            key="_post_partition",
        )
        logger.info(
            "Finished routing final post rows in %.1fs",
            time.monotonic() - posts_started,
        )
    else:
        logger.info("No final post rows to route; creating an empty routed dataset")
        final_posts_routed_path.mkdir(parents=True, exist_ok=False)

    random_source_paths = sorted(random_candidate_shards_path.glob("*.parquet"))
    candidate_lfs = []
    if random_source_paths:
        candidate_lfs.append(pl.scan_parquet(random_source_paths))
    if political_post_paths:
        candidate_lfs.append(
            pl.scan_parquet(political_post_paths).select(
                "subject_uri",
                pl.lit("political").alias("candidate_source"),
            )
        )
    if candidate_lfs:
        candidates_started = time.monotonic()
        logger.info(
            "Routing final candidate-source rows from %s random and %s political shards",
            f"{len(random_source_paths):,}",
            f"{len(political_post_paths):,}",
        )
        sink_partitioned(
            pl.concat(candidate_lfs).with_columns(
                post_data.post_partition_expr(partition_count)
            ),
            output_path=final_candidates_routed_path,
            key="_post_partition",
        )
        logger.info(
            "Finished routing final candidate-source rows in %.1fs",
            time.monotonic() - candidates_started,
        )
    else:
        logger.info("No candidate-source rows to route; creating an empty routed dataset")
        final_candidates_routed_path.mkdir(parents=True, exist_ok=False)


def write_and_validate_public_outputs(
    *,
    final_posts_routed_path: Path,
    final_candidates_routed_path: Path,
    posts_path: Path,
    required_posts_path: Path,
    candidate_sources_path: Path,
    missing_required_posts_path: Path,
    config: PostSelectionConfig,
    logger: logging.Logger,
) -> dict[str, int]:
    """Publish and validate one aligned public URI partition at a time."""
    posts_path.mkdir(parents=True, exist_ok=False)
    candidate_sources_path.mkdir(parents=True, exist_ok=False)
    counts = {
        "post_count": 0,
        "candidate_source_count": 0,
        "random_candidate_count": 0,
        "political_candidate_count": 0,
        "random_and_political_candidate_count": 0,
        "inference_covered_post_count": 0,
        "political_labeled_post_count": 0,
    }
    validation_started = time.monotonic()
    for partition_id in range(config.post_selection_partition_count):
        posts_df = (
            _load_paths(
                post_data.partition_parquet_paths(
                    final_posts_routed_path,
                    partition_id,
                ),
                post_data.POST_SCHEMA,
            )
            .select(post_data.POST_COLUMNS)
            .unique(subset="subject_uri")
            .sort("subject_uri")
        )
        candidate_sources_df = (
            _load_paths(
                post_data.partition_parquet_paths(
                    final_candidates_routed_path,
                    partition_id,
                ),
                post_data.CANDIDATE_SOURCE_SCHEMA,
            )
            .select(post_data.CANDIDATE_SOURCE_COLUMNS)
            .unique()
            .sort(["subject_uri", "candidate_source"])
        )
        _write_if_not_empty(
            posts_df,
            posts_path / f"part-{partition_id:05d}.parquet",
        )
        _write_if_not_empty(
            candidate_sources_df,
            candidate_sources_path / f"part-{partition_id:05d}.parquet",
        )

        required_df = _public_part(
            required_posts_path,
            partition_id,
            post_data.REQUIRED_POST_SCHEMA,
        )
        missing_df = _public_part(
            missing_required_posts_path,
            partition_id,
            post_data.REQUIRED_POST_SCHEMA,
        )
        post_data.validate_public_partition(
            posts_df=posts_df,
            required_posts_df=required_df,
            candidate_sources_df=candidate_sources_df,
            missing_required_posts_df=missing_df,
            partition_id=partition_id,
            partition_count=config.post_selection_partition_count,
            political_score_threshold=config.political_score_threshold,
        )

        counts["post_count"] += posts_df.height
        counts["candidate_source_count"] += candidate_sources_df.height
        counts["random_candidate_count"] += candidate_sources_df.filter(
            pl.col("candidate_source") == "random"
        ).height
        counts["political_candidate_count"] += candidate_sources_df.filter(
            pl.col("candidate_source") == "political"
        ).height
        overlaps = (
            candidate_sources_df.group_by("subject_uri")
            .agg(pl.col("candidate_source").n_unique().alias("_source_count"))
            .filter(pl.col("_source_count") > 1)
        )
        counts["random_and_political_candidate_count"] += overlaps.height
        counts["inference_covered_post_count"] += posts_df.filter(
            pl.col("political_inference_indexed_at").is_not_null()
        ).height
        counts["political_labeled_post_count"] += posts_df.filter(
            pl.col("is_political")
        ).height
        completed_partition_count = partition_id + 1
        if (
            completed_partition_count == 1
            or completed_partition_count == config.post_selection_partition_count
            or completed_partition_count % 16 == 0
        ):
            logger.info(
                "Wrote and validated public partition %s/%s: cumulative_posts=%s "
                "cumulative_candidate_sources=%s",
                completed_partition_count,
                config.post_selection_partition_count,
                f"{counts['post_count']:,}",
                f"{counts['candidate_source_count']:,}",
            )

    for path, schema in (
        (posts_path, post_data.POST_SCHEMA),
        (required_posts_path, post_data.REQUIRED_POST_SCHEMA),
        (candidate_sources_path, post_data.CANDIDATE_SOURCE_SCHEMA),
        (missing_required_posts_path, post_data.REQUIRED_POST_SCHEMA),
    ):
        _ensure_nonempty_dataset(path, schema)
    logger.info(
        "Finished writing and validating public datasets in %.1fs",
        time.monotonic() - validation_started,
    )
    return counts
