"""Bounded disk-partition processing for the Stage 3 post universe."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Protocol

import polars as pl

from engagement_prediction.data import partition_workers
from engagement_prediction.data import post_selection as post_data
from engagement_prediction.data.parquet import (
    ensure_typed_parquet_dataset,
    read_parquet_parts,
    sink_partitioned_parquet,
    write_parquet_part_if_not_empty,
)


class PostSelectionConfig(Protocol):
    """Structural settings required by the Stage 3 artifact helpers."""

    random_candidate_sampling_fraction: float
    random_seed: int


def materialize_required_rows(
    *,
    query_positives_lf: pl.LazyFrame,
    history_post_uris_lf: pl.LazyFrame,
    output_path: Path,
    partition_count: int,
) -> None:
    """Route positive/history requirement rows by URI without a global union.

    Role flags are deliberately not deduplicated here. Once all copies of a URI
    are colocated, :func:`build_required_posts` can combine them locally and keep
    both flags when a post is used as a positive and as history.
    """
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
    sink_partitioned_parquet(
        required_rows_lf.with_columns(post_data.post_partition_expr(partition_count)),
        output_path=output_path,
        key="_post_partition",
    )


def _process_uri_partition(
    *,
    required_rows_path: Path,
    post_metadata_path: Path,
    posts_path: Path,
    required_posts_path: Path,
    candidate_sources_path: Path,
    missing_required_posts_path: Path,
    random_candidate_sampling_fraction: float,
    random_seed: int,
    partition_id: int,
    partition_count: int,
) -> dict[str, Any]:
    """Resolve and write one independently owned Stage 3 URI partition.

    Required posts are the lossless positive/history universe. The public
    ``posts`` table is their resolved metadata unioned with the sampled root-post
    reservoir; unresolved histories are reported separately, while a missing or
    reply-only positive is a fatal lineage-integrity error.
    """

    partition_started = time.monotonic()
    required_rows_df = read_parquet_parts(
        post_data.partition_parquet_paths(required_rows_path, partition_id),
        empty=post_data.empty_frame(post_data.REQUIRED_POST_SCHEMA),
    )
    required_posts_df = post_data.build_required_posts(required_rows_df)
    metadata_file = post_metadata_path / f"part-{partition_id:05d}.parquet"
    resolved_posts_df = read_parquet_parts(
        [metadata_file] if metadata_file.exists() else [],
        empty=post_data.empty_frame(post_data.POST_SCHEMA),
    )
    root_posts_df = resolved_posts_df.filter(~pl.col("is_reply"))
    reply_posts_df = resolved_posts_df.filter(pl.col("is_reply"))

    required_found_with_metadata_df = resolved_posts_df.join(
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
    missing_positive_count = missing_required_df.filter(pl.col("is_positive")).height
    reply_positive_count = required_found_with_metadata_df.filter(
        pl.col("is_positive") & pl.col("is_reply")
    ).height
    if missing_positive_count:
        raise ValueError(
            f"{missing_positive_count} required positive posts are absent from the exact root snapshot"
        )
    if reply_positive_count:
        raise ValueError(
            f"{reply_positive_count} required positive posts resolved only as replies"
        )

    random_posts_df = root_posts_df.filter(
        post_data.random_candidate_expr(
            random_candidate_sampling_fraction,
            random_seed,
        )
    )
    required_found_posts_df = required_found_with_metadata_df.select(
        post_data.POST_COLUMNS
    )
    posts_df = (
        pl.concat([required_found_posts_df, random_posts_df])
        .unique(subset="subject_uri")
        .sort("subject_uri")
    )
    candidate_sources_df = random_posts_df.select(
        "subject_uri",
        pl.lit("random").alias("candidate_source"),
    ).sort(["subject_uri", "candidate_source"])

    post_data.validate_public_partition(
        posts_df=posts_df,
        required_posts_df=required_posts_df,
        candidate_sources_df=candidate_sources_df,
        missing_required_posts_df=missing_required_df,
        partition_id=partition_id,
        partition_count=partition_count,
    )
    write_parquet_part_if_not_empty(
        posts_df,
        posts_path / f"part-{partition_id:05d}.parquet",
    )
    write_parquet_part_if_not_empty(
        required_posts_df,
        required_posts_path / f"part-{partition_id:05d}.parquet",
    )
    write_parquet_part_if_not_empty(
        candidate_sources_df,
        candidate_sources_path / f"part-{partition_id:05d}.parquet",
    )
    write_parquet_part_if_not_empty(
        missing_required_df,
        missing_required_posts_path / f"part-{partition_id:05d}.parquet",
    )

    required_counts = {
        "required_post_count": required_posts_df.height,
        "positive_required_post_count": required_posts_df.filter(
            pl.col("is_positive")
        ).height,
        "history_required_post_count": required_posts_df.filter(
            pl.col("is_history")
        ).height,
        "positive_and_history_required_post_count": required_posts_df.filter(
            pl.col("is_positive") & pl.col("is_history")
        ).height,
        "found_required_post_count": found_required_df.height,
        "missing_required_post_count": missing_required_df.height,
        "found_positive_required_post_count": found_required_df.filter(
            pl.col("is_positive")
        ).height,
        "found_history_required_post_count": found_required_df.filter(
            pl.col("is_history")
        ).height,
        "missing_positive_required_post_count": missing_positive_count,
        "missing_history_required_post_count": missing_required_df.filter(
            pl.col("is_history")
        ).height,
        "history_resolved_as_root_count": required_found_with_metadata_df.filter(
            pl.col("is_history") & ~pl.col("is_reply")
        ).height,
        "history_resolved_as_reply_count": required_found_with_metadata_df.filter(
            pl.col("is_history") & pl.col("is_reply")
        ).height,
        "root_reply_overlap_count": 0,
    }
    output_counts = {
        "post_count": posts_df.height,
        "root_post_count": posts_df.filter(~pl.col("is_reply")).height,
        "reply_post_count": posts_df.filter(pl.col("is_reply")).height,
        "candidate_source_count": candidate_sources_df.height,
        "random_candidate_count": candidate_sources_df.height,
    }
    return {
        "partition_id": partition_id,
        "required_post_stats": required_counts,
        "output_stats": output_counts,
        "canonical_root_count": root_posts_df.height,
        "canonical_reply_count": reply_posts_df.height,
        "runtime_seconds": time.monotonic() - partition_started,
    }


def _merge_counts(results: list[dict[str, Any]], key: str) -> dict[str, int]:
    """Sum like-named counters returned by independent partitions."""

    merged: dict[str, int] = {}
    for result in results:
        for name, value in result[key].items():
            merged[name] = merged.get(name, 0) + int(value)
    return merged


def process_uri_partitions(
    *,
    required_rows_path: Path,
    post_metadata_path: Path,
    posts_path: Path,
    required_posts_path: Path,
    candidate_sources_path: Path,
    missing_required_posts_path: Path,
    config: PostSelectionConfig,
    partition_count: int,
    worker_count: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Resolve, validate, and write one public URI partition at a time."""
    for path in (
        posts_path,
        required_posts_path,
        candidate_sources_path,
        missing_required_posts_path,
    ):
        path.mkdir(parents=True, exist_ok=False)

    processing_started = time.monotonic()
    logger.info(
        "Processing %s URI partitions with up to %s worker processes",
        partition_count,
        worker_count,
    )

    def log_result(result: dict[str, Any]) -> None:
        logger.info(
            "Resolved URI partition %s/%s in %.1fs: roots=%s replies=%s "
            "required=%s missing_history=%s random=%s",
            result["partition_id"] + 1,
            partition_count,
            result["runtime_seconds"],
            f"{result['canonical_root_count']:,}",
            f"{result['canonical_reply_count']:,}",
            f"{result['required_post_stats']['required_post_count']:,}",
            f"{result['required_post_stats']['missing_history_required_post_count']:,}",
            f"{result['output_stats']['random_candidate_count']:,}",
        )

    results, effective_worker_count = partition_workers.run_partition_jobs(
        worker=_process_uri_partition,
        worker_kwargs=[
            {
                "required_rows_path": required_rows_path,
                "post_metadata_path": post_metadata_path,
                "posts_path": posts_path,
                "required_posts_path": required_posts_path,
                "candidate_sources_path": candidate_sources_path,
                "missing_required_posts_path": missing_required_posts_path,
                "random_candidate_sampling_fraction": (
                    config.random_candidate_sampling_fraction
                ),
                "random_seed": config.random_seed,
                "partition_id": partition_id,
                "partition_count": partition_count,
            }
            for partition_id in range(partition_count)
        ],
        worker_count=worker_count,
        on_result=log_result,
    )
    required_counts = _merge_counts(results, "required_post_stats")
    output_counts = _merge_counts(results, "output_stats")

    logger.info(
        "Finished all URI partitions in %.1fs",
        time.monotonic() - processing_started,
    )
    for path, schema in (
        (posts_path, post_data.POST_SCHEMA),
        (required_posts_path, post_data.REQUIRED_POST_SCHEMA),
        (candidate_sources_path, post_data.CANDIDATE_SOURCE_SCHEMA),
        (missing_required_posts_path, post_data.REQUIRED_POST_SCHEMA),
    ):
        ensure_typed_parquet_dataset(path, schema)
    return {
        "required_post_stats": required_counts,
        "output_stats": output_counts,
        "partition_worker_count": effective_worker_count,
        "partition_stats": [
            {
                "partition_id": result["partition_id"],
                "runtime_seconds": result["runtime_seconds"],
                **result["required_post_stats"],
                **result["output_stats"],
            }
            for result in results
        ],
    }
