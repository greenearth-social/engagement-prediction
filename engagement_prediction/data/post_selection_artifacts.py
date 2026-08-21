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
    reply_paths: list[str],
    config: PostSelectionConfig,
    normalized_posts_path: Path,
    normalized_replies_path: Path,
    logger: logging.Logger,
) -> None:
    """Normalize and URI-partition exact root-post and reply snapshots."""
    for label, paths, output_path, is_reply in (
        ("root-post", post_paths, normalized_posts_path, False),
        ("reply", reply_paths, normalized_replies_path, True),
    ):
        started = time.monotonic()
        logger.info(
            "Scanning and stream-sinking %s %s source files into URI partitions",
            f"{len(paths):,}",
            label,
        )
        normalized_lf = post_data.normalize_posts(
            ingex.scan_parquet_files(paths),
            posts_start=config.posts_start,
            posts_end=config.posts_end,
            is_reply=is_reply,
        ).with_columns(post_data.post_partition_expr(config.post_selection_partition_count))
        sink_partitioned(
            normalized_lf,
            output_path=output_path,
            key="_post_partition",
        )
        logger.info(
            "Finished partitioning %s source rows in %.1fs",
            label,
            time.monotonic() - started,
        )


def process_uri_partitions(
    *,
    required_rows_path: Path,
    normalized_posts_path: Path,
    normalized_replies_path: Path,
    posts_path: Path,
    required_posts_path: Path,
    candidate_sources_path: Path,
    missing_required_posts_path: Path,
    config: PostSelectionConfig,
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

    root_stats: list[dict[str, int]] = []
    reply_stats: list[dict[str, int]] = []
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
        "history_resolved_as_root_count": 0,
        "history_resolved_as_reply_count": 0,
        "root_reply_overlap_count": 0,
    }
    output_counts = {
        "post_count": 0,
        "root_post_count": 0,
        "reply_post_count": 0,
        "candidate_source_count": 0,
        "random_candidate_count": 0,
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
        # de-duped required posts [subject_uri, is_positive, is_history]
        required_posts_df = post_data.build_required_posts(required_rows_df)
        normalized_root_df = _load_paths(
            post_data.partition_parquet_paths(normalized_posts_path, partition_id),
            post_data.NORMALIZED_POST_SCHEMA,
        )
        normalized_reply_df = _load_paths(
            post_data.partition_parquet_paths(normalized_replies_path, partition_id),
            post_data.NORMALIZED_POST_SCHEMA,
        )
        # de-duped raw posts and replies
        root_posts_df, partition_root_stats = post_data.select_latest_post_rows(
            normalized_root_df
        )
        reply_posts_df, partition_reply_stats = post_data.select_latest_post_rows(
            normalized_reply_df
        )
        root_stats.append(partition_root_stats)
        reply_stats.append(partition_reply_stats)
        # handle posts that are in both root and reply dfs, combine [subject_uri, post_created_at, author_did, is_reply]
        resolved_posts_df, overlap_count = post_data.resolve_root_and_reply_posts(
            root_posts_df,
            reply_posts_df,
        )

        # [subject_uri, post_created_at, author_did, is_reply, is_positive, is_history]
        required_found_with_metadata_df = resolved_posts_df.join(
            required_posts_df,
            on="subject_uri",
            how="inner",
        )
        # [subject_uri, is_positive, is_history]
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
                config.random_candidate_sampling_fraction,
                config.random_seed,
            )
        )
        # [subject_uri, post_created_at, author_did, is_reply]
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

        post_data.validate_public_partition(
            posts_df=base_posts_df,
            required_posts_df=required_posts_df,
            candidate_sources_df=random_candidate_sources_df,
            missing_required_posts_df=missing_required_df,
            partition_id=partition_id,
            partition_count=config.post_selection_partition_count,
        )
        _write_if_not_empty(
            base_posts_df,
            posts_path / f"part-{partition_id:05d}.parquet",
        )
        _write_if_not_empty(
            required_posts_df,
            required_posts_path / f"part-{partition_id:05d}.parquet",
        )
        _write_if_not_empty(
            random_candidate_sources_df,
            candidate_sources_path / f"part-{partition_id:05d}.parquet",
        )
        _write_if_not_empty(
            missing_required_df,
            missing_required_posts_path / f"part-{partition_id:05d}.parquet",
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
        required_counts["missing_positive_required_post_count"] += missing_positive_count
        required_counts["missing_history_required_post_count"] += missing_required_df.filter(
            pl.col("is_history")
        ).height
        required_counts["history_resolved_as_root_count"] += (
            required_found_with_metadata_df.filter(
                pl.col("is_history") & ~pl.col("is_reply")
            ).height
        )
        required_counts["history_resolved_as_reply_count"] += (
            required_found_with_metadata_df.filter(
                pl.col("is_history") & pl.col("is_reply")
            ).height
        )
        required_counts["root_reply_overlap_count"] += overlap_count
        output_counts["post_count"] += base_posts_df.height
        output_counts["root_post_count"] += base_posts_df.filter(
            ~pl.col("is_reply")
        ).height
        output_counts["reply_post_count"] += base_posts_df.filter(
            pl.col("is_reply")
        ).height
        output_counts["candidate_source_count"] += random_candidate_sources_df.height
        output_counts["random_candidate_count"] += random_candidate_sources_df.height
        logger.info(
            "Resolved URI partition %s/%s in %.1fs: roots=%s replies=%s "
            "required=%s missing_history=%s random=%s overlaps=%s",
            partition_id + 1,
            config.post_selection_partition_count,
            time.monotonic() - partition_started,
            f"{root_posts_df.height:,}",
            f"{reply_posts_df.height:,}",
            f"{required_posts_df.height:,}",
            f"{missing_required_df.height:,}",
            f"{random_posts_df.height:,}",
            f"{overlap_count:,}",
        )

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
        _ensure_nonempty_dataset(path, schema)
    return {
        "root_source_stats": _merge_numeric_stats(root_stats),
        "reply_source_stats": _merge_numeric_stats(reply_stats),
        "required_post_stats": required_counts,
        "output_stats": output_counts,
    }
