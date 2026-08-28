"""Stage 3: resolve required posts and build a random root-post reservoir."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict

import polars as pl

from engagement_prediction.data import (
    ingex,
    post_selection as post_data,
    post_selection_artifacts,
    timestamps,
)
from engagement_prediction.data.parquet import find_artifact_path, scan_parquet_artifact
from engagement_prediction.data.source_metadata_artifacts import (
    load_source_metadata_artifact,
)
from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline.artifacts import PartialArtifactBundle
from engagement_prediction.pipeline.lineage import resolve_recorded_stage_lineage
from engagement_prediction.pipeline.logging import get_stage_logger


@dataclass(frozen=True)
class PostSelectionConfig:
    """Validated common source window and deterministic sampling settings."""

    gcs_bucket: str
    posts_start: datetime
    posts_end: datetime
    random_candidate_sampling_fraction: float
    random_seed: int
    data_partition_worker_count: int


def build_config(args: argparse.Namespace) -> PostSelectionConfig:
    """Parse Stage 3 settings; the physical partition count belongs to Stage 00."""

    posts_start = timestamps.parse_utc_datetime(args.posts_start, field_name="posts_start")
    posts_end = timestamps.parse_utc_datetime(args.posts_end, field_name="posts_end")
    if posts_start is None or posts_end is None:
        raise ValueError("posts_start and posts_end are required for post_selection")
    timestamps.validate_half_open_utc_window(
        start=posts_start,
        end=posts_end,
        start_field_name="posts_start",
        end_field_name="posts_end",
    )
    fraction = float(args.random_candidate_sampling_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("random_candidate_sampling_fraction must be between 0 and 1")
    worker_count = int(args.data_partition_worker_count)
    if worker_count <= 0:
        raise ValueError("data_partition_worker_count must be positive")
    return PostSelectionConfig(
        gcs_bucket=str(args.gcs_bucket),
        posts_start=posts_start,
        posts_end=posts_end,
        random_candidate_sampling_fraction=fraction,
        random_seed=int(args.random_seed),
        data_partition_worker_count=worker_count,
    )


def _validate_query_window(
    queries_lf: pl.LazyFrame,
    config: PostSelectionConfig,
) -> tuple[datetime, datetime]:
    schema = queries_lf.collect_schema()
    if "query_hour" not in schema:
        raise ValueError("Stage 1 queries artifact is missing query_hour")
    dtype = schema["query_hour"]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
        raise ValueError(f"Stage 1 query_hour must be a UTC datetime, found {dtype}")
    bounds = queries_lf.select(
        pl.col("query_hour").min().alias("min_query_hour"),
        pl.col("query_hour").max().alias("max_query_hour"),
    ).collect(engine="streaming")
    minimum = bounds.item(0, "min_query_hour")
    maximum = bounds.item(0, "max_query_hour")
    if minimum is None or maximum is None:
        raise ValueError("Stage 1 queries artifact contains no queries")
    if config.posts_start > minimum or config.posts_end <= maximum:
        raise ValueError(
            "posts_start/posts_end must cover every selected query hour: "
            f"query range is {minimum.isoformat()} to {maximum.isoformat()}"
        )
    return minimum, maximum


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    """Join requirements to Stage 00 metadata and publish the post universe."""

    out_dir = context.new_stage_dir("03_post_selection")
    logger = get_stage_logger("03_POST_SELECTION", log_file=out_dir / "stage.log")
    started_at = time.time()
    config = build_config(args)

    logger.info("Phase 1/4: resolving Stage 00-2 lineage and public inputs")
    lineage = resolve_recorded_stage_lineage(
        context,
        terminal_stage_folder="02_user_history",
        ancestor_stage_folders=("00_source_metadata", "01_query_selection"),
    )
    source_metadata_dir = lineage["00_source_metadata"]
    query_selection_dir = lineage["01_query_selection"]
    user_history_dir = lineage["02_user_history"]
    source_artifact = load_source_metadata_artifact(source_metadata_dir)
    if (
        source_artifact.post_snapshot.gcs_bucket != config.gcs_bucket
        or source_artifact.post_snapshot.start != config.posts_start
        or source_artifact.post_snapshot.end != config.posts_end
    ):
        raise ValueError(
            "Stage 3 configuration does not match the aligned Stage 00 source bucket/window"
        )
    partition_count = source_artifact.partition_count

    queries_lf = scan_parquet_artifact(find_artifact_path(query_selection_dir, "queries_"))
    min_query_hour, max_query_hour = _validate_query_window(queries_lf, config)
    query_positives_lf = scan_parquet_artifact(
        find_artifact_path(query_selection_dir, "query_positives_")
    )
    try:
        history_post_uris_path = find_artifact_path(user_history_dir, "history_post_uris_")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Stage 3 requires the new Stage 2 history_post_uris_* artifact; rerun Stage 2"
        ) from exc
    history_post_uris_lf = scan_parquet_artifact(history_post_uris_path)
    logger.info(
        "Using %s canonical Stage 00 URI partitions; no raw post/reply rescan is needed",
        partition_count,
    )

    artifact_suffix = out_dir.name
    publication = PartialArtifactBundle.create(
        output_dir=out_dir,
        bundle_name=f"post_universe_{artifact_suffix}",
        staging_name=f"_post_selection_staging_{artifact_suffix}.partial",
        dataset_schemas={
            "posts": post_data.POST_SCHEMA,
            "required_posts": post_data.REQUIRED_POST_SCHEMA,
            "candidate_sources": post_data.CANDIDATE_SOURCE_SCHEMA,
            "missing_required_posts": post_data.REQUIRED_POST_SCHEMA,
        },
    )
    bundle_path = publication.final_path
    posts_path = publication.public_path("posts")
    required_posts_path = publication.public_path("required_posts")
    candidate_sources_path = publication.public_path("candidate_sources")
    missing_required_posts_path = publication.public_path("missing_required_posts")
    post_sources_path = publication.public_path(f"post_sources_{artifact_suffix}.json")
    reply_sources_path = publication.public_path(f"reply_sources_{artifact_suffix}.json")
    ingex.write_source_manifest(post_sources_path, source_artifact.post_snapshot.manifest)
    ingex.write_source_manifest(reply_sources_path, source_artifact.reply_snapshot.manifest)

    staging_root = publication.staging_path
    required_rows_path = staging_root / "required_rows"

    logger.info("Phase 2/4: routing positive and history requirements by URI")
    post_selection_artifacts.materialize_required_rows(
        query_positives_lf=query_positives_lf,
        history_post_uris_lf=history_post_uris_lf,
        output_path=required_rows_path,
        partition_count=partition_count,
    )
    logger.info("Phase 3/4: joining requirements and sampling roots by URI partition")
    selection_stats = post_selection_artifacts.process_uri_partitions(
        required_rows_path=required_rows_path,
        post_metadata_path=source_artifact.post_metadata_path,
        posts_path=posts_path,
        required_posts_path=required_posts_path,
        candidate_sources_path=candidate_sources_path,
        missing_required_posts_path=missing_required_posts_path,
        config=config,
        partition_count=partition_count,
        worker_count=config.data_partition_worker_count,
        logger=logger,
    )
    required_stats = selection_stats["required_post_stats"]
    required_stats["root_reply_overlap_count"] = int(
        source_artifact.summary["index"]["root_reply_overlap_count"]
    )
    output_stats = selection_stats["output_stats"]

    logger.info("Phase 4/4: validating and atomically publishing the bundle")
    final_posts_path = bundle_path / "posts"
    final_required_posts_path = bundle_path / "required_posts"
    final_candidate_sources_path = bundle_path / "candidate_sources"
    final_missing_required_posts_path = bundle_path / "missing_required_posts"
    final_post_sources_path = bundle_path / post_sources_path.name
    final_reply_sources_path = bundle_path / reply_sources_path.name

    runtime_seconds = time.time() - started_at
    source_index = source_artifact.summary["index"]
    summary = {
        "gcs_bucket": config.gcs_bucket,
        "posts_start": config.posts_start.isoformat(),
        "posts_end": config.posts_end.isoformat(),
        "parameters": {
            "random_candidate_sampling_fraction": config.random_candidate_sampling_fraction,
            "source_metadata_partition_count": partition_count,
            "data_partition_worker_count": config.data_partition_worker_count,
            "random_seed": config.random_seed,
        },
        "input": {
            "source_metadata_dir": str(source_metadata_dir),
            "query_selection_dir": str(query_selection_dir),
            "user_history_dir": str(user_history_dir),
            "query_range": {
                "min_query_hour": min_query_hour.isoformat(),
                "max_query_hour": max_query_hour.isoformat(),
            },
            "post_file_count": len(source_artifact.post_snapshot.file_uris),
            "reply_file_count": len(source_artifact.reply_snapshot.file_uris),
            "root_source_stats": source_index["root_source_stats"],
            "reply_source_stats": source_index["reply_source_stats"],
        },
        "required_post_stats": required_stats,
        "partition_processing": {
            "partition_worker_count": selection_stats["partition_worker_count"],
            "partition_stats": selection_stats["partition_stats"],
        },
        "outputs": {
            "post_universe_path": bundle_path.name,
            "posts_path": str(Path(bundle_path.name) / "posts"),
            "required_posts_path": str(Path(bundle_path.name) / "required_posts"),
            "candidate_sources_path": str(Path(bundle_path.name) / "candidate_sources"),
            "missing_required_posts_path": str(
                Path(bundle_path.name) / "missing_required_posts"
            ),
            "post_sources_path": str(Path(bundle_path.name) / final_post_sources_path.name),
            "reply_sources_path": str(Path(bundle_path.name) / final_reply_sources_path.name),
            **output_stats,
        },
        "runtime_seconds": runtime_seconds,
    }
    stage_info = "\n".join([
            "stage: post_selection",
            f"runtime_seconds: {runtime_seconds:.2f}",
            f"source_metadata_partition_count: {partition_count}",
            f"data_partition_worker_count: {config.data_partition_worker_count}",
            f"effective_partition_worker_count: {selection_stats['partition_worker_count']}",
            f"post_count: {output_stats['post_count']}",
            f"root_post_count: {output_stats['root_post_count']}",
            f"reply_post_count: {output_stats['reply_post_count']}",
            f"random_candidate_count: {output_stats['random_candidate_count']}",
            f"missing_history_post_count: {required_stats['missing_history_required_post_count']}",
            f"post_sources_path: {Path(bundle_path.name) / final_post_sources_path.name}",
            f"reply_sources_path: {Path(bundle_path.name) / final_reply_sources_path.name}",
        ]) + "\n"
    publication.publish(
        summary=summary,
        stage_info=stage_info,
    )
    logger.info(
        "Post selection completed in %.2fs: posts=%s roots=%s replies=%s random=%s",
        runtime_seconds,
        f"{output_stats['post_count']:,}",
        f"{output_stats['root_post_count']:,}",
        f"{output_stats['reply_post_count']:,}",
        f"{output_stats['random_candidate_count']:,}",
    )
    return {
        "output_dir": out_dir,
        "artifacts": {
            "post_universe_path": str(bundle_path),
            "posts_path": str(final_posts_path),
            "required_posts_path": str(final_required_posts_path),
            "candidate_sources_path": str(final_candidate_sources_path),
            "missing_required_posts_path": str(final_missing_required_posts_path),
            "post_sources_path": str(final_post_sources_path),
            "reply_sources_path": str(final_reply_sources_path),
        },
    }
