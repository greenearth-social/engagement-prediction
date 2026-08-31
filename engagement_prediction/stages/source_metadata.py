"""Stage 00: build the reusable canonical root/reply metadata index.

This is the only active stage that lists the raw post and reply snapshots.
Stages 1, 3, and 6 reuse its narrow canonical metadata instead of repeating
the expensive source scan. Stage 7 reuses the exact recorded file manifests
when it must return to the raw rows for embedding payloads.

The public ``post_metadata`` dataset contains one row per URI. Physical URI
partitions are an implementation detail that lets duplicate resolution and
root-over-reply precedence happen without collecting the full source window.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict

from engagement_prediction.data import (
    ingex,
    source_metadata,
    source_metadata_artifacts,
    timestamps,
)
from engagement_prediction.pipeline.artifacts import PartialArtifactBundle
from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline.logging import get_stage_logger


@dataclass(frozen=True)
class SourceMetadataConfig:
    """Validated source window, bucket, and physical partition layout."""

    gcs_bucket: str
    posts_start: datetime
    posts_end: datetime
    source_metadata_partition_count: int
    data_partition_worker_count: int


def build_config(args: argparse.Namespace) -> SourceMetadataConfig:
    """Parse the common source window and Stage 00 physical settings."""

    posts_start = timestamps.parse_utc_datetime(args.posts_start, field_name="posts_start")
    posts_end = timestamps.parse_utc_datetime(args.posts_end, field_name="posts_end")
    if posts_start is None or posts_end is None:
        raise ValueError("posts_start and posts_end are required for source_metadata")
    timestamps.validate_half_open_utc_window(
        start=posts_start,
        end=posts_end,
        start_field_name="posts_start",
        end_field_name="posts_end",
    )
    partition_count = int(args.source_metadata_partition_count)
    if partition_count <= 0:
        raise ValueError("source_metadata_partition_count must be positive")
    worker_count = int(args.data_partition_worker_count)
    if worker_count <= 0:
        raise ValueError("data_partition_worker_count must be positive")
    gcs_bucket = str(args.gcs_bucket).strip()
    if not gcs_bucket:
        raise ValueError("gcs_bucket must not be empty")
    return SourceMetadataConfig(
        gcs_bucket=gcs_bucket,
        posts_start=posts_start,
        posts_end=posts_end,
        source_metadata_partition_count=partition_count,
        data_partition_worker_count=worker_count,
    )


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    """Build and atomically publish the exact-window canonical metadata index."""

    out_dir = context.new_stage_dir("00_source_metadata")
    logger = get_stage_logger("00_SOURCE_METADATA", log_file=out_dir / "stage.log")
    started_at = time.time()
    config = build_config(args)

    logger.info(
        "Phase 1/4: listing exact root and reply snapshots for [%s, %s)",
        config.posts_start.isoformat(),
        config.posts_end.isoformat(),
    )
    post_paths, post_timestamps = ingex.list_ingex_parquet_files(
        gcs_bucket=config.gcs_bucket,
        blob_prefix="bsky_posts",
        start=config.posts_start,
        end=config.posts_end,
    )
    reply_paths, reply_timestamps = ingex.list_ingex_parquet_files(
        gcs_bucket=config.gcs_bucket,
        blob_prefix="bsky_replies",
        start=config.posts_start,
        end=config.posts_end,
    )
    if not post_paths:
        raise ValueError("No bsky_posts Parquet files found for the Stage 00 source window")
    if not reply_paths:
        raise ValueError("No bsky_replies Parquet files found for the Stage 00 source window")
    logger.info(
        "Resolved %s root files and %s reply files",
        f"{len(post_paths):,}",
        f"{len(reply_paths):,}",
    )

    artifact_suffix = out_dir.name
    # Everything beneath the public bundle is written under one partial
    # directory. The final rename is the publication boundary: downstream
    # stages cannot mistake an interrupted build for a completed artifact.
    publication = PartialArtifactBundle.create(
        output_dir=out_dir,
        bundle_name=f"source_metadata_{artifact_suffix}",
        staging_name=f"_source_metadata_staging_{artifact_suffix}.partial",
        dataset_schemas={"post_metadata": source_metadata.POST_METADATA_SCHEMA},
    )
    bundle_path = publication.final_path
    post_metadata_path = publication.public_path("post_metadata")
    post_sources_path = publication.public_path(
        f"post_sources_{artifact_suffix}.json"
    )
    reply_sources_path = publication.public_path(
        f"reply_sources_{artifact_suffix}.json"
    )
    # These manifests freeze both the exact physical files and the common
    # source window. Downstream stages validate against them rather than
    # relisting GCS, where new exports could otherwise change a rerun.
    post_manifest = ingex.build_source_manifest(
        gcs_bucket=config.gcs_bucket,
        blob_prefix="bsky_posts",
        start=config.posts_start,
        end=config.posts_end,
        paths=post_paths,
        timestamps=post_timestamps,
    )
    reply_manifest = ingex.build_source_manifest(
        gcs_bucket=config.gcs_bucket,
        blob_prefix="bsky_replies",
        start=config.posts_start,
        end=config.posts_end,
        paths=reply_paths,
        timestamps=reply_timestamps,
    )
    ingex.write_source_manifest(post_sources_path, post_manifest)
    ingex.write_source_manifest(reply_sources_path, reply_manifest)

    staging_root = publication.staging_path
    normalized_posts_path = staging_root / "normalized_posts"
    normalized_replies_path = staging_root / "normalized_replies"

    # Roots and replies remain separate while routing so they can be
    # deduplicated independently. The same URI hash sends every occurrence of
    # one URI to the same bounded partition, regardless of source file.
    logger.info("Phase 2/4: routing narrow raw metadata by URI")
    source_metadata_artifacts.materialize_source_routes(
        post_paths=post_paths,
        reply_paths=reply_paths,
        normalized_posts_path=normalized_posts_path,
        normalized_replies_path=normalized_replies_path,
        config=config,
        logger=logger,
    )
    # Each partition now contains every source row needed to choose one
    # canonical record for its URIs. If a URI exists in both sources, its root
    # row wins and the reply row is counted as an overlap.
    logger.info("Phase 3/4: deduplicating metadata and applying root precedence")
    index_stats = source_metadata_artifacts.process_uri_partitions(
        normalized_posts_path=normalized_posts_path,
        normalized_replies_path=normalized_replies_path,
        post_metadata_path=post_metadata_path,
        partition_count=config.source_metadata_partition_count,
        worker_count=config.data_partition_worker_count,
        logger=logger,
    )

    logger.info("Phase 4/4: validating and publishing the metadata bundle")
    final_metadata_path = bundle_path / "post_metadata"
    final_post_sources_path = bundle_path / post_sources_path.name
    final_reply_sources_path = bundle_path / reply_sources_path.name
    runtime_seconds = time.time() - started_at
    summary = {
        "gcs_bucket": config.gcs_bucket,
        "posts_start": config.posts_start.isoformat(),
        "posts_end": config.posts_end.isoformat(),
        "parameters": {
            "source_metadata_partition_count": config.source_metadata_partition_count,
            "data_partition_worker_count": config.data_partition_worker_count,
        },
        "source_file_counts": {
            "post_file_count": len(post_paths),
            "reply_file_count": len(reply_paths),
        },
        "index": index_stats,
        "outputs": {
            "source_metadata_path": bundle_path.name,
            "post_metadata_path": str(Path(bundle_path.name) / "post_metadata"),
            "post_sources_path": str(Path(bundle_path.name) / final_post_sources_path.name),
            "reply_sources_path": str(Path(bundle_path.name) / final_reply_sources_path.name),
        },
        "runtime_seconds": runtime_seconds,
    }
    stage_info = "\n".join([
            "stage: source_metadata",
            f"runtime_seconds: {runtime_seconds:.2f}",
            f"source_metadata_partition_count: {config.source_metadata_partition_count}",
            f"data_partition_worker_count: {config.data_partition_worker_count}",
            f"effective_partition_worker_count: {index_stats['partition_worker_count']}",
            f"canonical_record_count: {index_stats['canonical_record_count']}",
            f"canonical_root_count: {index_stats['canonical_root_count']}",
            f"canonical_reply_count: {index_stats['canonical_reply_count']}",
            f"root_reply_overlap_count: {index_stats['root_reply_overlap_count']}",
            f"post_metadata_path: {Path(bundle_path.name) / 'post_metadata'}",
            f"post_sources_path: {Path(bundle_path.name) / final_post_sources_path.name}",
            f"reply_sources_path: {Path(bundle_path.name) / final_reply_sources_path.name}",
        ]) + "\n"
    publication.publish(
        summary=summary,
        stage_info=stage_info,
    )
    logger.info(
        "Source metadata completed in %.2fs: records=%s roots=%s replies=%s",
        runtime_seconds,
        f"{index_stats['canonical_record_count']:,}",
        f"{index_stats['canonical_root_count']:,}",
        f"{index_stats['canonical_reply_count']:,}",
    )
    return {
        "output_dir": out_dir,
        "artifacts": {
            "source_metadata_path": str(bundle_path),
            "post_metadata_path": str(final_metadata_path),
            "post_sources_path": str(final_post_sources_path),
            "reply_sources_path": str(final_reply_sources_path),
        },
    }
