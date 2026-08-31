"""Bounded artifact construction for the reusable Stage 00 metadata index.

The module implements a disk-backed shuffle:

1. Stream narrow root and reply rows into stable URI-hash partitions.
2. Load one root/reply partition pair at a time.
3. Deduplicate each source, apply root precedence, and immediately publish the
   canonical partition.

This avoids a global in-memory group-by while still producing one globally
unique metadata row per URI.
"""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Protocol

import polars as pl

from engagement_prediction.data import (
    ingex,
    partition_workers,
    source_manifests,
    source_metadata,
)
from engagement_prediction.data.parquet import (
    ensure_typed_parquet_dataset,
    find_artifact_path,
    read_parquet_parts,
    sink_partitioned_parquet,
    write_parquet_part_if_not_empty,
)


class SourceMetadataConfig(Protocol):
    """Structural settings required to build the Stage 00 metadata index."""

    posts_start: datetime
    posts_end: datetime
    source_metadata_partition_count: int
    data_partition_worker_count: int


@dataclass(frozen=True)
class SourceMetadataArtifact:
    """Resolved public datasets and immutable contract recorded by Stage 00."""

    stage_dir: Path
    bundle_path: Path
    post_metadata_path: Path
    post_snapshot: source_manifests.SourceSnapshot
    reply_snapshot: source_manifests.SourceSnapshot
    partition_count: int
    summary: dict[str, Any]


def load_source_metadata_artifact(stage_dir: Path) -> SourceMetadataArtifact:
    """Load and validate the reusable Stage 00 public contract.

    Consumers receive the canonical dataset, exact immutable source
    snapshots, and the partition count as one object. Keeping these together
    prevents a downstream stage from accidentally mixing metadata from one
    run with raw files or hash partitions from another.
    """

    stage_dir = Path(stage_dir).resolve()
    bundle_path = find_artifact_path(stage_dir, "source_metadata_")
    post_metadata_path = bundle_path / "post_metadata"
    if not post_metadata_path.is_dir():
        raise FileNotFoundError(
            f"Stage 00 artifact is missing post_metadata/: {bundle_path}"
        )
    post_snapshot = source_manifests.load_source_snapshot(
        bundle_path,
        manifest_prefix="post_sources_",
        expected_blob_prefix="bsky_posts",
    )
    reply_snapshot = source_manifests.load_source_snapshot(
        bundle_path,
        manifest_prefix="reply_sources_",
        expected_blob_prefix="bsky_replies",
    )
    source_manifests.validate_aligned_source_snapshots(
        (post_snapshot, reply_snapshot),
        description="Stage 00 post and reply snapshots",
    )
    summary_path = stage_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text())
        partition_count = int(
            summary["parameters"]["source_metadata_partition_count"]
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Stage 00 summary does not record source_metadata_partition_count: {summary_path}"
        ) from exc
    if partition_count <= 0:
        raise ValueError("Stage 00 source_metadata_partition_count must be positive")
    return SourceMetadataArtifact(
        stage_dir=stage_dir,
        bundle_path=bundle_path,
        post_metadata_path=post_metadata_path,
        post_snapshot=post_snapshot,
        reply_snapshot=reply_snapshot,
        partition_count=partition_count,
        summary=summary,
    )


def _merge_numeric_stats(stats: list[dict[str, int]]) -> dict[str, int]:
    """Sum like-named source-quality counters from independent partitions."""

    merged: dict[str, int] = {}
    for values in stats:
        for key, value in values.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def materialize_source_routes(
    *,
    post_paths: list[str],
    reply_paths: list[str],
    normalized_posts_path: Path,
    normalized_replies_path: Path,
    config: SourceMetadataConfig,
    logger: logging.Logger,
) -> None:
    """Scan each exact source once and route narrow rows by stable URI hash.

    Only URI, creation time, author, source type, and validity are written.
    Large payload columns such as content embeddings never enter Stage 00's
    staging data.
    """

    for label, paths, output_path, is_reply in (
        ("root-post", post_paths, normalized_posts_path, False),
        ("reply", reply_paths, normalized_replies_path, True),
    ):
        started = time.monotonic()
        logger.info(
            "Scanning and stream-sinking %s exact %s files into %s URI partitions",
            f"{len(paths):,}",
            label,
            config.source_metadata_partition_count,
        )
        normalized_lf = source_metadata.normalize_source_records(
            ingex.scan_parquet_files(paths),
            posts_start=config.posts_start,
            posts_end=config.posts_end,
            is_reply=is_reply,
        ).with_columns(
            source_metadata.uri_partition_expr(config.source_metadata_partition_count)
        )
        sink_partitioned_parquet(
            normalized_lf,
            output_path=output_path,
            key="_post_partition",
        )
        logger.info("Finished routing %s rows in %.1fs", label, time.monotonic() - started)


def _process_uri_partition(
    *,
    normalized_posts_path: Path,
    normalized_replies_path: Path,
    post_metadata_path: Path,
    partition_id: int,
    partition_count: int,
) -> dict[str, Any]:
    """Deduplicate and publish one independently owned metadata partition."""

    started = time.monotonic()
    # These are the complete root and reply row sets for the current URI
    # range, not arbitrary file shards. It is therefore safe to resolve
    # duplicates without looking at any other partition.
    roots = read_parquet_parts(
        source_metadata.partition_parquet_paths(normalized_posts_path, partition_id),
        empty=source_metadata.empty_frame(source_metadata.NORMALIZED_METADATA_SCHEMA),
    )
    replies = read_parquet_parts(
        source_metadata.partition_parquet_paths(normalized_replies_path, partition_id),
        empty=source_metadata.empty_frame(source_metadata.NORMALIZED_METADATA_SCHEMA),
    )
    canonical_roots, root_partition_stats = (
        source_metadata.select_latest_metadata_rows(roots)
    )
    canonical_replies, reply_partition_stats = (
        source_metadata.select_latest_metadata_rows(replies)
    )
    metadata, partition_overlap_count = source_metadata.apply_root_precedence(
        canonical_roots,
        canonical_replies,
    )
    source_metadata.validate_metadata_partition(
        metadata,
        partition_id=partition_id,
        partition_count=partition_count,
    )
    write_parquet_part_if_not_empty(
        metadata,
        post_metadata_path / f"part-{partition_id:05d}.parquet",
    )

    partition_root_count = metadata.filter(~pl.col("is_reply")).height
    partition_reply_count = metadata.filter(pl.col("is_reply")).height
    return {
        "partition_id": partition_id,
        "root_source_stats": root_partition_stats,
        "reply_source_stats": reply_partition_stats,
        "canonical_record_count": metadata.height,
        "canonical_root_count": partition_root_count,
        "canonical_reply_count": partition_reply_count,
        "root_reply_overlap_count": partition_overlap_count,
        "runtime_seconds": time.monotonic() - started,
    }


def process_uri_partitions(
    *,
    normalized_posts_path: Path,
    normalized_replies_path: Path,
    post_metadata_path: Path,
    partition_count: int,
    worker_count: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Deduplicate and publish canonical metadata partitions concurrently.

    Because all occurrences of a URI share a hash partition, local duplicate
    resolution and root precedence are globally complete. Each finished frame
    is validated and written by the process that owns its partition file.
    """

    post_metadata_path.mkdir(parents=True, exist_ok=False)
    logger.info(
        "Processing %s metadata partitions with up to %s worker processes",
        partition_count,
        worker_count,
    )

    def log_result(result: dict[str, Any]) -> None:
        logger.info(
            "Indexed URI partition %s/%s in %.1fs: roots=%s replies=%s overlaps=%s",
            result["partition_id"] + 1,
            partition_count,
            result["runtime_seconds"],
            f"{result['canonical_root_count']:,}",
            f"{result['canonical_reply_count']:,}",
            f"{result['root_reply_overlap_count']:,}",
        )

    results, effective_worker_count = partition_workers.run_partition_jobs(
        worker=_process_uri_partition,
        worker_kwargs=[
            {
                "normalized_posts_path": normalized_posts_path,
                "normalized_replies_path": normalized_replies_path,
                "post_metadata_path": post_metadata_path,
                "partition_id": partition_id,
                "partition_count": partition_count,
            }
            for partition_id in range(partition_count)
        ],
        worker_count=worker_count,
        on_result=log_result,
    )

    # Parquet dataset readers require at least one physical file. Publish a
    # typed empty part when every raw row was invalid.
    ensure_typed_parquet_dataset(
        post_metadata_path,
        source_metadata.POST_METADATA_SCHEMA,
    )
    return {
        "root_source_stats": _merge_numeric_stats([
            result["root_source_stats"] for result in results
        ]),
        "reply_source_stats": _merge_numeric_stats([
            result["reply_source_stats"] for result in results
        ]),
        "root_reply_overlap_count": sum(
            int(result["root_reply_overlap_count"]) for result in results
        ),
        "canonical_record_count": sum(
            int(result["canonical_record_count"]) for result in results
        ),
        "canonical_root_count": sum(
            int(result["canonical_root_count"]) for result in results
        ),
        "canonical_reply_count": sum(
            int(result["canonical_reply_count"]) for result in results
        ),
        "partition_worker_count": effective_worker_count,
        "partition_stats": [
            {
                key: value
                for key, value in result.items()
                if key not in {"root_source_stats", "reply_source_stats"}
            }
            for result in results
        ],
    }
