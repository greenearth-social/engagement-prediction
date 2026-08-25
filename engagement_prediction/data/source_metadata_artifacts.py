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

from engagement_prediction.data import ingex, source_manifests, source_metadata
from engagement_prediction.data.parquet import (
    find_artifact_path,
    read_parquet_parts,
    sink_partitioned_parquet,
)


class SourceMetadataConfig(Protocol):
    posts_start: datetime
    posts_end: datetime
    source_metadata_partition_count: int


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


def process_uri_partitions(
    *,
    normalized_posts_path: Path,
    normalized_replies_path: Path,
    post_metadata_path: Path,
    partition_count: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Deduplicate and publish one canonical metadata partition at a time.

    Because all occurrences of a URI share a hash partition, local duplicate
    resolution and root precedence are globally complete. Each finished frame
    is validated and written before the next partition is loaded.
    """

    post_metadata_path.mkdir(parents=True, exist_ok=False)
    root_stats: list[dict[str, int]] = []
    reply_stats: list[dict[str, int]] = []
    overlap_count = 0
    root_count = 0
    reply_count = 0
    partition_stats: list[dict[str, Any]] = []

    for partition_id in range(partition_count):
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
        if not metadata.is_empty():
            metadata.write_parquet(
                post_metadata_path / f"part-{partition_id:05d}.parquet",
                compression="zstd",
            )

        root_stats.append(root_partition_stats)
        reply_stats.append(reply_partition_stats)
        overlap_count += partition_overlap_count
        partition_root_count = metadata.filter(~pl.col("is_reply")).height
        partition_reply_count = metadata.filter(pl.col("is_reply")).height
        root_count += partition_root_count
        reply_count += partition_reply_count
        elapsed = time.monotonic() - started
        partition_stats.append({
            "partition_id": partition_id,
            "canonical_record_count": metadata.height,
            "canonical_root_count": partition_root_count,
            "canonical_reply_count": partition_reply_count,
            "root_reply_overlap_count": partition_overlap_count,
            "runtime_seconds": elapsed,
        })
        logger.info(
            "Indexed URI partition %s/%s in %.1fs: roots=%s replies=%s overlaps=%s",
            partition_id + 1,
            partition_count,
            elapsed,
            f"{partition_root_count:,}",
            f"{partition_reply_count:,}",
            f"{partition_overlap_count:,}",
        )

    # Parquet dataset readers require at least one physical file. Publish a
    # typed empty part when every raw row was invalid.
    if not list(post_metadata_path.glob("*.parquet")):
        source_metadata.empty_frame(source_metadata.POST_METADATA_SCHEMA).write_parquet(
            post_metadata_path / "part-00000.parquet",
            compression="zstd",
        )
    return {
        "root_source_stats": _merge_numeric_stats(root_stats),
        "reply_source_stats": _merge_numeric_stats(reply_stats),
        "root_reply_overlap_count": overlap_count,
        "canonical_record_count": root_count + reply_count,
        "canonical_root_count": root_count,
        "canonical_reply_count": reply_count,
        "partition_stats": partition_stats,
    }
