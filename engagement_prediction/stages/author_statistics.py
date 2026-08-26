"""Stage 6: build model-independent, training-window author statistics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict

import polars as pl

from engagement_prediction.data import author_statistics
from engagement_prediction.data import author_statistics_artifacts
from engagement_prediction.data import ingex
from engagement_prediction.data import source_manifests
from engagement_prediction.data.parquet import find_artifact_path
from engagement_prediction.data.source_metadata_artifacts import (
    SourceMetadataArtifact,
    load_source_metadata_artifact,
)
from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline.lineage import resolve_recorded_stage_lineage
from engagement_prediction.pipeline.logging import get_stage_logger


@dataclass(frozen=True)
class AuthorStatisticsConfig:
    """Validated training-support window and bounded aggregation layout."""

    support_start: datetime
    support_end: datetime
    partition_count: int
    source_metadata_partition_count: int


def build_config(
    args: argparse.Namespace,
    *,
    support_start: datetime,
    support_end: datetime,
    source_metadata_partition_count: int,
) -> AuthorStatisticsConfig:
    """Validate the training-only support window and physical partition count."""

    partition_count = int(args.author_statistics_partition_count)
    if partition_count <= 0:
        raise ValueError("author_statistics_partition_count must be positive")
    if source_metadata_partition_count <= 0:
        raise ValueError("Stage 00 source_metadata_partition_count must be positive")
    if support_end <= support_start:
        raise ValueError("Author-statistics support_end must be after support_start")
    for field_name, value in (
        ("support_start", support_start),
        ("support_end", support_end),
    ):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field_name} must be timezone-aware UTC")
        if value.utcoffset().total_seconds() != 0:
            raise ValueError(f"{field_name} must use UTC")
        if value.minute or value.second or value.microsecond:
            raise ValueError(f"{field_name} must be aligned to an hour")
    return AuthorStatisticsConfig(
        support_start=support_start,
        support_end=support_end,
        partition_count=partition_count,
        source_metadata_partition_count=source_metadata_partition_count,
    )


def _load_support_window(query_selection_dir: Path) -> tuple[datetime, datetime]:
    """Derive author support as the half-open ``[posts_start, val_start)`` window."""

    summary_path = query_selection_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Stage 1 artifact is missing summary.json: {query_selection_dir}"
        )
    try:
        summary = json.loads(summary_path.read_text())
        support_start = ingex.parse_utc_datetime(
            summary["posts_start"],
            field_name="posts_start",
        )
        support_end = ingex.parse_utc_datetime(
            summary["parameters"]["val_start"],
            field_name="val_start",
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Stage 1 summary does not record valid posts_start and val_start: {summary_path}"
        ) from exc
    if support_start is None or support_end is None:
        raise ValueError("Stage 1 support-window boundaries cannot be null")
    return support_start, support_end


def _load_exact_source_snapshots(
    *,
    source_artifact: SourceMetadataArtifact,
    post_liker_history_dir: Path,
    post_selection_dir: Path,
    query_selection_dir: Path,
    support_start: datetime,
    support_end: datetime,
) -> tuple[
    source_manifests.SourceSnapshot,
    source_manifests.SourceSnapshot,
    source_manifests.SourceSnapshot,
]:
    """Load aligned root, reply, and like snapshots without relisting GCS."""

    stage3_bundle = find_artifact_path(post_selection_dir, "post_universe_")
    stage5_bundle = find_artifact_path(
        post_liker_history_dir,
        "post_liker_histories_",
    )
    stage1_like_snapshot = source_manifests.load_source_snapshot(
        query_selection_dir,
        manifest_prefix="like_sources_",
        expected_blob_prefix="bsky_likes",
    )
    post_snapshot = source_manifests.load_source_snapshot(
        stage3_bundle,
        manifest_prefix="post_sources_",
        expected_blob_prefix="bsky_posts",
    )
    reply_snapshot = source_manifests.load_source_snapshot(
        stage3_bundle,
        manifest_prefix="reply_sources_",
        expected_blob_prefix="bsky_replies",
    )
    like_snapshot = source_manifests.load_source_snapshot(
        stage5_bundle,
        manifest_prefix="like_sources_",
        expected_blob_prefix="bsky_likes",
    )
    if post_snapshot.manifest != source_artifact.post_snapshot.manifest:
        raise ValueError("Stage 3 post source snapshot does not match its Stage 00 ancestor")
    if reply_snapshot.manifest != source_artifact.reply_snapshot.manifest:
        raise ValueError("Stage 3 reply source snapshot does not match its Stage 00 ancestor")
    if like_snapshot.manifest != stage1_like_snapshot.manifest:
        raise ValueError("Stage 5 like source snapshot does not match its Stage 1 ancestor")
    source_start, source_end = source_manifests.validate_aligned_source_snapshots(
        (post_snapshot, reply_snapshot, like_snapshot),
        description="Author-statistics source snapshots",
    )
    if source_start != support_start or source_end < support_end:
        raise ValueError(
            "Author-statistics source snapshot does not cover [posts_start, val_start)"
        )
    return post_snapshot, reply_snapshot, like_snapshot


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    """Run Stage 6 and publish unfiltered training-window author statistics."""

    out_dir = context.new_stage_dir("06_author_statistics")
    logger = get_stage_logger("06_AUTHOR_STATISTICS", log_file=out_dir / "stage.log")
    started_at = time.time()

    logger.info("Phase 1/6: resolving and validating Stage 00-5 lineage and snapshots")
    lineage = resolve_recorded_stage_lineage(
        context,
        terminal_stage_folder="05_post_liker_history",
        ancestor_stage_folders=(
            "00_source_metadata",
            "01_query_selection",
            "02_user_history",
            "03_post_selection",
            "04_negative_selection",
        ),
    )
    source_metadata_dir = lineage["00_source_metadata"]
    post_liker_history_dir = lineage["05_post_liker_history"]
    negative_selection_dir = lineage["04_negative_selection"]
    post_selection_dir = lineage["03_post_selection"]
    user_history_dir = lineage["02_user_history"]
    query_selection_dir = lineage["01_query_selection"]
    source_artifact = load_source_metadata_artifact(source_metadata_dir)
    support_start, support_end = _load_support_window(query_selection_dir)
    config = build_config(
        args,
        support_start=support_start,
        support_end=support_end,
        source_metadata_partition_count=source_artifact.partition_count,
    )
    post_snapshot, reply_snapshot, like_snapshot = _load_exact_source_snapshots(
        source_artifact=source_artifact,
        post_liker_history_dir=post_liker_history_dir,
        post_selection_dir=post_selection_dir,
        query_selection_dir=query_selection_dir,
        support_start=support_start,
        support_end=support_end,
    )
    like_paths = list(like_snapshot.file_uris)
    logger.info(
        "Starting author statistics: support_window=[%s, %s) roots=%s replies=%s "
        "likes=%s partitions=%s",
        support_start.isoformat(),
        support_end.isoformat(),
        f"{len(post_snapshot.file_uris):,}",
        f"{len(reply_snapshot.file_uris):,}",
        f"{len(like_paths):,}",
        config.partition_count,
    )

    artifact_suffix = out_dir.name
    bundle_path = out_dir / f"author_statistics_{artifact_suffix}"
    bundle_partial_path = out_dir / f"author_statistics_{artifact_suffix}.partial"
    bundle_partial_path.mkdir(parents=True, exist_ok=False)
    author_statistics_path = bundle_partial_path / "author_statistics"
    post_sources_path = bundle_partial_path / f"post_sources_{artifact_suffix}.json"
    reply_sources_path = bundle_partial_path / f"reply_sources_{artifact_suffix}.json"
    like_sources_path = bundle_partial_path / f"like_sources_{artifact_suffix}.json"
    ingex.write_source_manifest(post_sources_path, post_snapshot.manifest)
    ingex.write_source_manifest(reply_sources_path, reply_snapshot.manifest)
    ingex.write_source_manifest(like_sources_path, like_snapshot.manifest)

    staging_root = out_dir / f"_author_statistics_staging_{artifact_suffix}.partial"
    staging_root.mkdir(parents=True, exist_ok=False)
    normalized_likes_path = staging_root / "normalized_likes"
    per_post_shards_path = staging_root / "per_post_shards"
    per_post_by_author_path = staging_root / "per_post_by_author"

    # URI partitioning first resolves authoritative metadata and received-like
    # counts at the post level. Only then can rows be safely repartitioned by
    # author without splitting one post's support across workers.
    logger.info("Phase 2/6: routing the exact training-window like snapshot by URI")
    author_statistics_artifacts.materialize_like_routes(
        like_paths=like_paths,
        normalized_likes_path=normalized_likes_path,
        config=config,
        logger=logger,
    )

    logger.info("Phase 3/6: filtering Stage 00 metadata and collapsing raw likes per post")
    post_stats = author_statistics_artifacts.process_uri_partitions(
        post_metadata_path=source_artifact.post_metadata_path,
        normalized_likes_path=normalized_likes_path,
        per_post_shards_path=per_post_shards_path,
        config=config,
        logger=logger,
    )
    source_index = source_artifact.summary["index"]
    post_stats["root_reply_overlap_count"] = source_index["root_reply_overlap_count"]
    post_stats["root_source_stats"] = source_index["root_source_stats"]
    post_stats["reply_source_stats"] = source_index["reply_source_stats"]

    # The second hash key changes ownership from post URI to author DID. All of
    # one author's posts then land in one bounded aggregation partition.
    logger.info("Phase 4/6: routing narrow per-post rows by author DID")
    author_statistics_artifacts.route_per_post_rows_by_author(
        per_post_shards_path=per_post_shards_path,
        per_post_by_author_path=per_post_by_author_path,
        partition_count=config.partition_count,
    )

    logger.info("Phase 5/6: aggregating and publishing all author statistics")
    aggregation_stats = author_statistics_artifacts.process_author_partitions(
        per_post_by_author_path=per_post_by_author_path,
        author_statistics_path=author_statistics_path,
        config=config,
        logger=logger,
    )
    public_stats = author_statistics.validate_author_statistics_dataset(
        pl.scan_parquet(sorted(author_statistics_path.glob("*.parquet")))
    )
    if public_stats["author_count"] != aggregation_stats["author_count"]:
        raise ValueError("Published author count does not match aggregated author count")

    logger.info("Phase 6/6: removing internal staging and publishing the bundle")
    shutil.rmtree(staging_root)
    bundle_partial_path.replace(bundle_path)
    final_author_statistics_path = bundle_path / "author_statistics"
    final_post_sources_path = bundle_path / post_sources_path.name
    final_reply_sources_path = bundle_path / reply_sources_path.name
    final_like_sources_path = bundle_path / like_sources_path.name

    runtime_seconds = time.time() - started_at
    summary = {
        "parameters": {
            "author_statistics_partition_count": config.partition_count,
            "source_metadata_partition_count": config.source_metadata_partition_count,
        },
        "input": {
            "source_metadata_dir": str(source_metadata_dir),
            "post_liker_history_dir": str(post_liker_history_dir),
            "negative_selection_dir": str(negative_selection_dir),
            "post_selection_dir": str(post_selection_dir),
            "user_history_dir": str(user_history_dir),
            "query_selection_dir": str(query_selection_dir),
            "support_start": support_start.isoformat(),
            "support_end": support_end.isoformat(),
            "post_file_count": len(post_snapshot.file_uris),
            "reply_file_count": len(reply_snapshot.file_uris),
            "like_file_count": len(like_paths),
        },
        "post_aggregation": post_stats,
        "author_aggregation": aggregation_stats,
        "public_validation": public_stats,
        "outputs": {
            "author_statistics_path": bundle_path.name,
            "author_statistics_dataset_path": str(
                Path(bundle_path.name) / "author_statistics"
            ),
            "post_sources_path": str(
                Path(bundle_path.name) / final_post_sources_path.name
            ),
            "reply_sources_path": str(
                Path(bundle_path.name) / final_reply_sources_path.name
            ),
            "like_sources_path": str(
                Path(bundle_path.name) / final_like_sources_path.name
            ),
        },
        "runtime_seconds": runtime_seconds,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "stage_info.txt").write_text(
        "\n".join([
            "stage: author_statistics",
            f"runtime_seconds: {runtime_seconds:.2f}",
            f"support_window: [{support_start.isoformat()}, {support_end.isoformat()})",
            f"author_statistics_partition_count: {config.partition_count}",
            f"source_metadata_partition_count: {config.source_metadata_partition_count}",
            f"resolved_post_count: {post_stats['resolved_post_count']}",
            f"matched_like_event_count: {post_stats['matched_like_event_count']}",
            f"unmatched_like_event_count: {post_stats['unmatched_like_event_count']}",
            f"author_count: {aggregation_stats['author_count']}",
            f"author_statistics_dataset_path: {Path(bundle_path.name) / 'author_statistics'}",
            f"post_sources_path: {Path(bundle_path.name) / final_post_sources_path.name}",
            f"reply_sources_path: {Path(bundle_path.name) / final_reply_sources_path.name}",
            f"like_sources_path: {Path(bundle_path.name) / final_like_sources_path.name}",
        ])
        + "\n"
    )
    logger.info(
        "Author statistics completed in %.2fs: posts=%s matched_likes=%s authors=%s",
        runtime_seconds,
        f"{post_stats['resolved_post_count']:,}",
        f"{post_stats['matched_like_event_count']:,}",
        f"{aggregation_stats['author_count']:,}",
    )
    return {
        "output_dir": out_dir,
        "artifacts": {
            "author_statistics_path": str(bundle_path),
            "author_statistics_dataset_path": str(final_author_statistics_path),
            "post_sources_path": str(final_post_sources_path),
            "reply_sources_path": str(final_reply_sources_path),
            "like_sources_path": str(final_like_sources_path),
        },
    }
