"""Stage 5: extract complete post-liker event histories for selected posts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict

from engagement_prediction.data import ingex
from engagement_prediction.data import post_liker_history
from engagement_prediction.data import post_liker_history_artifacts
from engagement_prediction.data import source_manifests
from engagement_prediction.data.parquet import find_artifact_path
from engagement_prediction.data.source_metadata_artifacts import load_source_metadata_artifact
from engagement_prediction.pipeline.artifacts import PartialArtifactBundle
from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline.lineage import resolve_recorded_stage_lineage
from engagement_prediction.pipeline.logging import get_stage_logger


@dataclass(frozen=True)
class PostLikerHistoryConfig:
    """Validated URI partition count for complete liker-event extraction."""

    partition_count: int


def build_config(args: argparse.Namespace) -> PostLikerHistoryConfig:
    """Parse the Stage 5 partition setting."""

    partition_count = int(args.post_liker_history_partition_count)
    if partition_count <= 0:
        raise ValueError("post_liker_history_partition_count must be positive")
    return PostLikerHistoryConfig(partition_count=partition_count)


def _load_stage3_bundle(post_selection_dir: Path) -> tuple[Path, Path, Path]:
    """Locate resolved requirements and the unresolved-history anti-join."""

    bundle_path = find_artifact_path(post_selection_dir, "post_universe_")
    required_posts_path = bundle_path / "required_posts"
    missing_required_posts_path = bundle_path / "missing_required_posts"
    if not required_posts_path.is_dir() or not missing_required_posts_path.is_dir():
        raise FileNotFoundError(
            "Stage 5 requires Stage 3 required_posts/ and missing_required_posts/"
        )
    return bundle_path, required_posts_path, missing_required_posts_path


def _load_stage4_bundle(negative_selection_dir: Path) -> tuple[Path, Path]:
    """Locate the final negative-URI dataset."""

    bundle_path = find_artifact_path(negative_selection_dir, "negative_candidates_")
    negative_post_uris_path = bundle_path / "negative_post_uris"
    if not negative_post_uris_path.is_dir():
        raise FileNotFoundError(
            "Stage 5 requires Stage 4 negative_post_uris/"
        )
    return bundle_path, negative_post_uris_path


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    """Run Stage 5 and publish every raw liker event for selected posts."""

    out_dir = context.new_stage_dir("05_post_liker_history")
    logger = get_stage_logger("05_POST_LIKER_HISTORY", log_file=out_dir / "stage.log")
    started_at = time.time()
    config = build_config(args)

    logger.info("Phase 1/5: resolving and validating Stage 00-4 lineage")
    lineage = resolve_recorded_stage_lineage(
        context,
        terminal_stage_folder="04_negative_selection",
        ancestor_stage_folders=(
            "00_source_metadata",
            "01_query_selection",
            "02_user_history",
            "03_post_selection",
        ),
    )
    source_metadata_dir = lineage["00_source_metadata"]
    negative_selection_dir = lineage["04_negative_selection"]
    post_selection_dir = lineage["03_post_selection"]
    user_history_dir = lineage["02_user_history"]
    query_selection_dir = lineage["01_query_selection"]
    stage3_partition_count = load_source_metadata_artifact(
        source_metadata_dir
    ).partition_count
    (
        post_universe_path,
        required_posts_path,
        missing_required_posts_path,
    ) = _load_stage3_bundle(post_selection_dir)
    negative_candidates_path, negative_post_uris_path = _load_stage4_bundle(
        negative_selection_dir
    )
    stage4_like_snapshot = source_manifests.load_source_snapshot(
        negative_candidates_path,
        manifest_prefix="like_sources_",
        expected_blob_prefix="bsky_likes",
    )
    stage1_like_snapshot = source_manifests.load_source_snapshot(
        query_selection_dir,
        manifest_prefix="like_sources_",
        expected_blob_prefix="bsky_likes",
    )
    if stage4_like_snapshot.manifest != stage1_like_snapshot.manifest:
        raise ValueError("Stage 4 like source snapshot does not match its Stage 1 ancestor")
    like_paths = list(stage4_like_snapshot.file_uris)
    source_start = stage4_like_snapshot.start
    source_end = stage4_like_snapshot.end
    logger.info(
        "Starting post-liker extraction: source_window=[%s, %s) files=%s "
        "Stage3_partitions=%s Stage5_partitions=%s",
        source_start.isoformat(),
        source_end.isoformat(),
        f"{len(like_paths):,}",
        stage3_partition_count,
        config.partition_count,
    )

    artifact_suffix = out_dir.name
    publication = PartialArtifactBundle.create(
        output_dir=out_dir,
        bundle_name=f"post_liker_histories_{artifact_suffix}",
        staging_name=f"_post_liker_history_staging_{artifact_suffix}.partial",
        dataset_schemas={
            "post_liker_events": post_liker_history.POST_LIKER_EVENT_SCHEMA,
            "post_liker_posts": post_liker_history.POST_LIKER_POST_SCHEMA,
        },
    )
    bundle_path = publication.final_path
    post_liker_events_path = publication.public_path("post_liker_events")
    post_liker_posts_path = publication.public_path("post_liker_posts")
    like_sources_path = publication.public_path(f"like_sources_{artifact_suffix}.json")
    ingex.write_source_manifest(like_sources_path, stage4_like_snapshot.manifest)

    staging_root = publication.staging_path
    selected_post_shards_path = staging_root / "selected_post_shards"
    selected_post_routes_path = staging_root / "selected_post_routes"
    normalized_likes_path = staging_root / "normalized_likes"

    # Role construction excludes unresolved histories and unused Stage 3
    # reservoir posts, then re-hashes the surviving URI set for this stage.
    logger.info(
        "Phase 2/5: building and routing the resolved positive/history/final-negative universe"
    )
    post_liker_history_artifacts.materialize_selected_post_routes(
        required_posts_path=required_posts_path,
        missing_required_posts_path=missing_required_posts_path,
        negative_post_uris_path=negative_post_uris_path,
        stage3_partition_count=stage3_partition_count,
        output_path=selected_post_routes_path,
        shard_path=selected_post_shards_path,
        partition_count=config.partition_count,
    )

    # This scan deliberately retains all valid source-window events, including
    # duplicate rows and likes from users absent from query selection.
    logger.info(
        "Phase 3/5: stream-sinking all valid events from %s exact like files",
        f"{len(like_paths):,}",
    )
    post_liker_history_artifacts.materialize_normalized_likes(
        like_paths=like_paths,
        source_start=source_start,
        source_end=source_end,
        output_path=normalized_likes_path,
        partition_count=config.partition_count,
    )

    # Selected keys and likes share the same URI hash, so each output partition
    # can independently prove event-to-post integrity and exact event counts.
    logger.info("Phase 4/5: matching events and writing public URI partitions")
    extraction_stats = post_liker_history_artifacts.process_uri_partitions(
        selected_post_routes_path=selected_post_routes_path,
        normalized_likes_path=normalized_likes_path,
        post_liker_events_path=post_liker_events_path,
        post_liker_posts_path=post_liker_posts_path,
        source_start=source_start,
        source_end=source_end,
        partition_count=config.partition_count,
        logger=logger,
    )

    logger.info("Phase 5/5: validating and publishing the bundle")
    final_events_path = bundle_path / "post_liker_events"
    final_posts_path = bundle_path / "post_liker_posts"
    final_like_sources_path = bundle_path / like_sources_path.name

    runtime_seconds = time.time() - started_at
    summary = {
        "parameters": {
            "post_liker_history_partition_count": config.partition_count,
        },
        "input": {
            "source_metadata_dir": str(source_metadata_dir),
            "negative_selection_dir": str(negative_selection_dir),
            "negative_candidates_path": str(negative_candidates_path),
            "post_selection_dir": str(post_selection_dir),
            "post_universe_path": str(post_universe_path),
            "user_history_dir": str(user_history_dir),
            "query_selection_dir": str(query_selection_dir),
            "like_file_count": len(like_paths),
            "source_start": source_start.isoformat(),
            "source_end": source_end.isoformat(),
        },
        "extraction": extraction_stats,
        "outputs": {
            "post_liker_histories_path": bundle_path.name,
            "post_liker_events_path": str(
                Path(bundle_path.name) / "post_liker_events"
            ),
            "post_liker_posts_path": str(
                Path(bundle_path.name) / "post_liker_posts"
            ),
            "like_sources_path": str(
                Path(bundle_path.name) / final_like_sources_path.name
            ),
        },
        "runtime_seconds": runtime_seconds,
    }
    stage_info = (
        "\n".join([
            "stage: post_liker_history",
            f"runtime_seconds: {runtime_seconds:.2f}",
            f"post_liker_history_partition_count: {config.partition_count}",
            f"selected_post_count: {extraction_stats['selected_post_count']}",
            f"positive_post_count: {extraction_stats['positive_post_count']}",
            f"history_post_count: {extraction_stats['history_post_count']}",
            f"negative_post_count: {extraction_stats['negative_post_count']}",
            f"valid_source_like_count: {extraction_stats['valid_source_like_count']}",
            f"matched_like_event_count: {extraction_stats['matched_like_event_count']}",
            f"posts_with_likes_count: {extraction_stats['posts_with_likes_count']}",
            f"posts_without_likes_count: {extraction_stats['posts_without_likes_count']}",
            f"max_like_events_per_post: {extraction_stats['max_like_events_per_post']}",
            f"post_liker_events_path: {Path(bundle_path.name) / 'post_liker_events'}",
            f"post_liker_posts_path: {Path(bundle_path.name) / 'post_liker_posts'}",
            f"like_sources_path: {Path(bundle_path.name) / final_like_sources_path.name}",
        ])
        + "\n"
    )
    publication.publish(
        summary=summary,
        stage_info=stage_info,
    )
    logger.info(
        "Post-liker history completed in %.2fs: posts=%s events=%s zero_like_posts=%s",
        runtime_seconds,
        f"{extraction_stats['selected_post_count']:,}",
        f"{extraction_stats['matched_like_event_count']:,}",
        f"{extraction_stats['posts_without_likes_count']:,}",
    )
    return {
        "output_dir": out_dir,
        "artifacts": {
            "post_liker_histories_path": str(bundle_path),
            "post_liker_events_path": str(final_events_path),
            "post_liker_posts_path": str(final_posts_path),
            "like_sources_path": str(final_like_sources_path),
        },
    }
