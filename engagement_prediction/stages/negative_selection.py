"""Stage 4: calculate candidate popularity and select hourly negatives.

The random Stage 3 reservoir is scored with strict as-of like counts for each
query hour. Bounded local finalists are then merged by hour to apply globally
exact popular-first and random-fill quotas without a full post-hour matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict

import polars as pl

from engagement_prediction.data import candidate_popularity
from engagement_prediction.data import ingex
from engagement_prediction.data import negative_selection
from engagement_prediction.data import negative_selection_artifacts
from engagement_prediction.data import source_manifests
from engagement_prediction.data.parquet import find_artifact_path, scan_parquet_artifact
from engagement_prediction.data.source_metadata_artifacts import load_source_metadata_artifact
from engagement_prediction.pipeline.artifacts import PartialArtifactBundle
from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline.lineage import resolve_recorded_stage_lineage
from engagement_prediction.pipeline.logging import get_stage_logger


@dataclass(frozen=True)
class NegativeSelectionConfig:
    """Validated popularity threshold, quota, age, and partition settings."""

    negative_candidates_per_hour: int
    min_likes_for_popular_candidate: int
    popular_candidate_fraction: float
    max_candidate_age_hours: int
    partition_count: int
    random_seed: int


def build_config(
    args: argparse.Namespace,
    *,
    partition_count: int,
) -> NegativeSelectionConfig:
    """Parse Stage 4 settings using Stage 00's physical URI partition count."""

    negative_candidates_per_hour = int(args.negative_candidates_per_hour)
    if negative_candidates_per_hour < 0:
        raise ValueError("negative_candidates_per_hour must be non-negative")
    min_likes = int(args.min_likes_for_popular_candidate)
    if min_likes < 0:
        raise ValueError("min_likes_for_popular_candidate must be non-negative")
    popular_fraction = float(args.popular_candidate_fraction)
    if not 0.0 <= popular_fraction <= 1.0:
        raise ValueError("popular_candidate_fraction must be between 0 and 1")
    max_age = int(args.max_candidate_age_hours)
    if max_age <= 0:
        raise ValueError("max_candidate_age_hours must be positive")
    if partition_count <= 0:
        raise ValueError("Stage 00 source_metadata_partition_count must be positive")
    return NegativeSelectionConfig(
        negative_candidates_per_hour=negative_candidates_per_hour,
        min_likes_for_popular_candidate=min_likes,
        popular_candidate_fraction=popular_fraction,
        max_candidate_age_hours=max_age,
        partition_count=partition_count,
        random_seed=int(args.random_seed),
    )


def _load_stage3_bundle(post_selection_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve the Stage 3 metadata and candidate-reservoir datasets."""

    bundle_path = find_artifact_path(post_selection_dir, "post_universe_")
    posts_path = bundle_path / "posts"
    candidate_sources_path = bundle_path / "candidate_sources"
    if not posts_path.is_dir() or not candidate_sources_path.is_dir():
        raise FileNotFoundError(
            "Stage 4 requires Stage 3 posts/ and candidate_sources/ datasets"
        )
    return bundle_path, posts_path, candidate_sources_path


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    """Run Stage 4 popularity calculation and shared hourly negative selection."""

    out_dir = context.new_stage_dir("04_negative_selection")
    logger = get_stage_logger("04_NEGATIVE_SELECTION", log_file=out_dir / "stage.log")
    started_at = time.time()

    logger.info("Phase 1/7: resolving and validating Stage 00-3 lineage")
    lineage = resolve_recorded_stage_lineage(
        context,
        terminal_stage_folder="03_post_selection",
        ancestor_stage_folders=(
            "00_source_metadata",
            "01_query_selection",
            "02_user_history",
        ),
    )
    source_metadata_dir = lineage["00_source_metadata"]
    post_selection_dir = lineage["03_post_selection"]
    user_history_dir = lineage["02_user_history"]
    query_selection_dir = lineage["01_query_selection"]
    partition_count = load_source_metadata_artifact(source_metadata_dir).partition_count
    config = build_config(args, partition_count=partition_count)
    post_universe_path, posts_path, candidate_sources_path = _load_stage3_bundle(
        post_selection_dir
    )
    queries_lf = scan_parquet_artifact(
        find_artifact_path(query_selection_dir, "queries_")
    )
    query_hours_df = candidate_popularity.validate_query_hours(
        queries_lf.select("query_hour").unique().collect(engine="streaming")
    )
    like_snapshot = source_manifests.load_source_snapshot(
        query_selection_dir,
        manifest_prefix="like_sources_",
        expected_blob_prefix="bsky_likes",
    )
    like_paths = list(like_snapshot.file_uris)
    source_start = like_snapshot.start
    source_end = like_snapshot.end
    popular_quota = negative_selection.calculate_popular_quota(
        config.negative_candidates_per_hour,
        config.popular_candidate_fraction,
    )
    logger.info(
        "Starting negative selection: query_hours=%s K=%s popular_quota=%s "
        "min_likes=%s max_age_hours=%s URI_partitions=%s",
        f"{query_hours_df.height:,}",
        config.negative_candidates_per_hour,
        popular_quota,
        config.min_likes_for_popular_candidate,
        config.max_candidate_age_hours,
        config.partition_count,
    )

    artifact_suffix = out_dir.name
    publication = PartialArtifactBundle.create(
        output_dir=out_dir,
        bundle_name=f"negative_candidates_{artifact_suffix}",
        staging_name=f"_negative_selection_staging_{artifact_suffix}.partial",
        dataset_schemas={
            "hourly_candidates": negative_selection.HOURLY_CANDIDATE_SCHEMA,
            "negative_post_uris": negative_selection.NEGATIVE_POST_URI_SCHEMA,
        },
    )
    bundle_path = publication.final_path
    hourly_candidates_path = publication.public_path("hourly_candidates")
    negative_post_uris_path = publication.public_path("negative_post_uris")
    like_sources_path = publication.public_path(f"like_sources_{artifact_suffix}.json")
    ingex.write_source_manifest(like_sources_path, like_snapshot.manifest)

    staging_root = publication.staging_path
    normalized_likes_path = staging_root / "normalized_likes"
    local_finalists_path = staging_root / "local_finalists"
    routed_finalists_path = staging_root / "routed_finalists"
    selected_uri_routes_path = staging_root / "selected_uri_routes"

    logger.info(
        "Phase 2/7: stream-sinking %s exact like files into URI partitions",
        f"{len(like_paths):,}",
    )
    if config.negative_candidates_per_hour == 0 or query_hours_df.is_empty():
        normalized_likes_path.mkdir(parents=True, exist_ok=False)
    else:
        negative_selection_artifacts.materialize_normalized_likes(
            like_paths=like_paths,
            source_start=source_start,
            source_end=source_end,
            output_path=normalized_likes_path,
            partition_count=config.partition_count,
        )

    # Each URI partition produces only its best K popular and K random rows per
    # hour. Those bounded local finalists are sufficient to recover the global
    # top K without materializing the complete candidate-hour table on disk.
    logger.info("Phase 3/7: calculating strictly prior candidate popularity")
    popularity_stats = negative_selection_artifacts.process_uri_partitions(
        posts_path=posts_path,
        candidate_sources_path=candidate_sources_path,
        normalized_likes_path=normalized_likes_path,
        query_hours_df=query_hours_df,
        local_finalists_path=local_finalists_path,
        config=config,
        logger=logger,
    )

    # The first partitioning key is URI for joins and cumulative popularity;
    # the second is query hour so the final quota is globally exact per hour.
    logger.info("Phase 4/7: routing bounded method finalists by query hour")
    negative_selection_artifacts.route_local_finalists_by_hour(
        local_finalists_path=local_finalists_path,
        output_path=routed_finalists_path,
        partition_count=config.partition_count,
    )

    logger.info("Phase 5/7: applying global popular-first hourly quotas")
    selection_stats = negative_selection_artifacts.process_hour_partitions(
        routed_finalists_path=routed_finalists_path,
        hourly_candidates_path=hourly_candidates_path,
        query_hours_df=query_hours_df,
        config=config,
        logger=logger,
    )

    # A post may be selected in many hours. Repartitioning the final rows by URI
    # creates the compact unique set consumed by Stage 5.
    logger.info("Phase 6/7: globally deduplicating and validating selected post URIs")
    unique_negative_post_count = negative_selection_artifacts.build_negative_post_uris(
        hourly_candidates_path=hourly_candidates_path,
        negative_post_uris_path=negative_post_uris_path,
        uri_routes_path=selected_uri_routes_path,
        posts_path=posts_path,
        candidate_sources_path=candidate_sources_path,
        config=config,
    )

    logger.info("Phase 7/7: validating and publishing the completed bundle")
    final_hourly_candidates_path = bundle_path / "hourly_candidates"
    final_negative_post_uris_path = bundle_path / "negative_post_uris"
    final_like_sources_path = bundle_path / like_sources_path.name

    merged_hourly_stats = {}
    for query_hour in query_hours_df.get_column("query_hour").to_list():
        key = query_hour.isoformat()
        merged_hourly_stats[key] = {
            **popularity_stats["candidate_hour_stats"].get(key, {
                "eligible_candidate_count": 0,
                "zero_like_candidate_count": 0,
                "popular_eligible_candidate_count": 0,
            }),
            **selection_stats["hourly_selection_stats"][key],
        }

    runtime_seconds = time.time() - started_at
    summary = {
        "parameters": {
            "negative_candidates_per_hour": config.negative_candidates_per_hour,
            "min_likes_for_popular_candidate": config.min_likes_for_popular_candidate,
            "popular_candidate_fraction": config.popular_candidate_fraction,
            "popular_candidate_quota": popular_quota,
            "max_candidate_age_hours": config.max_candidate_age_hours,
            "source_metadata_partition_count": config.partition_count,
            "random_seed": config.random_seed,
        },
        "input": {
            "source_metadata_dir": str(source_metadata_dir),
            "post_selection_dir": str(post_selection_dir),
            "post_universe_path": str(post_universe_path),
            "user_history_dir": str(user_history_dir),
            "query_selection_dir": str(query_selection_dir),
            "like_file_count": len(like_paths),
            "source_start": source_start.isoformat(),
            "source_end": source_end.isoformat(),
            **{
                key: value
                for key, value in popularity_stats.items()
                if key != "candidate_hour_stats"
            },
        },
        "selection": {
            **{
                key: value
                for key, value in selection_stats.items()
                if key != "hourly_selection_stats"
            },
            "unique_negative_post_count": unique_negative_post_count,
            "hourly_stats": merged_hourly_stats,
        },
        "outputs": {
            "negative_candidates_path": bundle_path.name,
            "hourly_candidates_path": str(
                Path(bundle_path.name) / "hourly_candidates"
            ),
            "negative_post_uris_path": str(
                Path(bundle_path.name) / "negative_post_uris"
            ),
            "like_sources_path": str(Path(bundle_path.name) / final_like_sources_path.name),
        },
        "runtime_seconds": runtime_seconds,
    }
    stage_info = (
        "\n".join([
            "stage: negative_selection",
            f"runtime_seconds: {runtime_seconds:.2f}",
            f"query_hour_count: {selection_stats['query_hour_count']}",
            f"candidate_reservoir_count: {popularity_stats['candidate_reservoir_count']}",
            f"eligible_candidate_hour_count: {popularity_stats['eligible_candidate_hour_count']}",
            f"selected_candidate_row_count: {selection_stats['selected_candidate_row_count']}",
            f"popular_selected_count: {selection_stats['popular_selected_count']}",
            f"random_selected_count: {selection_stats['random_selected_count']}",
            f"short_query_hour_count: {selection_stats['short_query_hour_count']}",
            f"total_shortfall_count: {selection_stats['total_shortfall_count']}",
            f"unique_negative_post_count: {unique_negative_post_count}",
            f"like_sources_path: {Path(bundle_path.name) / final_like_sources_path.name}",
        ])
        + "\n"
    )
    publication.publish(
        summary=summary,
        stage_info=stage_info,
    )
    logger.info(
        "Negative selection completed in %.2fs: query_hours=%s selected_rows=%s "
        "unique_posts=%s popular=%s random=%s short_hours=%s",
        runtime_seconds,
        f"{selection_stats['query_hour_count']:,}",
        f"{selection_stats['selected_candidate_row_count']:,}",
        f"{unique_negative_post_count:,}",
        f"{selection_stats['popular_selected_count']:,}",
        f"{selection_stats['random_selected_count']:,}",
        f"{selection_stats['short_query_hour_count']:,}",
    )
    return {
        "output_dir": out_dir,
        "artifacts": {
            "negative_candidates_path": str(bundle_path),
            "hourly_candidates_path": str(final_hourly_candidates_path),
            "negative_post_uris_path": str(final_negative_post_uris_path),
            "like_sources_path": str(final_like_sources_path),
        },
    }
