"""Stage 3: build the required-post universe and candidate reservoirs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict

import polars as pl

from engagement_prediction.data import ingex
from engagement_prediction.data import post_selection_artifacts
from engagement_prediction.data.parquet import find_artifact_path, scan_parquet_artifact
from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline.dependencies import load_stage_manifest
from utils.helpers import get_stage_logger


@dataclass(frozen=True)
class PostSelectionConfig:
    gcs_bucket: str
    posts_start: datetime
    posts_end: datetime
    random_candidate_sampling_fraction: float
    max_political_candidates_per_creation_hour: int
    political_score_threshold: float
    political_inference_window_padding_days: int
    post_selection_partition_count: int
    random_seed: int


def _require_hour_aligned(value: datetime, field_name: str) -> None:
    if value.minute or value.second or value.microsecond:
        raise ValueError(f"{field_name} must be aligned to the start of an hour")


def build_config(args: argparse.Namespace) -> PostSelectionConfig:
    posts_start = ingex.parse_utc_datetime(args.posts_start, field_name="posts_start")
    posts_end = ingex.parse_utc_datetime(args.posts_end, field_name="posts_end")
    if posts_start is None or posts_end is None:
        raise ValueError("posts_start and posts_end are required for post_selection")
    _require_hour_aligned(posts_start, "posts_start")
    _require_hour_aligned(posts_end, "posts_end")
    if posts_end <= posts_start:
        raise ValueError("posts_end must be after posts_start")

    fraction = float(args.random_candidate_sampling_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("random_candidate_sampling_fraction must be between 0 and 1")
    political_cap = int(args.max_political_candidates_per_creation_hour)
    if political_cap < 0:
        raise ValueError("max_political_candidates_per_creation_hour must be non-negative")
    political_threshold = float(args.political_score_threshold)
    if not 0.0 <= political_threshold <= 1.0:
        raise ValueError("political_score_threshold must be between 0 and 1")
    padding_days = int(args.political_inference_window_padding_days)
    if padding_days < 0:
        raise ValueError("political_inference_window_padding_days must be non-negative")
    partition_count = int(args.post_selection_partition_count)
    if partition_count <= 0:
        raise ValueError("post_selection_partition_count must be positive")
    return PostSelectionConfig(
        gcs_bucket=str(args.gcs_bucket),
        posts_start=posts_start,
        posts_end=posts_end,
        random_candidate_sampling_fraction=fraction,
        max_political_candidates_per_creation_hour=political_cap,
        political_score_threshold=political_threshold,
        political_inference_window_padding_days=padding_days,
        post_selection_partition_count=partition_count,
        random_seed=int(args.random_seed),
    )


def _validate_upstream_lineage(
    *,
    query_selection_dir: Path,
    user_history_dir: Path,
) -> None:
    manifest = load_stage_manifest(user_history_dir)
    recorded_query_selection = manifest.get("inputs", {}).get("01_query_selection")
    if not recorded_query_selection:
        raise ValueError(
            f"Stage 3 requires a new user_history artifact, but '{user_history_dir}' "
            "does not record an 01_query_selection input"
        )
    if Path(recorded_query_selection).resolve() != query_selection_dir.resolve():
        raise ValueError(
            f"Stage 2 artifact '{user_history_dir}' was built from "
            f"'{Path(recorded_query_selection).resolve()}', not selected Stage 1 "
            f"'{query_selection_dir.resolve()}'"
        )


def _validate_query_window(
    queries_lf: pl.LazyFrame,
    config: PostSelectionConfig,
) -> tuple[datetime, datetime]:
    schema = queries_lf.collect_schema()
    if "query_hour" not in schema:
        raise ValueError("Stage 1 queries artifact is missing query_hour")
    query_hour_dtype = schema["query_hour"]
    if not isinstance(query_hour_dtype, pl.Datetime) or query_hour_dtype.time_zone != "UTC":
        raise ValueError(f"Stage 1 query_hour must be a UTC datetime, found {query_hour_dtype}")
    bounds = queries_lf.select(
        pl.col("query_hour").min().alias("min_query_hour"),
        pl.col("query_hour").max().alias("max_query_hour"),
    ).collect(engine="streaming")
    min_query_hour = bounds.item(0, "min_query_hour")
    max_query_hour = bounds.item(0, "max_query_hour")
    if min_query_hour is None or max_query_hour is None:
        raise ValueError("Stage 1 queries artifact contains no queries")
    if config.posts_start > min_query_hour or config.posts_end <= max_query_hour:
        raise ValueError(
            "posts_start/posts_end must cover every selected query hour: "
            f"query range is {min_query_hour.isoformat()} to {max_query_hour.isoformat()}"
        )
    return min_query_hour, max_query_hour


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = context.new_stage_dir("03_post_selection")
    logger = get_stage_logger("03_POST_SELECTION", log_file=out_dir / "stage.log")
    started_at = time.time()
    config = build_config(args)
    logger.info(
        "Starting post selection: posts_window=[%s, %s) partitions=%s "
        "random_fraction=%s political_cap_per_hour=%s",
        config.posts_start.isoformat(),
        config.posts_end.isoformat(),
        config.post_selection_partition_count,
        config.random_candidate_sampling_fraction,
        config.max_political_candidates_per_creation_hour,
    )

    logger.info("Phase 1/7: resolving and validating Stage 1/2 inputs")
    user_history_dir = context.resolve_prior_output(
        "02_user_history",
        prior_path=context.prior_outputs.get("02_user_history"),
    )
    query_selection_dir = context.resolve_prior_output(
        "01_query_selection",
        prior_path=context.prior_outputs.get("01_query_selection"),
    )
    _validate_upstream_lineage(
        query_selection_dir=query_selection_dir,
        user_history_dir=user_history_dir,
    )
    queries_lf = scan_parquet_artifact(
        find_artifact_path(query_selection_dir, "queries_")
    )
    min_query_hour, max_query_hour = _validate_query_window(queries_lf, config)
    query_positives_lf = scan_parquet_artifact(
        find_artifact_path(query_selection_dir, "query_positives_")
    )
    try:
        history_post_uris_path = find_artifact_path(
            user_history_dir,
            "history_post_uris_",
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Stage 3 requires the new Stage 2 history_post_uris_* artifact; "
            "rerun user_history with the current implementation"
        ) from exc
    history_post_uris_lf = scan_parquet_artifact(history_post_uris_path)
    logger.info(
        "Resolved inputs: query_selection=%s user_history=%s query_range=[%s, %s]",
        query_selection_dir,
        user_history_dir,
        min_query_hour.isoformat(),
        max_query_hour.isoformat(),
    )

    source_listing_started = time.monotonic()
    logger.info("Phase 2/7: listing bsky_posts source files")
    post_paths, post_timestamps = ingex.list_ingex_parquet_files(
        gcs_bucket=config.gcs_bucket,
        blob_prefix="bsky_posts",
        start=config.posts_start,
        end=config.posts_end,
    )
    if not post_paths:
        raise ValueError(
            f"No bsky_posts Parquet files found for {config.posts_start.isoformat()} "
            f"to {config.posts_end.isoformat()}"
        )
    logger.info(
        "Found %s bsky_posts files in %.1fs",
        f"{len(post_paths):,}",
        time.monotonic() - source_listing_started,
    )
    politics_enabled = config.max_political_candidates_per_creation_hour > 0
    inference_paths: list[str] = []
    inference_timestamps: list[datetime] = []
    inference_end = config.posts_end + timedelta(
        days=config.political_inference_window_padding_days
    )
    if politics_enabled:
        inference_listing_started = time.monotonic()
        logger.info(
            "Listing bsky_inferences source files over [%s, %s)",
            config.posts_start.isoformat(),
            inference_end.isoformat(),
        )
        inference_paths, inference_timestamps = ingex.list_ingex_parquet_files(
            gcs_bucket=config.gcs_bucket,
            blob_prefix="bsky_inferences",
            start=config.posts_start,
            end=inference_end,
        )
        if not inference_paths:
            raise ValueError(
                "Political candidate discovery is enabled but no bsky_inferences "
                "Parquet files were found"
            )
        logger.info(
            "Found %s bsky_inferences files in %.1fs",
            f"{len(inference_paths):,}",
            time.monotonic() - inference_listing_started,
        )
    else:
        logger.info("Political candidate discovery is disabled; skipping inference listing")

    artifact_suffix = out_dir.name
    bundle_path = out_dir / f"post_universe_{artifact_suffix}"
    bundle_partial_path = out_dir / f"post_universe_{artifact_suffix}.partial"
    bundle_partial_path.mkdir(parents=True, exist_ok=False)
    posts_path = bundle_partial_path / "posts"
    required_posts_path = bundle_partial_path / "required_posts"
    candidate_sources_path = bundle_partial_path / "candidate_sources"
    missing_required_posts_path = bundle_partial_path / "missing_required_posts"
    post_sources_path = bundle_partial_path / f"post_sources_{artifact_suffix}.json"
    inference_sources_path = (
        bundle_partial_path / f"inference_sources_{artifact_suffix}.json"
        if politics_enabled
        else None
    )
    ingex.write_source_manifest(
        post_sources_path,
        ingex.build_source_manifest(
            gcs_bucket=config.gcs_bucket,
            blob_prefix="bsky_posts",
            start=config.posts_start,
            end=config.posts_end,
            paths=post_paths,
            timestamps=post_timestamps,
        ),
    )
    if inference_sources_path is not None:
        ingex.write_source_manifest(
            inference_sources_path,
            ingex.build_source_manifest(
                gcs_bucket=config.gcs_bucket,
                blob_prefix="bsky_inferences",
                start=config.posts_start,
                end=inference_end,
                paths=inference_paths,
                timestamps=inference_timestamps,
            ),
        )
    logger.info("Saved exact source-file manifests under %s", bundle_partial_path)

    staging_root = out_dir / f"_post_selection_staging_{artifact_suffix}.partial"
    staging_root.mkdir(parents=True, exist_ok=False)
    required_rows_path = staging_root / "required_rows"
    normalized_posts_path = staging_root / "normalized_posts"
    normalized_inferences_path = (
        staging_root / "normalized_inferences" if politics_enabled else None
    )
    base_posts_shards_path = staging_root / "base_posts_shards"
    random_candidate_shards_path = staging_root / "random_candidate_shards"
    political_eligible_shards_path = staging_root / "political_eligible_shards"
    political_by_date_path = staging_root / "political_by_date"
    selected_political_shards_path = staging_root / "selected_political_shards"
    final_posts_routed_path = staging_root / "final_posts_routed"
    final_candidates_routed_path = staging_root / "final_candidates_routed"

    required_started = time.monotonic()
    logger.info(
        "Phase 3/7: routing positive and history requirement rows into %s URI partitions",
        config.post_selection_partition_count,
    )
    post_selection_artifacts.materialize_required_rows(
        query_positives_lf=query_positives_lf,
        history_post_uris_lf=history_post_uris_lf,
        output_path=required_rows_path,
        partition_count=config.post_selection_partition_count,
    )
    logger.info(
        "Finished routing required-post rows in %.1fs",
        time.monotonic() - required_started,
    )
    logger.info("Phase 4/7: normalizing and partitioning raw post source rows")
    post_selection_artifacts.materialize_source_rows(
        post_paths=post_paths,
        inference_paths=inference_paths,
        config=config,
        normalized_posts_path=normalized_posts_path,
        normalized_inferences_path=normalized_inferences_path,
        logger=logger,
    )
    logger.info("Phase 5/7: resolving metadata and memberships by URI partition")
    selection_stats = post_selection_artifacts.process_uri_partitions(
        required_rows_path=required_rows_path,
        normalized_posts_path=normalized_posts_path,
        normalized_inferences_path=normalized_inferences_path,
        required_posts_path=required_posts_path,
        missing_required_posts_path=missing_required_posts_path,
        base_posts_shards_path=base_posts_shards_path,
        random_candidate_shards_path=random_candidate_shards_path,
        political_eligible_shards_path=political_eligible_shards_path,
        config=config,
        logger=logger,
    )
    logger.info(
        "Resolved URI partitions: valid_posts=%s required=%s missing_required=%s",
        f"{selection_stats['post_source_stats'].get('unique_valid_post_count', 0):,}",
        f"{selection_stats['required_post_stats']['required_post_count']:,}",
        f"{selection_stats['required_post_stats']['missing_required_post_count']:,}",
    )
    logger.info("Phase 6/7: applying political caps and routing final datasets")
    political_hour_stats = post_selection_artifacts.select_political_candidates(
        political_eligible_shards_path=political_eligible_shards_path,
        political_by_date_path=political_by_date_path,
        selected_political_shards_path=selected_political_shards_path,
        config=config,
        logger=logger,
    )
    post_selection_artifacts.materialize_final_routes(
        base_posts_shards_path=base_posts_shards_path,
        random_candidate_shards_path=random_candidate_shards_path,
        selected_political_shards_path=selected_political_shards_path,
        final_posts_routed_path=final_posts_routed_path,
        final_candidates_routed_path=final_candidates_routed_path,
        partition_count=config.post_selection_partition_count,
        logger=logger,
    )
    logger.info("Phase 7/7: writing and validating public artifact partitions")
    output_stats = post_selection_artifacts.write_and_validate_public_outputs(
        final_posts_routed_path=final_posts_routed_path,
        final_candidates_routed_path=final_candidates_routed_path,
        posts_path=posts_path,
        required_posts_path=required_posts_path,
        candidate_sources_path=candidate_sources_path,
        missing_required_posts_path=missing_required_posts_path,
        config=config,
        logger=logger,
    )

    logger.info("Removing successful staging data and publishing the completed bundle")
    shutil.rmtree(staging_root)
    bundle_partial_path.replace(bundle_path)
    logger.info("Published post-universe bundle at %s", bundle_path)
    final_posts_path = bundle_path / "posts"
    final_required_posts_path = bundle_path / "required_posts"
    final_candidate_sources_path = bundle_path / "candidate_sources"
    final_missing_required_posts_path = bundle_path / "missing_required_posts"
    final_post_sources_path = bundle_path / post_sources_path.name
    final_inference_sources_path = (
        bundle_path / inference_sources_path.name
        if inference_sources_path is not None
        else None
    )

    runtime_seconds = time.time() - started_at
    political_eligible_count = sum(
        row["eligible_count"] for row in political_hour_stats
    )
    political_selected_count = sum(
        row["selected_count"] for row in political_hour_stats
    )
    summary = {
        "parameters": {
            "posts_start": config.posts_start.isoformat(),
            "posts_end": config.posts_end.isoformat(),
            "random_candidate_sampling_fraction": (
                config.random_candidate_sampling_fraction
            ),
            "max_political_candidates_per_creation_hour": (
                config.max_political_candidates_per_creation_hour
            ),
            "political_score_threshold": config.political_score_threshold,
            "political_inference_window_padding_days": (
                config.political_inference_window_padding_days
            ),
            "post_selection_partition_count": config.post_selection_partition_count,
            "random_seed": config.random_seed,
        },
        "input": {
            "query_selection_dir": str(query_selection_dir),
            "user_history_dir": str(user_history_dir),
            "history_post_uris_path": str(history_post_uris_path),
            "min_query_hour": min_query_hour.isoformat(),
            "max_query_hour": max_query_hour.isoformat(),
            "post_file_count": len(post_paths),
            "inference_file_count": len(inference_paths),
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
            "inference_sources_path": (
                str(Path(bundle_path.name) / final_inference_sources_path.name)
                if final_inference_sources_path is not None
                else None
            ),
            **output_stats,
        },
        **selection_stats,
        "political_candidate_stats": {
            "eligible_count": political_eligible_count,
            "selected_count": political_selected_count,
            "discarded_count": political_eligible_count - political_selected_count,
            "by_creation_hour": political_hour_stats,
        },
        "runtime_seconds": runtime_seconds,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    stage_info_lines = [
        "stage: post_selection",
        f"runtime_seconds: {runtime_seconds:.2f}",
        f"post_universe_path: {bundle_path.name}",
        f"posts_path: {Path(bundle_path.name) / 'posts'}",
        f"required_posts_path: {Path(bundle_path.name) / 'required_posts'}",
        f"candidate_sources_path: {Path(bundle_path.name) / 'candidate_sources'}",
        f"missing_required_posts_path: {Path(bundle_path.name) / 'missing_required_posts'}",
        f"post_sources_path: {Path(bundle_path.name) / final_post_sources_path.name}",
        f"post_file_count: {len(post_paths)}",
        f"inference_file_count: {len(inference_paths)}",
        f"post_source_row_count: {selection_stats['post_source_stats'].get('post_source_row_count', 0)}",
        f"invalid_post_row_count: {selection_stats['post_source_stats'].get('invalid_post_row_count', 0)}",
        f"duplicate_post_row_count: {selection_stats['post_source_stats'].get('duplicate_post_row_count', 0)}",
        f"duplicate_post_uri_count: {selection_stats['post_source_stats'].get('duplicate_post_uri_count', 0)}",
        f"inference_source_row_count: {selection_stats['inference_source_stats'].get('inference_source_row_count', 0)}",
        f"invalid_inference_row_count: {selection_stats['inference_source_stats'].get('invalid_inference_row_count', 0)}",
        f"post_count: {output_stats['post_count']}",
        f"required_post_count: {selection_stats['required_post_stats']['required_post_count']}",
        f"found_required_post_count: {selection_stats['required_post_stats']['found_required_post_count']}",
        f"missing_required_post_count: {selection_stats['required_post_stats']['missing_required_post_count']}",
        f"candidate_source_count: {output_stats['candidate_source_count']}",
        f"random_candidate_count: {output_stats['random_candidate_count']}",
        f"political_candidate_count: {output_stats['political_candidate_count']}",
        f"random_and_political_candidate_count: {output_stats['random_and_political_candidate_count']}",
        f"inference_covered_post_count: {output_stats['inference_covered_post_count']}",
        f"political_labeled_post_count: {output_stats['political_labeled_post_count']}",
        f"political_eligible_count: {political_eligible_count}",
        f"political_selected_count: {political_selected_count}",
        f"political_discarded_count: {political_eligible_count - political_selected_count}",
    ]
    if final_inference_sources_path is not None:
        stage_info_lines.append(
            f"inference_sources_path: {Path(bundle_path.name) / final_inference_sources_path.name}"
        )
    (out_dir / "stage_info.txt").write_text("\n".join(stage_info_lines) + "\n")
    logger.info(
        "Post selection completed in %.2fs: posts=%s required=%s random=%s political=%s",
        runtime_seconds,
        f"{output_stats['post_count']:,}",
        f"{selection_stats['required_post_stats']['required_post_count']:,}",
        f"{output_stats['random_candidate_count']:,}",
        f"{output_stats['political_candidate_count']:,}",
    )
    artifacts = {
        "post_universe_path": str(bundle_path),
        "posts_path": str(final_posts_path),
        "required_posts_path": str(final_required_posts_path),
        "candidate_sources_path": str(final_candidate_sources_path),
        "missing_required_posts_path": str(final_missing_required_posts_path),
        "post_sources_path": str(final_post_sources_path),
    }
    if final_inference_sources_path is not None:
        artifacts["inference_sources_path"] = str(final_inference_sources_path)
    return {"output_dir": out_dir, "artifacts": artifacts}
