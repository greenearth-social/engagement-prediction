"""Stage 3: resolve required posts/replies and build a random candidate reservoir."""

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
    partition_count = int(args.post_selection_partition_count)
    if partition_count <= 0:
        raise ValueError("post_selection_partition_count must be positive")
    return PostSelectionConfig(
        gcs_bucket=str(args.gcs_bucket),
        posts_start=posts_start,
        posts_end=posts_end,
        random_candidate_sampling_fraction=fraction,
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


def _find_post_sources_path(query_selection_dir: Path) -> Path:
    candidates = sorted(query_selection_dir.glob("post_sources_*.json"))
    if not candidates:
        raise FileNotFoundError(
            "Stage 3 requires post_sources_*.json from query_selection. "
            "Rerun Stage 1 with the current query-selection implementation."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Expected one post_sources_*.json under {query_selection_dir}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _load_exact_post_snapshot(
    query_selection_dir: Path,
    config: PostSelectionConfig,
) -> tuple[dict[str, Any], list[str]]:
    source_path = _find_post_sources_path(query_selection_dir)
    manifest = ingex.load_source_manifest(source_path)
    if manifest.get("gcs_bucket") != config.gcs_bucket:
        raise ValueError("Stage 1 post source bucket does not match Stage 3 gcs_bucket")
    if manifest.get("blob_prefix") != "bsky_posts":
        raise ValueError("Stage 1 post source manifest must use bsky_posts")
    manifest_start = ingex.parse_utc_datetime(manifest.get("start"), field_name="posts_start")
    manifest_end = ingex.parse_utc_datetime(manifest.get("end"), field_name="posts_end")
    if manifest_start != config.posts_start or manifest_end != config.posts_end:
        raise ValueError(
            "Stage 1 post source bounds do not match Stage 3 posts_start/posts_end"
        )
    return manifest, [entry["uri"] for entry in manifest["files"]]


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = context.new_stage_dir("03_post_selection")
    logger = get_stage_logger("03_POST_SELECTION", log_file=out_dir / "stage.log")
    started_at = time.time()
    config = build_config(args)
    logger.info(
        "Starting post selection: source_window=[%s, %s) partitions=%s random_fraction=%s",
        config.posts_start.isoformat(),
        config.posts_end.isoformat(),
        config.post_selection_partition_count,
        config.random_candidate_sampling_fraction,
    )

    logger.info("Phase 1/6: resolving and validating Stage 1/2 inputs")
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
    queries_lf = scan_parquet_artifact(find_artifact_path(query_selection_dir, "queries_"))
    min_query_hour, max_query_hour = _validate_query_window(queries_lf, config)
    query_positives_lf = scan_parquet_artifact(
        find_artifact_path(query_selection_dir, "query_positives_")
    )
    try:
        history_post_uris_path = find_artifact_path(user_history_dir, "history_post_uris_")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Stage 3 requires the new Stage 2 history_post_uris_* artifact; "
            "rerun user_history with the current implementation"
        ) from exc
    history_post_uris_lf = scan_parquet_artifact(history_post_uris_path)

    logger.info("Phase 2/6: loading exact root snapshot and listing replies")
    post_manifest, post_paths = _load_exact_post_snapshot(query_selection_dir, config)
    reply_listing_started = time.monotonic()
    reply_paths, reply_timestamps = ingex.list_ingex_parquet_files(
        gcs_bucket=config.gcs_bucket,
        blob_prefix="bsky_replies",
        start=config.posts_start,
        end=config.posts_end,
    )
    if not reply_paths:
        raise ValueError(
            f"No bsky_replies Parquet files found for {config.posts_start.isoformat()} "
            f"to {config.posts_end.isoformat()}"
        )
    logger.info(
        "Resolved %s exact bsky_posts files and found %s bsky_replies files in %.1fs",
        f"{len(post_paths):,}",
        f"{len(reply_paths):,}",
        time.monotonic() - reply_listing_started,
    )

    artifact_suffix = out_dir.name
    bundle_path = out_dir / f"post_universe_{artifact_suffix}"
    bundle_partial_path = out_dir / f"post_universe_{artifact_suffix}.partial"
    bundle_partial_path.mkdir(parents=True, exist_ok=False)
    posts_path = bundle_partial_path / "posts"
    required_posts_path = bundle_partial_path / "required_posts"
    candidate_sources_path = bundle_partial_path / "candidate_sources"
    missing_required_posts_path = bundle_partial_path / "missing_required_posts"
    post_sources_path = bundle_partial_path / f"post_sources_{artifact_suffix}.json"
    reply_sources_path = bundle_partial_path / f"reply_sources_{artifact_suffix}.json"
    ingex.write_source_manifest(post_sources_path, post_manifest)
    ingex.write_source_manifest(
        reply_sources_path,
        ingex.build_source_manifest(
            gcs_bucket=config.gcs_bucket,
            blob_prefix="bsky_replies",
            start=config.posts_start,
            end=config.posts_end,
            paths=reply_paths,
            timestamps=reply_timestamps,
        ),
    )

    staging_root = out_dir / f"_post_selection_staging_{artifact_suffix}.partial"
    staging_root.mkdir(parents=True, exist_ok=False)
    required_rows_path = staging_root / "required_rows"
    normalized_posts_path = staging_root / "normalized_posts"
    normalized_replies_path = staging_root / "normalized_replies"
    base_posts_shards_path = staging_root / "base_posts_shards"
    random_candidate_shards_path = staging_root / "random_candidate_shards"

    logger.info(
        "Phase 3/6: routing positive and history requirements into %s URI partitions",
        config.post_selection_partition_count,
    )
    # write required (positive and history) posts, partitioned
    post_selection_artifacts.materialize_required_rows(
        query_positives_lf=query_positives_lf,
        history_post_uris_lf=history_post_uris_lf,
        output_path=required_rows_path,
        partition_count=config.post_selection_partition_count,
    )

    # write ALL posts and replies, partitioned (subset of cols)
    logger.info("Phase 4/6: normalizing and partitioning root and reply source rows")
    post_selection_artifacts.materialize_source_rows(
        post_paths=post_paths,
        reply_paths=reply_paths,
        config=config,
        normalized_posts_path=normalized_posts_path,
        normalized_replies_path=normalized_replies_path,
        logger=logger,
    )

    logger.info("Phase 5/6: resolving metadata and memberships by URI partition")
    selection_stats = post_selection_artifacts.process_uri_partitions(
        required_rows_path=required_rows_path,
        normalized_posts_path=normalized_posts_path,
        normalized_replies_path=normalized_replies_path,
        required_posts_path=required_posts_path,
        missing_required_posts_path=missing_required_posts_path,
        base_posts_shards_path=base_posts_shards_path,
        random_candidate_shards_path=random_candidate_shards_path,
        config=config,
        logger=logger,
    )
    required_stats = selection_stats["required_post_stats"]
    logger.info(
        "Resolved URI partitions: roots=%s replies=%s required=%s missing_history=%s",
        f"{selection_stats['root_source_stats'].get('unique_valid_count', 0):,}",
        f"{selection_stats['reply_source_stats'].get('unique_valid_count', 0):,}",
        f"{required_stats['required_post_count']:,}",
        f"{required_stats['missing_history_required_post_count']:,}",
    )

    logger.info("Phase 6/6: validating and publishing public datasets")
    output_stats = post_selection_artifacts.write_and_validate_public_outputs(
        base_posts_shards_path=base_posts_shards_path,
        random_candidate_shards_path=random_candidate_shards_path,
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
    final_posts_path = bundle_path / "posts"
    final_required_posts_path = bundle_path / "required_posts"
    final_candidate_sources_path = bundle_path / "candidate_sources"
    final_missing_required_posts_path = bundle_path / "missing_required_posts"
    final_post_sources_path = bundle_path / post_sources_path.name
    final_reply_sources_path = bundle_path / reply_sources_path.name
    logger.info("Published post-universe bundle at %s", bundle_path)

    runtime_seconds = time.time() - started_at
    summary = {
        "gcs_bucket": config.gcs_bucket,
        "posts_start": config.posts_start.isoformat(),
        "posts_end": config.posts_end.isoformat(),
        "parameters": {
            "random_candidate_sampling_fraction": config.random_candidate_sampling_fraction,
            "post_selection_partition_count": config.post_selection_partition_count,
            "random_seed": config.random_seed,
        },
        "input": {
            "query_selection_dir": str(query_selection_dir),
            "user_history_dir": str(user_history_dir),
            "query_range": {
                "min_query_hour": min_query_hour.isoformat(),
                "max_query_hour": max_query_hour.isoformat(),
            },
            "post_file_count": len(post_paths),
            "reply_file_count": len(reply_paths),
            "root_source_stats": selection_stats["root_source_stats"],
            "reply_source_stats": selection_stats["reply_source_stats"],
        },
        "required_post_stats": required_stats,
        "outputs": {
            "post_universe_path": bundle_path.name,
            "posts_path": str(Path(bundle_path.name) / "posts"),
            "required_posts_path": str(Path(bundle_path.name) / "required_posts"),
            "candidate_sources_path": str(Path(bundle_path.name) / "candidate_sources"),
            "missing_required_posts_path": str(
                Path(bundle_path.name) / "missing_required_posts"
            ),
            "post_sources_path": str(Path(bundle_path.name) / final_post_sources_path.name),
            "reply_sources_path": str(
                Path(bundle_path.name) / final_reply_sources_path.name
            ),
            **output_stats,
        },
        "runtime_seconds": runtime_seconds,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "stage_info.txt").write_text(
        "\n".join(
            [
                "stage: post_selection",
                f"runtime_seconds: {runtime_seconds:.2f}",
                f"post_file_count: {len(post_paths)}",
                f"reply_file_count: {len(reply_paths)}",
                f"post_count: {output_stats['post_count']}",
                f"root_post_count: {output_stats['root_post_count']}",
                f"reply_post_count: {output_stats['reply_post_count']}",
                f"random_candidate_count: {output_stats['random_candidate_count']}",
                f"missing_history_post_count: {required_stats['missing_history_required_post_count']}",
                f"history_resolved_as_root_count: {required_stats['history_resolved_as_root_count']}",
                f"history_resolved_as_reply_count: {required_stats['history_resolved_as_reply_count']}",
                f"root_reply_overlap_count: {required_stats['root_reply_overlap_count']}",
                f"post_sources_path: {Path(bundle_path.name) / final_post_sources_path.name}",
                f"reply_sources_path: {Path(bundle_path.name) / final_reply_sources_path.name}",
            ]
        )
        + "\n"
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
