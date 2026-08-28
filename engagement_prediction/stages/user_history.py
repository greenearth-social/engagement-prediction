"""Stage 2: select bounded as-of user histories for Stage 1 queries.

The expensive history construction is split into stable DID-hash partitions.
While processing each partition, the stage writes a physical URI shard holding
the locally unique posts retained in its histories. It then stream-repartitions
those shards by post-URI hash and deduplicates one resulting URI partition at a
time. This second partitioning is what provides global URI uniqueness without
collecting the full post set in memory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import shutil
import time
from typing import Any, Dict

import polars as pl

from engagement_prediction.data import ingex, likes, partition_workers, source_manifests
from engagement_prediction.data.parquet import (
    load_parquet_from_prior,
    sink_partitioned_parquet,
)
from engagement_prediction.data import user_history as history_data
from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline.lineage import resolve_recorded_stage_lineage
from engagement_prediction.pipeline.logging import get_stage_logger


@dataclass(frozen=True)
class UserHistoryConfig:
    """Validated history cap and physical user-partition count."""

    max_history_posts_per_query: int
    user_history_partition_count: int
    data_partition_worker_count: int


def build_config(args: argparse.Namespace) -> UserHistoryConfig:
    """Parse the Stage 2 CLI settings that bound each partition's work."""

    max_history_posts_per_query = int(args.max_history_posts_per_query)
    if max_history_posts_per_query <= 0:
        raise ValueError("max_history_posts_per_query must be positive")
    user_history_partition_count = int(args.user_history_partition_count)
    if user_history_partition_count <= 0:
        raise ValueError("user_history_partition_count must be positive")
    data_partition_worker_count = int(args.data_partition_worker_count)
    if data_partition_worker_count <= 0:
        raise ValueError("data_partition_worker_count must be positive")
    return UserHistoryConfig(
        max_history_posts_per_query=max_history_posts_per_query,
        user_history_partition_count=user_history_partition_count,
        data_partition_worker_count=data_partition_worker_count,
    )


def _publish_partitioned_dataset(
    lf: pl.LazyFrame,
    *,
    partial_path: Path,
    final_path: Path,
) -> None:
    """Stream a hash-partitioned relation and publish it by atomic rename."""

    sink_partitioned_parquet(
        lf,
        output_path=partial_path,
        key="_user_partition",
    )
    partial_path.replace(final_path)


def _materialize_partitioned_inputs(
    *,
    queries_lf: pl.LazyFrame,
    source_like_paths: list[str],
    source_start,
    source_end,
    max_query_hour,
    partition_count: int,
    query_partitions_partial_path: Path,
    query_partitions_path: Path,
    queried_user_likes_partial_path: Path,
    queried_user_likes_path: Path,
    logger: logging.Logger,
) -> None:
    """Co-locate each queried user's queries and source-window likes.

    The semi-join limits the raw like rescan to DIDs that survived Stage 1.
    Likes remain otherwise complete because unselected activity is valid user
    history for a later query.
    """

    partition_expr = history_data.user_partition_expr(partition_count)
    logger.info("Partitioning selected queries into %s stable user buckets", partition_count)
    _publish_partitioned_dataset(
        queries_lf.with_columns(partition_expr),
        partial_path=query_partitions_partial_path,
        final_path=query_partitions_path,
    )

    selected_users_lf = queries_lf.select("did").unique()
    logger.info("Scanning like history for users represented in selected queries")
    queried_user_likes_lf = (
        likes.prepare_likes(
            ingex.scan_parquet_files(source_like_paths),
            start=source_start,
            end=source_end,
        )
        .filter(pl.col("like_created_at") < pl.lit(max_query_hour))
        .join(selected_users_lf, on="did", how="semi")
        .with_columns(partition_expr)
    )
    _publish_partitioned_dataset(
        queried_user_likes_lf,
        partial_path=queried_user_likes_partial_path,
        final_path=queried_user_likes_path,
    )


def _write_query_histories(
    *,
    query_partitions_path: Path,
    queried_user_likes_path: Path,
    partial_output_path: Path,
    history_post_uri_shards_path: Path,
    config: UserHistoryConfig,
    logger: logging.Logger,
) -> tuple[dict[str, dict[str, int]], int, list[dict[str, Any]]]:
    """Write histories and one locally deduplicated URI shard per user partition.

    The shard files reflect the DID-based processing layout. They are not yet
    globally unique because the same post may occur in histories from users in
    different partitions.
    """
    partial_output_path.mkdir(parents=True, exist_ok=False)
    history_post_uri_shards_path.mkdir(parents=True, exist_ok=False)
    worker_kwargs: list[dict[str, Any]] = []
    for partition_id in range(config.user_history_partition_count):
        query_paths = history_data.partition_parquet_paths(
            query_partitions_path,
            partition_id,
        )
        if not query_paths:
            continue
        like_paths = history_data.partition_parquet_paths(
            queried_user_likes_path,
            partition_id,
        )
        worker_kwargs.append({
            "query_paths": query_paths,
            "like_paths": like_paths,
            "partial_output_path": partial_output_path,
            "history_post_uri_shards_path": history_post_uri_shards_path,
            "partition_id": partition_id,
            "max_history_posts_per_query": config.max_history_posts_per_query,
        })

    logger.info(
        "Constructing histories across %s populated user partitions with up to %s workers",
        len(worker_kwargs),
        config.data_partition_worker_count,
    )

    def log_result(result: dict[str, Any]) -> None:
        logger.info(
            "Completed user-history partition %s in %.1fs with %s queries",
            result["partition_id"],
            result["runtime_seconds"],
            f"{result['query_count']:,}",
        )

    results, effective_worker_count = partition_workers.run_partition_jobs(
        worker=history_data.write_query_history_partition,
        worker_kwargs=worker_kwargs,
        worker_count=config.data_partition_worker_count,
        on_result=log_result,
    )
    output_rows = sum(int(result["query_count"]) for result in results)
    if output_rows == 0:
        raise ValueError("User history selection produced no query histories")
    return (
        history_data.merge_partition_stats([result["stats"] for result in results]),
        effective_worker_count,
        [
            {
                "partition_id": result["partition_id"],
                "query_count": result["query_count"],
                "runtime_seconds": result["runtime_seconds"],
            }
            for result in results
        ],
    )


def _write_history_post_uris(
    *,
    history_post_uri_shards_path: Path,
    routed_history_post_uris_path: Path,
    partial_output_path: Path,
    partition_count: int,
    worker_count: int,
    logger: logging.Logger,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Repartition local URI shards and publish globally unique URI partitions.

    A source shard can contain URIs belonging to many destination partitions.
    The streaming sink redistributes rows by stable ``subject_uri`` hash, which
    guarantees that every occurrence of one URI lands in the same destination
    partition. Loading, deduplicating, and writing one such partition at a time
    bounds the in-memory working set.
    """
    partial_output_path.mkdir(parents=True, exist_ok=False)
    shard_paths = sorted(history_post_uri_shards_path.glob("*.parquet"))
    if not shard_paths:
        empty_df = history_data.history_post_uri_frame(set())
        history_data.validate_history_post_uri_partition(
            empty_df,
            partition_id=0,
            partition_count=partition_count,
        )
        empty_df.write_parquet(
            partial_output_path / "part-00000.parquet",
            compression="zstd",
        )
        return 0, 0, []

    # Repartition physical user-partition shards by the logical URI hash key.
    sink_partitioned_parquet(
        pl.scan_parquet(shard_paths).with_columns(
            history_data.history_post_partition_expr(partition_count)
        ),
        output_path=routed_history_post_uris_path,
        key="_history_post_partition",
    )

    # Each URI is confined to one partition, so partition-local uniqueness also
    # guarantees uniqueness across the complete published dataset.
    worker_kwargs: list[dict[str, Any]] = []
    for partition_id in range(partition_count):
        partition_paths = history_data.history_post_partition_parquet_paths(
            routed_history_post_uris_path,
            partition_id,
        )
        if not partition_paths:
            continue
        worker_kwargs.append({
            "partition_paths": partition_paths,
            "partial_output_path": partial_output_path,
            "partition_id": partition_id,
            "partition_count": partition_count,
        })

    logger.info(
        "Deduplicating %s populated history-post URI partitions with up to %s workers",
        len(worker_kwargs),
        worker_count,
    )

    def log_result(result: dict[str, Any]) -> None:
        logger.info(
            "Completed history-post URI partition %s/%s in %.1fs: unique=%s",
            result["partition_id"] + 1,
            partition_count,
            result["runtime_seconds"],
            f"{result['unique_history_post_count']:,}",
        )

    results, effective_worker_count = partition_workers.run_partition_jobs(
        worker=history_data.write_history_post_uri_partition,
        worker_kwargs=worker_kwargs,
        worker_count=worker_count,
        on_result=log_result,
    )
    return (
        sum(int(result["unique_history_post_count"]) for result in results),
        effective_worker_count,
        results,
    )


def _log_stats(logger: logging.Logger, stats: dict[str, dict[str, int]]) -> None:
    """Log history coverage and truncation separately for every split."""

    for split, values in stats.items():
        logger.info(
            "%s: queries=%s users=%s empty=%s truncated=%s retained_items=%s",
            split,
            f"{values['query_count']:,}",
            f"{values['unique_user_count']:,}",
            f"{values['empty_history_count']:,}",
            f"{values['truncated_history_count']:,}",
            f"{values['retained_history_item_count']:,}",
        )


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    """Run Stage 2 and publish aligned histories plus their unique post URIs."""

    out_dir = context.new_stage_dir("02_user_history")
    logger = get_stage_logger("02_USER_HISTORY", log_file=out_dir / "stage.log")
    started_at = time.time()
    config = build_config(args)

    # Stage 1's manifest fixes both query keys and the raw-like snapshot. A
    # direct Stage 2 rerun must use those exact inputs rather than relist GCS.
    lineage = resolve_recorded_stage_lineage(
        context,
        terminal_stage_folder="01_query_selection",
        ancestor_stage_folders=("00_source_metadata",),
    )
    source_metadata_dir = lineage["00_source_metadata"]
    query_selection_dir = lineage["01_query_selection"]
    queries_lf = load_parquet_from_prior(query_selection_dir, "queries_")
    history_data.validate_queries_schema(queries_lf)
    query_summary = queries_lf.select(
        pl.len().alias("query_count"),
        pl.col("query_hour").min().alias("min_query_hour"),
        pl.col("query_hour").max().alias("max_query_hour"),
    ).collect(engine="streaming")
    query_count = query_summary.item(0, "query_count")
    if query_count == 0:
        raise ValueError("query_selection produced no queries")
    min_query_hour = query_summary.item(0, "min_query_hour")
    max_query_hour = query_summary.item(0, "max_query_hour")

    like_snapshot = source_manifests.load_source_snapshot(
        query_selection_dir,
        manifest_prefix="like_sources_",
        expected_blob_prefix="bsky_likes",
    )
    like_sources_path = like_snapshot.path
    source_manifest = like_snapshot.manifest
    source_like_paths = list(like_snapshot.file_uris)
    source_start = like_snapshot.start
    source_end = like_snapshot.end
    if min_query_hour < source_start or max_query_hour >= source_end:
        raise ValueError("Stage 1 query hours must fall within its recorded like source window")

    artifact_suffix = out_dir.name
    query_partitions_path = out_dir / f"_query_partitions_{artifact_suffix}"
    query_partitions_partial_path = out_dir / f"_query_partitions_{artifact_suffix}.partial"
    queried_user_likes_path = out_dir / f"_queried_user_likes_{artifact_suffix}"
    queried_user_likes_partial_path = out_dir / f"_queried_user_likes_{artifact_suffix}.partial"
    query_histories_path = out_dir / f"query_histories_{artifact_suffix}"
    query_histories_partial_path = out_dir / f"query_histories_{artifact_suffix}.partial"
    history_post_uri_shards_path = out_dir / f"_history_post_uri_shards_{artifact_suffix}.partial"
    routed_history_post_uris_path = (
        out_dir / f"_history_post_uri_partitions_{artifact_suffix}.partial"
    )
    history_post_uris_path = out_dir / f"history_post_uris_{artifact_suffix}"
    history_post_uris_partial_path = out_dir / f"history_post_uris_{artifact_suffix}.partial"

    # First partition by DID so every query and every eligible like for a user
    # is processed together. The intermediate datasets are deleted on success.
    _materialize_partitioned_inputs(
        queries_lf=queries_lf,
        source_like_paths=source_like_paths,
        source_start=source_start,
        source_end=source_end,
        max_query_hour=max_query_hour,
        partition_count=config.user_history_partition_count,
        query_partitions_partial_path=query_partitions_partial_path,
        query_partitions_path=query_partitions_path,
        queried_user_likes_partial_path=queried_user_likes_partial_path,
        queried_user_likes_path=queried_user_likes_path,
        logger=logger,
    )
    stats, history_worker_count, history_partition_stats = _write_query_histories(
        query_partitions_path=query_partitions_path,
        queried_user_likes_path=queried_user_likes_path,
        partial_output_path=query_histories_partial_path,
        history_post_uri_shards_path=history_post_uri_shards_path,
        config=config,
        logger=logger,
    )

    output_query_count = sum(values["query_count"] for values in stats.values())
    if output_query_count != query_count:
        raise ValueError(
            f"Query history count {output_query_count:,} does not match Stage 1 query count {query_count:,}"
        )
    # History construction emits locally unique URI shards. Repartitioning the
    # shards by URI makes partition-local deduplication globally sufficient.
    (
        unique_history_post_count,
        history_post_worker_count,
        history_post_partition_stats,
    ) = _write_history_post_uris(
        history_post_uri_shards_path=history_post_uri_shards_path,
        routed_history_post_uris_path=routed_history_post_uris_path,
        partial_output_path=history_post_uris_partial_path,
        partition_count=config.user_history_partition_count,
        worker_count=config.data_partition_worker_count,
        logger=logger,
    )
    shutil.rmtree(history_post_uri_shards_path)
    if routed_history_post_uris_path.exists():
        shutil.rmtree(routed_history_post_uris_path)
    query_histories_partial_path.replace(query_histories_path)
    history_post_uris_partial_path.replace(history_post_uris_path)
    # These DID-partitioned datasets are required only while constructing and
    # validating the two public outputs above. Keep them on failed runs for
    # diagnosis, but do not retain them in a successfully completed artifact.
    logger.info("Removing successful-run query and queried-user-like staging datasets")
    shutil.rmtree(query_partitions_path)
    shutil.rmtree(queried_user_likes_path)

    _log_stats(logger, stats)
    logger.info(
        "Selected %s unique history post URIs",
        f"{unique_history_post_count:,}",
    )

    runtime_seconds = time.time() - started_at
    summary = {
        "parameters": {
            "max_history_posts_per_query": config.max_history_posts_per_query,
            "user_history_partition_count": config.user_history_partition_count,
            "data_partition_worker_count": config.data_partition_worker_count,
        },
        "input": {
            "source_metadata_dir": str(source_metadata_dir),
            "query_selection_dir": str(query_selection_dir),
            "query_count": query_count,
            "like_sources_file": like_sources_path.name,
            "like_file_count": len(source_like_paths),
            "source_start": source_manifest.get("start"),
            "source_end": source_manifest.get("end"),
        },
        "outputs": {
            "query_histories_path": query_histories_path.name,
            "history_post_uris_path": history_post_uris_path.name,
            "query_count": output_query_count,
            "unique_history_post_count": unique_history_post_count,
        },
        "partition_processing": {
            "history_partition_worker_count": history_worker_count,
            "history_partition_stats": history_partition_stats,
            "history_post_partition_worker_count": history_post_worker_count,
            "history_post_partition_stats": history_post_partition_stats,
        },
        "selection_stats_by_split": stats,
        "runtime_seconds": runtime_seconds,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "stage_info.txt").write_text(
        "\n".join([
            "stage: user_history",
            f"runtime_seconds: {runtime_seconds:.2f}",
            f"input_queries: {query_count}",
            f"input_like_files: {len(source_like_paths)}",
            f"max_history_posts_per_query: {config.max_history_posts_per_query}",
            f"user_history_partition_count: {config.user_history_partition_count}",
            f"data_partition_worker_count: {config.data_partition_worker_count}",
            f"effective_history_partition_worker_count: {history_worker_count}",
            f"effective_history_post_partition_worker_count: {history_post_worker_count}",
            f"query_histories_path: {query_histories_path.name}",
            f"history_post_uris_path: {history_post_uris_path.name}",
            f"unique_history_post_count: {unique_history_post_count}",
        ])
        + "\n"
    )
    logger.info(
        "User history selection completed in %.2fs with %s query histories",
        runtime_seconds,
        f"{output_query_count:,}",
    )
    return {
        "output_dir": out_dir,
        "artifacts": {
            "query_histories_path": str(query_histories_path),
            "history_post_uris_path": str(history_post_uris_path),
        },
    }
