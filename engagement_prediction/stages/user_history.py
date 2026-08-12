"""Stage 2: select bounded as-of user histories for Stage 1 queries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict

import polars as pl

from engagement_prediction.data import ingex, likes
from engagement_prediction.data.parquet import load_parquet_from_prior
from engagement_prediction.data import user_history as history_data
from engagement_prediction.pipeline.core import Context
from utils.helpers import get_stage_logger


@dataclass(frozen=True)
class UserHistoryConfig:
    max_history_posts_per_query: int
    user_history_partition_count: int


def build_config(args: argparse.Namespace) -> UserHistoryConfig:
    max_history_posts_per_query = int(args.max_history_posts_per_query)
    if max_history_posts_per_query <= 0:
        raise ValueError("max_history_posts_per_query must be positive")
    user_history_partition_count = int(args.user_history_partition_count)
    if user_history_partition_count <= 0:
        raise ValueError("user_history_partition_count must be positive")
    return UserHistoryConfig(
        max_history_posts_per_query=max_history_posts_per_query,
        user_history_partition_count=user_history_partition_count,
    )


def _find_like_sources_path(query_selection_dir: Path) -> Path:
    candidates = sorted(query_selection_dir.glob("like_sources_*.json"))
    if not candidates:
        raise FileNotFoundError(
            "Stage 2 requires like_sources_*.json from query_selection. "
            "Rerun Stage 1 with the current query-selection implementation."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Expected one like_sources_*.json artifact under {query_selection_dir}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _publish_partitioned_dataset(
    lf: pl.LazyFrame,
    *,
    partial_path: Path,
    final_path: Path,
) -> None:
    partial_path.mkdir(parents=True, exist_ok=False)
    lf.sink_parquet(
        pl.PartitionBy(
            partial_path,
            key="_user_partition",
            include_key=False,
            approximate_bytes_per_file="auto",
        ),
        compression="zstd",
        maintain_order=False,
        engine="streaming",
    )
    partial_path.replace(final_path)


def _load_partition(paths: list[Path], *, empty: pl.DataFrame | None = None) -> pl.DataFrame:
    if not paths:
        if empty is None:
            raise ValueError("Expected a non-empty Parquet partition")
        return empty
    return pl.read_parquet(paths)


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
    partition_expr = history_data.user_partition_expr(partition_count)
    logger.info("Partitioning selected queries into %s stable user buckets", partition_count)
    _publish_partitioned_dataset(
        queries_lf.with_columns(partition_expr),
        partial_path=query_partitions_partial_path,
        final_path=query_partitions_path,
    )

    selected_users_lf = queries_lf.select("did").unique()
    selected_user_count = selected_users_lf.select(pl.len()).collect(engine="streaming").item()
    logger.info("Scanning like history for %s queried users", f"{selected_user_count:,}")
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
    final_output_path: Path,
    config: UserHistoryConfig,
    logger: logging.Logger,
) -> dict[str, dict[str, int]]:
    partial_output_path.mkdir(parents=True, exist_ok=False)
    all_partition_stats: list[dict[str, dict[str, int]]] = []
    output_rows = 0
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
        queries_df = _load_partition(query_paths)
        likes_df = _load_partition(like_paths, empty=history_data.empty_likes())
        history_df, partition_stats = history_data.build_query_histories_for_partition(
            queries_df,
            likes_df,
            max_history_posts_per_query=config.max_history_posts_per_query,
        )
        history_df.write_parquet(
            partial_output_path / f"part-{partition_id:05d}.parquet",
            compression="zstd",
        )
        output_rows += history_df.height
        all_partition_stats.append(partition_stats)
        logger.info(
            "Completed user-history partition %s with %s queries",
            partition_id,
            f"{history_df.height:,}",
        )

    if output_rows == 0:
        raise ValueError("User history selection produced no query histories")
    partial_output_path.replace(final_output_path)
    return history_data.merge_partition_stats(all_partition_stats)


def _log_stats(logger: logging.Logger, stats: dict[str, dict[str, int]]) -> None:
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
    out_dir = context.new_stage_dir("02_user_history")
    logger = get_stage_logger("02_USER_HISTORY", log_file=out_dir / "stage.log")
    started_at = time.time()
    config = build_config(args)

    query_selection_dir = context.resolve_prior_output(
        "01_query_selection",
        prior_path=context.prior_outputs.get("01_query_selection"),
    )
    queries_lf = load_parquet_from_prior(query_selection_dir, "queries_")
    history_data.validate_queries_schema(queries_lf)
    query_count = queries_lf.select(pl.len()).collect(engine="streaming").item()
    if query_count == 0:
        raise ValueError("query_selection produced no queries")
    max_query_hour = queries_lf.select(pl.col("query_hour").max()).collect().item()

    like_sources_path = _find_like_sources_path(query_selection_dir)
    source_manifest = ingex.load_source_manifest(like_sources_path)
    source_like_paths = [entry["uri"] for entry in source_manifest["files"]]
    source_start = ingex.parse_utc_datetime(source_manifest.get("start"), field_name="likes_start")
    source_end = ingex.parse_utc_datetime(source_manifest.get("end"), field_name="likes_end")

    artifact_suffix = out_dir.name
    query_partitions_path = out_dir / f"_query_partitions_{artifact_suffix}"
    query_partitions_partial_path = out_dir / f"_query_partitions_{artifact_suffix}.partial"
    queried_user_likes_path = out_dir / f"_queried_user_likes_{artifact_suffix}"
    queried_user_likes_partial_path = out_dir / f"_queried_user_likes_{artifact_suffix}.partial"
    query_histories_path = out_dir / f"query_histories_{artifact_suffix}"
    query_histories_partial_path = out_dir / f"query_histories_{artifact_suffix}.partial"

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
    stats = _write_query_histories(
        query_partitions_path=query_partitions_path,
        queried_user_likes_path=queried_user_likes_path,
        partial_output_path=query_histories_partial_path,
        final_output_path=query_histories_path,
        config=config,
        logger=logger,
    )
    _log_stats(logger, stats)

    output_query_count = sum(values["query_count"] for values in stats.values())
    if output_query_count != query_count:
        raise ValueError(
            f"Query history count {output_query_count:,} does not match Stage 1 query count {query_count:,}"
        )

    runtime_seconds = time.time() - started_at
    summary = {
        "parameters": {
            "max_history_posts_per_query": config.max_history_posts_per_query,
            "user_history_partition_count": config.user_history_partition_count,
        },
        "input": {
            "query_selection_dir": str(query_selection_dir),
            "query_count": query_count,
            "like_sources_file": like_sources_path.name,
            "like_file_count": len(source_like_paths),
            "likes_start": source_manifest.get("start"),
            "likes_end": source_manifest.get("end"),
        },
        "outputs": {
            "query_histories_path": query_histories_path.name,
            "query_count": output_query_count,
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
            f"query_histories_path: {query_histories_path.name}",
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
        },
    }
