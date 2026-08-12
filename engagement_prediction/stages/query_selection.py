"""Stage 1: select bounded user-hour queries and their positive posts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

import polars as pl

from engagement_prediction.data import ingex, likes
from engagement_prediction.pipeline.core import Context
from utils.helpers import get_stage_logger


QUERY_KEY = ["did", "query_hour"]
POSITIVE_KEY = ["did", "query_hour", "subject_uri"]
RAW_POSITIVE_COUNT_COLUMN = "raw_positive_count"
SPLITS = (
    "train",
    "val",
    "val_unseen_users",
    "holdout_seen_users",
    "holdout_unseen_users",
)
HASH_BUCKET_COUNT = 1_000_000


@dataclass(frozen=True)
class QuerySelectionConfig:
    unseen_user_fraction: float
    max_hours_per_user_per_split: int
    max_train_query_hours: Optional[int]
    max_eval_query_hours_per_split: Optional[int]
    max_positives_per_user_hour: int
    random_seed: int
    likes_start: Optional[datetime]
    likes_end: Optional[datetime]
    train_start: datetime
    val_start: datetime
    holdout_start: Optional[datetime]
    holdout_end: Optional[datetime]


def _validate_hour_aligned(value: Optional[datetime], field_name: str) -> None:
    if value is None:
        return
    if value.minute != 0 or value.second != 0 or value.microsecond != 0:
        raise ValueError(f"{field_name} must be aligned to the start of an hour")


def _optional_nonnegative_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative when provided")
    return parsed


def build_config(args: argparse.Namespace) -> QuerySelectionConfig:
    likes_start = ingex.parse_utc_datetime(args.likes_start, field_name="likes_start")
    likes_end = ingex.parse_utc_datetime(args.likes_end, field_name="likes_end")
    train_start_raw = args.train_start if args.train_start is not None else args.likes_start
    train_start = ingex.parse_utc_datetime(train_start_raw, field_name="train_start")
    val_start = ingex.parse_utc_datetime(args.val_start, field_name="val_start")
    holdout_start = ingex.parse_utc_datetime(args.holdout_start, field_name="holdout_start")
    holdout_end = ingex.parse_utc_datetime(args.holdout_end, field_name="holdout_end")

    if train_start is None:
        raise ValueError("train_start is required when likes_start is not provided")
    if val_start is None:
        raise ValueError("val_start is required")
    if val_start <= train_start:
        raise ValueError("val_start must be after train_start")
    if holdout_start is not None and holdout_start <= val_start:
        raise ValueError("holdout_start must be after val_start")
    if holdout_start is None and holdout_end is not None:
        raise ValueError("holdout_start is required when holdout_end is provided")
    if holdout_end is not None and holdout_start is not None and holdout_end <= holdout_start:
        raise ValueError("holdout_end must be after holdout_start")
    if likes_start is not None and likes_end is not None and likes_end <= likes_start:
        raise ValueError("likes_end must be after likes_start")
    if likes_start is not None and train_start < likes_start:
        raise ValueError("train_start must not be before likes_start")
    if likes_end is not None and train_start >= likes_end:
        raise ValueError("train_start must be before likes_end")
    if holdout_end is not None and likes_end is not None and holdout_end > likes_end:
        raise ValueError("holdout_end must not be after likes_end")

    for field_name, value in (
        ("likes_start", likes_start),
        ("likes_end", likes_end),
        ("train_start", train_start),
        ("val_start", val_start),
        ("holdout_start", holdout_start),
        ("holdout_end", holdout_end),
    ):
        _validate_hour_aligned(value, field_name)

    unseen_user_fraction = float(args.unseen_user_fraction)
    if not 0.0 <= unseen_user_fraction < 1.0:
        raise ValueError("unseen_user_fraction must be in [0, 1)")
    max_hours_per_user_per_split = int(args.max_hours_per_user_per_split)
    if max_hours_per_user_per_split <= 0:
        raise ValueError("max_hours_per_user_per_split must be positive")
    max_positives_per_user_hour = int(args.max_positives_per_user_hour)
    if max_positives_per_user_hour <= 0:
        raise ValueError("max_positives_per_user_hour must be positive")

    return QuerySelectionConfig(
        unseen_user_fraction=unseen_user_fraction,
        max_hours_per_user_per_split=max_hours_per_user_per_split,
        max_train_query_hours=_optional_nonnegative_int(
            args.max_train_query_hours,
            "max_train_query_hours",
        ),
        max_eval_query_hours_per_split=_optional_nonnegative_int(
            args.max_eval_query_hours_per_split,
            "max_eval_query_hours_per_split",
        ),
        max_positives_per_user_hour=max_positives_per_user_hour,
        random_seed=int(args.random_seed),
        likes_start=likes_start,
        likes_end=likes_end,
        train_start=train_start,
        val_start=val_start,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )


def _with_user_cohort(
    lf: pl.LazyFrame,
    *,
    unseen_user_fraction: float,
    random_seed: int,
) -> pl.LazyFrame:
    unseen_bucket_count = int(unseen_user_fraction * HASH_BUCKET_COUNT)
    cohort_bucket = (
        pl.concat_str([pl.lit("user-cohort"), pl.col("did")], separator="|")
        .hash(seed=random_seed)
        % pl.lit(HASH_BUCKET_COUNT, dtype=pl.UInt64)
    )
    return lf.with_columns(
        pl.when(cohort_bucket < pl.lit(unseen_bucket_count, dtype=pl.UInt64))
        .then(pl.lit("unseen_eval"))
        .otherwise(pl.lit("trainval"))
        .alias("user_cohort")
    )


def _with_split(lf: pl.LazyFrame, config: QuerySelectionConfig) -> pl.LazyFrame:
    timestamp = pl.col("like_created_at")
    trainval = pl.col("user_cohort") == "trainval"
    unseen = pl.col("user_cohort") == "unseen_eval"
    before_end = timestamp < pl.lit(config.holdout_end) if config.holdout_end is not None else pl.lit(True)

    train_window = (timestamp >= pl.lit(config.train_start)) & (timestamp < pl.lit(config.val_start))
    if config.holdout_start is None:
        val_window = timestamp >= pl.lit(config.val_start)
        holdout_window = pl.lit(False)
    else:
        val_window = (timestamp >= pl.lit(config.val_start)) & (timestamp < pl.lit(config.holdout_start))
        holdout_window = timestamp >= pl.lit(config.holdout_start)

    return (
        lf.with_columns(
            pl.when(trainval & train_window)
            .then(pl.lit("train"))
            .when(trainval & val_window)
            .then(pl.lit("val"))
            .when(trainval & holdout_window)
            .then(pl.lit("holdout_seen_users"))
            .when(unseen & val_window)
            .then(pl.lit("val_unseen_users"))
            .when(unseen & holdout_window)
            .then(pl.lit("holdout_unseen_users"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("split")
        )
        .filter(before_end & pl.col("split").is_not_null())
    )


def _query_priority_expr(random_seed: int) -> pl.Expr:
    return pl.concat_str(
        [
            pl.lit("query-hour"),
            pl.col("did"),
            pl.col("split"),
            pl.col("query_hour").dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
        ],
        separator="|",
    ).hash(seed=random_seed)


def _cap_queries_per_user(queries_lf: pl.LazyFrame, config: QuerySelectionConfig) -> pl.LazyFrame:
    return (
        queries_lf
        .sort(["did", "split", "_query_priority", "query_hour"])
        .group_by(["did", "split"], maintain_order=True)
        .head(config.max_hours_per_user_per_split)
    )


def _cap_queries_per_split(queries_lf: pl.LazyFrame, config: QuerySelectionConfig) -> pl.LazyFrame:
    columns = [
        "did",
        "query_hour",
        "user_cohort",
        "split",
        RAW_POSITIVE_COUNT_COLUMN,
        "_query_priority",
    ]
    train_lf = queries_lf.filter(pl.col("split") == "train").sort(
        ["_query_priority", "did", "query_hour"]
    )
    if config.max_train_query_hours is not None:
        train_lf = train_lf.head(config.max_train_query_hours)

    eval_lf = queries_lf.filter(pl.col("split") != "train").sort(
        ["split", "_query_priority", "did", "query_hour"]
    )
    if config.max_eval_query_hours_per_split is not None:
        eval_lf = eval_lf.group_by("split", maintain_order=True).head(
            config.max_eval_query_hours_per_split
        )
    return pl.concat([train_lf.select(columns), eval_lf.select(columns)], how="vertical_relaxed")


def _prepare_likes(
    likes_lf: pl.LazyFrame,
    config: QuerySelectionConfig,
) -> pl.LazyFrame:
    filtered_lf = likes.prepare_likes(
        likes_lf,
        start=config.likes_start,
        end=config.likes_end,
    )

    return (
        _with_split(
            _with_user_cohort(
                filtered_lf.filter(
                    pl.col("did").is_not_null()
                    & pl.col("subject_uri").is_not_null()
                    & pl.col("like_created_at").is_not_null()
                ),
                unseen_user_fraction=config.unseen_user_fraction,
                random_seed=config.random_seed,
            ),
            config,
        )
        .with_columns(pl.col("like_created_at").dt.truncate("1h").alias("query_hour"))
        .select(["did", "query_hour", "user_cohort", "split", "subject_uri", "like_created_at"])
    )


def _candidate_query_counts_lf(positive_rows_lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        positive_rows_lf
        .group_by(["did", "query_hour", "user_cohort", "split"])
        .agg(pl.len().cast(pl.UInt32).alias(RAW_POSITIVE_COUNT_COLUMN))
    )


def _build_query_lazyframes_from_counts(
    *,
    positive_rows_lf: pl.LazyFrame,
    candidate_query_counts_lf: pl.LazyFrame,
    config: QuerySelectionConfig,
) -> Dict[str, pl.LazyFrame]:
    candidate_queries_lf = candidate_query_counts_lf.with_columns(
        _query_priority_expr(config.random_seed).alias("_query_priority")
    )
    after_user_cap_lf = _cap_queries_per_user(candidate_queries_lf, config)
    after_split_cap_lf = _cap_queries_per_split(after_user_cap_lf, config)

    # Apply the cap to raw rows after query sampling. Deduplication does not make
    # an otherwise oversized query eligible, and discarded queries are not backfilled.
    selected_query_counts_lf = after_split_cap_lf.filter(
        pl.col(RAW_POSITIVE_COUNT_COLUMN) <= config.max_positives_per_user_hour
    )

    # Only selected user-hours are deduplicated. The same user/post pair may be
    # retained in different query-hours.
    positives_lf = (
        positive_rows_lf
        .join(selected_query_counts_lf.select(QUERY_KEY), on=QUERY_KEY, how="inner")
        .group_by(POSITIVE_KEY)
        .agg(pl.col("like_created_at").min())
        .select(["did", "query_hour", "subject_uri", "like_created_at"])
        .sort(["query_hour", "did", "subject_uri"])
    )
    final_counts_lf = positives_lf.group_by(QUERY_KEY).agg(
        pl.len().cast(pl.UInt32).alias("positive_count")
    )
    queries_lf = (
        selected_query_counts_lf
        .select(["did", "query_hour", "user_cohort", "split"])
        .join(final_counts_lf, on=QUERY_KEY, how="inner")
        .select(["did", "query_hour", "user_cohort", "split", "positive_count"])
        .sort(["query_hour", "did"])
    )
    return {
        "candidate_queries": candidate_queries_lf,
        "after_user_cap": after_user_cap_lf,
        "after_split_cap": after_split_cap_lf,
        "queries": queries_lf,
        "positives": positives_lf,
    }


def _split_summary_lf(lf: pl.LazyFrame, phase: str) -> pl.LazyFrame:
    return (
        lf.group_by("split")
        .agg(
            pl.len().cast(pl.UInt64).alias("query_count"),
            pl.col("did").n_unique().cast(pl.UInt64).alias("unique_user_count"),
        )
        .with_columns(pl.lit(phase).alias("phase"))
        .select(["phase", "split", "query_count", "unique_user_count"])
    )


def _positive_count_distribution_lf(
    lf: pl.LazyFrame,
    phase: str,
    count_column: str,
) -> pl.LazyFrame:
    return (
        lf.group_by(["split", count_column])
        .agg(pl.len().cast(pl.UInt64).alias("query_count"))
        .with_columns(
            pl.lit(phase).alias("phase"),
            pl.col(count_column).alias("positive_count"),
        )
        .select(["phase", "split", "positive_count", "query_count"])
    )


def collect_query_artifacts(
    lazyframes: Dict[str, pl.LazyFrame],
    config: QuerySelectionConfig,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:
    summaries_lf = pl.concat(
        [
            _split_summary_lf(lazyframes["candidate_queries"], "candidate"),
            _split_summary_lf(lazyframes["after_user_cap"], "after_user_cap"),
            _split_summary_lf(lazyframes["after_split_cap"], "after_split_cap"),
            _split_summary_lf(lazyframes["queries"], "final"),
        ],
        how="vertical_relaxed",
    )
    distributions_lf = pl.concat(
        [
            _positive_count_distribution_lf(
                lazyframes["candidate_queries"],
                "candidate",
                RAW_POSITIVE_COUNT_COLUMN,
            ),
            _positive_count_distribution_lf(lazyframes["queries"], "final", "positive_count"),
        ],
        how="vertical_relaxed",
    )
    oversized_lf = (
        lazyframes["after_split_cap"]
        .filter(pl.col(RAW_POSITIVE_COUNT_COLUMN) > config.max_positives_per_user_hour)
        .group_by("split")
        .agg(pl.len().cast(pl.UInt64).alias("query_count"))
    )

    summaries_df, distributions_df, oversized_df, queries_df, positives_df = pl.collect_all(
        [
            summaries_lf,
            distributions_lf,
            oversized_lf,
            lazyframes["queries"],
            lazyframes["positives"],
        ],
        engine="streaming",
    )
    if queries_df.is_empty():
        raise ValueError("Query selection produced no queries")

    _validate_artifacts(queries_df, positives_df)
    return queries_df, positives_df, _build_stats(summaries_df, distributions_df, oversized_df)


def _validate_artifacts(queries_df: pl.DataFrame, positives_df: pl.DataFrame) -> None:
    expected_query_columns = ["did", "query_hour", "user_cohort", "split", "positive_count"]
    expected_positive_columns = ["did", "query_hour", "subject_uri", "like_created_at"]
    if queries_df.columns != expected_query_columns:
        raise ValueError(f"Unexpected queries columns: {queries_df.columns}")
    if positives_df.columns != expected_positive_columns:
        raise ValueError(f"Unexpected query positives columns: {positives_df.columns}")
    if queries_df.unique(subset=QUERY_KEY).height != queries_df.height:
        raise ValueError("queries contains duplicate (did, query_hour) keys")
    if positives_df.unique(subset=POSITIVE_KEY).height != positives_df.height:
        raise ValueError("query_positives contains duplicate (did, query_hour, subject_uri) keys")
    orphan_count = positives_df.join(queries_df.select(QUERY_KEY), on=QUERY_KEY, how="anti").height
    if orphan_count:
        raise ValueError(f"query_positives contains {orphan_count:,} rows without a query")
    counts_df = positives_df.group_by(QUERY_KEY).agg(
        pl.col("subject_uri").n_unique().cast(pl.UInt32).alias("actual_positive_count")
    )
    mismatches = (
        queries_df
        .join(counts_df, on=QUERY_KEY, how="left")
        .filter(pl.col("positive_count") != pl.col("actual_positive_count"))
        .height
    )
    if mismatches:
        raise ValueError(f"Found {mismatches:,} queries with an incorrect positive_count")
    for column in ("query_hour",):
        dtype = queries_df.schema[column]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
            raise ValueError(f"{column} must be a UTC datetime, found {dtype}")
    like_dtype = positives_df.schema["like_created_at"]
    if not isinstance(like_dtype, pl.Datetime) or like_dtype.time_zone != "UTC":
        raise ValueError(f"like_created_at must be a UTC datetime, found {like_dtype}")


def _build_stats(
    summaries_df: pl.DataFrame,
    distributions_df: pl.DataFrame,
    oversized_df: pl.DataFrame,
) -> Dict[str, Any]:
    by_phase: Dict[str, Dict[str, Dict[str, int]]] = {}
    for phase in ("candidate", "after_user_cap", "after_split_cap", "final"):
        by_phase[phase] = {
            split: {"query_count": 0, "unique_user_count": 0}
            for split in SPLITS
        }
    for row in summaries_df.iter_rows(named=True):
        by_phase[row["phase"]][row["split"]] = {
            "query_count": int(row["query_count"]),
            "unique_user_count": int(row["unique_user_count"]),
        }

    positive_count_distribution: Dict[str, Dict[str, Dict[str, int]]] = {
        "candidate": {split: {} for split in SPLITS},
        "final": {split: {} for split in SPLITS},
    }
    for row in distributions_df.iter_rows(named=True):
        positive_count_distribution[row["phase"]][row["split"]][str(row["positive_count"])] = int(
            row["query_count"]
        )

    oversized_by_split = {split: 0 for split in SPLITS}
    for row in oversized_df.iter_rows(named=True):
        oversized_by_split[row["split"]] = int(row["query_count"])

    return {
        "queries_by_phase_and_split": by_phase,
        "positive_count_distribution": positive_count_distribution,
        "oversized_query_count_by_split": oversized_by_split,
    }


def _log_stats(logger: logging.Logger, stats: Dict[str, Any]) -> None:
    for split in SPLITS:
        candidate = stats["queries_by_phase_and_split"]["candidate"][split]
        final = stats["queries_by_phase_and_split"]["final"][split]
        oversized = stats["oversized_query_count_by_split"][split]
        logger.info(
            "%s: candidate_queries=%s candidate_users=%s final_queries=%s final_users=%s oversized_dropped=%s",
            split,
            f"{candidate['query_count']:,}",
            f"{candidate['unique_user_count']:,}",
            f"{final['query_count']:,}",
            f"{final['unique_user_count']:,}",
            f"{oversized:,}",
        )


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = context.new_stage_dir("01_query_selection")
    logger = get_stage_logger("01_QUERY_SELECTION", log_file=out_dir / "stage.log")
    started_at = time.time()
    config = build_config(args)

    like_paths, like_file_timestamps = ingex.list_ingex_parquet_files(
        gcs_bucket=str(args.gcs_bucket),
        blob_prefix="bsky_likes",
        start=config.likes_start,
        end=config.likes_end,
    )
    if not like_paths:
        raise ValueError(
            f"No likes Parquet files found for {config.likes_start} to {config.likes_end}"
        )
    logger.info("Found %s likes Parquet files", f"{len(like_paths):,}")

    artifact_suffix = out_dir.name
    like_sources_path = out_dir / f"like_sources_{artifact_suffix}.json"
    ingex.write_source_manifest(
        like_sources_path,
        ingex.build_source_manifest(
            gcs_bucket=str(args.gcs_bucket),
            blob_prefix="bsky_likes",
            start=config.likes_start,
            end=config.likes_end,
            paths=like_paths,
            timestamps=like_file_timestamps,
        ),
    )
    candidate_query_counts_path = out_dir / f"_candidate_query_counts_{artifact_suffix}.parquet"
    partial_candidate_query_counts_path = (
        out_dir / f"_candidate_query_counts_{artifact_suffix}.partial.parquet"
    )

    positive_rows_lf = _prepare_likes(
        ingex.scan_parquet_files(like_paths),
        config,
    )
    logger.info("Aggregating raw likes into candidate user-hour counts")
    _candidate_query_counts_lf(positive_rows_lf).sink_parquet(
        partial_candidate_query_counts_path,
        compression="zstd",
        maintain_order=False,
        engine="streaming",
    )
    partial_candidate_query_counts_path.replace(candidate_query_counts_path)
    logger.info("Saved candidate user-hour counts to %s", candidate_query_counts_path)

    # Start a new plan from the compact user-hour table, then rescan likes only
    # to populate positives for the selected query keys.
    selected_positive_rows_lf = _prepare_likes(
        ingex.scan_parquet_files(like_paths),
        config,
    )
    lazyframes = _build_query_lazyframes_from_counts(
        positive_rows_lf=selected_positive_rows_lf,
        candidate_query_counts_lf=pl.scan_parquet(candidate_query_counts_path),
        config=config,
    )

    # the collection and validation of the query artifacts happens in here
    queries_df, positives_df, stats = collect_query_artifacts(lazyframes, config)
    _log_stats(logger, stats)

    queries_path = out_dir / f"queries_{artifact_suffix}.parquet"
    query_positives_path = out_dir / f"query_positives_{artifact_suffix}.parquet"
    queries_df.write_parquet(queries_path, compression="zstd")
    positives_df.write_parquet(query_positives_path, compression="zstd")

    runtime_seconds = time.time() - started_at
    summary = {
        "gcs_bucket": str(args.gcs_bucket),
        "likes_start": config.likes_start.isoformat() if config.likes_start is not None else None,
        "likes_end": config.likes_end.isoformat() if config.likes_end is not None else None,
        "parameters": {
            "train_start": config.train_start.isoformat(),
            "val_start": config.val_start.isoformat(),
            "holdout_start": config.holdout_start.isoformat() if config.holdout_start is not None else None,
            "holdout_end": config.holdout_end.isoformat() if config.holdout_end is not None else None,
            "unseen_user_fraction": config.unseen_user_fraction,
            "max_hours_per_user_per_split": config.max_hours_per_user_per_split,
            "max_train_query_hours": config.max_train_query_hours,
            "max_eval_query_hours_per_split": config.max_eval_query_hours_per_split,
            "max_positives_per_user_hour": config.max_positives_per_user_hour,
            "random_seed": config.random_seed,
        },
        "input": {
            "like_file_count": len(like_paths),
            "first_like_file_timestamp": like_file_timestamps[0].isoformat(),
            "last_like_file_timestamp": like_file_timestamps[-1].isoformat(),
        },
        "outputs": {
            "like_sources_file": like_sources_path.name,
            "candidate_query_counts_file": candidate_query_counts_path.name,
            "queries_file": queries_path.name,
            "query_positives_file": query_positives_path.name,
            "query_count": queries_df.height,
            "positive_count": positives_df.height,
        },
        "selection_stats": stats,
        "runtime_seconds": runtime_seconds,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "stage_info.txt").write_text(
        "\n".join(
            [
                "stage: query_selection",
                f"runtime_seconds: {runtime_seconds:.2f}",
                f"input_like_files: {len(like_paths)}",
                f"queries: {queries_df.height}",
                f"query_positives: {positives_df.height}",
                f"like_sources_file: {like_sources_path.name}",
                f"candidate_query_counts_file: {candidate_query_counts_path.name}",
                f"queries_file: {queries_path.name}",
                f"query_positives_file: {query_positives_path.name}",
            ]
        )
        + "\n"
    )
    logger.info(
        "Query selection completed in %.2fs with %s queries and %s positives",
        runtime_seconds,
        f"{queries_df.height:,}",
        f"{positives_df.height:,}",
    )

    return {
        "output_dir": out_dir,
        "artifacts": {
            "like_sources_path": str(like_sources_path),
            "queries_path": str(queries_path),
            "query_positives_path": str(query_positives_path),
        },
    }
