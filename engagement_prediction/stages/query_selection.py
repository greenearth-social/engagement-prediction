"""Stage 1: select bounded user-hour queries and their positive posts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import shutil
import time
from typing import Any, Dict, Optional, Tuple

import polars as pl

from engagement_prediction.data import (
    ingex,
    likes,
    query_selection_artifacts,
    source_metadata_artifacts,
)
from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline.lineage import resolve_recorded_stage_lineage
from engagement_prediction.pipeline.logging import get_stage_logger


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
    """Validated Stage 1 settings shared by selection and artifact construction."""

    unseen_user_fraction: float
    max_hours_per_user_per_split: int
    max_train_query_hours: Optional[int]
    max_eval_query_hours_per_split: Optional[int]
    max_positives_per_user_hour: int
    random_seed: int
    posts_start: datetime
    posts_end: datetime
    train_start: datetime
    val_start: datetime
    holdout_start: Optional[datetime]
    holdout_end: Optional[datetime]


def _validate_hour_aligned(value: Optional[datetime], field_name: str) -> None:
    """Reject boundaries that cannot be represented as query-hour buckets."""

    if value is None:
        return
    if value.minute != 0 or value.second != 0 or value.microsecond != 0:
        raise ValueError(f"{field_name} must be aligned to the start of an hour")


def _optional_nonnegative_int(value: Any, field_name: str) -> Optional[int]:
    """Parse an optional query budget while preserving ``None`` as unbounded."""

    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative when provided")
    return parsed


def build_config(args: argparse.Namespace) -> QuerySelectionConfig:
    """Parse CLI values and validate the common source and target windows."""

    posts_start = ingex.parse_utc_datetime(args.posts_start, field_name="posts_start")
    posts_end = ingex.parse_utc_datetime(args.posts_end, field_name="posts_end")
    train_start = ingex.parse_utc_datetime(args.train_start, field_name="train_start")
    val_start = ingex.parse_utc_datetime(args.val_start, field_name="val_start")
    holdout_start = ingex.parse_utc_datetime(args.holdout_start, field_name="holdout_start")
    holdout_end = ingex.parse_utc_datetime(args.holdout_end, field_name="holdout_end")

    if posts_start is None or posts_end is None:
        raise ValueError("posts_start and posts_end are required for query_selection")
    for field_name, value in (
        ("posts_start", posts_start),
        ("posts_end", posts_end),
        ("train_start", train_start),
        ("val_start", val_start),
        ("holdout_start", holdout_start),
        ("holdout_end", holdout_end),
    ):
        _validate_hour_aligned(value, field_name)

    if posts_end <= posts_start:
        raise ValueError("posts_end must be after posts_start")
    if train_start is None:
        raise ValueError("train_start is required")
    if train_start < posts_start:
        raise ValueError("train_start must not be before posts_start")
    if train_start >= posts_end:
        raise ValueError("train_start must be before posts_end")
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
    if val_start >= posts_end:
        raise ValueError("val_start must be before posts_end")
    if holdout_start is not None and holdout_start >= posts_end:
        raise ValueError("holdout_start must be before posts_end")
    if holdout_end is not None and holdout_end > posts_end:
        raise ValueError("holdout_end must not be after posts_end")

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
        posts_start=posts_start,
        posts_end=posts_end,
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
    """Assign stable seen/unseen membership using only DID and the seed.

    User activity and the set of available query hours therefore cannot affect
    whether a user is held out for unseen-user evaluation.
    """

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
    """Map target likes to one of the five cohort-aware temporal splits."""

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


_PER_USER_CAP_HASH_NAMESPACE = "query-hour-per-user-cap"
_SPLIT_CAP_HASH_NAMESPACE = "query-hour-split-cap"
_PER_USER_PRIORITY_COLUMN = "_per_user_query_priority"
_SPLIT_PRIORITY_COLUMN = "_split_query_priority"


def _query_priority_expr(random_seed: int, *, namespace: str) -> pl.Expr:
    """Return a stable namespaced pseudo-random rank for one query hour."""

    return pl.concat_str(
        [
            pl.lit(namespace),
            pl.col("did"),
            pl.col("split"),
            pl.col("query_hour").dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
        ],
        separator="|",
    ).hash(seed=random_seed)


def _cap_queries_per_user(queries_lf: pl.LazyFrame, config: QuerySelectionConfig) -> pl.LazyFrame:
    """Limit one user's contribution independently within every split."""

    return (
        queries_lf
        .sort(["did", "split", _PER_USER_PRIORITY_COLUMN, "query_hour"])
        .group_by(["did", "split"], maintain_order=True)
        .head(config.max_hours_per_user_per_split)
    )


def _cap_queries_per_split(queries_lf: pl.LazyFrame, config: QuerySelectionConfig) -> pl.LazyFrame:
    """Apply the optional train and per-evaluation query-hour budgets."""

    columns = [
        "did",
        "query_hour",
        "user_cohort",
        "split",
        RAW_POSITIVE_COUNT_COLUMN,
    ]
    train_lf = queries_lf.filter(pl.col("split") == "train").sort(
        [_SPLIT_PRIORITY_COLUMN, "did", "query_hour"]
    )
    if config.max_train_query_hours is not None:
        train_lf = train_lf.head(config.max_train_query_hours)

    eval_lf = queries_lf.filter(pl.col("split") != "train").sort(
        ["split", _SPLIT_PRIORITY_COLUMN, "did", "query_hour"]
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
    """Normalize target likes and attach cohort, split, and query-hour fields."""

    filtered_lf = likes.prepare_likes(
        likes_lf,
        start=config.posts_start,
        end=config.posts_end,
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
    """Collapse raw target-like rows to narrow provisional user-hour counts."""

    return (
        positive_rows_lf
        .group_by(["did", "query_hour", "user_cohort", "split"])
        .agg(pl.len().cast(pl.UInt32).alias(RAW_POSITIVE_COUNT_COLUMN))
    )


def _build_provisional_query_lazyframes(
    candidate_query_counts_lf: pl.LazyFrame,
    config: QuerySelectionConfig,
) -> Dict[str, pl.LazyFrame]:
    """Build candidate, per-user-capped, and split-capped query plans."""

    # These ranks must be independent. Reusing the per-user rank globally
    # favors highly active users because their retained rows are low order
    # statistics from a larger set of hashes.
    candidate_queries_lf = candidate_query_counts_lf.with_columns(
        _query_priority_expr(
            config.random_seed,
            namespace=_PER_USER_CAP_HASH_NAMESPACE,
        ).alias(_PER_USER_PRIORITY_COLUMN),
        _query_priority_expr(
            config.random_seed,
            namespace=_SPLIT_CAP_HASH_NAMESPACE,
        ).alias(_SPLIT_PRIORITY_COLUMN),
    )
    after_user_cap_lf = _cap_queries_per_user(candidate_queries_lf, config)
    after_split_cap_lf = _cap_queries_per_split(after_user_cap_lf, config)
    return {
        "candidate_queries": candidate_queries_lf,
        "after_user_cap": after_user_cap_lf,
        "after_split_cap": after_split_cap_lf,
    }


def _build_final_query_lazyframes(
    *,
    provisional_query_lazyframes: Dict[str, pl.LazyFrame],
    eligible_positive_rows_lf: pl.LazyFrame,
    config: QuerySelectionConfig,
) -> Dict[str, pl.LazyFrame]:
    """Finalize queries after root-post membership has filtered positives.

    Sampling happens before root-post filtering. A query that becomes empty or
    oversized here is therefore dropped without replacement.
    """

    # Replace the sampled query's raw row count with the number of deduplicated
    # positives that actually resolved in the exact post snapshot.
    after_split_cap_lf = provisional_query_lazyframes["after_split_cap"]
    eligible_counts_lf = eligible_positive_rows_lf.group_by(QUERY_KEY).agg(
        pl.len().cast(pl.UInt32).alias("positive_count")
    )
    final_counts_lf = eligible_counts_lf.filter(
        pl.col("positive_count") <= config.max_positives_per_user_hour
    )
    # The inner join removes both zero-positive and oversized query keys while
    # preserving cohort and split metadata assigned before sampling.
    queries_lf = (
        after_split_cap_lf
        .select(["did", "query_hour", "user_cohort", "split"])
        .join(final_counts_lf, on=QUERY_KEY, how="inner")
        .select(["did", "query_hour", "user_cohort", "split", "positive_count"])
        .sort(["query_hour", "did"])
    )
    # Filter positives through the same key set to establish referential
    # integrity between the two public Stage 1 artifacts.
    positives_lf = (
        eligible_positive_rows_lf
        .join(final_counts_lf.select(QUERY_KEY), on=QUERY_KEY, how="inner")
        .select(["did", "query_hour", "subject_uri", "like_created_at"])
        .sort(["query_hour", "did", "subject_uri"])
    )
    return {
        **provisional_query_lazyframes,
        "eligible_counts": eligible_counts_lf,
        "queries": queries_lf,
        "positives": positives_lf,
    }


def _split_summary_lf(lf: pl.LazyFrame, phase: str) -> pl.LazyFrame:
    """Summarize query and unique-user counts for one selection phase."""

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
    """Build an exact positive-count histogram by split for one phase."""

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
    positive_filter_stats_by_split: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:
    """Collect final public rows and diagnostic selection summaries."""

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
        lazyframes["eligible_counts"]
        .filter(pl.col("positive_count") > config.max_positives_per_user_hour)
        .join(
            lazyframes["after_split_cap"].select([*QUERY_KEY, "split"]),
            on=QUERY_KEY,
            how="inner",
        )
        .group_by("split")
        .agg(pl.len().cast(pl.UInt64).alias("query_count"))
    )
    zero_positive_lf = (
        lazyframes["after_split_cap"]
        .join(
            lazyframes["eligible_counts"].select(QUERY_KEY),
            on=QUERY_KEY,
            how="anti",
        )
        .group_by("split")
        .agg(pl.len().cast(pl.UInt64).alias("query_count"))
    )

    (
        summaries_df,
        distributions_df,
        oversized_df,
        zero_positive_df,
        queries_df,
        positives_df,
    ) = pl.collect_all(
        [
            summaries_lf,
            distributions_lf,
            oversized_lf,
            zero_positive_lf,
            lazyframes["queries"],
            lazyframes["positives"],
        ],
        engine="streaming",
    )
    if queries_df.is_empty():
        raise ValueError("Query selection produced no queries")

    _validate_artifacts(queries_df, positives_df)
    return queries_df, positives_df, _build_stats(
        summaries_df,
        distributions_df,
        oversized_df,
        zero_positive_df,
        positive_filter_stats_by_split,
    )


def _validate_artifacts(queries_df: pl.DataFrame, positives_df: pl.DataFrame) -> None:
    """Enforce schemas, unique keys, counts, and positive-to-query linkage."""

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
    zero_positive_df: pl.DataFrame,
    positive_filter_stats_by_split: Optional[Dict[str, Dict[str, int]]],
) -> Dict[str, Any]:
    """Convert diagnostic frames into JSON-serializable stage statistics."""

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

    zero_positive_by_split = {split: 0 for split in SPLITS}
    for row in zero_positive_df.iter_rows(named=True):
        zero_positive_by_split[row["split"]] = int(row["query_count"])

    positive_filter_by_split = {
        split: {
            "selected_like_row_count": 0,
            "provisional_positive_count": 0,
            "retained_positive_count": 0,
            "missing_post_positive_count": 0,
        }
        for split in SPLITS
    }
    if positive_filter_stats_by_split is not None:
        for split, values in positive_filter_stats_by_split.items():
            positive_filter_by_split[split] = {
                key: int(value)
                for key, value in values.items()
            }
    positive_filter_totals = {
        field_name: sum(values[field_name] for values in positive_filter_by_split.values())
        for field_name in (
            "selected_like_row_count",
            "provisional_positive_count",
            "retained_positive_count",
            "missing_post_positive_count",
        )
    }

    return {
        "queries_by_phase_and_split": by_phase,
        "positive_count_distribution": positive_count_distribution,
        "oversized_query_count_by_split": oversized_by_split,
        "zero_positive_query_count_by_split": zero_positive_by_split,
        "positive_filter_by_split": positive_filter_by_split,
        "positive_filter_totals": positive_filter_totals,
    }


def _log_stats(logger: logging.Logger, stats: Dict[str, Any]) -> None:
    """Log per-split attrition from candidate hours to final queries."""

    for split in SPLITS:
        candidate = stats["queries_by_phase_and_split"]["candidate"][split]
        final = stats["queries_by_phase_and_split"]["final"][split]
        oversized = stats["oversized_query_count_by_split"][split]
        zero_positive = stats["zero_positive_query_count_by_split"][split]
        positive_filter = stats["positive_filter_by_split"][split]
        logger.info(
            "%s: candidate_queries=%s candidate_users=%s final_queries=%s "
            "final_users=%s provisional_positives=%s retained_positives=%s "
            "missing_post_positives=%s zero_positive_dropped=%s oversized_dropped=%s",
            split,
            f"{candidate['query_count']:,}",
            f"{candidate['unique_user_count']:,}",
            f"{final['query_count']:,}",
            f"{final['unique_user_count']:,}",
            f"{positive_filter['provisional_positive_count']:,}",
            f"{positive_filter['retained_positive_count']:,}",
            f"{positive_filter['missing_post_positive_count']:,}",
            f"{zero_positive:,}",
            f"{oversized:,}",
        )


def _validate_post_window(
    sampled_queries_lf: pl.LazyFrame,
    config: QuerySelectionConfig,
) -> tuple[datetime, datetime]:
    """Ensure the post snapshot covers every provisionally sampled hour."""

    bounds = sampled_queries_lf.select(
        pl.col("query_hour").min().alias("min_query_hour"),
        pl.col("query_hour").max().alias("max_query_hour"),
    ).collect(engine="streaming")
    min_query_hour = bounds.item(0, "min_query_hour")
    max_query_hour = bounds.item(0, "max_query_hour")
    if min_query_hour is None or max_query_hour is None:
        raise ValueError("Query selection produced no provisionally sampled query-hours")
    if config.posts_start > min_query_hour or config.posts_end <= max_query_hour:
        raise ValueError(
            "posts_start/posts_end must cover every provisionally selected query hour: "
            f"query range is {min_query_hour.isoformat()} to {max_query_hour.isoformat()}"
        )
    return min_query_hour, max_query_hour


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    """Run Stage 1 and publish query, positive, and source-snapshot artifacts."""

    out_dir = context.new_stage_dir("01_query_selection")
    logger = get_stage_logger("01_QUERY_SELECTION", log_file=out_dir / "stage.log")
    started_at = time.time()
    config = build_config(args)
    lineage = resolve_recorded_stage_lineage(
        context,
        terminal_stage_folder="00_source_metadata",
        ancestor_stage_folders=(),
    )
    source_metadata_dir = lineage["00_source_metadata"]
    source_artifact = source_metadata_artifacts.load_source_metadata_artifact(
        source_metadata_dir
    )
    if (
        source_artifact.post_snapshot.gcs_bucket != str(args.gcs_bucket)
        or source_artifact.post_snapshot.start != config.posts_start
        or source_artifact.post_snapshot.end != config.posts_end
    ):
        raise ValueError(
            "Stage 1 configuration does not match the pinned Stage 00 source bucket/window"
        )
    partition_count = source_artifact.partition_count
    query_positive_end = (
        config.holdout_end if config.holdout_end is not None else config.posts_end
    )
    logger.info(
        "Starting query selection: source_manifest_window=[%s, %s) "
        "query_positive_window=[%s, %s) val_start=%s holdout_start=%s "
        "source_metadata_partitions=%s",
        config.posts_start.isoformat(),
        config.posts_end.isoformat(),
        config.train_start.isoformat(),
        query_positive_end.isoformat(),
        config.val_start.isoformat(),
        config.holdout_start.isoformat() if config.holdout_start is not None else None,
        partition_count,
    )

    logger.info("Phase 1/5: validating Stage 00 metadata and listing exact likes")
    like_paths, like_file_timestamps = ingex.list_ingex_parquet_files(
        gcs_bucket=str(args.gcs_bucket),
        blob_prefix="bsky_likes",
        start=config.posts_start,
        end=config.posts_end,
    )
    if not like_paths:
        raise ValueError(
            f"No likes Parquet files found for {config.posts_start} to {config.posts_end}"
        )
    logger.info("Found %s likes Parquet files", f"{len(like_paths):,}")
    artifact_suffix = out_dir.name
    like_sources_path = out_dir / f"like_sources_{artifact_suffix}.json"
    ingex.write_source_manifest(
        like_sources_path,
        ingex.build_source_manifest(
            gcs_bucket=str(args.gcs_bucket),
            blob_prefix="bsky_likes",
            start=config.posts_start,
            end=config.posts_end,
            paths=like_paths,
            timestamps=like_file_timestamps,
        ),
    )
    logger.info("Saved exact likes source-file manifest")

    # Materializing the narrow counts prevents later ranking and capping plans
    # from repeatedly executing the all-like user-hour aggregation.
    candidate_query_counts_path = out_dir / f"_candidate_query_counts_{artifact_suffix}.parquet"
    partial_candidate_query_counts_path = (
        out_dir / f"_candidate_query_counts_{artifact_suffix}.partial.parquet"
    )

    logger.info("Phase 2/5: aggregating raw likes and sampling provisional query-hours")
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

    provisional_query_lazyframes = _build_provisional_query_lazyframes(
        pl.scan_parquet(candidate_query_counts_path),
        config,
    )
    sampled_queries_path = out_dir / f"_sampled_query_hours_{artifact_suffix}.partial.parquet"
    provisional_query_lazyframes["after_split_cap"].sink_parquet(
        sampled_queries_path,
        compression="zstd",
        maintain_order=False,
        engine="streaming",
    )
    sampled_queries_lf = pl.scan_parquet(sampled_queries_path)
    min_query_hour, max_query_hour = _validate_post_window(sampled_queries_lf, config)
    logger.info(
        "Saved provisional query-hours with range [%s, %s]",
        min_query_hour.isoformat(),
        max_query_hour.isoformat(),
    )

    logger.info("Phase 3/5: partitioning selected positive-like rows by metadata URI")
    # Matching URI partitions let Phase 4 use bounded local semi-joins instead
    # of one global posts-to-positives join.
    provisional_positive_rows_path = (
        out_dir / f"_provisional_positive_rows_{artifact_suffix}.partial"
    )
    eligible_positive_rows_path = (
        out_dir / f"_eligible_positive_rows_{artifact_suffix}.partial"
    )
    query_selection_artifacts.materialize_provisional_positive_rows(
        positive_rows_lf=_prepare_likes(
            ingex.scan_parquet_files(like_paths),
            config,
        ),
        sampled_queries_lf=sampled_queries_lf,
        partition_count=partition_count,
        output_path=provisional_positive_rows_path,
        logger=logger,
    )

    logger.info("Phase 4/5: filtering positives to canonical Stage 00 root URIs")
    membership_stats = query_selection_artifacts.filter_positive_partitions(
        provisional_positive_rows_path=provisional_positive_rows_path,
        post_metadata_path=source_artifact.post_metadata_path,
        eligible_positive_rows_path=eligible_positive_rows_path,
        partition_count=partition_count,
        splits=SPLITS,
        logger=logger,
    )

    materialized_query_lazyframes = {
        **provisional_query_lazyframes,
        "after_split_cap": sampled_queries_lf,
    }
    lazyframes = _build_final_query_lazyframes(
        provisional_query_lazyframes=materialized_query_lazyframes,
        eligible_positive_rows_lf=query_selection_artifacts.scan_eligible_positive_rows(
            eligible_positive_rows_path
        ),
        config=config,
    )

    logger.info("Phase 5/5: building, validating, and publishing final query artifacts")
    queries_df, positives_df, stats = collect_query_artifacts(
        lazyframes,
        config,
        membership_stats["positive_filter_stats_by_split"],
    )
    _log_stats(logger, stats)

    queries_path = out_dir / f"queries_{artifact_suffix}.parquet"
    query_positives_path = out_dir / f"query_positives_{artifact_suffix}.parquet"
    partial_queries_path = out_dir / f"queries_{artifact_suffix}.partial.parquet"
    partial_query_positives_path = (
        out_dir / f"query_positives_{artifact_suffix}.partial.parquet"
    )
    # Public files use temporary names until both have been written. The
    # pipeline manifest is created only after the runner returns successfully.
    queries_df.write_parquet(partial_queries_path, compression="zstd")
    positives_df.write_parquet(partial_query_positives_path, compression="zstd")
    partial_queries_path.replace(queries_path)
    partial_query_positives_path.replace(query_positives_path)

    sampled_queries_path.unlink()
    shutil.rmtree(provisional_positive_rows_path)
    shutil.rmtree(eligible_positive_rows_path)
    logger.info("Removed successful post-membership staging data")

    runtime_seconds = time.time() - started_at
    summary = {
        "gcs_bucket": str(args.gcs_bucket),
        "posts_start": config.posts_start.isoformat(),
        "posts_end": config.posts_end.isoformat(),
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
            "source_metadata_partition_count": partition_count,
        },
        "input": {
            "source_metadata_dir": str(source_metadata_dir),
            "like_file_count": len(like_paths),
            "first_like_file_timestamp": like_file_timestamps[0].isoformat(),
            "last_like_file_timestamp": like_file_timestamps[-1].isoformat(),
            "post_file_count": len(source_artifact.post_snapshot.file_uris),
            "post_source_stats": source_artifact.summary["index"]["root_source_stats"],
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
                f"source_metadata_dir: {source_metadata_dir}",
                f"input_post_files: {len(source_artifact.post_snapshot.file_uris)}",
                f"queries: {queries_df.height}",
                f"query_positives: {positives_df.height}",
                "provisional_query_positives: "
                f"{stats['positive_filter_totals']['provisional_positive_count']}",
                "post_snapshot_query_positives: "
                f"{stats['positive_filter_totals']['retained_positive_count']}",
                "positives_absent_from_post_snapshot: "
                f"{stats['positive_filter_totals']['missing_post_positive_count']}",
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
            "source_metadata_path": str(source_artifact.bundle_path),
            "queries_path": str(queries_path),
            "query_positives_path": str(query_positives_path),
        },
    }
