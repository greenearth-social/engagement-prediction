"""Deterministic hourly negative selection for Stage 4."""

from __future__ import annotations

import math

import polars as pl

from engagement_prediction.data import candidate_popularity


UTC_DATETIME = candidate_popularity.UTC_DATETIME
HOURLY_CANDIDATE_COLUMNS = [
    "query_hour",
    "subject_uri",
    "selection_source",
    "prior_like_count",
]
HOURLY_CANDIDATE_SCHEMA = {
    "query_hour": UTC_DATETIME,
    "subject_uri": pl.String,
    "selection_source": pl.String,
    "prior_like_count": pl.UInt64,
}
NEGATIVE_POST_URI_SCHEMA = {"subject_uri": pl.String}
LOCAL_FINALIST_COLUMNS = [
    "query_hour",
    "subject_uri",
    "post_created_at",
    "prior_like_count",
    "selection_method",
    "_selection_rank",
]
LOCAL_FINALIST_SCHEMA = {
    "query_hour": UTC_DATETIME,
    "subject_uri": pl.String,
    "post_created_at": UTC_DATETIME,
    "prior_like_count": pl.UInt64,
    "selection_method": pl.String,
    "_selection_rank": pl.UInt64,
}


def empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Create a typed empty frame for quota edge cases."""

    return pl.DataFrame(schema=schema)


def calculate_popular_quota(
    negative_candidates_per_hour: int,
    popular_candidate_fraction: float,
) -> int:
    """Round K * P to the nearest integer, with exact halves rounded upward."""
    if negative_candidates_per_hour < 0:
        raise ValueError("negative_candidates_per_hour must be non-negative")
    if not 0.0 <= popular_candidate_fraction <= 1.0:
        raise ValueError("popular_candidate_fraction must be between 0 and 1")
    return math.floor(
        negative_candidates_per_hour * popular_candidate_fraction + 0.5
    )


def selection_rank_expr(selection_method: str, random_seed: int) -> pl.Expr:
    """Return the stable source-specific rank used for uniform sampling."""
    if selection_method not in {"popular", "random"}:
        raise ValueError(f"Unsupported selection method: {selection_method}")
    return (
        pl.concat_str(
            [
                pl.lit(f"negative-selection-{selection_method}"),
                pl.col("query_hour").dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
                pl.col("subject_uri"),
            ],
            separator="|",
        )
        .hash(seed=random_seed)
        .alias("_selection_rank")
    )


def _top_per_hour(
    candidate_hours_df: pl.DataFrame,
    *,
    selection_method: str,
    limit: int,
    random_seed: int,
) -> pl.DataFrame:
    """Select one URI partition's best deterministic rows for each hour."""

    if limit == 0 or candidate_hours_df.is_empty():
        return empty_frame(LOCAL_FINALIST_SCHEMA)
    return (
        candidate_hours_df.with_columns(
            pl.lit(selection_method).alias("selection_method"),
            selection_rank_expr(selection_method, random_seed),
        )
        .sort(["query_hour", "_selection_rank", "subject_uri"])
        .group_by("query_hour", maintain_order=True)
        .head(limit)
        .select(LOCAL_FINALIST_COLUMNS)
        .sort(["query_hour", "selection_method", "_selection_rank", "subject_uri"])
    )


def select_local_finalists(
    candidate_hours_df: pl.DataFrame,
    *,
    negative_candidates_per_hour: int,
    min_likes_for_popular_candidate: int,
    random_seed: int,
) -> pl.DataFrame:
    """Retain bounded per-partition finalists for both selection methods.

    Keeping the first K rows for each method in every URI partition is sufficient
    to recover the exact global result: the final popular selection can choose at
    most K rows, and after those rows are excluded the random phase also needs at
    most K rows in total.
    """
    if negative_candidates_per_hour < 0:
        raise ValueError("negative_candidates_per_hour must be non-negative")
    if min_likes_for_popular_candidate < 0:
        raise ValueError("min_likes_for_popular_candidate must be non-negative")
    if candidate_hours_df.columns != candidate_popularity.CANDIDATE_HOUR_COLUMNS:
        raise ValueError("Candidate-hour popularity columns are invalid")
    if candidate_hours_df.schema != pl.Schema(candidate_popularity.CANDIDATE_HOUR_SCHEMA):
        raise ValueError(f"Candidate-hour popularity schema is invalid: {candidate_hours_df.schema}")
    if candidate_hours_df.height != candidate_hours_df.unique(
        subset=["query_hour", "subject_uri"]
    ).height:
        raise ValueError("Candidate-hour popularity contains duplicate keys")

    popular = _top_per_hour(
        candidate_hours_df.filter(
            pl.col("prior_like_count") >= min_likes_for_popular_candidate
        ),
        selection_method="popular",
        limit=negative_candidates_per_hour,
        random_seed=random_seed,
    )
    random = _top_per_hour(
        candidate_hours_df,
        selection_method="random",
        limit=negative_candidates_per_hour,
        random_seed=random_seed,
    )
    return pl.concat([popular, random]).sort(
        ["query_hour", "selection_method", "_selection_rank", "subject_uri"]
    )


def select_hourly_candidates(
    local_finalists_df: pl.DataFrame,
    *,
    negative_candidates_per_hour: int,
    popular_candidate_fraction: float,
) -> pl.DataFrame:
    """Apply the global popular-first quota and random shortfall filling."""
    popular_quota = calculate_popular_quota(
        negative_candidates_per_hour,
        popular_candidate_fraction,
    )
    if local_finalists_df.columns != LOCAL_FINALIST_COLUMNS:
        raise ValueError("Local finalist columns are invalid")
    if local_finalists_df.schema != pl.Schema(LOCAL_FINALIST_SCHEMA):
        raise ValueError(f"Local finalist schema is invalid: {local_finalists_df.schema}")
    if local_finalists_df.filter(
        ~pl.col("selection_method").is_in(["popular", "random"])
    ).height:
        raise ValueError("Local finalists contain an unsupported selection method")
    if negative_candidates_per_hour == 0 or local_finalists_df.is_empty():
        return empty_frame(HOURLY_CANDIDATE_SCHEMA)

    if popular_quota == 0:
        popular = empty_frame(LOCAL_FINALIST_SCHEMA)
    else:
        popular = (
            local_finalists_df.filter(pl.col("selection_method") == "popular")
            .unique(subset=["query_hour", "subject_uri"], keep="first")
            .sort(["query_hour", "_selection_rank", "subject_uri"])
            .group_by("query_hour", maintain_order=True)
            .head(popular_quota)
        )
    popular_keys = popular.select("query_hour", "subject_uri")
    popular_counts = popular.group_by("query_hour").len(name="_popular_count")
    random = (
        local_finalists_df.filter(pl.col("selection_method") == "random")
        .unique(subset=["query_hour", "subject_uri"], keep="first")
        .join(popular_keys, on=["query_hour", "subject_uri"], how="anti")
        .join(popular_counts, on="query_hour", how="left")
        .with_columns(pl.col("_popular_count").fill_null(0))
        .sort(["query_hour", "_selection_rank", "subject_uri"])
        .with_columns(pl.int_range(pl.len()).over("query_hour").alias("_position"))
        .filter(
            pl.col("_position")
            < pl.lit(negative_candidates_per_hour) - pl.col("_popular_count")
        )
    )

    popular_output = popular.select(
        "query_hour",
        "subject_uri",
        pl.lit("popular").alias("selection_source"),
        "prior_like_count",
    )
    random_output = random.select(
        "query_hour",
        "subject_uri",
        pl.lit("random").alias("selection_source"),
        "prior_like_count",
    )
    return (
        pl.concat([popular_output, random_output])
        .select(HOURLY_CANDIDATE_COLUMNS)
        .sort(["query_hour", "subject_uri"])
    )


def validate_hourly_candidates(
    hourly_candidates_df: pl.DataFrame,
    *,
    negative_candidates_per_hour: int,
) -> None:
    """Validate one completed hour-partition of the public candidate dataset."""
    if hourly_candidates_df.columns != HOURLY_CANDIDATE_COLUMNS:
        raise ValueError("Hourly candidate columns are invalid")
    if hourly_candidates_df.schema != pl.Schema(HOURLY_CANDIDATE_SCHEMA):
        raise ValueError(f"Hourly candidate schema is invalid: {hourly_candidates_df.schema}")
    if hourly_candidates_df.select(pl.any_horizontal(pl.all().is_null())).to_series().any():
        raise ValueError("Hourly candidates contain null values")
    if hourly_candidates_df.filter(
        ~pl.col("selection_source").is_in(["popular", "random"])
    ).height:
        raise ValueError("Hourly candidates contain an unsupported selection source")
    if hourly_candidates_df.height != hourly_candidates_df.unique(
        subset=["query_hour", "subject_uri"]
    ).height:
        raise ValueError("Hourly candidates contain duplicate query-hour/post keys")
    if not hourly_candidates_df.equals(
        hourly_candidates_df.sort(["query_hour", "subject_uri"])
    ):
        raise ValueError("Hourly candidates are not deterministically sorted")
    if hourly_candidates_df.group_by("query_hour").len().filter(
        pl.col("len") > negative_candidates_per_hour
    ).height:
        raise ValueError("Hourly candidate count exceeds negative_candidates_per_hour")
