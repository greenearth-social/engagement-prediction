"""Reusable transformations and schemas for Stage 3 post selection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


UTC_DATETIME = pl.Datetime("us", "UTC")
HASH_BUCKET_COUNT = 1_000_000
POST_COLUMNS = [
    "subject_uri",
    "post_created_at",
    "author_did",
    "news_social_concern_score",
    "political_inference_indexed_at",
    "is_political",
]
REQUIRED_POST_COLUMNS = ["subject_uri", "is_positive", "is_history"]
CANDIDATE_SOURCE_COLUMNS = ["subject_uri", "candidate_source"]
POST_SCHEMA = {
    "subject_uri": pl.String,
    "post_created_at": UTC_DATETIME,
    "author_did": pl.String,
    "news_social_concern_score": pl.Float64,
    "political_inference_indexed_at": UTC_DATETIME,
    "is_political": pl.Boolean,
}
REQUIRED_POST_SCHEMA = {
    "subject_uri": pl.String,
    "is_positive": pl.Boolean,
    "is_history": pl.Boolean,
}
CANDIDATE_SOURCE_SCHEMA = {
    "subject_uri": pl.String,
    "candidate_source": pl.String,
}
NORMALIZED_POST_SCHEMA = {
    "subject_uri": pl.String,
    "post_created_at": UTC_DATETIME,
    "author_did": pl.String,
    "_post_row_valid": pl.Boolean,
}
NORMALIZED_INFERENCE_SCHEMA = {
    "subject_uri": pl.String,
    "political_inference_indexed_at": UTC_DATETIME,
    "news_social_concern_score": pl.Float64,
    "_raw_news_social_concern_score": pl.Float64,
    "_inference_row_valid": pl.Boolean,
    "_score_missing_or_invalid": pl.Boolean,
}
LATEST_INFERENCE_SCHEMA = {
    "subject_uri": pl.String,
    "news_social_concern_score": pl.Float64,
    "political_inference_indexed_at": UTC_DATETIME,
    "is_political": pl.Boolean,
}
NEWS_SOCIAL_CONCERN_JSON_PATH = (
    '$.text["message.commit.record.text"].topic["News & Social Concern"]'
)


def empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def post_partition_expr(partition_count: int) -> pl.Expr:
    """Assign a post URI to a stable partition for bounded processing."""
    if partition_count <= 0:
        raise ValueError("post_selection_partition_count must be positive")
    return (
        pl.concat_str([pl.lit("post-selection"), pl.col("subject_uri")], separator="|")
        .hash(seed=0)
        .mod(pl.lit(partition_count, dtype=pl.UInt64))
        .cast(pl.UInt32)
        .alias("_post_partition")
    )


def random_candidate_expr(fraction: float, random_seed: int) -> pl.Expr:
    """Select an approximate stable fraction of unique posts by URI hash."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("random_candidate_sampling_fraction must be between 0 and 1")
    selected_bucket_count = int(fraction * HASH_BUCKET_COUNT)
    if selected_bucket_count == 0:
        return pl.lit(False)
    if selected_bucket_count == HASH_BUCKET_COUNT:
        return pl.lit(True)
    bucket = (
        pl.concat_str([pl.lit("random-candidate"), pl.col("subject_uri")], separator="|")
        .hash(seed=random_seed)
        .mod(pl.lit(HASH_BUCKET_COUNT, dtype=pl.UInt64))
    )
    return bucket < pl.lit(selected_bucket_count, dtype=pl.UInt64)


def political_priority_expr(random_seed: int) -> pl.Expr:
    return (
        pl.concat_str([pl.lit("political-candidate"), pl.col("subject_uri")], separator="|")
        .hash(seed=random_seed)
        .alias("_political_priority")
    )


def _utc_timestamp_expr(lf: pl.LazyFrame, column: str) -> pl.Expr:
    schema = lf.collect_schema()
    if column not in schema:
        raise ValueError(f"Input data is missing required column {column!r}")
    dtype = schema[column]
    value = pl.col(column)
    if dtype == pl.String:
        has_timezone = value.str.contains(r"(Z|[+-]\d{2}:?\d{2})$")
        normalized = pl.when(has_timezone).then(value).otherwise(value + pl.lit("Z"))
        return normalized.str.to_datetime(
            format="%Y-%m-%dT%H:%M:%S%.f%#z",
            time_zone="UTC",
            strict=False,
        )
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is None:
            return value.dt.replace_time_zone("UTC")
        return value.dt.convert_time_zone("UTC")
    raise ValueError(f"{column} must be a string or datetime column, found {dtype}")


def normalize_posts(
    posts_lf: pl.LazyFrame,
    *,
    posts_start: datetime,
    posts_end: datetime,
) -> pl.LazyFrame:
    """Normalize narrow post metadata while retaining an input-validity flag."""
    required = {"at_uri", "record_created_at", "did"}
    missing = required - set(posts_lf.collect_schema().names())
    if missing:
        raise ValueError(f"Input posts are missing required columns: {', '.join(sorted(missing))}")
    normalized = posts_lf.select(
        pl.col("at_uri").cast(pl.String).alias("subject_uri"),
        _utc_timestamp_expr(posts_lf, "record_created_at").alias("post_created_at"),
        pl.col("did").cast(pl.String).alias("author_did"),
    )
    return normalized.with_columns(
        (
            pl.col("subject_uri").is_not_null()
            & (pl.col("subject_uri").str.len_chars() > 0)
            & pl.col("post_created_at").is_not_null()
            & (pl.col("post_created_at") >= pl.lit(posts_start))
            & (pl.col("post_created_at") < pl.lit(posts_end))
            & pl.col("author_did").is_not_null()
            & (pl.col("author_did").str.len_chars() > 0)
        ).alias("_post_row_valid")
    )


def select_latest_post_rows(
    normalized_posts_df: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Keep the latest valid metadata row for each URI with a stable author tie-break."""
    invalid_row_count = normalized_posts_df.filter(
        ~pl.col("_post_row_valid")
    ).height
    valid = normalized_posts_df.filter(pl.col("_post_row_valid"))
    duplicate_counts = valid.group_by("subject_uri").len()
    duplicate_uri_count = duplicate_counts.filter(pl.col("len") > 1).height
    duplicate_row_count = int(
        duplicate_counts.select((pl.col("len") - 1).clip(lower_bound=0).sum()).item()
        or 0
    )
    selected = (
        valid.sort(
            ["subject_uri", "post_created_at", "author_did"],
            descending=[False, True, False],
        )
        .unique(subset="subject_uri", keep="first", maintain_order=True)
        .select("subject_uri", "post_created_at", "author_did")
        .sort("subject_uri")
    )
    return selected, {
        "post_source_row_count": normalized_posts_df.height,
        "invalid_post_row_count": invalid_row_count,
        "duplicate_post_row_count": duplicate_row_count,
        "duplicate_post_uri_count": duplicate_uri_count,
        "unique_valid_post_count": selected.height,
    }


def normalize_inferences(inferences_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Extract only the News & Social Concern score and normalize inference time."""
    required = {"at_uri", "indexed_at", "inferences"}
    missing = required - set(inferences_lf.collect_schema().names())
    if missing:
        raise ValueError(
            f"Input inferences are missing required columns: {', '.join(sorted(missing))}"
        )
    score = (
        pl.col("inferences")
        .cast(pl.String)
        .str.json_path_match(NEWS_SOCIAL_CONCERN_JSON_PATH)
        .cast(pl.Float64, strict=False)
    )
    normalized = inferences_lf.select(
        pl.col("at_uri").cast(pl.String).alias("subject_uri"),
        _utc_timestamp_expr(inferences_lf, "indexed_at").alias(
            "political_inference_indexed_at"
        ),
        score.alias("_raw_news_social_concern_score"),
    )
    valid_score = (
        pl.col("_raw_news_social_concern_score").is_not_null()
        & pl.col("_raw_news_social_concern_score").is_finite()
        & pl.col("_raw_news_social_concern_score").is_between(0.0, 1.0, closed="both")
    )
    return normalized.select(
        pl.col("subject_uri"),
        pl.col("political_inference_indexed_at"),
        pl.when(valid_score)
        .then(pl.col("_raw_news_social_concern_score"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("news_social_concern_score"),
        pl.col("_raw_news_social_concern_score"),
        (
            pl.col("subject_uri").is_not_null()
            & (pl.col("subject_uri").str.len_chars() > 0)
            & pl.col("political_inference_indexed_at").is_not_null()
        ).alias("_inference_row_valid"),
        (~valid_score).alias("_score_missing_or_invalid"),
    )


def select_latest_inferences(
    normalized_inferences_df: pl.DataFrame,
    *,
    political_score_threshold: float,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Select the latest inference per URI and reject conflicting timestamp ties."""
    invalid_row_count = normalized_inferences_df.filter(
        ~pl.col("_inference_row_valid")
    ).height
    valid = normalized_inferences_df.filter(pl.col("_inference_row_valid"))
    conflicts = (
        valid.group_by(["subject_uri", "political_inference_indexed_at"])
        .agg(
            pl.col("_raw_news_social_concern_score")
            .n_unique()
            .alias("_score_count")
        )
        .filter(pl.col("_score_count") > 1)
    )
    if not conflicts.is_empty():
        sample = conflicts.select("subject_uri", "political_inference_indexed_at").row(
            0, named=True
        )
        raise ValueError(
            "Conflicting political inference rows share the latest-selection key "
            f"for {sample['subject_uri']} at {sample['political_inference_indexed_at']}"
        )
    latest = (
        valid.sort(
            ["subject_uri", "political_inference_indexed_at"],
            descending=[False, True],
        )
        .unique(
            subset=["subject_uri", "political_inference_indexed_at"],
            keep="first",
            maintain_order=True,
        )
        .unique(subset="subject_uri", keep="first", maintain_order=True)
        .with_columns(
            (
                pl.col("news_social_concern_score").is_not_null()
                & (pl.col("news_social_concern_score") >= political_score_threshold)
            ).alias("is_political")
        )
        .select(list(LATEST_INFERENCE_SCHEMA))
        .sort("subject_uri")
    )
    return latest, {
        "inference_source_row_count": normalized_inferences_df.height,
        "invalid_inference_row_count": invalid_row_count,
        "missing_or_invalid_inference_score_count": valid.filter(
            pl.col("_score_missing_or_invalid")
        ).height,
        "latest_inference_count": latest.height,
    }


def build_required_posts(required_rows_df: pl.DataFrame) -> pl.DataFrame:
    if required_rows_df.is_empty():
        return empty_frame(REQUIRED_POST_SCHEMA)
    return (
        required_rows_df.group_by("subject_uri")
        .agg(
            pl.col("is_positive").max(),
            pl.col("is_history").max(),
        )
        .select(list(REQUIRED_POST_SCHEMA))
        .sort("subject_uri")
    )


def label_posts(
    posts_df: pl.DataFrame,
    latest_inferences_df: pl.DataFrame,
) -> pl.DataFrame:
    if latest_inferences_df.is_empty():
        return posts_df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("news_social_concern_score"),
            pl.lit(None, dtype=UTC_DATETIME).alias("political_inference_indexed_at"),
            pl.lit(None, dtype=pl.Boolean).alias("is_political"),
        ).select(list(POST_SCHEMA))
    return posts_df.join(latest_inferences_df, on="subject_uri", how="left").select(
        list(POST_SCHEMA)
    )


def select_political_candidates_for_day(
    eligible_df: pl.DataFrame,
    *,
    max_candidates_per_creation_hour: int,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Apply a deterministic per-creation-hour cap within one creation day."""
    if eligible_df.is_empty():
        return eligible_df, []
    ranked = eligible_df.with_columns(
        pl.col("post_created_at").dt.truncate("1h").alias("_post_created_hour")
    ).sort(["_post_created_hour", "_political_priority", "subject_uri"])
    selected = (
        ranked.group_by("_post_created_hour", maintain_order=True)
        .head(max_candidates_per_creation_hour)
        .sort(["_post_created_hour", "_political_priority", "subject_uri"])
    )
    eligible_counts = ranked.group_by("_post_created_hour").len().rename(
        {"len": "eligible_count"}
    )
    selected_counts = selected.group_by("_post_created_hour").len().rename(
        {"len": "selected_count"}
    )
    stats = (
        eligible_counts.join(selected_counts, on="_post_created_hour", how="left")
        .with_columns(pl.col("selected_count").fill_null(0))
        .sort("_post_created_hour")
    )
    hour_stats = [
        {
            "post_created_hour": row["_post_created_hour"].isoformat(),
            "eligible_count": int(row["eligible_count"]),
            "selected_count": int(row["selected_count"]),
            "discarded_count": int(row["eligible_count"] - row["selected_count"]),
        }
        for row in stats.iter_rows(named=True)
    ]
    return selected.drop("_post_created_hour"), hour_stats


def partition_parquet_paths(
    dataset_path: Path,
    partition_id: int,
    *,
    key: str = "_post_partition",
) -> list[Path]:
    partition_dir = Path(dataset_path) / f"{key}={partition_id}"
    return sorted(partition_dir.rglob("*.parquet")) if partition_dir.exists() else []


def validate_public_partition(
    *,
    posts_df: pl.DataFrame,
    required_posts_df: pl.DataFrame,
    candidate_sources_df: pl.DataFrame,
    missing_required_posts_df: pl.DataFrame,
    partition_id: int,
    partition_count: int,
    political_score_threshold: float,
) -> None:
    """Validate one aligned URI-hash partition across all public datasets."""
    expected = (
        (posts_df, POST_COLUMNS, POST_SCHEMA, ["subject_uri"]),
        (required_posts_df, REQUIRED_POST_COLUMNS, REQUIRED_POST_SCHEMA, ["subject_uri"]),
        (
            candidate_sources_df,
            CANDIDATE_SOURCE_COLUMNS,
            CANDIDATE_SOURCE_SCHEMA,
            ["subject_uri", "candidate_source"],
        ),
        (
            missing_required_posts_df,
            REQUIRED_POST_COLUMNS,
            REQUIRED_POST_SCHEMA,
            ["subject_uri"],
        ),
    )
    for frame, columns, schema, sort_columns in expected:
        if frame.columns != columns or frame.schema != pl.Schema(schema):
            raise ValueError(f"Unexpected public post-selection schema: {frame.schema}")
        if frame.get_column("subject_uri").null_count():
            raise ValueError("Public post-selection artifact contains a null subject_uri")
        if frame.height != frame.unique(subset=sort_columns).height:
            raise ValueError(f"Duplicate rows in public post-selection artifact {columns}")
        if frame.sort(sort_columns).to_dicts() != frame.to_dicts():
            raise ValueError(f"Public post-selection artifact is not sorted by {sort_columns}")
        if not frame.is_empty():
            assigned = (
                frame.select("subject_uri")
                .with_columns(post_partition_expr(partition_count))
                .get_column("_post_partition")
                .unique()
                .to_list()
            )
            if assigned != [partition_id]:
                raise ValueError(
                    f"Post-selection partition {partition_id} contains rows assigned to {assigned}"
                )

    if required_posts_df.filter(~pl.col("is_positive") & ~pl.col("is_history")).height:
        raise ValueError("required_posts contains a row with no required role")
    if candidate_sources_df.filter(
        ~pl.col("candidate_source").is_in(["random", "political"])
    ).height:
        raise ValueError("candidate_sources contains an unsupported source")
    if candidate_sources_df.join(posts_df.select("subject_uri"), on="subject_uri", how="anti").height:
        raise ValueError("candidate_sources contains a URI missing from posts")
    expected_missing = required_posts_df.join(
        posts_df.select("subject_uri"), on="subject_uri", how="anti"
    ).sort("subject_uri")
    if not expected_missing.equals(missing_required_posts_df):
        raise ValueError("missing_required_posts does not equal required_posts anti-joined to posts")

    inferred = posts_df.filter(pl.col("political_inference_indexed_at").is_not_null())
    uninferred = posts_df.filter(pl.col("political_inference_indexed_at").is_null())
    if uninferred.filter(
        pl.col("news_social_concern_score").is_not_null() | pl.col("is_political").is_not_null()
    ).height:
        raise ValueError("Posts without an inference timestamp contain political labels")
    if inferred.filter(pl.col("is_political").is_null()).height:
        raise ValueError("Posts with an inference timestamp are missing is_political")
    inconsistent = inferred.filter(
        pl.col("is_political")
        != (
            pl.col("news_social_concern_score").is_not_null()
            & (pl.col("news_social_concern_score") >= political_score_threshold)
        )
    )
    if inconsistent.height:
        raise ValueError("Post political score and label are inconsistent")
    political_uris = candidate_sources_df.filter(
        pl.col("candidate_source") == "political"
    ).select("subject_uri")
    if political_uris.join(
        posts_df.filter(pl.col("is_political")).select("subject_uri"),
        on="subject_uri",
        how="anti",
    ).height:
        raise ValueError("Political candidate source contains a non-political post")
