"""Reusable transformations and schemas for Stage 3 post selection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl


UTC_DATETIME = pl.Datetime("us", "UTC")
HASH_BUCKET_COUNT = 1_000_000
POST_COLUMNS = ["subject_uri", "post_created_at", "author_did", "is_reply"]
REQUIRED_POST_COLUMNS = ["subject_uri", "is_positive", "is_history"]
CANDIDATE_SOURCE_COLUMNS = ["subject_uri", "candidate_source"]
POST_SCHEMA = {
    "subject_uri": pl.String,
    "post_created_at": UTC_DATETIME,
    "author_did": pl.String,
    "is_reply": pl.Boolean,
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
    "is_reply": pl.Boolean,
    "_post_row_valid": pl.Boolean,
}


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
    """Select an approximate stable fraction of unique root posts by URI hash."""
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
    is_reply: bool,
) -> pl.LazyFrame:
    """Normalize narrow root/reply metadata while retaining a validity flag."""
    required = {"at_uri", "record_created_at", "did"}
    missing = required - set(posts_lf.collect_schema().names())
    if missing:
        raise ValueError(f"Input posts are missing required columns: {', '.join(sorted(missing))}")
    normalized = posts_lf.select(
        pl.col("at_uri").cast(pl.String).alias("subject_uri"),
        _utc_timestamp_expr(posts_lf, "record_created_at").alias("post_created_at"),
        pl.col("did").cast(pl.String).alias("author_did"),
        pl.lit(is_reply, dtype=pl.Boolean).alias("is_reply"),
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
    """Keep latest valid metadata per URI with a stable author tie-break."""
    invalid_row_count = normalized_posts_df.filter(~pl.col("_post_row_valid")).height
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
        .select(POST_COLUMNS)
        .sort("subject_uri")
    )
    return selected, {
        "source_row_count": normalized_posts_df.height,
        "invalid_row_count": invalid_row_count,
        "duplicate_row_count": duplicate_row_count,
        "duplicate_uri_count": duplicate_uri_count,
        "unique_valid_count": selected.height,
    }


def build_required_posts(required_rows_df: pl.DataFrame) -> pl.DataFrame:
    if required_rows_df.is_empty():
        return empty_frame(REQUIRED_POST_SCHEMA)
    return (
        required_rows_df.group_by("subject_uri")
        .agg(pl.col("is_positive").max(), pl.col("is_history").max())
        .select(REQUIRED_POST_COLUMNS)
        .sort("subject_uri")
    )


def resolve_root_and_reply_posts(
    root_posts_df: pl.DataFrame,
    reply_posts_df: pl.DataFrame,
) -> tuple[pl.DataFrame, int]:
    """Combine metadata with deterministic root precedence for URI collisions."""
    root_uris = root_posts_df.select("subject_uri")
    overlap_count = reply_posts_df.join(root_uris, on="subject_uri", how="semi").height
    replies_without_roots = reply_posts_df.join(root_uris, on="subject_uri", how="anti")
    return (
        pl.concat([root_posts_df, replies_without_roots])
        .sort("subject_uri"),
        overlap_count,
    )


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
        if not frame.equals(frame.sort(sort_columns)):
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

    if posts_df.get_column("is_reply").null_count():
        raise ValueError("posts contains a null is_reply value")
    if required_posts_df.filter(~pl.col("is_positive") & ~pl.col("is_history")).height:
        raise ValueError("required_posts contains a row with no required role")
    if candidate_sources_df.filter(pl.col("candidate_source") != "random").height:
        raise ValueError("candidate_sources contains an unsupported source")
    candidate_posts = candidate_sources_df.join(posts_df, on="subject_uri", how="left")
    if candidate_posts.filter(pl.col("post_created_at").is_null()).height:
        raise ValueError("candidate_sources contains a URI missing from posts")
    if candidate_posts.filter(pl.col("is_reply")).height:
        raise ValueError("candidate_sources contains a reply")
    expected_missing = required_posts_df.join(
        posts_df.select("subject_uri"), on="subject_uri", how="anti"
    ).sort("subject_uri")
    if not expected_missing.equals(missing_required_posts_df):
        raise ValueError("missing_required_posts does not equal required_posts anti-joined to posts")
    if missing_required_posts_df.filter(pl.col("is_positive")).height:
        raise ValueError("A required positive post is missing from root-post metadata")
    positive_posts = required_posts_df.filter(pl.col("is_positive")).join(
        posts_df, on="subject_uri", how="inner"
    )
    if positive_posts.filter(pl.col("is_reply")).height:
        raise ValueError("A required positive resolved only as a reply")
