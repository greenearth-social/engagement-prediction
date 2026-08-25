"""Schemas and reusable transformations for Stage 6 author statistics."""

from __future__ import annotations

from datetime import datetime

import polars as pl


UTC_DATETIME = pl.Datetime("us", "UTC")
PER_POST_COLUMNS = [
    "subject_uri",
    "author_did",
    "post_created_at",
    "is_reply",
    "received_like_count",
]
PER_POST_SCHEMA = {
    "subject_uri": pl.String,
    "author_did": pl.String,
    "post_created_at": UTC_DATETIME,
    "is_reply": pl.Boolean,
    "received_like_count": pl.UInt64,
}
AUTHOR_STAT_COLUMNS = [
    "author_did",
    "post_count",
    "root_post_count",
    "reply_post_count",
    "received_like_count",
    "root_received_like_count",
    "reply_received_like_count",
    "liked_post_count",
    "mean_likes_per_post",
    "median_likes_per_post",
    "max_likes_per_post",
    "first_post_created_at",
    "last_post_created_at",
]
AUTHOR_STAT_SCHEMA = {
    "author_did": pl.String,
    "post_count": pl.UInt64,
    "root_post_count": pl.UInt64,
    "reply_post_count": pl.UInt64,
    "received_like_count": pl.UInt64,
    "root_received_like_count": pl.UInt64,
    "reply_received_like_count": pl.UInt64,
    "liked_post_count": pl.UInt64,
    "mean_likes_per_post": pl.Float64,
    "median_likes_per_post": pl.Float64,
    "max_likes_per_post": pl.UInt64,
    "first_post_created_at": UTC_DATETIME,
    "last_post_created_at": UTC_DATETIME,
}


def empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Create a typed empty frame for sparse post or author partitions."""

    return pl.DataFrame(schema=schema)


def author_partition_expr(partition_count: int) -> pl.Expr:
    """Assign an author DID to a stable partition for bounded aggregation."""
    if partition_count <= 0:
        raise ValueError("author_statistics_partition_count must be positive")
    return (
        pl.concat_str(
            [pl.lit("author-statistics"), pl.col("author_did")],
            separator="|",
        )
        .hash(seed=0)
        .mod(pl.lit(partition_count, dtype=pl.UInt64))
        .cast(pl.UInt32)
        .alias("_author_partition")
    )


def build_per_post_statistics(
    resolved_posts_df: pl.DataFrame,
    like_counts_df: pl.DataFrame,
) -> pl.DataFrame:
    """Attach raw received-like counts to every resolved in-window record."""
    if like_counts_df.is_empty():
        like_counts_df = empty_frame({
            "subject_uri": pl.String,
            "received_like_count": pl.UInt64,
        })
    return (
        resolved_posts_df.join(like_counts_df, on="subject_uri", how="left")
        .with_columns(
            pl.col("received_like_count").fill_null(0).cast(pl.UInt64),
        )
        .select(PER_POST_COLUMNS)
        .sort("subject_uri")
    )


def aggregate_author_statistics(
    per_post_df: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate one descriptive statistics row for every author."""
    if per_post_df.is_empty():
        return empty_frame(AUTHOR_STAT_SCHEMA)
    return (
        per_post_df.group_by("author_did")
        .agg(
            pl.len().cast(pl.UInt64).alias("post_count"),
            (~pl.col("is_reply")).sum().cast(pl.UInt64).alias("root_post_count"),
            pl.col("is_reply").sum().cast(pl.UInt64).alias("reply_post_count"),
            pl.col("received_like_count").sum().cast(pl.UInt64).alias("received_like_count"),
            pl.when(~pl.col("is_reply"))
            .then(pl.col("received_like_count"))
            .otherwise(0)
            .sum()
            .cast(pl.UInt64)
            .alias("root_received_like_count"),
            pl.when(pl.col("is_reply"))
            .then(pl.col("received_like_count"))
            .otherwise(0)
            .sum()
            .cast(pl.UInt64)
            .alias("reply_received_like_count"),
            (pl.col("received_like_count") > 0)
            .sum()
            .cast(pl.UInt64)
            .alias("liked_post_count"),
            pl.col("received_like_count").mean().cast(pl.Float64).alias("mean_likes_per_post"),
            pl.col("received_like_count").median().cast(pl.Float64).alias("median_likes_per_post"),
            pl.col("received_like_count").max().cast(pl.UInt64).alias("max_likes_per_post"),
            pl.col("post_created_at").min().alias("first_post_created_at"),
            pl.col("post_created_at").max().alias("last_post_created_at"),
        )
        .select(AUTHOR_STAT_COLUMNS)
        .sort("author_did")
    )


def validate_per_post_statistics(
    per_post_df: pl.DataFrame,
    *,
    support_start: datetime,
    support_end: datetime,
) -> None:
    """Validate one-row-per-post counts, source bounds, and deterministic order."""

    if per_post_df.columns != PER_POST_COLUMNS or per_post_df.schema != pl.Schema(
        PER_POST_SCHEMA
    ):
        raise ValueError(f"Unexpected per-post author-statistics schema: {per_post_df.schema}")
    if per_post_df.get_column("subject_uri").null_count():
        raise ValueError("Per-post author statistics contain a null subject_uri")
    if per_post_df.get_column("author_did").null_count():
        raise ValueError("Per-post author statistics contain a null author_did")
    if per_post_df.height != per_post_df.unique("subject_uri").height:
        raise ValueError("Per-post author statistics contain duplicate subject_uri values")
    if not per_post_df.equals(per_post_df.sort("subject_uri")):
        raise ValueError("Per-post author statistics are not sorted by subject_uri")
    if per_post_df.filter(
        (pl.col("post_created_at") < support_start)
        | (pl.col("post_created_at") >= support_end)
    ).height:
        raise ValueError("Per-post author statistics contain a timestamp outside the support window")


def validate_author_partition(author_stats_df: pl.DataFrame) -> None:
    """Validate one author-hash partition before it is published."""

    if author_stats_df.columns != AUTHOR_STAT_COLUMNS or author_stats_df.schema != pl.Schema(
        AUTHOR_STAT_SCHEMA
    ):
        raise ValueError(f"Unexpected author-statistics schema: {author_stats_df.schema}")
    if author_stats_df.get_column("author_did").null_count():
        raise ValueError("Author statistics contain a null author_did")
    if author_stats_df.height != author_stats_df.unique("author_did").height:
        raise ValueError("Author statistics contain duplicate author_did values")
    if not author_stats_df.equals(author_stats_df.sort("author_did")):
        raise ValueError("Author statistics are not sorted by author_did")
    invalid = author_stats_df.filter(
        (pl.col("post_count") != pl.col("root_post_count") + pl.col("reply_post_count"))
        | (
            pl.col("received_like_count")
            != pl.col("root_received_like_count") + pl.col("reply_received_like_count")
        )
        | (pl.col("liked_post_count") > pl.col("post_count"))
        | (pl.col("first_post_created_at") > pl.col("last_post_created_at"))
    )
    if invalid.height:
        raise ValueError("Author statistics contain inconsistent aggregate values")


def validate_author_statistics_dataset(author_stats_lf: pl.LazyFrame) -> dict[str, int]:
    """Validate the complete, unfiltered public author-statistics dataset."""
    schema = author_stats_lf.collect_schema()
    if schema != pl.Schema(AUTHOR_STAT_SCHEMA):
        raise ValueError(f"Unexpected public author-statistics schema: {schema}")
    checks = author_stats_lf.select(
        pl.len().alias("author_count"),
        pl.col("author_did").null_count().alias("null_author_count"),
        pl.col("author_did").n_unique().alias("unique_author_count"),
    ).collect(engine="streaming").row(0, named=True)
    author_count = int(checks["author_count"])
    if checks["null_author_count"]:
        raise ValueError("Public author statistics contain a null author_did")
    if int(checks["unique_author_count"]) != author_count:
        raise ValueError("Public author statistics contain duplicate author_did values")
    return {"author_count": author_count}
