"""Reusable transformations and schemas for Stage 3 post selection."""

from __future__ import annotations

import polars as pl

from engagement_prediction.data import source_metadata


UTC_DATETIME = source_metadata.UTC_DATETIME
HASH_BUCKET_COUNT = 1_000_000
POST_COLUMNS = source_metadata.POST_METADATA_COLUMNS
REQUIRED_POST_COLUMNS = ["subject_uri", "is_positive", "is_history"]
CANDIDATE_SOURCE_COLUMNS = ["subject_uri", "candidate_source"]
POST_SCHEMA = source_metadata.POST_METADATA_SCHEMA
REQUIRED_POST_SCHEMA = {
    "subject_uri": pl.String,
    "is_positive": pl.Boolean,
    "is_history": pl.Boolean,
}
CANDIDATE_SOURCE_SCHEMA = {
    "subject_uri": pl.String,
    "candidate_source": pl.String,
}
NORMALIZED_POST_SCHEMA = source_metadata.NORMALIZED_METADATA_SCHEMA
empty_frame = source_metadata.empty_frame
post_partition_expr = source_metadata.uri_partition_expr


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


_utc_timestamp_expr = source_metadata._utc_timestamp_expr
normalize_posts = source_metadata.normalize_source_records
select_latest_post_rows = source_metadata.select_latest_metadata_rows


def build_required_posts(required_rows_df: pl.DataFrame) -> pl.DataFrame:
    """Collapse repeated requirement rows while preserving both role flags."""

    if required_rows_df.is_empty():
        return empty_frame(REQUIRED_POST_SCHEMA)
    return (
        required_rows_df.group_by("subject_uri")
        .agg(pl.col("is_positive").max(), pl.col("is_history").max())
        .select(REQUIRED_POST_COLUMNS)
        .sort("subject_uri")
    )


resolve_root_and_reply_posts = source_metadata.apply_root_precedence
partition_parquet_paths = source_metadata.partition_parquet_paths


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
