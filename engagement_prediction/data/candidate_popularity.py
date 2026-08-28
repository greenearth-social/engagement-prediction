"""As-of popularity calculation for Stage 4 negative candidates."""

from __future__ import annotations

import polars as pl

from engagement_prediction.data import like_counts


UTC_DATETIME = pl.Datetime("us", "UTC")
CANDIDATE_COLUMNS = ["subject_uri", "post_created_at"]
CANDIDATE_SCHEMA = {
    "subject_uri": pl.String,
    "post_created_at": UTC_DATETIME,
}
NORMALIZED_LIKE_COLUMNS = ["subject_uri", "like_created_at"]
NORMALIZED_LIKE_SCHEMA = {
    "subject_uri": pl.String,
    "like_created_at": UTC_DATETIME,
}
CANDIDATE_HOUR_COLUMNS = [
    "query_hour",
    "subject_uri",
    "post_created_at",
    "prior_like_count",
]
CANDIDATE_HOUR_SCHEMA = {
    "query_hour": UTC_DATETIME,
    "subject_uri": pl.String,
    "post_created_at": UTC_DATETIME,
    "prior_like_count": pl.UInt64,
}


def empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Create a typed empty frame for sparse partitions and edge cases."""

    return pl.DataFrame(schema=schema)


def validate_query_hours(query_hours_df: pl.DataFrame) -> pl.DataFrame:
    """Validate and deterministically normalize the Stage 1 query-hour set."""
    if query_hours_df.columns != ["query_hour"]:
        raise ValueError("Query hours must contain only query_hour")
    dtype = query_hours_df.schema["query_hour"]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
        raise ValueError(f"query_hour must be a UTC datetime, found {dtype}")
    if query_hours_df.get_column("query_hour").null_count():
        raise ValueError("Query hours contain a null query_hour")
    if query_hours_df.filter(
        (pl.col("query_hour").dt.minute() != 0)
        | (pl.col("query_hour").dt.second() != 0)
        | (pl.col("query_hour").dt.microsecond() != 0)
    ).height:
        raise ValueError("query_hour values must be aligned to the start of an hour")
    return query_hours_df.unique().sort("query_hour")


def build_candidate_reservoir(
    posts_df: pl.DataFrame,
    candidate_sources_df: pl.DataFrame,
) -> pl.DataFrame:
    """Return unique root metadata for posts represented in candidate_sources."""
    required_post_columns = {"subject_uri", "post_created_at", "is_reply"}
    if not required_post_columns.issubset(posts_df.columns):
        raise ValueError("Stage 3 posts are missing candidate metadata columns")
    if not {"subject_uri", "candidate_source"}.issubset(candidate_sources_df.columns):
        raise ValueError("Stage 3 candidate_sources has an unexpected schema")

    candidate_uris = candidate_sources_df.select("subject_uri").unique()
    candidates = candidate_uris.join(
        posts_df.select("subject_uri", "post_created_at", "is_reply"),
        on="subject_uri",
        how="left",
    )
    if candidates.get_column("post_created_at").null_count():
        raise ValueError("Stage 3 candidate_sources contains a URI missing from posts")
    if candidates.filter(pl.col("is_reply")).height:
        raise ValueError("Stage 3 candidate_sources contains a reply")
    result = candidates.select(CANDIDATE_COLUMNS).sort("subject_uri")
    if result.schema != pl.Schema(CANDIDATE_SCHEMA):
        raise ValueError(f"Unexpected candidate reservoir schema: {result.schema}")
    return result


def build_candidate_hour_popularity(
    candidates_df: pl.DataFrame,
    likes_df: pl.DataFrame,
    query_hours_df: pl.DataFrame,
    *,
    max_candidate_age_hours: int,
) -> pl.DataFrame:
    """Build eligible candidate-hour rows with strictly prior raw-like counts.

    Candidate creation timestamps are bucketed to UTC hours. A candidate is
    eligible in its creation-hour bucket and the following
    ``max_candidate_age_hours - 1`` buckets. Like rows are intentionally not
    deduplicated; each valid source row contributes one to the cumulative count.
    """
    if max_candidate_age_hours <= 0:
        raise ValueError("max_candidate_age_hours must be positive")
    query_hours_df = validate_query_hours(query_hours_df)
    if candidates_df.columns != CANDIDATE_COLUMNS or candidates_df.schema != pl.Schema(
        CANDIDATE_SCHEMA
    ):
        raise ValueError(f"Unexpected candidate schema: {candidates_df.schema}")
    if likes_df.columns != NORMALIZED_LIKE_COLUMNS or likes_df.schema != pl.Schema(
        NORMALIZED_LIKE_SCHEMA
    ):
        raise ValueError(f"Unexpected normalized-like schema: {likes_df.schema}")
    if candidates_df.get_column("subject_uri").null_count() or candidates_df.get_column(
        "post_created_at"
    ).null_count():
        raise ValueError("Candidate reservoir contains null values")
    if candidates_df.height != candidates_df.unique(subset="subject_uri").height:
        raise ValueError("Candidate reservoir contains duplicate subject_uri values")
    if candidates_df.is_empty() or query_hours_df.is_empty():
        return empty_frame(CANDIDATE_HOUR_SCHEMA)

    offsets_df = pl.DataFrame({
        "_age_hours": pl.Series(range(max_candidate_age_hours), dtype=pl.Int64),
    })
    eligible = (
        candidates_df.with_columns(
            pl.col("post_created_at").dt.truncate("1h").alias("_creation_hour")
        )
        .join(offsets_df, how="cross")
        .with_columns(
            (
                pl.col("_creation_hour")
                + pl.duration(hours=pl.col("_age_hours"))
            ).alias("query_hour")
        )
        .join(query_hours_df, on="query_hour", how="semi")
        .select("query_hour", "subject_uri", "post_created_at")
        .sort(["subject_uri", "query_hour"])
    )
    if eligible.is_empty():
        return empty_frame(CANDIDATE_HOUR_SCHEMA)

    # Stage 4 already has the complete eligible candidate-hour frame, including
    # post metadata. Filter before aggregation, then attach cumulative counts
    # directly so we do not normalize/sort the large key frame and hash-join it
    # back solely to recover ``post_created_at``.
    matched_likes = likes_df.join(
        candidates_df.select("subject_uri"), on="subject_uri", how="semi"
    )
    if matched_likes.is_empty():
        return (
            eligible.with_columns(
                pl.lit(0, dtype=pl.UInt64).alias("prior_like_count")
            )
            .select(CANDIDATE_HOUR_COLUMNS)
            .sort(["query_hour", "subject_uri"])
        )

    cumulative_likes = like_counts.build_cumulative_like_counts(matched_likes)
    return (
        eligible.join_asof(
            cumulative_likes,
            left_on="query_hour",
            right_on="like_hour",
            by="subject_uri",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .with_columns(
            pl.col("cumulative_like_count")
            .fill_null(0)
            .cast(pl.UInt64)
            .alias("prior_like_count")
        )
        .select(CANDIDATE_HOUR_COLUMNS)
        .sort(["query_hour", "subject_uri"])
    )
