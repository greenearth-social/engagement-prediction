"""Canonical strict as-of counting for raw like events."""

from __future__ import annotations

import polars as pl


UTC_DATETIME = pl.Datetime("us", "UTC")
POST_HOUR_COLUMNS = ["subject_uri", "query_hour"]
POST_HOUR_SCHEMA = {
    "subject_uri": pl.String,
    "query_hour": UTC_DATETIME,
}
LIKE_EVENT_COLUMNS = ["subject_uri", "like_created_at"]
LIKE_EVENT_SCHEMA = {
    "subject_uri": pl.String,
    "like_created_at": UTC_DATETIME,
}
POST_HOUR_COUNT_COLUMNS = [*POST_HOUR_COLUMNS, "prior_like_count"]
POST_HOUR_COUNT_SCHEMA = {
    **POST_HOUR_SCHEMA,
    "prior_like_count": pl.UInt64,
}
CUMULATIVE_LIKE_COUNT_COLUMNS = [
    "subject_uri",
    "like_hour",
    "cumulative_like_count",
]
CUMULATIVE_LIKE_COUNT_SCHEMA = {
    "subject_uri": pl.String,
    "like_hour": UTC_DATETIME,
    "cumulative_like_count": pl.UInt64,
}


def _validated_post_hours(post_hours_df: pl.DataFrame) -> pl.DataFrame:
    missing = set(POST_HOUR_COLUMNS) - set(post_hours_df.columns)
    if missing:
        raise ValueError(
            f"Post-hour rows are missing required columns: {', '.join(sorted(missing))}"
        )
    post_hours = (
        post_hours_df.select(POST_HOUR_COLUMNS)
        .unique()
        .sort(POST_HOUR_COLUMNS)
    )
    if post_hours.schema["subject_uri"] != pl.String:
        raise ValueError(f"subject_uri must be a string, found {post_hours.schema['subject_uri']}")
    if post_hours.schema["query_hour"] != UTC_DATETIME:
        raise ValueError(
            f"query_hour must be a microsecond-resolution UTC datetime, "
            f"found {post_hours.schema['query_hour']}"
        )
    if any(post_hours.get_column(column).null_count() for column in POST_HOUR_COLUMNS):
        raise ValueError("Post-hour rows contain null keys")
    if post_hours.filter(
        (pl.col("query_hour").dt.minute() != 0)
        | (pl.col("query_hour").dt.second() != 0)
        | (pl.col("query_hour").dt.microsecond() != 0)
    ).height:
        raise ValueError("query_hour values must be aligned to the start of an hour")
    return post_hours


def _validated_like_events(events_df: pl.DataFrame) -> pl.DataFrame:
    missing = set(LIKE_EVENT_COLUMNS) - set(events_df.columns)
    if missing:
        raise ValueError(
            f"Like events are missing required columns: {', '.join(sorted(missing))}"
        )
    events = events_df.select(LIKE_EVENT_COLUMNS)
    if events.schema["subject_uri"] != pl.String:
        raise ValueError(f"subject_uri must be a string, found {events.schema['subject_uri']}")
    if events.schema["like_created_at"] != UTC_DATETIME:
        raise ValueError(
            f"like_created_at must be a microsecond-resolution UTC datetime, "
            f"found {events.schema['like_created_at']}"
        )
    if any(events.get_column(column).null_count() for column in LIKE_EVENT_COLUMNS):
        raise ValueError("Like events contain null values")
    return events


def _build_cumulative_like_counts(events: pl.DataFrame) -> pl.DataFrame:
    if events.is_empty():
        return pl.DataFrame(schema=CUMULATIVE_LIKE_COUNT_SCHEMA)
    return (
        events.with_columns(
            pl.col("like_created_at").dt.truncate("1h").alias("like_hour")
        )
        .group_by("subject_uri", "like_hour")
        .len(name="_hour_like_count")
        .sort(["subject_uri", "like_hour"])
        .with_columns(
            pl.col("_hour_like_count")
            .cum_sum()
            .over("subject_uri")
            .cast(pl.UInt64)
            .alias("cumulative_like_count")
        )
        .select(CUMULATIVE_LIKE_COUNT_COLUMNS)
    )


def build_cumulative_like_counts(events_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate raw events into cumulative per-post hourly counts."""

    return _build_cumulative_like_counts(_validated_like_events(events_df))


def _lookup_prior_like_counts(
    post_hours: pl.DataFrame,
    cumulative_likes: pl.DataFrame,
) -> pl.DataFrame:
    if post_hours.is_empty():
        return pl.DataFrame(schema=POST_HOUR_COUNT_SCHEMA)
    if cumulative_likes.is_empty():
        return post_hours.with_columns(
            pl.lit(0, dtype=pl.UInt64).alias("prior_like_count")
        ).select(POST_HOUR_COUNT_COLUMNS)
    return (
        post_hours.join_asof(
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
        .select(POST_HOUR_COUNT_COLUMNS)
        .sort(POST_HOUR_COLUMNS)
    )


def lookup_prior_like_counts(
    post_hours_df: pl.DataFrame,
    cumulative_likes_df: pl.DataFrame,
) -> pl.DataFrame:
    """Look up strict counts from ``build_cumulative_like_counts`` output."""

    post_hours = _validated_post_hours(post_hours_df)
    missing = set(CUMULATIVE_LIKE_COUNT_COLUMNS) - set(cumulative_likes_df.columns)
    if missing:
        raise ValueError(
            "Cumulative like counts are missing required columns: "
            f"{', '.join(sorted(missing))}"
        )
    cumulative_likes = cumulative_likes_df.select(CUMULATIVE_LIKE_COUNT_COLUMNS)
    if cumulative_likes.schema != pl.Schema(CUMULATIVE_LIKE_COUNT_SCHEMA):
        raise ValueError(
            f"Unexpected cumulative like-count schema: {cumulative_likes.schema}"
        )
    return _lookup_prior_like_counts(post_hours, cumulative_likes)


def calculate_prior_like_counts(
    post_hours_df: pl.DataFrame,
    events_df: pl.DataFrame,
) -> pl.DataFrame:
    """Count raw events strictly before each unique post/query-hour pair.

    Events are first collapsed into hourly counts, then cumulatively summed by
    post. A backward as-of join with exact matches disabled excludes every
    event in the query hour while retaining duplicate source events.
    """

    post_hours = _validated_post_hours(post_hours_df)
    if post_hours.is_empty():
        return pl.DataFrame(schema=POST_HOUR_COUNT_SCHEMA)

    events = _validated_like_events(events_df).join(
        post_hours.select("subject_uri").unique(),
        on="subject_uri",
        how="semi",
    )
    if events.is_empty():
        return (
            post_hours.with_columns(
                pl.lit(0, dtype=pl.UInt64).alias("prior_like_count")
            )
            .select(POST_HOUR_COUNT_COLUMNS)
        )

    cumulative_likes = _build_cumulative_like_counts(events)
    return _lookup_prior_like_counts(post_hours, cumulative_likes)
