"""Schemas and reusable transformations for Stage 5 post-liker histories.

Stage 5 preserves every valid raw like event for the selected post universe;
it does not deduplicate likers or apply an as-of cutoff. Query-time popularity
and future liker-embedding replay can therefore derive their own strict
``like_created_at < query_hour`` views from this lossless event artifact.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from engagement_prediction.data import post_selection


UTC_DATETIME = pl.Datetime("us", "UTC")
POST_LIKER_EVENT_COLUMNS = ["subject_uri", "liker_did", "like_created_at"]
POST_LIKER_EVENT_SCHEMA = {
    "subject_uri": pl.String,
    "liker_did": pl.String,
    "like_created_at": UTC_DATETIME,
}
EVENT_AUDIT_COLUMNS = [
    "subject_uri",
    "like_event_count",
    "first_like_created_at",
    "last_like_created_at",
    "null_liker_did_count",
    "empty_liker_did_count",
    "null_timestamp_count",
]
EVENT_AUDIT_SCHEMA = {
    "subject_uri": pl.String,
    "like_event_count": pl.UInt64,
    "first_like_created_at": UTC_DATETIME,
    "last_like_created_at": UTC_DATETIME,
    "null_liker_did_count": pl.UInt64,
    "empty_liker_did_count": pl.UInt64,
    "null_timestamp_count": pl.UInt64,
}
POST_LIKER_POST_COLUMNS = [
    "subject_uri",
    "is_positive",
    "is_history",
    "is_negative",
    "like_event_count",
    "first_like_created_at",
    "last_like_created_at",
]
POST_LIKER_POST_SCHEMA = {
    "subject_uri": pl.String,
    "is_positive": pl.Boolean,
    "is_history": pl.Boolean,
    "is_negative": pl.Boolean,
    "like_event_count": pl.UInt64,
    "first_like_created_at": UTC_DATETIME,
    "last_like_created_at": UTC_DATETIME,
}
SELECTED_POST_COLUMNS = ["subject_uri", "is_positive", "is_history", "is_negative"]
SELECTED_POST_SCHEMA = {
    "subject_uri": pl.String,
    "is_positive": pl.Boolean,
    "is_history": pl.Boolean,
    "is_negative": pl.Boolean,
}


def empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Create a typed empty frame for sparse URI partitions."""

    return pl.DataFrame(schema=schema)


def build_selected_posts(
    required_posts_df: pl.DataFrame,
    missing_required_posts_df: pl.DataFrame,
    negative_post_uris_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build unique post roles from resolved requirements and final negatives.

    Missing Stage 3 requirements are deliberately excluded because they have
    no resolved metadata or embedding source. A URI can retain several roles.
    """
    resolved_required_df = (
        required_posts_df.join(
            missing_required_posts_df.select("subject_uri"),
            on="subject_uri",
            how="anti",
        )
        .select(
            "subject_uri",
            "is_positive",
            "is_history",
            pl.lit(False, dtype=pl.Boolean).alias("is_negative"),
        )
    )
    negative_roles_df = negative_post_uris_df.select(
        "subject_uri",
        pl.lit(False, dtype=pl.Boolean).alias("is_positive"),
        pl.lit(False, dtype=pl.Boolean).alias("is_history"),
        pl.lit(True, dtype=pl.Boolean).alias("is_negative"),
    )
    selected_rows = pl.concat(
        [resolved_required_df, negative_roles_df],
        how="vertical",
    )
    if selected_rows.is_empty():
        return empty_frame(SELECTED_POST_SCHEMA)
    return (
        selected_rows.group_by("subject_uri")
        .agg(
            pl.col("is_positive").max(),
            pl.col("is_history").max(),
            pl.col("is_negative").max(),
        )
        .select(SELECTED_POST_COLUMNS)
        .sort("subject_uri")
    )


def build_post_liker_posts(
    selected_posts_df: pl.DataFrame,
    event_stats_df: pl.DataFrame,
) -> pl.DataFrame:
    """Attach exact event counts and timestamp bounds to every selected post."""
    if event_stats_df.is_empty():
        event_stats_df = empty_frame({
            "subject_uri": pl.String,
            "like_event_count": pl.UInt64,
            "first_like_created_at": UTC_DATETIME,
            "last_like_created_at": UTC_DATETIME,
        })
    return (
        selected_posts_df.join(event_stats_df, on="subject_uri", how="left")
        .with_columns(
            pl.col("like_event_count").fill_null(0).cast(pl.UInt64),
            pl.col("first_like_created_at").cast(UTC_DATETIME),
            pl.col("last_like_created_at").cast(UTC_DATETIME),
        )
        .select(POST_LIKER_POST_COLUMNS)
        .sort("subject_uri")
    )


def audit_event_partition(events_lf: pl.LazyFrame) -> pl.DataFrame:
    """Scan persisted events once into per-post statistics and validity counts."""

    event_schema = events_lf.collect_schema()
    if event_schema != pl.Schema(POST_LIKER_EVENT_SCHEMA):
        raise ValueError(f"Unexpected post-liker event schema: {event_schema}")
    audit_df = (
        events_lf.group_by("subject_uri")
        .agg(
            pl.len().cast(pl.UInt64).alias("like_event_count"),
            pl.col("like_created_at").min().alias("first_like_created_at"),
            pl.col("like_created_at").max().alias("last_like_created_at"),
            pl.col("liker_did").is_null().sum().cast(pl.UInt64).alias(
                "null_liker_did_count"
            ),
            (
                pl.col("liker_did").is_not_null()
                & (pl.col("liker_did").str.len_chars() == 0)
            ).sum().cast(pl.UInt64).alias("empty_liker_did_count"),
            pl.col("like_created_at").is_null().sum().cast(pl.UInt64).alias(
                "null_timestamp_count"
            ),
        )
        .sort("subject_uri")
        .collect(engine="streaming")
    )
    if audit_df.columns != EVENT_AUDIT_COLUMNS or audit_df.schema != pl.Schema(
        EVENT_AUDIT_SCHEMA
    ):
        raise ValueError(f"Unexpected post-liker event audit schema: {audit_df.schema}")
    return audit_df


def event_stats_from_audit(event_audit_df: pl.DataFrame) -> pl.DataFrame:
    """Project the persisted-event audit to the public per-post statistics."""

    return event_audit_df.select(
        "subject_uri",
        "like_event_count",
        "first_like_created_at",
        "last_like_created_at",
    )


def validate_selected_posts(selected_posts_df: pl.DataFrame) -> None:
    """Validate the locally constructed selected-post role table."""
    if (
        selected_posts_df.columns != SELECTED_POST_COLUMNS
        or selected_posts_df.schema != pl.Schema(SELECTED_POST_SCHEMA)
    ):
        raise ValueError(f"Unexpected selected-post schema: {selected_posts_df.schema}")
    if selected_posts_df.get_column("subject_uri").null_count():
        raise ValueError("Selected posts contain a null subject_uri")
    if selected_posts_df.height != selected_posts_df.unique("subject_uri").height:
        raise ValueError("Selected posts contain duplicate subject_uri values")
    if selected_posts_df.filter(
        ~pl.col("is_positive") & ~pl.col("is_history") & ~pl.col("is_negative")
    ).height:
        raise ValueError("Selected posts contain a row without a post role")
    if not selected_posts_df.equals(selected_posts_df.sort("subject_uri")):
        raise ValueError("Selected posts are not sorted by subject_uri")


def validate_public_partition(
    *,
    event_audit_df: pl.DataFrame,
    post_liker_posts_df: pl.DataFrame,
    selected_posts_df: pl.DataFrame,
    source_start: datetime,
    source_end: datetime,
    partition_id: int,
    partition_count: int,
) -> None:
    """Validate Stage 5 schemas, keys, counts, bounds, and URI routing."""
    validate_selected_posts(selected_posts_df)
    if (
        event_audit_df.columns != EVENT_AUDIT_COLUMNS
        or event_audit_df.schema != pl.Schema(EVENT_AUDIT_SCHEMA)
    ):
        raise ValueError(
            f"Unexpected post-liker event audit schema: {event_audit_df.schema}"
        )
    if (
        post_liker_posts_df.columns != POST_LIKER_POST_COLUMNS
        or post_liker_posts_df.schema != pl.Schema(POST_LIKER_POST_SCHEMA)
    ):
        raise ValueError(
            f"Unexpected post-liker post schema: {post_liker_posts_df.schema}"
        )
    expected_keys = selected_posts_df.select("subject_uri")
    actual_keys = post_liker_posts_df.select("subject_uri")
    if not expected_keys.equals(actual_keys):
        raise ValueError("post_liker_posts keys do not equal the selected post universe")
    if post_liker_posts_df.height != post_liker_posts_df.unique("subject_uri").height:
        raise ValueError("post_liker_posts contains duplicate subject_uri values")
    if not post_liker_posts_df.equals(post_liker_posts_df.sort("subject_uri")):
        raise ValueError("post_liker_posts is not sorted by subject_uri")

    null_subject_uri_count = int(
        event_audit_df.filter(pl.col("subject_uri").is_null()).get_column(
            "like_event_count"
        ).sum()
        or 0
    )
    invalid_event_count = null_subject_uri_count + sum(
        int(event_audit_df.get_column(column).sum() or 0)
        for column in (
            "null_liker_did_count",
            "empty_liker_did_count",
            "null_timestamp_count",
        )
    )
    if invalid_event_count:
        raise ValueError("post_liker_events contains an invalid key or timestamp")
    min_timestamp = event_audit_df.get_column("first_like_created_at").min()
    max_timestamp = event_audit_df.get_column("last_like_created_at").max()
    if min_timestamp is not None and min_timestamp < source_start:
        raise ValueError("post_liker_events contains a timestamp before the source window")
    if max_timestamp is not None and max_timestamp >= source_end:
        raise ValueError("post_liker_events contains a timestamp at or after source_end")

    event_stats_df = event_stats_from_audit(event_audit_df)
    unexpected_events = event_stats_df.select("subject_uri").join(
        expected_keys,
        on="subject_uri",
        how="anti",
    ).height
    if unexpected_events:
        raise ValueError("post_liker_events contains a URI outside post_liker_posts")

    expected_posts_df = build_post_liker_posts(
        selected_posts_df,
        event_stats_df,
    )
    if not expected_posts_df.equals(post_liker_posts_df):
        raise ValueError("post_liker_posts counts or timestamp bounds do not match events")

    for frame_name, frame in (
        ("post_liker_posts", post_liker_posts_df.select("subject_uri")),
        ("post_liker_events", event_stats_df.select("subject_uri")),
    ):
        if frame.is_empty():
            continue
        assigned = (
            frame.with_columns(post_selection.post_partition_expr(partition_count))
            .get_column("_post_partition")
            .unique()
            .to_list()
        )
        if assigned != [partition_id]:
            raise ValueError(
                f"{frame_name} partition {partition_id} contains rows assigned to {assigned}"
            )
