"""Reusable normalization and filtering for Ingex like exports."""

from __future__ import annotations

from datetime import datetime

import polars as pl


LIKE_TIMESTAMP_COLUMN = "record_created_at"


def like_timestamp_expr(lf: pl.LazyFrame) -> pl.Expr:
    """Normalize the raw like timestamp column to a nullable UTC datetime."""

    schema = lf.collect_schema()
    if LIKE_TIMESTAMP_COLUMN not in schema:
        raise ValueError(f"Input likes are missing required column {LIKE_TIMESTAMP_COLUMN!r}")
    dtype = schema[LIKE_TIMESTAMP_COLUMN]
    timestamp = pl.col(LIKE_TIMESTAMP_COLUMN)
    if dtype == pl.String:
        has_timezone = timestamp.str.contains(r"(Z|[+-]\d{2}:?\d{2})$")
        normalized_text = pl.when(has_timezone).then(timestamp).otherwise(timestamp + pl.lit("Z"))
        return normalized_text.str.to_datetime(
            format="%Y-%m-%dT%H:%M:%S%.f%#z",
            time_zone="UTC",
            strict=False,
        )
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is None:
            return timestamp.dt.replace_time_zone("UTC")
        return timestamp.dt.convert_time_zone("UTC")
    raise ValueError(
        f"{LIKE_TIMESTAMP_COLUMN} must be a string or datetime column, found {dtype}"
    )


def normalize_likes(likes_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Select the canonical like fields and normalize timestamps to UTC."""
    required = {"did", "subject_uri", LIKE_TIMESTAMP_COLUMN}
    missing = required - set(likes_lf.collect_schema().names())
    if missing:
        raise ValueError(f"Input likes are missing required columns: {', '.join(sorted(missing))}")

    return likes_lf.select(
        pl.col("did").cast(pl.String),
        pl.col("subject_uri").cast(pl.String),
        like_timestamp_expr(likes_lf).alias("like_created_at"),
    )


def prepare_likes(
    likes_lf: pl.LazyFrame,
    *,
    start: datetime | None,
    end: datetime | None,
) -> pl.LazyFrame:
    """Normalize likes, remove invalid rows, and apply an inclusive/exclusive window."""
    filtered_lf = normalize_likes(likes_lf).filter(
        pl.col("did").is_not_null()
        & pl.col("subject_uri").is_not_null()
        & pl.col("like_created_at").is_not_null()
    )
    if start is not None:
        filtered_lf = filtered_lf.filter(pl.col("like_created_at") >= pl.lit(start))
    if end is not None:
        filtered_lf = filtered_lf.filter(pl.col("like_created_at") < pl.lit(end))
    return filtered_lf
