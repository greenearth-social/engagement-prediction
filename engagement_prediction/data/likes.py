"""Reusable normalization and filtering for Ingex like exports."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from engagement_prediction.data import timestamps


LIKE_TIMESTAMP_COLUMN = "record_created_at"


def normalize_likes(likes_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Select the canonical like fields and normalize timestamps to UTC."""
    required = {"did", "subject_uri", LIKE_TIMESTAMP_COLUMN}
    missing = required - set(likes_lf.collect_schema().names())
    if missing:
        raise ValueError(f"Input likes are missing required columns: {', '.join(sorted(missing))}")

    return likes_lf.select(
        pl.col("did").cast(pl.String),
        pl.col("subject_uri").cast(pl.String),
        timestamps.utc_timestamp_expr(
            likes_lf,
            LIKE_TIMESTAMP_COLUMN,
        ).alias("like_created_at"),
    )


def valid_identifier_expr(column: str) -> pl.Expr:
    """Accept only non-null identifiers containing a non-whitespace character."""

    identifier = pl.col(column)
    return identifier.is_not_null() & (
        identifier.str.strip_chars().str.len_chars() > 0
    )


def prepare_likes(
    likes_lf: pl.LazyFrame,
    *,
    start: datetime | None,
    end: datetime | None,
) -> pl.LazyFrame:
    """Normalize likes, remove invalid rows, and apply an inclusive/exclusive window."""
    filtered_lf = normalize_likes(likes_lf).filter(
        valid_identifier_expr("did")
        & valid_identifier_expr("subject_uri")
        & pl.col("like_created_at").is_not_null()
    )
    return filtered_lf.filter(
        timestamps.half_open_window_expr(
            "like_created_at",
            start=start,
            end=end,
        )
    )
