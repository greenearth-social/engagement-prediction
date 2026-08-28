"""Canonical UTC timestamp normalization and source-window validation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import polars as pl


UTC_DATETIME = pl.Datetime("us", "UTC")


def parse_utc_datetime(value: Any | None, *, field_name: str) -> datetime | None:
    """Parse a scalar timestamp, treating timezone-naive values as UTC."""

    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        raise ValueError(f"{field_name} must not be empty")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name} timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def utc_timestamp_expr(lf: pl.LazyFrame, column: str) -> pl.Expr:
    """Normalize one string or datetime column to microsecond-resolution UTC."""

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
        ).cast(UTC_DATETIME)
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is None:
            normalized = value.dt.replace_time_zone("UTC")
        else:
            normalized = value.dt.convert_time_zone("UTC")
        return normalized.cast(UTC_DATETIME)
    raise ValueError(f"{column} must be a string or datetime column, found {dtype}")


def half_open_window_expr(
    column: str,
    *,
    start: datetime | None,
    end: datetime | None,
) -> pl.Expr:
    """Return the canonical ``start <= column < end`` filter expression."""

    within_window = pl.lit(True)
    if start is not None:
        within_window &= pl.col(column) >= pl.lit(start)
    if end is not None:
        within_window &= pl.col(column) < pl.lit(end)
    return within_window


def validate_utc_hour_aligned(
    value: datetime | None,
    *,
    field_name: str,
) -> None:
    """Validate that an optional boundary is timezone-aware UTC on an exact hour."""

    if value is None:
        return
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    if value.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field_name} must use UTC")
    if value.minute or value.second or value.microsecond:
        raise ValueError(f"{field_name} must be aligned to the start of an hour")


def validate_half_open_utc_window(
    *,
    start: datetime,
    end: datetime,
    start_field_name: str,
    end_field_name: str,
) -> None:
    """Validate an ordered, hour-aligned UTC source window."""

    validate_utc_hour_aligned(start, field_name=start_field_name)
    validate_utc_hour_aligned(end, field_name=end_field_name)
    if end <= start:
        raise ValueError(f"{end_field_name} must be after {start_field_name}")
