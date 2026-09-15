"""Daily raw-row coverage diagnostics for the exact ingestion snapshots."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Any

import polars as pl

from engagement_prediction.data import timestamps


def scan_staged_timestamps(path: Path, *, timestamp_column: str) -> pl.LazyFrame:
    """Read every routed part, including null-key partitions and empty routes."""

    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Staged source directory does not exist: {path}")
    parts = sorted(path.rglob("*.parquet"))
    if not parts:
        return pl.DataFrame(schema={timestamp_column: timestamps.UTC_DATETIME}).lazy()
    return pl.scan_parquet(parts).select(timestamp_column)


def count_daily_rows(
    source_lf: pl.LazyFrame,
    *,
    timestamp_column: str,
    posts_start: datetime,
    posts_end: datetime,
) -> dict[str, Any]:
    """Count raw rows by UTC creation day without collecting the source rows.

    Only timestamps enter the scan. Excluded rows share three fixed buckets,
    keeping the aggregation bounded by the configured number of days even if
    malformed source data contains creation dates far outside the window.
    """

    timestamps.validate_half_open_utc_window(
        start=posts_start,
        end=posts_end,
        start_field_name="posts_start",
        end_field_name="posts_end",
    )
    normalized = source_lf.select(
        timestamps.utc_timestamp_expr(source_lf, timestamp_column).alias("_created_at")
    )
    created_at = pl.col("_created_at")
    within_window = timestamps.half_open_window_expr(
        "_created_at", start=posts_start, end=posts_end
    )
    buckets = (
        normalized.select(
            pl.when(created_at.is_null())
            .then(pl.lit("invalid_timestamp"))
            .when(created_at < pl.lit(posts_start))
            .then(pl.lit("before_window"))
            .when(created_at >= pl.lit(posts_end))
            .then(pl.lit("at_or_after_end"))
            .otherwise(pl.lit("in_window"))
            .alias("category"),
            pl.when(within_window)
            .then(created_at.dt.date())
            .otherwise(None)
            .alias("date"),
        )
        .group_by("category", "date")
        .agg(
            pl.col("category").is_not_null().cast(pl.UInt64).sum().alias("row_count")
        )
        .collect(engine="streaming")
    )
    daily_counts: dict[str, int] = {}
    category_counts = {
        "invalid_timestamp": 0,
        "before_window": 0,
        "at_or_after_end": 0,
        "in_window": 0,
    }
    for row in buckets.iter_rows(named=True):
        count = int(row["row_count"])
        category_counts[row["category"]] += count
        if row["category"] == "in_window":
            daily_counts[row["date"].isoformat()] = count

    dates: list[str] = []
    partial_dates: list[str] = []
    day_start = posts_start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day_start < posts_end:
        day_end = day_start + timedelta(days=1)
        date = day_start.date().isoformat()
        dates.append(date)
        if day_start < posts_start or day_end > posts_end:
            partial_dates.append(date)
        day_start = day_end

    return {
        "posts_start": posts_start.isoformat(),
        "posts_end": posts_end.isoformat(),
        "dates": dates,
        "counts": [daily_counts.get(date, 0) for date in dates],
        "partial_dates": partial_dates,
        "total_row_count": sum(category_counts.values()),
        "in_window_row_count": category_counts["in_window"],
        "invalid_timestamp_count": category_counts["invalid_timestamp"],
        "before_window_row_count": category_counts["before_window"],
        "at_or_after_end_row_count": category_counts["at_or_after_end"],
    }


def daily_count_labels(diagnostics: dict[str, Any]) -> list[str]:
    """Label intentionally partial boundary days in the coverage charts."""

    partial_dates = set(diagnostics["partial_dates"])
    return [
        f"{date} (partial)" if date in partial_dates else date
        for date in diagnostics["dates"]
    ]


def log_daily_counts(
    logger: logging.Logger,
    *,
    source_name: str,
    diagnostics: dict[str, Any],
) -> None:
    """Log counters that reconcile plotted rows with all raw source rows."""

    logger.info(
        "Raw %s daily coverage for [%s, %s): total_rows=%s in_window_rows=%s "
        "invalid_timestamp_rows=%s before_window_rows=%s at_or_after_end_rows=%s",
        source_name,
        diagnostics["posts_start"],
        diagnostics["posts_end"],
        diagnostics["total_row_count"],
        diagnostics["in_window_row_count"],
        diagnostics["invalid_timestamp_count"],
        diagnostics["before_window_row_count"],
        diagnostics["at_or_after_end_row_count"],
    )
