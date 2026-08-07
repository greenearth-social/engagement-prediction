"""Helpers for locating and scanning timestamped Ingex Parquet exports."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Optional, Sequence, Tuple

import polars as pl


TIMESTAMP_SUFFIX_PATTERN = r"_(\d{8})_(\d{6})\.parquet$"


def parse_utc_datetime(value: Optional[str], *, field_name: str) -> Optional[datetime]:
    """Parse a CLI timestamp and normalize it to UTC."""
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


def parse_ingex_blob_timestamp(blob_name: str, blob_prefix: str) -> Optional[datetime]:
    """Parse the UTC timestamp from an Ingex export filename."""
    pattern = re.compile(re.escape(blob_prefix) + TIMESTAMP_SUFFIX_PATTERN)
    match = pattern.fullmatch(blob_name)
    if match is None:
        return None
    date_part, time_part = match.groups()
    return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def _list_ingex_parquet_files(
    client: Any,
    *,
    gcs_bucket: str,
    blob_prefix: str,
    start: Optional[datetime],
    end: Optional[datetime],
) -> Tuple[list[str], list[datetime]]:
    matches: list[tuple[datetime, str]] = []
    for blob in client.list_blobs(gcs_bucket):
        timestamp = parse_ingex_blob_timestamp(blob.name, blob_prefix)
        if timestamp is None:
            continue
        if start is not None and timestamp < start:
            continue
        if end is not None and timestamp >= end:
            continue
        matches.append((timestamp, f"gs://{gcs_bucket}/{blob.name}"))
    matches.sort(key=lambda item: (item[0], item[1]))
    return [uri for _, uri in matches], [timestamp for timestamp, _ in matches]


def list_ingex_parquet_files(
    *,
    gcs_bucket: str,
    blob_prefix: str,
    start: Optional[datetime],
    end: Optional[datetime],
) -> Tuple[list[str], list[datetime]]:
    """List matching Ingex Parquet URIs in deterministic timestamp order."""
    from google.cloud import storage

    return _list_ingex_parquet_files(
        storage.Client(),
        gcs_bucket=gcs_bucket,
        blob_prefix=blob_prefix,
        start=start,
        end=end,
    )


def scan_parquet_files(paths: Sequence[str]) -> pl.LazyFrame:
    """Create a lazy Parquet scan for an explicit non-empty path collection."""
    if not paths:
        raise ValueError("Cannot scan an empty collection of Parquet files")
    return pl.scan_parquet(list(paths))
