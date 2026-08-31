"""Helpers for locating and scanning timestamped Ingex Parquet exports."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Optional, Sequence, Tuple

import polars as pl


TIMESTAMP_SUFFIX_PATTERN = r"_(\d{8})_(\d{6})\.parquet$"
SOURCE_MANIFEST_VERSION = 1


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
    """List one export family and apply half-open bounds to filename timestamps.

    The GCS prefix is applied server-side; parsing still rejects unrelated or
    malformed object names. Sorting by timestamp and URI makes source manifests
    reproducible even if GCS returns objects in a different order.
    """

    matches: list[tuple[datetime, str]] = []
    for blob in client.list_blobs(gcs_bucket, prefix=f"{blob_prefix}_"):
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


def scan_parquet_files(
    paths: Sequence[str],
    *,
    include_file_paths: str | None = None,
) -> pl.LazyFrame:
    """Create a lazy Parquet scan for an explicit non-empty path collection."""
    if not paths:
        raise ValueError("Cannot scan an empty collection of Parquet files")
    return pl.scan_parquet(
        list(paths),
        include_file_paths=include_file_paths,
    )


def build_source_manifest(
    *,
    gcs_bucket: str,
    blob_prefix: str,
    start: Optional[datetime],
    end: Optional[datetime],
    paths: Sequence[str],
    timestamps: Sequence[datetime],
) -> dict[str, Any]:
    """Describe the exact Ingex files used by a stage.

    Downstream stages rescan these recorded URIs instead of relisting GCS. This
    freezes the raw-data snapshot and prevents late-arriving exports from changing
    the contents of a rerun partway through the lineage.
    """
    if len(paths) != len(timestamps):
        raise ValueError("Ingex source paths and timestamps must have the same length")
    return {
        "version": SOURCE_MANIFEST_VERSION,
        "gcs_bucket": str(gcs_bucket),
        "blob_prefix": str(blob_prefix),
        "start": start.isoformat() if start is not None else None,
        "end": end.isoformat() if end is not None else None,
        "files": [
            {"uri": str(uri), "export_timestamp": timestamp.isoformat()}
            for uri, timestamp in zip(paths, timestamps)
        ],
    }


def write_source_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a source snapshot in stable, human-readable JSON form."""

    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_source_manifest(path: Path) -> dict[str, Any]:
    """Load and validate an Ingex source manifest."""
    path = Path(path)
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse Ingex source manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Ingex source manifest {path} must contain a JSON object")
    if manifest.get("version") != SOURCE_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported Ingex source manifest version in {path}: {manifest.get('version')!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Ingex source manifest {path} must contain a non-empty files list")
    for entry in files:
        if not isinstance(entry, dict) or not entry.get("uri") or not entry.get("export_timestamp"):
            raise ValueError(f"Ingex source manifest {path} contains an invalid file entry")
    return manifest
