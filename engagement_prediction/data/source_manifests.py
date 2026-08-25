"""Typed loading and alignment checks for immutable Ingex source snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from engagement_prediction.data import ingex


@dataclass(frozen=True)
class SourceSnapshot:
    """One validated manifest describing the exact files scanned by a stage."""

    path: Path
    manifest: dict[str, Any]
    file_uris: tuple[str, ...]
    start: datetime
    end: datetime
    gcs_bucket: str
    blob_prefix: str


def find_source_manifest(directory: Path, manifest_prefix: str) -> Path:
    """Find exactly one source manifest with ``manifest_prefix`` in a directory."""

    directory = Path(directory)
    matches = sorted(directory.glob(f"{manifest_prefix}*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {manifest_prefix}*.json under {directory}, "
            f"found {len(matches)}"
        )
    return matches[0]


def load_source_snapshot(
    directory: Path,
    *,
    manifest_prefix: str,
    expected_blob_prefix: str,
) -> SourceSnapshot:
    """Locate and validate one non-empty, bounded Ingex source snapshot."""

    manifest_path = find_source_manifest(directory, manifest_prefix)
    manifest = ingex.load_source_manifest(manifest_path)
    blob_prefix = manifest.get("blob_prefix")
    if blob_prefix != expected_blob_prefix:
        raise ValueError(
            f"Source manifest {manifest_path} must use {expected_blob_prefix}, "
            f"found {blob_prefix!r}"
        )
    start = ingex.parse_utc_datetime(manifest.get("start"), field_name="posts_start")
    end = ingex.parse_utc_datetime(manifest.get("end"), field_name="posts_end")
    if start is None or end is None or end <= start:
        raise ValueError(f"Source manifest {manifest_path} has invalid source bounds")
    gcs_bucket = manifest.get("gcs_bucket")
    if not isinstance(gcs_bucket, str) or not gcs_bucket:
        raise ValueError(f"Source manifest {manifest_path} has no GCS bucket")
    return SourceSnapshot(
        path=manifest_path,
        manifest=manifest,
        file_uris=tuple(str(entry["uri"]) for entry in manifest["files"]),
        start=start,
        end=end,
        gcs_bucket=gcs_bucket,
        blob_prefix=blob_prefix,
    )


def validate_aligned_source_snapshots(
    snapshots: Iterable[SourceSnapshot],
    *,
    description: str,
) -> tuple[datetime, datetime]:
    """Require source snapshots to share one bucket and half-open time window."""

    snapshots = tuple(snapshots)
    if not snapshots:
        raise ValueError("At least one source snapshot is required")
    contracts = {
        (snapshot.gcs_bucket, snapshot.start, snapshot.end)
        for snapshot in snapshots
    }
    if len(contracts) != 1:
        raise ValueError(f"{description} do not share one source bucket and window")
    return snapshots[0].start, snapshots[0].end
