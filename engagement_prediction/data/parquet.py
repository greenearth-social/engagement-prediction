"""Canonical helpers for locating and scanning pipeline Parquet artifacts."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def find_artifact_path(prior_path: Path, prefix: str) -> Path:
    """Find the newest matching Parquet file or partitioned dataset directory."""
    prior_path = Path(prior_path)
    candidates = [
        path
        for path in prior_path.glob(f"{prefix}*")
        if not path.name.endswith(".partial")
        and not path.name.endswith(".partial.parquet")
        and (path.is_dir() or path.suffix == ".parquet")
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No {prefix}*.parquet file or partitioned dataset found under {prior_path}")
    return candidates[0]


def scan_parquet_artifact(path: Path) -> pl.LazyFrame:
    """Scan a single Parquet file or all parts in a partitioned dataset."""
    path = Path(path)
    if path.is_file():
        return pl.scan_parquet(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Parquet artifact does not exist: {path}")
    parts = sorted(path.rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No Parquet parts found under partitioned dataset {path}")
    return pl.scan_parquet(parts)


def load_parquet_from_prior(prior_path: Path, prefix: str) -> pl.LazyFrame:
    """Locate and lazily scan a Parquet artifact from a prior stage directory."""
    return scan_parquet_artifact(find_artifact_path(prior_path, prefix))


def sink_partitioned_parquet(
    lf: pl.LazyFrame,
    *,
    output_path: Path,
    key: str,
) -> None:
    """Stream a lazy frame into Parquet partitions using an existing key column."""
    output_path.mkdir(parents=True, exist_ok=False)
    lf.sink_parquet(
        pl.PartitionBy(
            output_path,
            key=key,
            include_key=False,
            approximate_bytes_per_file="auto",
        ),
        compression="zstd",
        maintain_order=False,
        engine="streaming",
    )


def read_parquet_parts(
    paths: list[Path],
    *,
    empty: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Read bounded Parquet parts, optionally returning a schema-correct empty frame."""
    if paths:
        return pl.read_parquet(paths)
    if empty is None:
        raise ValueError("Expected at least one Parquet part")
    return empty
