"""Shared local-artifact helpers for canonical trained models."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import polars as pl

from engagement_prediction.data.parquet import scan_parquet_artifact


AUTHOR_MAP_SCHEMA = {
    "author_did": pl.String,
    "author_idx": pl.UInt32,
}


def write_json_atomically(path: Path, payload: Any) -> None:
    """Write deterministic portable JSON without exposing a partial final file."""

    path = Path(path)
    partial_path = path.with_name(f"{path.name}.partial")
    try:
        partial_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        partial_path.replace(path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def file_sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a local file."""

    digest = sha256()
    with Path(path).open("rb") as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_author_map(
    *,
    authors_path: Path,
    output_path: Path,
    author_table_num_rows: int,
) -> dict[str, int]:
    """Write and validate the model-independent serving author mapping."""

    if author_table_num_rows < 2:
        raise ValueError("Model author table must reserve PAD=0 and UNK=1")
    authors_lf = scan_parquet_artifact(authors_path)
    source_schema = authors_lf.collect_schema()
    for column, dtype in AUTHOR_MAP_SCHEMA.items():
        if source_schema.get(column) != dtype:
            raise ValueError(
                f"Stage 7 authors artifact must contain {column} with dtype {dtype}"
            )

    author_map_lf = authors_lf.select(list(AUTHOR_MAP_SCHEMA)).sort("author_did")
    validation = author_map_lf.select(
        pl.len().alias("author_count"),
        pl.col("author_did").null_count().alias("null_did_count"),
        pl.col("author_idx").null_count().alias("null_idx_count"),
        pl.col("author_did").n_unique().alias("unique_did_count"),
        pl.col("author_idx").n_unique().alias("unique_idx_count"),
        pl.col("author_idx").min().alias("min_author_idx"),
        pl.col("author_idx").max().alias("max_author_idx"),
    ).collect().row(0, named=True)
    author_count = int(validation["author_count"])
    if validation["null_did_count"] or validation["null_idx_count"]:
        raise ValueError("Stage 7 author vocabulary contains null keys")
    if int(validation["unique_did_count"]) != author_count:
        raise ValueError("Stage 7 author vocabulary contains duplicate author DIDs")
    if int(validation["unique_idx_count"]) != author_count:
        raise ValueError("Stage 7 author vocabulary contains duplicate author indices")
    if author_table_num_rows != author_count + 2:
        raise ValueError(
            "Stage 7 author vocabulary size does not match the model author table"
        )
    if author_count:
        if int(validation["min_author_idx"]) != 2:
            raise ValueError("Stage 7 author vocabulary must begin at index 2")
        if int(validation["max_author_idx"]) != author_table_num_rows - 1:
            raise ValueError("Stage 7 author vocabulary indices must be dense")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    try:
        author_map_lf.sink_parquet(
            partial_path,
            compression="zstd",
            maintain_order=True,
            engine="streaming",
        )
        if pl.read_parquet_schema(partial_path) != pl.Schema(AUTHOR_MAP_SCHEMA):
            raise ValueError("Published author map has an unexpected schema")
        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return {
        "author_count": author_count,
        "author_table_num_rows": author_table_num_rows,
        "file_size_bytes": output_path.stat().st_size,
    }
