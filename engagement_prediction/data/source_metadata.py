"""Canonical schemas and transformations for indexed post/reply metadata.

These functions define Stage 00's logical record contract independently of
artifact I/O. Raw records are first normalized and hash-routed; one physical
partition can then be loaded eagerly to resolve duplicates and root/reply
collisions while total memory remains bounded.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl


UTC_DATETIME = pl.Datetime("us", "UTC")
POST_METADATA_COLUMNS = ["subject_uri", "post_created_at", "author_did", "is_reply"]
POST_METADATA_SCHEMA = {
    "subject_uri": pl.String,
    "post_created_at": UTC_DATETIME,
    "author_did": pl.String,
    "is_reply": pl.Boolean,
}
NORMALIZED_METADATA_SCHEMA = {
    **POST_METADATA_SCHEMA,
    "_post_row_valid": pl.Boolean,
}


def empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Create a typed empty frame for sparse partitions and empty artifacts."""

    return pl.DataFrame(schema=schema)


def uri_partition_expr(partition_count: int) -> pl.Expr:
    """Assign a URI to the stable physical partition shared by metadata consumers.

    Equal URIs always receive the same partition. Consequently, uniqueness
    within every completed partition also proves global URI uniqueness.
    """

    if partition_count <= 0:
        raise ValueError("source_metadata_partition_count must be positive")
    return (
        pl.concat_str([pl.lit("post-selection"), pl.col("subject_uri")], separator="|")
        .hash(seed=0)
        .mod(pl.lit(partition_count, dtype=pl.UInt64))
        .cast(pl.UInt32)
        .alias("_post_partition")
    )


def _utc_timestamp_expr(lf: pl.LazyFrame, column: str) -> pl.Expr:
    """Normalize one string or datetime source column to UTC."""

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
        )
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is None:
            return value.dt.replace_time_zone("UTC")
        return value.dt.convert_time_zone("UTC")
    raise ValueError(f"{column} must be a string or datetime column, found {dtype}")


def normalize_source_records(
    source_lf: pl.LazyFrame,
    *,
    posts_start: datetime,
    posts_end: datetime,
    is_reply: bool,
    passthrough_columns: tuple[str, ...] = (),
) -> pl.LazyFrame:
    """Project raw roots/replies to narrow metadata while retaining validity.

    Invalid rows are marked rather than immediately dropped. Stage 00 routes
    them with the valid rows so partition processing can report complete
    invalid-row statistics before excluding them from the public index.
    Optional passthrough columns are used by later consumers such as Stage 7
    when physical source-file provenance must survive normalization.
    """

    required = {"at_uri", "record_created_at", "did", *passthrough_columns}
    missing = required - set(source_lf.collect_schema().names())
    if missing:
        raise ValueError(
            f"Input source records are missing required columns: {', '.join(sorted(missing))}"
        )
    normalized = source_lf.select(
        pl.col("at_uri").cast(pl.String).alias("subject_uri"),
        _utc_timestamp_expr(source_lf, "record_created_at").alias("post_created_at"),
        pl.col("did").cast(pl.String).alias("author_did"),
        pl.lit(is_reply, dtype=pl.Boolean).alias("is_reply"),
        *(pl.col(column) for column in passthrough_columns),
    )
    return normalized.with_columns(
        (
            pl.col("subject_uri").is_not_null()
            & (pl.col("subject_uri").str.len_chars() > 0)
            & pl.col("post_created_at").is_not_null()
            & (pl.col("post_created_at") >= pl.lit(posts_start))
            & (pl.col("post_created_at") < pl.lit(posts_end))
            & pl.col("author_did").is_not_null()
            & (pl.col("author_did").str.len_chars() > 0)
        ).alias("_post_row_valid")
    )


def select_latest_metadata_rows(
    normalized_df: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Keep latest valid metadata per URI with an ascending-author tie-break.

    This runs independently for roots and replies. Sorting before ``unique``
    makes duplicate selection deterministic even if source-file or Parquet
    row order changes.
    """

    invalid_row_count = normalized_df.filter(~pl.col("_post_row_valid")).height
    valid = normalized_df.filter(pl.col("_post_row_valid"))
    duplicate_counts = valid.group_by("subject_uri").len()
    duplicate_uri_count = duplicate_counts.filter(pl.col("len") > 1).height
    duplicate_row_count = int(
        duplicate_counts.select((pl.col("len") - 1).clip(lower_bound=0).sum()).item()
        or 0
    )
    selected = (
        valid.sort(
            ["subject_uri", "post_created_at", "author_did"],
            descending=[False, True, False],
        )
        .unique(subset="subject_uri", keep="first", maintain_order=True)
        .select(POST_METADATA_COLUMNS)
        .sort("subject_uri")
    )
    return selected, {
        "source_row_count": normalized_df.height,
        "invalid_row_count": invalid_row_count,
        "duplicate_row_count": duplicate_row_count,
        "duplicate_uri_count": duplicate_uri_count,
        "unique_valid_count": selected.height,
    }


def apply_root_precedence(
    root_df: pl.DataFrame,
    reply_df: pl.DataFrame,
) -> tuple[pl.DataFrame, int]:
    """Combine canonical sources, dropping reply rows whose URI is also a root.

    Ingex normally represents roots and replies in disjoint datasets. Root
    precedence makes the index deterministic if a malformed or duplicated URI
    nevertheless occurs in both snapshots.
    """

    root_uris = root_df.select("subject_uri")
    overlap_count = reply_df.join(root_uris, on="subject_uri", how="semi").height
    replies_without_roots = reply_df.join(root_uris, on="subject_uri", how="anti")
    return pl.concat([root_df, replies_without_roots]).sort("subject_uri"), overlap_count


def partition_parquet_paths(
    dataset_path: Path,
    partition_id: int,
    *,
    key: str = "_post_partition",
) -> list[Path]:
    """Return physical Parquet files for one stable URI partition."""

    partition_dir = Path(dataset_path) / f"{key}={partition_id}"
    return sorted(partition_dir.rglob("*.parquet")) if partition_dir.exists() else []


def validate_metadata_partition(
    frame: pl.DataFrame,
    *,
    partition_id: int,
    partition_count: int,
) -> None:
    """Validate schema, uniqueness, ordering, and physical partition assignment."""

    if frame.columns != POST_METADATA_COLUMNS or frame.schema != pl.Schema(
        POST_METADATA_SCHEMA
    ):
        raise ValueError(f"Unexpected source metadata schema: {frame.schema}")
    if frame.get_column("subject_uri").null_count():
        raise ValueError("Source metadata contains a null subject_uri")
    if frame.height != frame.unique(subset="subject_uri").height:
        raise ValueError("Source metadata contains duplicate subject_uri rows")
    if not frame.equals(frame.sort("subject_uri")):
        raise ValueError("Source metadata is not sorted by subject_uri")
    if not frame.is_empty():
        assigned = (
            frame.select("subject_uri")
            .with_columns(uri_partition_expr(partition_count))
            .get_column("_post_partition")
            .unique()
            .to_list()
        )
        if assigned != [partition_id]:
            raise ValueError(
                f"Source metadata partition {partition_id} contains rows assigned to {assigned}"
            )
