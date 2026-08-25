"""Schemas and bounded transformations for the hydrated training dataset.

This module owns logical row semantics: embedding selection, author-index
fallback, strict as-of like counts, and public schemas. Disk-backed shuffles,
partition iteration, and artifact publication live in
``dataset_hydration_artifacts``.
"""

from __future__ import annotations

from bisect import bisect_left
import binascii
from datetime import datetime
import json
from typing import Any
import zlib

import numpy as np
import polars as pl

from engagement_prediction.data import post_selection
from shared.input_data_helpers import get_expanded_embedding_vector


UTC_DATETIME = pl.Datetime("us", "UTC")
AUTHOR_PAD_IDX = 0
AUTHOR_UNK_IDX = 1

# Public post rows align URI metadata and role flags with ``embeddings.npy``.
# ``emb_idx`` is dense from zero and addresses one row of that memmap.
POST_COLUMNS = [
    "subject_uri",
    "emb_idx",
    "post_created_at",
    "author_did",
    "author_idx",
    "is_reply",
    "is_positive",
    "is_history",
    "is_negative",
]
POST_SCHEMA = {
    "subject_uri": pl.String,
    "emb_idx": pl.UInt32,
    "post_created_at": UTC_DATETIME,
    "author_did": pl.String,
    "author_idx": pl.UInt32,
    "is_reply": pl.Boolean,
    "is_positive": pl.Boolean,
    "is_history": pl.Boolean,
    "is_negative": pl.Boolean,
}
HYDRATED_POST_METADATA_COLUMNS = [
    "subject_uri",
    "emb_idx",
    "post_created_at",
    "author_did",
    "is_reply",
    "is_positive",
    "is_history",
    "is_negative",
]
HYDRATED_POST_METADATA_SCHEMA = {
    column: dtype
    for column, dtype in POST_SCHEMA.items()
    if column != "author_idx"
}

# Query artifacts repeat the Stage 1 keys but contain only uses that survived
# content-embedding hydration.
QUERY_COLUMNS = ["did", "query_hour", "user_cohort", "split", "positive_count"]
QUERY_SCHEMA = {
    "did": pl.String,
    "query_hour": UTC_DATETIME,
    "user_cohort": pl.String,
    "split": pl.String,
    "positive_count": pl.UInt32,
}

QUERY_POSITIVE_COLUMNS = [
    "did",
    "query_hour",
    "subject_uri",
    "like_created_at",
    "emb_idx",
    "post_created_at",
    "author_idx",
    "prior_like_count",
]
QUERY_POSITIVE_SCHEMA = {
    "did": pl.String,
    "query_hour": UTC_DATETIME,
    "subject_uri": pl.String,
    "like_created_at": UTC_DATETIME,
    "emb_idx": pl.UInt32,
    "post_created_at": UTC_DATETIME,
    "author_idx": pl.UInt32,
    "prior_like_count": pl.UInt64,
}

QUERY_HISTORY_COLUMNS = [
    "did",
    "query_hour",
    "history_subject_uris",
    "history_like_created_ats",
    "history_emb_indices",
    "history_author_indices",
    "history_prior_like_counts",
]
QUERY_HISTORY_SCHEMA = {
    "did": pl.String,
    "query_hour": UTC_DATETIME,
    "history_subject_uris": pl.List(pl.String),
    "history_like_created_ats": pl.List(UTC_DATETIME),
    "history_emb_indices": pl.List(pl.UInt32),
    "history_author_indices": pl.List(pl.UInt32),
    "history_prior_like_counts": pl.List(pl.UInt64),
}

# Negative pools remain shared by query hour. User-specific positive overlap is
# resolved by the dataloader when it builds a batch label matrix.
HOURLY_NEGATIVE_COLUMNS = [
    "query_hour",
    "subject_uri",
    "selection_source",
    "emb_idx",
    "post_created_at",
    "author_idx",
    "prior_like_count",
]
HOURLY_NEGATIVE_SCHEMA = {
    "query_hour": UTC_DATETIME,
    "subject_uri": pl.String,
    "selection_source": pl.String,
    "emb_idx": pl.UInt32,
    "post_created_at": UTC_DATETIME,
    "author_idx": pl.UInt32,
    "prior_like_count": pl.UInt64,
}

POST_HOUR_COUNT_COLUMNS = ["subject_uri", "query_hour", "prior_like_count"]
POST_HOUR_COUNT_SCHEMA = {
    "subject_uri": pl.String,
    "query_hour": UTC_DATETIME,
    "prior_like_count": pl.UInt64,
}

VALID_EMBEDDING_METADATA_SCHEMA = {
    "subject_uri": pl.String,
    "source_post_created_at": UTC_DATETIME,
    "source_author_did": pl.String,
}
VALID_EMBEDDING_KEY_SCHEMA = {"subject_uri": pl.String}
VALID_EMBEDDING_SCHEMA = {
    **VALID_EMBEDDING_METADATA_SCHEMA,
    "_emb_vec": pl.List(pl.Float32),
}


def empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Create a typed empty frame for sparse hydration partitions."""

    return pl.DataFrame(schema=schema)


def normalize_embedding_source_rows(
    source_lf: pl.LazyFrame,
    *,
    posts_start: datetime,
    posts_end: datetime,
    is_reply: bool,
) -> pl.LazyFrame:
    """Normalize raw metadata plus encoded embeddings for Phase 3's second pass.

    This relation still contains the Ingex payload representation and may have
    several rows per URI. Decoding, validation, and deterministic duplicate
    selection happen later, one URI partition at a time.
    """
    required = {"at_uri", "record_created_at", "did", "embeddings"}
    missing = required - set(source_lf.collect_schema().names())
    if missing:
        raise ValueError(
            "Embedding source rows are missing columns: " + ", ".join(sorted(missing))
        )
    normalized = source_lf.select(
        pl.col("at_uri").cast(pl.String).alias("subject_uri"),
        post_selection._utc_timestamp_expr(  # canonical Stage 3 timestamp handling
            source_lf,
            "record_created_at",
        ).alias("post_created_at"),
        pl.col("did").cast(pl.String).alias("author_did"),
        pl.lit(is_reply, dtype=pl.Boolean).alias("is_reply"),
        pl.col("embeddings"),
    )
    return normalized.filter(
        pl.col("subject_uri").is_not_null()
        & (pl.col("subject_uri").str.len_chars() > 0)
        & pl.col("post_created_at").is_not_null()
        & (pl.col("post_created_at") >= pl.lit(posts_start))
        & (pl.col("post_created_at") < pl.lit(posts_end))
        & pl.col("author_did").is_not_null()
        & (pl.col("author_did").str.len_chars() > 0)
    )


def normalize_embedding_source_keys(
    source_lf: pl.LazyFrame,
    *,
    posts_start: datetime,
    posts_end: datetime,
    is_reply: bool,
    source_path_column: str | None = None,
) -> pl.LazyFrame:
    """Normalize source keys without reading the embedding payload column.

    ``source_path_column`` retains Polars' file-provenance column so a batch
    scan can identify which physical files need a second, payload-bearing
    pass. The root/reply flag remains part of the shared normalization call,
    even though the narrow output needs only URI and optional provenance.
    """

    normalized_lf = post_selection.normalize_posts(
        source_lf,
        posts_start=posts_start,
        posts_end=posts_end,
        is_reply=is_reply,
        passthrough_columns=(source_path_column,) if source_path_column else (),
    ).filter(pl.col("_post_row_valid"))
    if source_path_column is None:
        return normalized_lf.select("subject_uri")
    return normalized_lf.select("subject_uri", source_path_column)


def _stable_payload_key(value: Any) -> str:
    """Serialize an embedding payload for deterministic final tie-breaking."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def select_latest_valid_embedding_payloads(
    source_rows_df: pl.DataFrame,
    *,
    embedding_model: str,
    embedding_dim: int,
) -> tuple[list[tuple[str, datetime, str, Any]], dict[str, int]]:
    """Select compact payloads for the latest usable vector for every URI.

    This operates on one URI-hash partition. A later row lacking a usable
    vector does not erase an older valid vector. Equal timestamps prefer the
    ascending author DID and then a stable serialized payload ordering. The
    decoded vector is discarded after validation so a large partition does not
    retain Python float arrays for every selected post.
    """
    required = {"subject_uri", "post_created_at", "author_did", "embeddings"}
    missing = required - set(source_rows_df.columns)
    if missing:
        raise ValueError(
            "Embedding source rows are missing columns: " + ", ".join(sorted(missing))
        )

    best: dict[str, tuple[datetime, str, str, Any]] = {}
    stats = {
        "source_row_count": source_rows_df.height,
        "null_embedding_count": 0,
        "invalid_embedding_count": 0,
        "wrong_dimension_count": 0,
        "non_finite_embedding_count": 0,
        "valid_embedding_row_count": 0,
        "duplicate_valid_embedding_row_count": 0,
    }
    for row in source_rows_df.iter_rows(named=True):
        uri = row["subject_uri"]
        created_at = row["post_created_at"]
        author_did = row["author_did"]
        if not uri or created_at is None or not author_did:
            stats["invalid_embedding_count"] += 1
            continue
        try:
            vector = get_expanded_embedding_vector(row["embeddings"], embedding_model)
        except (TypeError, ValueError, RuntimeError, binascii.Error, zlib.error):
            stats["invalid_embedding_count"] += 1
            continue
        if vector is None:
            stats["null_embedding_count"] += 1
            continue
        if len(vector) != embedding_dim:
            stats["wrong_dimension_count"] += 1
            continue
        array = np.asarray(vector, dtype=np.float32)
        if not np.isfinite(array).all():
            stats["non_finite_embedding_count"] += 1
            continue
        stats["valid_embedding_row_count"] += 1
        payload_key = _stable_payload_key(row["embeddings"])
        existing = best.get(uri)
        replace = existing is None
        if existing is not None:
            stats["duplicate_valid_embedding_row_count"] += 1
            existing_created_at, existing_author, existing_payload, _ = existing
            replace = (
                created_at > existing_created_at
                or (
                    created_at == existing_created_at
                    and (author_did, payload_key) < (existing_author, existing_payload)
                )
            )
        if replace:
            best[str(uri)] = (
                created_at,
                str(author_did),
                payload_key,
                row["embeddings"],
            )

    selected = [
        (uri, values[0], values[1], values[3])
        for uri, values in sorted(best.items())
    ]
    stats["unique_valid_embedding_count"] = len(selected)
    return selected, stats


def select_latest_valid_embeddings(
    source_rows_df: pl.DataFrame,
    *,
    embedding_model: str,
    embedding_dim: int,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Return selected vectors as a DataFrame for bounded transformations/tests."""
    selected, stats = select_latest_valid_embedding_payloads(
        source_rows_df,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )

    rows = [
        {
            "subject_uri": uri,
            "source_post_created_at": created_at,
            "source_author_did": author_did,
            "_emb_vec": np.asarray(
                get_expanded_embedding_vector(payload, embedding_model),
                dtype=np.float32,
            ).tolist(),
        }
        for uri, created_at, author_did, payload in selected
    ]
    result = (
        pl.from_dicts(rows, schema=VALID_EMBEDDING_SCHEMA)
        if rows
        else empty_frame(VALID_EMBEDDING_SCHEMA)
    )
    return result, stats


def build_hydrated_post_metadata(
    selected_metadata_df: pl.DataFrame,
    embedding_indices_df: pl.DataFrame,
) -> pl.DataFrame:
    """Attach dense embedding indices while retaining raw author DIDs."""
    result = (
        selected_metadata_df.select(
            "subject_uri",
            "post_created_at",
            "author_did",
            "is_reply",
            "is_positive",
            "is_history",
            "is_negative",
        )
        .join(
            embedding_indices_df.select("subject_uri", "emb_idx"),
            on="subject_uri",
            how="inner",
        )
        .with_columns(pl.col("emb_idx").cast(pl.UInt32))
        .select(HYDRATED_POST_METADATA_COLUMNS)
        .sort("subject_uri")
    )
    validate_frame(result, HYDRATED_POST_METADATA_SCHEMA, key=["subject_uri"])
    return result


def attach_post_author_indices(
    hydrated_post_metadata_df: pl.DataFrame,
    authors_df: pl.DataFrame,
) -> pl.DataFrame:
    """Map Stage 7 vocabulary authors and send every other author to UNK."""

    result = (
        hydrated_post_metadata_df.join(
            authors_df.select("author_did", "author_idx"),
            on="author_did",
            how="left",
        )
        .with_columns(
            pl.col("author_idx").fill_null(AUTHOR_UNK_IDX).cast(pl.UInt32)
        )
        .select(POST_COLUMNS)
        .sort("subject_uri")
    )
    validate_frame(result, POST_SCHEMA, key=["subject_uri"])
    return result


def calculate_prior_like_counts(
    post_hours_df: pl.DataFrame,
    events_df: pl.DataFrame,
) -> pl.DataFrame:
    """Count raw like rows strictly before each used post/query-hour pair.

    Stage 5 deliberately preserves duplicate events, so the sorted timestamp
    lists also preserve them. ``bisect_left`` enforces the strict boundary:
    likes exactly at ``query_hour`` are excluded.
    """
    if post_hours_df.is_empty():
        return empty_frame(POST_HOUR_COUNT_SCHEMA)
    events_by_uri: dict[str, list[datetime]] = {}
    for uri, created_at in events_df.select(
        "subject_uri", "like_created_at"
    ).sort(["subject_uri", "like_created_at"]).iter_rows():
        events_by_uri.setdefault(str(uri), []).append(created_at)
    rows = []
    for uri, query_hour in post_hours_df.select(
        "subject_uri", "query_hour"
    ).unique().sort(["subject_uri", "query_hour"]).iter_rows():
        count = bisect_left(events_by_uri.get(str(uri), []), query_hour)
        rows.append({
            "subject_uri": uri,
            "query_hour": query_hour,
            "prior_like_count": count,
        })
    return pl.from_dicts(rows, schema=POST_HOUR_COUNT_SCHEMA)


def validate_frame(
    df: pl.DataFrame,
    schema: dict[str, pl.DataType],
    *,
    key: list[str] | None,
) -> None:
    """Validate an exact schema, non-null key, and partition-local uniqueness."""

    if df.columns != list(schema) or df.schema != pl.Schema(schema):
        raise ValueError(f"Unexpected hydrated dataset schema: {df.schema}")
    if key:
        if any(df.get_column(column).null_count() for column in key):
            raise ValueError(f"Hydrated dataset has null key values in {key}")
        if df.height != df.unique(key).height:
            raise ValueError(f"Hydrated dataset has duplicate keys: {key}")


def validate_query_histories(df: pl.DataFrame) -> None:
    """Validate aligned Stage 7 history lists, ordering, and as-of timing."""

    validate_frame(df, QUERY_HISTORY_SCHEMA, key=["did", "query_hour"])
    for row in df.iter_rows(named=True):
        lengths = {
            len(row[column])
            for column in QUERY_HISTORY_COLUMNS[2:]
        }
        if len(lengths) != 1:
            raise ValueError("Hydrated query-history lists are not aligned")
        if row["history_like_created_ats"] != sorted(
            row["history_like_created_ats"], reverse=True
        ):
            raise ValueError("Hydrated query histories are not reverse chronological")
        if any(value >= row["query_hour"] for value in row["history_like_created_ats"]):
            raise ValueError("Hydrated query history contains a same-hour or future like")
