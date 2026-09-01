"""Training-only liker-user vocabulary and compact event schemas.

Stage 5 preserves raw liker DIDs because it is model-independent. Stage 7
turns only the events visible to surviving training features into a bounded
embedding vocabulary. Events from every other valid user remain useful: they
map to the shared UNK row rather than disappearing from the pooled history.
"""

from __future__ import annotations

import polars as pl


POST_LIKER_USER_PAD_IDX = 0
POST_LIKER_USER_UNK_IDX = 1

POST_LIKER_USER_VOCABULARY_COLUMNS = [
    "liker_did",
    "liker_idx",
    "training_event_count",
]
POST_LIKER_USER_VOCABULARY_SCHEMA = {
    "liker_did": pl.String,
    "liker_idx": pl.UInt32,
    "training_event_count": pl.UInt64,
}

POST_LIKER_USE_WINDOW_COLUMNS = [
    "subject_uri",
    "emb_idx",
    "final_use_query_hour",
    "final_training_use_query_hour",
]
POST_LIKER_USE_WINDOW_SCHEMA = {
    "subject_uri": pl.String,
    "emb_idx": pl.UInt32,
    "final_use_query_hour": pl.Datetime("us", "UTC"),
    "final_training_use_query_hour": pl.Datetime("us", "UTC"),
}

POST_LIKER_FEATURE_EVENT_COLUMNS = [
    "emb_idx",
    "liker_did",
    "like_created_at",
    "is_training_visible",
]
POST_LIKER_FEATURE_EVENT_SCHEMA = {
    "emb_idx": pl.UInt32,
    "liker_did": pl.String,
    "like_created_at": pl.Datetime("us", "UTC"),
    "is_training_visible": pl.Boolean,
}

INDEXED_POST_LIKER_EVENT_COLUMNS = [
    "emb_idx",
    "liker_idx",
    "like_created_at",
]
INDEXED_POST_LIKER_EVENT_SCHEMA = {
    "emb_idx": pl.UInt32,
    "liker_idx": pl.UInt32,
    "like_created_at": pl.Datetime("us", "UTC"),
}


def empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Create a typed empty frame for sparse event and vocabulary partitions."""

    return pl.DataFrame(schema=schema)


def support_partition_expr(partition_count: int) -> pl.Expr:
    """Assign each liker DID to one stable aggregation partition."""

    if partition_count <= 0:
        raise ValueError("post-liker user support partition count must be positive")
    return (
        pl.concat_str(
            [pl.lit("post-liker-user-training-support"), pl.col("liker_did")],
            separator="|",
        )
        .hash(seed=0)
        .mod(pl.lit(partition_count, dtype=pl.UInt64))
        .cast(pl.UInt32)
        .alias("_liker_partition")
    )


def add_liker_indices(selected_support_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Assign deterministic dense indices after bounded vocabulary selection."""

    return (
        selected_support_lf.sort("liker_did")
        .with_row_index("liker_idx", offset=2)
        .with_columns(
            pl.col("liker_idx").cast(pl.UInt32),
            pl.col("training_event_count").cast(pl.UInt64),
        )
        .select(POST_LIKER_USER_VOCABULARY_COLUMNS)
    )


def validate_post_liker_user_vocabulary(
    vocabulary_lf: pl.LazyFrame,
    *,
    min_training_event_count: int,
    max_vocabulary_size: int,
) -> dict[str, int]:
    """Validate the public bounded vocabulary and return table statistics."""

    if min_training_event_count < 1:
        raise ValueError("min_post_liker_user_training_event_count must be at least 1")
    if max_vocabulary_size < 0:
        raise ValueError("max_post_liker_user_vocabulary_size may not be negative")
    schema = vocabulary_lf.collect_schema()
    if schema != pl.Schema(POST_LIKER_USER_VOCABULARY_SCHEMA):
        raise ValueError(f"Unexpected post-liker user vocabulary schema: {schema}")
    checks = vocabulary_lf.select(
        pl.len().alias("user_count"),
        pl.col("liker_did").null_count().alias("null_user_count"),
        pl.col("liker_did").n_unique().alias("unique_user_count"),
        pl.col("liker_idx").n_unique().alias("unique_index_count"),
        pl.col("liker_idx").min().alias("min_liker_idx"),
        pl.col("liker_idx").max().alias("max_liker_idx"),
        pl.col("training_event_count").min().alias("min_training_event_count"),
        pl.col("training_event_count").sum().alias("training_event_count"),
    ).collect(engine="streaming").row(0, named=True)
    user_count = int(checks["user_count"])
    if checks["null_user_count"]:
        raise ValueError("Post-liker user vocabulary contains a null DID")
    if int(checks["unique_user_count"]) != user_count:
        raise ValueError("Post-liker user vocabulary contains duplicate DIDs")
    if int(checks["unique_index_count"]) != user_count:
        raise ValueError("Post-liker user vocabulary contains duplicate indices")
    if user_count > max_vocabulary_size:
        raise ValueError("Post-liker user vocabulary exceeds its configured cap")
    if user_count:
        if int(checks["min_liker_idx"]) != 2 or int(checks["max_liker_idx"]) != user_count + 1:
            raise ValueError("Post-liker user indices are not dense from 2")
        if int(checks["min_training_event_count"]) < min_training_event_count:
            raise ValueError("Post-liker user vocabulary contains a user below threshold")
    invalid_order_count = (
        vocabulary_lf.select("liker_did", "liker_idx")
        .sort("liker_did")
        .with_row_index("expected_idx", offset=2)
        .filter(pl.col("liker_idx") != pl.col("expected_idx"))
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if invalid_order_count:
        raise ValueError("Post-liker user indices do not follow ascending DID order")
    return {
        "user_count": user_count,
        "user_table_num_rows": user_count + 2,
        "training_event_count": int(checks["training_event_count"] or 0),
    }
