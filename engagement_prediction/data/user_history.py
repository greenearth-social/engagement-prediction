"""Query-conditioned user-history selection."""

from __future__ import annotations

from bisect import bisect_left
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


QUERY_KEY = ["did", "query_hour"]
QUERY_COLUMNS = ["did", "query_hour", "user_cohort", "split", "positive_count"]
HISTORY_COLUMNS = [
    "did",
    "query_hour",
    "history_subject_uris",
    "history_like_created_ats",
]
UTC_DATETIME = pl.Datetime("us", "UTC")
HISTORY_SCHEMA = {
    "did": pl.String,
    "query_hour": UTC_DATETIME,
    "history_subject_uris": pl.List(pl.String),
    "history_like_created_ats": pl.List(UTC_DATETIME),
}


def user_partition_expr(partition_count: int) -> pl.Expr:
    if partition_count <= 0:
        raise ValueError("user_history_partition_count must be positive")
    return (
        pl.concat_str([pl.lit("user-history"), pl.col("did")], separator="|")
        .hash(seed=0)
        .mod(pl.lit(partition_count, dtype=pl.UInt64))
        .cast(pl.UInt32)
        .alias("_user_partition")
    )


def validate_queries_schema(queries_lf: pl.LazyFrame) -> None:
    schema = queries_lf.collect_schema()
    if schema.names() != QUERY_COLUMNS:
        raise ValueError(f"Unexpected queries columns: {schema.names()}")
    if schema["did"] != pl.String:
        raise ValueError(f"queries.did must be String, found {schema['did']}")
    query_hour_dtype = schema["query_hour"]
    if not isinstance(query_hour_dtype, pl.Datetime) or query_hour_dtype.time_zone != "UTC":
        raise ValueError(f"queries.query_hour must be a UTC datetime, found {query_hour_dtype}")
    if schema["user_cohort"] != pl.String or schema["split"] != pl.String:
        raise ValueError("queries.user_cohort and queries.split must be String")
    if not schema["positive_count"].is_integer():
        raise ValueError("queries.positive_count must be an integer")


def empty_likes() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "did": pl.String,
            "subject_uri": pl.String,
            "like_created_at": UTC_DATETIME,
        }
    )


def _history_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.from_dicts(rows, schema=HISTORY_SCHEMA)


def build_query_histories_for_partition(
    queries_df: pl.DataFrame,
    likes_df: pl.DataFrame,
    *,
    max_history_posts_per_query: int,
) -> tuple[pl.DataFrame, dict[str, dict[str, int]]]:
    """Build bounded as-of histories for one stable user partition."""
    if max_history_posts_per_query <= 0:
        raise ValueError("max_history_posts_per_query must be positive")

    query_schema = queries_df.lazy().collect_schema()
    if any(column not in query_schema for column in QUERY_COLUMNS):
        raise ValueError(f"Partition queries must contain {QUERY_COLUMNS}")
    like_schema = likes_df.lazy().collect_schema()
    required_like_columns = {"did", "subject_uri", "like_created_at"}
    missing_like_columns = required_like_columns - set(like_schema.names())
    if missing_like_columns:
        raise ValueError(
            f"Partition likes are missing required columns: {', '.join(sorted(missing_like_columns))}"
        )

    sorted_likes = likes_df.select(
        pl.col("did").cast(pl.String),
        pl.col("subject_uri").cast(pl.String),
        pl.col("like_created_at").cast(UTC_DATETIME),
    ).sort(
        ["did", "like_created_at", "subject_uri"],
        descending=[False, False, True],
    )
    likes_by_user: dict[str, list[tuple[datetime, str]]] = {}
    for did, subject_uri, like_created_at in sorted_likes.iter_rows():
        likes_by_user.setdefault(did, []).append((like_created_at, subject_uri))
    like_times_by_user = {
        did: [event[0] for event in user_likes]
        for did, user_likes in likes_by_user.items()
    }

    rows: list[dict[str, Any]] = []
    stats: dict[str, dict[str, int]] = {}
    sorted_queries = queries_df.sort(["did", "query_hour"])
    for query in sorted_queries.iter_rows(named=True):
        did = query["did"]
        query_hour = query["query_hour"]
        split = query["split"]
        user_likes = likes_by_user.get(did, [])
        like_times = like_times_by_user.get(did, [])
        eligible_count = bisect_left(like_times, query_hour)
        selected = user_likes[max(0, eligible_count - max_history_posts_per_query):eligible_count]
        selected.reverse()

        history_subject_uris = [subject_uri for _, subject_uri in selected]
        history_like_created_ats = [like_created_at for like_created_at, _ in selected]
        rows.append({
            "did": did,
            "query_hour": query_hour,
            "history_subject_uris": history_subject_uris,
            "history_like_created_ats": history_like_created_ats,
        })

        split_stats = stats.setdefault(
            split,
            {
                "query_count": 0,
                "unique_user_count": 0,
                "empty_history_count": 0,
                "truncated_history_count": 0,
                "retained_history_item_count": 0,
            },
        )
        split_stats["query_count"] += 1
        split_stats["empty_history_count"] += int(not selected)
        split_stats["truncated_history_count"] += int(
            eligible_count > max_history_posts_per_query
        )
        split_stats["retained_history_item_count"] += len(selected)

    for split in stats:
        stats[split]["unique_user_count"] = sorted_queries.filter(
            pl.col("split") == split
        )["did"].n_unique()

    history_df = _history_frame(rows).sort(["query_hour", "did"])
    validate_partition_artifact(
        queries_df=queries_df,
        history_df=history_df,
        max_history_posts_per_query=max_history_posts_per_query,
    )
    return history_df, stats


def validate_partition_artifact(
    *,
    queries_df: pl.DataFrame,
    history_df: pl.DataFrame,
    max_history_posts_per_query: int,
) -> None:
    if history_df.columns != HISTORY_COLUMNS:
        raise ValueError(f"Unexpected query history columns: {history_df.columns}")
    if history_df.schema != pl.Schema(HISTORY_SCHEMA):
        raise ValueError(f"Unexpected query history schema: {history_df.schema}")
    if history_df.height != queries_df.height:
        raise ValueError("Query history row count does not match the query count")
    if history_df.unique(subset=QUERY_KEY).height != history_df.height:
        raise ValueError("Query histories contain duplicate (did, query_hour) keys")

    expected_keys = queries_df.select(QUERY_KEY).sort(QUERY_KEY)
    actual_keys = history_df.select(QUERY_KEY).sort(QUERY_KEY)
    if not expected_keys.equals(actual_keys):
        raise ValueError("Query history keys do not exactly match the Stage 1 query keys")

    for row in history_df.iter_rows(named=True):
        uris = row["history_subject_uris"]
        timestamps = row["history_like_created_ats"]
        if len(uris) != len(timestamps):
            raise ValueError("Query history URI and timestamp lists are not aligned")
        if len(uris) > max_history_posts_per_query:
            raise ValueError("Query history exceeds max_history_posts_per_query")
        if any(timestamp >= row["query_hour"] for timestamp in timestamps):
            raise ValueError("Query history contains a like from the query hour or future")
        ordered = sorted(zip(timestamps, uris), key=lambda item: (-item[0].timestamp(), item[1]))
        if list(zip(timestamps, uris)) != ordered:
            raise ValueError("Query history is not deterministically ordered by recency")


def merge_partition_stats(
    partition_stats: list[dict[str, dict[str, int]]],
) -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = {}
    for stats in partition_stats:
        for split, values in stats.items():
            target = merged.setdefault(split, {key: 0 for key in values})
            for key, value in values.items():
                target[key] += int(value)
    return dict(sorted(merged.items()))


def partition_parquet_paths(dataset_path: Path, partition_id: int) -> list[Path]:
    partition_dir = Path(dataset_path) / f"_user_partition={partition_id}"
    return sorted(partition_dir.rglob("*.parquet")) if partition_dir.exists() else []
