"""Training-feature support and the canonical Stage 7 author vocabulary.

Unlike Stage 6's descriptive statistics, vocabulary membership is based only on
author occurrences in the final model-facing training features. This prevents
validation/holdout exposure or globally prolific but unused authors from earning
a dedicated embedding-table row.
"""

from __future__ import annotations

import polars as pl


AUTHOR_SUPPORT_COLUMNS = [
    "author_did",
    "training_feature_count",
    "training_positive_count",
    "training_history_count",
    "training_negative_count",
]
AUTHOR_SUPPORT_SCHEMA = {
    "author_did": pl.String,
    "training_feature_count": pl.UInt64,
    "training_positive_count": pl.UInt64,
    "training_history_count": pl.UInt64,
    "training_negative_count": pl.UInt64,
}
AUTHOR_VOCABULARY_COLUMNS = ["author_did", "author_idx", *AUTHOR_SUPPORT_COLUMNS[1:]]
AUTHOR_VOCABULARY_SCHEMA = {
    "author_did": pl.String,
    "author_idx": pl.UInt32,
    **{column: AUTHOR_SUPPORT_SCHEMA[column] for column in AUTHOR_SUPPORT_COLUMNS[1:]},
}


def empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Create a typed empty frame for sparse support partitions."""

    return pl.DataFrame(schema=schema)


def support_partition_expr(partition_count: int) -> pl.Expr:
    """Assign one author to a deterministic support-aggregation partition."""

    if partition_count <= 0:
        raise ValueError("author_statistics_partition_count must be positive")
    return (
        pl.concat_str(
            [pl.lit("author-training-support"), pl.col("author_did")],
            separator="|",
        )
        .hash(seed=0)
        .mod(pl.lit(partition_count, dtype=pl.UInt64))
        .cast(pl.UInt32)
        .alias("_author_partition")
    )


def aggregate_support_rows(exposure_rows_df: pl.DataFrame) -> pl.DataFrame:
    """Collapse role-tagged model-facing occurrences to one row per author."""

    if exposure_rows_df.is_empty():
        return empty_frame(AUTHOR_SUPPORT_SCHEMA)
    result = (
        exposure_rows_df.group_by("author_did")
        .agg(
            pl.col("training_positive_count").sum().cast(pl.UInt64),
            pl.col("training_history_count").sum().cast(pl.UInt64),
            pl.col("training_negative_count").sum().cast(pl.UInt64),
        )
        .with_columns(
            (
                pl.col("training_positive_count")
                + pl.col("training_history_count")
                + pl.col("training_negative_count")
            )
            .cast(pl.UInt64)
            .alias("training_feature_count")
        )
        .select(AUTHOR_SUPPORT_COLUMNS)
        .sort("author_did")
    )
    validate_support_partition(result)
    return result


def add_author_indices(author_support_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Globally sort eligible authors and reserve 0=PAD and 1=UNK."""

    return (
        author_support_lf.sort("author_did")
        .with_row_index("author_idx", offset=2)
        .with_columns(pl.col("author_idx").cast(pl.UInt32))
        .select(AUTHOR_VOCABULARY_COLUMNS)
    )


def validate_support_partition(author_support_df: pl.DataFrame) -> None:
    """Validate one bounded author-support aggregation partition."""

    if author_support_df.columns != AUTHOR_SUPPORT_COLUMNS or author_support_df.schema != pl.Schema(
        AUTHOR_SUPPORT_SCHEMA
    ):
        raise ValueError(f"Unexpected author-support schema: {author_support_df.schema}")
    if author_support_df.get_column("author_did").null_count():
        raise ValueError("Author support contains a null author_did")
    if author_support_df.height != author_support_df.unique("author_did").height:
        raise ValueError("Author support contains duplicate author_did values")
    if not author_support_df.equals(author_support_df.sort("author_did")):
        raise ValueError("Author support is not sorted by author_did")
    inconsistent = author_support_df.filter(
        pl.col("training_feature_count")
        != (
            pl.col("training_positive_count")
            + pl.col("training_history_count")
            + pl.col("training_negative_count")
        )
    ).height
    if inconsistent:
        raise ValueError("Author support contains inconsistent role counts")


def validate_author_vocabulary(
    authors_lf: pl.LazyFrame,
    *,
    min_training_feature_count: int,
) -> dict[str, int]:
    """Validate the public dense vocabulary and return aggregate counts."""

    schema = authors_lf.collect_schema()
    if schema != pl.Schema(AUTHOR_VOCABULARY_SCHEMA):
        raise ValueError(f"Unexpected author-vocabulary schema: {schema}")
    checks = authors_lf.select(
        pl.len().alias("author_count"),
        pl.col("author_did").null_count().alias("null_author_count"),
        pl.col("author_did").n_unique().alias("unique_author_count"),
        pl.col("author_idx").n_unique().alias("unique_index_count"),
        pl.col("author_idx").min().alias("min_author_idx"),
        pl.col("author_idx").max().alias("max_author_idx"),
        pl.col("training_feature_count").min().alias("min_training_feature_count"),
        pl.col("training_feature_count").sum().alias("training_feature_count"),
        pl.col("training_positive_count").sum().alias("training_positive_count"),
        pl.col("training_history_count").sum().alias("training_history_count"),
        pl.col("training_negative_count").sum().alias("training_negative_count"),
        (
            pl.col("training_feature_count")
            != (
                pl.col("training_positive_count")
                + pl.col("training_history_count")
                + pl.col("training_negative_count")
            )
        )
        .sum()
        .alias("inconsistent_count"),
    ).collect(engine="streaming").row(0, named=True)
    author_count = int(checks["author_count"])
    if checks["null_author_count"]:
        raise ValueError("Author vocabulary contains a null author_did")
    if int(checks["unique_author_count"]) != author_count:
        raise ValueError("Author vocabulary contains duplicate author_did values")
    if int(checks["unique_index_count"]) != author_count:
        raise ValueError("Author vocabulary contains duplicate author_idx values")
    if checks["inconsistent_count"]:
        raise ValueError("Author vocabulary contains inconsistent role counts")
    if author_count:
        if int(checks["min_author_idx"]) != 2 or int(checks["max_author_idx"]) != author_count + 1:
            raise ValueError("Author indices are not dense from 2")
        if int(checks["min_training_feature_count"]) < min_training_feature_count:
            raise ValueError("Author vocabulary contains an author below its support threshold")
    unsorted_count = (
        authors_lf.select("author_did", "author_idx")
        .sort("author_did")
        .with_row_index("expected_idx", offset=2)
        .filter(pl.col("author_idx") != pl.col("expected_idx"))
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if unsorted_count:
        raise ValueError("Author indices do not follow ascending author_did order")
    return {
        "author_count": author_count,
        "author_table_num_rows": author_count + 2,
        "training_feature_count": int(checks["training_feature_count"] or 0),
        "training_positive_count": int(checks["training_positive_count"] or 0),
        "training_history_count": int(checks["training_history_count"] or 0),
        "training_negative_count": int(checks["training_negative_count"] or 0),
    }
