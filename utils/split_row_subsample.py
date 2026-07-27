#!/usr/bin/env python3
"""Deterministic global row subsampling for individual dataset splits."""

from __future__ import annotations

from typing import Optional

import polars as pl


def apply_split_row_target(
    lf: pl.LazyFrame,
    split_name: str,
    target_rows: Optional[int],
    random_seed: int,
    *,
    split_col: str = "split",
    user_col: str = "target_did",
    like_col: str = "like_uri",
) -> pl.LazyFrame:
    """Keep a deterministic uniform row subsample of one named split.

    The selected rows are ordered by a stable hash of the user--like key, so a
    smaller target is a strict subset of a larger target with the same seed.
    Rows in all other splits pass through unchanged. Targets that are absent,
    zero, or at least as large as the selected split are no-ops.

    Args:
        lf: Target-posts LazyFrame.
        split_name: Value in ``split_col`` to subsample (for example ``train``).
        target_rows: Number of rows to retain, or ``None``/non-positive for no-op.
        random_seed: Stable hash seed shared across comparable cells.
        split_col: Split indicator column.
        user_col: User ID column included in the deterministic row key.
        like_col: Like ID column included in the deterministic row key.
    """
    if target_rows is None or target_rows <= 0:
        return lf

    row_key = pl.concat_str([pl.col(user_col), pl.col(like_col)]).hash(seed=random_seed)
    row_rank = row_key.rank("ordinal").over(split_col)
    return (
        lf.with_columns(row_rank.alias("_row_subsample_rank"))
        .filter(
            (pl.col(split_col) != split_name)
            | (pl.col("_row_subsample_rank") <= int(target_rows))
        )
        .drop("_row_subsample_rank")
    )
