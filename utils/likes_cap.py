#!/usr/bin/env python3

"""
Shared helper for applying a per-user random cap to a likes table.

Used by both:
- Stage 1 (get_data): the original ``--max-likes-per-user`` ingestion-time cap.
- Stage 2 (target_posts) and Stage 3 (user_history): the
  ``--effective-likes-cap`` cap applied at training-prep time, so cap sweeps can
  reuse a single Stage 1 ingestion.

The cap is deterministic and idempotent: with the same seed, the result of
``cap=N`` is a strict subset of ``cap=M`` for ``N < M``.  This nesting property
makes cap sweeps (e.g. 200 → 100 → 50 → 20 → 10 → 5) interpretable, since each
tighter cap is sampling from the same hash-rank ordering.
"""

from __future__ import annotations

from typing import Optional

import polars as pl


def apply_per_user_random_cap(
    likes_lf: pl.LazyFrame,
    max_likes_per_user: Optional[int],
    random_seed: int,
    *,
    user_col: str = "did",
    post_col: str = "subject_uri",
) -> pl.LazyFrame:
    """Apply a per-user random cap to ``likes_lf`` in a deterministic way.

    For each user, the (user, post) pairs are hashed with the given seed and
    ranked.  Pairs with rank above ``max_likes_per_user`` are dropped.  Hashing
    on the ``(user, post)`` concatenation (rather than ``post`` alone) gives
    each user an independent random ordering, avoiding correlated subsets
    across users.

    Args:
        likes_lf: A polars LazyFrame containing at least ``user_col`` and
            ``post_col`` columns.
        max_likes_per_user: Maximum number of likes to retain per user.  If
            ``None`` or non-positive, the input is returned unchanged.
        random_seed: Seed for the hash; reuse the same seed across cells in a
            sweep to ensure that tighter caps are nested subsets of looser
            ones.
        user_col: Name of the user column (default ``"did"``).
        post_col: Name of the post column (default ``"subject_uri"``).

    Returns:
        A LazyFrame with the cap applied; the schema is unchanged.
    """
    if max_likes_per_user is None or max_likes_per_user <= 0:
        return likes_lf
    return (
        likes_lf
        .with_columns(
            pl.concat_str([pl.col(user_col), pl.col(post_col)])
              .hash(seed=random_seed)
              .alias("_rand_key")
        )
        .with_columns(
            pl.col("_rand_key").rank("ordinal").over(user_col).alias("_rand_order")
        )
        .filter(pl.col("_rand_order") <= max_likes_per_user)
        .drop(["_rand_key", "_rand_order"])
    )
