"""Shared helpers for evaluation modules.

Underscore-prefixed so ``discover_modules()`` skips this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Inference loading / unnesting
# ---------------------------------------------------------------------------

STRUCT_PREFIX = "message.commit.record.text"


def _load_inferences(run_dir: Path) -> pl.LazyFrame:
    """Locate and scan inferences_core from the 01_get_data stage output."""
    from utils.pipeline.core import select_prior_output
    from utils.helpers import load_parquet_from_prior

    prior = select_prior_output(run_dir, "01_get_data")
    if prior is None:
        raise FileNotFoundError("No 01_get_data output found")
    return load_parquet_from_prior(prior, "inferences_core_")


def _unnest_text_inferences(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Unnest inferences -> text -> message.commit.record.text into top-level group columns.

    Returns the flattened DataFrame and the list of inference group column names
    (only the struct fields from the text-body path, not sibling structs).
    """
    partially = (
        df
        .unnest("inferences")
        .unnest("text")
        .rename({STRUCT_PREFIX: "_text_inf"})
    )
    group_names = [
        f.name for f in partially.schema["_text_inf"].fields
        if isinstance(f.dtype, pl.Struct)
    ]
    return partially.unnest("_text_inf"), group_names


# ---------------------------------------------------------------------------
# User filtering
# ---------------------------------------------------------------------------

MIN_USER_POSTS = 20


def _filter_eligible_users(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only users with >= MIN_USER_POSTS posts and at least 1 pos + 1 neg."""
    eligible = (
        df.group_by("did")
        .agg(pl.len().alias("n"), pl.col("y_true").sum().alias("n_pos"))
        .filter(
            (pl.col("n") >= MIN_USER_POSTS)
            & (pl.col("n_pos") >= 1)
            & (pl.col("n_pos") < pl.col("n"))
        )
        .select("did")
    )
    return df.join(eligible, on="did", how="semi")


# ---------------------------------------------------------------------------
# Volume-split helper
# ---------------------------------------------------------------------------

def _givers_of_half_the_likes(
    user_ids: np.ndarray,
    did_to_likes: Dict[Any, int],
) -> Tuple[Dict[Any, bool], float, str]:
    """Return a mapping uid -> is_high for the 50%-of-volume split."""
    total = sum(did_to_likes.get(u, 0) for u in user_ids)
    if total == 0:
        return {u: False for u in user_ids}, 0.0, "0.0"

    sorted_uids = sorted(user_ids, key=lambda u: did_to_likes.get(u, 0),
                          reverse=True)
    cum = 0
    high_set: set = set()
    for uid in sorted_uids:
        cum += did_to_likes.get(uid, 0)
        high_set.add(uid)
        if cum >= total * 0.5:
            break

    pct_hi = 100.0 * len(high_set) / len(user_ids)
    is_high = {u: u in high_set for u in user_ids}
    return is_high, pct_hi, f"{pct_hi:.1f}"
