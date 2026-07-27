"""Tests for deterministic global target-post row subsampling."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.split_row_subsample import apply_split_row_target


def _build_target_posts() -> pl.DataFrame:
    rows = []
    for split, count in [("train", 30), ("val", 12), ("holdout_unseen_users", 8)]:
        for i in range(count):
            rows.append({
                "split": split,
                "target_did": f"{split}_user_{i % 7}",
                "like_uri": f"at://{split}/{i}",
                "payload": i,
            })
    return pl.DataFrame(rows)


def _keys(frame: pl.DataFrame, split: str) -> set[tuple[str, str]]:
    selected = frame.filter(pl.col("split") == split)
    return set(zip(selected["target_did"].to_list(), selected["like_uri"].to_list()))


def test_none_zero_or_negative_target_is_passthrough():
    source = _build_target_posts()
    for target in (None, 0, -1):
        out = apply_split_row_target(source.lazy(), "train", target, 42).collect()
        assert out.sort("like_uri").equals(source.sort("like_uri"))


def test_target_exactly_limits_only_named_split():
    source = _build_target_posts()
    out = apply_split_row_target(source.lazy(), "train", 11, 42).collect()
    assert out.filter(pl.col("split") == "train").height == 11
    assert out.filter(pl.col("split") == "val").height == 12
    assert out.filter(pl.col("split") == "holdout_unseen_users").height == 8


def test_same_seed_is_deterministic_and_different_seed_changes_selection():
    source = _build_target_posts()
    first = apply_split_row_target(source.lazy(), "train", 10, 42).collect()
    second = apply_split_row_target(source.lazy(), "train", 10, 42).collect()
    other_seed = apply_split_row_target(source.lazy(), "train", 10, 99).collect()
    assert first.sort("like_uri").equals(second.sort("like_uri"))
    assert _keys(first, "train") != _keys(other_seed, "train")


def test_smaller_target_is_nested_within_larger_target():
    source = _build_target_posts()
    large = apply_split_row_target(source.lazy(), "train", 20, 42).collect()
    small = apply_split_row_target(source.lazy(), "train", 9, 42).collect()
    assert _keys(small, "train").issubset(_keys(large, "train"))


def test_target_at_or_above_actual_is_noop_and_holdout_is_untouched():
    source = _build_target_posts()
    out = apply_split_row_target(source.lazy(), "train", 30, 42).collect()
    above = apply_split_row_target(source.lazy(), "train", 100, 42).collect()
    assert out.sort("like_uri").equals(source.sort("like_uri"))
    assert above.sort("like_uri").equals(source.sort("like_uri"))
    assert _keys(out, "holdout_unseen_users") == _keys(source, "holdout_unseen_users")
