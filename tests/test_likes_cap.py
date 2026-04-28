"""Tests for the shared ``apply_per_user_random_cap`` helper.

The key invariant we rely on for cap sweeps is *nesting*: with the same seed,
the result of cap=N must be a strict subset of cap=M for N < M.  This makes
each tighter cap level a downsample of the looser one, instead of an
independent random subset, which keeps cap-level comparisons interpretable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

# Make ``utils.likes_cap`` importable when running pytest from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.likes_cap import apply_per_user_random_cap


def _build_likes_df(users: int = 5, likes_per_user: int = 30) -> pl.DataFrame:
    """Build a synthetic likes table with ``users`` users and ``likes_per_user`` likes each."""
    rows = [
        {
            "did": f"user_{u}",
            "subject_uri": f"post_{u}_{p}",
            "emb_idx": u * likes_per_user + p,
        }
        for u in range(users)
        for p in range(likes_per_user)
    ]
    return pl.DataFrame(rows)


def test_cap_none_or_zero_is_passthrough():
    likes = _build_likes_df()
    out_none = apply_per_user_random_cap(likes.lazy(), None, random_seed=42).collect()
    out_zero = apply_per_user_random_cap(likes.lazy(), 0, random_seed=42).collect()
    out_negative = apply_per_user_random_cap(likes.lazy(), -5, random_seed=42).collect()
    assert out_none.height == likes.height
    assert out_zero.height == likes.height
    assert out_negative.height == likes.height


def test_cap_enforces_per_user_limit():
    likes = _build_likes_df(users=10, likes_per_user=50)
    cap = 7
    out = apply_per_user_random_cap(likes.lazy(), cap, random_seed=42).collect()
    counts = out.group_by("did").len().rename({"len": "n"})
    assert counts["n"].max() == cap
    assert counts["n"].min() == cap, "every user should hit the cap when they have >= cap likes"
    assert out.height == 10 * cap


def test_cap_is_deterministic_for_same_seed():
    likes = _build_likes_df()
    a = apply_per_user_random_cap(likes.lazy(), 12, random_seed=42).collect()
    b = apply_per_user_random_cap(likes.lazy(), 12, random_seed=42).collect()
    a_sorted = a.sort(["did", "subject_uri"])
    b_sorted = b.sort(["did", "subject_uri"])
    assert a_sorted.equals(b_sorted)


def test_cap_n_is_subset_of_cap_m_for_n_lt_m():
    """The core nesting property used by cap sweeps."""
    likes = _build_likes_df(users=8, likes_per_user=40)
    seed = 42
    cap_levels = [40, 30, 20, 10, 5, 2]

    sets_by_cap = {}
    for cap in cap_levels:
        out = apply_per_user_random_cap(likes.lazy(), cap, random_seed=seed).collect()
        sets_by_cap[cap] = set(zip(out["did"].to_list(), out["subject_uri"].to_list()))

    for tighter, looser in zip(cap_levels[1:], cap_levels[:-1]):
        assert sets_by_cap[tighter].issubset(sets_by_cap[looser]), (
            f"cap={tighter} must be a subset of cap={looser} at the same seed; "
            f"violations: {sets_by_cap[tighter] - sets_by_cap[looser]}"
        )
        # Non-trivial: the tighter cap actually drops rows.
        assert len(sets_by_cap[tighter]) < len(sets_by_cap[looser])


def test_different_seeds_produce_different_subsets():
    likes = _build_likes_df(users=6, likes_per_user=30)
    cap = 10
    a = apply_per_user_random_cap(likes.lazy(), cap, random_seed=1).collect()
    b = apply_per_user_random_cap(likes.lazy(), cap, random_seed=2).collect()
    set_a = set(zip(a["did"].to_list(), a["subject_uri"].to_list()))
    set_b = set(zip(b["did"].to_list(), b["subject_uri"].to_list()))
    assert set_a != set_b, "different seeds should produce different selections"
    assert len(set_a) == len(set_b) == 6 * cap


def test_users_with_fewer_than_cap_pass_through_unchanged():
    rows = [
        {"did": "small_user", "subject_uri": "p1", "emb_idx": 1},
        {"did": "small_user", "subject_uri": "p2", "emb_idx": 2},
    ]
    for i in range(20):
        rows.append({"did": "big_user", "subject_uri": f"p_{i}", "emb_idx": 100 + i})
    likes = pl.DataFrame(rows)

    out = apply_per_user_random_cap(likes.lazy(), 10, random_seed=42).collect()
    by_user = out.group_by("did").len().rename({"len": "n"})
    by_user_dict = dict(zip(by_user["did"].to_list(), by_user["n"].to_list()))
    assert by_user_dict["small_user"] == 2
    assert by_user_dict["big_user"] == 10


def test_custom_user_and_post_columns():
    rows = [
        {"user": f"u_{u}", "post": f"q_{u}_{p}"}
        for u in range(3)
        for p in range(20)
    ]
    likes = pl.DataFrame(rows)
    out = apply_per_user_random_cap(
        likes.lazy(), 5, random_seed=42, user_col="user", post_col="post",
    ).collect()
    counts = out.group_by("user").len().rename({"len": "n"})
    assert counts["n"].to_list() == [5, 5, 5]


@pytest.mark.parametrize("seed", [1, 7, 42, 12345])
def test_nesting_holds_across_seeds(seed):
    likes = _build_likes_df(users=4, likes_per_user=25)
    big = apply_per_user_random_cap(likes.lazy(), 20, random_seed=seed).collect()
    small = apply_per_user_random_cap(likes.lazy(), 5, random_seed=seed).collect()
    big_set = set(zip(big["did"].to_list(), big["subject_uri"].to_list()))
    small_set = set(zip(small["did"].to_list(), small["subject_uri"].to_list()))
    assert small_set.issubset(big_set)
