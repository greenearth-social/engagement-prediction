#!/usr/bin/env python3
"""Emit real D1/D2 paper artifacts from one completed synthetic-feed eval.

This is deliberately a post-processor: it reuses the deterministic pool and
the saved ``synthetic_feed_topk.json`` rather than re-scoring a model.  It is
therefore CPU-only and can run after every Stage-05 cell without changing the
model or the feed realization.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
synthetic = importlib.import_module("utils.05_evaluate.evals.synthetic_feed")

# (synthetic-feed group, native feature column) -> paper-facing alias.
FOCAL_TRAITS = {
    ("sentiment", "Negative"): "Neg. Sentiment",
    ("sentiment", "Positive"): "Pos. Sentiment",
    ("topic", "News & Social Concern"): "News & Soc. Concern",
    ("topic", "Diaries & Daily Life"): "Diaries",
    ("moderation", "hate"): "Hate",
    ("emotion_sentiment", "Annoyance"): "Annoyance",
    ("emotion_sentiment", "Gratitude"): "Gratitude",
    ("emotion_sentiment", "Anger"): "Anger",
}


def lorenz50_power_users(likes: pl.DataFrame, min_likes: int) -> set[str]:
    counts = (
        likes.group_by("did").len().filter(pl.col("len") >= min_likes)
        .sort("len", descending=True)
    )
    cumulative = counts["len"].cum_sum()
    cutoff = next(i for i, value in enumerate(cumulative) if value >= cumulative[-1] * 0.5)
    return set(counts.head(cutoff)["did"].cast(pl.String).to_list())


def stage_file(run_dir: Path, prefix: str) -> Path:
    from utils.pipeline.core import select_prior_output
    from utils.helpers import load_parquet_from_prior

    get_data = select_prior_output(run_dir, "01_get_data")
    if get_data is None:
        raise FileNotFoundError(f"No Stage-1 output under {run_dir}")
    # Resolve the concrete source path so we can retain text/time columns.
    candidates = sorted(get_data.glob(f"{prefix}*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No {prefix} parquet under {get_data}")
    return candidates[0]


def trait_rows(
    pool_values: np.ndarray,
    liked_values: np.ndarray,
    liked_dids: np.ndarray,
    topk: dict[str, list[int]],
    dids: list[str],
) -> dict[str, float | int] | None:
    actual: dict[str, np.ndarray] = {}
    for did in dids:
        vals = liked_values[liked_dids == did]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= synthetic.MIN_USER_LIKES:
            actual[did] = vals
    feed = {
        did: pool_values[np.asarray(indices, dtype=int)]
        for did, indices in topk.items()
        if did in dids
    }
    feed = {did: vals[np.isfinite(vals)] for did, vals in feed.items()}
    result = synthetic._compute_trait_decomposition(pool_values, actual, feed, dids)
    if result is None:
        return None
    return {
        "mean_user_pref_std": float(np.mean(result.user_pref_std)),
        "mean_model_amp_std": float(np.mean(result.model_amp_std)),
        "mean_model_excess_std": float(np.mean(result.model_excess_std)),
        "mean_user_pref_abs": float(np.mean(result.user_pref_std) * result.pool_sd),
        "mean_model_amp_abs": float(np.mean(result.model_amp_std) * result.pool_sd),
        "mean_model_excess_abs": float(np.mean(result.model_excess_std) * result.pool_sd),
        "pool_mean": float(result.pool_mean),
        "pool_sd": float(result.pool_sd),
        "n_users": result.n_users,
        "cohen_d_amp": float(result.cohen_d_amp),
        "p_amp": float(result.p_amp),
    }


def structural_features(frame: pl.DataFrame) -> pl.DataFrame:
    text = pl.col("record_text").fill_null("")
    timestamp = (
        pl.col("record_created_at").cast(pl.String)
        .str.to_datetime(strict=False, time_zone="UTC")
    )
    return frame.with_columns(
        [
            text.str.count_matches(r"\S+").cast(pl.Float64).alias("word_count"),
            text.str.contains(r"https?://", literal=False).cast(pl.Float64).alias("has_url"),
            text.str.contains(r"#\w+", literal=False).cast(pl.Float64).alias("has_hashtag"),
            timestamp.dt.hour().cast(pl.Float64).alias("hour_of_day_utc"),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--eval-dir", required=True, type=Path,
                        help="Stage-05 eval directory containing synthetic_feed/")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--likes-core", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--min-likes", type=int, default=5)
    args = parser.parse_args()

    checkpoint = json.loads(
        (args.eval_dir / "synthetic_feed" / "synthetic_feed_topk.json").read_text()
    )
    topk = {str(did): indices for did, indices in checkpoint["user_topk"].items()}
    posts = pl.read_parquet(stage_file(args.run_dir, "posts_core_"))
    inferences = pl.read_parquet(stage_file(args.run_dir, "inferences_core_"))
    pool = (
        posts.filter(pl.col("in_random_sample"))
        .join(inferences.select("at_uri", "inferences"), on="at_uri", how="inner")
        .sample(n=checkpoint["n_pool"], seed=42)
    )
    if "pool_at_uris" in checkpoint:
        if pool["at_uri"].to_list() != checkpoint["pool_at_uris"]:
            raise RuntimeError(
                "Deterministic pool reconstruction differs from the scored pool; "
                "refusing to apply saved top-K indices to different posts."
            )
    predictions = pl.read_parquet(args.predictions).with_columns(pl.col("did").cast(pl.String))
    liked = (
        predictions.filter(pl.col("y_true") == 1).select("did", "post_id")
        .join(posts, left_on="post_id", right_on="at_uri", how="inner")
        .join(inferences.select("at_uri", "inferences"), left_on="post_id", right_on="at_uri", how="inner")
    )
    power = lorenz50_power_users(pl.read_parquet(args.likes_core).select("did"), args.min_likes)
    typical = sorted(set(topk) - power)

    pool_flat, groups = synthetic._unnest_text_inferences(pool)
    liked_flat, _ = synthetic._unnest_text_inferences(liked)
    liked_dids = liked_flat["did"].cast(pl.String).to_numpy()
    d1_rows: list[dict[str, object]] = []
    for group in groups:
        pool_group = pool_flat.select(group).unnest(group)
        liked_group = liked_flat.select(group).unnest(group)
        for trait in pool_group.columns:
            alias = FOCAL_TRAITS.get((group, trait))
            if alias is None:
                continue
            stats = trait_rows(
                pool_group[trait].to_numpy().astype(float),
                liked_group[trait].to_numpy().astype(float),
                liked_dids,
                topk,
                typical,
            )
            if stats:
                d1_rows.append(
                    {"group": group, "trait": trait, "alias": alias, "stratum": "typical", **stats}
                )

    pool_struct = structural_features(pool).select(
        "word_count", "has_url", "has_hashtag", "hour_of_day_utc"
    )
    liked_struct = structural_features(liked).select(
        "word_count", "has_url", "has_hashtag", "hour_of_day_utc"
    )
    d2_rows: list[dict[str, object]] = []
    all_dids = sorted(set(topk))
    for feature in pool_struct.columns:
        stats = trait_rows(
            pool_struct[feature].to_numpy().astype(float),
            liked_struct[feature].to_numpy().astype(float),
            liked_dids,
            topk,
            all_dids,
        )
        if stats:
            d2_rows.append({"feature": feature, "stratum": "all", **stats})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(d1_rows).write_parquet(args.out_dir / "typical_axis_a_focal.parquet")
    pl.DataFrame(d2_rows).write_parquet(args.out_dir / "negative_controls.parquet")
    manifest = {
        "n_pool": pool.height,
        "n_liked_rows": liked.height,
        "n_scored_users": len(topk),
        "n_typical_users": len(typical),
        "n_power_users": len(set(topk) & power),
        "d1_rows": len(d1_rows),
        "d2_rows": len(d2_rows),
    }
    (args.out_dir / "paper_quality_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
