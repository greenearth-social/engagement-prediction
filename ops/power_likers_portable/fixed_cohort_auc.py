#!/usr/bin/env python3
"""Compare baseline and remedy AUC on one fixed, baseline-defined cohort.

The remedy's native prediction file is not an apples-to-apples power-liker
test: excluded users disappear from its training substrate.  Score the remedy
checkpoint on the *baseline* substrate with ``run_holdout_pred.py
--substrate-run-dir`` first, then pass that output here.  The script limits
both models to the identical (DID, post_id, label) prediction rows and records
both user and row denominators by Lorenz-50 stratum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
from sklearn.metrics import roc_auc_score


def auc(frame: pl.DataFrame) -> float | None:
    labels = frame["y_true"].to_list()
    if not labels or len(set(labels)) != 2:
        return None
    return float(roc_auc_score(labels, frame["y_pred_proba"].to_list()))


def power_likers(likes: pl.DataFrame, min_likes: int) -> set[str]:
    counts = (
        likes.group_by("did")
        .len()
        .filter(pl.col("len") >= min_likes)
        .sort("len", descending=True)
    )
    cumulative = counts["len"].cum_sum()
    cutoff = next(index for index, value in enumerate(cumulative) if value >= cumulative[-1] * 0.5)
    return set(counts.head(cutoff + 1)["did"].cast(pl.String).to_list())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remedy", type=Path, required=True)
    parser.add_argument("--likes-core", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-likes", type=int, default=5)
    args = parser.parse_args()

    columns = ["did", "post_id", "y_true", "y_pred_proba"]
    baseline = pl.read_parquet(args.baseline).select(columns).with_columns(pl.col("did").cast(pl.String))
    remedy = pl.read_parquet(args.remedy).select(columns).with_columns(pl.col("did").cast(pl.String))
    keys = ["did", "post_id", "y_true"]
    paired = baseline.join(remedy, on=keys, how="inner", suffix="_remedy")
    if paired.height == 0:
        raise RuntimeError("No identical prediction rows; check that remedy was scored on the baseline substrate.")

    power = power_likers(pl.read_parquet(args.likes_core).select("did"), args.min_likes)
    paired = paired.with_columns(
        pl.col("did").is_in(power).alias("power_liker"),
    )

    summary: dict[str, object] = {
        "cohort_definition": {
            "substrate": str(args.baseline),
            "comparison": str(args.remedy),
            "stratum": "Lorenz-50 on likes_core",
            "min_likes": args.min_likes,
            "comparison_keys": keys,
        },
        "common_rows": paired.height,
        "common_users": paired["did"].n_unique(),
        "strata": {},
    }
    for name, frame in {
        "all": paired,
        "typical": paired.filter(~pl.col("power_liker")),
        "power_liker": paired.filter(pl.col("power_liker")),
    }.items():
        baseline_frame = frame.select(["y_true", "y_pred_proba"])
        remedy_frame = frame.select(["y_true", "y_pred_proba_remedy"]).rename({"y_pred_proba_remedy": "y_pred_proba"})
        baseline_auc = auc(baseline_frame)
        remedy_auc = auc(remedy_frame)
        summary["strata"][name] = {
            "users": frame["did"].n_unique(),
            "rows": frame.height,
            "baseline_auc": baseline_auc,
            "remedy_auc": remedy_auc,
            "delta_auc": None if baseline_auc is None or remedy_auc is None else remedy_auc - baseline_auc,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
