#!/usr/bin/env python3
"""Compare baseline and remedy AUC on one fixed, baseline-defined cohort.

The remedy's native prediction file is not an apples-to-apples utility test:
excluded users disappear from its training substrate.  Score the remedy
checkpoint on the *baseline* substrate with ``run_holdout_pred.py
--substrate-run-dir`` first, then pass that output here.

The normal path asserts identical, ordered ``(did, post_id, y_true)`` rows and
pairs predictions positionally.  This is intentional: prediction parquet
files may contain repeated triplets, and joining on those triplets turns every
duplicate into a Cartesian expansion.  If ordered rows differ, the script
fails by default rather than silently changing the estimand.  An explicit
``--allow-dedup-fallback`` option is available for diagnostics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score


KEYS = ["did", "post_id", "y_true"]
PRED_COLUMNS = [*KEYS, "y_pred_proba"]


def auc(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    labels = np.asarray(labels)
    if labels.size == 0 or np.unique(labels).size != 2:
        return None
    return float(roc_auc_score(labels, predictions))


def lorenz50_power_users(
    likes: pl.DataFrame, min_likes: int
) -> tuple[set[str], dict[str, int | float]]:
    """Match ``stratified_auc.py``'s Lorenz-50 convention exactly.

    The first user whose cumulative likes crosses 50% is *not* included.  This
    keeps the portable evaluator aligned with the historical table, avoiding
    the previous one-user cutoff discrepancy.
    """
    counts = (
        likes.group_by("did")
        .len()
        .filter(pl.col("len") >= min_likes)
        .sort("len", descending=True)
    )
    cumulative = counts["len"].cum_sum()
    cutoff = next(
        index for index, value in enumerate(cumulative) if value >= cumulative[-1] * 0.5
    )
    power = set(counts.head(cutoff)["did"].cast(pl.String).to_list())
    return power, {
        "lorenz_cutoff_index": cutoff,
        "eligible_users": counts.height,
        "n_power": len(power),
        "n_typical": counts.height - len(power),
        "eligible_likes": int(cumulative[-1]),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duplicate_summary(frame: pl.DataFrame) -> dict[str, int]:
    counts = frame.group_by(KEYS).len()
    duplicate_groups = counts.filter(pl.col("len") > 1)
    return {
        "duplicate_key_groups": duplicate_groups.height,
        "rows_in_duplicate_key_groups": int(
            duplicate_groups["len"].sum() if duplicate_groups.height else 0
        ),
    }


def pair_predictions(
    baseline: pl.DataFrame, remedy: pl.DataFrame, allow_dedup_fallback: bool
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Produce one paired row per evaluated example without Cartesian joins."""
    base_keys = baseline.select(KEYS)
    remedy_keys = remedy.select(KEYS)
    base_dupes = duplicate_summary(baseline)
    remedy_dupes = duplicate_summary(remedy)
    diagnostics: dict[str, object] = {
        "baseline": base_dupes,
        "remedy": remedy_dupes,
        "pairing": "positional",
    }

    if base_keys.equals(remedy_keys):
        return (
            pl.DataFrame(
                {
                    "did": baseline["did"],
                    "post_id": baseline["post_id"],
                    "y_true": baseline["y_true"],
                    "y_pred_proba": baseline["y_pred_proba"],
                    "y_pred_proba_remedy": remedy["y_pred_proba"],
                }
            ),
            diagnostics,
        )

    if not allow_dedup_fallback:
        raise RuntimeError(
            "Prediction keys differ in content or order. Refusing an unsafe join; "
            "regenerate remedy predictions on the baseline substrate, or use "
            "--allow-dedup-fallback only to diagnose a legacy artifact."
        )

    def collapse(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.group_by(KEYS, maintain_order=True).agg(
            pl.col("y_pred_proba").mean().alias("y_pred_proba")
        )

    base_collapsed = collapse(baseline)
    remedy_collapsed = collapse(remedy)
    paired = base_collapsed.join(
        remedy_collapsed, on=KEYS, how="inner", suffix="_remedy", maintain_order="left"
    )
    diagnostics["pairing"] = "mean_collapsed_key_join"
    diagnostics["fallback_warning"] = (
        "Keys differed; duplicate prediction keys were mean-collapsed before joining. "
        "This is a diagnostic fallback, not the preregistered primary estimator."
    )
    return paired, diagnostics


def stratum_summary(frame: pl.DataFrame) -> dict[str, int | float | None]:
    labels = frame["y_true"].to_numpy()
    base_pred = frame["y_pred_proba"].to_numpy()
    remedy_pred = frame["y_pred_proba_remedy"].to_numpy()
    baseline_auc = auc(labels, base_pred)
    remedy_auc = auc(labels, remedy_pred)
    return {
        "users": frame["did"].n_unique(),
        "rows": frame.height,
        "baseline_auc": baseline_auc,
        "remedy_auc": remedy_auc,
        "delta_auc": (
            None if baseline_auc is None or remedy_auc is None else remedy_auc - baseline_auc
        ),
    }


def bootstrap_delta_auc(
    frame: pl.DataFrame, repetitions: int, seed: int
) -> dict[str, float | int | None]:
    """User-clustered percentile bootstrap for the paired AUC contrast."""
    if frame.height == 0 or repetitions == 0:
        return {"repetitions": repetitions, "ci_95_low": None, "ci_95_high": None}

    grouped = frame.partition_by("did", maintain_order=True)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(grouped), size=len(grouped))
        sampled = pl.concat([grouped[index] for index in indices], rechunk=False)
        point = stratum_summary(sampled)["delta_auc"]
        if point is not None:
            deltas.append(float(point))
    if not deltas:
        return {"repetitions": repetitions, "ci_95_low": None, "ci_95_high": None}
    return {
        "repetitions": repetitions,
        "ci_95_low": float(np.quantile(deltas, 0.025)),
        "ci_95_high": float(np.quantile(deltas, 0.975)),
    }


def source_commit() -> str | None:
    if os.environ.get("PL_SOURCE_COMMIT"):
        return os.environ["PL_SOURCE_COMMIT"]
    index = os.environ.get("PL_EXPORT_INDEX")
    if index:
        path = Path(index)
    else:
        path = Path.home() / "power-likers" / "SHA256SUMS.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("source_commit")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remedy", type=Path, required=True)
    parser.add_argument("--likes-core", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-likes", type=int, default=5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--architecture", default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--exclusion-file", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--allow-dedup-fallback", action="store_true")
    args = parser.parse_args()

    baseline = (
        pl.read_parquet(args.baseline)
        .select(PRED_COLUMNS)
        .with_columns(pl.col("did").cast(pl.String))
    )
    remedy = (
        pl.read_parquet(args.remedy)
        .select(PRED_COLUMNS)
        .with_columns(pl.col("did").cast(pl.String))
    )
    paired, pairing = pair_predictions(baseline, remedy, args.allow_dedup_fallback)
    if paired.height == 0:
        raise RuntimeError("No identical prediction rows; check that remedy was scored on the baseline substrate.")

    power, lorenz = lorenz50_power_users(
        pl.read_parquet(args.likes_core).select("did"), args.min_likes
    )
    paired = paired.with_columns(
        pl.col("did").is_in(power).alias("power_liker"),
    )

    summary: dict[str, object] = {
        "cohort_definition": {
            "substrate": str(args.baseline),
            "comparison": str(args.remedy),
            "stratum": "Lorenz-50 on likes_core",
            "min_likes": args.min_likes,
            "comparison_keys": KEYS,
        },
        "lorenz": lorenz,
        "pairing": pairing,
        "common_rows": paired.height,
        "common_users": paired["did"].n_unique(),
        "provenance": {
            "condition": args.condition,
            "architecture": args.architecture,
            "model_seed": args.model_seed,
            "source_commit": source_commit(),
            "baseline_sha256": sha256(args.baseline),
            "remedy_sha256": sha256(args.remedy),
            "likes_core_sha256": sha256(args.likes_core),
            "config": str(args.config) if args.config else None,
            "config_sha256": sha256(args.config) if args.config else None,
            "exclusion_file": str(args.exclusion_file) if args.exclusion_file else None,
            "exclusion_file_sha256": (
                sha256(args.exclusion_file) if args.exclusion_file else None
            ),
            "utc_created_at": datetime.now(timezone.utc).isoformat(),
        },
        "strata": {},
    }
    for name, frame in {
        "all": paired,
        "typical": paired.filter(~pl.col("power_liker")),
        "power_liker": paired.filter(pl.col("power_liker")),
    }.items():
        summary["strata"][name] = {
            **stratum_summary(frame),
            "bootstrap": bootstrap_delta_auc(
                frame, args.bootstrap_repetitions, args.bootstrap_seed
            ),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    if args.summary_csv:
        rows = []
        for stratum, values in summary["strata"].items():
            bootstrap = values["bootstrap"]
            rows.append(
                {
                    "condition": args.condition,
                    "architecture": args.architecture,
                    "seed": args.model_seed,
                    "stratum": stratum,
                    **{key: value for key, value in values.items() if key != "bootstrap"},
                    **bootstrap,
                    "baseline": str(args.baseline),
                    "remedy": str(args.remedy),
                }
            )
        append = args.summary_csv.exists()
        with args.summary_csv.open("a") as stream:
            pl.DataFrame(rows).write_csv(stream, include_header=not append)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
