#!/usr/bin/env python3
"""Aggregate fixed-cohort paired-AUC outputs into paper-ready conclusions.

Each input JSON is produced by ``fixed_cohort_auc.py`` for one matched
baseline/remedy model-seed pair.  The primary contrast is the equally weighted
mean of those seed-pair deltas.  The report deliberately does not pool legacy
native-holdout AUCs, because their cohorts vary by remedy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def verdict(ci_low: float | None, ci_high: float | None, margin: float) -> str:
    if ci_low is None or ci_high is None:
        return "insufficient_data"
    if ci_low > -margin:
        return "non_inferior" if ci_high > margin else "flat"
    return "utility_cost_exceeds_margin"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--stratum", default="typical")
    parser.add_argument(
        "--architecture",
        help="Require every input to carry this architecture in its provenance.",
    )
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        document = json.loads(path.read_text())
        actual_architecture = document.get("provenance", {}).get("architecture")
        if args.architecture is not None and actual_architecture != args.architecture:
            raise RuntimeError(
                f"{path} has architecture {actual_architecture!r}, expected "
                f"{args.architecture!r}."
            )
        cell = document["strata"].get(args.stratum)
        if cell is None or cell["delta_auc"] is None:
            continue
        rows.append(
            {
                "source": str(path),
                "delta_auc": cell["delta_auc"],
                "baseline_auc": cell["baseline_auc"],
                "remedy_auc": cell["remedy_auc"],
                "users": cell["users"],
                "rows": cell["rows"],
            }
        )
    if not rows:
        raise RuntimeError(f"No usable {args.stratum!r} estimates in inputs.")

    deltas = np.array([row["delta_auc"] for row in rows], dtype=float)
    rng = np.random.default_rng(args.bootstrap_seed)
    # This is a seed-pair uncertainty interval. Per-cell JSON carries the
    # user-clustered interval; full user-resampling across all paired parquet
    # files is intentionally left to the final artifact harvester.
    bootstrap_means = np.mean(
        rng.choice(deltas, size=(args.bootstrap_repetitions, len(deltas)), replace=True),
        axis=1,
    )
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975]).tolist()
    report = {
        "estimand": "mean paired fixed-cohort delta AUC over matched model seeds",
        "architecture": args.architecture,
        "stratum": args.stratum,
        "non_inferiority_margin": args.margin,
        "n_seed_pairs": len(rows),
        "mean_delta_auc": float(deltas.mean()),
        "seed_pair_bootstrap_ci_95": {"low": ci_low, "high": ci_high},
        "verdict": verdict(ci_low, ci_high, args.margin),
        "per_seed": rows,
        "caveat": (
            "The interval resamples seed pairs. The individual inputs retain "
            "user-clustered bootstrap intervals; final paper reporting should "
            "also run a joint user-clustered bootstrap from prediction parquets."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
