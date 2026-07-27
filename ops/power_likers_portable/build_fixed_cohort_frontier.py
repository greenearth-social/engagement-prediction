#!/usr/bin/env python3
"""Join matched fixed-cohort utility and synthetic-feed bias contrasts.

Each seed comparison joins user-level outcomes by ``did`` and trait/feature, so
the reported bias change has the same matched baseline/remedy cohort discipline
as the paired AUC analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl


def load_cells(stage1: Path) -> dict[str, dict[int, Path]]:
    cells: dict[str, dict[int, Path]] = defaultdict(dict)
    for manifest_path in stage1.glob("**/fixed_cohort_bias/fixed_cohort_bias_manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        condition = manifest["condition"]
        seed = int(manifest["model_seed"])
        if seed in cells[condition]:
            raise RuntimeError(f"Duplicate fixed-bias cell for {condition} seed {seed}.")
        cells[condition][seed] = manifest_path.parent
    return dict(cells)


def paired_seed_means(
    baseline: Path, remedy: Path, filename: str, keys: list[str]
) -> tuple[float, int]:
    base = pl.read_parquet(baseline / filename).select(*keys, "model_excess_abs")
    other = pl.read_parquet(remedy / filename).select(*keys, "model_excess_abs")
    paired = base.join(other, on=keys, how="inner", suffix="_remedy")
    if paired.is_empty():
        raise RuntimeError(f"No matched rows in {filename} for {remedy}.")
    delta = paired["model_excess_abs_remedy"] - paired["model_excess_abs"]
    return float(delta.mean()), paired.height


def seed_bootstrap_ci(values: list[float], repetitions: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values_array = np.asarray(values, dtype=float)
    means = np.array(
        [
            values_array[rng.integers(0, len(values_array), size=len(values_array))].mean()
            for _ in range(repetitions)
        ]
    )
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def utility_summary(reports_dir: Path) -> dict[str, dict[str, object]]:
    result = {}
    for path in reports_dir.glob("*_typical_paired_auc.json"):
        document = json.loads(path.read_text())
        result[path.stem.removesuffix("_typical_paired_auc")] = document
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows produced for {path.name}.")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", required=True, type=Path)
    parser.add_argument("--reports-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    args = parser.parse_args()

    cells = load_cells(args.stage1)
    baseline = cells.get("baseline", {})
    if set(baseline) != {1, 2, 3, 4, 5}:
        raise RuntimeError(f"Expected five baseline fixed-bias cells; got seeds {sorted(baseline)}.")
    utility = utility_summary(args.reports_dir)
    conditions = sorted(condition for condition in cells if condition != "baseline")
    missing_utility = sorted(set(conditions) - set(utility))
    if missing_utility:
        raise RuntimeError(f"Missing paired-AUC reports for {missing_utility}.")

    focal_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    for condition in conditions:
        remedies = cells[condition]
        if set(remedies) != set(baseline):
            raise RuntimeError(
                f"{condition} fixed-bias seeds {sorted(remedies)} do not match "
                f"baseline seeds {sorted(baseline)}."
            )
        utility_doc = utility[condition]
        focal_by_trait: dict[tuple[str, str, str], list[tuple[float, int]]] = defaultdict(list)
        control_by_feature: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for model_seed in sorted(baseline):
            base_dir, remedy_dir = baseline[model_seed], remedies[model_seed]
            focal = pl.read_parquet(base_dir / "fixed_cohort_typical_axis_a_user_level.parquet")
            for group, trait, alias in focal.select("group", "trait", "alias").unique().iter_rows():
                base_subset = focal.filter((pl.col("group") == group) & (pl.col("trait") == trait))
                remedy_subset = pl.read_parquet(
                    remedy_dir / "fixed_cohort_typical_axis_a_user_level.parquet"
                ).filter((pl.col("group") == group) & (pl.col("trait") == trait))
                paired = base_subset.join(
                    remedy_subset.select("did", "model_excess_abs"),
                    on="did",
                    how="inner",
                    suffix="_remedy",
                )
                if paired.is_empty():
                    raise RuntimeError(f"No paired focal rows for {condition} seed {model_seed} {trait}.")
                focal_by_trait[(group, trait, alias)].append(
                    (
                        float(
                            (
                                paired["model_excess_abs_remedy"]
                                - paired["model_excess_abs"]
                            ).mean()
                        ),
                        paired.height,
                    )
                )
            controls = pl.read_parquet(base_dir / "fixed_cohort_negative_controls_user_level.parquet")
            for feature in controls["feature"].unique().to_list():
                base_subset = controls.filter(pl.col("feature") == feature)
                remedy_subset = pl.read_parquet(
                    remedy_dir / "fixed_cohort_negative_controls_user_level.parquet"
                ).filter(pl.col("feature") == feature)
                paired = base_subset.join(
                    remedy_subset.select("did", "model_excess_abs"),
                    on="did",
                    how="inner",
                    suffix="_remedy",
                )
                control_by_feature[feature].append(
                    (
                        float(
                            (
                                paired["model_excess_abs_remedy"]
                                - paired["model_excess_abs"]
                            ).mean()
                        ),
                        paired.height,
                    )
                )
        for (group, trait, alias), seed_values in sorted(focal_by_trait.items()):
            deltas = [value for value, _ in seed_values]
            low, high = seed_bootstrap_ci(
                deltas,
                args.bootstrap_repetitions,
                seed=stable_seed(condition, group, trait),
            )
            focal_rows.append(
                {
                    "condition": condition,
                    "group": group,
                    "trait": trait,
                    "alias": alias,
                    "mean_delta_model_excess_pp": np.mean(deltas) * 100,
                    "seed_pair_ci_95_low_pp": low * 100,
                    "seed_pair_ci_95_high_pp": high * 100,
                    "seed_pairs": len(seed_values),
                    "mean_paired_users": np.mean([n for _, n in seed_values]),
                    "mean_delta_auc": utility_doc["mean_delta_auc"],
                    "auc_ci_95_low": utility_doc["seed_pair_bootstrap_ci_95"]["low"],
                    "auc_ci_95_high": utility_doc["seed_pair_bootstrap_ci_95"]["high"],
                    "auc_verdict": utility_doc["verdict"],
                }
            )
        for feature, seed_values in sorted(control_by_feature.items()):
            deltas = [value for value, _ in seed_values]
            low, high = seed_bootstrap_ci(
                deltas,
                args.bootstrap_repetitions,
                seed=stable_seed(condition, feature),
            )
            control_rows.append(
                {
                    "condition": condition,
                    "feature": feature,
                    "mean_delta_model_excess_raw": np.mean(deltas),
                    "seed_pair_ci_95_low": low,
                    "seed_pair_ci_95_high": high,
                    "seed_pairs": len(seed_values),
                    "mean_paired_users": np.mean([n for _, n in seed_values]),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "fixed_cohort_bias_utility_frontier.csv", focal_rows)
    write_csv(args.out_dir / "fixed_cohort_negative_control_deltas.csv", control_rows)
    (args.out_dir / "fixed_cohort_frontier_manifest.json").write_text(
        json.dumps(
            {
                "stage1": str(args.stage1),
                "reports_dir": str(args.reports_dir),
                "conditions": conditions,
                "focal_rows": len(focal_rows),
                "control_rows": len(control_rows),
                "bootstrap_repetitions": args.bootstrap_repetitions,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"focal_rows": len(focal_rows), "control_rows": len(control_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
