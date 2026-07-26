#!/usr/bin/env python3
"""Emit paper-quality holdout-strata and per-cell provenance artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from fixed_cohort_auc import lorenz50_power_users


def digest(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hash_.update(block)
    return hash_.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-dir", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--likes-core", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--exclusion-file", type=Path, default=None)
    parser.add_argument("--substrate-id", required=True)
    parser.add_argument("--min-likes", type=int, default=5)
    args = parser.parse_args()

    out_dir = args.out_dir or args.cell_dir / "paper_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = pl.read_parquet(args.predictions).with_columns(pl.col("did").cast(pl.String))
    likes = pl.read_parquet(args.likes_core).select(pl.col("did").cast(pl.String))
    power, lorenz = lorenz50_power_users(likes, args.min_likes)
    stratified = predictions.with_columns(
        pl.when(pl.col("did").is_in(power)).then(pl.lit("power_liker"))
        .otherwise(pl.lit("typical")).alias("stratum")
    )
    excluded_users: set[str] = set()
    if args.exclusion_file:
        excluded = pl.read_parquet(args.exclusion_file)
        did_col = "did" if "did" in excluded.columns else excluded.columns[0]
        excluded_users = set(excluded[did_col].cast(pl.String).to_list())

    strata = []
    for name in ("typical", "power_liker"):
        frame = stratified.filter(pl.col("stratum") == name)
        strata.append(
            {
                "stratum": name,
                "users": frame["did"].n_unique(),
                "rows": frame.height,
                "positive_rows": int(frame["y_true"].sum()),
                "negative_rows": int(frame.height - frame["y_true"].sum()),
                "users_excluded_from_training": len(set(frame["did"].to_list()) & excluded_users),
            }
        )
    (out_dir / "holdout_strata.json").write_text(
        json.dumps(
            {
                "prediction_file": str(args.predictions),
                "prediction_sha256": digest(args.predictions),
                "substrate_id": args.substrate_id,
                "min_likes": args.min_likes,
                "lorenz": lorenz,
                "strata": strata,
            },
            indent=2,
        )
        + "\n"
    )

    config_path = args.cell_dir / "training_config.json"
    config = json.loads(config_path.read_text())
    manifest = {
        "utc_created_at": datetime.now(timezone.utc).isoformat(),
        "cell_dir": str(args.cell_dir),
        "substrate_id": args.substrate_id,
        "training_config": str(config_path),
        "training_config_sha256": digest(config_path),
        "model_type": config.get("model_type"),
        "random_seed": config.get("random_seed"),
        "best_val_auc": config.get("best_val_auc"),
        "exclusion_file": str(args.exclusion_file) if args.exclusion_file else None,
        "exclusion_file_sha256": digest(args.exclusion_file) if args.exclusion_file else None,
        "n_excluded_users": len(excluded_users),
        "holdout_predictions": str(args.predictions),
        "holdout_predictions_sha256": digest(args.predictions),
    }
    (out_dir / "cell_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
