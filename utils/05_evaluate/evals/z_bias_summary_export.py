#!/usr/bin/env python3

"""
Bias summary export for cross-cell aggregation.

NOTE: This file is deliberately prefixed with ``z_`` so it sorts last in
``pkgutil.iter_modules``, guaranteeing it runs *after* ``synthetic_feed``
(whose summary it consumes).  Do not rename without preserving that ordering.

The :mod:`synthetic_feed` module already computes the per-user A / B /
model-amplification decomposition for each NLP trait and writes a nested
``synthetic_feed_summary.json``.  For cap × architecture × seed sweeps we want
a *flat* representation of those numbers, tagged with the sweep cell's
identity (effective_likes_cap, random_seed, model_type, user_encoder), so an
analysis notebook can ``pl.read_parquet("**/bias_by_trait_*.parquet")`` and
group by cell without walking nested directories or parsing JSON.

This module is a pure post-processor: it reads the synthetic_feed summary
that the prior eval-module run just wrote, joins in cell metadata from the
training_config.json sitting two levels above the eval output dir, and emits
two artifacts:

- ``bias_by_trait.parquet``: one row per (group, trait) with all the
  decomposition stats and the cell-identity columns.
- ``bias_summary.json``: a JSON copy of the same with a small headline.

If the upstream synthetic_feed run did not produce its summary (e.g. it was
skipped), this module skips silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from . import EvalContext, EvalModule


# Columns we pull from training_config.json so each row in the flat parquet
# is self-describing.  None values are kept as-is (rendered as null in the
# parquet) so cross-cell joins are robust to evolving training schemas.
_CELL_METADATA_KEYS: List[str] = [
    "model_type",
    "user_encoder",
    "user_summarization",
    "ema_alpha",
    "effective_likes_cap",
    "effective_likes_cap_seed",
    "max_likes_per_user",
    "random_seed",
    "epochs",
    "batch_size",
    "best_val_auc",
]


def _find_training_config(eval_out_dir: Path) -> Optional[Path]:
    """Stage 5 nests its output as <train_dir>/evals/<ts>/, so the training
    config sits two parents up.
    """
    candidate = eval_out_dir.parent.parent / "training_config.json"
    return candidate if candidate.exists() else None


def _flatten_groups(groups: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group_name, traits in (groups or {}).items():
        for trait_label, stats in (traits or {}).items():
            row: Dict[str, Any] = {
                "group": group_name,
                "trait": trait_label,
            }
            row.update(stats)
            rows.append(row)
    return rows


def _load_synthetic_feed_summary(eval_out_dir: Path) -> Optional[Dict[str, Any]]:
    summary_path = eval_out_dir / "synthetic_feed" / "synthetic_feed_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path, "r") as f:
        return json.load(f)


def _load_cell_metadata(eval_out_dir: Path) -> Dict[str, Any]:
    cfg_path = _find_training_config(eval_out_dir)
    if cfg_path is None:
        return {k: None for k in _CELL_METADATA_KEYS}
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    return {k: cfg.get(k) for k in _CELL_METADATA_KEYS}


class BiasSummaryExportModule(EvalModule):
    """Flatten synthetic_feed's per-trait decomposition into a tagged parquet."""

    name = "bias_summary_export"
    description = (
        "Flattens synthetic_feed_summary.json into a per-trait parquet with "
        "cell-identity columns from training_config.json for cross-cell "
        "aggregation across cap/arch/seed sweeps"
    )

    def run(self, ctx: EvalContext) -> Dict[str, Any]:
        out_dir = self.get_output_dir(ctx)

        synth = _load_synthetic_feed_summary(ctx.output_dir)
        if synth is None:
            return {
                "skipped": True,
                "reason": "synthetic_feed_summary.json not found "
                          "(the synthetic_feed module did not run or was skipped)",
            }

        cell_metadata = _load_cell_metadata(ctx.output_dir)

        rows = _flatten_groups(synth.get("groups", {}))
        for row in rows:
            row.update(cell_metadata)
            row["eval_timestamp"] = ctx.timestamp

        if not rows:
            return {
                "skipped": True,
                "reason": "synthetic_feed produced no per-trait records",
            }

        df = pl.DataFrame(rows)

        parquet_path = out_dir / f"bias_by_trait_{ctx.timestamp}.parquet"
        df.write_parquet(parquet_path, compression="zstd")

        json_path = out_dir / f"bias_summary_{ctx.timestamp}.json"
        flat_summary = {
            "headline": synth.get("headline"),
            "n_pool_posts": synth.get("n_pool_posts"),
            "n_users_eligible": synth.get("n_users_eligible"),
            "n_users_scored": synth.get("n_users_scored"),
            "cell_metadata": cell_metadata,
            "n_traits": len(rows),
            "rows": rows,
        }
        self.save_json(flat_summary, json_path)

        return {
            "parquet_path": str(parquet_path),
            "json_path": str(json_path),
            "n_traits": len(rows),
            "cell_metadata": cell_metadata,
        }
