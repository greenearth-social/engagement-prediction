#!/usr/bin/env python3

"""
Content Biases Evaluation Module

Measures whether the model's predicted engagement probability is correlated
with NLP content features (topic, sentiment, toxicity, etc.) on holdout
*negative* samples.  A strong correlation on negatives indicates the model
has learned a content bias independent of actual user preference.

For each inference group (e.g. emotion_sentiment, topic, moderation) a
horizontal bar chart of Spearman rank correlations is saved, plus a JSON
summary.

Outputs (under content_biases/):
- <group>_bias.png: one bar chart per inference group
- content_biases_summary.json: all correlations and metadata
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from . import EvalContext, EvalModule

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


def _correlations_for_group(
    y_pred: np.ndarray,
    group_df: pl.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, tuple[float, float, float]]:
    """Spearman correlation + Fisher-z CI between y_pred and each column.

    Returns {label: (rho, ci_lo, ci_hi)} for each column with enough data.
    """
    from scipy.stats import norm

    z_crit = norm.ppf(1 - alpha / 2)
    corrs: Dict[str, tuple[float, float, float]] = {}
    for col in group_df.columns:
        vals = group_df[col].to_numpy()
        mask = np.isfinite(vals)
        n = int(mask.sum())
        if n < 10:
            continue
        rho, _ = spearmanr(y_pred[mask], vals[mask])
        se = 1.0 / np.sqrt(n - 3)
        z = np.arctanh(rho)
        ci_lo = float(np.tanh(z - z_crit * se))
        ci_hi = float(np.tanh(z + z_crit * se))
        corrs[col] = (float(rho), ci_lo, ci_hi)
    return corrs


def _plot_group(
    group_name: str,
    corrs: Dict[str, tuple[float, float, float]],
    out_dir: Path,
) -> Path:
    labels = sorted(corrs, key=lambda k: abs(corrs[k][0]), reverse=True)
    rhos = [corrs[k][0] for k in labels]
    ci_lo = [corrs[k][1] for k in labels]
    ci_hi = [corrs[k][2] for k in labels]
    xerr_neg = [r - lo for r, lo in zip(rhos, ci_lo)]
    xerr_pos = [hi - r for r, hi in zip(rhos, ci_hi)]
    colors = ["#4878CF" if v >= 0 else "#D65F5F" for v in rhos]

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(labels))))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, rhos, color=colors, edgecolor="white", linewidth=0.5,
            xerr=[xerr_neg, xerr_pos], error_kw=dict(ecolor="#333333", capsize=2, linewidth=0.8))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ with predicted P(engagement)")
    ax.set_title(group_name.replace("_", " ").title())
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()

    path = out_dir / f"{group_name}_bias.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


class ContentBiasesModule(EvalModule):
    name = "content_biases"
    description = "Correlation between predicted engagement and NLP content features"

    def run(self, ctx: EvalContext) -> Dict[str, Any]:
        out_dir = self.get_output_dir(ctx)
        run_dir = ctx.config.get("run_dir")
        if run_dir is None:
            return {"skipped": True, "reason": "run_dir not in eval config"}

        try:
            inferences_lf = _load_inferences(Path(run_dir))
        except FileNotFoundError as e:
            return {"skipped": True, "reason": str(e)}

        # Predictions -> Polars, negatives only
        preds = (
            pl.from_pandas(ctx.predictions_df)
            .filter(pl.col("y_true") == 0)
            .select("post_id", "y_pred_proba")
        )
        n_negatives = len(preds)

        # Join predictions to inferences on post_id == at_uri
        joined = (
            inferences_lf
            .join(preds.lazy(), left_on="at_uri", right_on="post_id", how="inner")
            .collect()
        )
        n_matched = len(joined)
        if n_matched < 30:
            return {
                "skipped": True,
                "reason": f"only {n_matched} negatives matched inferences",
            }

        # Unnest to get inference group structs as top-level columns
        flat, group_names = _unnest_text_inferences(joined)
        y_pred = flat["y_pred_proba"].to_numpy()

        all_corrs: Dict[str, Dict[str, tuple[float, float, float]]] = {}
        plot_paths: list[str] = []

        for gname in group_names:
            group_df = flat.select(gname).unnest(gname)
            corrs = _correlations_for_group(y_pred, group_df)
            if not corrs:
                continue
            all_corrs[gname] = corrs
            path = _plot_group(gname, corrs, out_dir)
            plot_paths.append(str(path))

        # Flatten tuples for JSON: {group: {label: {rho, ci_lo, ci_hi}}}
        corrs_for_json = {
            g: {label: {"rho": r, "ci_lo": lo, "ci_hi": hi}
                for label, (r, lo, hi) in labels.items()}
            for g, labels in all_corrs.items()
        }
        summary = {
            "n_negatives": n_negatives,
            "n_matched": n_matched,
            "coverage_pct": round(100.0 * n_matched / n_negatives, 2) if n_negatives else 0,
            "groups": list(all_corrs.keys()),
            "correlations": corrs_for_json,
        }
        self.save_json(summary, out_dir / "content_biases_summary.json")

        return {
            "n_negatives": n_negatives,
            "n_matched": n_matched,
            "groups_plotted": len(all_corrs),
            "plot_paths": plot_paths,
        }
