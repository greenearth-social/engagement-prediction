#!/usr/bin/env python3

"""
Trait Amplification Evaluation Module

Compares the model's within-user association between predicted engagement and
each NLP content trait against the *actual* within-user association (based on
y_true).  The difference -- "amplification" -- reveals traits the model
over- or under-weights relative to real user preferences.

Uses the full holdout set (positives + negatives).  Within-user demeaning
removes user-level base-rate differences so the pooled correlations reflect
purely within-user variation.  Bootstrap resampling over users provides CIs.

Outputs (under trait_amplification/):
- amplification_scatter.png: headline scatter of rho_true vs rho_pred
- <group>_amplification.png: paired-bar detail per inference group
- trait_amplification_summary.json: all correlations, deltas, CIs
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from . import EvalContext, EvalModule
from .content_biases import _load_inferences, _unnest_text_inferences

MIN_USER_POSTS = 20
N_BOOTSTRAP = 500
ALPHA = 0.05

_GROUP_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_eligible_users(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only users with >= MIN_USER_POSTS posts and at least 1 pos + 1 neg."""
    eligible = (
        df.group_by("did")
        .agg(pl.len().alias("n"), pl.col("y_true").sum().alias("n_pos"))
        .filter(
            (pl.col("n") >= MIN_USER_POSTS)
            & (pl.col("n_pos") >= 1)
            & (pl.col("n_pos") < pl.col("n"))
        )
        .select("did")
    )
    return df.join(eligible, on="did", how="semi")


def _demean_within_user(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("y_true") - pl.col("y_true").mean().over("did")).alias("y_true_c"),
        (pl.col("y_pred_proba") - pl.col("y_pred_proba").mean().over("did")).alias("y_pred_c"),
    )


# key = "group::label", value = (rho_true, rho_pred, delta)
CorrelationResults = Dict[str, Tuple[float, float, float]]


def _compute_all_correlations(
    y_true_c: np.ndarray,
    y_pred_c: np.ndarray,
    trait_arrays: Dict[str, np.ndarray],
    finite_masks: Dict[str, np.ndarray],
) -> CorrelationResults:
    """Point estimates for every trait across all groups."""
    results: CorrelationResults = {}
    for key, vals in trait_arrays.items():
        mask = finite_masks[key]
        if mask.sum() < 10:
            continue
        rt, _ = spearmanr(y_true_c[mask], vals[mask])
        rp, _ = spearmanr(y_pred_c[mask], vals[mask])
        results[key] = (float(rt), float(rp), float(rp - rt))
    return results


def _bootstrap_cis(
    user_ids: np.ndarray,
    user_to_rows: Dict[Any, np.ndarray],
    y_true_c: np.ndarray,
    y_pred_c: np.ndarray,
    trait_arrays: Dict[str, np.ndarray],
    finite_masks: Dict[str, np.ndarray],
    valid_keys: set[str],
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float, float, float, float]]:
    """Single bootstrap pass across ALL traits.

    Returns {key: (rt_lo, rt_hi, rp_lo, rp_hi, delta_lo, delta_hi)}.
    """
    rng = np.random.default_rng(seed)
    n_users = len(user_ids)

    boot_rt: Dict[str, List[float]] = defaultdict(list)
    boot_rp: Dict[str, List[float]] = defaultdict(list)
    boot_d: Dict[str, List[float]] = defaultdict(list)

    for _ in range(n_bootstrap):
        sampled = rng.choice(user_ids, size=n_users, replace=True)
        idx = np.concatenate([user_to_rows[u] for u in sampled])
        yt = y_true_c[idx]
        yp = y_pred_c[idx]

        for key in valid_keys:
            tv = trait_arrays[key][idx]
            m = finite_masks[key][idx]
            if m.sum() < 10:
                continue
            rt, _ = spearmanr(yt[m], tv[m])
            rp, _ = spearmanr(yp[m], tv[m])
            boot_rt[key].append(rt)
            boot_rp[key].append(rp)
            boot_d[key].append(rp - rt)

    lo_q = alpha / 2 * 100
    hi_q = (1 - alpha / 2) * 100
    cis = {}
    for key in valid_keys:
        if key not in boot_d or len(boot_d[key]) < 10:
            continue
        cis[key] = (
            float(np.percentile(boot_rt[key], lo_q)),
            float(np.percentile(boot_rt[key], hi_q)),
            float(np.percentile(boot_rp[key], lo_q)),
            float(np.percentile(boot_rp[key], hi_q)),
            float(np.percentile(boot_d[key], lo_q)),
            float(np.percentile(boot_d[key], hi_q)),
        )
    return cis


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_scatter(
    group_results: Dict[str, Dict[str, Tuple[float, float, float]]],
    cis: Dict[str, Tuple],
    group_color_map: Dict[str, str],
    out_dir: Path,
) -> Path:
    """rho_true (x) vs rho_pred (y), one dot per trait, colored by group."""
    fig, ax = plt.subplots(figsize=(7, 7))

    for gname, traits in group_results.items():
        color = group_color_map[gname]
        xs, ys = [], []
        xel, xeh, yel, yeh = [], [], [], []
        for label, (rt, rp, _) in traits.items():
            key = f"{gname}::{label}"
            if key not in cis:
                continue
            ci = cis[key]
            xs.append(rt); ys.append(rp)
            xel.append(rt - ci[0]); xeh.append(ci[1] - rt)
            yel.append(rp - ci[2]); yeh.append(ci[3] - rp)

        if not xs:
            continue
        ax.errorbar(
            xs, ys, xerr=[xel, xeh], yerr=[yel, yeh],
            fmt="o", ms=4, color=color, ecolor=color, elinewidth=0.5,
            capsize=0, alpha=0.7, label=gname.replace("_", " "),
        )

    lims = list(ax.get_xlim()) + list(ax.get_ylim())
    lo, hi = min(lims), max(lims)
    margin = (hi - lo) * 0.05
    lo -= margin; hi += margin
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.7, alpha=0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("ρ(actual preference, trait)  [within-user]")
    ax.set_ylabel("ρ(predicted preference, trait)  [within-user]")
    ax.set_title("Trait Amplification: Predicted vs Actual")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=0.3)
    plt.tight_layout()

    path = out_dir / "amplification_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_group_bars(
    group_name: str,
    traits: Dict[str, Tuple[float, float, float]],
    cis: Dict[str, Tuple],
    out_dir: Path,
) -> Path:
    """Paired horizontal bars: rho_true (gray) vs rho_pred (colored)."""
    labels = sorted(traits, key=lambda k: abs(traits[k][2]), reverse=True)
    n = len(labels)

    rt_vals = [traits[k][0] for k in labels]
    rp_vals = [traits[k][1] for k in labels]

    def _err(label_list, pos, val_list):
        lo_list, hi_list = [], []
        for lab, v in zip(label_list, val_list):
            key = f"{group_name}::{lab}"
            if key in cis:
                ci = cis[key]
                lo_list.append(v - ci[pos])
                hi_list.append(ci[pos + 1] - v)
            else:
                lo_list.append(0); hi_list.append(0)
        return [lo_list, hi_list]

    bar_h = 0.35
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * n)))
    y_pos = np.arange(n)

    ax.barh(
        y_pos - bar_h / 2, rt_vals, height=bar_h,
        color="#999999", edgecolor="white", linewidth=0.5, label="actual (ρ_true)",
        xerr=_err(labels, 0, rt_vals),
        error_kw=dict(ecolor="#555555", capsize=1.5, linewidth=0.6),
    )
    pred_colors = ["#4878CF" if v >= 0 else "#D65F5F" for v in rp_vals]
    ax.barh(
        y_pos + bar_h / 2, rp_vals, height=bar_h,
        color=pred_colors, edgecolor="white", linewidth=0.5, label="predicted (ρ_pred)",
        xerr=_err(labels, 2, rp_vals),
        error_kw=dict(ecolor="#333333", capsize=1.5, linewidth=0.6),
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Spearman ρ  [within-user demeaned]")
    mean_abs_d = np.mean([abs(traits[k][2]) for k in labels])
    ax.set_title(f"{group_name.replace('_', ' ').title()}   (mean |δ| = {mean_abs_d:.4f})")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()

    path = out_dir / f"{group_name}_amplification.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class TraitAmplificationModule(EvalModule):
    name = "trait_amplification"
    description = "Measures model amplification/suppression of NLP content traits vs actual user preferences"

    def run(self, ctx: EvalContext) -> Dict[str, Any]:
        out_dir = self.get_output_dir(ctx)
        run_dir = ctx.config.get("run_dir")
        if run_dir is None:
            return {"skipped": True, "reason": "run_dir not in eval config"}

        try:
            inferences_lf = _load_inferences(Path(run_dir))
        except FileNotFoundError as e:
            return {"skipped": True, "reason": str(e)}

        # --- Full holdout, filter to eligible users ---
        preds = pl.from_pandas(ctx.predictions_df)
        n_users_total = preds["did"].n_unique()
        preds = _filter_eligible_users(preds)
        n_users_eligible = preds["did"].n_unique()
        if n_users_eligible < 5:
            return {"skipped": True, "reason": f"only {n_users_eligible} eligible users"}

        # --- Join to inferences, demean, unnest ---
        joined = (
            inferences_lf
            .join(preds.lazy(), left_on="at_uri", right_on="post_id", how="inner")
            .collect()
        )
        n_posts_matched = len(joined)
        if n_posts_matched < 50:
            return {"skipped": True, "reason": f"only {n_posts_matched} posts matched inferences"}

        joined = _demean_within_user(joined)
        flat, group_names = _unnest_text_inferences(joined)

        # --- Pre-extract all numpy arrays (keyed as "group::label") ---
        y_true_c = flat["y_true_c"].to_numpy()
        y_pred_c = flat["y_pred_c"].to_numpy()
        user_col = flat["did"].to_numpy()

        user_ids = np.unique(user_col)
        user_to_rows: Dict[Any, np.ndarray] = {
            u: np.where(user_col == u)[0] for u in user_ids
        }

        trait_arrays: Dict[str, np.ndarray] = {}
        finite_masks: Dict[str, np.ndarray] = {}
        group_labels: Dict[str, List[str]] = {}

        for gname in group_names:
            gdf = flat.select(gname).unnest(gname)
            cols = gdf.columns
            group_labels[gname] = cols
            for col in cols:
                key = f"{gname}::{col}"
                arr = gdf[col].to_numpy()
                trait_arrays[key] = arr
                finite_masks[key] = np.isfinite(arr)

        # --- Point estimates ---
        all_corrs = _compute_all_correlations(y_true_c, y_pred_c, trait_arrays, finite_masks)

        # --- Single bootstrap pass across all traits ---
        all_cis = _bootstrap_cis(
            user_ids, user_to_rows, y_true_c, y_pred_c,
            trait_arrays, finite_masks,
            valid_keys=set(all_corrs.keys()),
        )

        # --- Partition results back by group ---
        group_color_map = {g: _GROUP_COLORS[i % len(_GROUP_COLORS)] for i, g in enumerate(group_names)}
        group_results: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
        plot_paths: list[str] = []

        for gname in group_names:
            traits: Dict[str, Tuple[float, float, float]] = {}
            for label in group_labels.get(gname, []):
                key = f"{gname}::{label}"
                if key in all_corrs:
                    traits[label] = all_corrs[key]
            if not traits:
                continue
            group_results[gname] = traits
            path = _plot_group_bars(gname, traits, all_cis, out_dir)
            plot_paths.append(str(path))

        if group_results:
            scatter_path = _plot_scatter(group_results, all_cis, group_color_map, out_dir)
            plot_paths.insert(0, str(scatter_path))

        # --- Summary JSON ---
        groups_json: Dict[str, Any] = {}
        all_abs_deltas: list[float] = []
        for gname, traits in group_results.items():
            gdict: Dict[str, Any] = {}
            for label, (rt, rp, delta) in traits.items():
                all_abs_deltas.append(abs(delta))
                entry: Dict[str, float] = {"rho_true": rt, "rho_pred": rp, "delta": delta}
                key = f"{gname}::{label}"
                if key in all_cis:
                    ci = all_cis[key]
                    entry.update({
                        "rho_true_ci_lo": ci[0], "rho_true_ci_hi": ci[1],
                        "rho_pred_ci_lo": ci[2], "rho_pred_ci_hi": ci[3],
                        "delta_ci_lo": ci[4], "delta_ci_hi": ci[5],
                    })
                gdict[label] = entry
            groups_json[gname] = gdict

        summary = {
            "n_users_total": n_users_total,
            "n_users_eligible": n_users_eligible,
            "min_user_posts": MIN_USER_POSTS,
            "n_posts_matched": n_posts_matched,
            "n_bootstrap": N_BOOTSTRAP,
            "mean_abs_amplification": float(np.mean(all_abs_deltas)) if all_abs_deltas else 0.0,
            "groups": groups_json,
        }
        self.save_json(summary, out_dir / "trait_amplification_summary.json")

        return {
            "n_users_eligible": n_users_eligible,
            "n_posts_matched": n_posts_matched,
            "groups_plotted": len(group_results),
            "mean_abs_amplification": summary["mean_abs_amplification"],
            "plot_paths": plot_paths,
        }
