#!/usr/bin/env python3
"""Score one remedy's synthetic feed on a fixed baseline cohort.

Unlike a native Stage-5 evaluation, this runner fixes *all* cohort-defining
inputs to the baseline substrate: random-post pool, user histories, holdout
users, positive examples, and Lorenz-50 power-user split.  Only the checkpoint
varies.  Its D1/D2 outputs can therefore be joined to fixed-cohort paired AUC
without native-holdout composition artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

synthetic = importlib.import_module("utils.05_evaluate.evals.synthetic_feed")


def load_artifact_helpers():
    """Load sibling artifact helpers without making ``ops`` a package."""
    path = Path(__file__).with_name("generate_paper_quality_artifacts.py")
    spec = importlib.util.spec_from_file_location("paper_quality_artifacts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_for_cell(cell: Path, model_type: str) -> Path:
    pattern = synthetic._MODEL_TYPE_TO_CKPT_GLOB.get(model_type)
    if pattern is None:
        raise ValueError(f"Unsupported model type {model_type!r}")
    candidates = sorted(
        (cell / "checkpoints").glob(pattern),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint matching {pattern!r} in {cell / 'checkpoints'}")
    rich = [
        candidate
        for candidate in candidates
        if "_best" not in candidate.stem and "_weights" not in candidate.stem
    ]
    return rich[0] if rich else candidates[0]


def reconstruct_pool(
    baseline_run_dir: Path,
    pool_at_uris: list[str] | None,
    n_pool: int,
    artifacts,
) -> pl.DataFrame:
    posts = pl.read_parquet(artifacts.stage_file(baseline_run_dir, "posts_core_"))
    inferences = pl.read_parquet(artifacts.stage_file(baseline_run_dir, "inferences_core_"))
    eligible = (
        posts.filter(pl.col("in_random_sample"))
        .join(inferences.select("at_uri", "inferences"), on="at_uri", how="inner")
    )
    if pool_at_uris is None:
        # Historical evaluations predate the persisted pool identity. Their
        # evaluator sampled this exact eligible frame with seed 42; reproduce
        # it once, then persist the resolved URIs in this fixed-cohort output.
        # The resulting URI list is the durable pool identity for every cell.
        return eligible.sample(n=n_pool, seed=42)
    pool_order = pl.DataFrame({"at_uri": pool_at_uris, "_pool_order": range(len(pool_at_uris))})
    pool = pool_order.join(eligible, on="at_uri", how="left").sort("_pool_order")
    if pool.height != len(pool_at_uris) or pool["inferences"].null_count() > 0:
        raise RuntimeError("Baseline synthetic-feed pool cannot be reconstructed.")
    return pool.drop("_pool_order")


def trait_decomposition(
    pool_values: np.ndarray,
    liked_values: np.ndarray,
    liked_dids: np.ndarray,
    topk: dict[str, list[int]],
    dids: list[str],
):
    """Return the finite-value decomposition and its paired user identifiers."""
    actual = {}
    for did in dids:
        values = liked_values[liked_dids == did]
        values = values[np.isfinite(values)]
        if len(values) >= synthetic.MIN_USER_LIKES:
            actual[did] = values
    feed = {}
    for did, indices in topk.items():
        if did not in actual:
            continue
        values = pool_values[np.asarray(indices, dtype=int)]
        values = values[np.isfinite(values)]
        if values.size:
            feed[did] = values
    result = synthetic._compute_trait_decomposition(pool_values, actual, feed, dids)
    if result is None:
        return None
    if not (
        np.isfinite(result.user_pref_std).all()
        and np.isfinite(result.model_amp_std).all()
        and np.isfinite(result.model_excess_std).all()
    ):
        raise RuntimeError("Fixed-cohort decomposition produced non-finite user statistics.")
    return result


def summary_row(result) -> dict[str, float | int]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cell-dir", required=True, type=Path)
    parser.add_argument("--baseline-run-dir", required=True, type=Path)
    parser.add_argument(
        "--baseline-eval-dir",
        required=True,
        type=Path,
        help="Baseline eval directory whose saved pool identities define the fixed pool.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Remedy checkpoint predictions already scored on the baseline substrate.",
    )
    parser.add_argument("--likes-core", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model-seed", required=True, type=int)
    parser.add_argument("--min-likes", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    artifacts = load_artifact_helpers()
    train_cell = args.train_cell_dir.resolve()
    baseline_run_dir = args.baseline_run_dir.resolve()
    predictions_path = args.predictions.resolve()
    checkpoint_path = checkpoint_for_cell(
        train_cell,
        json.loads((train_cell / "training_config.json").read_text()).get("model_type", "mlp"),
    )
    checkpoint = json.loads(
        (args.baseline_eval_dir / "synthetic_feed" / "synthetic_feed_topk.json").read_text()
    )
    predictions = pl.read_parquet(predictions_path).with_columns(pl.col("did").cast(pl.String))
    pool_at_uris = checkpoint.get("pool_at_uris")
    pool = reconstruct_pool(
        baseline_run_dir,
        pool_at_uris,
        int(checkpoint["n_pool"]),
        artifacts,
    )
    posts = pl.read_parquet(artifacts.stage_file(baseline_run_dir, "posts_core_"))
    inferences = pl.read_parquet(artifacts.stage_file(baseline_run_dir, "inferences_core_"))
    liked = (
        predictions.filter(pl.col("y_true") == 1)
        .select("did", "post_id")
        .join(posts, left_on="post_id", right_on="at_uri", how="inner")
        .join(
            inferences.select("at_uri", "inferences"),
            left_on="post_id",
            right_on="at_uri",
            how="inner",
        )
    )

    random_posts_lf, embeddings = synthetic._load_random_pool(baseline_run_dir)
    del random_posts_lf  # Pool identities are pinned above; only its embeddings are needed.
    history_lf = synthetic._load_user_histories(baseline_run_dir)
    model, ckpt = synthetic._load_model(checkpoint_path, args.device)
    holdout_dids = predictions["did"].unique().to_list()
    user_summaries = synthetic._compute_user_summaries(holdout_dids, history_lf, embeddings, ckpt)
    positive_counts = (
        predictions.filter(pl.col("y_true") == 1)
        .group_by("did")
        .len()
        .filter(pl.col("len") >= synthetic.MIN_USER_LIKES)
    )
    eligible_dids = [did for did in positive_counts["did"].to_list() if did in user_summaries]
    if len(eligible_dids) < 10:
        raise RuntimeError(f"Only {len(eligible_dids)} fixed-cohort users have history summaries.")

    pool_embeddings = embeddings[pool["emb_idx"].to_numpy()].copy()
    scores = synthetic._score_pool_for_users(
        model,
        {did: user_summaries[did] for did in eligible_dids},
        pool_embeddings,
        ckpt,
        args.device,
    )
    top_k = min(synthetic.TOP_K, pool.height)
    topk = {did: np.argsort(values)[-top_k:][::-1].tolist() for did, values in scores.items()}

    power = artifacts.lorenz50_power_users(
        pl.read_parquet(args.likes_core).select("did"),
        args.min_likes,
    )
    typical = sorted(set(topk) - power)
    pool_flat, groups = synthetic._unnest_text_inferences(pool)
    liked_flat, _ = synthetic._unnest_text_inferences(liked)
    liked_dids = liked_flat["did"].cast(pl.String).to_numpy()

    d1_rows: list[dict[str, object]] = []
    d1_user_rows: list[dict[str, object]] = []
    for group in groups:
        pool_group = pool_flat.select(group).unnest(group)
        liked_group = liked_flat.select(group).unnest(group)
        for trait in pool_group.columns:
            alias = artifacts.FOCAL_TRAITS.get((group, trait))
            if alias is None:
                continue
            result = trait_decomposition(
                pool_group[trait].to_numpy().astype(float),
                liked_group[trait].to_numpy().astype(float),
                liked_dids,
                topk,
                typical,
            )
            if result is not None:
                d1_rows.append(
                    {
                        "group": group,
                        "trait": trait,
                        "alias": alias,
                        "stratum": "typical",
                        **summary_row(result),
                    }
                )
                for did, pref, amp, excess in zip(
                    result.user_dids,
                    result.user_pref_std,
                    result.model_amp_std,
                    result.model_excess_std,
                ):
                    d1_user_rows.append(
                        {
                            "group": group,
                            "trait": trait,
                            "alias": alias,
                            "stratum": "typical",
                            "did": did,
                            "user_pref_abs": float(pref * result.pool_sd),
                            "model_amp_abs": float(amp * result.pool_sd),
                            "model_excess_abs": float(excess * result.pool_sd),
                        }
                    )

    pool_struct = artifacts.structural_features(pool).select(
        "word_count", "has_url", "has_hashtag", "hour_of_day_utc"
    )
    liked_struct = artifacts.structural_features(liked).select(
        "word_count", "has_url", "has_hashtag", "hour_of_day_utc"
    )
    d2_rows: list[dict[str, object]] = []
    d2_user_rows: list[dict[str, object]] = []
    for feature in pool_struct.columns:
        result = trait_decomposition(
            pool_struct[feature].to_numpy().astype(float),
            liked_struct[feature].to_numpy().astype(float),
            liked_dids,
            topk,
            sorted(topk),
        )
        if result is not None:
            d2_rows.append({"feature": feature, "stratum": "all", **summary_row(result)})
            for did, pref, amp, excess in zip(
                result.user_dids,
                result.user_pref_std,
                result.model_amp_std,
                result.model_excess_std,
            ):
                d2_user_rows.append(
                    {
                        "feature": feature,
                        "stratum": "all",
                        "did": did,
                        "user_pref_abs": float(pref * result.pool_sd),
                        "model_amp_abs": float(amp * result.pool_sd),
                        "model_excess_abs": float(excess * result.pool_sd),
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(d1_rows).write_parquet(args.out_dir / "fixed_cohort_typical_axis_a_focal.parquet")
    pl.DataFrame(d1_user_rows).write_parquet(
        args.out_dir / "fixed_cohort_typical_axis_a_user_level.parquet"
    )
    pl.DataFrame(d2_rows).write_parquet(args.out_dir / "fixed_cohort_negative_controls.parquet")
    pl.DataFrame(d2_user_rows).write_parquet(
        args.out_dir / "fixed_cohort_negative_controls_user_level.parquet"
    )
    (args.out_dir / "fixed_cohort_synthetic_feed_topk.json").write_text(
        json.dumps(
            {
                "baseline_pool_at_uris": pool["at_uri"].to_list(),
                "pool_identity_source": (
                    "saved_baseline_eval" if pool_at_uris is not None
                    else "historical_seed42_reconstruction"
                ),
                "top_k": top_k,
                "user_topk": topk,
            }
        )
        + "\n"
    )
    manifest = {
        "condition": args.condition,
        "model_seed": args.model_seed,
        "baseline_run_dir": str(baseline_run_dir),
        "baseline_eval_dir": str(args.baseline_eval_dir),
        "pool_identity_source": (
            "saved_baseline_eval" if pool_at_uris is not None
            else "historical_seed42_reconstruction"
        ),
        "train_cell_dir": str(train_cell),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "predictions": str(predictions_path),
        "predictions_sha256": sha256(predictions_path),
        "n_pool": pool.height,
        "n_scored_users": len(topk),
        "n_typical_users": len(typical),
        "n_power_users": len(set(topk) & power),
        "d1_rows": len(d1_rows),
        "d1_user_rows": len(d1_user_rows),
        "d2_rows": len(d2_rows),
        "d2_user_rows": len(d2_user_rows),
    }
    (args.out_dir / "fixed_cohort_bias_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
