#!/usr/bin/env python3

"""
Stage 3: Relevel users using topic discovery + (optional) uniform-mixture-balanced selection.

Inputs:
- embedding_bundle_*.pkl from Stage 2 (featurize)

Outputs under <run_dir>/relevel/<timestamp>/:
- topic_model.pkl (MiniBatchKMeans)
- topic_pca.pkl (optional PCA)
- user_topic_mixtures.parquet
- retained_users.json (if relevel selection applied)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from utils.pipeline.core import new_stage_timestamp_dir, select_prior_output
from utils.helpers import (
    discover_topics,
    compute_user_topic_mixtures,
    relevel_uniform_mixture,
)
import time


def run(context, args) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '03_relevel')

    # Locate embedding bundle from prior featurize stage
    prior_featurize = select_prior_output(run_dir, '02_featurize', use_latest=context.use_latest, prior_path=context.prior_outputs.get('02_featurize'))
    if prior_featurize is None:
        raise FileNotFoundError("Featurize output not found. Run Stage 2 first or provide --prior-output-featurize.")
    bundle_candidates = sorted(prior_featurize.glob('embedding_bundle_*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not bundle_candidates:
        raise FileNotFoundError(f"No embedding_bundle_*.pkl found under {prior_featurize}")
    bundle_path = bundle_candidates[0]

    # Load bundle
    import pickle
    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)
    posts_emb_df: pd.DataFrame = bundle['posts_emb_df']
    likes_df: pd.DataFrame = bundle['likes_df']
    join_like: str = str(bundle['join_like'])
    join_post: str = str(bundle['join_post'])

    # Eligibility (for mixtures; selection can be applied later)
    available_posts = set(posts_emb_df[join_post].astype(str).unique())
    likes_df_local = likes_df.copy()
    if join_like not in likes_df_local.columns:
        raise KeyError(f"likes_df missing join_like column: {join_like}")
    likes_df_local[join_like] = likes_df_local[join_like].astype(str)
    likes_joinable = likes_df_local[likes_df_local[join_like].isin(available_posts)]

    # Topic discovery
    global_topic_k = int(getattr(args, 'global_topic_k', 20))
    random_seed = int(getattr(args, 'random_seed', 42))
    t0 = time.time()
    artifacts = discover_topics(posts_emb_df, likes_joinable, join_like, join_post, global_topic_k=global_topic_k, random_seed=random_seed)
    if artifacts.topic_model is None:
        raise RuntimeError("Topic discovery unavailable (scikit-learn missing or no joinable likes)")

    # Compute mixtures
    mixtures = compute_user_topic_mixtures(artifacts, posts_emb_df, likes_joinable, join_like, join_post)
    if mixtures is None or mixtures.empty:
        raise RuntimeError("Failed to compute user topic mixtures")

    # Save mixtures
    mixtures_path = out_dir / 'user_topic_mixtures.parquet'
    mixtures.to_parquet(mixtures_path, index=True)

    # Optional selection
    relevel_strategy = str(getattr(args, 'relevel_strategy', 'uniform_mixture_balanced'))
    relevel_alpha = float(getattr(args, 'relevel_alpha', 0.35))
    relevel_min_users_per_topic = int(getattr(args, 'relevel_min_users_per_topic', 0))

    retained_users_path = None
    if relevel_strategy == 'uniform_mixture_balanced' and artifacts.global_topic_k:
        # Eligible users based on min likes per user
        min_likes_per_user = int(getattr(args, 'min_likes_per_user', 4))
        counts = likes_joinable.groupby('did', observed=True)[join_like].nunique().astype(int)
        eligible_users = counts[counts >= min_likes_per_user].index.astype(str).tolist()
        kept_users = relevel_uniform_mixture(
            users=eligible_users,
            user_topic_probs=mixtures,
            global_topic_k=int(artifacts.global_topic_k),
            alpha=float(relevel_alpha),
            min_users_per_topic=int(relevel_min_users_per_topic),
            random_seed=random_seed,
        )
        retained_users_path = out_dir / 'retained_users.json'
        with open(retained_users_path, 'w') as f:
            json.dump({'retained_users': kept_users}, f, indent=2)

    # Save topic artifacts
    topic_model_path = out_dir / 'topic_model.pkl'
    with open(topic_model_path, 'wb') as f:
        pickle.dump(artifacts.topic_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    pca_model_path = None
    if artifacts.pca_model is not None:
        pca_model_path = out_dir / 'topic_pca.pkl'
        with open(pca_model_path, 'wb') as f:
            pickle.dump(artifacts.pca_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Stage info
    info_lines = [
        f"stage: relevel",
        f"runtime_seconds: {time.time()-t0:.2f}",
        f"settings: global_topic_k={global_topic_k}, relevel_strategy={relevel_strategy}, relevel_alpha={relevel_alpha}, relevel_min_users_per_topic={relevel_min_users_per_topic}",
        f"inputs: embedding_bundle",
        f"N_posts_emb: {len(posts_emb_df)}",
        f"N_likes_joinable: {len(likes_joinable)}",
        f"N_users_mixtures: {len(mixtures)}",
        f"N_retained_users: {len(kept_users) if 'kept_users' in locals() else 0}",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    return {
        'output_dir': out_dir,
        'artifacts': {
            'mixtures_path': str(mixtures_path),
            **({'retained_users_path': str(retained_users_path)} if retained_users_path else {}),
            'topic_model_path': str(topic_model_path),
            **({'topic_pca_path': str(pca_model_path)} if pca_model_path else {}),
            'embedding_bundle_path': str(bundle_path.resolve()),
        }
    }



