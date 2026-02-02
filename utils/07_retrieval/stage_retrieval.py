#!/usr/bin/env python3

"""
Stage 7: Retrieve top-k posts for users via ANN + cross-encoder reranking.

This stage implements a two-stage retrieval pipeline:
1. Fast candidate retrieval using Approximate Nearest Neighbor (ANN) search
2. Precise reranking using a cross-encoder transformer

Inputs:
- Trained two-tower model checkpoint from Stage 5
- embedding_bundle_*.pkl from Stage 2
- user_splits.json from Stage 4

Outputs under <run_dir>/retrieval/<timestamp>/:
- user_recommendations.parquet (user_id, ranked post_ids with scores)
- retrieval_metrics.json (recall@k, precision@k, NDCG@k)
- ann_index/ (saved FAISS/Annoy index for post embeddings)
- stage_info.txt
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from utils.pipeline.core import new_stage_timestamp_dir, select_prior_output, Context
from utils.helpers import get_stage_logger, log_operation_start, get_device


def run(context: Context, args) -> Dict[str, Any]:
    """
    Execute Stage 7: Top-k retrieval with ANN + cross-encoder reranking.
    
    Pipeline:
    1. Load trained two-tower model
    2. Encode all candidate posts once (cache in ANN index)
    3. For each user:
       a. Encode user from history
       b. ANN search for top-N candidates (N >> k, e.g., N=1000)
       c. Cross-encoder rerank candidates to final top-k (e.g., k=50)
    4. Save recommendations and metrics
    """
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '07_retrieval')
    
    # Initialize logger
    logger = get_stage_logger('STAGE_07_RETRIEVAL', log_file=out_dir / 'stage.log')
    t0 = time.time()
    
    # =========================================================================
    # Step 1: Resolve assets (model, bundle, splits)
    # =========================================================================
    log_operation_start('Resolve assets (model, bundle, splits)', 'STAGE_07_RETRIEVAL', logger)
    model_path, bundle_path, splits_path = _resolve_assets(run_dir, context, args)
    device = get_device(args.device)
    
    # Retrieval parameters
    top_n_candidates = int(getattr(args, 'retrieval_top_n', 1000))  # ANN candidates
    top_k_final = int(getattr(args, 'retrieval_top_k', 50))  # Final reranked results
    use_cross_encoder = bool(getattr(args, 'use_cross_encoder', True))
    ann_index_type = str(getattr(args, 'ann_index_type', 'faiss'))  # 'faiss' or 'annoy'
    
    # =========================================================================
    # Step 2: Load model and embedding bundle
    # =========================================================================
    log_operation_start('Load two-tower model and embedding bundle', 'STAGE_07_RETRIEVAL', logger)
    model, checkpoint = _load_model(model_path, device)
    bundle = _load_bundle(bundle_path)
    
    posts_emb_df = bundle['posts_emb_df']
    likes_df = bundle['likes_df']
    join_like = bundle['join_like']
    join_post = bundle['join_post']
    embedding_dim = bundle['embedding_dim']
    
    # =========================================================================
    # Step 3: Build ANN index for all post embeddings
    # =========================================================================
    log_operation_start(f'Build ANN index ({ann_index_type}) for {len(posts_emb_df)} posts', 'STAGE_07_RETRIEVAL', logger)
    from .ann_retrieval import build_ann_index, encode_all_posts
    
    # Encode all posts through post tower
    post_ids, post_embeddings_encoded = encode_all_posts(
        model=model,
        posts_emb_df=posts_emb_df,
        join_post=join_post,
        embedding_dim=embedding_dim,
        device=device,
        batch_size=1024,
    )
    
    # Build ANN index
    ann_index = build_ann_index(
        embeddings=post_embeddings_encoded,
        index_type=ann_index_type,
        shared_dim=model.shared_dim,
    )
    
    # Save ANN index
    ann_index_dir = out_dir / 'ann_index'
    ann_index_dir.mkdir(exist_ok=True)
    ann_index.save(str(ann_index_dir / f'{ann_index_type}_index'))
    np.save(ann_index_dir / 'post_ids.npy', post_ids)
    
    # =========================================================================
    # Step 4: Load user splits and build user histories
    # =========================================================================
    log_operation_start('Load user splits and build histories', 'STAGE_07_RETRIEVAL', logger)
    with open(splits_path) as f:
        splits = json.load(f)
    
    # TODO: Choose which users to generate recommendations for
    # Options: train_users, val_users, holdout_users, or all
    target_users = splits['holdout_users']  # Example: generate for holdout users
    
    if len(target_users) == 0:
        logger.warning("No target users found. Skipping retrieval.")
        return {
            'output_dir': out_dir,
            'artifacts': {
                'recommendations_path': None,
                'metrics_path': None,
            }
        }
    
    # Build user histories (liked posts before some cutoff timestamp)
    user_histories = _build_user_histories(
        target_users=target_users,
        likes_df=likes_df,
        posts_emb_df=posts_emb_df,
        join_like=join_like,
        join_post=join_post,
        embedding_dim=embedding_dim,
        max_history_len=int(getattr(args, 'max_history_len', 20)),
    )
    
    # =========================================================================
    # Step 5: Generate recommendations for each user
    # =========================================================================
    log_operation_start(f'Generate recommendations for {len(user_histories)} users', 'STAGE_07_RETRIEVAL', logger)
    
    recommendations = []
    
    for user_id, history_data in user_histories.items():
        # Step 5a: Encode user from history
        user_emb = _encode_user(
            model=model,
            history_embeddings=history_data['history_embeddings'],
            device=device,
        )
        
        # Step 5b: ANN search for top-N candidates
        candidate_post_ids, candidate_scores = ann_index.search(
            query=user_emb,
            k=top_n_candidates,
        )
        
        # Map indices back to post IDs
        candidate_post_ids = [post_ids[idx] for idx in candidate_post_ids[0]]
        
        # Step 5c: Cross-encoder reranking (if enabled)
        if use_cross_encoder and top_k_final < top_n_candidates:
            from .cross_encoder_reranker import rerank_with_cross_encoder
            
            reranked_post_ids, reranked_scores = rerank_with_cross_encoder(
                user_id=user_id,
                user_history_data=history_data,
                candidate_post_ids=candidate_post_ids[:top_n_candidates],  # May trim if needed
                posts_emb_df=posts_emb_df,
                join_post=join_post,
                device=device,
                top_k=top_k_final,
            )
            
            final_post_ids = reranked_post_ids
            final_scores = reranked_scores
        else:
            # No reranking: use ANN scores directly
            final_post_ids = candidate_post_ids[:top_k_final]
            final_scores = candidate_scores[0][:top_k_final]
        
        recommendations.append({
            'user_id': user_id,
            'recommended_post_ids': final_post_ids,
            'scores': final_scores.tolist() if isinstance(final_scores, np.ndarray) else final_scores,
        })
    
    # =========================================================================
    # Step 6: Save recommendations
    # =========================================================================
    log_operation_start('Save user recommendations', 'STAGE_07_RETRIEVAL', logger)
    recommendations_path = out_dir / 'user_recommendations.parquet'
    
    # Flatten to DataFrame format
    recs_data = []
    for rec in recommendations:
        for rank, (post_id, score) in enumerate(zip(rec['recommended_post_ids'], rec['scores'])):
            recs_data.append({
                'user_id': rec['user_id'],
                'rank': rank + 1,
                'post_id': post_id,
                'score': float(score),
            })
    
    recs_df = pd.DataFrame(recs_data)
    recs_df.to_parquet(recommendations_path, index=False)
    
    # =========================================================================
    # Step 7: Compute retrieval metrics (if ground truth available)
    # =========================================================================
    log_operation_start('Compute retrieval metrics', 'STAGE_07_RETRIEVAL', logger)
    metrics = _compute_retrieval_metrics(
        recommendations=recommendations,
        likes_df=likes_df,
        user_histories=user_histories,
        k_values=[10, 20, 50],
    )
    
    metrics_path = out_dir / 'retrieval_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # =========================================================================
    # Step 8: Write stage info
    # =========================================================================
    runtime = time.time() - t0
    info_lines = [
        f"stage: retrieval",
        f"runtime_seconds: {runtime:.2f}",
        f"settings: top_n_candidates={top_n_candidates}, top_k_final={top_k_final}, use_cross_encoder={use_cross_encoder}",
        f"ann_index_type: {ann_index_type}",
        f"num_posts_indexed: {len(post_ids)}",
        f"num_users: {len(user_histories)}",
        f"avg_recommendations_per_user: {len(recs_df) / len(user_histories) if len(user_histories) > 0 else 0:.1f}",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')
    
    logger.info(f"Stage 7 complete (runtime: {runtime:.2f}s)")
    
    return {
        'output_dir': out_dir,
        'artifacts': {
            'recommendations_path': str(recommendations_path),
            'metrics_path': str(metrics_path),
            'ann_index_dir': str(ann_index_dir),
        }
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _resolve_assets(run_dir: Path, context: Context, args) -> tuple[str, str, str]:
    """Locate model checkpoint, embedding bundle, and user splits."""
    # Find model checkpoint from Stage 5
    prior_train = select_prior_output(
        run_dir, 
        '05_train_two_tower',  # or '05_train' for MLP
        use_latest=context.use_latest,
        prior_path=context.prior_outputs.get('05_train_two_tower')
    )
    if prior_train is None:
        raise FileNotFoundError("Training output not found (Stage 5).")
    
    checkpoint_candidates = list(prior_train.glob('checkpoints/two_tower_best.pth'))
    if not checkpoint_candidates:
        checkpoint_candidates = list(prior_train.glob('checkpoints/two_tower_*.pth'))
    if not checkpoint_candidates:
        raise FileNotFoundError(f"No model checkpoint found in {prior_train / 'checkpoints'}")
    model_path = str(checkpoint_candidates[0])
    
    # Find embedding bundle from Stage 2
    prior_featurize = select_prior_output(run_dir, '02_featurize', use_latest=context.use_latest)
    if prior_featurize is None:
        raise FileNotFoundError("Featurize output not found (Stage 2).")
    bundle_candidates = sorted(prior_featurize.glob('embedding_bundle_*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not bundle_candidates:
        raise FileNotFoundError(f"No embedding bundle found in {prior_featurize}")
    bundle_path = str(bundle_candidates[0])
    
    # Find user splits from Stage 4
    prior_split = select_prior_output(run_dir, '04_split', use_latest=context.use_latest)
    if prior_split is None:
        raise FileNotFoundError("Split output not found (Stage 4).")
    splits_path = prior_split / 'user_splits.json'
    if not splits_path.exists():
        raise FileNotFoundError(f"user_splits.json not found in {prior_split}")
    
    return model_path, bundle_path, str(splits_path)


def _load_model(model_path: str, device: str):
    """Load trained two-tower model from checkpoint."""
    import torch
    from utils.05_train.stage_train_two_tower import TwoTowerEngagement
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model hyperparameters from checkpoint
    model = TwoTowerEngagement(
        post_embedding_dim=checkpoint.get('post_embedding_dim', 384),
        shared_dim=checkpoint.get('shared_dim', 128),
        user_hidden_dim=checkpoint.get('user_hidden_dim', 256),
        post_hidden_dim=checkpoint.get('post_hidden_dim', 256),
        num_attention_heads=checkpoint.get('num_attention_heads', 4),
        num_attention_layers=checkpoint.get('num_attention_layers', 2),
        max_history_len=checkpoint.get('max_history_len', 50),
        dropout_rate=checkpoint.get('dropout_rate', 0.1),
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def _load_bundle(bundle_path: str) -> Dict[str, Any]:
    """Load embedding bundle from Stage 2."""
    import pickle
    with open(bundle_path, 'rb') as f:
        return pickle.load(f)


def _build_user_histories(
    target_users: List[str],
    likes_df: pd.DataFrame,
    posts_emb_df: pd.DataFrame,
    join_like: str,
    join_post: str,
    embedding_dim: int,
    max_history_len: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Build user history sequences from their liked posts.
    
    TODO: Implement temporal split to separate history from held-out test items.
    For now, uses all liked posts as history (cheating for demonstration).
    """
    user_histories = {}
    
    # Get embedding column names
    emb_cols = [f"post_emb_{i}" for i in range(embedding_dim)]
    
    # Create lookup for post embeddings
    post_emb_lookup = {}
    for _, row in posts_emb_df.iterrows():
        pid = str(row[join_post])
        post_emb_lookup[pid] = row[emb_cols].values
    
    for user_id in target_users:
        user_likes = likes_df[likes_df['did'] == user_id]
        liked_post_ids = user_likes[join_like].astype(str).unique().tolist()
        
        # Get embeddings for liked posts
        history_embs = []
        for pid in liked_post_ids[:max_history_len]:  # Cap length
            if pid in post_emb_lookup:
                history_embs.append(post_emb_lookup[pid])
        
        if len(history_embs) > 0:
            user_histories[user_id] = {
                'history_embeddings': np.stack(history_embs),  # [seq_len, embedding_dim]
                'history_post_ids': liked_post_ids[:len(history_embs)],
            }
    
    return user_histories


def _encode_user(model, history_embeddings: np.ndarray, device: str) -> np.ndarray:
    """Encode user from history embeddings using the user tower."""
    import torch
    
    # Add batch dimension and create mask
    history_tensor = torch.tensor(history_embeddings, dtype=torch.float32, device=device).unsqueeze(0)  # [1, seq_len, D]
    history_mask = torch.ones(1, history_embeddings.shape[0], dtype=torch.bool, device=device)  # [1, seq_len]
    
    with torch.no_grad():
        user_emb = model.encode_user(history_tensor, history_mask)  # [1, shared_dim]
    
    return user_emb.cpu().numpy()  # [1, shared_dim]


def _compute_retrieval_metrics(
    recommendations: List[Dict],
    likes_df: pd.DataFrame,
    user_histories: Dict,
    k_values: List[int],
) -> Dict[str, Any]:
    """
    Compute retrieval metrics: Recall@k, Precision@k, NDCG@k.
    
    TODO: Implement proper evaluation with held-out test sets.
    This requires temporal splitting of likes into history vs. test.
    """
    # Placeholder metrics
    metrics = {
        'num_users': len(recommendations),
        'avg_recommendations_per_user': np.mean([len(rec['recommended_post_ids']) for rec in recommendations]),
    }
    
    for k in k_values:
        # TODO: Compute Recall@k = (relevant items in top-k) / (total relevant items)
        # TODO: Compute Precision@k = (relevant items in top-k) / k
        # TODO: Compute NDCG@k = normalized discounted cumulative gain
        
        metrics[f'recall@{k}'] = 0.0  # Placeholder
        metrics[f'precision@{k}'] = 0.0  # Placeholder
        metrics[f'ndcg@{k}'] = 0.0  # Placeholder
    
    return metrics
