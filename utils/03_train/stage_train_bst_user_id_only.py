#!/usr/bin/env python3

"""Stage 3 experiment: BST ranker using only post-liker user ID features."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from shared.input_data_helpers import get_padded_history_time_deltas
from utils.dataloaders import (
    BucketedBatchSampler,
    PostLikerEventLookup,
    POST_LIKER_USER_UNK_IDX,
    _build_post_liker_user_idx_by_did,
    _fill_post_liker_event_slice,
    _list_to_float_array,
    _list_to_int_array,
    _post_split_window_for_like_split,
    _resolve_prior,
    _timestamp_to_epoch_us,
    load_post_liker_event_artifacts,
)
from utils.helpers import (
    clear_cuda_memory,
    get_device,
    get_stage_logger,
    load_parquet_from_prior,
    log_operation_start,
    log_prior_stage_inputs,
    plot_training_history,
    set_random_seeds,
    validate_dataframe_schema,
)
from utils.matrix_ranking import (
    MatrixBatchScores,
    calc_baseline_rank_metrics_for_batch,
    empty_rank_metric_sums,
    evaluate_matrix_scorer,
    finalize_rank_metrics,
    finalize_zero_history_rank_metrics,
    log_zero_history_rank_metrics,
    rank_metric_sums_for_batch,
    stage_info_metric_lines,
    zero_history_rank_metric_sums_for_batch,
)
from utils.pipeline.core import Context
from utils.ranker_utilities import LinearPredictionHead


stage_train_bst_ranker = importlib.import_module("utils.03_train.stage_train_bst_ranker")
PostLikerUserPooler = stage_train_bst_ranker.PostLikerUserPooler

STAGE_LOG_NAME = "STAGE_03_TRAIN_BST_USER_ID_ONLY"


def _validate_time_delta_bucket_boundaries(boundaries_hours: Sequence[float]) -> tuple[float, ...]:
    boundaries = tuple(float(boundary) for boundary in boundaries_hours)
    if len(boundaries) == 0:
        raise ValueError("time delta bucket boundaries must not be empty")
    previous = 0.0
    for boundary in boundaries:
        if boundary <= 0.0:
            raise ValueError("time delta bucket boundaries must be positive")
        if boundary <= previous:
            raise ValueError("time delta bucket boundaries must be strictly increasing")
        previous = boundary
    return boundaries


def _get_post_liker_user_table_num_rows(user_idx_df: Optional[Any]) -> int:
    if user_idx_df is None or len(user_idx_df) == 0:
        return 2
    if "user_idx" not in user_idx_df.columns:
        raise ValueError("post-liker user_idx artifact is missing user_idx column")
    max_user_idx = user_idx_df["user_idx"].max()
    if max_user_idx is None:
        return 2
    return max(2, int(max_user_idx) + 1)


class UserIdOnlyBucketedEngagementDataset(Dataset):
    """Bucketed ranker dataset that never loads post content embeddings."""

    def __init__(
        self,
        likes_core_df: pl.DataFrame,
        posts_core_df: pl.DataFrame,
        history_df: pl.DataFrame,
        split: str,
        max_history_len: int,
        post_liker_event_lookup: PostLikerEventLookup,
        post_liker_user_idx_df: pl.DataFrame,
        max_post_liker_replay_events_per_post: int,
        bst_additional_batch_negatives: Optional[int] = None,
        seed: int = 0,
        logger: Optional[Any] = None,
    ):
        if max_history_len <= 0:
            raise ValueError("max_history_len must be positive")
        if max_post_liker_replay_events_per_post <= 0:
            raise ValueError("max_post_liker_replay_events_per_post must be positive")
        if bst_additional_batch_negatives is not None and bst_additional_batch_negatives <= 0:
            raise ValueError("bst_additional_batch_negatives must be positive when provided")

        self.split = str(split)
        self.max_history_len = int(max_history_len)
        self.post_liker_event_lookup = post_liker_event_lookup
        self.post_liker_user_idx_by_did = _build_post_liker_user_idx_by_did(post_liker_user_idx_df)
        self.max_post_liker_replay_events_per_post = int(max_post_liker_replay_events_per_post)
        self.bst_additional_batch_negatives = int(bst_additional_batch_negatives) if bst_additional_batch_negatives is not None else None
        self.seed = int(seed)

        likes_columns = ["did", "subject_uri", "split", "like_hour_bucket", "emb_idx"]
        posts_columns = ["at_uri", "in_random_sample", "negative_hour_bucket", "split_window", "emb_idx"]
        history_columns = ["did", "like_hour_bucket", "prior_emb_indices"]
        self.has_history_time_deltas = "prior_like_age_hours_at_bucket_start" in history_df.columns
        if self.has_history_time_deltas:
            history_columns.append("prior_like_age_hours_at_bucket_start")
        elif logger:
            logger.warning(
                "UserIdOnlyBucketedEngagementDataset history input is missing "
                "prior_like_age_hours_at_bucket_start; emitting zero history_time_deltas_hours"
            )

        validate_dataframe_schema(likes_core_df, dict.fromkeys(likes_columns, None))
        validate_dataframe_schema(posts_core_df, dict.fromkeys(posts_columns, None))
        validate_dataframe_schema(history_df, dict.fromkeys(history_columns, None))

        like_ordered_df = (
            likes_core_df
            .filter(pl.col("split") == self.split)
            .with_row_index(name="_like_order")
        )
        user_hour_df = (
            like_ordered_df
            .group_by(["did", "like_hour_bucket"])
            .agg([
                pl.col("subject_uri").sort_by("_like_order").alias("liked_post_ids"),
                pl.col("emb_idx").sort_by("_like_order").alias("liked_post_emb_indices"),
                pl.col("_like_order").min().alias("_first_like_order"),
            ])
            .sort("_first_like_order")
        )
        joined = (
            user_hour_df
            .join(
                history_df.select(history_columns),
                on=["did", "like_hour_bucket"],
                how="left",
                maintain_order="left",
            )
        )
        if logger:
            logger.info(f"  UserIdOnlyBucketedEngagementDataset('{self.split}'): {len(joined):,} user-hour rows")

        self.user_ids = joined["did"].to_list()
        self.target_user_indices = [
            self.post_liker_user_idx_by_did.get(str(user_id), POST_LIKER_USER_UNK_IDX)
            for user_id in self.user_ids
        ]
        self.like_hour_buckets = joined["like_hour_bucket"].to_list()
        self.liked_post_ids = joined["liked_post_ids"].to_list()
        self.liked_post_emb_indices = [
            _list_to_int_array(value)
            for value in joined["liked_post_emb_indices"].to_list()
        ]
        self.prior_emb_indices = [
            _list_to_int_array(value)
            for value in joined["prior_emb_indices"].to_list()
        ]
        if self.has_history_time_deltas:
            self.prior_like_age_hours_at_bucket_start = [
                _list_to_float_array(value)
                for value in joined["prior_like_age_hours_at_bucket_start"].to_list()
            ]
        else:
            self.prior_like_age_hours_at_bucket_start = [
                np.array([], dtype=np.float32)
                for _ in self.prior_emb_indices
            ]

        self.row_indices_by_bucket: Dict[Any, List[int]] = {}
        for row_idx, bucket in enumerate(self.like_hour_buckets):
            self.row_indices_by_bucket.setdefault(bucket, []).append(row_idx)

        post_split_window = _post_split_window_for_like_split(self.split)
        sampled_posts_df = posts_core_df.filter(
            (pl.col("split_window") == post_split_window)
            & pl.col("in_random_sample")
            & pl.col("negative_hour_bucket").is_not_null()
        )
        self.sampled_posts_by_bucket: Dict[Any, List[Dict[str, Any]]] = {}
        for row in sampled_posts_df.iter_rows(named=True):
            post = {
                "post_id": row["at_uri"],
                "emb_idx": int(row["emb_idx"]),
            }
            self.sampled_posts_by_bucket.setdefault(row["negative_hour_bucket"], []).append(post)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: Any) -> Dict[str, Any]:
        if isinstance(idx, tuple):
            row_idx, epoch = idx
        else:
            row_idx = idx
            epoch = 0
        row_idx = int(row_idx)
        return {
            "row_idx": row_idx,
            "bucket": self.like_hour_buckets[row_idx],
            "user_id": self.user_ids[row_idx],
            "epoch": int(epoch),
        }

    def _padded_history_mask_for_row(self, row_idx: int) -> torch.Tensor:
        hist_indices = self.prior_emb_indices[row_idx]
        seq_len = min(len(hist_indices), self.max_history_len)
        mask = np.zeros((self.max_history_len,), dtype=np.bool_)
        if seq_len > 0:
            mask[:seq_len] = True
        return torch.from_numpy(mask)

    def _padded_time_deltas_for_row(self, row_idx: int) -> torch.Tensor:
        deltas = self.prior_like_age_hours_at_bucket_start[row_idx]
        padded = get_padded_history_time_deltas(deltas, self.max_history_len)
        return torch.from_numpy(padded)

    def _post_liker_features_for_emb_idx(
        self,
        emb_idx: int,
        target_time_us: int,
        cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = (int(emb_idx), int(target_time_us))
        if key not in cache:
            cache[key] = self.post_liker_event_lookup.liker_events_before(
                emb_idx=int(emb_idx),
                target_time_us=int(target_time_us),
                max_replay_events=self.max_post_liker_replay_events_per_post,
            )
        return cache[key]

    def _sample_candidate_posts_for_batch(
        self,
        row_indices: List[int],
        bucket: Any,
        epoch: int,
        candidate_to_idx: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        sampled_posts = [
            post
            for post in self.sampled_posts_by_bucket.get(bucket, [])
            if post["post_id"] not in candidate_to_idx
        ]
        if self.bst_additional_batch_negatives is None:
            return sampled_posts
        if len(sampled_posts) <= self.bst_additional_batch_negatives:
            return sampled_posts

        sorted_row_indices = sorted(int(row_idx) for row_idx in row_indices)
        row_seed = sum((pos + 1) * (row_idx + 1) for pos, row_idx in enumerate(sorted_row_indices))
        rng = np.random.default_rng(self.seed + int(epoch) * max(len(self.user_ids), 1) + row_seed)
        selected_indices = sorted(rng.choice(len(sampled_posts), size=self.bst_additional_batch_negatives, replace=False).tolist())
        return [sampled_posts[idx] for idx in selected_indices]

    def collate_batch(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not items:
            raise ValueError("UserIdOnlyBucketedEngagementDataset.collate_batch received an empty batch")

        row_indices = [int(item["row_idx"]) for item in items]
        epochs = {int(item.get("epoch", 0)) for item in items}
        if len(epochs) != 1:
            raise ValueError("Bucketed batches must contain rows from exactly one sampling epoch")
        epoch = next(iter(epochs))
        bucket = self.like_hour_buckets[row_indices[0]]
        if any(self.like_hour_buckets[row_idx] != bucket for row_idx in row_indices):
            raise ValueError("Bucketed batches must contain rows from exactly one hour bucket")
        target_time_us = _timestamp_to_epoch_us(bucket)

        user_ids = [self.user_ids[row_idx] for row_idx in row_indices]
        user_to_batch_idx = {
            user_id: user_idx
            for user_idx, user_id in enumerate(user_ids)
        }

        history_mask_tensors = []
        time_delta_tensors = []
        history_user_indices_padded = np.zeros(
            (
                len(row_indices),
                self.max_history_len,
                self.max_post_liker_replay_events_per_post,
            ),
            dtype=np.int64,
        )
        history_time_gap_hours_padded = np.zeros(
            (
                len(row_indices),
                self.max_history_len,
                self.max_post_liker_replay_events_per_post,
            ),
            dtype=np.float32,
        )
        history_max_events = 0
        post_liker_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        for batch_row_idx, row_idx in enumerate(row_indices):
            history_mask_tensors.append(self._padded_history_mask_for_row(row_idx))
            time_delta_tensors.append(self._padded_time_deltas_for_row(row_idx))
            hist_indices = self.prior_emb_indices[row_idx]
            seq_len = min(len(hist_indices), self.max_history_len)
            for hist_pos in range(seq_len):
                user_indices, time_gap_hours = self._post_liker_features_for_emb_idx(
                    int(hist_indices[hist_pos]),
                    target_time_us,
                    post_liker_cache,
                )
                history_max_events = max(
                    history_max_events,
                    _fill_post_liker_event_slice(
                        user_indices,
                        time_gap_hours,
                        history_user_indices_padded[batch_row_idx, hist_pos],
                        history_time_gap_hours_padded[batch_row_idx, hist_pos],
                    ),
                )

        candidate_post_ids: List[str] = []
        candidate_emb_indices: List[int] = []
        candidate_to_idx: Dict[str, int] = {}

        def add_candidate(post_id: str, emb_idx: int) -> None:
            if post_id in candidate_to_idx:
                return
            candidate_to_idx[post_id] = len(candidate_post_ids)
            candidate_post_ids.append(post_id)
            candidate_emb_indices.append(int(emb_idx))

        for row_idx in row_indices:
            for post_id, emb_idx in zip(self.liked_post_ids[row_idx], self.liked_post_emb_indices[row_idx]):
                add_candidate(str(post_id), int(emb_idx))

        for post in self._sample_candidate_posts_for_batch(row_indices, bucket, epoch, candidate_to_idx):
            add_candidate(str(post["post_id"]), int(post["emb_idx"]))

        candidate_user_indices_padded = np.zeros(
            (len(candidate_emb_indices), self.max_post_liker_replay_events_per_post),
            dtype=np.int64,
        )
        candidate_time_gap_hours_padded = np.zeros(
            (len(candidate_emb_indices), self.max_post_liker_replay_events_per_post),
            dtype=np.float32,
        )
        candidate_max_events = 0
        for candidate_idx, emb_idx in enumerate(candidate_emb_indices):
            user_indices, time_gap_hours = self._post_liker_features_for_emb_idx(
                int(emb_idx),
                target_time_us,
                post_liker_cache,
            )
            candidate_max_events = max(
                candidate_max_events,
                _fill_post_liker_event_slice(
                    user_indices,
                    time_gap_hours,
                    candidate_user_indices_padded[candidate_idx],
                    candidate_time_gap_hours_padded[candidate_idx],
                ),
            )

        label_matrix = torch.zeros((len(user_ids), len(candidate_post_ids)), dtype=torch.float32)
        for row_idx in row_indices:
            user_idx = user_to_batch_idx[self.user_ids[row_idx]]
            for post_id in self.liked_post_ids[row_idx]:
                candidate_idx = candidate_to_idx.get(post_id)
                if candidate_idx is not None:
                    label_matrix[user_idx, candidate_idx] = 1.0

        return {
            "history_mask": torch.stack(history_mask_tensors, dim=0),
            "history_time_deltas_hours": torch.stack(time_delta_tensors, dim=0),
            "history_post_liker_user_indices": torch.from_numpy(history_user_indices_padded),
            "history_post_liker_time_gap_hours": torch.from_numpy(history_time_gap_hours_padded),
            "candidate_post_liker_user_indices": torch.from_numpy(candidate_user_indices_padded),
            "candidate_post_liker_time_gap_hours": torch.from_numpy(candidate_time_gap_hours_padded),
            "target_user_indices": torch.tensor(
                [self.target_user_indices[row_idx] for row_idx in row_indices],
                dtype=torch.long,
            ),
            "label_matrix": label_matrix,
            "user_id": user_ids,
            "candidate_post_id": candidate_post_ids,
            "bucket": bucket,
            "post_liker_replay_event_count": int(
                (history_user_indices_padded != 0).sum().item()
                + (candidate_user_indices_padded != 0).sum().item()
            ),
            "post_liker_replay_max_events_per_post": int(max(history_max_events, candidate_max_events)),
        }


def create_user_id_only_bucketed_data_loaders(
    train_dataset: UserIdOnlyBucketedEngagementDataset,
    val_dataset: UserIdOnlyBucketedEngagementDataset,
    val_unseen_dataset: UserIdOnlyBucketedEngagementDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    seed: int,
    train_resample_candidates_each_epoch: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    worker_kw: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        worker_kw.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BucketedBatchSampler(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            seed=seed,
            resample_candidates_each_epoch=train_resample_candidates_each_epoch,
        ),
        collate_fn=train_dataset.collate_batch,
        **worker_kw,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=BucketedBatchSampler(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            seed=seed,
        ),
        collate_fn=val_dataset.collate_batch,
        **worker_kw,
    )
    val_unseen_loader = DataLoader(
        val_unseen_dataset,
        batch_sampler=BucketedBatchSampler(
            val_unseen_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            seed=seed,
        ),
        collate_fn=val_unseen_dataset.collate_batch,
        **worker_kw,
    )
    return train_loader, val_loader, val_unseen_loader


class BSTUserIdOnlyRanker(nn.Module):
    """Behavior Sequence Transformer using only user-ID-derived post features."""

    __constants__ = ["post_liker_user_embedding_dim", "target_user_projection_dim", "prepend_target_user_token"]

    def __init__(
        self,
        post_liker_user_table_num_rows: int,
        post_liker_user_embedding_dim: int,
        post_liker_projection_dim: int,
        model_dim: int,
        time_embedding_dim: int,
        num_attention_heads: int,
        num_transformer_layers: int,
        transformer_ff_dim: int,
        dropout_rate: float,
        norm_first: bool,
        time_delta_bucket_boundaries_hours: List[float],
        prediction_hidden_dims: List[int],
        post_liker_pooling_tau_hours: float,
        target_user_projection_dim: int,
        post_liker_user_dropout_rate: float,
        target_user_dropout_rate: float,
        prepend_target_user_token: bool,
    ):
        super().__init__()
        if post_liker_user_table_num_rows < 2:
            raise ValueError("post_liker_user_table_num_rows must be at least 2")
        if post_liker_user_embedding_dim <= 0:
            raise ValueError("post_liker_user_embedding_dim must be positive")
        if post_liker_projection_dim <= 0:
            raise ValueError("post_liker_projection_dim must be positive")
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if time_embedding_dim <= 0:
            raise ValueError("time_embedding_dim must be positive")
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if num_transformer_layers <= 0:
            raise ValueError("num_transformer_layers must be positive")
        if transformer_ff_dim <= 0:
            raise ValueError("transformer_ff_dim must be positive")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be in [0, 1]")
        if post_liker_pooling_tau_hours <= 0.0:
            raise ValueError("post_liker_pooling_tau_hours must be positive")
        if target_user_projection_dim <= 0:
            raise ValueError("target_user_projection_dim must be positive")
        if not 0.0 <= post_liker_user_dropout_rate <= 1.0:
            raise ValueError("post_liker_user_dropout_rate must be in [0, 1]")
        if not 0.0 <= target_user_dropout_rate <= 1.0:
            raise ValueError("target_user_dropout_rate must be in [0, 1]")

        self.post_liker_user_table_num_rows = int(post_liker_user_table_num_rows)
        self.post_liker_user_embedding_dim = int(post_liker_user_embedding_dim)
        self.post_liker_projection_dim = int(post_liker_projection_dim)
        self.model_dim = int(model_dim)
        self.time_embedding_dim = int(time_embedding_dim)
        self.dropout_rate = float(dropout_rate)
        self.target_user_projection_dim = int(target_user_projection_dim)
        self.post_liker_user_dropout_rate = float(post_liker_user_dropout_rate)
        self.target_user_dropout_rate = float(target_user_dropout_rate)
        self.prepend_target_user_token = bool(prepend_target_user_token)
        self.time_delta_bucket_boundaries_hours = _validate_time_delta_bucket_boundaries(
            time_delta_bucket_boundaries_hours
        )
        self.register_buffer(
            "_time_delta_bucket_boundaries_tensor",
            torch.tensor(self.time_delta_bucket_boundaries_hours, dtype=torch.float32),
            persistent=False,
        )
        self.num_time_delta_buckets = len(self.time_delta_bucket_boundaries_hours) + 2
        self.transformer_input_dim = self.model_dim + self.time_embedding_dim
        if self.transformer_input_dim % int(num_attention_heads) != 0:
            raise ValueError("model_dim + time_embedding_dim must be divisible by num_attention_heads")

        self.post_liker_user_embedding = nn.Embedding(
            num_embeddings=self.post_liker_user_table_num_rows,
            embedding_dim=self.post_liker_user_embedding_dim,
            padding_idx=0,
        )
        nn.init.xavier_uniform_(self.post_liker_user_embedding.weight)
        with torch.no_grad():
            self.post_liker_user_embedding.weight[0].zero_()
        self.post_liker_user_pooler = PostLikerUserPooler(
            user_embedding_dim=self.post_liker_user_embedding_dim,
            pooling_tau_hours=post_liker_pooling_tau_hours,
        )
        self.post_liker_projection = nn.Linear(
            self.post_liker_user_embedding_dim,
            self.post_liker_projection_dim,
        )
        self.post_liker_projection_norm = nn.LayerNorm(self.post_liker_projection_dim)
        self.post_liker_projection_activation = nn.GELU()
        self.post_liker_fusion = nn.Linear(self.post_liker_projection_dim, self.model_dim)
        self.target_user_projection = nn.Linear(
            self.post_liker_user_embedding_dim,
            self.target_user_projection_dim,
        )
        self.target_user_projection_norm = nn.LayerNorm(self.target_user_projection_dim)
        self.target_user_projection_activation = nn.GELU()
        self.target_user_token_fusion = nn.Linear(self.target_user_projection_dim, self.model_dim)
        self.time_delta_embedding = nn.Embedding(
            num_embeddings=self.num_time_delta_buckets,
            embedding_dim=self.time_embedding_dim,
        )
        nn.init.xavier_uniform_(self.time_delta_embedding.weight)
        self.empty_history_token = nn.Parameter(torch.randn(self.transformer_input_dim) * 0.02)
        for layer in (
            self.post_liker_projection,
            self.post_liker_fusion,
            self.target_user_projection,
            self.target_user_token_fusion,
        ):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_input_dim,
            nhead=int(num_attention_heads),
            dim_feedforward=int(transformer_ff_dim),
            dropout=float(dropout_rate),
            activation="gelu",
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(num_transformer_layers),
            enable_nested_tensor=False,
        )
        self.prediction_head = LinearPredictionHead(
            input_dim=self.transformer_input_dim + self.target_user_projection_dim,
            hidden_dims=prediction_hidden_dims,
            dropout_rate=dropout_rate,
        )

    def _bucketize_time_deltas_hours(self, time_deltas_hours: torch.Tensor) -> torch.Tensor:
        deltas = time_deltas_hours
        if not torch.is_floating_point(deltas):
            deltas = deltas.to(dtype=torch.float32)
        deltas = torch.clamp(deltas, min=0.0)
        boundary_tensor = self._time_delta_bucket_boundaries_tensor.to(
            device=deltas.device,
            dtype=deltas.dtype,
        )
        positive_bucket_ids = torch.bucketize(deltas, boundary_tensor, right=False) + 1
        zero_bucket_ids = torch.zeros_like(positive_bucket_ids)
        return torch.where(deltas <= 0.0, zero_bucket_ids, positive_bucket_ids).to(dtype=torch.long)

    def _post_vectors_from_liker_tensors(
        self,
        user_indices: torch.Tensor,
        time_gap_hours: torch.Tensor,
    ) -> torch.Tensor:
        user_indices = self._apply_user_idx_unk_dropout(
            user_indices,
            self.post_liker_user_dropout_rate,
        )
        pooled = self.post_liker_user_pooler(
            user_indices,
            time_gap_hours,
            self.post_liker_user_embedding.weight,
        )
        projected = self.post_liker_projection_norm(
            self.post_liker_projection_activation(self.post_liker_projection(pooled))
        )
        return self.post_liker_fusion(projected)

    def _target_user_indices_for_features(self, target_user_indices: torch.Tensor) -> torch.Tensor:
        prepared_indices = target_user_indices.to(
            device=self.post_liker_user_embedding.weight.device,
            dtype=torch.long,
        )
        return self._apply_user_idx_unk_dropout(
            prepared_indices,
            self.target_user_dropout_rate,
        )

    def _target_user_features_from_prepared_indices(self, target_user_indices: torch.Tensor) -> torch.Tensor:
        target_user_embeddings = self.post_liker_user_embedding(target_user_indices)
        return self.target_user_projection_norm(
            self.target_user_projection_activation(
                self.target_user_projection(target_user_embeddings)
            )
        )

    def _target_user_features(self, target_user_indices: torch.Tensor) -> torch.Tensor:
        return self._target_user_features_from_prepared_indices(
            self._target_user_indices_for_features(target_user_indices)
        )

    def _target_user_token_input(self, target_user_features: torch.Tensor) -> torch.Tensor:
        batch_size = int(target_user_features.size(0))
        device = target_user_features.device
        target_user_post_vector = self.target_user_token_fusion(target_user_features).unsqueeze(1)
        target_user_time_bucket_ids = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
        target_user_time_embeddings = self.time_delta_embedding(target_user_time_bucket_ids)
        return torch.cat([target_user_post_vector, target_user_time_embeddings], dim=-1)

    def _maybe_prepend_target_user_token(
        self,
        history_input: torch.Tensor,
        history_mask: torch.Tensor,
        target_user_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.prepend_target_user_token:
            return history_input, history_mask
        batch_size = int(history_input.size(0))
        target_user_token = self._target_user_token_input(target_user_features)
        target_user_mask = torch.ones((batch_size, 1), device=history_mask.device, dtype=torch.bool)
        return torch.cat([target_user_token, history_input], dim=1), torch.cat([target_user_mask, history_mask], dim=1)

    def _apply_user_idx_unk_dropout(
        self,
        user_indices: torch.Tensor,
        dropout_rate: float,
    ) -> torch.Tensor:
        if not self.training or dropout_rate <= 0.0:
            return user_indices
        eligible = user_indices.gt(1)
        dropout_mask = eligible & torch.rand(user_indices.size(), device=user_indices.device).lt(dropout_rate)
        return torch.where(dropout_mask, torch.ones_like(user_indices), user_indices)

    def _inject_empty_history_token(
        self,
        history_input: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(history_input.size(0))
        max_history_len = int(history_input.size(1))
        token = self.empty_history_token.reshape(1, 1, self.transformer_input_dim)
        if max_history_len == 0:
            return (
                token.expand(batch_size, 1, self.transformer_input_dim),
                torch.ones((batch_size, 1), device=history_input.device, dtype=torch.bool),
            )

        has_history = history_mask.any(dim=1)
        if bool(has_history.all().item()):
            return history_input, history_mask

        inject = ~has_history
        inject_f = inject.to(dtype=history_input.dtype).reshape(batch_size, 1, 1)
        history_input = history_input.clone()
        history_input[:, 0:1, :] = (
            history_input[:, 0:1, :] * (1.0 - inject_f)
            + token.expand(batch_size, 1, self.transformer_input_dim) * inject_f
        )
        history_mask = history_mask.clone()
        history_mask[:, 0] = history_mask[:, 0] | inject
        return history_input, history_mask

    def _forward_transformer(
        self,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        history_post_liker_user_indices: torch.Tensor,
        history_post_liker_time_gap_hours: torch.Tensor,
        candidate_post_liker_user_indices: torch.Tensor,
        candidate_post_liker_time_gap_hours: torch.Tensor,
        target_user_features: torch.Tensor,
    ) -> torch.Tensor:
        if history_mask.dim() != 2:
            raise ValueError("history_mask must have shape [B, H]")
        batch_size, max_history_len = history_mask.shape
        if history_time_deltas_hours.shape != (batch_size, max_history_len):
            raise ValueError("history_time_deltas_hours must have shape [B, H]")
        if history_post_liker_user_indices.dim() != 3 or history_post_liker_user_indices.size(0) != batch_size or history_post_liker_user_indices.size(1) != max_history_len:
            raise ValueError("history_post_liker_user_indices must have shape [B, H, K]")
        if history_post_liker_time_gap_hours.shape != history_post_liker_user_indices.shape:
            raise ValueError("history_post_liker_time_gap_hours shape must match history_post_liker_user_indices")
        if candidate_post_liker_user_indices.dim() != 2 or candidate_post_liker_user_indices.size(0) != batch_size:
            raise ValueError("candidate_post_liker_user_indices must have shape [B, K]")
        if candidate_post_liker_time_gap_hours.shape != candidate_post_liker_user_indices.shape:
            raise ValueError("candidate_post_liker_time_gap_hours shape must match candidate_post_liker_user_indices")
        if target_user_features.shape != (batch_size, self.target_user_projection_dim):
            raise ValueError("target_user_features must have shape [B, target_user_projection_dim]")

        device = self.post_liker_user_embedding.weight.device
        history_mask = history_mask.to(device=device, dtype=torch.bool)
        history_time_deltas_hours = history_time_deltas_hours.to(device=device, dtype=torch.float32)
        history_post_liker_user_indices = history_post_liker_user_indices.to(device=device, dtype=torch.long)
        history_post_liker_time_gap_hours = history_post_liker_time_gap_hours.to(device=device, dtype=torch.float32)
        candidate_post_liker_user_indices = candidate_post_liker_user_indices.to(device=device, dtype=torch.long)
        candidate_post_liker_time_gap_hours = candidate_post_liker_time_gap_hours.to(device=device, dtype=torch.float32)

        history_post_vectors = self._post_vectors_from_liker_tensors(
            history_post_liker_user_indices,
            history_post_liker_time_gap_hours,
        )
        candidate_post_vector = self._post_vectors_from_liker_tensors(
            candidate_post_liker_user_indices,
            candidate_post_liker_time_gap_hours,
        ).unsqueeze(1)

        history_time_bucket_ids = self._bucketize_time_deltas_hours(history_time_deltas_hours)
        history_time_embeddings = self.time_delta_embedding(history_time_bucket_ids)
        history_input = torch.cat([history_post_vectors, history_time_embeddings], dim=-1)
        history_input, history_mask = self._inject_empty_history_token(history_input, history_mask)
        history_input, history_mask = self._maybe_prepend_target_user_token(
            history_input,
            history_mask,
            target_user_features,
        )
        candidate_time_bucket_ids = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
        candidate_time_embeddings = self.time_delta_embedding(candidate_time_bucket_ids)
        candidate_input = torch.cat([candidate_post_vector, candidate_time_embeddings], dim=-1)
        transformer_input = torch.cat([history_input, candidate_input], dim=1)

        candidate_is_not_padding = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
        src_key_padding_mask = torch.cat([~history_mask, candidate_is_not_padding], dim=1)
        encoded_sequence = self.transformer_encoder(
            transformer_input,
            src_key_padding_mask=src_key_padding_mask,
        )
        return encoded_sequence[:, -1, :]

    def _validate_one_layer_matrix_scorer(self) -> nn.TransformerEncoderLayer:
        layers = getattr(self.transformer_encoder, "layers", None)
        if layers is None or len(layers) != 1:
            raise RuntimeError("score_candidate_matrix requires exactly one transformer layer")
        layer = layers[0]
        if not isinstance(layer, nn.TransformerEncoderLayer):
            raise RuntimeError("score_candidate_matrix requires a standard TransformerEncoderLayer")
        self_attn = layer.self_attn
        if (
            not isinstance(self_attn, nn.MultiheadAttention)
            or not self_attn.batch_first
            or not getattr(self_attn, "_qkv_same_embed_dim", False)
            or self_attn.in_proj_weight is None
            or self_attn.in_proj_bias is None
            or self_attn.out_proj is None
        ):
            raise RuntimeError("score_candidate_matrix requires packed batch-first self-attention projections")
        if self_attn.embed_dim != self.transformer_input_dim:
            raise RuntimeError("score_candidate_matrix found a transformer dimension mismatch")
        return layer

    def _candidate_token_self_attention(
        self,
        layer: nn.TransformerEncoderLayer,
        history_input: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_input: torch.Tensor,
    ) -> torch.Tensor:
        self_attn = layer.self_attn
        num_users, max_history_len, embed_dim = history_input.shape
        num_candidates = int(candidate_input.size(0))
        num_heads = int(self_attn.num_heads)
        head_dim = embed_dim // num_heads
        scale = float(head_dim) ** -0.5

        in_proj_weight = self_attn.in_proj_weight
        in_proj_bias = self_attn.in_proj_bias
        if in_proj_weight is None or in_proj_bias is None:
            raise RuntimeError("score_candidate_matrix requires packed self-attention projections")
        q_weight, k_weight, v_weight = in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = in_proj_bias.chunk(3, dim=0)
        query = F.linear(candidate_input, q_weight, q_bias).view(num_candidates, num_heads, head_dim)
        history_key = F.linear(history_input, k_weight, k_bias).view(num_users, max_history_len, num_heads, head_dim)
        history_value = F.linear(history_input, v_weight, v_bias).view(num_users, max_history_len, num_heads, head_dim)
        candidate_key = F.linear(candidate_input, k_weight, k_bias).view(num_candidates, num_heads, head_dim)
        candidate_value = F.linear(candidate_input, v_weight, v_bias).view(num_candidates, num_heads, head_dim)

        history_scores = torch.einsum("cnd,uhnd->unch", query, history_key) * scale
        history_scores = history_scores.masked_fill(~history_mask[:, None, None, :], float("-inf"))
        candidate_scores = (query * candidate_key).sum(dim=-1).transpose(0, 1) * scale
        candidate_scores = candidate_scores.unsqueeze(0).unsqueeze(-1).expand(num_users, -1, -1, -1)
        attention_scores = torch.cat([history_scores, candidate_scores], dim=-1)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout_rate, training=self.training)

        if max_history_len == 0:
            history_context = torch.zeros(
                (num_users, num_candidates, num_heads, head_dim),
                device=history_input.device,
                dtype=history_input.dtype,
            )
        else:
            history_context = torch.einsum(
                "unch,uhnd->ucnd",
                attention_weights[..., :max_history_len],
                history_value,
            )
        candidate_context = (
            attention_weights[..., max_history_len].permute(0, 2, 1).unsqueeze(-1)
            * candidate_value.unsqueeze(0)
        )
        attention_output = (history_context + candidate_context).reshape(num_users, num_candidates, embed_dim)
        return F.linear(attention_output, self_attn.out_proj.weight, self_attn.out_proj.bias)

    def _candidate_token_feed_forward(
        self,
        layer: nn.TransformerEncoderLayer,
        candidate_state: torch.Tensor,
    ) -> torch.Tensor:
        hidden = F.linear(candidate_state, layer.linear1.weight, layer.linear1.bias)
        hidden = F.gelu(hidden)
        hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)
        hidden = F.linear(hidden, layer.linear2.weight, layer.linear2.bias)
        return F.dropout(hidden, p=self.dropout_rate, training=self.training)

    def score_candidate_matrix(
        self,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        history_post_liker_user_indices: torch.Tensor,
        history_post_liker_time_gap_hours: torch.Tensor,
        candidate_post_liker_user_indices: torch.Tensor,
        candidate_post_liker_time_gap_hours: torch.Tensor,
        target_user_indices: torch.Tensor,
    ) -> torch.Tensor:
        layer = self._validate_one_layer_matrix_scorer()
        if history_mask.dim() != 2:
            raise ValueError("history_mask must have shape [U, H]")
        num_users, max_history_len = history_mask.shape
        num_candidates = int(candidate_post_liker_user_indices.size(0))
        if history_time_deltas_hours.shape != (num_users, max_history_len):
            raise ValueError("history_time_deltas_hours must have shape [U, H]")
        if history_post_liker_user_indices.dim() != 3 or history_post_liker_user_indices.size(0) != num_users or history_post_liker_user_indices.size(1) != max_history_len:
            raise ValueError("history_post_liker_user_indices must have shape [U, H, K]")
        if history_post_liker_time_gap_hours.shape != history_post_liker_user_indices.shape:
            raise ValueError("history_post_liker_time_gap_hours shape must match history_post_liker_user_indices")
        if candidate_post_liker_user_indices.dim() != 2:
            raise ValueError("candidate_post_liker_user_indices must have shape [C, K]")
        if candidate_post_liker_time_gap_hours.shape != candidate_post_liker_user_indices.shape:
            raise ValueError("candidate_post_liker_time_gap_hours shape must match candidate_post_liker_user_indices")
        if target_user_indices.dim() != 1 or target_user_indices.size(0) != num_users:
            raise ValueError("target_user_indices must have shape [U]")

        device = self.post_liker_user_embedding.weight.device
        history_mask = history_mask.to(device=device, dtype=torch.bool)
        history_time_deltas_hours = history_time_deltas_hours.to(device=device, dtype=torch.float32)
        history_post_liker_user_indices = history_post_liker_user_indices.to(device=device, dtype=torch.long)
        history_post_liker_time_gap_hours = history_post_liker_time_gap_hours.to(device=device, dtype=torch.float32)
        candidate_post_liker_user_indices = candidate_post_liker_user_indices.to(device=device, dtype=torch.long)
        candidate_post_liker_time_gap_hours = candidate_post_liker_time_gap_hours.to(device=device, dtype=torch.float32)
        target_user_indices = self._target_user_indices_for_features(target_user_indices)
        target_user_features = self._target_user_features_from_prepared_indices(target_user_indices)

        history_post_vectors = self._post_vectors_from_liker_tensors(
            history_post_liker_user_indices,
            history_post_liker_time_gap_hours,
        )
        candidate_post_vectors = self._post_vectors_from_liker_tensors(
            candidate_post_liker_user_indices,
            candidate_post_liker_time_gap_hours,
        )
        history_time_bucket_ids = self._bucketize_time_deltas_hours(history_time_deltas_hours)
        history_time_embeddings = self.time_delta_embedding(history_time_bucket_ids)
        candidate_time_bucket_ids = torch.zeros((num_candidates,), device=device, dtype=torch.long)
        candidate_time_embeddings = self.time_delta_embedding(candidate_time_bucket_ids)
        history_input = torch.cat([history_post_vectors, history_time_embeddings], dim=-1)
        history_input, history_mask = self._inject_empty_history_token(history_input, history_mask)
        history_input, history_mask = self._maybe_prepend_target_user_token(
            history_input,
            history_mask,
            target_user_features,
        )
        candidate_input = torch.cat([candidate_post_vectors, candidate_time_embeddings], dim=-1)

        if layer.norm_first:
            normed_history_input = F.layer_norm(
                history_input,
                [self.transformer_input_dim],
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
            )
            normed_candidate_input = F.layer_norm(
                candidate_input,
                [self.transformer_input_dim],
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
            )
            attention_output = F.dropout(
                self._candidate_token_self_attention(
                    layer,
                    normed_history_input,
                    history_mask,
                    normed_candidate_input,
                ),
                p=self.dropout_rate,
                training=self.training,
            )
            candidate_state = candidate_input.unsqueeze(0) + attention_output
            normed_candidate_state = F.layer_norm(
                candidate_state,
                [self.transformer_input_dim],
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps,
            )
            candidate_state = candidate_state + self._candidate_token_feed_forward(
                layer,
                normed_candidate_state,
            )
        else:
            attention_output = F.dropout(
                self._candidate_token_self_attention(layer, history_input, history_mask, candidate_input),
                p=self.dropout_rate,
                training=self.training,
            )
            candidate_state = F.layer_norm(
                candidate_input.unsqueeze(0) + attention_output,
                [self.transformer_input_dim],
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
            )
            candidate_state = F.layer_norm(
                candidate_state + self._candidate_token_feed_forward(layer, candidate_state),
                [self.transformer_input_dim],
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps,
            )

        prediction_input = candidate_state.reshape(num_users * num_candidates, self.transformer_input_dim)
        expanded_target_features = target_user_features.unsqueeze(1).expand(
            num_users,
            num_candidates,
            self.target_user_projection_dim,
        ).reshape(num_users * num_candidates, self.target_user_projection_dim)
        prediction_input = torch.cat([prediction_input, expanded_target_features], dim=-1)
        logits = self.prediction_head(prediction_input)
        if logits.dim() == 2 and logits.shape == (num_users * num_candidates, 1):
            logits = logits.squeeze(-1)
        if logits.shape != (num_users * num_candidates,):
            raise RuntimeError("prediction_head must return logits with shape [U*C] or [U*C, 1]")
        return logits.reshape(num_users, num_candidates)

    def forward(
        self,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        history_post_liker_user_indices: torch.Tensor,
        history_post_liker_time_gap_hours: torch.Tensor,
        candidate_post_liker_user_indices: torch.Tensor,
        candidate_post_liker_time_gap_hours: torch.Tensor,
        target_user_indices: torch.Tensor,
    ) -> torch.Tensor:
        target_user_indices = self._target_user_indices_for_features(target_user_indices)
        target_user_features = self._target_user_features_from_prepared_indices(target_user_indices)
        transformer_output = self._forward_transformer(
            history_mask=history_mask,
            history_time_deltas_hours=history_time_deltas_hours,
            history_post_liker_user_indices=history_post_liker_user_indices,
            history_post_liker_time_gap_hours=history_post_liker_time_gap_hours,
            candidate_post_liker_user_indices=candidate_post_liker_user_indices,
            candidate_post_liker_time_gap_hours=candidate_post_liker_time_gap_hours,
            target_user_features=target_user_features,
        )
        prediction_input = torch.cat([transformer_output, target_user_features], dim=-1)
        logits = self.prediction_head(prediction_input)
        if logits.dim() == 2 and logits.shape == (transformer_output.size(0), 1):
            logits = logits.squeeze(-1)
        if logits.shape != (transformer_output.size(0),):
            raise RuntimeError("prediction_head must return logits with shape [B] or [B, 1]")
        return logits


def _compute_bst_user_id_only_listwise_loss_and_preds(
    model: BSTUserIdOnlyRanker,
    batch: Dict[str, Any],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    required_fields = (
        "history_mask",
        "history_time_deltas_hours",
        "history_post_liker_user_indices",
        "history_post_liker_time_gap_hours",
        "candidate_post_liker_user_indices",
        "candidate_post_liker_time_gap_hours",
        "target_user_indices",
        "label_matrix",
    )
    missing_fields = [field for field in required_fields if field not in batch]
    if missing_fields:
        raise RuntimeError("BST user-ID-only batches are missing required fields: " + ", ".join(missing_fields))

    history_mask = batch["history_mask"].to(device, dtype=torch.bool, non_blocking=True)
    history_time_deltas_hours = batch["history_time_deltas_hours"].to(device, dtype=torch.float32, non_blocking=True)
    history_post_liker_user_indices = batch["history_post_liker_user_indices"].to(device, dtype=torch.long, non_blocking=True)
    history_post_liker_time_gap_hours = batch["history_post_liker_time_gap_hours"].to(device, dtype=torch.float32, non_blocking=True)
    candidate_post_liker_user_indices = batch["candidate_post_liker_user_indices"].to(device, dtype=torch.long, non_blocking=True)
    candidate_post_liker_time_gap_hours = batch["candidate_post_liker_time_gap_hours"].to(device, dtype=torch.float32, non_blocking=True)
    target_user_indices = batch["target_user_indices"].to(device, dtype=torch.long, non_blocking=True)
    labels = batch["label_matrix"].to(device, dtype=torch.float32, non_blocking=True)

    scores = model.score_candidate_matrix(
        history_mask=history_mask,
        history_time_deltas_hours=history_time_deltas_hours,
        history_post_liker_user_indices=history_post_liker_user_indices,
        history_post_liker_time_gap_hours=history_post_liker_time_gap_hours,
        candidate_post_liker_user_indices=candidate_post_liker_user_indices,
        candidate_post_liker_time_gap_hours=candidate_post_liker_time_gap_hours,
        target_user_indices=target_user_indices,
    )
    if scores.shape != labels.shape:
        raise RuntimeError("Expected BST scores and label_matrix to have matching [num_users, num_candidates] shapes")
    positive_counts = labels.sum(dim=1, keepdim=True)
    if torch.any(positive_counts <= 0):
        raise RuntimeError("Each user row in label_matrix must contain at least one positive candidate")

    targets = labels / positive_counts
    loss_per_user = -(targets * F.log_softmax(scores, dim=1)).sum(dim=1)
    return loss_per_user.mean(), scores, labels


def run_bst_user_id_only_listwise_epoch(
    *,
    train: bool,
    split_name: str,
    model: BSTUserIdOnlyRanker,
    device: str,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    disable_progress: bool,
    gradient_clip_max_norm: float,
    metrics_top_ks: List[int],
    calc_baseline_metrics: bool,
    max_batches: Optional[int] = None,
) -> Tuple[float, Dict[str, Any], Dict[str, float]]:
    if train:
        if optimizer is None:
            raise ValueError("optimizer is required when train=True")
        model.train()
    else:
        model.eval()

    loss_sum = torch.zeros((), device=device)
    batches = 0
    baseline_metric_sums = empty_rank_metric_sums(metrics_top_ks)
    baseline_metric_user_count = 0
    metric_sums = empty_rank_metric_sums(metrics_top_ks)
    metric_user_count = 0
    zero_history_metric_sums = empty_rank_metric_sums(metrics_top_ks)
    zero_history_metric_user_count = 0

    with nullcontext() if train else torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=split_name, leave=False, disable=disable_progress)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if train and optimizer is not None:
                optimizer.zero_grad()

            loss, scores, labels = _compute_bst_user_id_only_listwise_loss_and_preds(model, batch, device)
            if calc_baseline_metrics:
                baseline_batch_metric_sums, baseline_batch_metric_user_count = calc_baseline_rank_metrics_for_batch(
                    labels,
                    metrics_top_ks,
                )
                baseline_metric_user_count += baseline_batch_metric_user_count
                for key, value in baseline_batch_metric_sums.items():
                    baseline_metric_sums[key] += value

            ranked_indices = torch.argsort(scores.detach(), dim=1, descending=True)
            ranked_labels = torch.gather(labels, dim=1, index=ranked_indices)
            batch_metric_sums, batch_metric_user_count = rank_metric_sums_for_batch(
                ranked_labels,
                metrics_top_ks,
            )
            batch_zero_history_metric_sums, batch_zero_history_metric_user_count = zero_history_rank_metric_sums_for_batch(
                batch,
                ranked_labels,
                metrics_top_ks,
            )

            if train and optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)
                optimizer.step()

            loss_sum += loss.detach()
            batches += 1
            metric_user_count += batch_metric_user_count
            for key, value in batch_metric_sums.items():
                metric_sums[key] += value
            zero_history_metric_user_count += batch_zero_history_metric_user_count
            for key, value in batch_zero_history_metric_sums.items():
                zero_history_metric_sums[key] += value

    loss = (loss_sum / max(batches, 1)).item()
    baseline_metrics = finalize_rank_metrics(baseline_metric_sums, baseline_metric_user_count)
    metrics: Dict[str, Any] = finalize_rank_metrics(metric_sums, metric_user_count)
    metrics.update(finalize_zero_history_rank_metrics(zero_history_metric_sums, zero_history_metric_user_count))
    metrics["loss"] = loss
    metrics["rank_metric_user_count"] = metric_user_count
    return loss, metrics, baseline_metrics


class BSTUserIdOnlyMatrixScorer:
    def __init__(self, model: BSTUserIdOnlyRanker):
        self.model = model

    def prepare_for_eval(self, device: str) -> None:
        self.model = self.model.to(device)
        self.model.eval()

    def score_batch(self, batch: Dict[str, Any], device: str) -> MatrixBatchScores:
        loss, scores, _ = _compute_bst_user_id_only_listwise_loss_and_preds(self.model, batch, device)
        return MatrixBatchScores(scores=scores, loss=loss)


def _listwise_history_metric_names(metrics_top_ks: List[int]) -> List[str]:
    names: List[str] = []
    for k in metrics_top_ks:
        names.extend([f"ndcg@{k}", f"recall@{k}"])
    names.append("mean_average_precision")
    return names


def _append_split_metrics_to_history(
    history: Dict[str, List[float]],
    split_name: str,
    metrics: Dict[str, Any],
    metric_names: List[str],
) -> None:
    for metric_name in metric_names:
        key = f"{split_name}_{metric_name}"
        metric_value = metrics.get(metric_name)
        history.setdefault(key, []).append(float(metric_value) if metric_value is not None else float("nan"))


def _log_bst_user_id_only_epoch_metrics(
    experiment_tracker: Optional[Any],
    iteration: int,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    val_unseen_metrics: Dict[str, Any],
    train_baseline_metrics: Dict[str, float],
    val_baseline_metrics: Dict[str, float],
    val_unseen_baseline_metrics: Dict[str, float],
    calc_baseline_metrics: bool,
    metrics_top_ks: List[int],
    primary_metric_name: str,
) -> None:
    if experiment_tracker is None:
        return
    primary_metric_key = primary_metric_name.replace("val_unseen_", "", 1)
    primary_metric_value = val_unseen_metrics.get(primary_metric_key)
    if primary_metric_value is not None:
        experiment_tracker.log_scalar(
            f"Primary Ranking Metric ({primary_metric_key})",
            f"Validation Unseen Users {primary_metric_key}",
            float(primary_metric_value),
            iteration,
        )
    for k in metrics_top_ks:
        if calc_baseline_metrics:
            for metric_name, metric_label in ((f"ndcg@{k}", f"NDCG@{k}"), (f"recall@{k}", f"Recall@{k}")):
                for split_label, metrics in (
                    ("Train", train_baseline_metrics),
                    ("Validation", val_baseline_metrics),
                    ("Validation Unseen Users", val_unseen_baseline_metrics),
                ):
                    experiment_tracker.log_scalar(
                        metric_label,
                        f"{split_label} {metric_label}",
                        float(metrics[metric_name]),
                        0,
                    )
        for metric_name, metric_label in ((f"ndcg@{k}", f"NDCG@{k}"), (f"recall@{k}", f"Recall@{k}")):
            for split_label, metrics in (
                ("Train", train_metrics),
                ("Validation", val_metrics),
                ("Validation Unseen Users", val_unseen_metrics),
            ):
                metric_value = metrics.get(metric_name)
                if metric_value is None:
                    continue
                experiment_tracker.log_scalar(metric_label, f"{split_label} {metric_label}", float(metric_value), iteration)
    log_zero_history_rank_metrics(
        experiment_tracker,
        {
            "train": train_metrics,
            "validation": val_metrics,
            "validation_unseen_users": val_unseen_metrics,
        },
        metrics_top_ks,
        iteration,
    )


def train_bst_user_id_only_model(
    model: BSTUserIdOnlyRanker,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_unseen_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    early_stopping_min_delta: float,
    checkpoints_dir: Optional[Path],
    disable_progress: bool,
    lr_scheduler_factor: float,
    lr_scheduler_patience: int,
    gradient_clip_max_norm: float,
    metrics_top_ks: Optional[List[int]] = None,
    bst_max_train_batches_per_epoch: Optional[int] = None,
    experiment_tracker: Optional[Any] = None,
) -> Dict[str, Any]:
    metrics_top_ks = list(metrics_top_ks or [30])
    if not metrics_top_ks:
        raise ValueError("metrics_top_ks must contain at least one value")
    if bst_max_train_batches_per_epoch is not None and bst_max_train_batches_per_epoch <= 0:
        raise ValueError("bst_max_train_batches_per_epoch must be positive when provided")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_scheduler_factor, patience=lr_scheduler_patience
    )

    primary_metric_name = f"val_unseen_ndcg@{metrics_top_ks[0]}"
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_unseen_loss": [],
    }
    listwise_metric_names = _listwise_history_metric_names(metrics_top_ks)
    for split_name in ("train", "val", "val_unseen"):
        for metric_name in listwise_metric_names:
            history[f"{split_name}_{metric_name}"] = []
    best_val_metric = float("-inf")
    best_val_loss = float("inf")
    best_reset_val_metric = float("-inf")
    patience_counter = 0
    best_state_dict = None

    for epoch in tqdm(range(epochs), desc="Training epochs", disable=disable_progress):
        calc_baseline_metrics = epoch == 0
        train_loss, train_metrics, train_baseline_metrics = run_bst_user_id_only_listwise_epoch(
            train=True,
            split_name="Train",
            model=model,
            device=device,
            dataloader=train_loader,
            optimizer=optimizer,
            disable_progress=disable_progress,
            gradient_clip_max_norm=gradient_clip_max_norm,
            metrics_top_ks=metrics_top_ks,
            calc_baseline_metrics=calc_baseline_metrics,
            max_batches=bst_max_train_batches_per_epoch,
        )
        val_loss, val_metrics, val_baseline_metrics = run_bst_user_id_only_listwise_epoch(
            train=False,
            split_name="Validation",
            model=model,
            device=device,
            dataloader=val_loader,
            optimizer=None,
            disable_progress=disable_progress,
            gradient_clip_max_norm=gradient_clip_max_norm,
            metrics_top_ks=metrics_top_ks,
            calc_baseline_metrics=calc_baseline_metrics,
        )
        val_unseen_loss, val_unseen_metrics, val_unseen_baseline_metrics = run_bst_user_id_only_listwise_epoch(
            train=False,
            split_name="Validation Unseen Users",
            model=model,
            device=device,
            dataloader=val_unseen_loader,
            optimizer=None,
            disable_progress=disable_progress,
            gradient_clip_max_norm=gradient_clip_max_norm,
            metrics_top_ks=metrics_top_ks,
            calc_baseline_metrics=calc_baseline_metrics,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_unseen_loss"].append(val_unseen_loss)
        _append_split_metrics_to_history(history, "train", train_metrics, listwise_metric_names)
        _append_split_metrics_to_history(history, "val", val_metrics, listwise_metric_names)
        _append_split_metrics_to_history(history, "val_unseen", val_unseen_metrics, listwise_metric_names)
        _log_bst_user_id_only_epoch_metrics(
            experiment_tracker,
            epoch + 1,
            train_metrics,
            val_metrics,
            val_unseen_metrics,
            train_baseline_metrics,
            val_baseline_metrics,
            val_unseen_baseline_metrics,
            calc_baseline_metrics,
            metrics_top_ks,
            primary_metric_name,
        )

        primary_metric_key = primary_metric_name.replace("val_unseen_", "", 1)
        primary_metric_value = val_unseen_metrics.get(primary_metric_key)
        primary_metric = float(primary_metric_value) if primary_metric_value is not None else None
        scheduler.step(primary_metric if primary_metric is not None else float("-inf"))

        if primary_metric is not None and primary_metric > best_val_metric:
            best_val_metric = primary_metric
            best_val_loss = val_unseen_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoints_dir is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state_dict,
                        "val_unseen_loss": val_unseen_loss,
                        "primary_metric_name": primary_metric_name,
                        "val_unseen_primary_metric": primary_metric,
                        "history": history,
                    },
                    checkpoints_dir / "bst_user_id_only_best.pth",
                )

        significant_improvement = (
            primary_metric is not None
            and primary_metric > best_reset_val_metric
            and (primary_metric - best_reset_val_metric) >= early_stopping_min_delta
        )
        if significant_improvement and primary_metric is not None:
            best_reset_val_metric = primary_metric
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_metric": best_val_metric,
        "primary_metric_name": primary_metric_name,
    }


def _load_user_id_only_training_data(
    context: Context,
    logger: Optional[Any] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if logger is None:
        logger = get_stage_logger("DATALOADERS")

    log_operation_start("Locate likes_core", "DATALOADERS", logger)
    get_data_dir = _resolve_prior(context, stage_key="get_data", folder="01_get_data")
    likes_core_df = load_parquet_from_prior(get_data_dir, "likes_core_").collect()
    logger.info(f"Loaded likes_core: {len(likes_core_df):,} rows")

    log_operation_start("Locate posts_core", "DATALOADERS", logger)
    posts_core_df = load_parquet_from_prior(get_data_dir, "posts_core_").collect()
    logger.info(f"Loaded posts_core: {len(posts_core_df):,} rows")

    log_operation_start("Locate user_history", "DATALOADERS", logger)
    history_dir = _resolve_prior(context, stage_key="user_history", folder="02_user_history")
    history_df = load_parquet_from_prior(history_dir, "history_posts_").collect()
    logger.info(f"Loaded user_history: {len(history_df):,} rows")

    post_liker_events_df, user_idx_df = load_post_liker_event_artifacts(context, logger=logger)
    return likes_core_df, posts_core_df, history_df, post_liker_events_df, user_idx_df


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    device = get_device(args.device)
    timestamp = context.run_timestamp

    run_tag = args.run_tag or "bst_user_id_only"
    out_dir = context.new_stage_dir("03_train", tag=run_tag)
    checkpoints_dir = out_dir / "checkpoints"
    plots_dir = out_dir / "plots"
    logs_dir = out_dir / "logs"
    for directory in (checkpoints_dir, plots_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    logger = get_stage_logger(STAGE_LOG_NAME, log_file=out_dir / "stage.log")
    log_operation_start("Stage 3 BST user-ID-only experiment training", STAGE_LOG_NAME, logger)
    t0 = time.time()

    clear_cuda_memory()
    random_seed = int(args.random_seed)
    set_random_seeds(random_seed)

    log_operation_start("Load lightweight training data from prior stages", STAGE_LOG_NAME, logger)
    likes_core_df, posts_core_df, history_df, post_liker_events_df, post_liker_user_idx_df = _load_user_id_only_training_data(
        context,
        logger=logger,
    )
    log_prior_stage_inputs(context, logger)

    max_history_len = int(args.max_history_len)
    model_dim = int(args.bst_model_dim)
    time_embedding_dim = int(args.bst_time_embedding_dim)
    num_attention_heads = int(args.bst_num_attention_heads)
    num_transformer_layers = int(args.bst_num_transformer_layers)
    transformer_ff_dim = int(args.bst_transformer_ff_dim)
    dropout_rate = float(args.bst_dropout_rate)
    norm_first = bool(args.bst_norm_first)
    time_delta_bucket_boundaries_hours = [float(v) for v in args.bst_time_delta_bucket_boundaries_hours]
    if args.prediction_hidden_dims is None:
        raise ValueError("prediction_hidden_dims is required for BST user-ID-only training")
    prediction_hidden_dims = [int(v) for v in args.prediction_hidden_dims]
    post_liker_user_embedding_dim = int(args.bst_post_liker_user_embedding_dim)
    post_liker_projection_dim = int(args.bst_post_liker_projection_dim)
    post_liker_pooling_tau_hours = float(args.bst_post_liker_pooling_tau_hours)
    target_user_projection_dim = int(args.bst_target_user_projection_dim)
    post_liker_user_dropout_rate = float(args.bst_post_liker_user_dropout_rate)
    target_user_dropout_rate = float(args.bst_target_user_dropout_rate)
    prepend_target_user_token = bool(args.bst_user_id_only_prepend_target_user_token)
    bst_max_post_liker_replay_events_per_post = args.bst_max_post_liker_replay_events_per_post
    if bst_max_post_liker_replay_events_per_post is None:
        raise ValueError("bst_max_post_liker_replay_events_per_post is required for BST user-ID-only training")
    bst_max_post_liker_replay_events_per_post = int(bst_max_post_liker_replay_events_per_post)
    batch_size = int(args.batch_size)
    bst_additional_batch_negatives = int(args.bst_additional_batch_negatives)
    bst_max_train_batches_per_epoch = args.bst_max_train_batches_per_epoch
    if bst_max_train_batches_per_epoch is not None:
        bst_max_train_batches_per_epoch = int(bst_max_train_batches_per_epoch)
    metrics_top_ks = list(args.metrics_top_ks)
    if not metrics_top_ks:
        raise ValueError("metrics_top_ks must contain at least one value")
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.bst_weight_decay)
    epochs = int(args.epochs)
    patience = int(args.patience)
    early_stopping_min_delta = float(args.early_stopping_min_delta)
    disable_progress = bool(args.disable_progress)
    generate_plots = not bool(args.no_plots)
    save_model = not bool(args.no_save_model)
    lr_scheduler_factor = float(args.lr_scheduler_factor)
    lr_scheduler_patience = int(args.lr_scheduler_patience)
    gradient_clip_max_norm = float(args.gradient_clip_max_norm)
    primary_metric_name = f"val_unseen_ndcg@{metrics_top_ks[0]}"

    if num_transformer_layers != 1:
        raise ValueError("BST user-ID-only experiment requires bst_num_transformer_layers=1")

    post_liker_event_lookup = PostLikerEventLookup.from_dataframe(post_liker_events_df)
    post_liker_user_table_num_rows = _get_post_liker_user_table_num_rows(post_liker_user_idx_df)
    logger.info(
        "BST user-ID-only experiment enabled: "
        f"user_embedding_dim={post_liker_user_embedding_dim}, "
        f"post_liker_projection_dim={post_liker_projection_dim}, "
        f"target_user_projection_dim={target_user_projection_dim}, "
        f"pooling_tau_hours={post_liker_pooling_tau_hours}, "
        f"post_liker_user_dropout_rate={post_liker_user_dropout_rate}, "
        f"target_user_dropout_rate={target_user_dropout_rate}, "
        f"prepend_target_user_token={prepend_target_user_token}, "
        f"user_table_num_rows={post_liker_user_table_num_rows}, "
        f"max_post_liker_replay_events_per_post={bst_max_post_liker_replay_events_per_post}, "
        f"post_liker_event_rows={len(post_liker_events_df):,}, "
        f"post_liker_user_idx_rows={len(post_liker_user_idx_df):,}"
    )
    logger.warning("BST user-ID-only checkpoints are experimental and not serving-compatible.")

    num_workers = int(args.num_dataloader_workers)
    pin_memory = bool(args.dataloader_pin_memory)
    persistent_workers = bool(args.dataloader_persistent_workers)
    prefetch_factor = int(args.dataloader_prefetch_factor)

    log_operation_start("Create lightweight bucketed BST user-ID-only datasets", STAGE_LOG_NAME, logger)
    train_dataset = UserIdOnlyBucketedEngagementDataset(
        likes_core_df=likes_core_df,
        posts_core_df=posts_core_df,
        history_df=history_df,
        split="train",
        max_history_len=max_history_len,
        post_liker_event_lookup=post_liker_event_lookup,
        post_liker_user_idx_df=post_liker_user_idx_df,
        max_post_liker_replay_events_per_post=bst_max_post_liker_replay_events_per_post,
        bst_additional_batch_negatives=bst_additional_batch_negatives,
        seed=random_seed,
        logger=logger,
    )
    val_dataset = UserIdOnlyBucketedEngagementDataset(
        likes_core_df=likes_core_df,
        posts_core_df=posts_core_df,
        history_df=history_df,
        split="val",
        max_history_len=max_history_len,
        post_liker_event_lookup=post_liker_event_lookup,
        post_liker_user_idx_df=post_liker_user_idx_df,
        max_post_liker_replay_events_per_post=bst_max_post_liker_replay_events_per_post,
        bst_additional_batch_negatives=bst_additional_batch_negatives,
        seed=random_seed,
        logger=logger,
    )
    val_unseen_dataset = UserIdOnlyBucketedEngagementDataset(
        likes_core_df=likes_core_df,
        posts_core_df=posts_core_df,
        history_df=history_df,
        split="val_unseen_users",
        max_history_len=max_history_len,
        post_liker_event_lookup=post_liker_event_lookup,
        post_liker_user_idx_df=post_liker_user_idx_df,
        max_post_liker_replay_events_per_post=bst_max_post_liker_replay_events_per_post,
        bst_additional_batch_negatives=bst_additional_batch_negatives,
        seed=random_seed,
        logger=logger,
    )

    config = {
        "model_type": "bst-user-id-only",
        "model_dim": model_dim,
        "time_embedding_dim": time_embedding_dim,
        "num_attention_heads": num_attention_heads,
        "num_transformer_layers": num_transformer_layers,
        "transformer_ff_dim": transformer_ff_dim,
        "dropout_rate": dropout_rate,
        "norm_first": norm_first,
        "time_delta_bucket_boundaries_hours": list(time_delta_bucket_boundaries_hours),
        "prediction_hidden_dims": list(prediction_hidden_dims),
        "max_history_len": max_history_len,
        "bst_additional_batch_negatives": bst_additional_batch_negatives,
        "bst_use_post_liker_user_pooling": True,
        "bst_post_liker_user_table_num_rows": post_liker_user_table_num_rows,
        "bst_post_liker_user_embedding_dim": post_liker_user_embedding_dim,
        "bst_post_liker_projection_dim": post_liker_projection_dim,
        "bst_post_liker_pooling_tau_hours": post_liker_pooling_tau_hours,
        "bst_target_user_projection_dim": target_user_projection_dim,
        "bst_post_liker_user_dropout_rate": post_liker_user_dropout_rate,
        "bst_target_user_dropout_rate": target_user_dropout_rate,
        "bst_user_id_only_prepend_target_user_token": prepend_target_user_token,
        "bst_max_post_liker_replay_events_per_post": bst_max_post_liker_replay_events_per_post,
        "requires_post_liker_user_pooling": True,
        "requires_target_user_indices": True,
        "experimental_offline_only": True,
    }
    training_config = {
        **config,
        "batch_size": batch_size,
        "bst_max_train_batches_per_epoch": bst_max_train_batches_per_epoch,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "random_seed": random_seed,
        "lr_scheduler_factor": lr_scheduler_factor,
        "lr_scheduler_patience": lr_scheduler_patience,
        "gradient_clip_max_norm": gradient_clip_max_norm,
        "primary_metric_name": primary_metric_name,
        "metrics_top_ks": metrics_top_ks,
        "num_dataloader_workers": num_workers,
        "dataloader_pin_memory": pin_memory,
        "dataloader_persistent_workers": persistent_workers,
        "dataloader_prefetch_factor": prefetch_factor,
        "save_model": save_model,
        "generate_plots": generate_plots,
    }
    training_config_path = out_dir / "training_config.json"
    with open(training_config_path, "w") as f:
        json.dump(training_config, f, indent=2)
    logger.info(f"Training config written to: {training_config_path}")

    train_loader, val_loader, val_unseen_loader = create_user_id_only_bucketed_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        val_unseen_dataset=val_unseen_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        seed=random_seed,
        train_resample_candidates_each_epoch=True,
    )
    del likes_core_df, posts_core_df, history_df, post_liker_events_df, post_liker_user_idx_df, post_liker_event_lookup

    log_operation_start("Create BST user-ID-only ranker model", STAGE_LOG_NAME, logger)
    model = BSTUserIdOnlyRanker(
        post_liker_user_table_num_rows=post_liker_user_table_num_rows,
        post_liker_user_embedding_dim=post_liker_user_embedding_dim,
        post_liker_projection_dim=post_liker_projection_dim,
        model_dim=model_dim,
        time_embedding_dim=time_embedding_dim,
        num_attention_heads=num_attention_heads,
        num_transformer_layers=num_transformer_layers,
        transformer_ff_dim=transformer_ff_dim,
        dropout_rate=dropout_rate,
        norm_first=norm_first,
        time_delta_bucket_boundaries_hours=time_delta_bucket_boundaries_hours,
        prediction_hidden_dims=prediction_hidden_dims,
        post_liker_pooling_tau_hours=post_liker_pooling_tau_hours,
        target_user_projection_dim=target_user_projection_dim,
        post_liker_user_dropout_rate=post_liker_user_dropout_rate,
        target_user_dropout_rate=target_user_dropout_rate,
        prepend_target_user_token=prepend_target_user_token,
    )

    log_operation_start(f"Train BST user-ID-only ranker (epochs={epochs}, batch_size={batch_size})", STAGE_LOG_NAME, logger)
    training_results = train_bst_user_id_only_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_unseen_loader=val_unseen_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        early_stopping_min_delta=early_stopping_min_delta,
        checkpoints_dir=checkpoints_dir,
        disable_progress=disable_progress,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        gradient_clip_max_norm=gradient_clip_max_norm,
        metrics_top_ks=metrics_top_ks,
        bst_max_train_batches_per_epoch=bst_max_train_batches_per_epoch,
        experiment_tracker=context.tracker,
    )
    trained_model: BSTUserIdOnlyRanker = training_results["model"]
    clear_cuda_memory()

    model_path = None
    if save_model:
        log_operation_start("Save BST user-ID-only checkpoint", STAGE_LOG_NAME, logger)
        model_path = checkpoints_dir / f"bst_user_id_only_{timestamp}.pth"
        torch.save(
            {
                "model_state_dict": trained_model.state_dict(),
                "config": config,
                "training_history": training_results["history"],
                "primary_metric_name": training_results["primary_metric_name"],
                "best_val_metric": training_results["best_val_metric"],
                "best_val_loss": training_results["best_val_loss"],
            },
            model_path,
        )
        logger.info(f"Model saved to: {model_path}")

    if generate_plots:
        hist = training_results["history"]
        try:
            primary_metric_name = training_results["primary_metric_name"]
            val_unseen_metric_history = hist.get(primary_metric_name, [])
            valid_metrics = [
                (idx + 1, float(value))
                for idx, value in enumerate(val_unseen_metric_history)
                if float(value) == float(value)
            ]
            best_epoch = max(valid_metrics, key=lambda item: item[1])[0] if valid_metrics else None
        except Exception as exc:
            logger.warning(f"Could not determine best epoch from BST user-ID-only training history: {exc}")
            best_epoch = None
        plot_training_history(hist, plots_dir / f"training_history_{timestamp}.png", best_epoch=best_epoch)

    bst_matrix_scorer = BSTUserIdOnlyMatrixScorer(trained_model)
    train_eval = evaluate_matrix_scorer(
        bst_matrix_scorer,
        train_loader,
        device=device,
        metrics_top_ks=metrics_top_ks,
        progress_desc="Evaluate train",
        disable_progress=disable_progress,
        max_batches=bst_max_train_batches_per_epoch,
    )
    val_eval = evaluate_matrix_scorer(
        bst_matrix_scorer,
        val_loader,
        device=device,
        metrics_top_ks=metrics_top_ks,
        progress_desc="Evaluate validation",
        disable_progress=disable_progress,
    )
    val_unseen_eval = evaluate_matrix_scorer(
        bst_matrix_scorer,
        val_unseen_loader,
        device=device,
        metrics_top_ks=metrics_top_ks,
        progress_desc="Evaluate validation unseen users",
        disable_progress=disable_progress,
    )
    train_metrics = train_eval["metrics"]
    val_metrics = val_eval["metrics"]
    val_unseen_metrics = val_unseen_eval["metrics"]
    train_loss = float(train_metrics["loss"]) if train_metrics.get("loss") is not None else 0.0
    val_loss = float(val_metrics["loss"]) if val_metrics.get("loss") is not None else 0.0
    val_unseen_loss = float(val_unseen_metrics["loss"]) if val_unseen_metrics.get("loss") is not None else 0.0
    logger.info(f"Train metrics: {train_metrics}")
    logger.info(f"Validation metrics: {val_metrics}")
    logger.info(f"Validation unseen users metrics: {val_unseen_metrics}")

    final_split_metrics: Dict[str, Dict[str, Any]] = {
        "train": train_metrics,
        "val": val_metrics,
        "val_unseen_users": val_unseen_metrics,
    }
    runtime = time.time() - t0
    training_results_path = out_dir / "training_results.json"
    end_of_training_values = {
        "runtime_seconds": runtime,
        "model_path": str(model_path) if model_path else None,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "val_unseen_samples": len(val_unseen_dataset),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_unseen_loss": val_unseen_loss,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "val_unseen_metrics": val_unseen_metrics,
        "primary_metric_name": training_results["primary_metric_name"],
        "best_val_metric": training_results["best_val_metric"],
        "best_val_loss": training_results["best_val_loss"],
        "training_history": training_results["history"],
    }
    with open(training_results_path, "w") as f:
        json.dump(end_of_training_values, f, indent=2)
    logger.info(f"Training results written to: {training_results_path}")

    info_lines = [
        "stage: train_bst_user_id_only",
        f"timestamp: {timestamp}",
        f"runtime_seconds: {runtime:.2f}",
        f"settings: batch_size={batch_size}, bst_additional_batch_negatives={bst_additional_batch_negatives}, lr={learning_rate}, epochs={epochs}, max_history_len={max_history_len}, early_stopping_min_delta={early_stopping_min_delta}",
        f"post_liker_user_table_num_rows: {post_liker_user_table_num_rows}",
        f"train_samples: {len(train_dataset)}",
        f"val_samples: {len(val_dataset)}",
        f"val_unseen_samples: {len(val_unseen_dataset)}",
        f"primary_metric_name: {training_results['primary_metric_name']}",
        f"best_val_metric: {training_results['best_val_metric']:.4f}",
    ]
    info_lines.extend(stage_info_metric_lines(final_split_metrics))
    (out_dir / "stage_info.txt").write_text("\n".join(info_lines) + "\n")

    logger.info(f"BST user-ID-only ranker training completed in {runtime:.2f}s")

    return {
        "output_dir": out_dir,
        "artifacts": {
            "model_path": str(model_path) if model_path else None,
            "training_config": str(training_config_path),
            "training_results": str(training_results_path),
        },
    }
