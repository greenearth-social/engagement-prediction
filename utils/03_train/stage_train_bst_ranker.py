#!/usr/bin/env python3

"""Stage 3 model components and training for a BST heavy ranker."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared.input_data_helpers import AUTHOR_PAD_IDX, AUTHOR_UNK_IDX
from utils.dataloaders import (
    RankerPairDataset,
    build_ranker_pair_negative_pools_by_split_window,
    create_ranker_pair_data_loaders,
    get_author_table_num_rows,
    load_bucketed_training_data,
)
from utils.helpers import (
    clear_cuda_memory,
    find_author_idx_artifact_path,
    get_device,
    get_stage_logger,
    log_operation_start,
    log_prior_stage_inputs,
    plot_training_history,
    set_random_seeds,
)
from utils.matrix_ranking import stage_info_metric_lines
from utils.pipeline.core import Context


STAGE_LOG_NAME = "STAGE_03_TRAIN_BST_RANKER"


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


def bucketize_time_deltas_hours(
    time_deltas_hours: torch.Tensor,
    boundaries_hours: Sequence[float],
) -> torch.Tensor:
    """Map raw hour deltas to embedding-table bucket IDs."""
    boundaries = _validate_time_delta_bucket_boundaries(boundaries_hours)
    deltas = time_deltas_hours
    if not torch.is_floating_point(deltas):
        deltas = deltas.to(dtype=torch.float32)
    deltas = torch.clamp(deltas, min=0.0)
    boundary_tensor = torch.tensor(boundaries, device=deltas.device, dtype=deltas.dtype)
    positive_bucket_ids = torch.bucketize(deltas, boundary_tensor, right=False) + 1
    zero_bucket_ids = torch.zeros_like(positive_bucket_ids)
    return torch.where(deltas <= 0.0, zero_bucket_ids, positive_bucket_ids).to(dtype=torch.long)


class BSTPostAuthorFeatureEncoder(nn.Module):
    """Fuse MiniLM post embeddings with author embeddings for the BST ranker."""

    def __init__(
        self,
        post_embedding_dim: int,
        author_table_num_rows: int,
        author_embedding_dim: int,
        model_dim: int,
        author_unknown_dropout_rate: float,
    ):
        super().__init__()
        if post_embedding_dim <= 0:
            raise ValueError("post_embedding_dim must be positive")
        if author_table_num_rows < 2:
            raise ValueError("author_table_num_rows must be at least 2")
        if author_embedding_dim <= 0:
            raise ValueError("author_embedding_dim must be positive")
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if not 0.0 <= author_unknown_dropout_rate <= 1.0:
            raise ValueError("author_unknown_dropout_rate must be in [0, 1]")

        self.post_embedding_dim = int(post_embedding_dim)
        self.model_dim = int(model_dim)
        self.author_unknown_dropout_rate = float(author_unknown_dropout_rate)
        self.author_embedding = nn.Embedding(
            num_embeddings=int(author_table_num_rows),
            embedding_dim=int(author_embedding_dim),
            padding_idx=AUTHOR_PAD_IDX,
        )
        nn.init.xavier_uniform_(self.author_embedding.weight)
        with torch.no_grad():
            self.author_embedding.weight[AUTHOR_PAD_IDX].zero_()

        self.fusion_layer = nn.Linear(
            int(post_embedding_dim) + int(author_embedding_dim),
            int(model_dim),
        )
        nn.init.xavier_uniform_(self.fusion_layer.weight)
        if self.fusion_layer.bias is not None:
            nn.init.zeros_(self.fusion_layer.bias)

    def forward(
        self,
        post_embeddings: torch.Tensor,
        author_indices: torch.Tensor,
    ) -> torch.Tensor:
        if post_embeddings.size(-1) != self.post_embedding_dim:
            raise ValueError(
                f"post_embeddings last dimension ({post_embeddings.size(-1)}) must match post_embedding_dim ({self.post_embedding_dim})"
            )
        if post_embeddings.shape[:-1] != author_indices.shape:
            raise ValueError("author_indices shape must match post_embeddings leading dimensions")

        author_indices = author_indices.to(device=post_embeddings.device, dtype=torch.long)
        if self.training and self.author_unknown_dropout_rate > 0.0:
            eligible = author_indices > AUTHOR_UNK_IDX
            if torch.any(eligible):
                dropout_mask = torch.rand(author_indices.shape, device=author_indices.device) < self.author_unknown_dropout_rate
                author_indices = torch.where(
                    eligible & dropout_mask,
                    torch.full_like(author_indices, AUTHOR_UNK_IDX),
                    author_indices,
                )

        author_embeddings = self.author_embedding(author_indices)
        fused_inputs = torch.cat([post_embeddings, author_embeddings], dim=-1)
        return self.fusion_layer(fused_inputs)


class LinearPredictionHead(nn.Module):
    """Linear-layer prediction head for BST candidate-pair encodings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout_rate: float,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be in [0, 1]")

        hidden_dims = tuple(int(hidden_dim) for hidden_dim in hidden_dims)
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden_dims must contain only positive values")

        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout_rate)),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, encoded_pair: torch.Tensor) -> torch.Tensor:
        return self.network(encoded_pair).squeeze(-1)


class BSTRanker(nn.Module):
    """Behavior Sequence Transformer encoder for one user-history/candidate pair."""

    def __init__(
        self,
        post_embedding_dim: int,
        author_table_num_rows: int,
        author_embedding_dim: int,
        model_dim: int,
        time_embedding_dim: int,
        num_attention_heads: int,
        num_transformer_layers: int,
        transformer_ff_dim: int,
        dropout_rate: float,
        author_unknown_dropout_rate: float,
        norm_first: bool,
        time_delta_bucket_boundaries_hours: Sequence[float],
        prediction_hidden_dims: Sequence[int],
    ):
        super().__init__()
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

        self.post_embedding_dim = int(post_embedding_dim)
        self.model_dim = int(model_dim)
        self.time_embedding_dim = int(time_embedding_dim)
        self.time_delta_bucket_boundaries_hours = _validate_time_delta_bucket_boundaries(
            time_delta_bucket_boundaries_hours
        )
        self.num_time_delta_buckets = len(self.time_delta_bucket_boundaries_hours) + 2
        self.transformer_input_dim = self.model_dim + self.time_embedding_dim
        if self.transformer_input_dim % int(num_attention_heads) != 0:
            raise ValueError("model_dim + time_embedding_dim must be divisible by num_attention_heads")

        self.post_feature_encoder = BSTPostAuthorFeatureEncoder(
            post_embedding_dim=post_embedding_dim,
            author_table_num_rows=author_table_num_rows,
            author_embedding_dim=author_embedding_dim,
            model_dim=model_dim,
            author_unknown_dropout_rate=author_unknown_dropout_rate,
        )
        self.time_delta_embedding = nn.Embedding(
            num_embeddings=self.num_time_delta_buckets,
            embedding_dim=self.time_embedding_dim,
        )
        nn.init.xavier_uniform_(self.time_delta_embedding.weight)

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
            input_dim=self.transformer_input_dim,
            hidden_dims=prediction_hidden_dims,
            dropout_rate=dropout_rate,
        )

    def _forward_transformer(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        candidate_post_author_idx: torch.Tensor,
    ) -> torch.Tensor:
        if history_embeddings.dim() != 3:
            raise ValueError("history_embeddings must have shape [B, H, D]")
        if candidate_post_embeddings.dim() != 2:
            raise ValueError("candidate_post_embeddings must have shape [B, D]")
        batch_size, max_history_len, embed_dim = history_embeddings.shape
        if embed_dim != self.post_embedding_dim:
            raise ValueError(
                f"history_embeddings last dimension ({embed_dim}) must match post_embedding_dim ({self.post_embedding_dim})"
            )
        if candidate_post_embeddings.shape != (batch_size, self.post_embedding_dim):
            raise ValueError("candidate_post_embeddings must have shape [B, post_embedding_dim]")
        if history_mask.shape != (batch_size, max_history_len):
            raise ValueError("history_mask must have shape [B, H]")
        if history_time_deltas_hours.shape != (batch_size, max_history_len):
            raise ValueError("history_time_deltas_hours must have shape [B, H]")
        if history_author_indices.shape != (batch_size, max_history_len):
            raise ValueError("history_author_indices must have shape [B, H]")
        if candidate_post_author_idx.shape != (batch_size,):
            raise ValueError("candidate_post_author_idx must have shape [B]")

        device = history_embeddings.device
        history_mask = history_mask.to(device=device, dtype=torch.bool)
        history_time_deltas_hours = history_time_deltas_hours.to(device=device)
        candidate_post_embeddings = candidate_post_embeddings.to(device=device)
        history_author_indices = history_author_indices.to(device=device, dtype=torch.long)
        candidate_post_author_idx = candidate_post_author_idx.to(device=device, dtype=torch.long)

        history_post_vectors = self.post_feature_encoder(history_embeddings, history_author_indices)
        candidate_post_vector = self.post_feature_encoder(
            candidate_post_embeddings,
            candidate_post_author_idx,
        ).unsqueeze(1)
        post_sequence = torch.cat([history_post_vectors, candidate_post_vector], dim=1)

        candidate_time_delta = torch.zeros((batch_size, 1), device=device, dtype=history_time_deltas_hours.dtype)
        sequence_time_deltas = torch.cat([history_time_deltas_hours, candidate_time_delta], dim=1)
        time_bucket_ids = bucketize_time_deltas_hours(
            sequence_time_deltas,
            self.time_delta_bucket_boundaries_hours,
        )
        time_embeddings = self.time_delta_embedding(time_bucket_ids)
        transformer_input = torch.cat([post_sequence, time_embeddings], dim=-1)

        candidate_is_not_padding = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
        src_key_padding_mask = torch.cat([~history_mask, candidate_is_not_padding], dim=1)
        encoded_sequence = self.transformer_encoder(
            transformer_input,
            src_key_padding_mask=src_key_padding_mask,
        )
        return encoded_sequence[:, -1, :]

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        candidate_post_author_idx: torch.Tensor,
    ) -> torch.Tensor:
        transformer_output = self._forward_transformer(
            history_embeddings=history_embeddings,
            history_mask=history_mask,
            history_time_deltas_hours=history_time_deltas_hours,
            candidate_post_embeddings=candidate_post_embeddings,
            history_author_indices=history_author_indices,
            candidate_post_author_idx=candidate_post_author_idx,
        )
        logits = self.prediction_head(transformer_output)
        if logits.dim() == 2 and logits.shape == (transformer_output.size(0), 1):
            logits = logits.squeeze(-1)
        if logits.shape != (transformer_output.size(0),):
            raise RuntimeError("prediction_head must return logits with shape [B] or [B, 1]")
        return logits

    def predict_proba(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        candidate_post_author_idx: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.forward(
                history_embeddings=history_embeddings,
                history_mask=history_mask,
                history_time_deltas_hours=history_time_deltas_hours,
                candidate_post_embeddings=candidate_post_embeddings,
                history_author_indices=history_author_indices,
                candidate_post_author_idx=candidate_post_author_idx,
            )
        )


def _flatten_ranker_pair_batch(batch: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    history_embeddings = batch["history_embeddings"].to(device, non_blocking=True)
    history_mask = batch["history_mask"].to(device, non_blocking=True)
    history_time_deltas_hours = batch["history_time_deltas_hours"].to(device, non_blocking=True)
    candidate_post_embeddings = batch["candidate_post_embeddings"].to(device, non_blocking=True)
    candidate_labels = batch["candidate_labels"].to(device, dtype=torch.float32, non_blocking=True)
    if "history_author_indices" not in batch or "candidate_post_author_idx" not in batch:
        raise RuntimeError("BST ranker batches must include author index tensors")
    history_author_indices = batch["history_author_indices"].to(device, dtype=torch.long, non_blocking=True)
    candidate_post_author_idx = batch["candidate_post_author_idx"].to(device, dtype=torch.long, non_blocking=True)

    if candidate_post_embeddings.dim() != 3:
        raise RuntimeError("candidate_post_embeddings must have shape [B, C, D]")
    if candidate_labels.shape != candidate_post_embeddings.shape[:2]:
        raise RuntimeError("candidate_labels must have shape [B, C]")
    if candidate_post_author_idx.shape != candidate_post_embeddings.shape[:2]:
        raise RuntimeError("candidate_post_author_idx must have shape [B, C]")

    batch_size, num_candidates, embed_dim = candidate_post_embeddings.shape
    return {
        "history_embeddings": history_embeddings.repeat_interleave(num_candidates, dim=0),
        "history_mask": history_mask.repeat_interleave(num_candidates, dim=0),
        "history_time_deltas_hours": history_time_deltas_hours.repeat_interleave(num_candidates, dim=0),
        "candidate_post_embeddings": candidate_post_embeddings.reshape(batch_size * num_candidates, embed_dim),
        "history_author_indices": history_author_indices.repeat_interleave(num_candidates, dim=0),
        "candidate_post_author_idx": candidate_post_author_idx.reshape(batch_size * num_candidates),
        "labels": candidate_labels.reshape(batch_size * num_candidates),
    }


def _compute_bst_loss_and_preds(
    model: BSTRanker,
    batch: Dict[str, Any],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flattened = _flatten_ranker_pair_batch(batch, device)
    labels = flattened["labels"]
    logits = model(
        history_embeddings=flattened["history_embeddings"],
        history_mask=flattened["history_mask"],
        history_time_deltas_hours=flattened["history_time_deltas_hours"],
        candidate_post_embeddings=flattened["candidate_post_embeddings"],
        history_author_indices=flattened["history_author_indices"],
        candidate_post_author_idx=flattened["candidate_post_author_idx"],
    )
    if logits.shape != labels.shape:
        raise RuntimeError("Expected BST logits and labels to have matching [num_pairs] shapes")
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss, logits, labels


def _classification_metrics_from_logits(labels: torch.Tensor, logits: torch.Tensor) -> Dict[str, Any]:
    labels_np = labels.detach().cpu().numpy()
    logits_np = logits.detach().cpu().numpy()
    positive_count = int(labels.detach().sum().item())
    metrics: Dict[str, Any] = {
        "classification_metric_pair_count": int(labels.numel()),
        "classification_metric_positive_count": positive_count,
    }
    if labels.numel() > 0 and torch.unique(labels.detach().cpu()).numel() > 1:
        metrics["auc_roc"] = float(roc_auc_score(labels_np, logits_np))
    else:
        metrics["auc_roc"] = None
    if positive_count > 0:
        metrics["average_precision"] = float(average_precision_score(labels_np, logits_np))
    else:
        metrics["average_precision"] = None
    return metrics


def run_bst_epoch(
    *,
    train: bool,
    split_name: str,
    model: BSTRanker,
    device: str,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    disable_progress: bool,
    gradient_clip_max_norm: float,
    compute_classification_metrics: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    if train:
        if optimizer is None:
            raise ValueError("optimizer is required when train=True")
        model.train()
    else:
        model.eval()

    loss_sum = torch.zeros((), device=device)
    batches = 0
    pair_count = 0
    positive_count = 0
    label_chunks: List[torch.Tensor] = []
    logit_chunks: List[torch.Tensor] = []

    with nullcontext() if train else torch.inference_mode():
        for batch in tqdm(dataloader, desc=split_name, leave=False, disable=disable_progress):
            if train and optimizer is not None:
                optimizer.zero_grad()

            loss, logits, labels = _compute_bst_loss_and_preds(model, batch, device)

            if train and optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)
                optimizer.step()

            loss_sum += loss.detach()
            batches += 1
            if compute_classification_metrics:
                label_chunks.append(labels.detach().cpu())
                logit_chunks.append(logits.detach().cpu())
            else:
                pair_count += int(labels.numel())
                positive_count += int(labels.detach().sum().item())

    loss = (loss_sum / max(batches, 1)).item()
    if compute_classification_metrics and label_chunks:
        all_labels = torch.cat(label_chunks)
        all_logits = torch.cat(logit_chunks)
        metrics = _classification_metrics_from_logits(all_labels, all_logits)
    elif compute_classification_metrics:
        metrics = {
            "classification_metric_pair_count": 0,
            "classification_metric_positive_count": 0,
            "auc_roc": None,
            "average_precision": None,
        }
    else:
        metrics = {
            "classification_metric_pair_count": pair_count,
            "classification_metric_positive_count": positive_count,
        }
    metrics["loss"] = loss
    return loss, metrics


def _log_bst_epoch_metrics(
    experiment_tracker: Optional[Any],
    iteration: int,
    train_loss: float,
    val_loss: float,
    val_unseen_loss: float,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    val_unseen_metrics: Dict[str, Any],
) -> None:
    if experiment_tracker is None:
        return
    experiment_tracker.log_scalar("Training Loss History", "Train Loss", float(train_loss), iteration)
    experiment_tracker.log_scalar("Training Loss History", "Validation Loss", float(val_loss), iteration)
    experiment_tracker.log_scalar("Training Loss History", "Validation Unseen Users Loss", float(val_unseen_loss), iteration)
    for metric_name, metric_label in (("auc_roc", "AUC-ROC"), ("average_precision", "Average Precision")):
        for split_label, metrics in (
            ("Train", train_metrics),
            ("Validation", val_metrics),
            ("Validation Unseen Users", val_unseen_metrics),
        ):
            metric_value = metrics.get(metric_name)
            if metric_value is None:
                continue
            experiment_tracker.log_scalar(
                title=f"{metric_label} by Split",
                series=f"{split_label} {metric_label}",
                value=float(metric_value),
                iteration=iteration,
            )


def train_bst_ranker_model(
    model: BSTRanker,
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
    bst_use_auc_as_primary: bool = False,
    experiment_tracker: Optional[Any] = None,
) -> Dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler_mode = "max" if bst_use_auc_as_primary else "min"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=scheduler_mode, factor=lr_scheduler_factor, patience=lr_scheduler_patience
    )

    primary_metric_name = "val_unseen_auc_roc" if bst_use_auc_as_primary else "val_unseen_loss"
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_unseen_loss": [],
    }
    if bst_use_auc_as_primary:
        history.update({
            "train_auc_roc": [],
            "val_auc_roc": [],
            "val_unseen_auc_roc": [],
            "train_average_precision": [],
            "val_average_precision": [],
            "val_unseen_average_precision": [],
        })
    best_val_metric = float("-inf") if bst_use_auc_as_primary else float("inf")
    best_reset_val_metric = float("-inf") if bst_use_auc_as_primary else float("inf")
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    for epoch in tqdm(range(epochs), desc="Training epochs", disable=disable_progress):
        train_loss, train_metrics = run_bst_epoch(
            train=True,
            split_name="Train",
            model=model,
            device=device,
            dataloader=train_loader,
            optimizer=optimizer,
            disable_progress=disable_progress,
            gradient_clip_max_norm=gradient_clip_max_norm,
            compute_classification_metrics=bst_use_auc_as_primary,
        )
        val_loss, val_metrics = run_bst_epoch(
            train=False,
            split_name="Validation",
            model=model,
            device=device,
            dataloader=val_loader,
            optimizer=None,
            disable_progress=disable_progress,
            gradient_clip_max_norm=gradient_clip_max_norm,
            compute_classification_metrics=bst_use_auc_as_primary,
        )
        val_unseen_loss, val_unseen_metrics = run_bst_epoch(
            train=False,
            split_name="Validation Unseen Users",
            model=model,
            device=device,
            dataloader=val_unseen_loader,
            optimizer=None,
            disable_progress=disable_progress,
            gradient_clip_max_norm=gradient_clip_max_norm,
            compute_classification_metrics=bst_use_auc_as_primary,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_unseen_loss"].append(val_unseen_loss)
        if bst_use_auc_as_primary:
            train_auc = train_metrics.get("auc_roc")
            val_auc = val_metrics.get("auc_roc")
            val_unseen_auc = val_unseen_metrics.get("auc_roc")
            train_ap = train_metrics.get("average_precision")
            val_ap = val_metrics.get("average_precision")
            val_unseen_ap = val_unseen_metrics.get("average_precision")
            history["train_auc_roc"].append(float(train_auc) if train_auc is not None else float("nan"))
            history["val_auc_roc"].append(float(val_auc) if val_auc is not None else float("nan"))
            history["val_unseen_auc_roc"].append(float(val_unseen_auc) if val_unseen_auc is not None else float("nan"))
            history["train_average_precision"].append(float(train_ap) if train_ap is not None else float("nan"))
            history["val_average_precision"].append(float(val_ap) if val_ap is not None else float("nan"))
            history["val_unseen_average_precision"].append(float(val_unseen_ap) if val_unseen_ap is not None else float("nan"))

        _log_bst_epoch_metrics(
            experiment_tracker,
            epoch + 1,
            train_loss,
            val_loss,
            val_unseen_loss,
            train_metrics,
            val_metrics,
            val_unseen_metrics,
        )

        if bst_use_auc_as_primary:
            val_unseen_auc = val_unseen_metrics.get("auc_roc")
            primary_metric = float(val_unseen_auc) if val_unseen_auc is not None else None
        else:
            primary_metric = float(val_unseen_loss)

        if primary_metric is not None:
            scheduler.step(primary_metric)
        else:
            scheduler.step(float("-inf") if bst_use_auc_as_primary else float("inf"))

        better_than_best = (
            primary_metric is not None
            and (
                primary_metric > best_val_metric
                if bst_use_auc_as_primary
                else primary_metric < best_val_metric
            )
        )
        if better_than_best and primary_metric is not None:
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
                    checkpoints_dir / "bst_ranker_best.pth",
                )

        significant_improvement = (
            primary_metric is not None
            and (
                (
                    primary_metric > best_reset_val_metric
                    and (primary_metric - best_reset_val_metric) >= early_stopping_min_delta
                )
                if bst_use_auc_as_primary
                else (
                    primary_metric < best_reset_val_metric
                    and (best_reset_val_metric - primary_metric) >= early_stopping_min_delta
                )
            )
        )
        if primary_metric is not None and significant_improvement:
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


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    device = get_device(args.device)
    timestamp = context.run_timestamp

    run_tag = args.run_tag or ""
    out_dir = context.new_stage_dir("03_train", tag=run_tag)
    checkpoints_dir = out_dir / "checkpoints"
    plots_dir = out_dir / "plots"
    logs_dir = out_dir / "logs"
    for directory in (checkpoints_dir, plots_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    logger = get_stage_logger(STAGE_LOG_NAME, log_file=out_dir / "stage.log")
    log_operation_start("Stage 3 BST ranker training", STAGE_LOG_NAME, logger)
    t0 = time.time()

    clear_cuda_memory()
    random_seed = int(args.random_seed)
    set_random_seeds(random_seed)

    log_operation_start("Load training data from prior stages", STAGE_LOG_NAME, logger)
    embeddings_mmap, likes_core_df, posts_core_df, history_df, author_idx_mapping_df, embed_dim = load_bucketed_training_data(
        context, logger=logger,
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
    time_delta_bucket_boundaries_hours = tuple(float(v) for v in args.bst_time_delta_bucket_boundaries_hours)
    if args.bst_prediction_hidden_dims is None:
        raise ValueError("bst_prediction_hidden_dims is required for BST ranker training")
    prediction_hidden_dims = tuple(int(v) for v in args.bst_prediction_hidden_dims)
    use_author_embedding_table = bool(args.use_author_embedding_table)
    author_embedding_dim = int(args.author_embedding_dim)
    author_unknown_dropout_rate = float(args.author_unknown_dropout_rate)
    batch_size = int(args.batch_size)
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
    bst_use_auc_as_primary = bool(args.bst_use_auc_as_primary)

    if not use_author_embedding_table:
        raise ValueError("BST ranker v1 requires use_author_embedding_table=True")
    if author_idx_mapping_df is None:
        raise FileNotFoundError(
            "author_idx artifact was not found in 01_get_data output, but use_author_embedding_table was enabled."
        )
    author_table_num_rows = get_author_table_num_rows(author_idx_mapping_df)
    logger.info(
        "Author embedding table enabled: "
        f"author_embedding_dim={author_embedding_dim}, "
        f"author_table_num_rows={author_table_num_rows}"
    )
    author_idx_artifact_path = find_author_idx_artifact_path(context)
    if author_idx_artifact_path is None:
        logger.warning("Author embedding table enabled, but no author_idx parquet path was found to log")
    elif context.tracker is not None:
        author_idx_artifact_id = context.tracker.log_file_artifact(
            name="author_idx_mapping",
            path=author_idx_artifact_path,
        )
        logger.info(f"Author index mapping artifact id: {author_idx_artifact_id}")

    num_workers = int(args.num_dataloader_workers)
    pin_memory = bool(args.dataloader_pin_memory)
    persistent_workers = bool(args.dataloader_persistent_workers)
    prefetch_factor = int(args.dataloader_prefetch_factor)

    log_operation_start("Build shared ranker negative pools", STAGE_LOG_NAME, logger)
    negative_pools_by_split_window = build_ranker_pair_negative_pools_by_split_window(
        posts_core_df,
        use_author_embedding_table=use_author_embedding_table,
    )
    train_negative_pool = negative_pools_by_split_window.get("train", {})
    val_negative_pool = negative_pools_by_split_window.get("val", {})

    log_operation_start("Create ranker pair datasets", STAGE_LOG_NAME, logger)
    train_dataset = RankerPairDataset(
        embeddings_mmap=embeddings_mmap,
        likes_core_df=likes_core_df,
        posts_core_df=posts_core_df,
        history_df=history_df,
        split="train",
        max_history_len=max_history_len,
        embed_dim=embed_dim,
        seed=random_seed,
        use_author_embedding_table=use_author_embedding_table,
        sampled_posts_by_bucket=train_negative_pool,
        logger=logger,
    )
    val_dataset = RankerPairDataset(
        embeddings_mmap=embeddings_mmap,
        likes_core_df=likes_core_df,
        posts_core_df=posts_core_df,
        history_df=history_df,
        split="val",
        max_history_len=max_history_len,
        embed_dim=embed_dim,
        seed=random_seed,
        use_author_embedding_table=use_author_embedding_table,
        sampled_posts_by_bucket=val_negative_pool,
        logger=logger,
    )
    val_unseen_dataset = RankerPairDataset(
        embeddings_mmap=embeddings_mmap,
        likes_core_df=likes_core_df,
        posts_core_df=posts_core_df,
        history_df=history_df,
        split="val_unseen_users",
        max_history_len=max_history_len,
        embed_dim=embed_dim,
        seed=random_seed,
        use_author_embedding_table=use_author_embedding_table,
        sampled_posts_by_bucket=val_negative_pool,
        logger=logger,
    )
    train_loader, val_loader, val_unseen_loader, _ = create_ranker_pair_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        val_unseen_dataset=val_unseen_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        seed=random_seed,
    )

    log_operation_start("Create BST ranker model", STAGE_LOG_NAME, logger)
    model = BSTRanker(
        post_embedding_dim=embed_dim,
        author_table_num_rows=author_table_num_rows,
        author_embedding_dim=author_embedding_dim,
        model_dim=model_dim,
        time_embedding_dim=time_embedding_dim,
        num_attention_heads=num_attention_heads,
        num_transformer_layers=num_transformer_layers,
        transformer_ff_dim=transformer_ff_dim,
        dropout_rate=dropout_rate,
        author_unknown_dropout_rate=author_unknown_dropout_rate,
        norm_first=norm_first,
        time_delta_bucket_boundaries_hours=time_delta_bucket_boundaries_hours,
        prediction_hidden_dims=prediction_hidden_dims,
    )

    log_operation_start(f"Train BST ranker (epochs={epochs}, batch_size={batch_size})", STAGE_LOG_NAME, logger)
    training_results = train_bst_ranker_model(
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
        bst_use_auc_as_primary=bst_use_auc_as_primary,
        experiment_tracker=context.tracker,
    )
    trained_model: BSTRanker = training_results["model"]
    clear_cuda_memory()

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
            if primary_metric_name.endswith("_loss"):
                best_epoch = min(valid_metrics, key=lambda item: item[1])[0] if valid_metrics else None
            else:
                best_epoch = max(valid_metrics, key=lambda item: item[1])[0] if valid_metrics else None
        except Exception as exc:
            logger.warning(f"Could not determine best epoch from BST training history: {exc}")
            best_epoch = None
        plot_training_history(hist, plots_dir / f"training_history_{timestamp}.png", best_epoch=best_epoch)

    train_loss, train_metrics = run_bst_epoch(
        train=False,
        split_name="Evaluate train",
        model=trained_model,
        device=device,
        dataloader=train_loader,
        optimizer=None,
        disable_progress=disable_progress,
        gradient_clip_max_norm=gradient_clip_max_norm,
        compute_classification_metrics=bst_use_auc_as_primary,
    )
    val_loss, val_metrics = run_bst_epoch(
        train=False,
        split_name="Evaluate validation",
        model=trained_model,
        device=device,
        dataloader=val_loader,
        optimizer=None,
        disable_progress=disable_progress,
        gradient_clip_max_norm=gradient_clip_max_norm,
        compute_classification_metrics=bst_use_auc_as_primary,
    )
    val_unseen_loss, val_unseen_metrics = run_bst_epoch(
        train=False,
        split_name="Evaluate validation unseen users",
        model=trained_model,
        device=device,
        dataloader=val_unseen_loader,
        optimizer=None,
        disable_progress=disable_progress,
        gradient_clip_max_norm=gradient_clip_max_norm,
        compute_classification_metrics=bst_use_auc_as_primary,
    )
    logger.info(f"Train metrics: {train_metrics}")
    logger.info(f"Validation metrics: {val_metrics}")
    logger.info(f"Validation unseen users metrics: {val_unseen_metrics}")

    config = {
        "model_type": "bst-ranker",
        "post_embedding_dim": embed_dim,
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
        "use_author_embedding_table": use_author_embedding_table,
        "author_embedding_dim": author_embedding_dim,
        "author_unknown_dropout_rate": author_unknown_dropout_rate,
        "author_table_num_rows": author_table_num_rows,
        "author_pad_idx": AUTHOR_PAD_IDX,
        "author_unk_idx": AUTHOR_UNK_IDX,
        "bst_use_auc_as_primary": bst_use_auc_as_primary,
    }

    model_path = None
    if save_model:
        log_operation_start("Save BST ranker checkpoint", STAGE_LOG_NAME, logger)
        model_path = checkpoints_dir / f"bst_ranker_{timestamp}.pth"
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

    final_split_metrics: Dict[str, Dict[str, Any]] = {
        "train": train_metrics,
        "val": val_metrics,
        "val_unseen_users": val_unseen_metrics,
    }
    training_config = {
        **config,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "random_seed": random_seed,
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
    }
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    runtime = time.time() - t0
    info_lines = [
        "stage: train_bst_ranker",
        f"timestamp: {timestamp}",
        f"runtime_seconds: {runtime:.2f}",
        f"settings: batch_size={batch_size}, lr={learning_rate}, epochs={epochs}, max_history_len={max_history_len}, early_stopping_min_delta={early_stopping_min_delta}",
        f"train_samples: {len(train_dataset)}",
        f"val_samples: {len(val_dataset)}",
        f"val_unseen_samples: {len(val_unseen_dataset)}",
        f"primary_metric_name: {training_results['primary_metric_name']}",
        f"best_val_metric: {training_results['best_val_metric']:.4f}",
    ]
    info_lines.extend(stage_info_metric_lines(final_split_metrics))
    (out_dir / "stage_info.txt").write_text("\n".join(info_lines) + "\n")

    logger.info(f"BST ranker training completed in {runtime:.2f}s")

    return {
        "output_dir": out_dir,
        "artifacts": {
            "model_path": str(model_path) if model_path else None,
            "training_config": str(out_dir / "training_config.json"),
        },
    }
