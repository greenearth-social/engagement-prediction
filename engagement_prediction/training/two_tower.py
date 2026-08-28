"""Reusable listwise training primitives for the canonical two-tower model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from engagement_prediction.training.listwise import (
    run_listwise_epoch,
    train_listwise_model,
)
from engagement_prediction.training.ranking import MatrixBatchScores


def compute_two_tower_listwise_loss_and_scores(
    model: nn.Module,
    batch: Dict[str, Any],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode each user and candidate once, then compute matrix listwise loss."""

    label_matrix = batch["label_matrix"]
    # Stage 7 batches are host tensors. Validate labels before the asynchronous
    # device transfer so bad input does not introduce a CUDA synchronization.
    input_positive_counts = label_matrix.sum(dim=1, keepdim=True)
    if torch.any(input_positive_counts <= 0):
        raise RuntimeError(
            "Each user row in label_matrix must contain at least one positive candidate"
        )
    if "history_author_indices" not in batch or "candidate_post_author_idx" not in batch:
        raise RuntimeError("Two-tower listwise batches must include author index tensors")

    history_embeddings = batch["history_embeddings"].to(
        device, non_blocking=True
    )
    history_mask = batch["history_mask"].to(
        device, dtype=torch.bool, non_blocking=True
    )
    history_author_indices = batch["history_author_indices"].to(
        device, dtype=torch.long, non_blocking=True
    )
    candidate_post_embeddings = batch["candidate_post_embeddings"].to(
        device, non_blocking=True
    )
    candidate_post_author_idx = batch["candidate_post_author_idx"].to(
        device, dtype=torch.long, non_blocking=True
    )
    labels = label_matrix.to(device, dtype=torch.float32, non_blocking=True)

    user_embeddings = model.encode_user(
        history_embeddings,
        history_mask,
        history_author_indices,
    )
    post_embeddings = model.encode_post(
        candidate_post_embeddings,
        candidate_post_author_idx,
    )
    if user_embeddings.dim() != 2 or post_embeddings.dim() != 2:
        raise RuntimeError("Two-tower encoders must return rank-two embedding matrices")
    output_embedding_dim = int(model.output_embedding_dim)
    if output_embedding_dim <= 0:
        raise RuntimeError("Two-tower output_embedding_dim must be positive")
    if user_embeddings.shape != (labels.size(0), output_embedding_dim):
        raise RuntimeError(
            "User tower output must have shape [num_users, output_embedding_dim]"
        )
    if post_embeddings.shape != (labels.size(1), output_embedding_dim):
        raise RuntimeError(
            "Post tower output must have shape [num_candidates, output_embedding_dim]"
        )

    similarity_temperature = float(model.similarity_temperature)
    if similarity_temperature <= 0.0:
        raise RuntimeError("Two-tower similarity_temperature must be positive")
    scores = torch.matmul(user_embeddings, post_embeddings.transpose(0, 1))
    scores = scores / similarity_temperature
    if scores.shape != labels.shape:
        raise RuntimeError(
            "Expected two-tower scores and label_matrix to have matching "
            "[num_users, num_candidates] shapes"
        )

    positive_counts = input_positive_counts.to(
        device, dtype=torch.float32, non_blocking=True
    )
    targets = labels / positive_counts
    loss_per_user = -(targets * F.log_softmax(scores, dim=1)).sum(dim=1)
    return loss_per_user.mean(), scores, labels


def run_two_tower_listwise_epoch(
    *,
    train: bool,
    split_name: str,
    model: nn.Module,
    device: str,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    disable_progress: bool,
    gradient_clip_max_norm: float,
    metrics_top_ks: List[int],
    calc_baseline_metrics: bool,
    max_batches: Optional[int],
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    """Run the shared listwise epoch with two-tower's NDCG-only output."""

    return run_listwise_epoch(
        compute_loss_and_scores=compute_two_tower_listwise_loss_and_scores,
        include_dcg_metrics=False,
        zero_grad_set_to_none=True,
        train=train,
        split_name=split_name,
        model=model,
        device=device,
        dataloader=dataloader,
        optimizer=optimizer,
        disable_progress=disable_progress,
        gradient_clip_max_norm=gradient_clip_max_norm,
        metrics_top_ks=metrics_top_ks,
        calc_baseline_metrics=calc_baseline_metrics,
        max_batches=max_batches,
    )


class TwoTowerMatrixScorer:
    """Matrix-ranking scorer for an in-memory canonical two-tower model."""

    def __init__(self, model: nn.Module):
        self.model = model

    def prepare_for_eval(self, device: str) -> None:
        self.model = self.model.to(device)
        self.model.eval()

    def score_batch(self, batch: Dict[str, Any], device: str) -> MatrixBatchScores:
        loss, scores, _ = compute_two_tower_listwise_loss_and_scores(
            self.model,
            batch,
            device,
        )
        return MatrixBatchScores(scores=scores, loss=loss)


def train_two_tower_model(
    *,
    model: nn.Module,
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
    metrics_top_ks: List[int],
    max_train_batches_per_epoch: Optional[int],
    checkpoint_metadata: Dict[str, Any],
    best_checkpoint_callback: Optional[Callable[[Path], None]],
    experiment_tracker: Optional[Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Train two-tower through the model-independent listwise lifecycle."""

    output_embedding_dim = int(model.output_embedding_dim)
    if output_embedding_dim <= 0:
        raise ValueError("model.output_embedding_dim must be positive")
    results = train_listwise_model(
        model=model,
        epoch_runner=run_two_tower_listwise_epoch,
        model_label="Two-tower",
        checkpoint_filename="two_tower_best.pth",
        checkpoint_extra_fields={"output_embedding_dim": output_embedding_dim},
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
        max_train_batches_per_epoch=max_train_batches_per_epoch,
        checkpoint_metadata=checkpoint_metadata,
        best_checkpoint_callback=best_checkpoint_callback,
        experiment_tracker=experiment_tracker,
        logger=logger,
    )
    results["output_embedding_dim"] = output_embedding_dim
    return results
