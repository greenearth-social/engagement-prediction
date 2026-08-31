"""Reusable listwise training primitives for the BST ranker."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from engagement_prediction.models.bst_ranker import BSTRanker
from engagement_prediction.training.listwise import (
    run_listwise_epoch,
    train_listwise_model,
)
from engagement_prediction.training.ranking import MatrixBatchScores


def compute_bst_listwise_loss_and_scores(
    model: BSTRanker,
    batch: Dict[str, Any],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score one shared slate and compute a multi-positive listwise loss.

    Each user's positive labels are normalized to a probability distribution,
    so a query with multiple positives contributes the same total target mass
    as a query with one positive.
    """

    label_matrix = batch["label_matrix"]
    # Native loader batches arrive on the host. Validate them before the
    # asynchronous CUDA copy so invalid-data detection does not introduce a
    # device-to-host synchronization into every training step.
    input_positive_counts = label_matrix.sum(dim=1, keepdim=True)
    if torch.any(input_positive_counts <= 0):
        raise RuntimeError(
            "Each user row in label_matrix must contain at least one positive candidate"
        )

    history_embeddings = batch["history_embeddings"].to(
        device, non_blocking=True
    )
    history_mask = batch["history_mask"].to(device, non_blocking=True)
    history_time_deltas_hours = batch["history_time_deltas_hours"].to(
        device, non_blocking=True
    )
    candidate_post_embeddings = batch["candidate_post_embeddings"].to(
        device, non_blocking=True
    )
    labels = label_matrix.to(device, dtype=torch.float32, non_blocking=True)
    if (
        "history_author_indices" not in batch
        or "candidate_post_author_idx" not in batch
    ):
        raise RuntimeError("BST listwise batches must include author index tensors")
    history_author_indices = batch["history_author_indices"].to(
        device, dtype=torch.long, non_blocking=True
    )
    candidate_post_author_idx = batch["candidate_post_author_idx"].to(
        device, dtype=torch.long, non_blocking=True
    )
    history_prior_cumulative_likes = None
    candidate_prior_cumulative_likes = None
    if model.use_popularity_feature:
        if (
            "history_prior_cumulative_likes" not in batch
            or "candidate_prior_cumulative_likes" not in batch
        ):
            raise RuntimeError(
                "BST listwise batches must include popularity tensors when "
                "popularity features are enabled"
            )
        history_prior_cumulative_likes = batch[
            "history_prior_cumulative_likes"
        ].to(device, dtype=torch.float32, non_blocking=True)
        candidate_prior_cumulative_likes = batch[
            "candidate_prior_cumulative_likes"
        ].to(device, dtype=torch.float32, non_blocking=True)

    scores = model.score_candidate_matrix_one_layer(
        history_embeddings=history_embeddings,
        history_mask=history_mask,
        history_time_deltas_hours=history_time_deltas_hours,
        candidate_post_embeddings=candidate_post_embeddings,
        history_author_indices=history_author_indices,
        candidate_post_author_idx=candidate_post_author_idx,
        history_prior_cumulative_likes=history_prior_cumulative_likes,
        candidate_prior_cumulative_likes=candidate_prior_cumulative_likes,
    )
    if scores.shape != labels.shape:
        raise RuntimeError(
            "Expected BST scores and label_matrix to have matching "
            "[num_users, num_candidates] shapes"
        )
    positive_counts = input_positive_counts.to(
        device, dtype=torch.float32, non_blocking=True
    )
    targets = labels / positive_counts
    loss_per_user = -(targets * F.log_softmax(scores, dim=1)).sum(dim=1)
    return loss_per_user.mean(), scores, labels


def run_bst_listwise_epoch(
    *,
    train: bool,
    split_name: str,
    model: BSTRanker,
    device: str,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    disable_progress: bool,
    gradient_clip_max_norm: float,
    metrics_top_ks: List[int],
    calc_baseline_metrics: bool,
    max_batches: Optional[int],
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    """Run the shared listwise epoch while retaining BST's DCG fields."""

    return run_listwise_epoch(
        compute_loss_and_scores=compute_bst_listwise_loss_and_scores,
        include_dcg_metrics=True,
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


class BSTRankerMatrixScorer:
    """Matrix-ranking scorer for an in-memory BST model."""

    def __init__(self, model: BSTRanker):
        self.model = model

    def prepare_for_eval(self, device: str) -> None:
        """Move the eager model once and disable training-only behavior."""

        self.model = self.model.to(device)
        self.model.eval()

    def score_batch(self, batch: Dict[str, Any], device: str) -> MatrixBatchScores:
        """Adapt the BST training primitive to the generic matrix evaluator."""

        loss, scores, _ = compute_bst_listwise_loss_and_scores(
            self.model, batch, device
        )
        return MatrixBatchScores(scores=scores, loss=loss)


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
    metrics_top_ks: List[int],
    bst_max_train_batches_per_epoch: Optional[int],
    checkpoint_metadata: Dict[str, Any],
    best_checkpoint_callback: Optional[Callable[[Path], None]],
    experiment_tracker: Optional[Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Train BST through the model-independent listwise lifecycle."""

    if (
        bst_max_train_batches_per_epoch is not None
        and bst_max_train_batches_per_epoch <= 0
    ):
        raise ValueError(
            "bst_max_train_batches_per_epoch must be positive when provided"
        )
    return train_listwise_model(
        model=model,
        epoch_runner=run_bst_listwise_epoch,
        model_label="BST",
        checkpoint_filename="bst_ranker_best.pth",
        checkpoint_extra_fields={},
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
        max_train_batches_per_epoch=bst_max_train_batches_per_epoch,
        checkpoint_metadata=checkpoint_metadata,
        best_checkpoint_callback=best_checkpoint_callback,
        experiment_tracker=experiment_tracker,
        logger=logger,
    )
