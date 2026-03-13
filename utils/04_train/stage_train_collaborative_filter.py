#!/usr/bin/env python3

"""
Stage 4 (Collaborative Filter): train a minimal implicit-feedback recommender.

This implementation now follows a positive-only collaborative-filtering setup:
    - Training uses only observed train likes as positive interactions.
    - It does NOT use the pipeline's sampled ``neg_emb_idx`` values for fitting.
    - Unobserved user-item pairs are treated as unlabeled zeros inside a dense
      implicit-feedback logistic matrix-factorization objective.
    - The scoring function is logistic, so the exported score is already a
      probability-like quantity in ``[0, 1]`` that can be thresholded.

The model is intentionally small:
    - One latent vector per user.
    - One latent vector per item.
    - One scalar bias per user.
    - One scalar bias per item.
    - One global bias.

At evaluation time, we score the existing candidate pairs from Stage 2
(``like_uri`` versus sampled ``neg_uri``) so the current Stage 5 evaluator can
keep operating on the same ``y_pred_proba`` parquet outputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from utils.dataloaders import load_training_data
from utils.helpers import (
    clear_cuda_memory,
    get_device,
    get_stage_logger,
    log_operation_start,
    plot_model_performance,
    set_random_seeds,
)
from utils.pipeline.core import Context

STAGE_LOG_NAME = "STAGE_04_TRAIN_COLLABORATIVE_FILTER"


class CollaborativeFilteringModel(nn.Module):
    """Minimal logistic matrix-factorization model for implicit feedback."""

    def __init__(self, num_users: int, num_items: int, latent_dim: int, unknown_user_index: int):
        super().__init__()
        # Store model sizes for checkpoints and debugging.
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        # Store the unseen-user bucket index for mean-user fallback handling.
        self.unknown_user_index = unknown_user_index
        # Learn one latent vector per user.
        self.user_factors = nn.Embedding(num_users, latent_dim)
        # Learn one latent vector per item.
        self.item_factors = nn.Embedding(num_items, latent_dim)
        # Learn one scalar bias per user.
        self.user_bias = nn.Embedding(num_users, 1)
        # Learn one scalar bias per item.
        self.item_bias = nn.Embedding(num_items, 1)
        # Learn one scalar bias shared by all interactions.
        self.global_bias = nn.Parameter(torch.zeros(1))
        # Initialize user factors with a small Gaussian.
        nn.init.normal_(self.user_factors.weight, mean=0.0, std=0.02)
        # Initialize item factors with the same small Gaussian.
        nn.init.normal_(self.item_factors.weight, mean=0.0, std=0.02)
        # Initialize user biases at zero.
        nn.init.zeros_(self.user_bias.weight)
        # Initialize item biases at zero.
        nn.init.zeros_(self.item_bias.weight)

    def _resolve_user_representation(self, user_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Look up the latent vector for each requested user.
        user_vec = self.user_factors(user_index)
        # Look up the user bias for each requested user.
        user_bias = self.user_bias(user_index)
        # Identify which positions correspond to the unseen-user fallback bucket.
        unknown_mask = user_index == self.unknown_user_index
        # Replace unseen-user rows with the mean seen-user representation when needed.
        if torch.any(unknown_mask):
            # Use the mean of all trained user vectors when at least one seen user exists.
            if self.unknown_user_index > 0:
                mean_user_vec = self.user_factors.weight[:self.unknown_user_index].mean(dim=0)
                # Use the mean of all trained user biases for the same fallback.
                mean_user_bias = self.user_bias.weight[:self.unknown_user_index].mean(dim=0)
            else:
                # Fall back to the reserved unknown-user row only if no seen users exist.
                mean_user_vec = self.user_factors.weight[self.unknown_user_index]
                # Mirror that fallback for the bias term.
                mean_user_bias = self.user_bias.weight[self.unknown_user_index]
            # Swap in the mean-user vector anywhere the batch contains an unseen user.
            user_vec = torch.where(unknown_mask.unsqueeze(-1), mean_user_vec.unsqueeze(0), user_vec)
            # Swap in the mean-user bias anywhere the batch contains an unseen user.
            user_bias = torch.where(unknown_mask.unsqueeze(-1), mean_user_bias.unsqueeze(0), user_bias)
        # Return the resolved user vectors and biases.
        return user_vec, user_bias

    def logits(self, user_index: torch.Tensor, item_index: torch.Tensor) -> torch.Tensor:
        # Resolve user vectors, replacing unseen users with the mean seen-user representation.
        user_vec, user_bias = self._resolve_user_representation(user_index)
        # Look up the latent vector for each item in the batch.
        item_vec = self.item_factors(item_index)
        # Compute the dot-product interaction term.
        interaction = (user_vec * item_vec).sum(dim=-1)
        # Look up the item bias terms.
        item_bias = self.item_bias(item_index).squeeze(-1)
        # Return the unnormalized logit.
        return interaction + user_bias.squeeze(-1) + item_bias + self.global_bias

    def forward(self, user_index: torch.Tensor, item_index: torch.Tensor) -> torch.Tensor:
        # Convert the pairwise logits into probabilities.
        return torch.sigmoid(self.logits(user_index, item_index))

    def logits_for_all_items(self, user_index: torch.Tensor, item_count: int) -> torch.Tensor:
        # Resolve user vectors, replacing unseen users with the mean seen-user representation.
        user_vec, user_bias = self._resolve_user_representation(user_index)
        # Slice the trainable item vectors, excluding the unknown-item bucket.
        item_vec = self.item_factors.weight[:item_count]
        # Compute all user-item dot products at once.
        interaction = user_vec @ item_vec.T
        # Slice the item bias row.
        item_bias = self.item_bias.weight[:item_count].T
        # Broadcast biases across the full user-item score matrix.
        return interaction + user_bias + item_bias + self.global_bias

    def compute_loss_and_preds(self, batch: Dict[str, Any], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # Move user indices onto the active device.
        user_index = batch["user_index"].to(device)
        # Move item indices onto the active device.
        item_index = batch["item_index"].to(device)
        # Move labels onto the active device.
        labels = batch["label"].to(device)
        # Score the candidate pairs.
        preds = self(user_index, item_index)
        # Compute the standard BCE loss for exported candidate-pair evaluation.
        loss = F.binary_cross_entropy(preds, labels)
        # Return both the loss and the probabilities.
        return loss, preds


class CandidateInteractionDataset(Dataset):
    """Pairwise candidate dataset used for val/holdout scoring and exports."""

    def __init__(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        labels: np.ndarray,
        user_ids: List[str],
        post_ids: List[str],
    ):
        # Store encoded user indices as a tensor.
        self.user_indices = torch.as_tensor(user_indices, dtype=torch.long)
        # Store encoded item indices as a tensor.
        self.item_indices = torch.as_tensor(item_indices, dtype=torch.long)
        # Store binary labels as a float tensor.
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        # Keep original user IDs for parquet outputs.
        self.user_ids = user_ids
        # Keep original post IDs for parquet outputs.
        self.post_ids = post_ids

    def __len__(self) -> int:
        # Return the number of candidate rows.
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Return the encoded user index.
        user_index = self.user_indices[idx]
        # Return the encoded item index.
        item_index = self.item_indices[idx]
        # Return the binary candidate label.
        label = self.labels[idx]
        # Return the original user ID.
        user_id = self.user_ids[idx]
        # Return the original post ID.
        post_id = self.post_ids[idx]
        # Package everything into the standard batch dict.
        return {
            "user_index": user_index,
            "item_index": item_index,
            "label": label,
            "user_id": user_id,
            "post_id": post_id,
        }


def _build_user_index(
    target_posts_df: pl.DataFrame,
    logger,
) -> Tuple[Dict[str, int], int]:
    """Build the train-user vocabulary and reserve one unseen-user bucket."""
    # Restrict the user vocabulary to train rows to avoid leakage.
    train_users = (
        target_posts_df
        .filter(pl.col("split") == "train")
        .select("target_did")
        .unique()
        .sort("target_did")
        .get_column("target_did")
        .to_list()
    )
    # Map every train user to a compact integer index.
    user_to_index = {did: idx for idx, did in enumerate(train_users)}
    # Reserve one extra index for any user absent from training.
    unknown_user_index = len(train_users)
    # Log the final vocabulary size.
    logger.info(
        f"User vocabulary built from train split: {len(train_users):,} seen users + 1 unknown-user bucket"
    )
    # Return the mapping plus the fallback index.
    return user_to_index, unknown_user_index


def _build_item_index(
    target_posts_df: pl.DataFrame,
    logger,
) -> Tuple[Dict[int, int], int]:
    """Build a compact item vocabulary from train-split positive likes only."""
    # Collect every observed positive train item index and ignore eval-time items entirely.
    train_like_items = (
        target_posts_df
        .filter((pl.col("split") == "train") & pl.col("like_emb_idx").is_not_null())
        .select("like_emb_idx")
        .unique()
        .sort("like_emb_idx")
        .get_column("like_emb_idx")
        .to_list()
    )
    # Deduplicate and sort the raw train-positive embedding indices for stable mapping.
    raw_item_ids = [int(item_id) for item_id in train_like_items]
    # Map every raw Stage 1 ``emb_idx`` into a compact item index.
    item_to_index = {raw_item_id: idx for idx, raw_item_id in enumerate(raw_item_ids)}
    # Reserve one extra item slot for defensive fallback handling.
    unknown_item_index = len(item_to_index)
    # Log the compact item vocabulary size and the train-only source of truth.
    logger.info(
        f"Item vocabulary built from train positive likes only: {len(raw_item_ids):,} known items + 1 unknown-item bucket"
    )
    # Return the compact mapping and the fallback index.
    return item_to_index, unknown_item_index


def _build_train_positive_lookup(
    target_posts_df: pl.DataFrame,
    user_to_index: Dict[str, int],
    item_to_index: Dict[int, int],
    logger,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, int]:
    """Build the positive-only user -> liked-items lookup from the train split."""
    # Keep only the positive train interactions.
    train_pairs = (
        target_posts_df
        .filter((pl.col("split") == "train") & pl.col("like_emb_idx").is_not_null())
        .select(["target_did", "like_emb_idx"])
        .unique()
        .sort(["target_did", "like_emb_idx"])
    )
    # Initialize the lookup map from compact user index to compact item IDs.
    user_positive_items: Dict[int, List[int]] = {}
    # Iterate through every unique train positive interaction.
    for target_did, like_emb_idx in train_pairs.iter_rows():
        # Resolve the compact user index.
        user_index = user_to_index[str(target_did)]
        # Resolve the compact item index.
        item_index = item_to_index[int(like_emb_idx)]
        # Append the positive item to this user's interaction list.
        user_positive_items.setdefault(user_index, []).append(item_index)
    # Convert each positive-item list into a compact numpy array.
    compact_lookup = {
        user_index: np.asarray(item_indices, dtype=np.int64)
        for user_index, item_indices in user_positive_items.items()
    }
    # Build the ordered array of train-user indices.
    train_user_indices = np.asarray(sorted(compact_lookup.keys()), dtype=np.int64)
    # Count how many unique positive user-item pairs exist.
    num_positive_pairs = int(sum(len(item_indices) for item_indices in compact_lookup.values()))
    # Log the positive-only interaction count.
    logger.info(
        f"Positive-only train matrix built from observed likes: {len(train_user_indices):,} users, "
        f"{num_positive_pairs:,} unique user-item positives"
    )
    # Return the lookup, the ordered train users, and the positive count.
    return compact_lookup, train_user_indices, num_positive_pairs


def _auto_balance_positive_weight(
    num_users: int,
    num_items: int,
    num_positive_pairs: int,
) -> float:
    """Choose the implicit positive weight by balancing positives vs unlabeled zeros."""
    # Count the total number of cells in the implicit user-item matrix.
    total_pairs = max(num_users * num_items, 1)
    # Count how many cells are unobserved.
    num_zero_pairs = max(total_pairs - num_positive_pairs, 1)
    # Balance the total positive weight against the total zero weight.
    return float(num_zero_pairs / max(num_positive_pairs, 1))


def _sanitize_item_indices(
    raw_item_indices: np.ndarray,
    item_to_index: Dict[int, int],
    unknown_item_index: int,
) -> np.ndarray:
    """Map raw Stage 1 ``emb_idx`` values into compact item IDs."""
    # Convert every raw item ID into its compact vocabulary index when available.
    mapped = [item_to_index.get(int(raw_item_index), unknown_item_index) for raw_item_index in raw_item_indices.tolist()]
    # Return the compact item-index array.
    return np.asarray(mapped, dtype=np.int64)


def _interleave_post_ids(like_uris: List[str], neg_uris: List[str]) -> List[str]:
    """Build the alternating [positive, negative, positive, negative, ...] post ID list."""
    # Pre-allocate the final output length.
    post_ids: List[str] = [""] * (2 * len(like_uris))
    # Place positive post IDs in even positions.
    post_ids[0::2] = like_uris
    # Place negative post IDs in odd positions.
    post_ids[1::2] = neg_uris
    # Return the alternating list.
    return post_ids


def _build_candidate_dataset(
    target_posts_df: pl.DataFrame,
    split: str,
    user_to_index: Dict[str, int],
    unknown_user_index: int,
    item_to_index: Dict[int, int],
    unknown_item_index: int,
    logger,
) -> Tuple[CandidateInteractionDataset, Dict[str, int]]:
    """Build the candidate-pair dataset used for validation and holdout scoring."""
    # Filter to the requested split and keep only rows with a candidate negative.
    split_rows = target_posts_df.filter(
        (pl.col("split") == split) & pl.col("neg_emb_idx").is_not_null()
    )
    # Log how many target rows survived the split filter.
    logger.info(f"  Split '{split}': {len(split_rows):,} target rows (candidate scoring set)")
    # Extract the raw user IDs.
    target_dids = split_rows["target_did"].to_list()
    # Extract the raw positive item IDs.
    like_item_indices = split_rows["like_emb_idx"].to_numpy().astype(np.int64)
    # Extract the raw negative item IDs.
    neg_item_indices = split_rows["neg_emb_idx"].to_numpy().astype(np.int64)
    # Extract the positive post URIs for parquet output.
    like_uris = split_rows["like_uri"].to_list()
    # Extract the negative post URIs for parquet output.
    neg_uris = split_rows["neg_uri"].to_list()
    # Count the split's target rows.
    n_rows = len(target_dids)
    # Map raw users into the train vocabulary plus unseen-user fallback.
    mapped_users = np.asarray([user_to_index.get(did, unknown_user_index) for did in target_dids], dtype=np.int64)
    # Map raw positive items into the compact item vocabulary.
    safe_like_items = _sanitize_item_indices(like_item_indices, item_to_index, unknown_item_index)
    # Map raw negative items into the compact item vocabulary.
    safe_neg_items = _sanitize_item_indices(neg_item_indices, item_to_index, unknown_item_index)
    # Interleave user indices so each row yields [positive, negative].
    user_indices = np.empty(2 * n_rows, dtype=np.int64)
    # Assign the positive user rows.
    user_indices[0::2] = mapped_users
    # Assign the negative user rows.
    user_indices[1::2] = mapped_users
    # Interleave positive and negative item indices.
    item_indices = np.empty(2 * n_rows, dtype=np.int64)
    # Assign the positive item rows.
    item_indices[0::2] = safe_like_items
    # Assign the negative item rows.
    item_indices[1::2] = safe_neg_items
    # Build the binary candidate labels.
    labels = np.empty(2 * n_rows, dtype=np.float32)
    # Mark positive candidate rows.
    labels[0::2] = 1.0
    # Mark negative candidate rows.
    labels[1::2] = 0.0
    # Duplicate each user ID so the parquet export lines up with the labels.
    user_ids = [did for did in target_dids for _ in (0, 1)]
    # Interleave positive and negative post IDs.
    post_ids = _interleave_post_ids(like_uris, neg_uris)
    # Count how many rows use the unseen-user fallback.
    n_unknown_users = int((mapped_users == unknown_user_index).sum())
    # Count how many pairwise rows use the unknown-item fallback.
    n_unknown_items = int((item_indices == unknown_item_index).sum())
    # Materialize the dataset object.
    dataset = CandidateInteractionDataset(
        user_indices=user_indices,
        item_indices=item_indices,
        labels=labels,
        user_ids=user_ids,
        post_ids=post_ids,
    )
    # Return both the dataset and the audit stats.
    return dataset, {
        "target_rows": n_rows,
        "samples": len(dataset),
        "unknown_users": n_unknown_users,
        "unknown_items": n_unknown_items,
    }


def _create_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    """Create a DataLoader using the repo's standard worker settings."""
    # Start from the loader settings shared by every split.
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    # Only pass worker-specific options when workers are enabled.
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    # Return the configured DataLoader.
    return DataLoader(dataset, **loader_kwargs)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute the small metric set shared by train/val/holdout exports."""
    # Always record the number of evaluated candidate rows.
    metrics: Dict[str, Any] = {"total_samples": int(len(y_true))}
    # Always record the number of positive candidate rows.
    metrics["positive_samples"] = int(y_true.sum())
    # Compute AUC only when both classes are present.
    if len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred))
    # Compute the default thresholded accuracy used elsewhere in the repo.
    metrics["accuracy@0.5"] = float(accuracy_score(y_true, (y_pred > 0.5).astype(int)))
    # Record the mean predicted probability for calibration inspection.
    metrics["mean_pred_proba"] = float(np.mean(y_pred))
    # Return the metric bundle.
    return metrics


def _evaluate_model(
    model: CollaborativeFilteringModel,
    loader: DataLoader,
    device: str,
) -> Dict[str, Any]:
    """Run one inference pass over candidate user-item pairs."""
    # Switch the model into inference mode.
    model.eval()
    # Accumulate predicted probabilities here.
    y_pred: List[float] = []
    # Accumulate labels here.
    y_true: List[float] = []
    # Accumulate raw user IDs here.
    user_ids: List[str] = []
    # Accumulate raw post IDs here.
    post_ids: List[str] = []
    # Disable autograd for evaluation.
    with torch.inference_mode():
        # Iterate through candidate batches.
        for batch in loader:
            # Reuse the pairwise scoring helper.
            _, preds = model.compute_loss_and_preds(batch, device)
            # Extend the probability list.
            y_pred.extend(preds.cpu().numpy().tolist())
            # Extend the label list.
            y_true.extend(batch["label"].cpu().numpy().tolist())
            # Extend the raw user ID list.
            user_ids.extend(batch["user_id"])
            # Extend the raw post ID list.
            post_ids.extend(batch["post_id"])
    # Convert labels into one numpy array.
    y_true_arr = np.asarray(y_true, dtype=np.float32)
    # Convert predictions into one numpy array.
    y_pred_arr = np.asarray(y_pred, dtype=np.float32)
    # Compute the summary metrics.
    metrics = _compute_metrics(y_true_arr, y_pred_arr)
    # Return both metrics and parquet-ready prediction columns.
    return {
        "metrics": metrics,
        "predictions": {
            "user_id": user_ids,
            "post_id": post_ids,
            "y_true": y_true_arr,
            "y_pred": y_pred_arr,
        },
    }


def _candidate_bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute BCE on candidate-pair probabilities for validation monitoring."""
    # Convert candidate labels into a torch tensor.
    labels = torch.as_tensor(y_true, dtype=torch.float32)
    # Convert candidate probabilities into a torch tensor.
    preds = torch.as_tensor(y_pred, dtype=torch.float32)
    # Return the scalar BCE value.
    return float(F.binary_cross_entropy(preds, labels).item())


def _implicit_logistic_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    """Logistic MF loss from positive-only implicit data.

    This implements the objective from Logistic Matrix Factorization for
    implicit feedback (Johnson, 2014):

        (1 + alpha * r_ui) * softplus(s_ui) - alpha * r_ui * s_ui

    where ``r_ui`` is 1 for observed train likes and 0 for unobserved pairs.
    """
    # Compute the implicit-feedback logistic loss elementwise.
    loss_matrix = (1.0 + positive_weight * targets) * F.softplus(logits) - positive_weight * targets * logits
    # Return the mean loss over the batch's user-item submatrix.
    return loss_matrix.mean()


def _build_batch_target_matrix(
    batch_user_indices: np.ndarray,
    user_positive_items: Dict[int, np.ndarray],
    item_count: int,
    device: str,
) -> torch.Tensor:
    """Build the dense 0/1 implicit target matrix for one user batch."""
    # Start from an all-zero unlabeled matrix.
    targets = torch.zeros((len(batch_user_indices), item_count), dtype=torch.float32, device=device)
    # Mark observed positive items for each user in the batch.
    for row_index, user_index in enumerate(batch_user_indices.tolist()):
        # Look up the compact positive-item list for this user.
        positive_items = user_positive_items.get(int(user_index))
        # Skip users that somehow have no positives.
        if positive_items is None or len(positive_items) == 0:
            continue
        # Materialize the compact item indices on the active device.
        positive_tensor = torch.as_tensor(positive_items, dtype=torch.long, device=device)
        # Mark the observed items as positives in the dense matrix.
        targets[row_index, positive_tensor] = 1.0
    # Return the dense target block.
    return targets


def train_collaborative_filter_model(
    model: CollaborativeFilteringModel,
    train_user_indices: np.ndarray,
    user_positive_items: Dict[int, np.ndarray],
    num_train_items: int,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    user_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    lr_scheduler_factor: float,
    lr_scheduler_patience: int,
    disable_progress: bool,
    positive_weight: float,
) -> Dict[str, Any]:
    """Train the implicit logistic MF model on positive-only train interactions."""
    # Import optimizers lazily to match the other Stage 4 scripts.
    import torch.optim as optim
    # Import the scheduler lazily for the same reason.
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    # Import tqdm lazily to keep module import time small.
    from tqdm import tqdm as _tqdm

    # Move the model onto the active device.
    model = model.to(device)
    # Use AdamW for stable optimization and simple weight decay.
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Reduce the learning rate when validation AUC plateaus.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
    )
    # Track train loss plus candidate-validation loss/AUC over time.
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
    }
    # Initialize best validation AUC tracking.
    best_val_auc = float("-inf")
    # Initialize best validation BCE tracking as a tie-breaker.
    best_val_loss = float("inf")
    # Initialize the no-improvement counter.
    patience_counter = 0
    # Hold the best model weights in memory.
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    # Iterate through optimization epochs.
    for _epoch in _tqdm(range(epochs), desc="Training epochs", disable=disable_progress):
        # Switch into training mode.
        model.train()
        # Shuffle the train-user order for this epoch.
        shuffled_users = np.random.permutation(train_user_indices)
        # Reset the epoch train-loss accumulator.
        epoch_train_loss = 0.0
        # Count how many user batches ran this epoch.
        num_batches = 0

        # Walk through the shuffled users in mini-batches.
        for start in _tqdm(range(0, len(shuffled_users), user_batch_size), desc="Training", leave=False, disable=disable_progress):
            # Slice the current user batch.
            batch_user_indices = shuffled_users[start:start + user_batch_size]
            # Skip empty slices defensively.
            if len(batch_user_indices) == 0:
                continue
            # Convert the compact user indices into a device tensor.
            batch_users = torch.as_tensor(batch_user_indices, dtype=torch.long, device=device)
            # Clear stale gradients.
            optimizer.zero_grad()
            # Score the current user batch against every trainable item.
            logits = model.logits_for_all_items(batch_users, item_count=num_train_items)
            # Build the dense positive-only target block for this batch.
            targets = _build_batch_target_matrix(
                batch_user_indices=batch_user_indices,
                user_positive_items=user_positive_items,
                item_count=num_train_items,
                device=device,
            )
            # Compute the implicit logistic MF loss.
            loss = _implicit_logistic_loss(logits, targets, positive_weight=positive_weight)
            # Backpropagate through user/item factors and biases.
            loss.backward()
            # Apply the optimizer step.
            optimizer.step()
            # Accumulate the epoch train loss.
            epoch_train_loss += float(loss.item())
            # Count the completed user batch.
            num_batches += 1

        # Normalize the epoch train loss.
        train_loss = epoch_train_loss / max(num_batches, 1)
        # Score the candidate validation pairs for early stopping.
        val_eval = _evaluate_model(model, val_loader, device)
        # Read the validation AUC from the candidate evaluation.
        val_auc = float(val_eval["metrics"].get("auc_roc", 0.0))
        # Compute BCE on the validation candidate predictions.
        val_loss = _candidate_bce_loss(
            val_eval["predictions"]["y_true"],
            val_eval["predictions"]["y_pred"],
        )
        # Append the epoch metrics to the history bundle.
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        # Let the LR scheduler react to candidate validation AUC.
        scheduler.step(val_auc)

        # Prefer higher validation AUC, then lower validation BCE.
        is_better = (val_auc > best_val_auc) or (val_auc == best_val_auc and val_loss < best_val_loss)
        # Save the best in-memory weights when validation improves.
        if is_better:
            # Update the best validation AUC.
            best_val_auc = val_auc
            # Update the best validation BCE.
            best_val_loss = val_loss
            # Clone the model weights into memory.
            best_state_dict = copy.deepcopy(model.state_dict())
            # Reset patience because we improved.
            patience_counter = 0
        else:
            # Count one more epoch without improvement.
            patience_counter += 1
            # Stop once the patience budget is exhausted.
            if patience_counter >= patience:
                break

    # Restore the best model weights before final scoring/export.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Return the trained model plus the tracked history.
    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_auc": best_val_auc,
    }


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    # Resolve the root pipeline run directory.
    run_dir = Path(context.run_dir).resolve()
    # Resolve the active device once at the top of the stage.
    device = get_device(args.device)
    # Reuse the pipeline run timestamp for outputs.
    timestamp = context.run_timestamp

    # --- output directories ---
    # Preserve the repo's run-tag behavior for sweep outputs.
    run_tag = args.run_tag or ""
    # Create the timestamped Stage 4 directory.
    out_dir = context.new_stage_dir("04_train", tag=run_tag)
    # Create the checkpoints directory.
    checkpoints_dir = out_dir / "checkpoints"
    # Create the plots directory.
    plots_dir = out_dir / "plots"
    # Create the logs directory.
    logs_dir = out_dir / "logs"
    # Materialize all stage subdirectories.
    for directory in (checkpoints_dir, plots_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Create the stage logger.
    logger = get_stage_logger(STAGE_LOG_NAME, log_file=out_dir / "stage.log")
    # Emit the standard stage start banner.
    log_operation_start("Stage 4 collaborative-filter training", STAGE_LOG_NAME, logger)
    # Start the wall-clock timer for stage metadata.
    t0 = time.time()

    # --- reproducibility and CUDA housekeeping ---
    # Clear stale CUDA allocations before building the model.
    clear_cuda_memory()
    # Read the stage random seed from args.
    random_seed = int(args.random_seed)
    # Seed Python, NumPy, and Torch deterministically.
    set_random_seeds(random_seed)

    # --- load data from prior stages ---
    # Reuse the shared artifact loader to stay aligned with the rest of Stage 4.
    log_operation_start("Load training data from prior stages", STAGE_LOG_NAME, logger)
    _embeddings_mmap, target_posts_df, _history_df, embed_dim = load_training_data(
        run_dir, context, logger=logger,
    )

    # --- hyperparams (extract all args once, use locals everywhere below) ---
    # Keep the placeholder user-encoder argument for config output clarity.
    user_encoder = args.user_encoder
    # Read the latent factor size.
    cf_latent_dim = int(args.cf_latent_dim)
    # Read the batch size; here it means users per optimization batch.
    batch_size = int(args.batch_size)
    # Read the epoch count.
    epochs = int(args.epochs)
    # Read the learning rate.
    learning_rate = float(args.learning_rate)
    # Read the collaborative-filter weight decay.
    weight_decay = float(args.weight_decay_collaborative_filter)
    # Read the early-stopping patience.
    patience = int(args.patience)
    # Read the progress-bar toggle.
    disable_progress = bool(args.disable_progress)
    # Read the plot toggle.
    generate_plots = not bool(args.no_plots)
    # Read the checkpoint-save toggle.
    save_model = not bool(args.no_save_model)
    # Read the LR scheduler factor.
    lr_scheduler_factor = float(args.lr_scheduler_factor)
    # Read the LR scheduler patience.
    lr_scheduler_patience = int(args.lr_scheduler_patience)
    # Read the holdout flavor emphasized in metadata/plots.
    eval_holdout_type = str(args.eval_holdout_type)
    # Read the shared DataLoader worker count used for candidate scoring.
    num_workers = int(args.num_dataloader_workers)
    # Read the shared pin_memory toggle.
    pin_memory = bool(args.dataloader_pin_memory)
    # Read the shared persistent_workers toggle.
    persistent_workers = bool(args.dataloader_persistent_workers)
    # Read the shared prefetch factor.
    prefetch_factor = int(args.dataloader_prefetch_factor)

    # Log the placeholder nature of user_encoder for this model type.
    logger.info(
        f"Collaborative-filter mode selected (user_encoder={user_encoder!r} is accepted as a placeholder and not used architecturally)"
    )

    # --- build vocabularies from train positives and candidate items ---
    # Build the train-user vocabulary plus unseen-user bucket.
    user_to_index, unknown_user_index = _build_user_index(target_posts_df, logger)
    # Build the compact item vocabulary from all candidate items.
    item_to_index, unknown_item_index = _build_item_index(target_posts_df, logger)
    # Count the trainable items excluding the unknown-item bucket.
    num_train_items = len(item_to_index)
    # Count the total items including the unknown-item bucket.
    num_items = num_train_items + 1
    # Count the total users including the unseen-user bucket.
    num_users = len(user_to_index) + 1

    # --- build positive-only train lookup ---
    # Build the train user -> positive items lookup from observed likes only.
    user_positive_items, train_user_indices, num_positive_pairs = _build_train_positive_lookup(
        target_posts_df=target_posts_df,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        logger=logger,
    )
    # Choose the implicit positive weight by balancing positives and unlabeled zeros.
    positive_weight = _auto_balance_positive_weight(
        num_users=len(train_user_indices),
        num_items=num_train_items,
        num_positive_pairs=num_positive_pairs,
    )
    # Log the auto-balanced positive weight for reproducibility.
    logger.info(f"Implicit positive weight (auto-balanced): {positive_weight:.2f}")

    # --- build candidate datasets for validation and later exports ---
    # Start the candidate-dataset construction section in the logs.
    log_operation_start("Create candidate datasets for scoring/evaluation", STAGE_LOG_NAME, logger)
    # Build the candidate train dataset.
    train_candidate_dataset, train_stats = _build_candidate_dataset(
        target_posts_df=target_posts_df,
        split="train",
        user_to_index=user_to_index,
        unknown_user_index=unknown_user_index,
        item_to_index=item_to_index,
        unknown_item_index=unknown_item_index,
        logger=logger,
    )
    # Build the candidate validation dataset.
    val_candidate_dataset, val_stats = _build_candidate_dataset(
        target_posts_df=target_posts_df,
        split="val",
        user_to_index=user_to_index,
        unknown_user_index=unknown_user_index,
        item_to_index=item_to_index,
        unknown_item_index=unknown_item_index,
        logger=logger,
    )
    # Build the train candidate DataLoader used for final exports.
    train_candidate_loader = _create_loader(
        dataset=train_candidate_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    # Build the validation candidate DataLoader used for early stopping.
    val_candidate_loader = _create_loader(
        dataset=val_candidate_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # --- build model ---
    # Instantiate the minimal implicit collaborative-filter model.
    model = CollaborativeFilteringModel(
        num_users=num_users,
        num_items=num_items,
        latent_dim=cf_latent_dim,
        unknown_user_index=unknown_user_index,
    )

    # --- train ---
    # Start the fitting section in the logs.
    log_operation_start(
        f"Training implicit collaborative-filter model (epochs={epochs}, user_batch_size={batch_size})",
        STAGE_LOG_NAME,
        logger,
    )
    # Run the positive-only implicit training loop.
    training_results = train_collaborative_filter_model(
        model=model,
        train_user_indices=train_user_indices,
        user_positive_items=user_positive_items,
        num_train_items=num_train_items,
        val_loader=val_candidate_loader,
        device=device,
        epochs=epochs,
        user_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        disable_progress=disable_progress,
        positive_weight=positive_weight,
    )
    # Pull the fitted model out of the result bundle.
    trained_model: CollaborativeFilteringModel = training_results["model"]
    # Read the best validation AUC for config output.
    best_val_auc = float(training_results["best_val_auc"])

    # --- evaluate on candidate train and val pairs ---
    # Score the train candidate pairs for export and metrics.
    train_eval = _evaluate_model(trained_model, train_candidate_loader, device)
    # Score the validation candidate pairs for export and metrics.
    val_eval = _evaluate_model(trained_model, val_candidate_loader, device)
    # Log the train metric summary.
    logger.info(f"Train candidate metrics: {train_eval['metrics']}")
    # Log the validation metric summary.
    logger.info(f"Validation candidate metrics: {val_eval['metrics']}")

    # --- plots ---
    # Optionally create ROC/PR diagnostics for candidate-pair scoring.
    if generate_plots:
        # Plot train candidate diagnostics.
        try:
            plot_model_performance(
                train_eval["predictions"]["y_true"],
                train_eval["predictions"]["y_pred"],
                plots_dir / f"train_performance_{timestamp}.png",
                title_suffix="(Train)",
            )
        except Exception as exc:
            logger.warning(f"Train performance plotting failed: {exc}")
        # Plot validation candidate diagnostics.
        try:
            plot_model_performance(
                val_eval["predictions"]["y_true"],
                val_eval["predictions"]["y_pred"],
                plots_dir / f"val_performance_{timestamp}.png",
                title_suffix="(Validation)",
            )
        except Exception as exc:
            logger.warning(f"Validation performance plotting failed: {exc}")

    # --- save model ---
    # Initialize the optional model path for the artifact dict.
    model_path: Optional[Path] = None
    # Persist the fitted model when checkpoint saving is enabled.
    if save_model:
        # Announce checkpoint writing in the stage log.
        log_operation_start("Save model checkpoint", STAGE_LOG_NAME, logger)
        # Build the final checkpoint path.
        model_path = checkpoints_dir / f"collaborative_filter_{timestamp}.pth"
        # Save weights plus the compact vocab metadata.
        torch.save(
            {
                "model_state_dict": trained_model.state_dict(),
                "model_type": "collaborative-filter",
                "user_encoder": user_encoder,
                "num_users": num_users,
                "num_items": num_items,
                "cf_latent_dim": cf_latent_dim,
                "unknown_user_index": unknown_user_index,
                "unknown_item_index": unknown_item_index,
                "positive_weight": positive_weight,
                "training_results": {
                    "history": training_results["history"],
                    "best_val_loss": training_results["best_val_loss"],
                    "best_val_auc": training_results["best_val_auc"],
                },
                "training_parameters": {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "patience": patience,
                },
            },
            model_path,
        )
        # Log the checkpoint path.
        logger.info(f"Model saved to: {model_path}")
        # Register the checkpoint with the experiment tracker when available.
        context.tracker.log_artifact(name="collaborative_filter_model", path=model_path)

    # --- save predictions ---
    # Create the standard predictions directory.
    predictions_dir = out_dir / "predictions"
    # Materialize the predictions directory.
    predictions_dir.mkdir(parents=True, exist_ok=True)
    # Write train candidate predictions in the shared schema.
    pl.DataFrame({
        "did": train_eval["predictions"]["user_id"],
        "post_id": train_eval["predictions"]["post_id"],
        "y_true": train_eval["predictions"]["y_true"],
        "y_pred_proba": train_eval["predictions"]["y_pred"],
    }).write_parquet(predictions_dir / "train.parquet")
    # Write validation candidate predictions in the shared schema.
    pl.DataFrame({
        "did": val_eval["predictions"]["user_id"],
        "post_id": val_eval["predictions"]["post_id"],
        "y_true": val_eval["predictions"]["y_true"],
        "y_pred_proba": val_eval["predictions"]["y_pred"],
    }).write_parquet(predictions_dir / "val.parquet")

    # --- holdout evaluation ---
    # Initialize the preferred holdout metric bundle.
    holdout_metrics: Dict[str, Any] = {}
    # Evaluate both holdout flavors so Stage 5 can load either one.
    for holdout_type in ["unseen_users", "seen_users"]:
        # Derive the Stage 2 split name from the holdout flavor.
        split_name = f"holdout_{holdout_type}"
        try:
            # Build the holdout candidate dataset.
            holdout_dataset, holdout_stats = _build_candidate_dataset(
                target_posts_df=target_posts_df,
                split=split_name,
                user_to_index=user_to_index,
                unknown_user_index=unknown_user_index,
                item_to_index=item_to_index,
                unknown_item_index=unknown_item_index,
                logger=logger,
            )
            # Skip empty holdout splits cleanly.
            if len(holdout_dataset) == 0:
                logger.info(f"No rows for split '{split_name}', skipping.")
                continue
            # Announce the holdout evaluation run in the logs.
            log_operation_start(f"Holdout evaluation ({holdout_type})", STAGE_LOG_NAME, logger)
            # Build the holdout DataLoader.
            holdout_loader = _create_loader(
                dataset=holdout_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
            # Score the holdout candidate pairs.
            holdout_eval = _evaluate_model(trained_model, holdout_loader, device)
            # Read the holdout metric bundle.
            split_metrics = holdout_eval["metrics"]
            # Log metrics together with fallback usage stats.
            logger.info(f"Holdout metrics ({holdout_type}): {split_metrics} | stats={holdout_stats}")
            # Keep the preferred holdout metrics for metadata output.
            if holdout_type == eval_holdout_type:
                holdout_metrics = split_metrics
            # Write the holdout predictions in the shared schema.
            pl.DataFrame({
                "did": holdout_eval["predictions"]["user_id"],
                "post_id": holdout_eval["predictions"]["post_id"],
                "y_true": holdout_eval["predictions"]["y_true"],
                "y_pred_proba": holdout_eval["predictions"]["y_pred"],
            }).write_parquet(predictions_dir / f"{split_name}.parquet")
            # Optionally create the main holdout ROC/PR plot.
            if generate_plots and holdout_type == eval_holdout_type:
                try:
                    plot_model_performance(
                        holdout_eval["predictions"]["y_true"],
                        holdout_eval["predictions"]["y_pred"],
                        plots_dir / f"holdout_performance_{timestamp}.png",
                        title_suffix="(Holdout)",
                    )
                except Exception as exc:
                    logger.warning(f"Holdout performance plotting failed: {exc}")
        except Exception as exc:
            # Keep holdout failures non-fatal to match the other Stage 4 trainers.
            logger.warning(f"Holdout evaluation ({holdout_type}) failed (non-fatal): {exc}")

    # --- training config ---
    # Assemble the compact JSON config summary.
    training_config = {
        "model_type": "collaborative-filter",
        "training_mode": "positive_only_implicit_logistic_mf",
        "user_encoder": user_encoder,
        "cf_latent_dim": cf_latent_dim,
        "num_users": num_users,
        "num_items": num_items,
        "unknown_user_index": unknown_user_index,
        "unknown_item_index": unknown_item_index,
        "positive_weight": positive_weight,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": patience,
        "random_seed": random_seed,
        "embed_dim": embed_dim,
        "num_train_positive_pairs": num_positive_pairs,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "train_metrics": train_eval["metrics"],
        "val_metrics": val_eval["metrics"],
        "holdout_metrics": holdout_metrics,
        "best_val_auc": best_val_auc,
    }
    # Write the config summary.
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    # --- stage info ---
    # Compute the total stage runtime.
    runtime = time.time() - t0
    # Build the human-readable stage metadata lines.
    info_lines = [
        "stage: train_collaborative_filter",
        f"timestamp: {timestamp}",
        f"runtime_seconds: {runtime:.2f}",
        f"settings: batch_size={batch_size}, lr={learning_rate}, epochs={epochs}, cf_latent_dim={cf_latent_dim}",
        "training_mode: positive_only_implicit_logistic_mf",
        f"train_positive_pairs: {num_positive_pairs}",
        f"best_val_auc: {best_val_auc:.4f}",
    ]
    # Add holdout AUC when present.
    if holdout_metrics.get("auc_roc"):
        info_lines.append(f"holdout_auc: {holdout_metrics['auc_roc']:.4f}")
    # Persist the stage metadata file.
    (out_dir / "stage_info.txt").write_text("\n".join(info_lines) + "\n")

    # Log stage completion.
    logger.info(f"Collaborative-filter training completed in {runtime:.2f}s")

    # Return the standard stage result payload.
    return {
        "output_dir": out_dir,
        "artifacts": {
            "model_path": str(model_path) if model_path else None,
            "training_config": str(out_dir / "training_config.json"),
        },
    }
