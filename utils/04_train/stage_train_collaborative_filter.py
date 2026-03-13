#!/usr/bin/env python3

"""
Stage 4 (Collaborative Filter): train a minimal matrix-factorization recommender.

This stage intentionally keeps the model small and easy to reason about:
    - Users are represented by learned latent vectors.
    - Posts are represented by learned latent vectors keyed by ``emb_idx``.
    - The score is a dot product plus user/item/global bias terms.
    - A sigmoid converts the score into ``y_pred_proba`` so the existing
      evaluation pipeline can consume the outputs without any special cases.

The stage reuses the same Stage 1-3 artifacts as the MLP and two-tower paths,
but it does not consume post embedding values directly.  Instead, it uses the
embedding memmap only to determine the size of the item vocabulary so that the
item IDs stay aligned with the rest of the pipeline.

The holdout user split includes users that were never seen during training, so
this implementation reserves a single ``unknown_user`` embedding and maps every
unseen user to that fallback vector at validation / holdout time.
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

from utils.dataloaders import filter_split_and_join_history, load_training_data
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
    """Minimal logistic matrix-factorization model.

    The model stays intentionally small:
        - ``user_factors`` learns one vector per seen user plus one unknown user.
        - ``item_factors`` learns one vector per post ``emb_idx`` plus one unknown item.
        - ``user_bias`` and ``item_bias`` capture marginal effects cheaply.
        - ``global_bias`` lets the model learn the base interaction rate.
    """

    def __init__(self, num_users: int, num_items: int, latent_dim: int):
        super().__init__()
        # Store sizes so the saved checkpoint can reconstruct the module later.
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        # Learn one latent vector per user.
        self.user_factors = nn.Embedding(num_users, latent_dim)
        # Learn one latent vector per item.
        self.item_factors = nn.Embedding(num_items, latent_dim)
        # Learn one scalar bias per user.
        self.user_bias = nn.Embedding(num_users, 1)
        # Learn one scalar bias per item.
        self.item_bias = nn.Embedding(num_items, 1)
        # Learn a single global bias shared by every example.
        self.global_bias = nn.Parameter(torch.zeros(1))
        # Initialize factor tables with a small normal distribution.
        nn.init.normal_(self.user_factors.weight, mean=0.0, std=0.02)
        # Initialize item factors the same way as user factors.
        nn.init.normal_(self.item_factors.weight, mean=0.0, std=0.02)
        # Start user biases at zero so the model begins near the global rate.
        nn.init.zeros_(self.user_bias.weight)
        # Start item biases at zero for the same reason.
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_index: torch.Tensor, item_index: torch.Tensor) -> torch.Tensor:
        # Fetch the latent vector for each user in the batch.
        user_vec = self.user_factors(user_index)
        # Fetch the latent vector for each item in the batch.
        item_vec = self.item_factors(item_index)
        # Compute the dot-product interaction score for each user-item pair.
        interaction = (user_vec * item_vec).sum(dim=-1)
        # Add user-specific bias terms.
        user_bias = self.user_bias(user_index).squeeze(-1)
        # Add item-specific bias terms.
        item_bias = self.item_bias(item_index).squeeze(-1)
        # Combine interaction and bias terms into one logit.
        logits = interaction + user_bias + item_bias + self.global_bias
        # Convert logits into probabilities for the shared evaluation pipeline.
        return torch.sigmoid(logits)

    def compute_loss_and_preds(self, batch: Dict[str, Any], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # Move the user indices to the active device.
        user_index = batch["user_index"].to(device)
        # Move the item indices to the active device.
        item_index = batch["item_index"].to(device)
        # Move the binary labels to the active device.
        labels = batch["label"].to(device)
        # Run the forward pass and obtain calibrated probabilities.
        preds = self(user_index, item_index)
        # Optimize binary cross-entropy on the predicted probabilities.
        loss = F.binary_cross_entropy(preds, labels)
        # Return both the scalar loss and the probabilities for metrics.
        return loss, preds


class InteractionDataset(Dataset):
    """Flat binary interaction dataset shared by train/val/holdout evaluation."""

    def __init__(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        labels: np.ndarray,
        user_ids: List[str],
        post_ids: List[str],
    ):
        # Cache user indices as a tensor so __getitem__ stays tiny.
        self.user_indices = torch.as_tensor(user_indices, dtype=torch.long)
        # Cache item indices the same way.
        self.item_indices = torch.as_tensor(item_indices, dtype=torch.long)
        # Cache labels as float tensors for BCE loss.
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        # Keep original string user IDs for prediction parquet outputs.
        self.user_ids = user_ids
        # Keep original string post IDs for prediction parquet outputs.
        self.post_ids = post_ids

    def __len__(self) -> int:
        # Return the total number of binary interaction rows.
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Return the encoded user index used by the model.
        user_index = self.user_indices[idx]
        # Return the encoded item index used by the model.
        item_index = self.item_indices[idx]
        # Return the binary label for this interaction.
        label = self.labels[idx]
        # Return the original user ID for downstream prediction exports.
        user_id = self.user_ids[idx]
        # Return the original post ID for downstream prediction exports.
        post_id = self.post_ids[idx]
        # Bundle everything into the standard batch dictionary shape.
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
    """Build a train-only user vocabulary plus one unknown-user bucket."""
    # Keep the vocabulary tied strictly to the train split to avoid leakage.
    train_users = (
        target_posts_df
        .filter((pl.col("split") == "train") & pl.col("neg_emb_idx").is_not_null())
        .select("target_did")
        .unique()
        .sort("target_did")
        .get_column("target_did")
        .to_list()
    )
    # Assign a stable integer index to every seen training user.
    user_to_index = {did: idx for idx, did in enumerate(train_users)}
    # Reserve one extra slot for any user that never appeared in training.
    unknown_user_index = len(train_users)
    # Log the vocabulary size so the stage metadata is easy to audit.
    logger.info(
        f"User vocabulary built from train split: {len(train_users):,} seen users + 1 unknown-user bucket"
    )
    # Return both the mapping and the fallback index.
    return user_to_index, unknown_user_index


def _sanitize_item_indices(
    raw_item_indices: np.ndarray,
    unknown_item_index: int,
) -> np.ndarray:
    """Map out-of-range item IDs to the reserved unknown-item bucket."""
    # Copy into int64 because PyTorch embedding lookups expect integer indices.
    item_indices = raw_item_indices.astype(np.int64, copy=True)
    # Identify any invalid item IDs before indexing the embedding table.
    invalid_mask = (item_indices < 0) | (item_indices >= unknown_item_index)
    # Replace invalid IDs with the fallback item index.
    item_indices[invalid_mask] = unknown_item_index
    # Return the sanitized array.
    return item_indices


def _interleave_post_ids(
    like_uris: List[str],
    neg_uris: List[str],
) -> List[str]:
    """Build the alternating [positive, negative, positive, negative, ...] post ID list."""
    # Pre-allocate the exact output length to avoid repeated list growth.
    post_ids: List[str] = [""] * (2 * len(like_uris))
    # Fill the even positions with liked post IDs.
    post_ids[0::2] = like_uris
    # Fill the odd positions with negative post IDs.
    post_ids[1::2] = neg_uris
    # Return the alternating sequence.
    return post_ids


def _build_interaction_dataset(
    target_posts_df: pl.DataFrame,
    history_df: pl.DataFrame,
    split: str,
    user_to_index: Dict[str, int],
    unknown_user_index: int,
    unknown_item_index: int,
    logger,
) -> Tuple[InteractionDataset, Dict[str, int]]:
    """Convert one split into a flat binary interaction dataset."""
    # Reuse the canonical target/history join used elsewhere in the pipeline.
    joined = filter_split_and_join_history(target_posts_df, history_df, split)
    # Log the number of target rows that survived split + negative filtering.
    logger.info(f"  Split '{split}': {len(joined):,} target rows (after dropping null neg_emb_idx)")
    # Extract the per-row user IDs.
    target_dids = joined["target_did"].to_list()
    # Extract the positive item IDs.
    like_item_indices = joined["like_emb_idx"].to_numpy()
    # Extract the negative item IDs.
    neg_item_indices = joined["neg_emb_idx"].to_numpy()
    # Extract the positive post URIs for output exports.
    like_uris = joined["like_uri"].to_list()
    # Extract the negative post URIs for output exports.
    neg_uris = joined["neg_uri"].to_list()
    # Count how many target rows this split contains.
    n_rows = len(target_dids)
    # Map every raw user ID into the train vocabulary, falling back when unseen.
    mapped_users = np.array([user_to_index.get(did, unknown_user_index) for did in target_dids], dtype=np.int64)
    # Sanitize positive item IDs before building the model inputs.
    safe_like_items = _sanitize_item_indices(like_item_indices, unknown_item_index)
    # Sanitize negative item IDs before building the model inputs.
    safe_neg_items = _sanitize_item_indices(neg_item_indices, unknown_item_index)
    # Interleave user indices so each target row becomes [positive, negative].
    user_indices = np.empty(2 * n_rows, dtype=np.int64)
    # Place the user index for every positive sample.
    user_indices[0::2] = mapped_users
    # Reuse the same user index for every negative sample.
    user_indices[1::2] = mapped_users
    # Interleave item indices to match the [positive, negative] row order.
    item_indices = np.empty(2 * n_rows, dtype=np.int64)
    # Fill even positions with the liked item IDs.
    item_indices[0::2] = safe_like_items
    # Fill odd positions with the negative item IDs.
    item_indices[1::2] = safe_neg_items
    # Build balanced binary labels in the same alternating order.
    labels = np.empty(2 * n_rows, dtype=np.float32)
    # Positive examples go in even positions.
    labels[0::2] = 1.0
    # Negative examples go in odd positions.
    labels[1::2] = 0.0
    # Duplicate the user ID strings in the same alternating order for exports.
    user_ids = [did for did in target_dids for _ in (0, 1)]
    # Interleave the positive and negative post IDs for exports.
    post_ids = _interleave_post_ids(like_uris, neg_uris)
    # Count how many users fell back to the unknown-user bucket.
    n_unknown_users = int((mapped_users == unknown_user_index).sum())
    # Count how many items fell back to the unknown-item bucket.
    n_unknown_items = int((item_indices == unknown_item_index).sum())
    # Build the PyTorch dataset object.
    dataset = InteractionDataset(
        user_indices=user_indices,
        item_indices=item_indices,
        labels=labels,
        user_ids=user_ids,
        post_ids=post_ids,
    )
    # Return the dataset together with small audit stats for logging/config output.
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
    """Create one DataLoader using the repo's standard worker settings."""
    # Start from the arguments that apply to every loader.
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    # Only pass worker-specific options when worker processes are enabled.
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    # Return the configured DataLoader instance.
    return DataLoader(dataset, **loader_kwargs)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute the small metric set shared across train/val/holdout outputs."""
    # Always record how many binary rows were evaluated.
    metrics: Dict[str, Any] = {"total_samples": int(len(y_true))}
    # Always record how many positive rows were present.
    metrics["positive_samples"] = int(y_true.sum())
    # Guard AUC because it requires both classes to be present.
    if len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred))
    # Record the default 0.5-threshold accuracy for quick debugging.
    metrics["accuracy@0.5"] = float(accuracy_score(y_true, (y_pred > 0.5).astype(int)))
    # Record the mean probability so calibration drift is visible in logs.
    metrics["mean_pred_proba"] = float(np.mean(y_pred))
    # Return the metric dictionary.
    return metrics


def _evaluate_model(
    model: CollaborativeFilteringModel,
    loader: DataLoader,
    device: str,
) -> Dict[str, Any]:
    """Run one full evaluation pass and collect metrics plus parquet-ready predictions."""
    # Switch the module into inference mode.
    model.eval()
    # Accumulate scalar predictions here.
    y_pred: List[float] = []
    # Accumulate ground-truth labels here.
    y_true: List[float] = []
    # Accumulate user IDs here for parquet export.
    user_ids: List[str] = []
    # Accumulate post IDs here for parquet export.
    post_ids: List[str] = []
    # Disable autograd because this pass is evaluation-only.
    with torch.inference_mode():
        # Step through the loader batch by batch.
        for batch in loader:
            # Reuse the model helper so the scoring path stays identical to training.
            _, preds = model.compute_loss_and_preds(batch, device)
            # Extend the probability list with CPU values.
            y_pred.extend(preds.cpu().numpy().tolist())
            # Extend the label list with CPU values.
            y_true.extend(batch["label"].cpu().numpy().tolist())
            # Extend the user ID list with raw strings from the batch.
            user_ids.extend(batch["user_id"])
            # Extend the post ID list with raw strings from the batch.
            post_ids.extend(batch["post_id"])
    # Convert labels to one compact numpy array.
    y_true_arr = np.asarray(y_true, dtype=np.float32)
    # Convert predictions to one compact numpy array.
    y_pred_arr = np.asarray(y_pred, dtype=np.float32)
    # Compute summary metrics from the full pass.
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


def train_collaborative_filter_model(
    model: CollaborativeFilteringModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    lr_scheduler_factor: float,
    lr_scheduler_patience: int,
    disable_progress: bool,
    gradient_clip_max_norm: float,
) -> Dict[str, Any]:
    """Train the collaborative-filter model with early stopping on validation AUC."""
    # Import optimizers lazily to mirror the other Stage 4 modules.
    import torch.optim as optim
    # Import the scheduler lazily for the same reason.
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    # Import tqdm lazily to keep module import time small.
    from tqdm import tqdm as _tqdm

    # Move model parameters onto the requested device.
    model = model.to(device)
    # Use AdamW for a small, stable optimization baseline.
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Reduce the learning rate when validation AUC plateaus.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
    )
    # Track train/val loss and AUC across epochs.
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_auc": [],
        "val_auc": [],
    }
    # Start best-AUC tracking at the lowest possible value.
    best_val_auc = float("-inf")
    # Track the best validation loss as a tie-breaker.
    best_val_loss = float("inf")
    # Start the no-improvement counter at zero.
    patience_counter = 0
    # Keep the best model weights in memory for final evaluation/export.
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    # Iterate over epochs with an optional progress bar.
    for _epoch in _tqdm(range(epochs), desc="Training epochs", disable=disable_progress):
        # Switch into training mode so gradients and embeddings update.
        model.train()
        # Reset the running train loss accumulator.
        epoch_train_loss = 0.0
        # Accumulate train probabilities for AUC.
        train_preds: List[float] = []
        # Accumulate train labels for AUC.
        train_labels: List[float] = []
        # Loop through the train loader.
        for batch in _tqdm(train_loader, desc="Training", leave=False, disable=disable_progress):
            # Clear stale gradients before the next backward pass.
            optimizer.zero_grad()
            # Compute the current batch loss and probabilities.
            loss, preds = model.compute_loss_and_preds(batch, device)
            # Backpropagate through the latent factors and bias terms.
            loss.backward()
            # Clip gradients to avoid occasional unstable updates.
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
            # Apply the optimizer update.
            optimizer.step()
            # Add this batch loss into the epoch average accumulator.
            epoch_train_loss += float(loss.item())
            # Cache train probabilities for train AUC.
            train_preds.extend(preds.detach().cpu().numpy().tolist())
            # Cache labels for train AUC.
            train_labels.extend(batch["label"].detach().cpu().numpy().tolist())

        # Normalize train loss by the number of batches.
        train_loss = epoch_train_loss / max(len(train_loader), 1)
        # Convert cached train labels into a numpy array.
        train_labels_arr = np.asarray(train_labels, dtype=np.float32)
        # Convert cached train predictions into a numpy array.
        train_preds_arr = np.asarray(train_preds, dtype=np.float32)
        # Compute train AUC when both classes are present.
        train_auc = float(roc_auc_score(train_labels_arr, train_preds_arr)) if len(np.unique(train_labels_arr)) > 1 else 0.0

        # Switch into evaluation mode for validation.
        model.eval()
        # Reset the running validation loss accumulator.
        epoch_val_loss = 0.0
        # Accumulate validation probabilities for AUC.
        val_preds: List[float] = []
        # Accumulate validation labels for AUC.
        val_labels: List[float] = []
        # Disable autograd during validation.
        with torch.inference_mode():
            # Loop through the validation loader.
            for batch in _tqdm(val_loader, desc="Validation", leave=False, disable=disable_progress):
                # Compute validation loss and probabilities with the same scoring path.
                loss, preds = model.compute_loss_and_preds(batch, device)
                # Add batch loss into the validation loss accumulator.
                epoch_val_loss += float(loss.item())
                # Cache validation probabilities.
                val_preds.extend(preds.detach().cpu().numpy().tolist())
                # Cache validation labels.
                val_labels.extend(batch["label"].detach().cpu().numpy().tolist())

        # Normalize validation loss by the number of batches.
        val_loss = epoch_val_loss / max(len(val_loader), 1)
        # Convert cached validation labels into a numpy array.
        val_labels_arr = np.asarray(val_labels, dtype=np.float32)
        # Convert cached validation predictions into a numpy array.
        val_preds_arr = np.asarray(val_preds, dtype=np.float32)
        # Compute validation AUC when both classes are present.
        val_auc = float(roc_auc_score(val_labels_arr, val_preds_arr)) if len(np.unique(val_labels_arr)) > 1 else 0.0

        # Append the current epoch statistics into the history object.
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        # Let the scheduler react to validation AUC.
        scheduler.step(val_auc)

        # Prefer higher AUC, then lower loss when AUC ties.
        is_better = (val_auc > best_val_auc) or (val_auc == best_val_auc and val_loss < best_val_loss)
        # Snapshot the model whenever validation improves.
        if is_better:
            # Update the best validation AUC.
            best_val_auc = val_auc
            # Update the best validation loss.
            best_val_loss = val_loss
            # Clone the state dict so later epochs cannot mutate it in place.
            best_state_dict = copy.deepcopy(model.state_dict())
            # Reset the patience counter after an improvement.
            patience_counter = 0
        else:
            # Count one more epoch without improvement.
            patience_counter += 1
            # Stop early once the patience budget is exhausted.
            if patience_counter >= patience:
                break

    # Restore the best model weights before returning.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Return the trained model and the tracked history/metrics.
    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_auc": best_val_auc,
    }


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    # Resolve the root run directory for prior artifact lookup.
    run_dir = Path(context.run_dir).resolve()
    # Resolve the training device once at the top of the stage.
    device = get_device(args.device)
    # Reuse the pipeline timestamp for all output file names.
    timestamp = context.run_timestamp

    # --- output directories ---
    # Preserve the repo's run-tag convention for sweep outputs.
    run_tag = args.run_tag or ""
    # Create the timestamped Stage 4 directory.
    out_dir = context.new_stage_dir("04_train", tag=run_tag)
    # Create the checkpoints directory.
    checkpoints_dir = out_dir / "checkpoints"
    # Create the plots directory.
    plots_dir = out_dir / "plots"
    # Create the logs directory.
    logs_dir = out_dir / "logs"
    # Materialize all Stage 4 subdirectories eagerly.
    for directory in (checkpoints_dir, plots_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Create the stage logger inside the output directory.
    logger = get_stage_logger(STAGE_LOG_NAME, log_file=out_dir / "stage.log")
    # Emit the standard stage start banner.
    log_operation_start("Stage 4 collaborative-filter training", STAGE_LOG_NAME, logger)
    # Start a wall-clock timer for stage metadata.
    t0 = time.time()

    # --- reproducibility and CUDA housekeeping ---
    # Clear any stale CUDA allocations before building the model.
    clear_cuda_memory()
    # Read the stage random seed from the CLI namespace.
    random_seed = int(args.random_seed)
    # Seed Python / NumPy / Torch deterministically.
    set_random_seeds(random_seed)

    # --- load data from prior stages ---
    # Reuse the shared loader so artifact lookup stays consistent with other models.
    log_operation_start("Load training data from prior stages", STAGE_LOG_NAME, logger)
    embeddings_mmap, target_posts_df, history_df, embed_dim = load_training_data(
        run_dir, context, logger=logger,
    )

    # --- hyperparams (extract all args once, use locals everywhere below) ---
    # Keep the placeholder user encoder local for config output clarity.
    user_encoder = args.user_encoder
    # Read the latent factor size for the matrix-factorization model.
    cf_latent_dim = int(args.cf_latent_dim)
    # Read the shared batch size.
    batch_size = int(args.batch_size)
    # Read the shared epoch count.
    epochs = int(args.epochs)
    # Read the shared learning rate.
    learning_rate = float(args.learning_rate)
    # Read the collaborative-filter-specific weight decay.
    weight_decay = float(args.weight_decay_collaborative_filter)
    # Read the shared early-stopping patience.
    patience = int(args.patience)
    # Read the progress-bar toggle.
    disable_progress = bool(args.disable_progress)
    # Read the plot toggle.
    generate_plots = not bool(args.no_plots)
    # Read the model-save toggle.
    save_model = not bool(args.no_save_model)
    # Read the learning-rate scheduler factor.
    lr_scheduler_factor = float(args.lr_scheduler_factor)
    # Read the learning-rate scheduler patience.
    lr_scheduler_patience = int(args.lr_scheduler_patience)
    # Read the gradient clipping threshold.
    gradient_clip_max_norm = float(args.gradient_clip_max_norm)
    # Read the holdout type that Stage 5 will emphasize.
    eval_holdout_type = str(args.eval_holdout_type)
    # Read the shared DataLoader worker count.
    num_workers = int(args.num_dataloader_workers)
    # Read the shared pin_memory toggle.
    pin_memory = bool(args.dataloader_pin_memory)
    # Read the shared persistent_workers toggle.
    persistent_workers = bool(args.dataloader_persistent_workers)
    # Read the shared prefetch factor.
    prefetch_factor = int(args.dataloader_prefetch_factor)

    # Log the fact that collaborative filtering ignores the user-encoder architecture flag.
    logger.info(f"Collaborative-filter mode selected (user_encoder={user_encoder!r} is accepted as a placeholder and not used)")

    # --- build vocabularies ---
    # Build the train-only user vocabulary and unknown-user fallback.
    user_to_index, unknown_user_index = _build_user_index(target_posts_df, logger)
    # Count all valid item rows from the Stage 1 memmap.
    num_known_items = int(embeddings_mmap.shape[0])
    # Reserve one extra item slot as a defensive unknown-item bucket.
    unknown_item_index = num_known_items
    # Derive the total item vocabulary size including the fallback bucket.
    num_items = num_known_items + 1
    # Derive the total user vocabulary size including the fallback bucket.
    num_users = len(user_to_index) + 1
    # Log the resulting vocabulary sizes.
    logger.info(
        f"Item vocabulary built from embeddings memmap: {num_known_items:,} known items + 1 unknown-item bucket"
    )

    # --- create datasets ---
    # Start the dataset construction section in the logs.
    log_operation_start("Create collaborative-filter datasets", STAGE_LOG_NAME, logger)
    # Build the train split dataset.
    train_dataset, train_stats = _build_interaction_dataset(
        target_posts_df=target_posts_df,
        history_df=history_df,
        split="train",
        user_to_index=user_to_index,
        unknown_user_index=unknown_user_index,
        unknown_item_index=unknown_item_index,
        logger=logger,
    )
    # Build the validation split dataset.
    val_dataset, val_stats = _build_interaction_dataset(
        target_posts_df=target_posts_df,
        history_df=history_df,
        split="val",
        user_to_index=user_to_index,
        unknown_user_index=unknown_user_index,
        unknown_item_index=unknown_item_index,
        logger=logger,
    )
    # Build the train DataLoader.
    train_loader = _create_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    # Build the validation DataLoader.
    val_loader = _create_loader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # --- build model ---
    # Instantiate the minimal collaborative-filter model.
    model = CollaborativeFilteringModel(
        num_users=num_users,
        num_items=num_items,
        latent_dim=cf_latent_dim,
    )

    # --- train ---
    # Start the training section in the logs.
    log_operation_start(
        f"Training collaborative-filter model (epochs={epochs}, batch_size={batch_size})",
        STAGE_LOG_NAME,
        logger,
    )
    # Run the training loop and capture the best model state/history.
    training_results = train_collaborative_filter_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        disable_progress=disable_progress,
        gradient_clip_max_norm=gradient_clip_max_norm,
    )
    # Pull the trained model out of the result bundle.
    trained_model: CollaborativeFilteringModel = training_results["model"]
    # Read the best validation AUC for config and metadata output.
    best_val_auc = float(training_results["best_val_auc"])

    # --- evaluate on train and val ---
    # Evaluate the fitted model on the train split for plots and exports.
    train_eval = _evaluate_model(trained_model, train_loader, device)
    # Evaluate the fitted model on the validation split for plots and exports.
    val_eval = _evaluate_model(trained_model, val_loader, device)
    # Log the train metric summary.
    logger.info(f"Train metrics: {train_eval['metrics']}")
    # Log the validation metric summary.
    logger.info(f"Validation metrics: {val_eval['metrics']}")

    # --- plots ---
    # Optionally create train/val diagnostic plots using the shared helper.
    if generate_plots:
        # Plot train ROC/PR diagnostics.
        try:
            plot_model_performance(
                train_eval["predictions"]["y_true"],
                train_eval["predictions"]["y_pred"],
                plots_dir / f"train_performance_{timestamp}.png",
                title_suffix="(Train)",
            )
        except Exception as exc:
            logger.warning(f"Train performance plotting failed: {exc}")
        # Plot validation ROC/PR diagnostics.
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
    # Initialize the optional model path to None for the artifact dict.
    model_path: Optional[Path] = None
    # Persist the learned latent-factor model when checkpoint saving is enabled.
    if save_model:
        # Announce checkpoint writing in the stage log.
        log_operation_start("Save model checkpoint", STAGE_LOG_NAME, logger)
        # Build the final checkpoint path inside the stage output directory.
        model_path = checkpoints_dir / f"collaborative_filter_{timestamp}.pth"
        # Save the model weights plus enough config to reload later.
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
        # Log the final checkpoint path for debugging.
        logger.info(f"Model saved to: {model_path}")
        # Register the checkpoint with the experiment tracker when available.
        context.tracker.log_artifact(name="collaborative_filter_model", path=model_path)

    # --- save predictions ---
    # Create the shared predictions output directory.
    predictions_dir = out_dir / "predictions"
    # Materialize the predictions directory before writing parquet files.
    predictions_dir.mkdir(parents=True, exist_ok=True)
    # Write train predictions in the standard schema consumed by evaluation.
    pl.DataFrame({
        "did": train_eval["predictions"]["user_id"],
        "post_id": train_eval["predictions"]["post_id"],
        "y_true": train_eval["predictions"]["y_true"],
        "y_pred_proba": train_eval["predictions"]["y_pred"],
    }).write_parquet(predictions_dir / "train.parquet")
    # Write validation predictions in the same standard schema.
    pl.DataFrame({
        "did": val_eval["predictions"]["user_id"],
        "post_id": val_eval["predictions"]["post_id"],
        "y_true": val_eval["predictions"]["y_true"],
        "y_pred_proba": val_eval["predictions"]["y_pred"],
    }).write_parquet(predictions_dir / "val.parquet")

    # --- holdout evaluation ---
    # Initialize the holdout metric bundle with an empty dict.
    holdout_metrics: Dict[str, Any] = {}
    # Evaluate both holdout flavors so Stage 5 can load either one later.
    for holdout_type in ["unseen_users", "seen_users"]:
        # Derive the Stage 2 split name from the holdout flavor.
        split_name = f"holdout_{holdout_type}"
        try:
            # Build the holdout dataset using the same user/item mappings.
            holdout_dataset, holdout_stats = _build_interaction_dataset(
                target_posts_df=target_posts_df,
                history_df=history_df,
                split=split_name,
                user_to_index=user_to_index,
                unknown_user_index=unknown_user_index,
                unknown_item_index=unknown_item_index,
                logger=logger,
            )
            # Skip empty holdout splits cleanly.
            if len(holdout_dataset) == 0:
                logger.info(f"No rows for split '{split_name}', skipping.")
                continue
            # Announce the holdout evaluation run in the stage log.
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
            # Run the holdout inference pass.
            holdout_eval = _evaluate_model(trained_model, holdout_loader, device)
            # Read the metric summary for this holdout split.
            split_metrics = holdout_eval["metrics"]
            # Log the metrics together with unknown-user/item counts.
            logger.info(f"Holdout metrics ({holdout_type}): {split_metrics} | stats={holdout_stats}")
            # Keep the preferred holdout metrics in the top-level config bundle.
            if holdout_type == eval_holdout_type:
                holdout_metrics = split_metrics
            # Write the holdout prediction parquet in the shared schema.
            pl.DataFrame({
                "did": holdout_eval["predictions"]["user_id"],
                "post_id": holdout_eval["predictions"]["post_id"],
                "y_true": holdout_eval["predictions"]["y_true"],
                "y_pred_proba": holdout_eval["predictions"]["y_pred"],
            }).write_parquet(predictions_dir / f"{split_name}.parquet")
            # Optionally emit the main holdout diagnostic plot.
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
    # Assemble a compact JSON summary for later inspection.
    training_config = {
        "model_type": "collaborative-filter",
        "user_encoder": user_encoder,
        "cf_latent_dim": cf_latent_dim,
        "num_users": num_users,
        "num_items": num_items,
        "unknown_user_index": unknown_user_index,
        "unknown_item_index": unknown_item_index,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": patience,
        "random_seed": random_seed,
        "embed_dim": embed_dim,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "train_metrics": train_eval["metrics"],
        "val_metrics": val_eval["metrics"],
        "holdout_metrics": holdout_metrics,
        "best_val_auc": best_val_auc,
    }
    # Write the JSON summary into the stage output directory.
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    # --- stage info ---
    # Compute the total stage runtime in seconds.
    runtime = time.time() - t0
    # Build the human-readable stage metadata file.
    info_lines = [
        "stage: train_collaborative_filter",
        f"timestamp: {timestamp}",
        f"runtime_seconds: {runtime:.2f}",
        f"settings: batch_size={batch_size}, lr={learning_rate}, epochs={epochs}, cf_latent_dim={cf_latent_dim}",
        "inputs: embeddings memmap size, target_posts, user_history",
        f"train_samples: {len(train_dataset)}",
        f"val_samples: {len(val_dataset)}",
        f"best_val_auc: {best_val_auc:.4f}",
    ]
    # Add holdout AUC when it is available.
    if holdout_metrics.get("auc_roc"):
        info_lines.append(f"holdout_auc: {holdout_metrics['auc_roc']:.4f}")
    # Persist the stage metadata file.
    (out_dir / "stage_info.txt").write_text("\n".join(info_lines) + "\n")

    # Announce stage completion in the log.
    logger.info(f"Collaborative-filter training completed in {runtime:.2f}s")

    # Return the standard stage result payload.
    return {
        "output_dir": out_dir,
        "artifacts": {
            "model_path": str(model_path) if model_path else None,
            "training_config": str(out_dir / "training_config.json"),
        },
    }
