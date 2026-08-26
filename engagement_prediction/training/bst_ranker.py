"""Reusable listwise training primitives for the BST ranker."""

from __future__ import annotations

from contextlib import nullcontext
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from engagement_prediction.models.bst_ranker import BSTRanker
from engagement_prediction.training.ranking import (
    MatrixBatchScores,
    calc_baseline_ndcg_tensor_sums_for_batch,
    empty_ndcg_metric_tensor_sums,
    finalize_rank_metrics,
    finalize_zero_history_rank_metrics,
    log_zero_history_rank_metrics,
    ndcg_metric_tensor_sums_for_batch,
    topk_ranked_labels_for_scores,
)


def compute_bst_listwise_loss_and_scores(
    model: BSTRanker,
    batch: Dict[str, Any],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    label_matrix = batch["label_matrix"]
    # Native loader batches arrive on the host. Validate them before the
    # asynchronous CUDA copy so invalid-data detection does not introduce a
    # device-to-host synchronization into every training step.
    input_positive_counts = label_matrix.sum(dim=1, keepdim=True)
    if torch.any(input_positive_counts <= 0):
        raise RuntimeError("Each user row in label_matrix must contain at least one positive candidate")

    history_embeddings = batch["history_embeddings"].to(device, non_blocking=True)
    history_mask = batch["history_mask"].to(device, non_blocking=True)
    history_time_deltas_hours = batch["history_time_deltas_hours"].to(device, non_blocking=True)
    candidate_post_embeddings = batch["candidate_post_embeddings"].to(device, non_blocking=True)
    labels = label_matrix.to(device, dtype=torch.float32, non_blocking=True)
    if "history_author_indices" not in batch or "candidate_post_author_idx" not in batch:
        raise RuntimeError("BST listwise batches must include author index tensors")
    history_author_indices = batch["history_author_indices"].to(device, dtype=torch.long, non_blocking=True)
    candidate_post_author_idx = batch["candidate_post_author_idx"].to(device, dtype=torch.long, non_blocking=True)
    history_prior_cumulative_likes = None
    candidate_prior_cumulative_likes = None
    if model.use_popularity_feature:
        if "history_prior_cumulative_likes" not in batch or "candidate_prior_cumulative_likes" not in batch:
            raise RuntimeError("BST listwise batches must include popularity tensors when popularity features are enabled")
        history_prior_cumulative_likes = batch["history_prior_cumulative_likes"].to(device, dtype=torch.float32, non_blocking=True)
        candidate_prior_cumulative_likes = batch["candidate_prior_cumulative_likes"].to(device, dtype=torch.float32, non_blocking=True)

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
        raise RuntimeError("Expected BST scores and label_matrix to have matching [num_users, num_candidates] shapes")
    positive_counts = input_positive_counts.to(device, dtype=torch.float32, non_blocking=True)
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
) -> Tuple[float, Dict[str, Any], Dict[str, float]]:
    if train:
        if optimizer is None:
            raise ValueError("optimizer is required when train=True")
        model.train()
    else:
        model.eval()

    loss_sum = torch.zeros((), device=device)
    batches = 0
    baseline_metric_sums = empty_ndcg_metric_tensor_sums(
        metrics_top_ks,
        device=device,
    )
    baseline_metric_user_count = torch.zeros((), device=device, dtype=torch.int64)
    metric_sums = empty_ndcg_metric_tensor_sums(
        metrics_top_ks,
        device=device,
    )
    metric_user_count = torch.zeros((), device=device, dtype=torch.int64)
    zero_history_metric_sums = empty_ndcg_metric_tensor_sums(
        metrics_top_ks,
        device=device,
    )
    zero_history_metric_user_count = torch.zeros((), device=device, dtype=torch.int64)
    with nullcontext() if train else torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=split_name, leave=False, disable=disable_progress)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if train and optimizer is not None:
                optimizer.zero_grad()

            history_mask = batch.get("history_mask")
            if history_mask is None or history_mask.dim() != 2:
                raise RuntimeError("history_mask must have shape [num_users, history_len]")
            # Compute the metric mask on the host before model inputs move to
            # CUDA. Only this B-length vector is copied for metrics; the full
            # BxH history mask is not transferred a second time.
            zero_history_mask = (~history_mask.any(dim=1)).to(
                device,
                dtype=torch.bool,
                non_blocking=True,
            )
            loss, scores, labels = compute_bst_listwise_loss_and_scores(model, batch, device)
            if calc_baseline_metrics:
                baseline_batch_metric_sums, baseline_batch_metric_user_count = calc_baseline_ndcg_tensor_sums_for_batch(
                    labels,
                    metrics_top_ks,
                )
                baseline_metric_user_count.add_(baseline_batch_metric_user_count)
                for key, value in baseline_batch_metric_sums.items():
                    baseline_metric_sums[key].add_(value)

            top_ranked_labels = topk_ranked_labels_for_scores(
                scores,
                labels,
                metrics_top_ks,
            )
            total_relevant = labels.sum(dim=1)
            batch_metric_sums, batch_metric_user_count = ndcg_metric_tensor_sums_for_batch(
                top_ranked_labels,
                total_relevant,
                metrics_top_ks,
            )
            batch_zero_history_metric_sums, batch_zero_history_metric_user_count = ndcg_metric_tensor_sums_for_batch(
                top_ranked_labels,
                total_relevant,
                metrics_top_ks,
                row_mask=zero_history_mask,
            )

            if train and optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)
                optimizer.step()

            loss_sum += loss.detach()
            batches += 1
            metric_user_count.add_(batch_metric_user_count)
            for key, value in batch_metric_sums.items():
                metric_sums[key].add_(value)
            zero_history_metric_user_count.add_(batch_zero_history_metric_user_count)
            for key, value in batch_zero_history_metric_sums.items():
                zero_history_metric_sums[key].add_(value)

    # Transfer all epoch statistics together. This is the only metric-related
    # CUDA synchronization in the epoch, replacing several synchronizations
    # per batch from Python scalar extraction.
    metric_names = list(metric_sums)
    packed_statistics = torch.stack(
        [
            loss_sum.to(dtype=torch.float64),
            baseline_metric_user_count.to(dtype=torch.float64),
            metric_user_count.to(dtype=torch.float64),
            zero_history_metric_user_count.to(dtype=torch.float64),
            *(baseline_metric_sums[name] for name in metric_names),
            *(metric_sums[name] for name in metric_names),
            *(zero_history_metric_sums[name] for name in metric_names),
        ]
    ).cpu().tolist()
    loss = float(packed_statistics[0]) / max(batches, 1)
    baseline_metric_user_count_value = int(packed_statistics[1])
    metric_user_count_value = int(packed_statistics[2])
    zero_history_metric_user_count_value = int(packed_statistics[3])
    cursor = 4
    baseline_metric_sums_value = dict(zip(metric_names, packed_statistics[cursor : cursor + len(metric_names)]))
    cursor += len(metric_names)
    metric_sums_value = dict(zip(metric_names, packed_statistics[cursor : cursor + len(metric_names)]))
    cursor += len(metric_names)
    zero_history_metric_sums_value = dict(zip(metric_names, packed_statistics[cursor : cursor + len(metric_names)]))
    baseline_metrics = finalize_rank_metrics(baseline_metric_sums_value, baseline_metric_user_count_value)
    metrics: Dict[str, Any] = finalize_rank_metrics(metric_sums_value, metric_user_count_value)
    metrics.update(
        finalize_zero_history_rank_metrics(
            zero_history_metric_sums_value,
            zero_history_metric_user_count_value,
        )
    )
    metrics["loss"] = loss
    metrics["rank_metric_user_count"] = metric_user_count_value
    return loss, metrics, baseline_metrics


class BSTRankerMatrixScorer:
    """Matrix-ranking scorer for an in-memory BST model."""

    def __init__(self, model: BSTRanker):
        self.model = model

    def prepare_for_eval(self, device: str) -> None:
        self.model = self.model.to(device)
        self.model.eval()

    def score_batch(self, batch: Dict[str, Any], device: str) -> MatrixBatchScores:
        loss, scores, _ = compute_bst_listwise_loss_and_scores(self.model, batch, device)
        return MatrixBatchScores(scores=scores, loss=loss)


def _log_bst_epoch_metrics(
    experiment_tracker: Optional[Any],
    iteration: int,
    train_loss: float,
    val_loss: float,
    val_unseen_loss: float,
) -> None:
    if experiment_tracker is None:
        return
    experiment_tracker.log_scalar("Training Loss History", "Train Loss", float(train_loss), iteration)
    experiment_tracker.log_scalar("Training Loss History", "Validation Loss", float(val_loss), iteration)
    experiment_tracker.log_scalar("Training Loss History", "Validation Unseen Users Loss", float(val_unseen_loss), iteration)


def _listwise_history_metric_names(metrics_top_ks: List[int]) -> List[str]:
    return [f"ndcg@{k}" for k in metrics_top_ks]


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


def _save_checkpoint_atomically(
    checkpoint: Dict[str, Any],
    checkpoint_path: Path,
) -> None:
    """Publish a complete checkpoint without exposing a partially written file."""
    partial_path = checkpoint_path.with_name(f"{checkpoint_path.name}.partial")
    torch.save(checkpoint, partial_path)
    partial_path.replace(checkpoint_path)


def _log_bst_listwise_epoch_metrics(
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
    primary_title = f"Primary Ranking Metric ({primary_metric_key})"
    primary_series = f"Validation Unseen Users {primary_metric_key}"
    if calc_baseline_metrics:
        baseline_primary_metric = val_unseen_baseline_metrics.get(primary_metric_key)
        if baseline_primary_metric is not None:
            experiment_tracker.log_scalar(
                primary_title,
                primary_series,
                float(baseline_primary_metric),
                0,
            )
    primary_metric_value = val_unseen_metrics.get(primary_metric_key)
    if primary_metric_value is not None:
        experiment_tracker.log_scalar(
            primary_title,
            primary_series,
            float(primary_metric_value),
            iteration,
        )
    for k in metrics_top_ks:
        metric_name = f"ndcg@{k}"
        metric_label = f"NDCG@{k}"
        if calc_baseline_metrics:
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
        for split_label, metrics in (
            ("Train", train_metrics),
            ("Validation", val_metrics),
            ("Validation Unseen Users", val_unseen_metrics),
        ):
            metric_value = metrics.get(metric_name)
            if metric_value is None:
                continue
            experiment_tracker.log_scalar(
                metric_label,
                f"{split_label} {metric_label}",
                float(metric_value),
                iteration,
            )
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
    metrics_top_ks = list(metrics_top_ks)
    if not metrics_top_ks:
        raise ValueError("metrics_top_ks must contain at least one value")
    if any(k <= 0 for k in metrics_top_ks):
        raise ValueError("metrics_top_ks values must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be nonnegative")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be nonnegative")
    if not 0.0 < lr_scheduler_factor < 1.0:
        raise ValueError("lr_scheduler_factor must be in (0, 1)")
    if lr_scheduler_patience < 0:
        raise ValueError("lr_scheduler_patience must be nonnegative")
    if gradient_clip_max_norm <= 0.0:
        raise ValueError("gradient_clip_max_norm must be positive")
    if bst_max_train_batches_per_epoch is not None and bst_max_train_batches_per_epoch <= 0:
        raise ValueError("bst_max_train_batches_per_epoch must be positive when provided")
    if best_checkpoint_callback is not None and checkpoints_dir is None:
        raise ValueError("checkpoints_dir is required when best_checkpoint_callback is provided")
    if checkpoints_dir is not None:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

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
    best_reset_val_metric = float("-inf")
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None
    best_epoch: Optional[int] = None
    epochs_completed = 0
    stopped_early = False
    baseline_metrics: Dict[str, Dict[str, float]] = {}

    for epoch in tqdm(range(epochs), desc="Training epochs", disable=disable_progress):
        calc_baseline_metrics = epoch == 0
        train_loss, train_metrics, train_baseline_metrics = run_bst_listwise_epoch(
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
        val_loss, val_metrics, val_baseline_metrics = run_bst_listwise_epoch(
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
            max_batches=None,
        )
        val_unseen_loss, val_unseen_metrics, val_unseen_baseline_metrics = run_bst_listwise_epoch(
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
            max_batches=None,
        )
        if calc_baseline_metrics:
            baseline_metrics = {
                "train": dict(train_baseline_metrics),
                "val": dict(val_baseline_metrics),
                "val_unseen_users": dict(val_unseen_baseline_metrics),
            }

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_unseen_loss"].append(val_unseen_loss)
        _append_split_metrics_to_history(history, "train", train_metrics, listwise_metric_names)
        _append_split_metrics_to_history(history, "val", val_metrics, listwise_metric_names)
        _append_split_metrics_to_history(history, "val_unseen", val_unseen_metrics, listwise_metric_names)

        _log_bst_epoch_metrics(
            experiment_tracker,
            epoch + 1,
            train_loss,
            val_loss,
            val_unseen_loss,
        )
        _log_bst_listwise_epoch_metrics(
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

        if primary_metric is not None:
            scheduler.step(primary_metric)
        else:
            scheduler.step(float("-inf"))

        epoch_number = epoch + 1
        epochs_completed = epoch_number
        new_checkpoint_best = (
            primary_metric is not None
            and primary_metric > best_val_metric
        )
        if new_checkpoint_best and primary_metric is not None:
            best_val_metric = primary_metric
            best_val_loss = val_unseen_loss
            best_epoch = epoch_number
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        significant_improvement = (
            primary_metric is not None
            and primary_metric > best_reset_val_metric
            and (primary_metric - best_reset_val_metric) >= early_stopping_min_delta
        )
        patience_reset = primary_metric is not None and significant_improvement
        if patience_reset and primary_metric is not None:
            best_reset_val_metric = primary_metric
            patience_counter = 0
        else:
            patience_counter += 1

        if new_checkpoint_best and checkpoints_dir is not None:
            checkpoint_path = checkpoints_dir / "bst_ranker_best.pth"
            _save_checkpoint_atomically(
                checkpoint={
                    "epoch": epoch_number,
                    "best_epoch": best_epoch,
                    "epochs_completed": epochs_completed,
                    "stopped_early": False,
                    "patience_counter": patience_counter,
                    "model_state_dict": best_state_dict,
                    "val_unseen_loss": val_unseen_loss,
                    "primary_metric_name": primary_metric_name,
                    "best_val_metric": best_val_metric,
                    "val_unseen_primary_metric": primary_metric,
                    "history": history,
                    "baseline_metrics": baseline_metrics,
                    "metadata": checkpoint_metadata,
                },
                checkpoint_path=checkpoint_path,
            )
            if best_checkpoint_callback is not None:
                best_checkpoint_callback(checkpoint_path)

        epochs_since_best = (
            epoch_number - best_epoch
            if best_epoch is not None
            else None
        )
        required_metric = best_reset_val_metric + early_stopping_min_delta
        current_metric_text = f"{primary_metric:.6f}" if primary_metric is not None else "n/a"
        best_metric_text = f"{best_val_metric:.6f}" if best_epoch is not None else "n/a"
        reference_text = f"{best_reset_val_metric:.6f}" if best_reset_val_metric != float("-inf") else "n/a"
        required_text = f"{required_metric:.6f}" if best_reset_val_metric != float("-inf") else "n/a"
        logger.info(
            "BST early-stopping status: epoch=%d current_%s=%s checkpoint_best=%s "
            "best_epoch=%s epochs_since_best=%s patience=%d/%d patience_reference=%s "
            "required_metric=%s min_delta=%.6f learning_rate=%.8g "
            "new_checkpoint_best=%s patience_reset=%s",
            epoch_number,
            primary_metric_name,
            current_metric_text,
            best_metric_text,
            best_epoch if best_epoch is not None else "n/a",
            epochs_since_best if epochs_since_best is not None else "n/a",
            patience_counter,
            patience,
            reference_text,
            required_text,
            early_stopping_min_delta,
            float(optimizer.param_groups[0]["lr"]),
            new_checkpoint_best,
            patience_reset,
        )

        if patience_counter >= patience:
            stopped_early = True
            logger.info(
                "BST early stopping triggered at epoch %d after patience reached %d/%d",
                epoch_number,
                patience_counter,
                patience,
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if checkpoints_dir is not None:
            # The weights still come from best_epoch; refresh the metadata so
            # the checkpoint also describes how the complete run finished.
            _save_checkpoint_atomically(
                checkpoint={
                    "epoch": best_epoch,
                    "best_epoch": best_epoch,
                    "epochs_completed": epochs_completed,
                    "stopped_early": stopped_early,
                    "patience_counter": patience_counter,
                    "model_state_dict": best_state_dict,
                    "val_unseen_loss": best_val_loss,
                    "primary_metric_name": primary_metric_name,
                    "best_val_metric": best_val_metric,
                    "val_unseen_primary_metric": best_val_metric,
                    "history": history,
                    "baseline_metrics": baseline_metrics,
                    "metadata": checkpoint_metadata,
                },
                checkpoint_path=checkpoints_dir / "bst_ranker_best.pth",
            )

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_metric": best_val_metric,
        "primary_metric_name": primary_metric_name,
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "stopped_early": stopped_early,
        "patience_counter": patience_counter,
        "baseline_metrics": baseline_metrics,
    }
