"""Reusable listwise training primitives for the canonical two-tower model."""

from __future__ import annotations

from contextlib import nullcontext
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from engagement_prediction.training.ranking import (
    MatrixBatchScores,
    calc_baseline_ndcg_tensor_sums_for_batch,
    empty_ndcg_metric_tensor_sums,
    finalize_rank_metrics,
    finalize_zero_history_rank_metrics,
    log_random_baseline_histogram,
    log_zero_history_rank_metrics,
    ndcg_metric_tensor_sums_for_batch,
    topk_ranked_labels_for_scores,
)


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
    """Run one listwise epoch with one device-to-host metric transfer."""

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
    baseline_metric_user_count = torch.zeros(
        (), device=device, dtype=torch.int64
    )
    baseline_zero_history_metric_sums = empty_ndcg_metric_tensor_sums(
        metrics_top_ks,
        device=device,
    )
    baseline_zero_history_metric_user_count = torch.zeros(
        (), device=device, dtype=torch.int64
    )
    metric_sums = empty_ndcg_metric_tensor_sums(metrics_top_ks, device=device)
    metric_user_count = torch.zeros((), device=device, dtype=torch.int64)
    zero_history_metric_sums = empty_ndcg_metric_tensor_sums(
        metrics_top_ks,
        device=device,
    )
    zero_history_metric_user_count = torch.zeros(
        (), device=device, dtype=torch.int64
    )

    with nullcontext() if train else torch.inference_mode():
        for batch_idx, batch in enumerate(
            tqdm(
                dataloader,
                desc=split_name,
                leave=False,
                disable=disable_progress,
            )
        ):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            history_mask = batch.get("history_mask")
            if history_mask is None or history_mask.dim() != 2:
                raise RuntimeError(
                    "history_mask must have shape [num_users, history_len]"
                )
            zero_history_mask = (~history_mask.any(dim=1)).to(
                device,
                dtype=torch.bool,
                non_blocking=True,
            )
            loss, scores, labels = compute_two_tower_listwise_loss_and_scores(
                model,
                batch,
                device,
            )

            if calc_baseline_metrics:
                (
                    baseline_batch_metric_sums,
                    baseline_batch_metric_user_count,
                ) = calc_baseline_ndcg_tensor_sums_for_batch(
                    labels,
                    metrics_top_ks,
                )
                baseline_metric_user_count.add_(baseline_batch_metric_user_count)
                for key, value in baseline_batch_metric_sums.items():
                    baseline_metric_sums[key].add_(value)
                (
                    baseline_zero_history_batch_metric_sums,
                    baseline_zero_history_batch_metric_user_count,
                ) = calc_baseline_ndcg_tensor_sums_for_batch(
                    labels[zero_history_mask],
                    metrics_top_ks,
                )
                baseline_zero_history_metric_user_count.add_(
                    baseline_zero_history_batch_metric_user_count
                )
                for key, value in baseline_zero_history_batch_metric_sums.items():
                    baseline_zero_history_metric_sums[key].add_(value)

            top_ranked_labels = topk_ranked_labels_for_scores(
                scores,
                labels,
                metrics_top_ks,
            )
            total_relevant = labels.sum(dim=1)
            (
                batch_metric_sums,
                batch_metric_user_count,
            ) = ndcg_metric_tensor_sums_for_batch(
                top_ranked_labels,
                total_relevant,
                metrics_top_ks,
            )
            (
                batch_zero_history_metric_sums,
                batch_zero_history_metric_user_count,
            ) = ndcg_metric_tensor_sums_for_batch(
                top_ranked_labels,
                total_relevant,
                metrics_top_ks,
                row_mask=zero_history_mask,
            )

            if train and optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip_max_norm,
                )
                optimizer.step()

            loss_sum.add_(loss.detach())
            batches += 1
            metric_user_count.add_(batch_metric_user_count)
            for key, value in batch_metric_sums.items():
                metric_sums[key].add_(value)
            zero_history_metric_user_count.add_(
                batch_zero_history_metric_user_count
            )
            for key, value in batch_zero_history_metric_sums.items():
                zero_history_metric_sums[key].add_(value)

    # Keep all epoch statistics on the device until one packed transfer. This
    # avoids the per-batch `.item()` synchronizations that throttled legacy
    # training while preserving exact top-K NDCG values.
    metric_names = list(metric_sums)
    packed_statistics = torch.stack(
        [
            loss_sum.to(dtype=torch.float64),
            baseline_metric_user_count.to(dtype=torch.float64),
            baseline_zero_history_metric_user_count.to(dtype=torch.float64),
            metric_user_count.to(dtype=torch.float64),
            zero_history_metric_user_count.to(dtype=torch.float64),
            *(baseline_metric_sums[name] for name in metric_names),
            *(
                baseline_zero_history_metric_sums[name]
                for name in metric_names
            ),
            *(metric_sums[name] for name in metric_names),
            *(zero_history_metric_sums[name] for name in metric_names),
        ]
    ).cpu().tolist()
    loss = float(packed_statistics[0]) / max(batches, 1)
    baseline_user_count = int(packed_statistics[1])
    baseline_zero_history_user_count = int(packed_statistics[2])
    metric_user_count_value = int(packed_statistics[3])
    zero_history_user_count = int(packed_statistics[4])
    cursor = 5
    baseline_sums = dict(
        zip(metric_names, packed_statistics[cursor : cursor + len(metric_names)])
    )
    cursor += len(metric_names)
    baseline_zero_history_sums = dict(
        zip(metric_names, packed_statistics[cursor : cursor + len(metric_names)])
    )
    cursor += len(metric_names)
    learned_sums = dict(
        zip(metric_names, packed_statistics[cursor : cursor + len(metric_names)])
    )
    cursor += len(metric_names)
    zero_history_sums = dict(
        zip(metric_names, packed_statistics[cursor : cursor + len(metric_names)])
    )

    baseline_metrics: Dict[str, Any] = {
        key: value
        for key, value in finalize_rank_metrics(
            baseline_sums,
            baseline_user_count,
        ).items()
        if key.startswith("ndcg@")
    }
    baseline_metrics.update({
        key: value
        for key, value in finalize_zero_history_rank_metrics(
            baseline_zero_history_sums,
            baseline_zero_history_user_count,
        ).items()
        if key.startswith("zero_history_ndcg@")
        or key == "zero_history_rank_metric_user_count"
    })
    baseline_metrics["rank_metric_user_count"] = baseline_user_count
    metrics: Dict[str, Any] = {
        key: value
        for key, value in finalize_rank_metrics(
            learned_sums,
            metric_user_count_value,
        ).items()
        if key.startswith("ndcg@")
    }
    metrics.update({
        key: value
        for key, value in finalize_zero_history_rank_metrics(
            zero_history_sums,
            zero_history_user_count,
        ).items()
        if key.startswith("zero_history_ndcg@")
        or key == "zero_history_rank_metric_user_count"
    })
    metrics["loss"] = loss
    metrics["rank_metric_user_count"] = metric_user_count_value
    return loss, metrics, baseline_metrics


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


def _append_split_metrics_to_history(
    history: Dict[str, List[float]],
    split_name: str,
    metrics: Dict[str, Any],
    metric_names: List[str],
) -> None:
    for metric_name in metric_names:
        value = metrics.get(metric_name)
        history[f"{split_name}_{metric_name}"].append(
            float(value) if value is not None else float("nan")
        )


def _save_checkpoint_atomically(
    checkpoint: Dict[str, Any],
    checkpoint_path: Path,
) -> None:
    partial_path = checkpoint_path.with_name(f"{checkpoint_path.name}.partial")
    try:
        torch.save(checkpoint, partial_path)
        partial_path.replace(checkpoint_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def _log_epoch_metrics(
    *,
    experiment_tracker: Optional[Any],
    iteration: int,
    train_loss: float,
    val_loss: float,
    val_unseen_loss: float,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    val_unseen_metrics: Dict[str, Any],
    metrics_top_ks: List[int],
    primary_metric_name: str,
) -> None:
    if experiment_tracker is None:
        return

    experiment_tracker.log_scalar(
        "Training Loss History", "Train Loss", float(train_loss), iteration
    )
    experiment_tracker.log_scalar(
        "Training Loss History", "Validation Loss", float(val_loss), iteration
    )
    experiment_tracker.log_scalar(
        "Training Loss History",
        "Validation Unseen Users Loss",
        float(val_unseen_loss),
        iteration,
    )
    primary_metric_key = primary_metric_name.replace("val_unseen_", "", 1)
    primary_title = f"Primary Ranking Metric ({primary_metric_key})"
    primary_series = f"Validation Unseen Users {primary_metric_key}"
    experiment_tracker.log_scalar(
        primary_title,
        primary_series,
        float(val_unseen_metrics[primary_metric_key]),
        iteration,
    )

    for k in metrics_top_ks:
        metric_name = f"ndcg@{k}"
        title = f"NDCG@{k}"
        for split_label, metrics in (
            ("Train", train_metrics),
            ("Validation", val_metrics),
            ("Validation Unseen Users", val_unseen_metrics),
        ):
            experiment_tracker.log_scalar(
                title,
                f"{split_label} {title}",
                float(metrics[metric_name]),
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
    """Train, checkpoint, and reload a canonical two-tower model."""

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
    if (
        max_train_batches_per_epoch is not None
        and max_train_batches_per_epoch <= 0
    ):
        raise ValueError(
            "max_train_batches_per_epoch must be positive when provided"
        )
    if best_checkpoint_callback is not None and checkpoints_dir is None:
        raise ValueError(
            "checkpoints_dir is required when best_checkpoint_callback is provided"
        )
    output_embedding_dim = int(model.output_embedding_dim)
    if output_embedding_dim <= 0:
        raise ValueError("model.output_embedding_dim must be positive")
    if checkpoints_dir is not None:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
    )

    primary_metric_name = f"val_unseen_ndcg@{metrics_top_ks[0]}"
    metric_names = [f"ndcg@{k}" for k in metrics_top_ks]
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_unseen_loss": [],
    }
    for split_name in ("train", "val", "val_unseen"):
        for metric_name in metric_names:
            history[f"{split_name}_{metric_name}"] = []

    best_val_metric = float("-inf")
    best_reset_val_metric = float("-inf")
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_epoch: Optional[int] = None
    epochs_completed = 0
    stopped_early = False
    baseline_metrics: Dict[str, Dict[str, Any]] = {}

    for epoch in tqdm(
        range(epochs),
        desc="Training epochs",
        disable=disable_progress,
    ):
        calc_baseline_metrics = epoch == 0
        train_loss, train_metrics, train_baseline_metrics = (
            run_two_tower_listwise_epoch(
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
                max_batches=max_train_batches_per_epoch,
            )
        )
        val_loss, val_metrics, val_baseline_metrics = (
            run_two_tower_listwise_epoch(
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
        )
        val_unseen_loss, val_unseen_metrics, val_unseen_baseline_metrics = (
            run_two_tower_listwise_epoch(
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
        )
        if calc_baseline_metrics:
            baseline_metrics = {
                "train": train_baseline_metrics,
                "val": val_baseline_metrics,
                "val_unseen_users": val_unseen_baseline_metrics,
            }
            log_random_baseline_histogram(
                experiment_tracker,
                baseline_metrics,
                metrics_top_ks,
            )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_unseen_loss"].append(val_unseen_loss)
        _append_split_metrics_to_history(
            history, "train", train_metrics, metric_names
        )
        _append_split_metrics_to_history(history, "val", val_metrics, metric_names)
        _append_split_metrics_to_history(
            history, "val_unseen", val_unseen_metrics, metric_names
        )
        _log_epoch_metrics(
            experiment_tracker=experiment_tracker,
            iteration=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            val_unseen_loss=val_unseen_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            val_unseen_metrics=val_unseen_metrics,
            metrics_top_ks=metrics_top_ks,
            primary_metric_name=primary_metric_name,
        )

        primary_metric_key = primary_metric_name.replace("val_unseen_", "", 1)
        primary_metric_value = val_unseen_metrics.get(primary_metric_key)
        primary_metric = (
            float(primary_metric_value)
            if primary_metric_value is not None
            else None
        )
        scheduler.step(
            primary_metric if primary_metric is not None else float("-inf")
        )

        epoch_number = epoch + 1
        epochs_completed = epoch_number
        new_checkpoint_best = (
            primary_metric is not None and primary_metric > best_val_metric
        )
        if new_checkpoint_best and primary_metric is not None:
            best_val_metric = primary_metric
            best_val_loss = val_unseen_loss
            best_epoch = epoch_number
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        significant_improvement = (
            primary_metric is not None
            and primary_metric > best_reset_val_metric
            and primary_metric - best_reset_val_metric
            >= early_stopping_min_delta
        )
        patience_reset = primary_metric is not None and significant_improvement
        if patience_reset and primary_metric is not None:
            best_reset_val_metric = primary_metric
            patience_counter = 0
        else:
            patience_counter += 1

        if new_checkpoint_best and checkpoints_dir is not None:
            checkpoint_path = checkpoints_dir / "two_tower_best.pth"
            _save_checkpoint_atomically(
                {
                    "epoch": epoch_number,
                    "best_epoch": best_epoch,
                    "epochs_completed": epochs_completed,
                    "stopped_early": False,
                    "patience_counter": patience_counter,
                    "model_state_dict": best_state_dict,
                    "output_embedding_dim": output_embedding_dim,
                    "val_unseen_loss": val_unseen_loss,
                    "primary_metric_name": primary_metric_name,
                    "best_val_metric": best_val_metric,
                    "val_unseen_primary_metric": primary_metric,
                    "history": history,
                    "baseline_metrics": baseline_metrics,
                    "metadata": checkpoint_metadata,
                },
                checkpoint_path,
            )
            if best_checkpoint_callback is not None:
                best_checkpoint_callback(checkpoint_path)

        epochs_since_best = (
            epoch_number - best_epoch if best_epoch is not None else None
        )
        required_metric = best_reset_val_metric + early_stopping_min_delta
        logger.info(
            "Two-tower early-stopping status: epoch=%d current_%s=%s "
            "checkpoint_best=%s best_epoch=%s epochs_since_best=%s "
            "patience=%d/%d patience_reference=%s required_metric=%s "
            "min_delta=%.6f learning_rate=%.8g new_checkpoint_best=%s "
            "patience_reset=%s",
            epoch_number,
            primary_metric_name,
            f"{primary_metric:.6f}" if primary_metric is not None else "n/a",
            f"{best_val_metric:.6f}" if best_epoch is not None else "n/a",
            best_epoch if best_epoch is not None else "n/a",
            epochs_since_best if epochs_since_best is not None else "n/a",
            patience_counter,
            patience,
            (
                f"{best_reset_val_metric:.6f}"
                if best_reset_val_metric != float("-inf")
                else "n/a"
            ),
            (
                f"{required_metric:.6f}"
                if best_reset_val_metric != float("-inf")
                else "n/a"
            ),
            early_stopping_min_delta,
            float(optimizer.param_groups[0]["lr"]),
            new_checkpoint_best,
            patience_reset,
        )

        if patience_counter >= patience:
            stopped_early = True
            logger.info(
                "Two-tower early stopping triggered at epoch %d after patience "
                "reached %d/%d",
                epoch_number,
                patience_counter,
                patience,
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if checkpoints_dir is not None:
            _save_checkpoint_atomically(
                {
                    "epoch": best_epoch,
                    "best_epoch": best_epoch,
                    "epochs_completed": epochs_completed,
                    "stopped_early": stopped_early,
                    "patience_counter": patience_counter,
                    "model_state_dict": best_state_dict,
                    "output_embedding_dim": output_embedding_dim,
                    "val_unseen_loss": best_val_loss,
                    "primary_metric_name": primary_metric_name,
                    "best_val_metric": best_val_metric,
                    "val_unseen_primary_metric": best_val_metric,
                    "history": history,
                    "baseline_metrics": baseline_metrics,
                    "metadata": checkpoint_metadata,
                },
                checkpoints_dir / "two_tower_best.pth",
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
        "output_embedding_dim": output_embedding_dim,
    }
