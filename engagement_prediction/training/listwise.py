"""Shared epoch and training lifecycle for canonical listwise models."""

from __future__ import annotations

from contextlib import nullcontext
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from engagement_prediction.training.model_artifacts import (
    write_torch_checkpoint_atomically,
)
from engagement_prediction.training.ranking import (
    calc_baseline_ndcg_tensor_sums_for_batch,
    empty_ndcg_metric_tensor_sums,
    finalize_rank_metrics,
    finalize_zero_history_rank_metrics,
    log_random_baseline_histogram,
    log_zero_history_rank_metrics,
    ndcg_metric_tensor_sums_for_batch,
    topk_ranked_labels_for_scores,
)


LossAndScores = Callable[
    [nn.Module, dict[str, Any], str],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
EpochRunner = Callable[..., tuple[float, dict[str, Any], dict[str, Any]]]


def _filtered_metrics(
    metric_sums: dict[str, float],
    user_count: int,
    *,
    include_dcg_metrics: bool,
) -> dict[str, float]:
    """Keep the metric subset requested by a particular model family."""

    metrics = finalize_rank_metrics(metric_sums, user_count)
    if include_dcg_metrics:
        return metrics
    return {key: value for key, value in metrics.items() if key.startswith("ndcg@")}


def _filtered_zero_history_metrics(
    metric_sums: dict[str, float],
    user_count: int,
    *,
    include_dcg_metrics: bool,
) -> dict[str, float | int]:
    """Apply the same model-specific filtering to the zero-history slice."""

    metrics = finalize_zero_history_rank_metrics(metric_sums, user_count)
    if include_dcg_metrics:
        return metrics
    return {
        key: value
        for key, value in metrics.items()
        if key.startswith("zero_history_ndcg@")
        or key == "zero_history_rank_metric_user_count"
    }


def run_listwise_epoch(
    *,
    compute_loss_and_scores: LossAndScores,
    include_dcg_metrics: bool,
    zero_grad_set_to_none: bool,
    train: bool,
    split_name: str,
    model: nn.Module,
    device: str,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    disable_progress: bool,
    gradient_clip_max_norm: float,
    metrics_top_ks: list[int],
    calc_baseline_metrics: bool,
    max_batches: Optional[int],
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    """Run one listwise epoch with one packed device-to-host metric transfer."""

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
                optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

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
            loss, scores, labels = compute_loss_and_scores(model, batch, device)

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
            batch_metric_sums, batch_metric_user_count = (
                ndcg_metric_tensor_sums_for_batch(
                    top_ranked_labels,
                    total_relevant,
                    metrics_top_ks,
                )
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

    # Packing all accumulators into one tensor produces one device-to-host
    # synchronization at the end of the epoch instead of one per metric and
    # batch.  Keep the packing/unpacking order in lockstep below.
    metric_names = list(metric_sums)
    packed_statistics = torch.stack(
        [
            loss_sum.to(dtype=torch.float64),
            baseline_metric_user_count.to(dtype=torch.float64),
            baseline_zero_history_metric_user_count.to(dtype=torch.float64),
            metric_user_count.to(dtype=torch.float64),
            zero_history_metric_user_count.to(dtype=torch.float64),
            *(baseline_metric_sums[name] for name in metric_names),
            *(baseline_zero_history_metric_sums[name] for name in metric_names),
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

    baseline_metrics: dict[str, Any] = _filtered_metrics(
        baseline_sums,
        baseline_user_count,
        include_dcg_metrics=include_dcg_metrics,
    )
    baseline_metrics.update(
        _filtered_zero_history_metrics(
            baseline_zero_history_sums,
            baseline_zero_history_user_count,
            include_dcg_metrics=include_dcg_metrics,
        )
    )
    baseline_metrics["rank_metric_user_count"] = baseline_user_count
    metrics: dict[str, Any] = _filtered_metrics(
        learned_sums,
        metric_user_count_value,
        include_dcg_metrics=include_dcg_metrics,
    )
    metrics.update(
        _filtered_zero_history_metrics(
            zero_history_sums,
            zero_history_user_count,
            include_dcg_metrics=include_dcg_metrics,
        )
    )
    metrics["loss"] = loss
    metrics["rank_metric_user_count"] = metric_user_count_value
    return loss, metrics, baseline_metrics


def _append_epoch_history(
    history: dict[str, list[float]],
    *,
    losses: Mapping[str, float],
    split_metrics: Mapping[str, Mapping[str, Any]],
    metric_names: list[str],
) -> None:
    """Append one epoch while preserving stable serialized history keys."""

    for split_name, loss in losses.items():
        history[f"{split_name}_loss"].append(float(loss))
    for split_name, metrics in split_metrics.items():
        for metric_name in metric_names:
            value = metrics.get(metric_name)
            history[f"{split_name}_{metric_name}"].append(
                float(value) if value is not None else float("nan")
            )


def _log_epoch_metrics(
    *,
    experiment_tracker: Optional[Any],
    iteration: int,
    losses: Mapping[str, float],
    split_metrics: Mapping[str, Mapping[str, Any]],
    metrics_top_ks: list[int],
    primary_metric_key: str,
) -> None:
    """Report learned metrics at their one-based epoch iteration."""

    if experiment_tracker is None:
        return
    for series, split_name in (
        ("Train Loss", "train"),
        ("Validation Loss", "val"),
        ("Validation Unseen Users Loss", "val_unseen"),
    ):
        experiment_tracker.log_scalar(
            "Training Loss History",
            series,
            float(losses[split_name]),
            iteration,
        )
    primary_value = split_metrics["val_unseen"].get(primary_metric_key)
    if primary_value is not None:
        experiment_tracker.log_scalar(
            f"Primary Ranking Metric ({primary_metric_key})",
            f"Validation Unseen Users {primary_metric_key}",
            float(primary_value),
            iteration,
        )
    for k in metrics_top_ks:
        metric_name = f"ndcg@{k}"
        title = f"NDCG@{k}"
        for split_label, split_name in (
            ("Train", "train"),
            ("Validation", "val"),
            ("Validation Unseen Users", "val_unseen"),
        ):
            value = split_metrics[split_name].get(metric_name)
            if value is not None:
                experiment_tracker.log_scalar(
                    title,
                    f"{split_label} {title}",
                    float(value),
                    iteration,
                )
    log_zero_history_rank_metrics(
        experiment_tracker,
        {
            "train": split_metrics["train"],
            "validation": split_metrics["val"],
            "validation_unseen_users": split_metrics["val_unseen"],
        },
        metrics_top_ks,
        iteration,
    )


def _checkpoint_payload(
    *,
    epoch: int,
    epochs_completed: int,
    stopped_early: bool,
    patience_counter: int,
    state_dict: Mapping[str, torch.Tensor],
    val_unseen_loss: float,
    primary_metric_name: str,
    best_val_metric: float,
    history: Mapping[str, list[float]],
    baseline_metrics: Mapping[str, Mapping[str, Any]],
    checkpoint_metadata: Mapping[str, Any],
    checkpoint_extra_fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge shared and model-specific state without ambiguous key overrides."""

    protected_keys = {
        "epoch",
        "best_epoch",
        "epochs_completed",
        "stopped_early",
        "patience_counter",
        "model_state_dict",
        "val_unseen_loss",
        "primary_metric_name",
        "best_val_metric",
        "val_unseen_primary_metric",
        "history",
        "baseline_metrics",
        "metadata",
    }
    collisions = protected_keys.intersection(checkpoint_extra_fields)
    if collisions:
        raise ValueError(
            "checkpoint_extra_fields cannot replace shared checkpoint keys: "
            f"{sorted(collisions)}"
        )
    return {
        "epoch": epoch,
        "best_epoch": epoch,
        "epochs_completed": epochs_completed,
        "stopped_early": stopped_early,
        "patience_counter": patience_counter,
        "model_state_dict": state_dict,
        **dict(checkpoint_extra_fields),
        "val_unseen_loss": val_unseen_loss,
        "primary_metric_name": primary_metric_name,
        "best_val_metric": best_val_metric,
        "val_unseen_primary_metric": best_val_metric,
        "history": history,
        "baseline_metrics": baseline_metrics,
        "metadata": checkpoint_metadata,
    }


def train_listwise_model(
    *,
    model: nn.Module,
    epoch_runner: EpochRunner,
    model_label: str,
    checkpoint_filename: str,
    checkpoint_extra_fields: Mapping[str, Any],
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
    metrics_top_ks: list[int],
    max_train_batches_per_epoch: Optional[int],
    checkpoint_metadata: Mapping[str, Any],
    best_checkpoint_callback: Optional[Callable[[Path], None]],
    experiment_tracker: Optional[Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Train, early-stop, checkpoint, and reload one canonical listwise model."""

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
    if max_train_batches_per_epoch is not None and max_train_batches_per_epoch <= 0:
        raise ValueError(
            "max_train_batches_per_epoch must be positive when provided"
        )
    if best_checkpoint_callback is not None and checkpoints_dir is None:
        raise ValueError(
            "checkpoints_dir is required when best_checkpoint_callback is provided"
        )
    if not model_label.strip():
        raise ValueError("model_label must not be empty")
    if not checkpoint_filename.strip():
        raise ValueError("checkpoint_filename must not be empty")
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
    primary_metric_key = f"ndcg@{metrics_top_ks[0]}"
    primary_metric_name = f"val_unseen_{primary_metric_key}"
    metric_names = [f"ndcg@{k}" for k in metrics_top_ks]
    # Preserve the established serialized history order: all split losses
    # precede the per-split ranking metrics in checkpoints and JSON outputs.
    history: dict[str, list[float]] = {
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
    best_state_dict: Optional[dict[str, torch.Tensor]] = None
    best_epoch: Optional[int] = None
    epochs_completed = 0
    stopped_early = False
    baseline_metrics: dict[str, dict[str, Any]] = {}

    for epoch in tqdm(
        range(epochs),
        desc="Training epochs",
        disable=disable_progress,
    ):
        # Random baselines depend only on each batch's labels.  Compute them
        # while epoch one's batches are already resident instead of making an
        # extra pre-training pass through all three loaders.
        calc_baseline_metrics = epoch == 0
        epoch_outputs = {}
        for split_name, display_name, loader, train_split, max_batches in (
            (
                "train",
                "Train",
                train_loader,
                True,
                max_train_batches_per_epoch,
            ),
            ("val", "Validation", val_loader, False, None),
            (
                "val_unseen",
                "Validation Unseen Users",
                val_unseen_loader,
                False,
                None,
            ),
        ):
            epoch_outputs[split_name] = epoch_runner(
                train=train_split,
                split_name=display_name,
                model=model,
                device=device,
                dataloader=loader,
                optimizer=optimizer if train_split else None,
                disable_progress=disable_progress,
                gradient_clip_max_norm=gradient_clip_max_norm,
                metrics_top_ks=metrics_top_ks,
                calc_baseline_metrics=calc_baseline_metrics,
                max_batches=max_batches,
            )

        losses = {
            split_name: float(output[0])
            for split_name, output in epoch_outputs.items()
        }
        split_metrics = {
            split_name: output[1]
            for split_name, output in epoch_outputs.items()
        }
        if calc_baseline_metrics:
            baseline_metrics = {
                "train": dict(epoch_outputs["train"][2]),
                "val": dict(epoch_outputs["val"][2]),
                "val_unseen_users": dict(epoch_outputs["val_unseen"][2]),
            }
            log_random_baseline_histogram(
                experiment_tracker,
                baseline_metrics,
                metrics_top_ks,
            )

        _append_epoch_history(
            history,
            losses=losses,
            split_metrics=split_metrics,
            metric_names=metric_names,
        )
        _log_epoch_metrics(
            experiment_tracker=experiment_tracker,
            iteration=epoch + 1,
            losses=losses,
            split_metrics=split_metrics,
            metrics_top_ks=metrics_top_ks,
            primary_metric_key=primary_metric_key,
        )

        primary_value = split_metrics["val_unseen"].get(primary_metric_key)
        primary_metric = (
            float(primary_value) if primary_value is not None else None
        )
        scheduler.step(
            primary_metric if primary_metric is not None else float("-inf")
        )
        epoch_number = epoch + 1
        epochs_completed = epoch_number
        # Checkpoint selection tracks every improvement, whereas patience is
        # reset only by an improvement of at least min_delta.  Keeping these
        # references separate avoids throwing away a genuinely best model just
        # because its gain was too small to extend training.
        new_checkpoint_best = (
            primary_metric is not None and primary_metric > best_val_metric
        )
        if new_checkpoint_best and primary_metric is not None:
            best_val_metric = primary_metric
            best_val_loss = losses["val_unseen"]
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
            checkpoint_path = checkpoints_dir / checkpoint_filename
            if best_epoch is None or best_state_dict is None:
                raise RuntimeError("Checkpoint-best state was not retained")
            write_torch_checkpoint_atomically(
                checkpoint_path,
                _checkpoint_payload(
                    epoch=best_epoch,
                    epochs_completed=epochs_completed,
                    stopped_early=False,
                    patience_counter=patience_counter,
                    state_dict=best_state_dict,
                    val_unseen_loss=best_val_loss,
                    primary_metric_name=primary_metric_name,
                    best_val_metric=best_val_metric,
                    history=history,
                    baseline_metrics=baseline_metrics,
                    checkpoint_metadata=checkpoint_metadata,
                    checkpoint_extra_fields=checkpoint_extra_fields,
                ),
            )
            if best_checkpoint_callback is not None:
                # Stage-specific callbacks export serving TorchScript from the
                # just-published checkpoint.  A callback failure is fatal so a
                # completed stage never advertises stale serving files.
                best_checkpoint_callback(checkpoint_path)

        epochs_since_best = (
            epoch_number - best_epoch if best_epoch is not None else None
        )
        required_metric = best_reset_val_metric + early_stopping_min_delta
        logger.info(
            "%s early-stopping status: epoch=%d current_%s=%s checkpoint_best=%s "
            "best_epoch=%s epochs_since_best=%s patience=%d/%d "
            "patience_reference=%s required_metric=%s min_delta=%.6f "
            "learning_rate=%.8g new_checkpoint_best=%s patience_reset=%s",
            model_label,
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
                "%s early stopping triggered at epoch %d after patience reached %d/%d",
                model_label,
                epoch_number,
                patience_counter,
                patience,
            )
            break

    if best_state_dict is not None and best_epoch is not None:
        model.load_state_dict(best_state_dict)
        if checkpoints_dir is not None:
            write_torch_checkpoint_atomically(
                checkpoints_dir / checkpoint_filename,
                _checkpoint_payload(
                    epoch=best_epoch,
                    epochs_completed=epochs_completed,
                    stopped_early=stopped_early,
                    patience_counter=patience_counter,
                    state_dict=best_state_dict,
                    val_unseen_loss=best_val_loss,
                    primary_metric_name=primary_metric_name,
                    best_val_metric=best_val_metric,
                    history=history,
                    baseline_metrics=baseline_metrics,
                    checkpoint_metadata=checkpoint_metadata,
                    checkpoint_extra_fields=checkpoint_extra_fields,
                ),
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
