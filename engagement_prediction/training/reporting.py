"""Canonical reporting helpers for model-training stages."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


def write_bst_training_history_plot(
    history: Dict[str, List[float]],
    output_path: Path,
    best_epoch: Optional[int],
) -> None:
    """Write loss and primary-ranking curves for a completed BST run."""

    train_losses = history.get("train_loss", [])
    if not train_losses:
        raise ValueError("BST training history does not contain any epochs")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_suffixes = sorted(
        key.removeprefix("train_")
        for key in history
        if key.startswith("train_ndcg@")
    )
    if not metric_suffixes:
        raise ValueError("BST training history does not contain an NDCG metric")
    metric_suffix = metric_suffixes[0]
    epochs = range(1, len(train_losses) + 1)
    figure, (loss_axis, metric_axis) = plt.subplots(1, 2, figsize=(10, 6))
    for split, label in (
        ("train", "Train"),
        ("val", "Validation"),
        ("val_unseen", "Validation Unseen Users"),
    ):
        loss_axis.plot(epochs, history[f"{split}_loss"], label=label)
        metric_axis.plot(
            epochs,
            history[f"{split}_{metric_suffix}"],
            label=label,
        )
    loss_axis.set_title("BST Training Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Listwise Loss")
    metric_axis.set_title(f"BST {metric_suffix}")
    metric_axis.set_xlabel("Epoch")
    metric_axis.set_ylabel(metric_suffix)
    for axis in (loss_axis, metric_axis):
        axis.grid(True, alpha=0.3)
        axis.legend()
        if best_epoch is not None:
            axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.6)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_two_tower_training_history_plot(
    history: Dict[str, List[float]],
    output_path: Path,
    best_epoch: Optional[int],
) -> None:
    """Write loss and NDCG curves for canonical two-tower training."""

    train_losses = history.get("train_loss", [])
    if not train_losses:
        raise ValueError("Two-tower training history does not contain any epochs")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_suffixes = sorted(
        key.removeprefix("train_")
        for key in history
        if key.startswith("train_ndcg@")
    )
    if not metric_suffixes:
        raise ValueError("Two-tower training history does not contain an NDCG metric")
    metric_suffix = metric_suffixes[0]
    epochs = range(1, len(train_losses) + 1)
    figure, (loss_axis, metric_axis) = plt.subplots(1, 2, figsize=(10, 6))
    for split, label in (
        ("train", "Train"),
        ("val", "Validation"),
        ("val_unseen", "Validation Unseen Users"),
    ):
        loss_axis.plot(epochs, history[f"{split}_loss"], label=label)
        metric_axis.plot(
            epochs,
            history[f"{split}_{metric_suffix}"],
            label=label,
        )
    loss_axis.set_title("Two-Tower Training Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Listwise Loss")
    metric_axis.set_title(f"Two-Tower {metric_suffix}")
    metric_axis.set_xlabel("Epoch")
    metric_axis.set_ylabel(metric_suffix)
    for axis in (loss_axis, metric_axis):
        axis.grid(True, alpha=0.3)
        axis.legend()
        if best_epoch is not None:
            axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.6)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
