"""Canonical reporting helpers for model-training stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from engagement_prediction.experiment_tracking.base import ExperimentTracker


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


def write_history_length_plots(
    *,
    final_metrics: Dict[str, Dict[str, Any]],
    metrics_top_ks: List[int],
    output_dir: Path,
    best_epoch: int,
    tracker: Optional[ExperimentTracker],
) -> Dict[str, Path]:
    """Plot final validation NDCG and eligible query counts by history length."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    splits = (
        ("val", "Validation"),
        ("val_unseen_users", "Validation Unseen Users"),
    )
    labels = [
        bucket["label"]
        for bucket in final_metrics["val"]["history_length_breakdown"]
    ]
    positions = list(range(len(labels)))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for k in metrics_top_ks:
        metric = f"ndcg@{k}"
        figure, (metric_axis, count_axis) = plt.subplots(
            2,
            1,
            figsize=(11, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        try:
            for split_index, (split, label) in enumerate(splits):
                buckets = final_metrics[split]["history_length_breakdown"]
                values = [
                    float("nan") if bucket[metric] is None else bucket[metric]
                    for bucket in buckets
                ]
                (line,) = metric_axis.plot(
                    positions, values, marker="o", label=label
                )
                count_axis.bar(
                    [position + (split_index - 0.5) * 0.38 for position in positions],
                    [bucket["query_count"] for bucket in buckets],
                    width=0.38,
                    color=line.get_color(),
                    label=label,
                )
            metric_axis.set_title(
                f"Final validation by history length (best epoch {best_epoch})"
            )
            metric_axis.set_ylabel(f"NDCG@{k}")
            metric_axis.set_ylim(-0.03, 1.03)
            metric_axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            count_axis.set_ylabel("Eligible query rows")
            count_axis.set_ylim(0, max(1, count_axis.get_ylim()[1]))
            count_axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            count_axis.set_xlabel("Likes in model-input history (after truncation)")
            count_axis.set_xticks(positions, labels)
            for axis in (metric_axis, count_axis):
                axis.grid(True, axis="y", alpha=0.3)
                axis.legend()
            figure.tight_layout()
            output_path = output_dir / f"history_length_ndcg_at_{k}.png"
            figure.savefig(output_path, dpi=300, bbox_inches="tight")
            if tracker is not None:
                tracker.log_plot(
                    title="NDCG by history length",
                    series=metric,
                    figure=figure,
                    iteration=best_epoch,
                )
            output_paths[metric] = output_path
        finally:
            plt.close(figure)
    return output_paths
