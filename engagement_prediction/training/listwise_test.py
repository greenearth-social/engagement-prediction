"""Tests for the model-independent listwise training lifecycle."""

from __future__ import annotations

import logging

import pytest
import torch
from torch import nn

from engagement_prediction.training.listwise import train_listwise_model


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))


def test_training_restores_best_state_and_refreshes_checkpoint_metadata(tmp_path):
    model = _TinyModel()
    unseen_metrics = [0.50, 0.40, 0.30]
    train_epoch = 0
    callback_checkpoints = []

    def epoch_runner(**kwargs):
        nonlocal train_epoch
        if kwargs["train"]:
            train_epoch += 1
            with torch.no_grad():
                kwargs["model"].weight.fill_(float(train_epoch))
        metric = (
            unseen_metrics[train_epoch - 1]
            if kwargs["split_name"] == "Validation Unseen Users"
            else 0.25
        )
        baseline = (
            {
                "ndcg@1": 0.20,
                "zero_history_ndcg@1": 0.10,
                "rank_metric_user_count": 2,
                "zero_history_rank_metric_user_count": 1,
            }
            if kwargs["calc_baseline_metrics"]
            else {}
        )
        return 1.0, {"ndcg@1": metric}, baseline

    def checkpoint_callback(path):
        callback_checkpoints.append(torch.load(path, weights_only=False))

    result = train_listwise_model(
        model=model,
        epoch_runner=epoch_runner,
        model_label="Tiny",
        checkpoint_filename="tiny_best.pth",
        checkpoint_extra_fields={"model_kind": "tiny"},
        train_loader=object(),
        val_loader=object(),
        val_unseen_loader=object(),
        device="cpu",
        epochs=5,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        patience=2,
        early_stopping_min_delta=0.10,
        checkpoints_dir=tmp_path,
        disable_progress=True,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=1,
        gradient_clip_max_norm=1.0,
        metrics_top_ks=[1],
        max_train_batches_per_epoch=None,
        checkpoint_metadata={"model_config": {"kind": "tiny"}},
        best_checkpoint_callback=checkpoint_callback,
        experiment_tracker=None,
        logger=logging.getLogger("listwise-training-test"),
    )

    assert result["best_epoch"] == 1
    assert result["epochs_completed"] == 3
    assert result["stopped_early"] is True
    assert result["patience_counter"] == 2
    assert result["model"].weight.item() == pytest.approx(1.0)
    assert list(result["history"]) == [
        "train_loss",
        "val_loss",
        "val_unseen_loss",
        "train_ndcg@1",
        "val_ndcg@1",
        "val_unseen_ndcg@1",
    ]
    assert len(callback_checkpoints) == 1
    assert callback_checkpoints[0]["epoch"] == 1
    assert callback_checkpoints[0]["epochs_completed"] == 1
    assert callback_checkpoints[0]["stopped_early"] is False

    checkpoint = torch.load(tmp_path / "tiny_best.pth", weights_only=False)
    assert checkpoint["epoch"] == checkpoint["best_epoch"] == 1
    assert checkpoint["epochs_completed"] == 3
    assert checkpoint["stopped_early"] is True
    assert checkpoint["patience_counter"] == 2
    assert checkpoint["model_kind"] == "tiny"
    assert checkpoint["metadata"] == {"model_config": {"kind": "tiny"}}
    assert checkpoint["model_state_dict"]["weight"].item() == pytest.approx(1.0)

