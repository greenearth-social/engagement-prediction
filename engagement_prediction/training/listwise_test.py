"""Tests for the model-independent listwise training lifecycle."""

from __future__ import annotations

import logging
import math

import pytest
import torch
from torch import nn

from engagement_prediction.training import listwise
from engagement_prediction.training.listwise import run_listwise_epoch, train_listwise_model


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
        assert kwargs["history_length_bucket_boundaries"] is None
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


def _history_batch(lengths, ranked_labels):
    """Represent the history mask received after loader-side truncation/padding."""

    return {
        "history_mask": torch.arange(max(lengths) + 2)[None, :]
        < torch.tensor(lengths)[:, None],
        "scores": torch.tensor([[2.0, 1.0]] * len(lengths)),
        "labels": torch.tensor(ranked_labels, dtype=torch.float32),
    }


def _run_history_epoch(batches, boundaries, include_dcg_metrics, compute):
    return run_listwise_epoch(
        compute_loss_and_scores=compute,
        include_dcg_metrics=include_dcg_metrics,
        zero_grad_set_to_none=True,
        train=False,
        split_name="Validation",
        model=_TinyModel(),
        device="cpu",
        dataloader=batches,
        optimizer=None,
        disable_progress=True,
        gradient_clip_max_norm=1.0,
        metrics_top_ks=[1, 2, 30],
        calc_baseline_metrics=False,
        max_batches=None,
        history_length_bucket_boundaries=boundaries,
    )


def _fixture_loss_and_scores(model, batch, device):
    return torch.tensor(1.0), batch["scores"], batch["labels"]


@pytest.mark.parametrize("include_dcg_metrics", [False, True])
def test_history_buckets_preserve_scoring_and_aggregate_eligible_rows(
    monkeypatch, include_dcg_metrics
):
    lengths = [0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 40]
    ranked_labels = [[1, 0] if idx % 2 == 0 else [0, 1] for idx in range(len(lengths))]
    # The first bucket's three rows span unequal batches with means 1 and 0.
    # A no-positive row in the second batch must not enter bucket counts.
    batches = [
        _history_batch(lengths, ranked_labels),
        _history_batch([0, 0, 5], [[0, 1], [0, 1], [0, 0]]),
    ]
    ordinary = _run_history_epoch(
        batches, None, include_dcg_metrics, _fixture_loss_and_scores
    )
    calls = {"score": 0, "rank": 0, "cpu": 0}
    original_rank = listwise.topk_ranked_labels_for_scores
    original_cpu = torch.Tensor.cpu

    def compute(model, batch, device):
        calls["score"] += 1
        return _fixture_loss_and_scores(model, batch, device)

    def rank(*args, **kwargs):
        calls["rank"] += 1
        return original_rank(*args, **kwargs)

    def cpu(tensor, *args, **kwargs):
        calls["cpu"] += 1
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(listwise, "topk_ranked_labels_for_scores", rank)
    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    loss, metrics, baseline = _run_history_epoch(
        batches, [0, 1, 2, 4, 8, 16, 32], include_dcg_metrics, compute
    )

    assert calls == {"score": 2, "rank": 2, "cpu": 1}
    assert loss == ordinary[0]
    assert baseline == ordinary[2]
    assert "history_length_breakdown" not in baseline
    assert {key: value for key, value in metrics.items() if key != "history_length_breakdown"} == ordinary[1]
    buckets = metrics["history_length_breakdown"]
    assert [bucket["query_count"] for bucket in buckets] == [3, 1, 1, 2, 2, 2, 2, 2]
    assert [bucket["label"] for bucket in buckets] == ["0", "1", "2", "3–4", "5–8", "9–16", "17–32", ">32"]
    assert buckets[-1]["lower_bound"] == 33
    assert buckets[-1]["upper_bound"] is None
    second_rank_ndcg = 1.0 / math.log2(3)
    assert buckets[0]["ndcg@1"] == pytest.approx(1.0 / 3)
    assert buckets[0]["ndcg@2"] == pytest.approx((1 + 2 * second_rank_ndcg) / 3)
    assert buckets[1]["ndcg@1"] == 0.0
    assert buckets[1]["ndcg@2"] == pytest.approx(second_rank_ndcg)
    assert buckets[2]["ndcg@1"] == 1.0
    assert buckets[3]["ndcg@1"] == 0.5
    for k in [1, 2, 30]:
        metric_name = f"ndcg@{k}"
        weighted_mean = sum(bucket[metric_name] * bucket["query_count"] for bucket in buckets) / sum(bucket["query_count"] for bucket in buckets)
        assert weighted_mean == pytest.approx(metrics[metric_name])
        assert buckets[0][metric_name] == metrics[f"zero_history_{metric_name}"]
    assert buckets[0]["query_count"] == metrics["zero_history_rank_metric_user_count"]
    assert sum(bucket["query_count"] for bucket in buckets) == metrics["rank_metric_user_count"]
    assert all(bucket["ndcg@30"] == bucket["ndcg@2"] for bucket in buckets)


def test_history_buckets_count_only_the_truncated_input_mask():
    batch = _history_batch([0, 2, 2], [[1, 0], [1, 0], [0, 1]])
    # Two long histories have already been truncated by the loader to two likes;
    # the mask's remaining columns are padding, including for zero history.
    batch["untruncated_history_lengths"] = [0, 10, 50]
    _, metrics, _ = _run_history_epoch(
        [batch], [0, 1, 2, 4], False, _fixture_loss_and_scores
    )
    buckets = metrics["history_length_breakdown"]
    assert [bucket["query_count"] for bucket in buckets] == [1, 0, 2, 0, 0]
    assert buckets[2]["ndcg@1"] == 0.5
    for idx in [1, 3, 4]:
        assert all(buckets[idx][f"ndcg@{k}"] is None for k in [1, 2, 30])


def test_history_buckets_are_retained_for_an_empty_validation_split():
    loss, metrics, _ = _run_history_epoch(
        [], [0, 2], False, _fixture_loss_and_scores
    )
    assert loss == 0.0
    assert metrics["rank_metric_user_count"] == 0
    buckets = metrics["history_length_breakdown"]
    assert [bucket["label"] for bucket in buckets] == ["0", "1–2", ">2"]
    assert all(bucket["query_count"] == 0 for bucket in buckets)
    assert all(bucket[f"ndcg@{k}"] is None for bucket in buckets for k in [1, 2, 30])
