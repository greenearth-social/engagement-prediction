import logging

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from engagement_prediction.models.two_tower import TwoTowerModel
from engagement_prediction.training import two_tower as two_tower_training


class _TinyTwoTower(nn.Module):
    def __init__(self, *, output_embedding_dim: int = 3, temperature: float = 0.5):
        super().__init__()
        self.output_embedding_dim = int(output_embedding_dim)
        self.similarity_temperature = float(temperature)
        self.user_projection = nn.Linear(2, output_embedding_dim, bias=False)
        self.post_projection = nn.Linear(2, output_embedding_dim, bias=False)
        self.author_embedding = nn.Embedding(8, 2)

    def encode_user(
        self,
        history_embeddings,
        history_mask,
        history_author_indices,
    ):
        mask = history_mask.unsqueeze(-1).to(dtype=history_embeddings.dtype)
        fused = history_embeddings + self.author_embedding(history_author_indices)
        pooled = (fused * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(self.user_projection(pooled), dim=-1)

    def encode_post(self, post_embeddings, post_author_indices):
        fused = post_embeddings + self.author_embedding(post_author_indices)
        return F.normalize(self.post_projection(fused), dim=-1)


def _batch():
    return {
        "history_embeddings": torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        "history_mask": torch.tensor([[True, True], [False, False]]),
        "history_author_indices": torch.tensor([[2, 3], [0, 0]]),
        "history_time_deltas_hours": torch.tensor(
            [[2.0, 12.0], [0.0, 0.0]]
        ),
        "history_prior_cumulative_likes": torch.tensor([[3, 4], [0, 0]]),
        "candidate_post_embeddings": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
        "candidate_post_author_idx": torch.tensor([3, 4, 5]),
        "candidate_prior_cumulative_likes": torch.tensor([4, 5, 10]),
        "candidate_post_age_hours": torch.tensor([0.0, 10.0, 100.0]),
        "label_matrix": torch.tensor(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        ),
    }


class _SingleBatchDataset(Dataset):
    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        return self.batch


class _RecordingTracker:
    def __init__(self):
        self.calls = []

    def log_scalar(self, title, series, value, iteration):
        self.calls.append(
            {
                "title": title,
                "series": series,
                "value": value,
                "iteration": iteration,
            }
        )


def _loader():
    return DataLoader(
        _SingleBatchDataset(_batch()),
        batch_size=None,
        shuffle=False,
    )


def test_compute_two_tower_listwise_loss_uses_native_author_fields():
    torch.manual_seed(3)
    model = _TinyTwoTower()
    batch = _batch()

    loss, scores, labels = (
        two_tower_training.compute_two_tower_listwise_loss_and_scores(
            model,
            batch,
            "cpu",
        )
    )
    user_embeddings = model.encode_user(
        batch["history_embeddings"],
        batch["history_mask"],
        batch["history_author_indices"],
    )
    post_embeddings = model.encode_post(
        batch["candidate_post_embeddings"],
        batch["candidate_post_author_idx"],
    )
    expected_scores = user_embeddings @ post_embeddings.T / 0.5
    targets = labels / labels.sum(dim=1, keepdim=True)
    expected_loss = -(targets * F.log_softmax(expected_scores, dim=1)).sum(1).mean()

    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(loss, expected_loss)
    assert scores.shape == labels.shape == (2, 3)


def test_compute_two_tower_loss_ignores_non_serving_features():
    model = _TinyTwoTower()
    batch = _batch()
    loss, scores, _ = two_tower_training.compute_two_tower_listwise_loss_and_scores(
        model, batch, "cpu"
    )
    changed = {
        **batch,
        "history_time_deltas_hours": torch.full((2, 2), 999.0),
        "history_prior_cumulative_likes": torch.full((2, 2), 999),
        "candidate_prior_cumulative_likes": torch.full((3,), 999),
        "candidate_post_age_hours": torch.full((3,), 999.0),
    }
    changed_loss, changed_scores, _ = (
        two_tower_training.compute_two_tower_listwise_loss_and_scores(
            model, changed, "cpu"
        )
    )

    torch.testing.assert_close(changed_scores, scores)
    torch.testing.assert_close(changed_loss, loss)


def test_compute_two_tower_loss_rejects_missing_authors_and_empty_positive_rows():
    model = _TinyTwoTower()
    missing_authors = _batch()
    del missing_authors["history_author_indices"]
    with pytest.raises(RuntimeError, match="author index tensors"):
        two_tower_training.compute_two_tower_listwise_loss_and_scores(
            model, missing_authors, "cpu"
        )

    no_positives = _batch()
    no_positives["label_matrix"][1] = 0
    with pytest.raises(RuntimeError, match="at least one positive"):
        two_tower_training.compute_two_tower_listwise_loss_and_scores(
            model, no_positives, "cpu"
        )


def test_two_tower_epoch_uses_topk_ndcg_only_and_updates_weights(monkeypatch):
    model = _TinyTwoTower()
    loader = _loader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    before = model.user_projection.weight.detach().clone()

    def fail_argsort(*_args, **_kwargs):
        raise AssertionError("canonical two-tower metrics must use topk")

    monkeypatch.setattr(torch, "argsort", fail_argsort)
    loss, metrics, baseline_metrics = (
        two_tower_training.run_two_tower_listwise_epoch(
            train=True,
            split_name="Train",
            model=model,
            device="cpu",
            dataloader=loader,
            optimizer=optimizer,
            disable_progress=True,
            gradient_clip_max_norm=1.0,
            metrics_top_ks=[1, 2],
            calc_baseline_metrics=True,
            max_batches=None,
        )
    )

    assert loss >= 0.0
    assert metrics["rank_metric_user_count"] == 2
    assert metrics["zero_history_rank_metric_user_count"] == 1
    assert set(baseline_metrics) == {"ndcg@1", "ndcg@2"}
    assert not any(key.startswith("dcg@") for key in metrics)
    assert not any("recall" in key or "average_precision" in key for key in metrics)
    assert not torch.equal(before, model.user_projection.weight)


def test_canonical_two_tower_runs_one_native_batch_optimizer_step():
    torch.manual_seed(9)
    model = TwoTowerModel(
        post_embedding_dim=2,
        author_table_num_rows=8,
        author_embedding_dim=2,
        content_projection_dim=4,
        author_projection_dim=2,
        user_hidden_dim=4,
        post_hidden_dim=4,
        output_embedding_dim=3,
        max_history_len=2,
        dropout_rate=0.0,
        author_unknown_dropout_rate=0.0,
        similarity_temperature=0.25,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    before = model.user_tower.history_encoder.output_projection[-1].weight.detach().clone()

    loss, metrics, baseline_metrics = (
        two_tower_training.run_two_tower_listwise_epoch(
            train=True,
            split_name="Train",
            model=model,
            device="cpu",
            dataloader=_loader(),
            optimizer=optimizer,
            disable_progress=True,
            gradient_clip_max_norm=1.0,
            metrics_top_ks=[1, 2],
            calc_baseline_metrics=True,
            max_batches=None,
        )
    )

    assert loss >= 0.0
    assert metrics["rank_metric_user_count"] == 2
    assert baseline_metrics["ndcg@2"] >= 0.0
    assert not torch.equal(
        before,
        model.user_tower.history_encoder.output_projection[-1].weight,
    )


def test_train_two_tower_uses_unseen_ndcg_for_checkpoint_and_min_delta_patience(
    tmp_path,
    monkeypatch,
    caplog,
):
    model = _TinyTwoTower()
    loader = _loader()
    tracker = _RecordingTracker()
    unseen_values = iter([0.50, 0.45, 0.56])
    callback_epochs = []

    def fake_epoch(**kwargs):
        ndcg = (
            next(unseen_values)
            if kwargs["split_name"] == "Validation Unseen Users"
            else 0.25
        )
        return (
            1.0,
            {
                "loss": 1.0,
                "ndcg@1": ndcg,
                "zero_history_ndcg@1": ndcg,
                "zero_history_rank_metric_user_count": 1,
                "rank_metric_user_count": 2,
            },
            {"dcg@1": 0.4, "ndcg@1": 0.4},
        )

    def callback(path):
        assert path == tmp_path / "two_tower_best.pth"
        assert path.is_file()
        assert not (tmp_path / "two_tower_best.pth.partial").exists()
        callback_epochs.append(torch.load(path, weights_only=False)["epoch"])

    monkeypatch.setattr(
        two_tower_training,
        "run_two_tower_listwise_epoch",
        fake_epoch,
    )
    logger = logging.getLogger("two-tower-training-test")
    caplog.set_level(logging.INFO, logger=logger.name)

    results = two_tower_training.train_two_tower_model(
        model=model,
        train_loader=loader,
        val_loader=loader,
        val_unseen_loader=loader,
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
        checkpoint_metadata={"model_config": {"output_embedding_dim": 3}},
        best_checkpoint_callback=callback,
        experiment_tracker=tracker,
        logger=logger,
    )

    assert results["primary_metric_name"] == "val_unseen_ndcg@1"
    assert results["best_epoch"] == 3
    assert results["epochs_completed"] == 3
    assert results["stopped_early"] is True
    assert results["patience_counter"] == 2
    assert results["output_embedding_dim"] == 3
    assert callback_epochs == [1, 3]

    checkpoint = torch.load(tmp_path / "two_tower_best.pth", weights_only=False)
    assert checkpoint["epoch"] == checkpoint["best_epoch"] == 3
    assert checkpoint["epochs_completed"] == 3
    assert checkpoint["stopped_early"] is True
    assert checkpoint["patience_counter"] == 2
    assert checkpoint["output_embedding_dim"] == 3
    assert checkpoint["best_val_metric"] == pytest.approx(0.56)
    assert checkpoint["metadata"] == {
        "model_config": {"output_embedding_dim": 3}
    }

    for series in (
        "Train NDCG@1",
        "Validation NDCG@1",
        "Validation Unseen Users NDCG@1",
    ):
        iterations = [
            call["iteration"]
            for call in tracker.calls
            if call["series"] == series
        ]
        assert iterations == [0, 1, 2, 3]
    assert "new_checkpoint_best=True" in caplog.text
    assert "patience_reset=False" in caplog.text
    assert "patience=2/2" in caplog.text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epochs": 0}, "epochs must be positive"),
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ({"weight_decay": -1.0}, "weight_decay must be nonnegative"),
        ({"patience": 0}, "patience must be positive"),
        ({"early_stopping_min_delta": -0.1}, "must be nonnegative"),
        ({"gradient_clip_max_norm": 0.0}, "must be positive"),
        ({"max_train_batches_per_epoch": 0}, "must be positive"),
    ],
)
def test_train_two_tower_validates_settings(tmp_path, kwargs, message):
    arguments = {
        "model": _TinyTwoTower(),
        "train_loader": _loader(),
        "val_loader": _loader(),
        "val_unseen_loader": _loader(),
        "device": "cpu",
        "epochs": 1,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "patience": 2,
        "early_stopping_min_delta": 0.0,
        "checkpoints_dir": tmp_path,
        "disable_progress": True,
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 1,
        "gradient_clip_max_norm": 1.0,
        "metrics_top_ks": [1],
        "max_train_batches_per_epoch": None,
        "checkpoint_metadata": {},
        "best_checkpoint_callback": None,
        "experiment_tracker": None,
        "logger": logging.getLogger("two-tower-validation-test"),
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        two_tower_training.train_two_tower_model(**arguments)
