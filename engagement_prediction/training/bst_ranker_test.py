from datetime import datetime, timezone
import logging

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from engagement_prediction.data import author_vocabulary, dataset_hydration, training_index
from engagement_prediction.data.datasets import (
    HydratedBucketedEngagementDataset,
    create_hydrated_data_loader,
)
from engagement_prediction.models.bst_ranker import BSTRanker
from engagement_prediction.training import bst_ranker as bst_training


def _model(*, use_popularity_feature: bool) -> BSTRanker:
    torch.manual_seed(4)
    return BSTRanker(
        post_embedding_dim=2,
        author_table_num_rows=8,
        author_embedding_dim=3,
        content_projection_dim=4,
        author_projection_dim=2,
        model_dim=4,
        time_embedding_dim=2,
        num_attention_heads=2,
        num_transformer_layers=1,
        transformer_ff_dim=8,
        dropout_rate=0.0,
        author_unknown_dropout_rate=0.0,
        norm_first=False,
        time_delta_bucket_boundaries_hours=[1.0, 6.0, 24.0],
        prediction_hidden_dims=[4],
        use_popularity_feature=use_popularity_feature,
        popularity_projection_dim=2 if use_popularity_feature else 0,
        popularity_log_mean=1.0,
        popularity_log_std=2.0,
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "history_embeddings": torch.tensor([[[1.0, 0.0]], [[0.0, 0.0]]]),
        "history_mask": torch.tensor([[True], [False]]),
        "history_author_indices": torch.tensor([[2], [0]]),
        "history_time_deltas_hours": torch.tensor([[2.0], [0.0]]),
        "history_prior_cumulative_likes": torch.tensor([[3.0], [0.0]]),
        "candidate_post_embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        "candidate_post_author_idx": torch.tensor([3, 4, 5]),
        "candidate_prior_cumulative_likes": torch.tensor([4.0, 5.0, 10.0]),
        "candidate_post_age_hours": torch.tensor([0.0, 10.0, 100.0]),
        "label_matrix": torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
    }


class _SingleBatchDataset(Dataset):
    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.batch


class _RecordingTracker:
    def __init__(self):
        self.calls = []

    def log_scalar(self, title, series, value, iteration):
        self.calls.append({
            "kind": "scalar",
            "title": title,
            "series": series,
            "value": value,
            "iteration": iteration,
        })

    def log_histogram(
        self,
        title,
        series,
        values,
        iteration=0,
        xlabels=None,
        xaxis=None,
        yaxis=None,
        labels=None,
        mode=None,
    ):
        self.calls.append({
            "kind": "histogram",
            "title": title,
            "series": series,
            "values": values,
            "iteration": iteration,
            "xlabels": xlabels,
            "xaxis": xaxis,
            "yaxis": yaxis,
            "labels": labels,
            "mode": mode,
        })


def _write_dataset(bundle, name, frame):
    dataset_path = bundle / name
    dataset_path.mkdir()
    frame.write_parquet(dataset_path / "part-00000.parquet")


def _native_bundle(tmp_path):
    bundle = tmp_path / "hydrated_training_data_test"
    bundle.mkdir()
    embeddings = np.lib.format.open_memmap(
        bundle / "embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(4, 2),
    )
    embeddings[:] = np.array([[1, 0], [0, 1], [2, 0], [0, 2]], dtype=np.float32)
    embeddings.flush()
    del embeddings
    hour = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    created = datetime(2026, 1, 1, 11, 30, tzinfo=timezone.utc)
    _write_dataset(bundle, "queries", pl.DataFrame({
        "did": ["u1", "u2"],
        "query_hour": [hour, hour],
        "user_cohort": ["seen", "seen"],
        "split": ["train", "train"],
        "positive_count": [1, 1],
    }, schema=dataset_hydration.QUERY_SCHEMA))
    _write_dataset(bundle, "query_positives", pl.DataFrame({
        "did": ["u1", "u2"],
        "query_hour": [hour, hour],
        "subject_uri": ["p1", "p2"],
        "like_created_at": [hour, hour],
        "emb_idx": [0, 1],
        "post_created_at": [created, created],
        "author_idx": [2, 3],
        "prior_like_count": [4, 5],
    }, schema=dataset_hydration.QUERY_POSITIVE_SCHEMA))
    _write_dataset(bundle, "query_histories", pl.DataFrame({
        "did": ["u1", "u2"],
        "query_hour": [hour, hour],
        "history_subject_uris": [["h1"], []],
        "history_like_created_ats": [[datetime(2026, 1, 1, 10, tzinfo=timezone.utc)], []],
        "history_emb_indices": [[2], []],
        "history_author_indices": [[4], []],
        "history_prior_like_counts": [[3], []],
    }, schema=dataset_hydration.QUERY_HISTORY_SCHEMA))
    _write_dataset(bundle, "hourly_negative_candidates", pl.DataFrame({
        "query_hour": [hour],
        "subject_uri": ["n1"],
        "selection_source": ["random"],
        "emb_idx": [3],
        "post_created_at": [created],
        "author_idx": [5],
        "prior_like_count": [10],
    }, schema=dataset_hydration.HOURLY_NEGATIVE_SCHEMA))
    _write_dataset(bundle, "posts", pl.DataFrame({
        "subject_uri": ["p1", "p2", "h1", "n1"],
        "emb_idx": [0, 1, 2, 3],
        "post_created_at": [created, created, created, created],
        "author_did": ["a2", "a3", "a4", "a5"],
        "author_idx": [2, 3, 4, 5],
        "is_reply": [False, False, False, False],
        "is_positive": [True, True, False, False],
        "is_history": [False, False, True, False],
        "is_negative": [False, False, False, True],
    }, schema=dataset_hydration.POST_SCHEMA))
    _write_dataset(bundle, "authors", pl.DataFrame({
        "author_did": ["a2", "a3", "a4", "a5"],
        "author_idx": [2, 3, 4, 5],
        "training_feature_count": [1, 1, 1, 1],
        "training_positive_count": [1, 1, 0, 0],
        "training_history_count": [0, 0, 1, 0],
        "training_negative_count": [0, 0, 0, 1],
    }, schema=author_vocabulary.AUTHOR_VOCABULARY_SCHEMA))
    training_index.build_loader_index(
        posts_path=bundle / "posts",
        queries_path=bundle / "queries",
        query_positives_path=bundle / "query_positives",
        query_histories_path=bundle / "query_histories",
        hourly_negative_candidates_path=bundle / "hourly_negative_candidates",
        embeddings_path=bundle / "embeddings.npy",
        authors_path=bundle / "authors",
        output_path=bundle / "loader_index",
        logger=None,
    )
    return bundle


def test_compute_bst_listwise_loss_uses_native_fields_and_ignores_candidate_age():
    model = _model(use_popularity_feature=True)
    batch = _batch()

    loss, scores, labels = bst_training.compute_bst_listwise_loss_and_scores(model, batch, "cpu")
    changed_age_batch = {**batch, "candidate_post_age_hours": torch.tensor([999.0, 888.0, 777.0])}
    changed_loss, changed_scores, _ = bst_training.compute_bst_listwise_loss_and_scores(
        model,
        changed_age_batch,
        "cpu",
    )

    assert torch.isfinite(loss)
    assert scores.shape == labels.shape == (2, 3)
    torch.testing.assert_close(changed_scores, scores)
    torch.testing.assert_close(changed_loss, loss)


def test_compute_bst_listwise_loss_rejects_rows_without_positives():
    batch = _batch()
    batch["label_matrix"][1] = 0

    with pytest.raises(RuntimeError, match="at least one positive"):
        bst_training.compute_bst_listwise_loss_and_scores(
            _model(use_popularity_feature=True),
            batch,
            "cpu",
        )


def test_native_stage7_batch_runs_one_optimizer_step(tmp_path):
    dataset = HydratedBucketedEngagementDataset(
        _native_bundle(tmp_path),
        split="train",
        max_history_len=2,
        bst_additional_batch_negatives=None,
        seed=7,
        logger=None,
    )
    loader = create_hydrated_data_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=1,
        seed=7,
        resample_candidates_each_epoch=False,
    )
    model = _model(use_popularity_feature=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    batch = next(iter(loader))

    before = model.post_feature_encoder.content_projection.weight.detach().clone()
    loss, metrics, baseline_metrics = bst_training.run_bst_listwise_epoch(
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

    assert "candidate_post_age_hours" in batch
    assert loss >= 0.0
    assert metrics["rank_metric_user_count"] == 2
    assert "mean_average_precision" not in metrics
    assert "zero_history_mean_average_precision" not in metrics
    assert "mean_average_precision" not in baseline_metrics
    assert baseline_metrics["rank_metric_user_count"] == 2
    assert baseline_metrics["zero_history_rank_metric_user_count"] == 1
    assert baseline_metrics["zero_history_ndcg@2"] >= 0.0
    assert not any("recall" in key for key in metrics | baseline_metrics)
    assert not torch.equal(before, model.post_feature_encoder.content_projection.weight)


def test_bst_epoch_metrics_do_not_require_full_argsort(monkeypatch):
    model = _model(use_popularity_feature=False)
    loader = DataLoader(_SingleBatchDataset(_batch()), batch_size=None, shuffle=False)

    def fail_argsort(*args, **kwargs):
        raise AssertionError("canonical BST metrics should use topk, not full argsort")

    monkeypatch.setattr(torch, "argsort", fail_argsort)

    loss, metrics, baseline_metrics = bst_training.run_bst_listwise_epoch(
        train=False,
        split_name="Validation",
        model=model,
        device="cpu",
        dataloader=loader,
        optimizer=None,
        disable_progress=True,
        gradient_clip_max_norm=1.0,
        metrics_top_ks=[1, 2],
        calc_baseline_metrics=True,
        max_batches=None,
    )

    assert loss >= 0.0
    assert metrics["rank_metric_user_count"] == 2
    assert metrics["ndcg@2"] >= 0.0
    assert baseline_metrics["ndcg@2"] >= 0.0


def test_train_bst_piggybacks_baseline_histogram_on_epoch_one_and_logs_detailed_early_stopping(
    tmp_path,
    monkeypatch,
    caplog,
):
    model = _model(use_popularity_feature=False)
    loader = DataLoader(_SingleBatchDataset(_batch()), batch_size=None, shuffle=False)
    tracker = _RecordingTracker()
    unseen_values = iter([0.50, 0.45, 0.56])
    callback_epochs = []
    baseline_by_split = {
        "Train": (0.10, 0.11),
        "Validation": (0.20, 0.21),
        "Validation Unseen Users": (0.30, 0.31),
    }
    epoch_calls = []

    def fake_epoch(**kwargs):
        split_name = kwargs["split_name"]
        epoch_calls.append((split_name, kwargs["calc_baseline_metrics"]))
        ndcg = next(unseen_values) if split_name == "Validation Unseen Users" else 0.25
        metrics = {
            "loss": 1.0,
            "ndcg@1": ndcg,
            "zero_history_ndcg@1": ndcg - 0.05,
            "zero_history_rank_metric_user_count": 1,
            "rank_metric_user_count": 2,
        }
        if kwargs["calc_baseline_metrics"]:
            baseline_ndcg, zero_history_baseline_ndcg = baseline_by_split[split_name]
            baseline = {
                "dcg@1": baseline_ndcg,
                "ndcg@1": baseline_ndcg,
                "zero_history_dcg@1": zero_history_baseline_ndcg,
                "zero_history_ndcg@1": zero_history_baseline_ndcg,
                "rank_metric_user_count": 10,
                "zero_history_rank_metric_user_count": 4,
            }
        else:
            baseline = {}
        return 1.0, metrics, baseline

    def best_checkpoint_callback(checkpoint_path):
        assert checkpoint_path == tmp_path / "bst_ranker_best.pth"
        assert checkpoint_path.is_file()
        assert not (tmp_path / "bst_ranker_best.pth.partial").exists()
        callback_epochs.append(torch.load(checkpoint_path, weights_only=False)["epoch"])

    monkeypatch.setattr(bst_training, "run_bst_listwise_epoch", fake_epoch)
    logger = logging.getLogger("bst-training-test")
    caplog.set_level(logging.INFO, logger=logger.name)

    results = bst_training.train_bst_ranker_model(
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
        bst_max_train_batches_per_epoch=None,
        checkpoint_metadata={"model_config": {"model_dim": 4}},
        best_checkpoint_callback=best_checkpoint_callback,
        experiment_tracker=tracker,
        logger=logger,
    )

    assert results["best_epoch"] == 3
    assert results["epochs_completed"] == 3
    assert results["stopped_early"] is True
    assert results["patience_counter"] == 2
    checkpoint = torch.load(tmp_path / "bst_ranker_best.pth", weights_only=False)
    assert checkpoint["epoch"] == 3
    assert checkpoint["best_epoch"] == 3
    assert checkpoint["epochs_completed"] == 3
    assert checkpoint["stopped_early"] is True
    assert checkpoint["patience_counter"] == 2
    assert checkpoint["best_val_metric"] == pytest.approx(0.56)
    assert checkpoint["metadata"] == {"model_config": {"model_dim": 4}}
    assert callback_epochs == [1, 3]
    assert epoch_calls == [
        ("Train", True),
        ("Validation", True),
        ("Validation Unseen Users", True),
        ("Train", False),
        ("Validation", False),
        ("Validation Unseen Users", False),
        ("Train", False),
        ("Validation", False),
        ("Validation Unseen Users", False),
    ]
    assert results["baseline_metrics"]["train"]["ndcg@1"] == pytest.approx(0.10)
    assert results["baseline_metrics"]["val"]["zero_history_ndcg@1"] == pytest.approx(0.21)
    assert results["baseline_metrics"]["val_unseen_users"]["ndcg@1"] == pytest.approx(0.30)
    histogram_calls = [call for call in tracker.calls if call["kind"] == "histogram"]
    assert histogram_calls == [{
        "kind": "histogram",
        "title": "Random Baseline NDCG@1",
        "series": "Random Baseline",
        "values": [[0.10, 0.20, 0.30], [0.11, 0.21, 0.31]],
        "iteration": 0,
        "xlabels": ["Train", "Validation", "Validation Unseen Users"],
        "xaxis": "Split",
        "yaxis": "NDCG@1",
        "labels": ["All observations", "Zero-history only"],
        "mode": "group",
    }]
    assert tracker.calls[0]["kind"] == "histogram"
    for series in (
        "Train NDCG@1",
        "Validation NDCG@1",
        "Validation Unseen Users NDCG@1",
    ):
        calls = [
            call
            for call in tracker.calls
            if call["kind"] == "scalar"
            and call["title"] == "NDCG@1"
            and call["series"] == series
        ]
        assert [call["iteration"] for call in calls] == [1, 2, 3]
    primary_calls = [
        call
        for call in tracker.calls
        if call["kind"] == "scalar"
        and call["title"] == "Primary Ranking Metric (ndcg@1)"
    ]
    assert [call["iteration"] for call in primary_calls] == [1, 2, 3]
    loss_calls = [
        call
        for call in tracker.calls
        if call["kind"] == "scalar" and call["title"] == "Training Loss History"
    ]
    assert min(call["iteration"] for call in loss_calls) == 1
    zero_history_calls = [
        call
        for call in tracker.calls
        if call["kind"] == "scalar" and "Zero-History" in call["series"]
    ]
    assert zero_history_calls
    assert min(call["iteration"] for call in zero_history_calls) == 1
    assert not any(
        call["kind"] == "scalar" and call["iteration"] == 0
        for call in tracker.calls
    )
    assert not any("Recall" in call["title"] or "Recall" in call["series"] for call in tracker.calls)
    assert not any("MAP" in call["title"] or "MAP" in call["series"] for call in tracker.calls)
    status_lines = [record.message for record in caplog.records if "BST early-stopping status" in record.message]
    assert len(status_lines) == 3
    assert "epoch=3" in status_lines[-1]
    assert "checkpoint_best=0.560000" in status_lines[-1]
    assert "best_epoch=3" in status_lines[-1]
    assert "epochs_since_best=0" in status_lines[-1]
    assert "patience=2/2" in status_lines[-1]
    assert "patience_reference=0.500000" in status_lines[-1]
    assert "required_metric=0.600000" in status_lines[-1]
    assert "new_checkpoint_best=True" in status_lines[-1]
    assert "patience_reset=False" in status_lines[-1]


def test_save_checkpoint_atomically_replaces_partial_file(tmp_path):
    checkpoint_path = tmp_path / "bst_ranker_best.pth"
    partial_path = tmp_path / "bst_ranker_best.pth.partial"
    checkpoint_path.write_bytes(b"old checkpoint")

    bst_training._save_checkpoint_atomically(
        checkpoint={"epoch": 2, "tensor": torch.tensor([1.0, 2.0])},
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.is_file()
    assert not partial_path.exists()
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["epoch"] == 2
    torch.testing.assert_close(checkpoint["tensor"], torch.tensor([1.0, 2.0]))


def test_best_checkpoint_callback_failure_propagates(tmp_path, monkeypatch):
    model = _model(use_popularity_feature=False)
    loader = DataLoader(_SingleBatchDataset(_batch()), batch_size=None, shuffle=False)

    def fake_epoch(**kwargs):
        baseline = {
            "dcg@1": 0.4,
            "ndcg@1": 0.4,
            "zero_history_dcg@1": 0.4,
            "zero_history_ndcg@1": 0.4,
            "rank_metric_user_count": 2,
            "zero_history_rank_metric_user_count": 1,
        } if kwargs["calc_baseline_metrics"] else {}
        return (
            1.0,
            {"loss": 1.0, "ndcg@1": 0.5, "rank_metric_user_count": 2},
            baseline,
        )

    def fail_callback(checkpoint_path):
        assert checkpoint_path.is_file()
        raise RuntimeError("export failed")

    monkeypatch.setattr(bst_training, "run_bst_listwise_epoch", fake_epoch)

    with pytest.raises(RuntimeError, match="export failed"):
        bst_training.train_bst_ranker_model(
            model=model,
            train_loader=loader,
            val_loader=loader,
            val_unseen_loader=loader,
            device="cpu",
            epochs=2,
            learning_rate=1.0e-3,
            weight_decay=0.0,
            patience=2,
            early_stopping_min_delta=0.0,
            checkpoints_dir=tmp_path,
            disable_progress=True,
            lr_scheduler_factor=0.5,
            lr_scheduler_patience=1,
            gradient_clip_max_norm=1.0,
            metrics_top_ks=[1],
            bst_max_train_batches_per_epoch=None,
            checkpoint_metadata={},
            best_checkpoint_callback=fail_callback,
            experiment_tracker=None,
            logger=logging.getLogger("bst-training-callback-failure-test"),
        )

    assert (tmp_path / "bst_ranker_best.pth").is_file()
    assert not (tmp_path / "bst_ranker_best.pth.partial").exists()
