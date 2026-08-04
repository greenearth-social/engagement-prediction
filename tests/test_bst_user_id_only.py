"""Tests for the experimental BST user-ID-only ranker."""
import importlib

import pytest
import torch


stage_train_bst_user_id_only = importlib.import_module("utils.03_train.stage_train_bst_user_id_only")
BSTUserIdOnlyRanker = stage_train_bst_user_id_only.BSTUserIdOnlyRanker
_compute_bst_user_id_only_listwise_loss_and_preds = (
    stage_train_bst_user_id_only._compute_bst_user_id_only_listwise_loss_and_preds
)
_log_bst_user_id_only_epoch_metrics = stage_train_bst_user_id_only._log_bst_user_id_only_epoch_metrics

DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS = [1.0, 3.0, 6.0, 12.0, 24.0]


class _RecordingTracker:
    def __init__(self) -> None:
        self.calls = []

    def log_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        self.calls.append(
            {
                "title": title,
                "series": series,
                "value": value,
                "iteration": iteration,
            }
        )


def _scalar_calls_by_series(calls, series: str):
    return [call for call in calls if call["series"] == series]


def _make_model(*, dropout_rate: float = 0.0, prepend_target_user_token: bool = False) -> BSTUserIdOnlyRanker:
    torch.manual_seed(123)
    return BSTUserIdOnlyRanker(
        post_liker_user_table_num_rows=8,
        post_liker_user_embedding_dim=3,
        post_liker_projection_dim=2,
        model_dim=5,
        time_embedding_dim=3,
        num_attention_heads=2,
        num_transformer_layers=1,
        transformer_ff_dim=16,
        dropout_rate=dropout_rate,
        norm_first=False,
        time_delta_bucket_boundaries_hours=DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS,
        prediction_hidden_dims=[8, 4],
        post_liker_pooling_tau_hours=10.0,
        target_user_projection_dim=2,
        post_liker_user_dropout_rate=0.0,
        target_user_dropout_rate=0.0,
        prepend_target_user_token=prepend_target_user_token,
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "history_mask": torch.tensor(
            [
                [True, True, False],
                [True, False, False],
            ],
            dtype=torch.bool,
        ),
        "history_time_deltas_hours": torch.tensor(
            [
                [2.0, 25.0, 999.0],
                [0.5, 777.0, 888.0],
            ],
            dtype=torch.float32,
        ),
        "history_post_liker_user_indices": torch.tensor(
            [
                [[2, 3, 0], [4, 0, 0], [0, 0, 0]],
                [[5, 2, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=torch.long,
        ),
        "history_post_liker_time_gap_hours": torch.tensor(
            [
                [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "candidate_post_liker_user_indices": torch.tensor(
            [
                [2, 4, 0],
                [3, 0, 0],
            ],
            dtype=torch.long,
        ),
        "candidate_post_liker_time_gap_hours": torch.tensor(
            [
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "target_user_indices": torch.tensor([2, 5], dtype=torch.long),
    }


def _mixed_zero_history_batch() -> dict[str, torch.Tensor]:
    batch = {key: value.clone() for key, value in _batch().items()}
    batch["history_mask"][1] = False
    return batch


def _expected_matrix_scores(model: BSTUserIdOnlyRanker, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    num_users = batch["history_mask"].shape[0]
    num_candidates = batch["candidate_post_liker_user_indices"].shape[0]
    rows = []
    for user_idx in range(num_users):
        row_scores = []
        for candidate_idx in range(num_candidates):
            score = model(
                history_mask=batch["history_mask"][user_idx:user_idx + 1],
                history_time_deltas_hours=batch["history_time_deltas_hours"][user_idx:user_idx + 1],
                history_post_liker_user_indices=batch["history_post_liker_user_indices"][user_idx:user_idx + 1],
                history_post_liker_time_gap_hours=batch["history_post_liker_time_gap_hours"][user_idx:user_idx + 1],
                candidate_post_liker_user_indices=batch["candidate_post_liker_user_indices"][candidate_idx:candidate_idx + 1],
                candidate_post_liker_time_gap_hours=batch["candidate_post_liker_time_gap_hours"][candidate_idx:candidate_idx + 1],
                target_user_indices=batch["target_user_indices"][user_idx:user_idx + 1],
            )
            row_scores.append(score.squeeze(0))
        rows.append(torch.stack(row_scores))
    return torch.stack(rows)


def test_bst_user_id_only_matrix_scorer_matches_repeated_forward_path():
    model = _make_model().eval()
    batch = _batch()

    with torch.no_grad():
        expected = _expected_matrix_scores(model, batch)
        actual = model.score_candidate_matrix(**batch)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_bst_user_id_only_target_user_token_matrix_scorer_matches_repeated_forward_path():
    model = _make_model(prepend_target_user_token=True).eval()
    batch = _batch()

    with torch.no_grad():
        expected = _expected_matrix_scores(model, batch)
        actual = model.score_candidate_matrix(**batch)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_bst_user_id_only_zero_history_rows_use_empty_history_token():
    model = _make_model()
    batch = _batch()
    batch["history_mask"] = torch.zeros_like(batch["history_mask"])

    scores = model.score_candidate_matrix(**batch)
    scores.sum().backward()

    assert torch.isfinite(scores).all()
    assert model.empty_history_token.grad is not None
    assert torch.isfinite(model.empty_history_token.grad).all()
    assert model.empty_history_token.grad.abs().sum() > 0


def test_bst_user_id_only_zero_history_matrix_scorer_matches_repeated_forward_path():
    model = _make_model().eval()
    batch = _mixed_zero_history_batch()

    with torch.no_grad():
        expected = _expected_matrix_scores(model, batch)
        actual = model.score_candidate_matrix(**batch)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_bst_user_id_only_target_user_token_zero_history_matches_repeated_forward_path():
    model = _make_model(prepend_target_user_token=True).eval()
    batch = _mixed_zero_history_batch()

    with torch.no_grad():
        expected = _expected_matrix_scores(model, batch)
        actual = model.score_candidate_matrix(**batch)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_bst_user_id_only_target_user_indices_change_scores():
    model = _make_model().eval()
    batch = _batch()

    with torch.no_grad():
        baseline = model.score_candidate_matrix(**batch)
        changed_batch = {key: value.clone() for key, value in batch.items()}
        changed_batch["target_user_indices"] = torch.tensor([5, 2], dtype=torch.long)
        changed = model.score_candidate_matrix(**changed_batch)

    assert not torch.allclose(baseline, changed)


def test_bst_user_id_only_target_user_token_changes_scores_beyond_post_head_feature():
    without_token = _make_model(prepend_target_user_token=False).eval()
    with_token = _make_model(prepend_target_user_token=True).eval()
    with_token.load_state_dict(without_token.state_dict())
    batch = _batch()

    with torch.no_grad():
        without_token_scores = without_token.score_candidate_matrix(**batch)
        with_token_scores = with_token.score_candidate_matrix(**batch)

    assert not torch.allclose(without_token_scores, with_token_scores)
    first_head_layer = with_token.prediction_head.network[0]
    assert first_head_layer.in_features == with_token.transformer_input_dim + with_token.target_user_projection_dim


def test_bst_user_id_only_user_idx_dropout_maps_supported_ids_to_unknown_only_in_training():
    model = BSTUserIdOnlyRanker(
        post_liker_user_table_num_rows=8,
        post_liker_user_embedding_dim=3,
        post_liker_projection_dim=2,
        model_dim=5,
        time_embedding_dim=3,
        num_attention_heads=2,
        num_transformer_layers=1,
        transformer_ff_dim=16,
        dropout_rate=0.0,
        norm_first=False,
        time_delta_bucket_boundaries_hours=DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS,
        prediction_hidden_dims=[8, 4],
        post_liker_pooling_tau_hours=10.0,
        target_user_projection_dim=2,
        post_liker_user_dropout_rate=1.0,
        target_user_dropout_rate=1.0,
        prepend_target_user_token=False,
    )
    user_indices = torch.tensor([[0, 1, 2, 7]], dtype=torch.long)

    model.train()
    dropped = model._apply_user_idx_unk_dropout(user_indices, model.post_liker_user_dropout_rate)
    model.eval()
    unchanged = model._apply_user_idx_unk_dropout(user_indices, model.post_liker_user_dropout_rate)

    assert dropped.tolist() == [[0, 1, 1, 1]]
    assert unchanged.tolist() == [[0, 1, 2, 7]]


def test_bst_user_id_only_has_no_content_or_author_parameters():
    model = _make_model()

    param_names = [name for name, _param in model.named_parameters()]

    assert not any("content" in name for name in param_names)
    assert not any("author" in name for name in param_names)


def test_bst_user_id_only_listwise_loss_trains_user_embedding_and_target_projection():
    model = _make_model()
    batch = {
        **_batch(),
        "label_matrix": torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
    }

    loss, scores, labels = _compute_bst_user_id_only_listwise_loss_and_preds(model, batch, "cpu")
    loss.backward()

    assert torch.isfinite(loss)
    assert scores.shape == labels.shape == (2, 2)
    assert model.post_liker_user_embedding.weight.grad is not None
    assert model.post_liker_user_embedding.weight.grad.abs().sum() > 0
    assert model.target_user_projection.weight.grad is not None
    assert model.target_user_projection.weight.grad.abs().sum() > 0


def test_bst_user_id_only_target_user_token_path_trains_token_fusion():
    model = _make_model(prepend_target_user_token=True)
    batch = {
        **_batch(),
        "label_matrix": torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
    }

    loss, _scores, _labels = _compute_bst_user_id_only_listwise_loss_and_preds(model, batch, "cpu")
    loss.backward()

    assert torch.isfinite(loss)
    assert model.target_user_token_fusion.weight.grad is not None
    assert model.target_user_token_fusion.weight.grad.abs().sum() > 0


def test_bst_user_id_only_baseline_metrics_log_on_normal_series_at_iteration_zero():
    tracker = _RecordingTracker()
    train_metrics = {"ndcg@1": 0.5, "recall@1": 0.6}
    val_metrics = {"ndcg@1": 0.4, "recall@1": 0.5}
    val_unseen_metrics = {"ndcg@1": 0.3, "recall@1": 0.4}
    train_baseline_metrics = {"ndcg@1": 0.2, "recall@1": 0.25}
    val_baseline_metrics = {"ndcg@1": 0.15, "recall@1": 0.2}
    val_unseen_baseline_metrics = {"ndcg@1": 0.1, "recall@1": 0.12}

    _log_bst_user_id_only_epoch_metrics(
        experiment_tracker=tracker,
        iteration=1,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_unseen_metrics=val_unseen_metrics,
        train_baseline_metrics=train_baseline_metrics,
        val_baseline_metrics=val_baseline_metrics,
        val_unseen_baseline_metrics=val_unseen_baseline_metrics,
        calc_baseline_metrics=True,
        metrics_top_ks=[1],
        primary_metric_name="val_unseen_ndcg@1",
    )

    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train NDCG@1")] == [0, 1]
    assert [call["value"] for call in _scalar_calls_by_series(tracker.calls, "Train NDCG@1")] == [0.2, 0.5]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation NDCG@1")] == [0, 1]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Unseen Users NDCG@1")] == [0, 1]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train Recall@1")] == [0, 1]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Recall@1")] == [0, 1]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Unseen Users Recall@1")] == [0, 1]
    for old_baseline_series in (
        "Train Baseline NDCG@1",
        "Validation Baseline NDCG@1",
        "Validation Unseen Users Baseline NDCG@1",
        "Train Baseline Recall@1",
        "Validation Baseline Recall@1",
        "Validation Unseen Users Baseline Recall@1",
    ):
        assert _scalar_calls_by_series(tracker.calls, old_baseline_series) == []


def test_bst_user_id_only_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="post_liker_user_embedding_dim"):
        BSTUserIdOnlyRanker(
            post_liker_user_table_num_rows=8,
            post_liker_user_embedding_dim=0,
            post_liker_projection_dim=2,
            model_dim=5,
            time_embedding_dim=3,
            num_attention_heads=2,
            num_transformer_layers=1,
            transformer_ff_dim=16,
            dropout_rate=0.0,
            norm_first=False,
            time_delta_bucket_boundaries_hours=DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS,
            prediction_hidden_dims=[8, 4],
            post_liker_pooling_tau_hours=10.0,
            target_user_projection_dim=2,
            post_liker_user_dropout_rate=0.0,
            target_user_dropout_rate=0.0,
            prepend_target_user_token=False,
        )


def test_bst_user_id_only_rejects_invalid_user_idx_dropout_rates():
    with pytest.raises(ValueError, match="post_liker_user_dropout_rate"):
        BSTUserIdOnlyRanker(
            post_liker_user_table_num_rows=8,
            post_liker_user_embedding_dim=3,
            post_liker_projection_dim=2,
            model_dim=5,
            time_embedding_dim=3,
            num_attention_heads=2,
            num_transformer_layers=1,
            transformer_ff_dim=16,
            dropout_rate=0.0,
            norm_first=False,
            time_delta_bucket_boundaries_hours=DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS,
            prediction_hidden_dims=[8, 4],
            post_liker_pooling_tau_hours=10.0,
            target_user_projection_dim=2,
            post_liker_user_dropout_rate=-0.1,
            target_user_dropout_rate=0.0,
            prepend_target_user_token=False,
        )
    with pytest.raises(ValueError, match="target_user_dropout_rate"):
        BSTUserIdOnlyRanker(
            post_liker_user_table_num_rows=8,
            post_liker_user_embedding_dim=3,
            post_liker_projection_dim=2,
            model_dim=5,
            time_embedding_dim=3,
            num_attention_heads=2,
            num_transformer_layers=1,
            transformer_ff_dim=16,
            dropout_rate=0.0,
            norm_first=False,
            time_delta_bucket_boundaries_hours=DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS,
            prediction_hidden_dims=[8, 4],
            post_liker_pooling_tau_hours=10.0,
            target_user_projection_dim=2,
            post_liker_user_dropout_rate=0.0,
            target_user_dropout_rate=1.1,
            prepend_target_user_token=False,
        )
