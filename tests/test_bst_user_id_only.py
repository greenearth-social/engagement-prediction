"""Tests for the experimental BST user-ID-only ranker."""
import importlib

import pytest
import torch


stage_train_bst_user_id_only = importlib.import_module("utils.03_train.stage_train_bst_user_id_only")
BSTUserIdOnlyRanker = stage_train_bst_user_id_only.BSTUserIdOnlyRanker
_compute_bst_user_id_only_listwise_loss_and_preds = (
    stage_train_bst_user_id_only._compute_bst_user_id_only_listwise_loss_and_preds
)

DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS = [1.0, 3.0, 6.0, 12.0, 24.0]


def _make_model(*, dropout_rate: float = 0.0) -> BSTUserIdOnlyRanker:
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


def test_bst_user_id_only_target_user_indices_change_scores():
    model = _make_model().eval()
    batch = _batch()

    with torch.no_grad():
        baseline = model.score_candidate_matrix(**batch)
        changed_batch = {key: value.clone() for key, value in batch.items()}
        changed_batch["target_user_indices"] = torch.tensor([5, 2], dtype=torch.long)
        changed = model.score_candidate_matrix(**changed_batch)

    assert not torch.allclose(baseline, changed)


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
        )
