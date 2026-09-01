"""Tests for the BST heavy ranker model components."""

import pytest
import torch
import torch.nn as nn


from engagement_prediction.models.bst_ranker import BSTRanker
from engagement_prediction.models.common import LinearPredictionHead, ProjectedPostFeatureEncoder

DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS = [1.0, 3.0, 6.0, 12.0, 24.0, 72.0, 168.0, 720.0, 2160.0]


def _make_model(
    *,
    dropout_rate: float = 0.0,
    num_attention_heads: int = 2,
    num_transformer_layers: int = 1,
    norm_first: bool = False,
    prediction_hidden_dims=(8, 4),
    use_popularity_feature: bool = False,
    popularity_projection_dim: int | None = None,
    use_post_liker_feature: bool = False,
    post_liker_user_unknown_dropout_rate: float = 0.0,
) -> BSTRanker:
    torch.manual_seed(123)
    return BSTRanker(
        post_embedding_dim=4,
        author_table_num_rows=8,
        author_embedding_dim=3,
        content_projection_dim=6,
        author_projection_dim=4,
        model_dim=5,
        time_embedding_dim=3,
        num_attention_heads=num_attention_heads,
        num_transformer_layers=num_transformer_layers,
        transformer_ff_dim=16,
        dropout_rate=dropout_rate,
        author_unknown_dropout_rate=0.0,
        norm_first=norm_first,
        time_delta_bucket_boundaries_hours=DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS,
        prediction_hidden_dims=prediction_hidden_dims,
        use_popularity_feature=use_popularity_feature,
        popularity_projection_dim=(2 if popularity_projection_dim is None else popularity_projection_dim) if use_popularity_feature else 0,
        popularity_log_mean=1.0,
        popularity_log_std=2.0,
        use_post_liker_feature=use_post_liker_feature,
        post_liker_user_table_num_rows=7,
        post_liker_user_embedding_dim=3,
        post_liker_projection_dim=2,
        post_liker_pooling_tau_hours=24.0,
        post_liker_user_unknown_dropout_rate=post_liker_user_unknown_dropout_rate,
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "history_embeddings": torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5], [9.0, 9.0, 9.0, 9.0]],
                [[0.0, 0.0, 1.0, 0.5], [1.0, 1.0, 0.0, 0.5], [8.0, 8.0, 8.0, 8.0]],
            ],
            dtype=torch.float32,
        ),
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
        "candidate_post_embeddings": torch.tensor(
            [
                [0.25, 0.5, 0.75, 1.0],
                [1.0, 0.75, 0.5, 0.25],
            ],
            dtype=torch.float32,
        ),
        "history_author_indices": torch.tensor(
            [
                [2, 3, 7],
                [4, 6, 7],
            ],
            dtype=torch.long,
        ),
        "candidate_post_author_idx": torch.tensor([5, 6], dtype=torch.long),
    }


def _batch_with_popularity() -> dict[str, torch.Tensor]:
    batch = _batch()
    return {
        **batch,
        "history_prior_cumulative_likes": torch.tensor(
            [
                [1.0, 5.0, 999.0],
                [10.0, 888.0, 777.0],
            ],
            dtype=torch.float32,
        ),
        "candidate_prior_cumulative_likes": torch.tensor([2.0, 20.0], dtype=torch.float32),
    }


def _packed_post_liker_batch() -> dict[str, torch.Tensor]:
    """Create row zero plus three post segments, one of which is empty."""

    return {
        "post_liker_event_user_indices": torch.tensor([2, 3, 1], dtype=torch.long),
        "post_liker_event_age_from_latest_hours": torch.tensor(
            [24.0, 0.0, 0.0], dtype=torch.float32
        ),
        "post_liker_event_offsets": torch.tensor([0, 0, 2, 2, 3], dtype=torch.long),
        "history_post_liker_rows": torch.tensor(
            [[1, 2, 0], [3, 0, 0]], dtype=torch.long
        ),
        "candidate_post_liker_rows": torch.tensor([1, 3], dtype=torch.long),
    }


def _expected_matrix_scores(model: BSTRanker, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    num_users = batch["history_embeddings"].shape[0]
    num_candidates = batch["candidate_post_embeddings"].shape[0]
    kwargs = {
        "history_embeddings": batch["history_embeddings"].repeat_interleave(num_candidates, dim=0),
        "history_mask": batch["history_mask"].repeat_interleave(num_candidates, dim=0),
        "history_time_deltas_hours": batch["history_time_deltas_hours"].repeat_interleave(num_candidates, dim=0),
        "candidate_post_embeddings": batch["candidate_post_embeddings"].repeat(num_users, 1),
        "history_author_indices": batch["history_author_indices"].repeat_interleave(num_candidates, dim=0),
        "candidate_post_author_idx": batch["candidate_post_author_idx"].repeat(num_users),
    }
    if "history_prior_cumulative_likes" in batch:
        kwargs["history_prior_cumulative_likes"] = batch["history_prior_cumulative_likes"].repeat_interleave(num_candidates, dim=0)
        kwargs["candidate_prior_cumulative_likes"] = batch["candidate_prior_cumulative_likes"].repeat(num_users)
    return model(**kwargs).reshape(num_users, num_candidates)


def _mixed_zero_history_batch() -> dict[str, torch.Tensor]:
    batch = {key: value.clone() for key, value in _batch().items()}
    batch["history_mask"][1] = False
    return batch


def _empty_history_batch() -> dict[str, torch.Tensor]:
    batch = _batch()
    num_users = batch["history_embeddings"].shape[0]
    return {
        **batch,
        "history_embeddings": torch.empty((num_users, 0, 4), dtype=torch.float32),
        "history_mask": torch.empty((num_users, 0), dtype=torch.bool),
        "history_time_deltas_hours": torch.empty((num_users, 0), dtype=torch.float32),
        "history_author_indices": torch.empty((num_users, 0), dtype=torch.long),
    }


def test_bst_ranker_forward_transformer_shape_and_builtin_transformer_encoder():
    model = _make_model()
    model.eval()
    batch = _batch()

    output = model._forward_transformer(**batch)

    assert isinstance(model.post_feature_encoder, ProjectedPostFeatureEncoder)
    assert isinstance(model.transformer_encoder, nn.TransformerEncoder)
    assert isinstance(model.prediction_head, LinearPredictionHead)
    assert output.shape == (2, model.transformer_input_dim)
    assert output.dtype == torch.float32


def test_bst_ranker_forward_returns_raw_logits():
    model = _make_model()
    model.eval()
    batch = _batch()

    logits = model(**batch)

    assert logits.shape == (2,)
    assert logits.dtype == torch.float32


def test_bst_ranker_zero_history_rows_use_empty_history_token():
    model = _make_model()
    batch = _mixed_zero_history_batch()

    scores = model.score_candidate_matrix_one_layer(**batch)
    scores.sum().backward()

    assert torch.isfinite(scores).all()
    assert model.empty_history_token.grad is not None
    assert torch.isfinite(model.empty_history_token.grad).all()
    assert model.empty_history_token.grad.abs().sum() > 0


@pytest.mark.parametrize("norm_first", [False, True])
def test_bst_ranker_score_candidate_matrix_one_layer_matches_repeated_path(norm_first):
    model = _make_model(norm_first=norm_first)
    model.eval()
    batch = _batch()

    with torch.inference_mode():
        expected = _expected_matrix_scores(model, batch)
        scores = model.score_candidate_matrix_one_layer(**batch)

    assert scores.shape == (2, 2)
    torch.testing.assert_close(scores, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("norm_first", [False, True])
def test_bst_ranker_zero_history_matrix_scorer_matches_repeated_path(norm_first):
    model = _make_model(norm_first=norm_first)
    model.eval()
    batch = _mixed_zero_history_batch()

    with torch.inference_mode():
        expected = _expected_matrix_scores(model, batch)
        scores = model.score_candidate_matrix_one_layer(**batch)

    torch.testing.assert_close(scores, expected, atol=1e-6, rtol=1e-6)


def test_bst_ranker_score_candidate_matrix_one_layer_supports_training_gradients():
    model = _make_model()
    batch = _batch()

    scores = model.score_candidate_matrix_one_layer(**batch)
    loss = scores.square().sum()
    loss.backward()

    assert scores.shape == (2, 2)
    assert scores.requires_grad
    grad_sum = sum(
        param.grad.abs().sum()
        for param in model.parameters()
        if param.grad is not None
    )
    assert grad_sum > 0


@pytest.mark.parametrize("norm_first", [False, True])
def test_bst_ranker_popularity_matrix_scorer_matches_repeated_path(norm_first):
    model = _make_model(norm_first=norm_first, use_popularity_feature=True)
    model.eval()
    batch = _batch_with_popularity()

    with torch.inference_mode():
        expected = _expected_matrix_scores(model, batch)
        scores = model.score_candidate_matrix_one_layer(**batch)

    assert scores.shape == (2, 2)
    torch.testing.assert_close(scores, expected, atol=1e-6, rtol=1e-6)


def test_bst_ranker_popularity_changes_scores_for_valid_tokens():
    model = _make_model(use_popularity_feature=True)
    model.eval()
    batch = _batch_with_popularity()

    output = model.score_candidate_matrix_one_layer(**batch)
    changed_batch = {key: value.clone() for key, value in batch.items()}
    changed_batch["candidate_prior_cumulative_likes"] = torch.tensor([2000.0, 20.0], dtype=torch.float32)
    changed_output = model.score_candidate_matrix_one_layer(**changed_batch)

    assert not torch.allclose(changed_output, output)


def test_bst_ranker_popularity_requires_tensors_when_enabled():
    model = _make_model(use_popularity_feature=True)
    batch = _batch()

    with pytest.raises(ValueError, match="history_prior_cumulative_likes"):
        model.score_candidate_matrix_one_layer(**batch)


def test_bst_ranker_score_candidate_matrix_one_layer_rejects_multi_layer_model():
    model = _make_model(num_transformer_layers=2)
    model.eval()
    batch = _batch()

    with pytest.raises(RuntimeError, match="exactly one transformer layer"):
        model.score_candidate_matrix_one_layer(**batch)


def test_bst_ranker_rejects_attention_head_mismatch():
    with pytest.raises(ValueError, match="divisible"):
        BSTRanker(
            post_embedding_dim=4,
            author_table_num_rows=8,
            author_embedding_dim=3,
            content_projection_dim=6,
            author_projection_dim=4,
            model_dim=5,
            time_embedding_dim=2,
            num_attention_heads=4,
            num_transformer_layers=1,
            transformer_ff_dim=16,
            dropout_rate=0.0,
            author_unknown_dropout_rate=0.0,
            norm_first=False,
            time_delta_bucket_boundaries_hours=DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS,
            prediction_hidden_dims=(7,),
            use_popularity_feature=False,
            popularity_projection_dim=0,
            popularity_log_mean=0.0,
            popularity_log_std=1.0,
            use_post_liker_feature=False,
            post_liker_user_table_num_rows=2,
            post_liker_user_embedding_dim=3,
            post_liker_projection_dim=2,
            post_liker_pooling_tau_hours=24.0,
            post_liker_user_unknown_dropout_rate=0.0,
        )


def test_bst_ranker_bucketizes_time_deltas_reserving_zero_and_clipping_tail():
    model = _make_model()
    deltas = torch.tensor([-2.0, 0.0, 0.5, 1.0, 1.1, 3.0, 2160.0, 2161.0])

    bucket_ids = model._bucketize_time_deltas_hours(deltas)

    assert bucket_ids.dtype == torch.long
    assert bucket_ids.tolist() == [0, 0, 1, 1, 2, 2, 9, 10]


def test_bst_ranker_masks_padded_history_positions():
    model = _make_model()
    model.eval()
    batch = _batch()

    output = model(**batch)
    changed_batch = {key: value.clone() for key, value in batch.items()}
    changed_batch["history_embeddings"][0, 2] = torch.tensor([1000.0, 1000.0, 1000.0, 1000.0])
    changed_batch["history_embeddings"][1, 1:] = torch.tensor(
        [[2000.0, 2000.0, 2000.0, 2000.0], [3000.0, 3000.0, 3000.0, 3000.0]]
    )
    changed_batch["history_time_deltas_hours"][0, 2] = 100000.0
    changed_batch["history_time_deltas_hours"][1, 1:] = torch.tensor([200000.0, 300000.0])
    changed_batch["history_author_indices"][0, 2] = 2
    changed_batch["history_author_indices"][1, 1:] = torch.tensor([3, 4])

    changed_output = model(**changed_batch)

    torch.testing.assert_close(changed_output, output, atol=1e-6, rtol=1e-6)


def test_bst_ranker_score_candidate_matrix_one_layer_masks_padded_history_positions():
    model = _make_model()
    model.eval()
    batch = _batch()

    output = model.score_candidate_matrix_one_layer(**batch)
    changed_batch = {key: value.clone() for key, value in batch.items()}
    changed_batch["history_embeddings"][0, 2] = torch.tensor([1000.0, 1000.0, 1000.0, 1000.0])
    changed_batch["history_embeddings"][1, 1:] = torch.tensor(
        [[2000.0, 2000.0, 2000.0, 2000.0], [3000.0, 3000.0, 3000.0, 3000.0]]
    )
    changed_batch["history_time_deltas_hours"][0, 2] = 100000.0
    changed_batch["history_time_deltas_hours"][1, 1:] = torch.tensor([200000.0, 300000.0])
    changed_batch["history_author_indices"][0, 2] = 2
    changed_batch["history_author_indices"][1, 1:] = torch.tensor([3, 4])

    changed_output = model.score_candidate_matrix_one_layer(**changed_batch)

    torch.testing.assert_close(changed_output, output, atol=1e-6, rtol=1e-6)


def test_bst_ranker_score_candidate_matrix_one_layer_masks_padded_history_popularity():
    model = _make_model(use_popularity_feature=True)
    model.eval()
    batch = _batch_with_popularity()

    output = model.score_candidate_matrix_one_layer(**batch)
    changed_batch = {key: value.clone() for key, value in batch.items()}
    changed_batch["history_prior_cumulative_likes"][0, 2] = 100000.0
    changed_batch["history_prior_cumulative_likes"][1, 1:] = torch.tensor([200000.0, 300000.0])

    changed_output = model.score_candidate_matrix_one_layer(**changed_batch)

    torch.testing.assert_close(changed_output, output, atol=1e-6, rtol=1e-6)


def test_bst_ranker_supports_candidate_only_sequence_with_zero_delta_bucket():
    model = _make_model()
    model.eval()
    history_time_deltas = torch.empty((2, 0), dtype=torch.float32)
    candidate_deltas = torch.zeros((2, 1), dtype=torch.float32)

    output = model(
        history_embeddings=torch.empty((2, 0, 4), dtype=torch.float32),
        history_mask=torch.empty((2, 0), dtype=torch.bool),
        history_time_deltas_hours=history_time_deltas,
        candidate_post_embeddings=torch.tensor(
            [
                [0.25, 0.5, 0.75, 1.0],
                [1.0, 0.75, 0.5, 0.25],
            ],
            dtype=torch.float32,
        ),
        history_author_indices=torch.empty((2, 0), dtype=torch.long),
        candidate_post_author_idx=torch.tensor([5, 6], dtype=torch.long),
    )

    assert model._bucketize_time_deltas_hours(candidate_deltas).tolist() == [[0], [0]]
    assert output.shape == (2,)


def test_bst_ranker_score_candidate_matrix_supports_zero_length_history():
    model = _make_model()
    model.eval()
    batch = _empty_history_batch()

    with torch.inference_mode():
        expected = _expected_matrix_scores(model, batch)
        scores = model.score_candidate_matrix(
            batch["history_embeddings"],
            batch["history_mask"],
            batch["history_time_deltas_hours"],
            batch["candidate_post_embeddings"],
            batch["history_author_indices"],
            batch["candidate_post_author_idx"],
        )

    assert torch.isfinite(scores).all()
    torch.testing.assert_close(scores, expected, atol=1e-6, rtol=1e-6)


def test_bst_ranker_gradients_flow_through_post_time_transformer_and_head_parameters():
    model = _make_model()
    batch = _batch()

    output = model(**batch)
    loss = output.square().sum()
    loss.backward()

    assert model.post_feature_encoder.content_projection.weight.grad is not None
    assert model.post_feature_encoder.content_projection.weight.grad.abs().sum() > 0
    assert model.post_feature_encoder.author_projection.weight.grad is not None
    assert model.post_feature_encoder.author_projection.weight.grad.abs().sum() > 0
    assert model.post_feature_encoder.fusion_layer.weight.grad is not None
    assert model.post_feature_encoder.fusion_layer.weight.grad.abs().sum() > 0
    assert model.time_delta_embedding.weight.grad is not None
    assert model.time_delta_embedding.weight.grad.abs().sum() > 0
    transformer_grad_sum = sum(
        param.grad.abs().sum()
        for param in model.transformer_encoder.parameters()
        if param.grad is not None
    )
    assert transformer_grad_sum > 0
    prediction_head_grad_sum = sum(
        param.grad.abs().sum()
        for param in model.prediction_head.parameters()
        if param.grad is not None
    )
    assert prediction_head_grad_sum > 0


def test_bst_ranker_gradients_flow_through_popularity_projection():
    model = _make_model(use_popularity_feature=True)
    batch = _batch_with_popularity()

    output = model(**batch)
    loss = output.square().sum()
    loss.backward()

    assert model.post_feature_encoder.popularity_projection.weight.grad is not None
    assert model.post_feature_encoder.popularity_projection.weight.grad.abs().sum() > 0


def test_bst_ranker_supports_direct_linear_prediction_head():
    model = _make_model(prediction_hidden_dims=())
    model.eval()
    batch = _batch()

    output = model(**batch)

    linear_layers = [m for m in model.prediction_head.modules() if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 1
    assert output.shape == (2,)


def test_bst_ranker_torchscript_forward_matches_eager():
    model = _make_model().eval()
    batch = _batch()

    with torch.no_grad():
        eager_output = model(**batch)
        scripted_model = torch.jit.script(model)
        scripted_output = scripted_model(
            batch["history_embeddings"],
            batch["history_mask"],
            batch["history_time_deltas_hours"],
            batch["candidate_post_embeddings"],
            batch["history_author_indices"],
            batch["candidate_post_author_idx"],
        )

    assert scripted_output.shape == eager_output.shape
    assert torch.allclose(scripted_output, eager_output, atol=1e-5)


def test_bst_ranker_torchscript_exports_matrix_scorer():
    model = _make_model().eval()
    batch = _batch()

    with torch.no_grad():
        expected = model.score_candidate_matrix_one_layer(**batch)
        scripted_model = torch.jit.script(model)
        scripted_scores = scripted_model.score_candidate_matrix(
            batch["history_embeddings"],
            batch["history_mask"],
            batch["history_time_deltas_hours"],
            batch["candidate_post_embeddings"],
            batch["history_author_indices"],
            batch["candidate_post_author_idx"],
        )

    assert scripted_scores.shape == expected.shape
    torch.testing.assert_close(scripted_scores, expected, atol=1e-5, rtol=1e-5)


def test_bst_ranker_can_script_differently_shaped_models_in_one_process():
    first_model = _make_model().eval()
    second_model = BSTRanker(
        post_embedding_dim=2,
        author_table_num_rows=6,
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
        use_popularity_feature=True,
        popularity_projection_dim=2,
        popularity_log_mean=1.0,
        popularity_log_std=2.0,
        use_post_liker_feature=False,
        post_liker_user_table_num_rows=2,
        post_liker_user_embedding_dim=3,
        post_liker_projection_dim=2,
        post_liker_pooling_tau_hours=24.0,
        post_liker_user_unknown_dropout_rate=0.0,
    ).eval()

    first_scripted = torch.jit.script(first_model)
    second_scripted = torch.jit.script(second_model)

    assert callable(first_scripted.score_candidate_matrix)
    assert callable(second_scripted.score_candidate_matrix)


def test_bst_ranker_torchscript_save_load_preserves_zero_history_paths(tmp_path):
    model = _make_model().eval()
    batch = _mixed_zero_history_batch()
    model_path = tmp_path / "ranker.pt"

    with torch.inference_mode():
        eager_output = model(**batch)
        eager_scores = model.score_candidate_matrix_one_layer(**batch)
        scripted_model = torch.jit.script(model)
        scripted_model.save(str(model_path))
        loaded_model = torch.jit.load(str(model_path)).eval()
        loaded_output = loaded_model(
            batch["history_embeddings"],
            batch["history_mask"],
            batch["history_time_deltas_hours"],
            batch["candidate_post_embeddings"],
            batch["history_author_indices"],
            batch["candidate_post_author_idx"],
        )
        loaded_scores = loaded_model.score_candidate_matrix(
            batch["history_embeddings"],
            batch["history_mask"],
            batch["history_time_deltas_hours"],
            batch["candidate_post_embeddings"],
            batch["history_author_indices"],
            batch["candidate_post_author_idx"],
        )

    torch.testing.assert_close(loaded_output, eager_output, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(loaded_scores, eager_scores, atol=1e-5, rtol=1e-5)


def test_bst_ranker_torchscript_supports_popularity_features():
    model = _make_model(use_popularity_feature=True).eval()
    batch = _batch_with_popularity()

    with torch.no_grad():
        eager_output = model(**batch)
        eager_scores = model.score_candidate_matrix_one_layer(**batch)
        scripted_model = torch.jit.script(model)
        scripted_output = scripted_model(
            batch["history_embeddings"],
            batch["history_mask"],
            batch["history_time_deltas_hours"],
            batch["candidate_post_embeddings"],
            batch["history_author_indices"],
            batch["candidate_post_author_idx"],
            batch["history_prior_cumulative_likes"],
            batch["candidate_prior_cumulative_likes"],
        )
        scripted_scores = scripted_model.score_candidate_matrix(
            batch["history_embeddings"],
            batch["history_mask"],
            batch["history_time_deltas_hours"],
            batch["candidate_post_embeddings"],
            batch["history_author_indices"],
            batch["candidate_post_author_idx"],
            batch["history_prior_cumulative_likes"],
            batch["candidate_prior_cumulative_likes"],
        )

    torch.testing.assert_close(scripted_output, eager_output, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(scripted_scores, eager_scores, atol=1e-5, rtol=1e-5)


def test_post_liker_pooling_matches_naive_weighted_mean_and_scores():
    model = _make_model(use_post_liker_feature=True).eval()
    batch = _batch()
    packed = _packed_post_liker_batch()

    with torch.inference_mode():
        pooled = model.post_liker_user_pooler(
            packed["post_liker_event_user_indices"],
            packed["post_liker_event_age_from_latest_hours"],
            packed["post_liker_event_offsets"],
        )
        table = model.lookup_post_liker_user_embeddings(torch.tensor([1, 2, 3]))
        expected_first_post = (
            torch.exp(torch.tensor(-1.0)) * table[1] + table[2]
        ) / (torch.exp(torch.tensor(-1.0)) + 1.0)
        torch.testing.assert_close(pooled[0], torch.zeros_like(pooled[0]))
        torch.testing.assert_close(pooled[1], expected_first_post)
        torch.testing.assert_close(pooled[2], torch.zeros_like(pooled[2]))
        torch.testing.assert_close(pooled[3], table[0])

        history_vectors = pooled.index_select(
            0, packed["history_post_liker_rows"].reshape(-1)
        ).reshape(2, 3, 3)
        candidate_vectors = pooled.index_select(
            0, packed["candidate_post_liker_rows"]
        )
        vector_scores = model.score_candidate_matrix(
            **batch,
            history_post_liker_vectors=history_vectors,
            candidate_post_liker_vectors=candidate_vectors,
        )
        event_scores = model.score_candidate_matrix_from_post_liker_events(
            **batch,
            history_prior_cumulative_likes=None,
            candidate_prior_cumulative_likes=None,
            **packed,
        )

    torch.testing.assert_close(event_scores, vector_scores)


def test_post_liker_feature_requires_both_pooled_vector_inputs():
    model = _make_model(use_post_liker_feature=True).eval()
    batch = _batch()
    history_vectors = torch.zeros((2, 3, 3))

    with pytest.raises(RuntimeError, match="history_post_liker_vectors is required"):
        model.score_candidate_matrix(**batch)
    with pytest.raises(RuntimeError, match="candidate_post_liker_vectors is required"):
        model.score_candidate_matrix(
            **batch,
            history_post_liker_vectors=history_vectors,
        )


def test_post_liker_event_scoring_backpropagates_to_user_table_and_scripts():
    model = _make_model(use_post_liker_feature=True)
    batch = _batch()
    packed = _packed_post_liker_batch()

    scores = model.score_candidate_matrix_from_post_liker_events(
        **batch,
        history_prior_cumulative_likes=None,
        candidate_prior_cumulative_likes=None,
        **packed,
    )
    scores.sum().backward()
    assert model.post_liker_user_pooler.user_embedding.weight.grad is not None
    assert model.post_feature_encoder.post_liker_projection.weight.grad is not None

    model.eval()
    scripted = torch.jit.script(model)
    with torch.inference_mode():
        expected = model.score_candidate_matrix_from_post_liker_events(
            **batch,
            history_prior_cumulative_likes=None,
            candidate_prior_cumulative_likes=None,
            **packed,
        )
        actual = scripted.score_candidate_matrix_from_post_liker_events(
            batch["history_embeddings"],
            batch["history_mask"],
            batch["history_time_deltas_hours"],
            batch["candidate_post_embeddings"],
            batch["history_author_indices"],
            batch["candidate_post_author_idx"],
            None,
            None,
            packed["post_liker_event_user_indices"],
            packed["post_liker_event_age_from_latest_hours"],
            packed["post_liker_event_offsets"],
            packed["history_post_liker_rows"],
            packed["candidate_post_liker_rows"],
        )
    torch.testing.assert_close(actual, expected)


def test_post_liker_unknown_dropout_maps_known_users_but_not_natural_unk():
    model = _make_model(
        use_post_liker_feature=True,
        post_liker_user_unknown_dropout_rate=1.0,
    ).train()
    offsets = torch.tensor([0, 2], dtype=torch.long)
    pooled = model.post_liker_user_pooler(
        torch.tensor([1, 2]),
        torch.zeros(2),
        offsets,
    )
    unk = model.lookup_post_liker_user_embeddings(torch.tensor([1]))[0]
    torch.testing.assert_close(pooled[0], unk)


def test_bst_ranker_rejects_invalid_prediction_hidden_dims():
    with pytest.raises(ValueError, match="hidden_dims"):
        _make_model(prediction_hidden_dims=[0])


def test_bst_ranker_rejects_invalid_popularity_projection_dim():
    with pytest.raises(ValueError, match="popularity_projection_dim"):
        _make_model(use_popularity_feature=True, popularity_projection_dim=0)


def test_bst_ranker_rejects_invalid_prediction_head_output_shape():
    model = _make_model()
    model.prediction_head = nn.Linear(8, 2)
    batch = _batch()

    with pytest.raises(RuntimeError, match="prediction_head"):
        model(**batch)
