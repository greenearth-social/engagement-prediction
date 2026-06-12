"""Tests for the BST heavy ranker model components."""
import importlib

import pytest
import torch
import torch.nn as nn


stage_train_bst_ranker = importlib.import_module("utils.03_train.stage_train_bst_ranker")
BSTRanker = stage_train_bst_ranker.BSTRanker
DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS = stage_train_bst_ranker.DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS
LinearPredictionHead = stage_train_bst_ranker.LinearPredictionHead
bucketize_time_deltas_hours = stage_train_bst_ranker.bucketize_time_deltas_hours


def _make_model(
    *,
    dropout_rate: float = 0.0,
    num_attention_heads: int = 2,
    prediction_hidden_dims=(8, 4),
) -> BSTRanker:
    torch.manual_seed(123)
    return BSTRanker(
        post_embedding_dim=4,
        author_table_num_rows=8,
        author_embedding_dim=3,
        model_dim=5,
        time_embedding_dim=3,
        num_attention_heads=num_attention_heads,
        num_transformer_layers=1,
        transformer_ff_dim=16,
        dropout_rate=dropout_rate,
        author_unknown_dropout_rate=0.0,
        norm_first=False,
        time_delta_bucket_boundaries_hours=DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS,
        prediction_hidden_dims=prediction_hidden_dims,
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


def test_bst_ranker_forward_transformer_shape_and_builtin_transformer_encoder():
    model = _make_model()
    model.eval()
    batch = _batch()

    output = model._forward_transformer(**batch)

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


def test_bst_ranker_predict_proba_applies_sigmoid_to_logits():
    model = _make_model()
    model.eval()
    batch = _batch()

    logits = model(**batch)
    probabilities = model.predict_proba(**batch)

    torch.testing.assert_close(probabilities, torch.sigmoid(logits))
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_bst_ranker_rejects_attention_head_mismatch():
    with pytest.raises(ValueError, match="divisible"):
        BSTRanker(
            post_embedding_dim=4,
            author_table_num_rows=8,
            author_embedding_dim=3,
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
        )


def test_bucketize_time_deltas_hours_reserves_zero_and_clips_tail():
    deltas = torch.tensor([-2.0, 0.0, 0.5, 1.0, 1.1, 3.0, 2160.0, 2161.0])

    bucket_ids = bucketize_time_deltas_hours(deltas)

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

    assert bucketize_time_deltas_hours(candidate_deltas).tolist() == [[0], [0]]
    assert output.shape == (2,)


def test_bst_ranker_gradients_flow_through_post_time_transformer_and_head_parameters():
    model = _make_model()
    batch = _batch()

    output = model(**batch)
    loss = output.square().sum()
    loss.backward()

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


def test_bst_ranker_supports_direct_linear_prediction_head():
    model = _make_model(prediction_hidden_dims=())
    model.eval()
    batch = _batch()

    output = model(**batch)

    linear_layers = [m for m in model.prediction_head.modules() if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 1
    assert output.shape == (2,)


def test_bst_ranker_rejects_invalid_prediction_hidden_dims():
    with pytest.raises(ValueError, match="hidden_dims"):
        _make_model(prediction_hidden_dims=[0])


def test_bst_ranker_rejects_invalid_prediction_head_output_shape():
    model = _make_model()
    model.prediction_head = nn.Linear(8, 2)
    batch = _batch()

    with pytest.raises(RuntimeError, match="prediction_head"):
        model(**batch)
