from __future__ import annotations

import pytest
import torch

from engagement_prediction.models.two_tower import (
    CrossAttentionHistoryEncoder,
    TwoTowerModel,
)


def _make_model(
    *,
    post_embedding_dim: int = 8,
    output_embedding_dim: int = 5,
    max_history_len: int = 4,
    similarity_temperature: float = 0.25,
) -> TwoTowerModel:
    return TwoTowerModel(
        post_embedding_dim=post_embedding_dim,
        author_table_num_rows=8,
        author_embedding_dim=3,
        content_projection_dim=6,
        author_projection_dim=2,
        user_hidden_dim=7,
        post_hidden_dim=9,
        output_embedding_dim=output_embedding_dim,
        max_history_len=max_history_len,
        dropout_rate=0.0,
        author_unknown_dropout_rate=0.0,
        similarity_temperature=similarity_temperature,
    )


def _inputs(
    *,
    users: int = 3,
    history_len: int = 4,
    candidates: int = 6,
    post_embedding_dim: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(users, history_len, post_embedding_dim),
        torch.ones(users, history_len, dtype=torch.bool),
        torch.randint(1, 8, (users, history_len), dtype=torch.long),
        torch.randn(candidates, post_embedding_dim),
        torch.randint(1, 8, (candidates,), dtype=torch.long),
    )


def test_two_tower_uses_only_cross_attention_and_shares_feature_encoder():
    model = _make_model()

    assert isinstance(
        model.user_tower.history_encoder,
        CrossAttentionHistoryEncoder,
    )
    assert not hasattr(model.user_tower.history_encoder, "transformer_layers")
    assert (
        model.user_tower.post_feature_encoder
        is model.post_tower.post_feature_encoder
        is model.post_feature_encoder
    )


def test_two_tower_encoders_return_configured_unit_vectors():
    model = _make_model(output_embedding_dim=5).eval()
    history, mask, history_authors, posts, post_authors = _inputs()

    with torch.no_grad():
        users = model.encode_user(history, mask, history_authors)
        candidates = model.encode_post(posts, post_authors)

    assert users.shape == (3, 5)
    assert candidates.shape == (6, 5)
    assert torch.allclose(users.norm(dim=-1), torch.ones(3), atol=1.0e-6)
    assert torch.allclose(candidates.norm(dim=-1), torch.ones(6), atol=1.0e-6)


def test_two_tower_forward_matches_independent_tower_matrix_scoring():
    model = _make_model(similarity_temperature=0.4).eval()
    history, mask, history_authors, posts, post_authors = _inputs()

    with torch.no_grad():
        scores = model(
            history,
            mask,
            posts,
            history_authors,
            post_authors,
        )
        expected = (
            model.encode_user(history, mask, history_authors)
            @ model.encode_post(posts, post_authors).T
        ) / 0.4

    assert scores.shape == (3, 6)
    assert torch.equal(scores, expected)


def test_two_tower_masked_padding_does_not_change_user_vector():
    torch.manual_seed(3)
    model = _make_model().eval()
    history, mask, history_authors, _, _ = _inputs(users=2)
    mask[:, 2:] = False

    changed_history = history.clone()
    changed_history[:, 2:] = torch.randn_like(changed_history[:, 2:]) * 1000.0
    changed_authors = history_authors.clone()
    changed_authors[:, 2:] = 7

    with torch.no_grad():
        original = model.encode_user(history, mask, history_authors)
        changed = model.encode_user(changed_history, mask, changed_authors)

    assert torch.equal(original, changed)


def test_two_tower_history_order_changes_user_vector():
    torch.manual_seed(4)
    model = _make_model().eval()
    history, mask, history_authors, _, _ = _inputs(users=1)

    reordered_history = history.flip(dims=(1,))
    reordered_authors = history_authors.flip(dims=(1,))
    with torch.no_grad():
        original = model.encode_user(history, mask, history_authors)
        reordered = model.encode_user(
            reordered_history,
            mask,
            reordered_authors,
        )

    assert not torch.allclose(original, reordered)


def test_two_tower_all_masked_history_uses_learned_token_with_gradients():
    model = _make_model()
    history = torch.zeros(2, 4, 8)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    history_authors = torch.zeros(2, 4, dtype=torch.long)

    user_vectors = model.encode_user(history, mask, history_authors)
    user_vectors[:, 0].sum().backward()

    assert user_vectors.shape == (2, 5)
    assert torch.isfinite(user_vectors).all()
    assert torch.allclose(user_vectors.norm(dim=-1), torch.ones(2), atol=1.0e-6)
    empty_token_grad = model.user_tower.history_encoder.empty_history_token.grad
    assert empty_token_grad is not None
    assert torch.isfinite(empty_token_grad).all()
    assert torch.count_nonzero(empty_token_grad) > 0


def test_two_tower_rejects_literal_zero_width_history():
    model = _make_model().eval()

    with pytest.raises(ValueError, match="positive history width"):
        model.encode_user(
            torch.empty(2, 0, 8),
            torch.empty(2, 0, dtype=torch.bool),
            torch.empty(2, 0, dtype=torch.long),
        )


def test_two_tower_backward_reaches_both_towers_and_shared_features():
    model = _make_model()
    history, mask, history_authors, posts, post_authors = _inputs()
    mask[0] = False

    scores = model(history, mask, posts, history_authors, post_authors)
    scores.square().mean().backward()

    parameter_groups = (
        model.user_tower.history_encoder.parameters(),
        model.post_tower.projection.parameters(),
        model.post_feature_encoder.parameters(),
    )
    for parameters in parameter_groups:
        gradients = [parameter.grad for parameter in parameters]
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("all_masked", [False, True])
def test_two_tower_serving_towers_script_save_and_reload_with_exact_parity(
    tmp_path,
    all_masked: bool,
):
    torch.manual_seed(8)
    model = _make_model().eval()
    history, mask, history_authors, posts, post_authors = _inputs()
    if all_masked:
        mask[:] = False
        history.zero_()
        history_authors.zero_()

    scripted_user = torch.jit.script(model.user_tower)
    scripted_post = torch.jit.script(model.post_tower)
    user_path = tmp_path / "user_tower.pt"
    post_path = tmp_path / "post_tower.pt"
    scripted_user.save(str(user_path))
    scripted_post.save(str(post_path))
    loaded_user = torch.jit.load(str(user_path))
    loaded_post = torch.jit.load(str(post_path))

    with torch.no_grad():
        eager_users = model.encode_user(history, mask, history_authors)
        eager_posts = model.encode_post(posts, post_authors)
        loaded_users = loaded_user(history, mask, history_authors)
        loaded_posts = loaded_post(posts, post_authors)

    assert torch.equal(loaded_users, eager_users)
    assert torch.equal(loaded_posts, eager_posts)
    assert torch.isfinite(loaded_users).all()
    assert torch.isfinite(loaded_posts).all()


@pytest.mark.parametrize("temperature", [0.0, -0.1])
def test_two_tower_requires_positive_similarity_temperature(temperature: float):
    with pytest.raises(ValueError, match="similarity_temperature"):
        _make_model(similarity_temperature=temperature)

