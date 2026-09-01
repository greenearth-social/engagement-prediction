import pytest
import torch

from engagement_prediction.models.common import LinearPredictionHead, ProjectedPostFeatureEncoder


def _encoder(*, use_popularity_feature: bool, author_unknown_dropout_rate: float):
    return ProjectedPostFeatureEncoder(
        post_embedding_dim=4,
        author_table_num_rows=6,
        author_embedding_dim=3,
        content_projection_dim=5,
        author_projection_dim=2,
        output_dim=7,
        author_unknown_dropout_rate=author_unknown_dropout_rate,
        use_popularity_feature=use_popularity_feature,
        popularity_projection_dim=2 if use_popularity_feature else 0,
        popularity_log_mean=1.0,
        popularity_log_std=2.0,
        use_post_liker_feature=False,
        post_liker_user_embedding_dim=3,
        post_liker_projection_dim=2,
    )


def test_projected_post_feature_encoder_preserves_zero_padding_row():
    encoder = _encoder(use_popularity_feature=False, author_unknown_dropout_rate=0.0)

    assert torch.equal(encoder.author_embedding.weight[0], torch.zeros(3))


def test_projected_post_feature_encoder_requires_popularity_when_enabled():
    encoder = _encoder(use_popularity_feature=True, author_unknown_dropout_rate=0.0)

    with pytest.raises(ValueError, match="prior_cumulative_likes"):
        encoder(torch.ones((2, 4)), torch.tensor([2, 3]))


def test_projected_post_feature_encoder_is_torchscriptable():
    encoder = _encoder(use_popularity_feature=True, author_unknown_dropout_rate=0.0).eval()
    embeddings = torch.ones((2, 4))
    authors = torch.tensor([2, 3])
    popularity = torch.tensor([1.0, 10.0])

    scripted = torch.jit.script(encoder)

    torch.testing.assert_close(
        scripted(embeddings, authors, popularity),
        encoder(embeddings, authors, popularity),
    )


def test_linear_prediction_head_supports_hidden_and_direct_shapes():
    inputs = torch.ones((3, 7))

    assert LinearPredictionHead(7, [4, 2], 0.0)(inputs).shape == (3,)
    assert LinearPredictionHead(7, [], 0.0)(inputs).shape == (3,)
