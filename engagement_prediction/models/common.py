"""Neural-network components shared by engagement-ranking models."""

from __future__ import annotations

from typing import Final, List, Optional

import torch
import torch.nn as nn

from engagement_prediction.data.author_indices import AUTHOR_PAD_IDX, AUTHOR_UNK_IDX


class ProjectedPostFeatureEncoder(nn.Module):
    """Map heterogeneous post features into one model-width representation.

    Content, author identity, optional as-of popularity, and optional pooled
    post-liker identity are normalized in separate branches before fusion.
    Keeping the branches separate prevents the much larger pretrained content
    vector scale from dominating learned categorical and scalar features.
    """

    __constants__ = ["use_popularity_feature", "use_post_liker_feature"]

    def __init__(
        self,
        post_embedding_dim: int,
        author_table_num_rows: int,
        author_embedding_dim: int,
        content_projection_dim: int,
        author_projection_dim: int,
        output_dim: int,
        author_unknown_dropout_rate: float,
        use_popularity_feature: bool,
        popularity_projection_dim: int,
        popularity_log_mean: float,
        popularity_log_std: float,
        use_post_liker_feature: bool,
        post_liker_user_embedding_dim: int,
        post_liker_projection_dim: int,
    ):
        super().__init__()
        if post_embedding_dim <= 0:
            raise ValueError("post_embedding_dim must be positive")
        if author_table_num_rows < 2:
            raise ValueError("author_table_num_rows must be at least 2")
        if author_embedding_dim <= 0:
            raise ValueError("author_embedding_dim must be positive")
        if content_projection_dim <= 0:
            raise ValueError("content_projection_dim must be positive")
        if author_projection_dim <= 0:
            raise ValueError("author_projection_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if not 0.0 <= author_unknown_dropout_rate <= 1.0:
            raise ValueError("author_unknown_dropout_rate must be in [0, 1]")
        if use_popularity_feature and popularity_projection_dim <= 0:
            raise ValueError("popularity_projection_dim must be positive when popularity features are enabled")
        if use_popularity_feature and popularity_log_std <= 0.0:
            raise ValueError("popularity_log_std must be positive when popularity features are enabled")
        if use_post_liker_feature and post_liker_user_embedding_dim <= 0:
            raise ValueError(
                "post_liker_user_embedding_dim must be positive when post-liker features are enabled"
            )
        if use_post_liker_feature and post_liker_projection_dim <= 0:
            raise ValueError(
                "post_liker_projection_dim must be positive when post-liker features are enabled"
            )

        self.post_embedding_dim = int(post_embedding_dim)
        self.content_projection_dim = int(content_projection_dim)
        self.author_projection_dim = int(author_projection_dim)
        self.output_dim = int(output_dim)
        self.author_unk_idx = int(AUTHOR_UNK_IDX)
        self.author_unknown_dropout_rate = float(author_unknown_dropout_rate)
        self.use_popularity_feature: Final[bool] = bool(use_popularity_feature)
        self.popularity_projection_dim = int(popularity_projection_dim) if self.use_popularity_feature else 0
        self.popularity_log_mean = float(popularity_log_mean)
        self.popularity_log_std = float(popularity_log_std)
        self.use_post_liker_feature: Final[bool] = bool(use_post_liker_feature)
        self.post_liker_user_embedding_dim = (
            int(post_liker_user_embedding_dim) if self.use_post_liker_feature else 0
        )
        self.post_liker_projection_dim = (
            int(post_liker_projection_dim) if self.use_post_liker_feature else 0
        )
        self.author_embedding = nn.Embedding(
            num_embeddings=int(author_table_num_rows),
            embedding_dim=int(author_embedding_dim),
            padding_idx=AUTHOR_PAD_IDX,
        )
        nn.init.xavier_uniform_(self.author_embedding.weight)
        with torch.no_grad():
            self.author_embedding.weight[AUTHOR_PAD_IDX].zero_()

        self.content_projection = nn.Linear(
            int(post_embedding_dim),
            self.content_projection_dim,
        )
        self.author_projection = nn.Linear(
            int(author_embedding_dim),
            self.author_projection_dim,
        )
        self.projection_activation = nn.GELU()
        self.content_projection_norm = nn.LayerNorm(self.content_projection_dim)
        self.author_projection_norm = nn.LayerNorm(self.author_projection_dim)
        if self.use_popularity_feature:
            self.popularity_projection = nn.Linear(
                1,
                self.popularity_projection_dim,
            )
            self.popularity_projection_norm = nn.LayerNorm(self.popularity_projection_dim)
        if self.use_post_liker_feature:
            self.post_liker_projection = nn.Linear(
                self.post_liker_user_embedding_dim,
                self.post_liker_projection_dim,
            )
            self.post_liker_projection_norm = nn.LayerNorm(
                self.post_liker_projection_dim
            )
        self.fusion_layer = nn.Linear(
            self.content_projection_dim
            + self.author_projection_dim
            + self.popularity_projection_dim
            + self.post_liker_projection_dim,
            self.output_dim,
        )
        layers = [self.content_projection, self.author_projection, self.fusion_layer]
        if self.use_popularity_feature:
            layers.append(self.popularity_projection)
        if self.use_post_liker_feature:
            layers.append(self.post_liker_projection)
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        post_embeddings: torch.Tensor,
        author_indices: torch.Tensor,
        prior_cumulative_likes: Optional[torch.Tensor] = None,
        post_liker_vectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if post_embeddings.size(-1) != self.post_embedding_dim:
            raise ValueError(
                f"post_embeddings last dimension ({post_embeddings.size(-1)}) must match post_embedding_dim ({self.post_embedding_dim})"
            )
        if post_embeddings.shape[:-1] != author_indices.shape:
            raise ValueError("author_indices shape must match post_embeddings leading dimensions")
        if self.use_popularity_feature:
            if prior_cumulative_likes is None:
                raise ValueError("prior_cumulative_likes is required when popularity features are enabled")
            if post_embeddings.shape[:-1] != prior_cumulative_likes.shape:
                raise ValueError("prior_cumulative_likes shape must match post_embeddings leading dimensions")
        if self.use_post_liker_feature:
            if post_liker_vectors is None:
                raise ValueError(
                    "post_liker_vectors is required when post-liker features are enabled"
                )
            if post_liker_vectors.shape[:-1] != post_embeddings.shape[:-1]:
                raise ValueError(
                    "post_liker_vectors leading dimensions must match post_embeddings"
                )
            if post_liker_vectors.size(-1) != self.post_liker_user_embedding_dim:
                raise ValueError("post_liker_vectors has the wrong feature dimension")
        elif post_liker_vectors is not None:
            raise ValueError(
                "post_liker_vectors must be omitted when post-liker features are disabled"
            )

        author_indices = author_indices.to(device=post_embeddings.device, dtype=torch.long)
        if self.training and self.author_unknown_dropout_rate > 0.0:
            eligible = author_indices > self.author_unk_idx
            # Keep dropout entirely on-device. Branching on torch.any(eligible)
            # would synchronize CUDA with Python once for every encoder call.
            # This intentionally draws a mask even for an all-PAD/UNK tensor;
            # outputs are unchanged, although the later RNG stream advances.
            dropout_mask = torch.rand(author_indices.shape, device=author_indices.device) < self.author_unknown_dropout_rate
            author_indices = torch.where(
                eligible & dropout_mask,
                torch.full_like(author_indices, self.author_unk_idx),
                author_indices,
            )

        author_embeddings = self.author_embedding(author_indices)
        content_features = self.content_projection_norm(
            self.projection_activation(self.content_projection(post_embeddings))
        )
        author_features = self.author_projection_norm(
            self.projection_activation(self.author_projection(author_embeddings))
        )
        feature_parts = [content_features, author_features]
        if self.use_popularity_feature:
            # Counts are raw cumulative likes as of the scoring hour.  The
            # training stage fits the log-space mean/std using training rows
            # only, and those constants travel with the exported model.
            popularity_counts_input = torch.jit._unwrap_optional(prior_cumulative_likes)
            popularity_counts = popularity_counts_input.to(device=post_embeddings.device, dtype=post_embeddings.dtype)
            popularity_log = torch.log1p(torch.clamp(popularity_counts, min=0.0))
            popularity_scaled = (popularity_log - self.popularity_log_mean) / self.popularity_log_std
            popularity_features = self.popularity_projection_norm(
                self.projection_activation(self.popularity_projection(popularity_scaled.unsqueeze(-1)))
            )
            feature_parts.append(popularity_features)
        if self.use_post_liker_feature:
            post_liker_input = torch.jit._unwrap_optional(post_liker_vectors).to(
                device=post_embeddings.device,
                dtype=post_embeddings.dtype,
            )
            post_liker_features = self.post_liker_projection_norm(
                self.projection_activation(
                    self.post_liker_projection(post_liker_input)
                )
            )
            feature_parts.append(post_liker_features)
        fused_inputs = torch.cat(feature_parts, dim=-1)
        return self.fusion_layer(fused_inputs)


class LinearPredictionHead(nn.Module):
    """Apply a configurable feed-forward head and return one logit per input row."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be in [0, 1]")

        hidden_dims = tuple(int(hidden_dim) for hidden_dim in hidden_dims)
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden_dims must contain only positive values")

        layers: list[nn.Module] = []
        previous_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout_rate)),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, encoded_pair: torch.Tensor) -> torch.Tensor:
        return self.network(encoded_pair).squeeze(-1)
