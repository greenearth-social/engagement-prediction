"""Cross-attention two-tower model for engagement retrieval."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from engagement_prediction.models.common import ProjectedPostFeatureEncoder


class CrossAttentionHistoryEncoder(nn.Module):
    """Pool a newest-first post history with one learned attention query.

    The encoder deliberately has no history self-attention stack. It combines a
    content-only masked mean with learned-query attention over position-aware
    history items, keeping computation linear in the configured history length.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_history_len: int,
        dropout_rate: float,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if max_history_len <= 0:
            raise ValueError("max_history_len must be positive")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be in [0, 1]")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.max_history_len = int(max_history_len)
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout_rate)),
        )
        self.positional_embedding = nn.Embedding(
            self.max_history_len,
            self.hidden_dim,
        )
        self.attention_query = nn.Parameter(
            torch.randn(1, 1, self.hidden_dim) * 0.02
        )
        self.empty_history_token = nn.Parameter(
            torch.randn(self.hidden_dim) * 0.02
        )
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool one fixed-width, newest-first history per batch row."""

        if history_features.dim() != 3:
            raise ValueError("history_features must have shape [B, H, D]")
        batch_size = int(history_features.size(0))
        history_len = int(history_features.size(1))
        feature_dim = int(history_features.size(2))
        if feature_dim != self.input_dim:
            raise ValueError("history_features has the wrong feature dimension")
        if history_len == 0:
            raise ValueError("history_features must have a positive history width")
        if history_len > self.max_history_len:
            raise ValueError("history width exceeds max_history_len")
        if history_mask.shape != (batch_size, history_len):
            raise ValueError("history_mask must have shape [B, H]")

        history_mask = history_mask.to(
            device=history_features.device,
            dtype=torch.bool,
        )
        projected = self.input_projection(history_features)

        # Keep empty histories modelable without synchronizing CUDA with Python.
        # Stage 7 and serving use a positive fixed width, so an all-masked row can
        # safely borrow position zero for this learned token.
        inject_empty = ~history_mask.any(dim=1)
        inject_weight = inject_empty.to(dtype=projected.dtype).reshape(
            batch_size,
            1,
            1,
        )
        empty_token = self.empty_history_token.reshape(1, 1, self.hidden_dim)
        projected = projected.clone()
        projected[:, 0:1, :] = (
            projected[:, 0:1, :] * (1.0 - inject_weight)
            + empty_token.expand(batch_size, 1, self.hidden_dim) * inject_weight
        )
        history_mask = history_mask.clone()
        history_mask[:, 0] = history_mask[:, 0] | inject_empty

        mask_float = history_mask.unsqueeze(-1).to(dtype=projected.dtype)
        mean_pooled = (projected * mask_float).sum(dim=1) / mask_float.sum(
            dim=1
        ).clamp(min=1.0)

        # Histories are newest-first. Index relative to max_history_len rather
        # than the observed width so a given recency position has the same
        # embedding in every batch (including short or truncated histories).
        positions = torch.arange(history_len, device=history_features.device)
        positions = (self.max_history_len - 1) - positions
        positioned = projected + self.positional_embedding(positions).unsqueeze(0)

        query = self.attention_query.expand(batch_size, 1, self.hidden_dim)
        attention_scores = torch.bmm(query, positioned.transpose(1, 2))
        ignore_mask = ~history_mask
        scores = attention_scores.masked_fill(ignore_mask.unsqueeze(1), -1.0e9)
        max_scores = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - max_scores).masked_fill(
            ignore_mask.unsqueeze(1),
            0.0,
        )
        attention_weights = exp_scores / exp_scores.sum(
            dim=-1,
            keepdim=True,
        ).clamp(min=1.0)
        attention_pooled = torch.bmm(attention_weights, positioned).squeeze(1)

        return self.output_projection(
            torch.cat([attention_pooled, mean_pooled], dim=-1)
        )


class TwoTowerUserTower(nn.Module):
    """Fuse history content and authors, pool it, and return unit vectors."""

    def __init__(
        self,
        post_feature_encoder: ProjectedPostFeatureEncoder,
        history_encoder: CrossAttentionHistoryEncoder,
    ):
        super().__init__()
        self.post_feature_encoder = post_feature_encoder
        self.history_encoder = history_encoder

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_author_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Encode newest-first histories into unit-length retrieval vectors."""

        history_features = self.post_feature_encoder(
            history_embeddings,
            history_author_indices,
        )
        history_mask = history_mask.to(
            device=history_embeddings.device,
            dtype=torch.bool,
        )
        history_features = history_features.masked_fill(
            ~history_mask.unsqueeze(-1),
            0.0,
        )
        return F.normalize(
            self.history_encoder(history_features, history_mask),
            p=2.0,
            dim=-1,
            eps=1.0e-12,
        )


class TwoTowerPostTower(nn.Module):
    """Fuse post content and author features and project to unit vectors."""

    def __init__(
        self,
        post_feature_encoder: ProjectedPostFeatureEncoder,
        feature_dim: int,
        hidden_dim: int,
        output_embedding_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if output_embedding_dim <= 0:
            raise ValueError("output_embedding_dim must be positive")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be in [0, 1]")
        self.post_feature_encoder = post_feature_encoder
        self.projection = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(int(hidden_dim), int(output_embedding_dim)),
        )
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        post_embeddings: torch.Tensor,
        post_author_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Encode independent candidate posts into unit-length vectors."""

        post_features = self.post_feature_encoder(
            post_embeddings,
            post_author_indices,
        )
        return F.normalize(
            self.projection(post_features),
            p=2.0,
            dim=-1,
            eps=1.0e-12,
        )


class TwoTowerModel(nn.Module):
    """Encode users and posts independently in one retrieval embedding space."""

    def __init__(
        self,
        post_embedding_dim: int,
        author_table_num_rows: int,
        author_embedding_dim: int,
        content_projection_dim: int,
        author_projection_dim: int,
        user_hidden_dim: int,
        post_hidden_dim: int,
        output_embedding_dim: int,
        max_history_len: int,
        dropout_rate: float,
        author_unknown_dropout_rate: float,
        similarity_temperature: float,
    ):
        super().__init__()
        if output_embedding_dim <= 0:
            raise ValueError("output_embedding_dim must be positive")
        if similarity_temperature <= 0.0:
            raise ValueError("similarity_temperature must be positive")

        self.post_embedding_dim = int(post_embedding_dim)
        self.output_embedding_dim = int(output_embedding_dim)
        self.similarity_temperature = float(similarity_temperature)

        # Both exported towers must use exactly the same learned content/author
        # transform. PyTorch deduplicates the shared parameters during training;
        # each separately scripted serving artifact receives the same weights.
        post_feature_encoder = ProjectedPostFeatureEncoder(
            post_embedding_dim=post_embedding_dim,
            author_table_num_rows=author_table_num_rows,
            author_embedding_dim=author_embedding_dim,
            content_projection_dim=content_projection_dim,
            author_projection_dim=author_projection_dim,
            output_dim=content_projection_dim,
            author_unknown_dropout_rate=author_unknown_dropout_rate,
            use_popularity_feature=False,
            popularity_projection_dim=0,
            popularity_log_mean=0.0,
            popularity_log_std=1.0,
        )
        self.user_tower = TwoTowerUserTower(
            post_feature_encoder=post_feature_encoder,
            history_encoder=CrossAttentionHistoryEncoder(
                input_dim=content_projection_dim,
                hidden_dim=user_hidden_dim,
                output_dim=output_embedding_dim,
                max_history_len=max_history_len,
                dropout_rate=dropout_rate,
            ),
        )
        self.post_tower = TwoTowerPostTower(
            post_feature_encoder=post_feature_encoder,
            feature_dim=content_projection_dim,
            hidden_dim=post_hidden_dim,
            output_embedding_dim=output_embedding_dim,
            dropout_rate=dropout_rate,
        )

    @property
    def post_feature_encoder(self) -> ProjectedPostFeatureEncoder:
        """Return the feature encoder shared by the user and post towers."""

        return self.user_tower.post_feature_encoder

    def encode_user(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_author_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run the serving-compatible user tower."""

        return self.user_tower(
            history_embeddings,
            history_mask,
            history_author_indices,
        )

    def encode_post(
        self,
        post_embeddings: torch.Tensor,
        post_author_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run the serving-compatible post tower."""

        return self.post_tower(post_embeddings, post_author_indices)

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        post_author_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return temperature-scaled scores for every user/post combination."""

        user_vectors = self.encode_user(
            history_embeddings,
            history_mask,
            history_author_indices,
        )
        post_vectors = self.encode_post(post_embeddings, post_author_indices)
        return torch.matmul(user_vectors, post_vectors.transpose(0, 1)) / self.similarity_temperature
