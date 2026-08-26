"""Behavior Sequence Transformer model for engagement ranking."""

from __future__ import annotations

from typing import Final, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from engagement_prediction.models.common import LinearPredictionHead, ProjectedPostFeatureEncoder


def _validate_time_delta_bucket_boundaries(
    boundaries_hours: Sequence[float],
) -> tuple[float, ...]:
    boundaries = tuple(float(boundary) for boundary in boundaries_hours)
    if len(boundaries) == 0:
        raise ValueError("time delta bucket boundaries must not be empty")
    previous = 0.0
    for boundary in boundaries:
        if boundary <= 0.0:
            raise ValueError("time delta bucket boundaries must be positive")
        if boundary <= previous:
            raise ValueError("time delta bucket boundaries must be strictly increasing")
        previous = boundary
    return boundaries


class BSTRanker(nn.Module):
    """Behavior Sequence Transformer encoder for one user-history/candidate pair."""

    __constants__ = ["use_popularity_feature"]

    def __init__(
        self,
        post_embedding_dim: int,
        author_table_num_rows: int,
        author_embedding_dim: int,
        content_projection_dim: int,
        author_projection_dim: int,
        model_dim: int,
        time_embedding_dim: int,
        num_attention_heads: int,
        num_transformer_layers: int,
        transformer_ff_dim: int,
        dropout_rate: float,
        author_unknown_dropout_rate: float,
        norm_first: bool,
        time_delta_bucket_boundaries_hours: List[float],
        prediction_hidden_dims: List[int],
        use_popularity_feature: bool,
        popularity_projection_dim: int,
        popularity_log_mean: float,
        popularity_log_std: float,
    ):
        super().__init__()
        if time_embedding_dim <= 0:
            raise ValueError("time_embedding_dim must be positive")
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if num_transformer_layers <= 0:
            raise ValueError("num_transformer_layers must be positive")
        if transformer_ff_dim <= 0:
            raise ValueError("transformer_ff_dim must be positive")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be in [0, 1]")
        if use_popularity_feature and popularity_projection_dim <= 0:
            raise ValueError("popularity_projection_dim must be positive when popularity features are enabled")
        if use_popularity_feature and popularity_log_std <= 0.0:
            raise ValueError("popularity_log_std must be positive when popularity features are enabled")

        self.post_embedding_dim = int(post_embedding_dim)
        self.content_projection_dim = int(content_projection_dim)
        self.author_projection_dim = int(author_projection_dim)
        self.model_dim = int(model_dim)
        self.time_embedding_dim = int(time_embedding_dim)
        self.dropout_rate = float(dropout_rate)
        self.use_popularity_feature: Final[bool] = bool(use_popularity_feature)
        self.popularity_projection_dim = int(popularity_projection_dim) if self.use_popularity_feature else 0
        self.popularity_log_mean = float(popularity_log_mean)
        self.popularity_log_std = float(popularity_log_std)
        self.time_delta_bucket_boundaries_hours = _validate_time_delta_bucket_boundaries(
            time_delta_bucket_boundaries_hours
        )
        self.register_buffer(
            "_time_delta_bucket_boundaries_tensor",
            torch.tensor(self.time_delta_bucket_boundaries_hours, dtype=torch.float32),
            persistent=False,
        )
        self.num_time_delta_buckets = len(self.time_delta_bucket_boundaries_hours) + 2
        self.transformer_input_dim = self.model_dim + self.time_embedding_dim
        if self.transformer_input_dim % int(num_attention_heads) != 0:
            raise ValueError("model_dim + time_embedding_dim must be divisible by num_attention_heads")

        self.post_feature_encoder = ProjectedPostFeatureEncoder(
            post_embedding_dim=post_embedding_dim,
            author_table_num_rows=author_table_num_rows,
            author_embedding_dim=author_embedding_dim,
            content_projection_dim=content_projection_dim,
            author_projection_dim=author_projection_dim,
            output_dim=model_dim,
            author_unknown_dropout_rate=author_unknown_dropout_rate,
            use_popularity_feature=use_popularity_feature,
            popularity_projection_dim=popularity_projection_dim,
            popularity_log_mean=popularity_log_mean,
            popularity_log_std=popularity_log_std,
        )
        self.time_delta_embedding = nn.Embedding(
            num_embeddings=self.num_time_delta_buckets,
            embedding_dim=self.time_embedding_dim,
        )
        nn.init.xavier_uniform_(self.time_delta_embedding.weight)
        self.empty_history_token = nn.Parameter(
            torch.randn(self.transformer_input_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_input_dim,
            nhead=int(num_attention_heads),
            dim_feedforward=int(transformer_ff_dim),
            dropout=float(dropout_rate),
            activation="gelu",
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(num_transformer_layers),
            enable_nested_tensor=False,
        )
        self.prediction_head = LinearPredictionHead(
            input_dim=self.transformer_input_dim,
            hidden_dims=prediction_hidden_dims,
            dropout_rate=dropout_rate,
        )

    def _bucketize_time_deltas_hours(self, time_deltas_hours: torch.Tensor) -> torch.Tensor:
        deltas = time_deltas_hours
        if not torch.is_floating_point(deltas):
            deltas = deltas.to(dtype=torch.float32)
        deltas = torch.clamp(deltas, min=0.0)
        boundary_tensor = self._time_delta_bucket_boundaries_tensor.to(
            device=deltas.device,
            dtype=deltas.dtype,
        )
        positive_bucket_ids = torch.bucketize(deltas, boundary_tensor, right=False) + 1
        zero_bucket_ids = torch.zeros_like(positive_bucket_ids)
        return torch.where(deltas <= 0.0, zero_bucket_ids, positive_bucket_ids).to(dtype=torch.long)

    def _inject_empty_history_token(
        self,
        history_input: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Give an empty-history user one learned, unmasked history token."""

        batch_size = int(history_input.size(0))
        max_history_len = int(history_input.size(1))
        token = self.empty_history_token.reshape(1, 1, self.transformer_input_dim)
        if max_history_len == 0:
            return (
                token.expand(batch_size, 1, self.transformer_input_dim),
                torch.ones(
                    (batch_size, 1),
                    device=history_input.device,
                    dtype=torch.bool,
                ),
            )

        has_history = history_mask.any(dim=1)
        # Always express empty-history injection as tensor operations. A Python
        # branch on has_history.all().item() stalls the host until CUDA has
        # finished the preceding model work on every batch. On an all-nonempty
        # batch the token now receives a zero gradient instead of no gradient;
        # model outputs remain identical.
        inject = ~has_history
        inject_float = inject.to(dtype=history_input.dtype).reshape(batch_size, 1, 1)
        history_input = history_input.clone()
        history_input[:, 0:1, :] = (
            history_input[:, 0:1, :] * (1.0 - inject_float)
            + token.expand(batch_size, 1, self.transformer_input_dim) * inject_float
        )
        history_mask = history_mask.clone()
        history_mask[:, 0] = history_mask[:, 0] | inject
        return history_input, history_mask

    def _forward_transformer(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        candidate_post_author_idx: torch.Tensor,
        history_prior_cumulative_likes: Optional[torch.Tensor] = None,
        candidate_prior_cumulative_likes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if history_embeddings.dim() != 3:
            raise ValueError("history_embeddings must have shape [B, H, D]")
        if candidate_post_embeddings.dim() != 2:
            raise ValueError("candidate_post_embeddings must have shape [B, D]")
        batch_size, max_history_len, embed_dim = history_embeddings.shape
        if embed_dim != self.post_embedding_dim:
            raise ValueError(
                f"history_embeddings last dimension ({embed_dim}) must match post_embedding_dim ({self.post_embedding_dim})"
            )
        if candidate_post_embeddings.shape != (batch_size, self.post_embedding_dim):
            raise ValueError("candidate_post_embeddings must have shape [B, post_embedding_dim]")
        if history_mask.shape != (batch_size, max_history_len):
            raise ValueError("history_mask must have shape [B, H]")
        if history_time_deltas_hours.shape != (batch_size, max_history_len):
            raise ValueError("history_time_deltas_hours must have shape [B, H]")
        if history_author_indices.shape != (batch_size, max_history_len):
            raise ValueError("history_author_indices must have shape [B, H]")
        if candidate_post_author_idx.shape != (batch_size,):
            raise ValueError("candidate_post_author_idx must have shape [B]")
        if self.use_popularity_feature:
            if history_prior_cumulative_likes is None:
                raise ValueError("history_prior_cumulative_likes is required when popularity features are enabled")
            if candidate_prior_cumulative_likes is None:
                raise ValueError("candidate_prior_cumulative_likes is required when popularity features are enabled")
            if history_prior_cumulative_likes.shape != (batch_size, max_history_len):
                raise ValueError("history_prior_cumulative_likes must have shape [B, H]")
            if candidate_prior_cumulative_likes.shape != (batch_size,):
                raise ValueError("candidate_prior_cumulative_likes must have shape [B]")

        device = history_embeddings.device
        history_mask = history_mask.to(device=device, dtype=torch.bool)
        history_time_deltas_hours = history_time_deltas_hours.to(device=device)
        candidate_post_embeddings = candidate_post_embeddings.to(device=device)
        history_author_indices = history_author_indices.to(device=device, dtype=torch.long)
        candidate_post_author_idx = candidate_post_author_idx.to(device=device, dtype=torch.long)
        history_prior_cumulative_likes_tensor = history_prior_cumulative_likes
        candidate_prior_cumulative_likes_tensor = candidate_prior_cumulative_likes
        if self.use_popularity_feature:
            history_prior_cumulative_likes_tensor = torch.jit._unwrap_optional(history_prior_cumulative_likes).to(device=device, dtype=torch.float32)
            candidate_prior_cumulative_likes_tensor = torch.jit._unwrap_optional(candidate_prior_cumulative_likes).to(device=device, dtype=torch.float32)

        history_post_vectors = self.post_feature_encoder(
            history_embeddings,
            history_author_indices,
            history_prior_cumulative_likes_tensor,
        )
        candidate_post_vector = self.post_feature_encoder(
            candidate_post_embeddings,
            candidate_post_author_idx,
            candidate_prior_cumulative_likes_tensor,
        ).unsqueeze(1)
        history_time_bucket_ids = self._bucketize_time_deltas_hours(history_time_deltas_hours)
        history_time_embeddings = self.time_delta_embedding(history_time_bucket_ids)
        history_input = torch.cat([history_post_vectors, history_time_embeddings], dim=-1)
        history_input, history_mask = self._inject_empty_history_token(
            history_input,
            history_mask,
        )
        # The candidate token represents the item being scored now. Post age is
        # a separate freshness concept and is intentionally not encoded here.
        candidate_time_bucket_ids = torch.zeros(
            (batch_size, 1),
            device=device,
            dtype=torch.long,
        )
        candidate_time_embeddings = self.time_delta_embedding(candidate_time_bucket_ids)
        candidate_input = torch.cat([candidate_post_vector, candidate_time_embeddings], dim=-1)
        transformer_input = torch.cat([history_input, candidate_input], dim=1)

        candidate_is_not_padding = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
        src_key_padding_mask = torch.cat([~history_mask, candidate_is_not_padding], dim=1)
        encoded_sequence = self.transformer_encoder(
            transformer_input,
            src_key_padding_mask=src_key_padding_mask,
        )
        return encoded_sequence[:, -1, :]

    def _validate_one_layer_matrix_scorer(self) -> nn.TransformerEncoderLayer:
        layers = getattr(self.transformer_encoder, "layers", None)
        if layers is None or len(layers) != 1:
            raise RuntimeError("score_candidate_matrix_one_layer requires exactly one transformer layer")
        layer = layers[0]
        if not isinstance(layer, nn.TransformerEncoderLayer):
            raise RuntimeError("score_candidate_matrix_one_layer requires a standard TransformerEncoderLayer")
        self_attn = layer.self_attn
        if (
            not isinstance(self_attn, nn.MultiheadAttention)
            or not self_attn.batch_first
            or not getattr(self_attn, "_qkv_same_embed_dim", False)
            or self_attn.in_proj_weight is None
            or self_attn.in_proj_bias is None
            or self_attn.out_proj is None
        ):
            raise RuntimeError("score_candidate_matrix_one_layer requires packed batch-first self-attention projections")
        if self_attn.embed_dim != self.transformer_input_dim:
            raise RuntimeError("score_candidate_matrix_one_layer found a transformer dimension mismatch")
        return layer

    def _candidate_token_self_attention(
        self,
        history_input: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_input: torch.Tensor,
    ) -> torch.Tensor:
        # Resolve the layer from this module instead of accepting it as a typed
        # argument. TorchScript otherwise binds the helper to the first
        # concrete TransformerEncoderLayer compiled in the process, which
        # prevents later best-checkpoint exports with a different model shape.
        layer = self.transformer_encoder.layers[0]
        self_attn = layer.self_attn
        num_users, max_history_len, embed_dim = history_input.shape
        num_candidates = int(candidate_input.size(0))
        num_heads = int(self_attn.num_heads)
        head_dim = embed_dim // num_heads
        scale = float(head_dim) ** -0.5

        in_proj_weight = self_attn.in_proj_weight
        in_proj_bias = self_attn.in_proj_bias
        if in_proj_weight is None or in_proj_bias is None:
            raise RuntimeError("score_candidate_matrix requires packed self-attention projections")
        q_weight, k_weight, v_weight = in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = in_proj_bias.chunk(3, dim=0)
        query = F.linear(candidate_input, q_weight, q_bias).view(num_candidates, num_heads, head_dim)
        history_key = F.linear(history_input, k_weight, k_bias).view(num_users, max_history_len, num_heads, head_dim)
        history_value = F.linear(history_input, v_weight, v_bias).view(num_users, max_history_len, num_heads, head_dim)
        candidate_key = F.linear(candidate_input, k_weight, k_bias).view(num_candidates, num_heads, head_dim)
        candidate_value = F.linear(candidate_input, v_weight, v_bias).view(num_candidates, num_heads, head_dim)

        history_scores = torch.einsum("cnd,uhnd->unch", query, history_key) * scale
        history_scores = history_scores.masked_fill(~history_mask[:, None, None, :], float("-inf"))
        candidate_scores = (query * candidate_key).sum(dim=-1).transpose(0, 1) * scale
        candidate_scores = candidate_scores.unsqueeze(0).unsqueeze(-1).expand(num_users, -1, -1, -1)
        attention_scores = torch.cat([history_scores, candidate_scores], dim=-1)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout_rate, training=self.training)

        if max_history_len == 0:
            history_context = torch.zeros(
                (num_users, num_candidates, num_heads, head_dim),
                device=history_input.device,
                dtype=history_input.dtype,
            )
        else:
            history_context = torch.einsum(
                "unch,uhnd->ucnd",
                attention_weights[..., :max_history_len],
                history_value,
            )
        candidate_context = (
            attention_weights[..., max_history_len].permute(0, 2, 1).unsqueeze(-1)
            * candidate_value.unsqueeze(0)
        )
        attention_output = (history_context + candidate_context).reshape(num_users, num_candidates, embed_dim)
        return F.linear(attention_output, self_attn.out_proj.weight, self_attn.out_proj.bias)

    def _candidate_token_feed_forward(
        self,
        candidate_state: torch.Tensor,
    ) -> torch.Tensor:
        layer = self.transformer_encoder.layers[0]
        hidden = F.linear(candidate_state, layer.linear1.weight, layer.linear1.bias)
        hidden = F.gelu(hidden)
        hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)
        hidden = F.linear(hidden, layer.linear2.weight, layer.linear2.bias)
        return F.dropout(hidden, p=self.dropout_rate, training=self.training)

    def score_candidate_matrix_one_layer(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        candidate_post_author_idx: torch.Tensor,
        history_prior_cumulative_likes: Optional[torch.Tensor] = None,
        candidate_prior_cumulative_likes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_one_layer_matrix_scorer()
        if history_embeddings.dim() != 3:
            raise ValueError("history_embeddings must have shape [U, H, D]")
        if candidate_post_embeddings.dim() != 2:
            raise ValueError("candidate_post_embeddings must have shape [C, D]")
        num_users, max_history_len, embed_dim = history_embeddings.shape
        num_candidates = int(candidate_post_embeddings.size(0))
        if embed_dim != self.post_embedding_dim:
            raise ValueError(
                f"history_embeddings last dimension ({embed_dim}) must match post_embedding_dim ({self.post_embedding_dim})"
            )
        if candidate_post_embeddings.shape != (num_candidates, self.post_embedding_dim):
            raise ValueError("candidate_post_embeddings must have shape [C, post_embedding_dim]")
        if history_mask.shape != (num_users, max_history_len):
            raise ValueError("history_mask must have shape [U, H]")
        if history_time_deltas_hours.shape != (num_users, max_history_len):
            raise ValueError("history_time_deltas_hours must have shape [U, H]")
        if history_author_indices.shape != (num_users, max_history_len):
            raise ValueError("history_author_indices must have shape [U, H]")
        if candidate_post_author_idx.shape != (num_candidates,):
            raise ValueError("candidate_post_author_idx must have shape [C]")
        if self.use_popularity_feature:
            if history_prior_cumulative_likes is None:
                raise ValueError("history_prior_cumulative_likes is required when popularity features are enabled")
            if candidate_prior_cumulative_likes is None:
                raise ValueError("candidate_prior_cumulative_likes is required when popularity features are enabled")
            if history_prior_cumulative_likes.shape != (num_users, max_history_len):
                raise ValueError("history_prior_cumulative_likes must have shape [U, H]")
            if candidate_prior_cumulative_likes.shape != (num_candidates,):
                raise ValueError("candidate_prior_cumulative_likes must have shape [C]")

        return self.score_candidate_matrix(
            history_embeddings=history_embeddings,
            history_mask=history_mask,
            history_time_deltas_hours=history_time_deltas_hours,
            candidate_post_embeddings=candidate_post_embeddings,
            history_author_indices=history_author_indices,
            candidate_post_author_idx=candidate_post_author_idx,
            history_prior_cumulative_likes=history_prior_cumulative_likes,
            candidate_prior_cumulative_likes=candidate_prior_cumulative_likes,
        )

    @torch.jit.export
    def score_candidate_matrix(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        candidate_post_author_idx: torch.Tensor,
        history_prior_cumulative_likes: Optional[torch.Tensor] = None,
        candidate_prior_cumulative_likes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if len(self.transformer_encoder.layers) != 1:
            raise RuntimeError("score_candidate_matrix requires exactly one transformer layer")
        layer = self.transformer_encoder.layers[0]

        num_users = int(history_embeddings.size(0))
        num_candidates = int(candidate_post_embeddings.size(0))
        device = history_embeddings.device
        history_mask = history_mask.to(device=device, dtype=torch.bool)
        history_time_deltas_hours = history_time_deltas_hours.to(device=device)
        candidate_post_embeddings = candidate_post_embeddings.to(device=device)
        history_author_indices = history_author_indices.to(device=device, dtype=torch.long)
        candidate_post_author_idx = candidate_post_author_idx.to(device=device, dtype=torch.long)
        history_prior_cumulative_likes_tensor = history_prior_cumulative_likes
        candidate_prior_cumulative_likes_tensor = candidate_prior_cumulative_likes
        if self.use_popularity_feature:
            if history_prior_cumulative_likes is None:
                raise RuntimeError("history_prior_cumulative_likes is required when popularity features are enabled")
            if candidate_prior_cumulative_likes is None:
                raise RuntimeError("candidate_prior_cumulative_likes is required when popularity features are enabled")
            if history_prior_cumulative_likes.size(0) != num_users or history_prior_cumulative_likes.size(1) != history_embeddings.size(1):
                raise RuntimeError("history_prior_cumulative_likes must have shape [U, H]")
            if candidate_prior_cumulative_likes.size(0) != num_candidates:
                raise RuntimeError("candidate_prior_cumulative_likes must have shape [C]")
            history_prior_cumulative_likes_tensor = torch.jit._unwrap_optional(history_prior_cumulative_likes).to(device=device, dtype=torch.float32)
            candidate_prior_cumulative_likes_tensor = torch.jit._unwrap_optional(candidate_prior_cumulative_likes).to(device=device, dtype=torch.float32)

        history_post_vectors = self.post_feature_encoder(
            history_embeddings,
            history_author_indices,
            history_prior_cumulative_likes_tensor,
        )
        candidate_post_vectors = self.post_feature_encoder(
            candidate_post_embeddings,
            candidate_post_author_idx,
            candidate_prior_cumulative_likes_tensor,
        )
        history_time_bucket_ids = self._bucketize_time_deltas_hours(history_time_deltas_hours)
        history_time_embeddings = self.time_delta_embedding(history_time_bucket_ids)
        candidate_time_bucket_ids = torch.zeros((num_candidates,), device=device, dtype=torch.long)
        candidate_time_embeddings = self.time_delta_embedding(candidate_time_bucket_ids)
        history_input = torch.cat([history_post_vectors, history_time_embeddings], dim=-1)
        history_input, history_mask = self._inject_empty_history_token(
            history_input,
            history_mask,
        )
        candidate_input = torch.cat([candidate_post_vectors, candidate_time_embeddings], dim=-1)

        if layer.norm_first:
            normed_history_input = F.layer_norm(
                history_input,
                [self.transformer_input_dim],
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
            )
            normed_candidate_input = F.layer_norm(
                candidate_input,
                [self.transformer_input_dim],
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
            )
            attention_output = F.dropout(
                self._candidate_token_self_attention(
                    normed_history_input,
                    history_mask,
                    normed_candidate_input,
                ),
                p=self.dropout_rate,
                training=self.training,
            )
            candidate_state = candidate_input.unsqueeze(0) + attention_output
            normed_candidate_state = F.layer_norm(
                candidate_state,
                [self.transformer_input_dim],
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps,
            )
            candidate_state = candidate_state + self._candidate_token_feed_forward(
                normed_candidate_state,
            )
        else:
            attention_output = F.dropout(
                self._candidate_token_self_attention(
                    history_input,
                    history_mask,
                    candidate_input,
                ),
                p=self.dropout_rate,
                training=self.training,
            )
            candidate_state = F.layer_norm(
                candidate_input.unsqueeze(0) + attention_output,
                [self.transformer_input_dim],
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
            )
            candidate_state = F.layer_norm(
                candidate_state + self._candidate_token_feed_forward(candidate_state),
                [self.transformer_input_dim],
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps,
            )

        logits = self.prediction_head(candidate_state.reshape(num_users * num_candidates, self.transformer_input_dim))
        if logits.dim() == 2 and logits.shape == (num_users * num_candidates, 1):
            logits = logits.squeeze(-1)
        if logits.shape != (num_users * num_candidates,):
            raise RuntimeError("prediction_head must return logits with shape [U*C] or [U*C, 1]")
        return logits.reshape(num_users, num_candidates)

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        candidate_post_author_idx: torch.Tensor,
        history_prior_cumulative_likes: Optional[torch.Tensor] = None,
        candidate_prior_cumulative_likes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        transformer_output = self._forward_transformer(
            history_embeddings=history_embeddings,
            history_mask=history_mask,
            history_time_deltas_hours=history_time_deltas_hours,
            candidate_post_embeddings=candidate_post_embeddings,
            history_author_indices=history_author_indices,
            candidate_post_author_idx=candidate_post_author_idx,
            history_prior_cumulative_likes=history_prior_cumulative_likes,
            candidate_prior_cumulative_likes=candidate_prior_cumulative_likes,
        )
        logits = self.prediction_head(transformer_output)
        if logits.dim() == 2 and logits.shape == (transformer_output.size(0), 1):
            logits = logits.squeeze(-1)
        if logits.shape != (transformer_output.size(0),):
            raise RuntimeError("prediction_head must return logits with shape [B] or [B, 1]")
        return logits
