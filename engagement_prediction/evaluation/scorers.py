"""TorchScript matrix scorers for supported comparison artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from engagement_prediction.evaluation.artifacts import (
    BST_MODEL_TYPE,
    TWO_TOWER_MODEL_TYPE,
    ModelArtifact,
)
from engagement_prediction.training.ranking import MatrixBatchScores


def _batch_tensor(
    batch: dict[str, Any],
    key: str,
    *,
    device: str,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fetch one required host tensor and transfer it non-blockingly."""

    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Ranking batch is missing tensor {key!r}")
    return value.to(
        device,
        dtype=dtype if dtype is not None else value.dtype,
        non_blocking=True,
    )


class BSTTorchScriptScorer:
    """Score one shared candidate matrix through bounded BST chunks."""

    def __init__(self, scripted_model_path: Path, candidate_chunk_size: int):
        if candidate_chunk_size <= 0:
            raise ValueError("BST candidate chunk size must be positive")
        self.scripted_model_path = Path(scripted_model_path)
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.model: Any = None
        self._device: str | None = None

    def prepare_for_eval(self, device: str) -> None:
        """Lazily load the script on the requested device and reuse it."""

        if self.model is not None and self._device == str(device):
            self.model.eval()
            return
        self.close()
        try:
            self.model = torch.jit.load(
                str(self.scripted_model_path),
                map_location=device,
            )
        except Exception as exc:
            raise ValueError(
                f"Could not load BST TorchScript {self.scripted_model_path}: {exc}"
            ) from exc
        self.model.eval()
        self._device = str(device)

    def score_batch(self, batch: dict[str, Any], device: str) -> MatrixBatchScores:
        """Score candidates in bounded chunks while reusing user histories."""

        if self.model is None:
            raise RuntimeError("BST scorer must be prepared before scoring")
        history_embeddings = _batch_tensor(
            batch, "history_embeddings", device=device
        )
        history_mask = _batch_tensor(
            batch, "history_mask", device=device, dtype=torch.bool
        )
        history_time_deltas_hours = _batch_tensor(
            batch, "history_time_deltas_hours", device=device
        )
        history_author_indices = _batch_tensor(
            batch, "history_author_indices", device=device, dtype=torch.long
        )
        history_prior_cumulative_likes = _batch_tensor(
            batch,
            "history_prior_cumulative_likes",
            device=device,
            dtype=torch.float32,
        )
        candidate_post_embeddings = _batch_tensor(
            batch, "candidate_post_embeddings", device=device
        )
        candidate_post_author_idx = _batch_tensor(
            batch,
            "candidate_post_author_idx",
            device=device,
            dtype=torch.long,
        )
        candidate_prior_cumulative_likes = _batch_tensor(
            batch,
            "candidate_prior_cumulative_likes",
            device=device,
            dtype=torch.float32,
        )
        candidate_count = int(candidate_post_embeddings.size(0))
        score_chunks = []
        for start in range(0, candidate_count, self.candidate_chunk_size):
            end = min(start + self.candidate_chunk_size, candidate_count)
            score_chunks.append(
                self.model.score_candidate_matrix(
                    history_embeddings,
                    history_mask,
                    history_time_deltas_hours,
                    candidate_post_embeddings[start:end],
                    history_author_indices,
                    candidate_post_author_idx[start:end],
                    history_prior_cumulative_likes,
                    candidate_prior_cumulative_likes[start:end],
                )
            )
        if score_chunks:
            scores = torch.cat(score_chunks, dim=1)
        else:
            scores = torch.empty(
                (int(history_embeddings.size(0)), 0),
                device=device,
                dtype=history_embeddings.dtype,
            )
        return MatrixBatchScores(scores=scores)

    def close(self) -> None:
        """Drop the loaded ScriptModule so its device memory can be released."""

        self.model = None
        self._device = None


class TwoTowerTorchScriptScorer:
    """Combine canonical user and post towers with training temperature."""

    def __init__(
        self,
        user_tower_path: Path,
        post_tower_path: Path,
        similarity_temperature: float,
        output_embedding_dim: int,
    ):
        if not math.isfinite(similarity_temperature) or similarity_temperature <= 0.0:
            raise ValueError("Two-tower similarity temperature must be positive")
        if output_embedding_dim <= 0:
            raise ValueError("Two-tower output embedding dimension must be positive")
        self.user_tower_path = Path(user_tower_path)
        self.post_tower_path = Path(post_tower_path)
        self.similarity_temperature = float(similarity_temperature)
        self.output_embedding_dim = int(output_embedding_dim)
        self.user_tower: Any = None
        self.post_tower: Any = None
        self._device: str | None = None

    def prepare_for_eval(self, device: str) -> None:
        """Lazily load the matching exported tower pair on one device."""

        if (
            self.user_tower is not None
            and self.post_tower is not None
            and self._device == str(device)
        ):
            self.user_tower.eval()
            self.post_tower.eval()
            return
        self.close()
        try:
            self.user_tower = torch.jit.load(
                str(self.user_tower_path),
                map_location=device,
            )
            self.post_tower = torch.jit.load(
                str(self.post_tower_path),
                map_location=device,
            )
        except Exception as exc:
            self.close()
            raise ValueError(
                "Could not load two-tower TorchScript artifacts "
                f"{self.user_tower_path} and {self.post_tower_path}: {exc}"
            ) from exc
        self.user_tower.eval()
        self.post_tower.eval()
        self._device = str(device)

    def score_batch(self, batch: dict[str, Any], device: str) -> MatrixBatchScores:
        """Encode each side once and form the complete score matrix."""

        if self.user_tower is None or self.post_tower is None:
            raise RuntimeError("Two-tower scorer must be prepared before scoring")
        history_embeddings = _batch_tensor(
            batch, "history_embeddings", device=device
        )
        history_mask = _batch_tensor(
            batch, "history_mask", device=device, dtype=torch.bool
        )
        history_author_indices = _batch_tensor(
            batch, "history_author_indices", device=device, dtype=torch.long
        )
        candidate_post_embeddings = _batch_tensor(
            batch, "candidate_post_embeddings", device=device
        )
        candidate_post_author_idx = _batch_tensor(
            batch,
            "candidate_post_author_idx",
            device=device,
            dtype=torch.long,
        )
        user_embeddings = self.user_tower(
            history_embeddings,
            history_mask,
            history_author_indices,
        )
        post_embeddings = self.post_tower(
            candidate_post_embeddings,
            candidate_post_author_idx,
        )
        expected_user_shape = (
            int(history_embeddings.size(0)),
            self.output_embedding_dim,
        )
        expected_post_shape = (
            int(candidate_post_embeddings.size(0)),
            self.output_embedding_dim,
        )
        if tuple(user_embeddings.shape) != expected_user_shape:
            raise RuntimeError(
                "User tower returned an unexpected shape: "
                f"{tuple(user_embeddings.shape)} != {expected_user_shape}"
            )
        if tuple(post_embeddings.shape) != expected_post_shape:
            raise RuntimeError(
                "Post tower returned an unexpected shape: "
                f"{tuple(post_embeddings.shape)} != {expected_post_shape}"
            )
        scores = torch.matmul(user_embeddings, post_embeddings.transpose(0, 1))
        scores = scores / self.similarity_temperature
        return MatrixBatchScores(scores=scores)

    def close(self) -> None:
        """Drop both ScriptModules so their device memory can be released."""

        self.user_tower = None
        self.post_tower = None
        self._device = None


def create_model_scorer(
    model: ModelArtifact,
    *,
    bst_candidate_chunk_size: int,
) -> BSTTorchScriptScorer | TwoTowerTorchScriptScorer:
    """Instantiate the scorer matching an already resolved artifact."""

    if model.model_type == BST_MODEL_TYPE:
        return BSTTorchScriptScorer(
            model.script_paths["ranker"],
            candidate_chunk_size=bst_candidate_chunk_size,
        )
    if model.model_type == TWO_TOWER_MODEL_TYPE:
        constructor = model.model_config["constructor_args"]
        assert model.similarity_temperature is not None
        return TwoTowerTorchScriptScorer(
            model.script_paths["user_tower"],
            model.script_paths["post_tower"],
            similarity_temperature=model.similarity_temperature,
            output_embedding_dim=int(constructor["output_embedding_dim"]),
        )
    raise ValueError(f"Unsupported model type: {model.model_type}")
