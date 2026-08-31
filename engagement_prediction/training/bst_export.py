"""Serving-safe TorchScript export for canonical BST checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from engagement_prediction.models.bst_ranker import BSTRanker
from engagement_prediction.training.model_artifacts import file_sha256


_MODEL_TYPE = "bst-ranker"


def _require_mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    """Narrow dynamically loaded checkpoint metadata to a mapping."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    return value


def _load_checkpoint_model(
    *,
    checkpoint_path: Path,
    expected_model_config: Mapping[str, Any],
    expected_popularity_stats: Mapping[str, Any],
) -> tuple[BSTRanker, int]:
    """Reconstruct the exact CPU model described by a canonical checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"BST checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise ValueError(f"Failed to load BST checkpoint {checkpoint_path}: {exc}") from exc

    checkpoint = _require_mapping(checkpoint, description="BST checkpoint")
    metadata = _require_mapping(
        checkpoint.get("metadata"),
        description="BST checkpoint metadata",
    )
    checkpoint_model_config = _require_mapping(
        metadata.get("model_config"),
        description="BST checkpoint model_config",
    )
    checkpoint_popularity_stats = _require_mapping(
        metadata.get("popularity_stats"),
        description="BST checkpoint popularity_stats",
    )
    if dict(checkpoint_model_config) != dict(expected_model_config):
        raise ValueError("BST checkpoint model_config does not match the expected model configuration")
    if dict(checkpoint_popularity_stats) != dict(expected_popularity_stats):
        raise ValueError("BST checkpoint popularity_stats do not match the expected popularity statistics")
    if checkpoint_model_config.get("model_type") != _MODEL_TYPE:
        raise ValueError(f"BST checkpoint model_type must be '{_MODEL_TYPE}'")

    constructor_args = _require_mapping(
        checkpoint_model_config.get("constructor_args"),
        description="BST checkpoint constructor_args",
    )
    popularity_enabled = checkpoint_popularity_stats.get("enabled")
    if not isinstance(popularity_enabled, bool):
        raise ValueError("BST checkpoint popularity_stats.enabled must be a boolean")
    if constructor_args.get("use_popularity_feature") is not popularity_enabled:
        raise ValueError(
            "BST checkpoint popularity configuration disagrees with constructor_args"
        )
    if constructor_args.get("popularity_log_mean") != checkpoint_popularity_stats.get("log_mean"):
        raise ValueError(
            "BST checkpoint popularity_log_mean disagrees with popularity_stats"
        )
    if constructor_args.get("popularity_log_std") != checkpoint_popularity_stats.get("log_std"):
        raise ValueError(
            "BST checkpoint popularity_log_std disagrees with popularity_stats"
        )
    if constructor_args.get("num_transformer_layers") != 1:
        raise ValueError("Serving BST checkpoints must contain exactly one transformer layer")

    best_epoch = checkpoint.get("best_epoch")
    if not isinstance(best_epoch, int) or isinstance(best_epoch, bool) or best_epoch < 1:
        raise ValueError("BST checkpoint best_epoch must be a positive integer")
    checkpoint_epoch = checkpoint.get("epoch")
    if checkpoint_epoch != best_epoch:
        raise ValueError("BST checkpoint epoch must match best_epoch")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("BST checkpoint model_state_dict must be a mapping")
    try:
        model = BSTRanker(**dict(constructor_args))
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ValueError(f"BST checkpoint cannot reconstruct its configured model: {exc}") from exc
    return model.cpu().eval(), best_epoch


def _parity_inputs(
    *,
    post_embedding_dim: int,
    author_table_num_rows: int,
    zero_length_history: bool,
) -> tuple[torch.Tensor, ...]:
    """Create deterministic inputs matching the inference service's eight arguments."""

    num_users = 2
    num_candidates = 3
    history_len = 0 if zero_length_history else 3
    history_value_count = num_users * history_len * post_embedding_dim
    if history_value_count:
        history_embeddings = torch.arange(
            history_value_count,
            dtype=torch.float32,
        ).reshape(num_users, history_len, post_embedding_dim)
        history_embeddings = history_embeddings / float(max(post_embedding_dim, 1) + 7)
    else:
        history_embeddings = torch.empty(
            (num_users, 0, post_embedding_dim),
            dtype=torch.float32,
        )
    candidate_post_embeddings = torch.arange(
        num_candidates * post_embedding_dim,
        dtype=torch.float32,
    ).reshape(num_candidates, post_embedding_dim)
    candidate_post_embeddings = candidate_post_embeddings / float(
        max(post_embedding_dim, 1) + 11
    )

    if zero_length_history:
        history_mask = torch.empty((num_users, 0), dtype=torch.bool)
        history_time_deltas_hours = torch.empty((num_users, 0), dtype=torch.float32)
        history_author_indices = torch.empty((num_users, 0), dtype=torch.long)
        history_prior_cumulative_likes = torch.empty(
            (num_users, 0),
            dtype=torch.long,
        )
    else:
        history_mask = torch.tensor(
            [[True, True, False], [False, False, False]],
            dtype=torch.bool,
        )
        history_time_deltas_hours = torch.tensor(
            [[0.0, 12.0, 500.0], [1.0, 2.0, 3.0]],
            dtype=torch.float32,
        )
        author_modulus = max(author_table_num_rows, 1)
        history_author_indices = (
            torch.arange(num_users * history_len, dtype=torch.long)
            .reshape(num_users, history_len)
            .remainder(author_modulus)
        )
        history_prior_cumulative_likes = torch.tensor(
            [[0, 5, 999], [1, 2, 3]],
            dtype=torch.long,
        )

    candidate_post_author_idx = torch.arange(
        num_candidates,
        dtype=torch.long,
    ).remainder(max(author_table_num_rows, 1))
    candidate_prior_cumulative_likes = torch.tensor(
        [0, 10, 100],
        dtype=torch.long,
    )
    return (
        history_embeddings,
        history_mask,
        history_time_deltas_hours,
        candidate_post_embeddings,
        history_author_indices,
        candidate_post_author_idx,
        history_prior_cumulative_likes,
        candidate_prior_cumulative_likes,
    )


def _validate_score_parity(
    *,
    eager_model: BSTRanker,
    scripted_model: torch.jit.ScriptModule,
) -> dict[str, Any]:
    """Require bitwise eager/scripted parity across serving edge cases."""

    case_results: list[dict[str, Any]] = []
    constructor_values = {
        "post_embedding_dim": eager_model.post_embedding_dim,
        "author_table_num_rows": eager_model.post_feature_encoder.author_embedding.num_embeddings,
    }
    with torch.inference_mode():
        for case_name, zero_length_history in (
            ("normal_and_all_masked", False),
            ("zero_length_history", True),
        ):
            inputs = _parity_inputs(
                **constructor_values,
                zero_length_history=zero_length_history,
            )
            eager_scores = eager_model.score_candidate_matrix(*inputs)
            scripted_scores = scripted_model.score_candidate_matrix(*inputs)
            if eager_scores.shape != (2, 3):
                raise RuntimeError(
                    f"BST export parity case '{case_name}' returned unexpected shape "
                    f"{tuple(eager_scores.shape)}"
                )
            if not bool(torch.isfinite(eager_scores).all().item()):
                raise RuntimeError(
                    f"BST export parity case '{case_name}' produced non-finite eager scores"
                )
            if not bool(torch.isfinite(scripted_scores).all().item()):
                raise RuntimeError(
                    f"BST export parity case '{case_name}' produced non-finite scripted scores"
                )
            if not torch.equal(eager_scores, scripted_scores):
                max_absolute_difference = float(
                    (eager_scores - scripted_scores).abs().max().item()
                )
                raise RuntimeError(
                    f"BST export parity case '{case_name}' was not exactly equal; "
                    f"max_absolute_difference={max_absolute_difference}"
                )
            case_results.append(
                {
                    "case": case_name,
                    "shape": [2, 3],
                    "exact_match": True,
                    "finite": True,
                }
            )
    return {
        "case_count": len(case_results),
        "all_exact": True,
        "cases": case_results,
    }


def validate_bst_ranker_export(
    *,
    checkpoint_path: Path,
    scripted_model_path: Path,
    expected_model_config: Mapping[str, Any],
    expected_popularity_stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that a saved ScriptModule exactly represents its best checkpoint."""

    eager_model, best_epoch = _load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        expected_model_config=expected_model_config,
        expected_popularity_stats=expected_popularity_stats,
    )
    scripted_model_path = Path(scripted_model_path)
    if not scripted_model_path.is_file():
        raise FileNotFoundError(
            f"BST TorchScript artifact does not exist: {scripted_model_path}"
        )
    try:
        scripted_model = torch.jit.load(str(scripted_model_path), map_location="cpu").eval()
    except Exception as exc:
        raise ValueError(
            f"Failed to load BST TorchScript artifact {scripted_model_path}: {exc}"
        ) from exc

    parity = _validate_score_parity(
        eager_model=eager_model,
        scripted_model=scripted_model,
    )
    return {
        "best_epoch": best_epoch,
        "size_bytes": scripted_model_path.stat().st_size,
        "sha256": file_sha256(scripted_model_path),
        "parity": parity,
    }


def export_bst_ranker_checkpoint(
    *,
    checkpoint_path: Path,
    output_path: Path,
    expected_model_config: Mapping[str, Any],
    expected_popularity_stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Script, validate, and atomically publish one canonical BST checkpoint."""

    eager_model, _ = _load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        expected_model_config=expected_model_config,
        expected_popularity_stats=expected_popularity_stats,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The partial suffix also keeps ClearML's ``*.pt`` framework hook from
    # observing an unvalidated intermediate model.
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    scripted_model = torch.jit.script(eager_model)
    scripted_model.save(str(partial_path))
    validation = validate_bst_ranker_export(
        checkpoint_path=checkpoint_path,
        scripted_model_path=partial_path,
        expected_model_config=expected_model_config,
        expected_popularity_stats=expected_popularity_stats,
    )
    partial_path.replace(output_path)
    return {
        "path": str(output_path),
        **validation,
    }
