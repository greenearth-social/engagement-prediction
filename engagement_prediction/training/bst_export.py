"""Serving-safe TorchScript export for canonical BST checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl
import torch

from engagement_prediction.data.parquet import scan_parquet_artifact
from engagement_prediction.data.post_liker_users import (
    POST_LIKER_USER_PAD_IDX,
    POST_LIKER_USER_UNK_IDX,
    POST_LIKER_USER_VOCABULARY_SCHEMA,
)
from engagement_prediction.models.bst_ranker import BSTRanker
from engagement_prediction.training.model_artifacts import (
    file_sha256,
    write_json_atomically,
)


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


def _post_liker_parity_inputs(
    *,
    eager_model: BSTRanker,
    base_inputs: tuple[torch.Tensor, ...],
    event_case: str,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    """Build packed events and the equivalent explicit pooled-vector inputs."""

    history_embeddings = base_inputs[0]
    candidate_embeddings = base_inputs[3]
    num_users = int(history_embeddings.size(0))
    history_len = int(history_embeddings.size(1))
    num_candidates = int(candidate_embeddings.size(0))
    num_post_rows = max(5, num_candidates)

    if event_case == "no_liker_events":
        event_user_indices = torch.empty((0,), dtype=torch.long)
        event_ages = torch.empty((0,), dtype=torch.float32)
        event_offsets = torch.zeros((num_post_rows + 1,), dtype=torch.long)
    elif event_case == "all_unknown_events":
        event_user_indices = torch.ones((num_post_rows,), dtype=torch.long)
        event_ages = torch.arange(num_post_rows, dtype=torch.float32) * 6.0
        event_offsets = torch.arange(num_post_rows + 1, dtype=torch.long)
    elif event_case == "mixed_events":
        table_rows = eager_model.post_liker_user_pooler.user_embedding.num_embeddings
        known_two = 2 if table_rows > 2 else POST_LIKER_USER_UNK_IDX
        known_three = 3 if table_rows > 3 else known_two
        event_user_indices = torch.tensor(
            [
                known_two,
                POST_LIKER_USER_UNK_IDX,
                known_three,
                POST_LIKER_USER_UNK_IDX,
            ],
            dtype=torch.long,
        )
        event_ages = torch.tensor([0.0, 24.0, 0.0, 168.0], dtype=torch.float32)
        # The first and final post rows deliberately have no events.
        event_offsets = torch.tensor([0, 0, 2, 3, 4, 4], dtype=torch.long)
    else:
        raise ValueError(f"Unknown post-liker export parity case: {event_case}")

    if history_len:
        history_rows = (
            torch.arange(num_users * history_len, dtype=torch.long)
            .reshape(num_users, history_len)
            .remainder(num_post_rows)
        )
    else:
        history_rows = torch.empty((num_users, 0), dtype=torch.long)
    candidate_rows = torch.arange(num_candidates, dtype=torch.long).remainder(
        num_post_rows
    )

    pooled = eager_model.post_liker_user_pooler(
        event_user_indices,
        event_ages,
        event_offsets,
    )
    history_vectors = pooled.index_select(0, history_rows.reshape(-1)).reshape(
        num_users,
        history_len,
        eager_model.post_liker_user_embedding_dim,
    )
    candidate_vectors = pooled.index_select(0, candidate_rows)
    event_inputs = (
        *base_inputs,
        event_user_indices,
        event_ages,
        event_offsets,
        history_rows,
        candidate_rows,
    )
    return event_inputs, history_vectors, candidate_vectors


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
    use_post_liker_feature = bool(eager_model.use_post_liker_feature)
    with torch.inference_mode():
        base_cases = (
            ("normal_and_all_masked", False),
            ("zero_length_history", True),
        )
        for case_name, zero_length_history in base_cases:
            inputs = _parity_inputs(
                **constructor_values,
                zero_length_history=zero_length_history,
            )
            if use_post_liker_feature:
                event_inputs, history_vectors, candidate_vectors = (
                    _post_liker_parity_inputs(
                        eager_model=eager_model,
                        base_inputs=inputs,
                        event_case="mixed_events",
                    )
                )
                eager_scores = eager_model.score_candidate_matrix(
                    *inputs,
                    history_vectors,
                    candidate_vectors,
                )
                scripted_scores = scripted_model.score_candidate_matrix(
                    *inputs,
                    history_vectors,
                    candidate_vectors,
                )
                eager_event_scores = (
                    eager_model.score_candidate_matrix_from_post_liker_events(
                        *event_inputs
                    )
                )
                scripted_event_scores = (
                    scripted_model.score_candidate_matrix_from_post_liker_events(
                        *event_inputs
                    )
                )
                for description, scores in (
                    ("eager event", eager_event_scores),
                    ("scripted event", scripted_event_scores),
                ):
                    if not torch.equal(eager_scores, scores):
                        difference = float((eager_scores - scores).abs().max().item())
                        raise RuntimeError(
                            f"BST export parity case '{case_name}' {description} "
                            "scores disagreed with explicit pooled-vector scores; "
                            f"max_absolute_difference={difference}"
                        )
            else:
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
                    **(
                        {"event_and_prepooled_exact_match": True}
                        if use_post_liker_feature
                        else {}
                    ),
                }
            )

        if use_post_liker_feature:
            # These cases exercise the two representations that are easy to
            # mishandle in serving: an entirely empty state and an all-UNK
            # state populated only by long-tail users.
            normal_inputs = _parity_inputs(
                **constructor_values,
                zero_length_history=False,
            )
            for case_name, event_case in (
                ("no_liker_events", "no_liker_events"),
                ("all_unknown_liker_events", "all_unknown_events"),
            ):
                event_inputs, history_vectors, candidate_vectors = (
                    _post_liker_parity_inputs(
                        eager_model=eager_model,
                        base_inputs=normal_inputs,
                        event_case=event_case,
                    )
                )
                eager_scores = eager_model.score_candidate_matrix(
                    *normal_inputs,
                    history_vectors,
                    candidate_vectors,
                )
                scripted_scores = scripted_model.score_candidate_matrix(
                    *normal_inputs,
                    history_vectors,
                    candidate_vectors,
                )
                eager_event_scores = (
                    eager_model.score_candidate_matrix_from_post_liker_events(
                        *event_inputs
                    )
                )
                scripted_event_scores = (
                    scripted_model.score_candidate_matrix_from_post_liker_events(
                        *event_inputs
                    )
                )
                if not (
                    torch.equal(eager_scores, scripted_scores)
                    and torch.equal(eager_scores, eager_event_scores)
                    and torch.equal(eager_scores, scripted_event_scores)
                ):
                    raise RuntimeError(
                        f"BST export parity case '{case_name}' was not exactly equal"
                    )
                if not bool(torch.isfinite(eager_scores).all().item()):
                    raise RuntimeError(
                        f"BST export parity case '{case_name}' produced non-finite scores"
                    )
                case_results.append(
                    {
                        "case": case_name,
                        "shape": [2, 3],
                        "exact_match": True,
                        "finite": True,
                        "event_and_prepooled_exact_match": True,
                    }
                )

            table_rows = eager_model.post_liker_user_pooler.user_embedding.num_embeddings
            lookup_indices = torch.tensor(
                [
                    POST_LIKER_USER_PAD_IDX,
                    POST_LIKER_USER_UNK_IDX,
                    max(table_rows - 1, POST_LIKER_USER_UNK_IDX),
                ],
                dtype=torch.long,
            )
            eager_lookup = eager_model.lookup_post_liker_user_embeddings(
                lookup_indices
            )
            scripted_lookup = scripted_model.lookup_post_liker_user_embeddings(
                lookup_indices
            )
            if not torch.equal(eager_lookup, scripted_lookup):
                raise RuntimeError(
                    "BST exported post-liker user embedding lookup was not exactly equal"
                )
    return {
        "case_count": len(case_results),
        "all_exact": True,
        "cases": case_results,
        **(
            {
                "post_liker_lookup": {
                    "row_count": 3,
                    "exact_match": True,
                    "finite": True,
                }
            }
            if use_post_liker_feature
            else {}
        ),
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


POST_LIKER_USER_MAP_SCHEMA = {
    "liker_did": pl.String,
    "liker_idx": pl.UInt32,
}


def _write_post_liker_user_map(
    *,
    vocabulary_path: Path,
    output_path: Path,
    expected_user_table_num_rows: int,
) -> dict[str, Any]:
    """Publish the Stage 7 vocabulary in the serving map's narrow schema."""

    vocabulary_lf = scan_parquet_artifact(Path(vocabulary_path))
    vocabulary_schema = vocabulary_lf.collect_schema()
    if vocabulary_schema != pl.Schema(POST_LIKER_USER_VOCABULARY_SCHEMA):
        raise ValueError(
            "Stage 7 post-liker user vocabulary has an unexpected schema: "
            f"{vocabulary_schema}"
        )
    user_map_lf = vocabulary_lf.select(list(POST_LIKER_USER_MAP_SCHEMA)).sort(
        "liker_did"
    )
    checks = user_map_lf.select(
        pl.len().alias("user_count"),
        pl.col("liker_did").null_count().alias("null_did_count"),
        pl.col("liker_idx").null_count().alias("null_idx_count"),
        pl.col("liker_did").n_unique().alias("unique_did_count"),
        pl.col("liker_idx").n_unique().alias("unique_idx_count"),
        pl.col("liker_idx").min().alias("min_liker_idx"),
        pl.col("liker_idx").max().alias("max_liker_idx"),
    ).collect(engine="streaming").row(0, named=True)
    user_count = int(checks["user_count"])
    if checks["null_did_count"] or checks["null_idx_count"]:
        raise ValueError("Stage 7 post-liker user vocabulary contains null keys")
    if int(checks["unique_did_count"]) != user_count:
        raise ValueError("Stage 7 post-liker user vocabulary contains duplicate DIDs")
    if int(checks["unique_idx_count"]) != user_count:
        raise ValueError("Stage 7 post-liker user vocabulary contains duplicate indices")
    if expected_user_table_num_rows != user_count + 2:
        raise ValueError(
            "Stage 7 post-liker vocabulary size does not match the checkpoint table"
        )
    if user_count and (
        int(checks["min_liker_idx"]) != 2
        or int(checks["max_liker_idx"]) != expected_user_table_num_rows - 1
    ):
        raise ValueError("Stage 7 post-liker user indices must be dense from 2")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    try:
        user_map_lf.sink_parquet(
            partial_path,
            compression="zstd",
            maintain_order=True,
            engine="streaming",
        )
        if pl.read_parquet_schema(partial_path) != pl.Schema(
            POST_LIKER_USER_MAP_SCHEMA
        ):
            raise ValueError("Published post-liker user map has an unexpected schema")
        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return {
        "user_count": user_count,
        "user_table_num_rows": expected_user_table_num_rows,
        "file_size_bytes": output_path.stat().st_size,
        "sha256": file_sha256(output_path),
    }


def _write_numpy_array_atomically(path: Path, array: np.ndarray) -> None:
    """Write one exact NumPy array without exposing an incomplete final file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f"{path.name}.partial")
    try:
        with partial_path.open("wb") as file_obj:
            np.save(file_obj, array, allow_pickle=False)
        partial_path.replace(path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def export_post_liker_serving_artifacts(
    *,
    checkpoint_path: Path,
    scripted_model_path: Path,
    expected_model_config: Mapping[str, Any],
    expected_popularity_stats: Mapping[str, Any],
    vocabulary_path: Path,
    user_map_output_path: Path,
    embeddings_output_path: Path,
    state_config_output_path: Path,
    max_replay_events_per_post: int,
) -> dict[str, Any]:
    """Export and cross-check the learned liker table and serving state contract.

    The table comes from the selected checkpoint, not the in-memory model left
    by the final evaluation pass. Both the eager checkpoint and the exported
    ScriptModule must return exactly the same rows before anything is published.
    """

    if max_replay_events_per_post <= 0:
        raise ValueError("max_replay_events_per_post must be positive")
    eager_model, best_epoch = _load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        expected_model_config=expected_model_config,
        expected_popularity_stats=expected_popularity_stats,
    )
    if not eager_model.use_post_liker_feature:
        raise ValueError(
            "Post-liker serving artifacts require a feature-enabled BST checkpoint"
        )
    scripted_model_path = Path(scripted_model_path)
    try:
        scripted_model = torch.jit.load(
            str(scripted_model_path),
            map_location="cpu",
        ).eval()
    except Exception as exc:
        raise ValueError(
            f"Failed to load BST TorchScript artifact {scripted_model_path}: {exc}"
        ) from exc

    user_embedding = eager_model.post_liker_user_pooler.user_embedding
    user_table_num_rows = int(user_embedding.num_embeddings)
    user_embedding_dim = int(user_embedding.embedding_dim)
    map_stats = _write_post_liker_user_map(
        vocabulary_path=vocabulary_path,
        output_path=user_map_output_path,
        expected_user_table_num_rows=user_table_num_rows,
    )

    lookup_indices = torch.arange(user_table_num_rows, dtype=torch.long)
    with torch.inference_mode():
        eager_table = eager_model.lookup_post_liker_user_embeddings(
            lookup_indices
        ).detach().cpu()
        scripted_table = scripted_model.lookup_post_liker_user_embeddings(
            lookup_indices
        ).detach().cpu()
    if eager_table.shape != (user_table_num_rows, user_embedding_dim):
        raise RuntimeError("BST checkpoint returned an invalid post-liker table shape")
    if eager_table.dtype != torch.float32:
        raise RuntimeError("BST post-liker user embedding table must be Float32")
    if not bool(torch.isfinite(eager_table).all().item()):
        raise RuntimeError("BST post-liker user embedding table contains non-finite values")
    if not torch.equal(eager_table, scripted_table):
        raise RuntimeError(
            "BST checkpoint and TorchScript post-liker embedding lookups disagree"
        )
    if not torch.equal(
        eager_table[POST_LIKER_USER_PAD_IDX],
        torch.zeros_like(eager_table[POST_LIKER_USER_PAD_IDX]),
    ):
        raise RuntimeError("BST post-liker PAD embedding row must remain zero")

    table_array = eager_table.numpy()
    _write_numpy_array_atomically(embeddings_output_path, table_array)
    reloaded_table = np.load(
        Path(embeddings_output_path),
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        reloaded_table.dtype != np.dtype("<f4")
        or reloaded_table.shape != table_array.shape
        or not np.array_equal(reloaded_table, table_array)
    ):
        raise RuntimeError("Published post-liker user embedding table failed validation")

    table_sha256 = file_sha256(embeddings_output_path)
    state_config = {
        "ranker_contract_version": 2,
        "post_liker_feature_enabled": True,
        "post_liker_user_pad_idx": POST_LIKER_USER_PAD_IDX,
        "post_liker_user_unk_idx": POST_LIKER_USER_UNK_IDX,
        "post_liker_user_table_num_rows": user_table_num_rows,
        "post_liker_user_embedding_dim": user_embedding_dim,
        "post_liker_pooling_tau_hours": float(
            eager_model.post_liker_user_pooler.pooling_tau_hours
        ),
        "max_post_liker_replay_events_per_post": int(
            max_replay_events_per_post
        ),
        "post_liker_user_idx_mapping_filename": Path(user_map_output_path).name,
        "post_liker_user_idx_mapping_sha256": map_stats["sha256"],
        "post_liker_user_embeddings_filename": Path(embeddings_output_path).name,
        "post_liker_user_embeddings_sha256": table_sha256,
        "liker_embedding_model_version": table_sha256,
        "incremental_state": {
            "stored_fields": [
                "pooled_embedding_mean",
                "decayed_weight",
                "reference_timestamp",
                "liker_embedding_model_version",
            ],
            "update_contract": {
                "decay": "exp(-(event_timestamp-reference_timestamp)/tau)",
                "decayed_weight": "decay*decayed_weight+1",
                "pooled_embedding_mean": (
                    "(decay*decayed_weight*pooled_embedding_mean+liker_embedding)"
                    "/updated_decayed_weight"
                ),
            },
            "training_replay_is_bounded_approximation": True,
            "production_recursive_state_may_retain_all_prior_weight": True,
            "out_of_order_events_supported": False,
            "event_deletion_reversal_supported": False,
        },
    }
    write_json_atomically(state_config_output_path, state_config)
    return {
        "best_epoch": best_epoch,
        "user_map_path": str(user_map_output_path),
        "user_map": map_stats,
        "embeddings_path": str(embeddings_output_path),
        "embeddings_shape": [user_table_num_rows, user_embedding_dim],
        "embeddings_dtype": "float32",
        "embeddings_size_bytes": Path(embeddings_output_path).stat().st_size,
        "embeddings_sha256": table_sha256,
        "state_config_path": str(state_config_output_path),
        "state_config_sha256": file_sha256(state_config_output_path),
        "checkpoint_script_lookup_exact_match": True,
    }
