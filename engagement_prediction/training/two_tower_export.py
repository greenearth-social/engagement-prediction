"""Serving-safe TorchScript export for canonical two-tower checkpoints."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import torch

from engagement_prediction.models.two_tower import TwoTowerModel


_MODEL_TYPE = "two-tower"


def _require_mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    return value


def _load_checkpoint_model(
    *,
    checkpoint_path: Path,
    expected_model_config: Mapping[str, Any],
) -> tuple[TwoTowerModel, int]:
    """Reconstruct the exact CPU model described by a canonical checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Two-tower checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to load two-tower checkpoint {checkpoint_path}: {exc}"
        ) from exc

    checkpoint = _require_mapping(checkpoint, description="Two-tower checkpoint")
    metadata = _require_mapping(
        checkpoint.get("metadata"),
        description="Two-tower checkpoint metadata",
    )
    checkpoint_model_config = _require_mapping(
        metadata.get("model_config"),
        description="Two-tower checkpoint model_config",
    )
    if dict(checkpoint_model_config) != dict(expected_model_config):
        raise ValueError(
            "Two-tower checkpoint model_config does not match the expected model configuration"
        )
    if checkpoint_model_config.get("model_type") != _MODEL_TYPE:
        raise ValueError(
            f"Two-tower checkpoint model_type must be '{_MODEL_TYPE}'"
        )
    if checkpoint_model_config.get("user_encoder_type") != "cross_attention":
        raise ValueError("Canonical two-tower checkpoints must use cross_attention")

    constructor_args = _require_mapping(
        checkpoint_model_config.get("constructor_args"),
        description="Two-tower checkpoint constructor_args",
    )
    output_embedding_dim = checkpoint_model_config.get("output_embedding_dim")
    if constructor_args.get("output_embedding_dim") != output_embedding_dim:
        raise ValueError(
            "Two-tower output_embedding_dim disagrees with constructor_args"
        )
    if checkpoint.get("output_embedding_dim") != output_embedding_dim:
        raise ValueError(
            "Two-tower checkpoint output_embedding_dim disagrees with model_config"
        )

    best_epoch = checkpoint.get("best_epoch")
    if not isinstance(best_epoch, int) or isinstance(best_epoch, bool) or best_epoch < 1:
        raise ValueError("Two-tower checkpoint best_epoch must be a positive integer")
    if checkpoint.get("epoch") != best_epoch:
        raise ValueError("Two-tower checkpoint epoch must match best_epoch")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Two-tower checkpoint model_state_dict must be a mapping")
    try:
        model = TwoTowerModel(**dict(constructor_args))
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ValueError(
            f"Two-tower checkpoint cannot reconstruct its configured model: {exc}"
        ) from exc
    return model.cpu().eval(), best_epoch


def _parity_inputs(
    model: TwoTowerModel,
    *,
    all_masked: bool,
) -> tuple[torch.Tensor, ...]:
    users = 2
    candidates = 3
    history_len = min(3, model.user_tower.history_encoder.max_history_len)
    post_embedding_dim = model.post_embedding_dim
    author_table_num_rows = (
        model.post_feature_encoder.author_embedding.num_embeddings
    )
    history_embeddings = torch.arange(
        users * history_len * post_embedding_dim,
        dtype=torch.float32,
    ).reshape(users, history_len, post_embedding_dim)
    history_embeddings = history_embeddings / float(post_embedding_dim + 7)
    history_mask = torch.ones((users, history_len), dtype=torch.bool)
    if history_len > 1:
        history_mask[0, -1] = False
    if all_masked:
        history_mask.zero_()
    history_author_indices = torch.arange(
        users * history_len,
        dtype=torch.long,
    ).reshape(users, history_len).remainder(author_table_num_rows)
    post_embeddings = torch.arange(
        candidates * post_embedding_dim,
        dtype=torch.float32,
    ).reshape(candidates, post_embedding_dim)
    post_embeddings = post_embeddings / float(post_embedding_dim + 11)
    post_author_indices = torch.arange(
        candidates,
        dtype=torch.long,
    ).remainder(author_table_num_rows)
    return (
        history_embeddings,
        history_mask,
        history_author_indices,
        post_embeddings,
        post_author_indices,
    )


def _validate_parity(
    *,
    eager_model: TwoTowerModel,
    scripted_user_tower: torch.jit.ScriptModule,
    scripted_post_tower: torch.jit.ScriptModule,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    with torch.inference_mode():
        for case_name, all_masked in (
            ("ordinary", False),
            ("all_masked", True),
        ):
            (
                history_embeddings,
                history_mask,
                history_author_indices,
                post_embeddings,
                post_author_indices,
            ) = _parity_inputs(eager_model, all_masked=all_masked)
            eager_users = eager_model.encode_user(
                history_embeddings,
                history_mask,
                history_author_indices,
            )
            eager_posts = eager_model.encode_post(
                post_embeddings,
                post_author_indices,
            )
            scripted_users = scripted_user_tower(
                history_embeddings,
                history_mask,
                history_author_indices,
            )
            scripted_posts = scripted_post_tower(
                post_embeddings,
                post_author_indices,
            )
            expected_user_shape = (2, eager_model.output_embedding_dim)
            expected_post_shape = (3, eager_model.output_embedding_dim)
            if eager_users.shape != expected_user_shape:
                raise RuntimeError(
                    f"Two-tower export case '{case_name}' returned unexpected user shape"
                )
            if eager_posts.shape != expected_post_shape:
                raise RuntimeError(
                    f"Two-tower export case '{case_name}' returned unexpected post shape"
                )
            tensors = (eager_users, eager_posts, scripted_users, scripted_posts)
            if not all(bool(torch.isfinite(value).all().item()) for value in tensors):
                raise RuntimeError(
                    f"Two-tower export case '{case_name}' produced non-finite outputs"
                )
            if not torch.equal(eager_users, scripted_users):
                raise RuntimeError(
                    f"Two-tower export case '{case_name}' user outputs were not exactly equal"
                )
            if not torch.equal(eager_posts, scripted_posts):
                raise RuntimeError(
                    f"Two-tower export case '{case_name}' post outputs were not exactly equal"
                )
            for name, value in (
                ("eager users", eager_users),
                ("eager posts", eager_posts),
                ("scripted users", scripted_users),
                ("scripted posts", scripted_posts),
            ):
                if not torch.allclose(
                    value.norm(dim=-1),
                    torch.ones(value.size(0)),
                    atol=1.0e-6,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        f"Two-tower export case '{case_name}' {name} were not unit normalized"
                    )
            eager_scores = (
                eager_users @ eager_posts.transpose(0, 1)
            ) / eager_model.similarity_temperature
            scripted_scores = (
                scripted_users @ scripted_posts.transpose(0, 1)
            ) / eager_model.similarity_temperature
            if not torch.equal(eager_scores, scripted_scores):
                raise RuntimeError(
                    f"Two-tower export case '{case_name}' combined scores were not exactly equal"
                )
            cases.append({
                "case": case_name,
                "user_shape": list(expected_user_shape),
                "post_shape": list(expected_post_shape),
                "exact_match": True,
                "finite": True,
                "unit_normalized": True,
            })
    return {"case_count": len(cases), "all_exact": True, "cases": cases}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_tower_pair_with_rollback(
    *,
    user_partial_path: Path,
    user_tower_path: Path,
    post_partial_path: Path,
    post_tower_path: Path,
) -> None:
    """Replace both serving files while restoring the prior pair on failure.

    A filesystem cannot atomically rename two independent files together. Keep
    the previously validated pair as rollback backups until both new files are
    installed so a failed second rename never leaves mixed checkpoint epochs.
    """

    replacements = (
        (Path(user_partial_path), Path(user_tower_path)),
        (Path(post_partial_path), Path(post_tower_path)),
    )
    states = []
    for _partial_path, final_path in replacements:
        backup_path = final_path.with_name(f"{final_path.name}.previous")
        if backup_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite stale two-tower rollback file: {backup_path}"
            )
        states.append((final_path, backup_path, final_path.exists()))

    try:
        for final_path, backup_path, existed in states:
            if existed:
                final_path.replace(backup_path)
        for partial_path, final_path in replacements:
            partial_path.replace(final_path)
    except Exception:
        rollback_errors = []
        for final_path, backup_path, existed in states:
            try:
                if backup_path.exists():
                    final_path.unlink(missing_ok=True)
                    backup_path.replace(final_path)
                elif not existed:
                    final_path.unlink(missing_ok=True)
            except Exception as rollback_exc:
                rollback_errors.append(f"{final_path}: {rollback_exc}")
        if rollback_errors:
            raise RuntimeError(
                "Two-tower publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            )
        raise
    else:
        for _final_path, backup_path, _existed in states:
            backup_path.unlink(missing_ok=True)


def validate_two_tower_export(
    *,
    checkpoint_path: Path,
    user_tower_path: Path,
    post_tower_path: Path,
    expected_model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that separately saved towers exactly represent one checkpoint."""

    eager_model, best_epoch = _load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        expected_model_config=expected_model_config,
    )
    user_tower_path = Path(user_tower_path)
    post_tower_path = Path(post_tower_path)
    if not user_tower_path.is_file():
        raise FileNotFoundError(f"User tower does not exist: {user_tower_path}")
    if not post_tower_path.is_file():
        raise FileNotFoundError(f"Post tower does not exist: {post_tower_path}")
    try:
        scripted_user_tower = torch.jit.load(
            str(user_tower_path), map_location="cpu"
        ).eval()
        scripted_post_tower = torch.jit.load(
            str(post_tower_path), map_location="cpu"
        ).eval()
    except Exception as exc:
        raise ValueError(f"Failed to load two-tower TorchScript artifacts: {exc}") from exc
    parity = _validate_parity(
        eager_model=eager_model,
        scripted_user_tower=scripted_user_tower,
        scripted_post_tower=scripted_post_tower,
    )
    return {
        "best_epoch": best_epoch,
        "output_embedding_dim": eager_model.output_embedding_dim,
        "parity": parity,
        "user_tower": {
            "size_bytes": user_tower_path.stat().st_size,
            "sha256": _file_sha256(user_tower_path),
        },
        "post_tower": {
            "size_bytes": post_tower_path.stat().st_size,
            "sha256": _file_sha256(post_tower_path),
        },
    }


def export_two_tower_checkpoint(
    *,
    checkpoint_path: Path,
    user_tower_path: Path,
    post_tower_path: Path,
    expected_model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Script, validate, and atomically publish both canonical towers."""

    eager_model, _ = _load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        expected_model_config=expected_model_config,
    )
    user_tower_path = Path(user_tower_path)
    post_tower_path = Path(post_tower_path)
    user_tower_path.parent.mkdir(parents=True, exist_ok=True)
    post_tower_path.parent.mkdir(parents=True, exist_ok=True)
    user_partial_path = user_tower_path.with_name(f"{user_tower_path.name}.partial")
    post_partial_path = post_tower_path.with_name(f"{post_tower_path.name}.partial")
    try:
        torch.jit.script(eager_model.user_tower).save(str(user_partial_path))
        torch.jit.script(eager_model.post_tower).save(str(post_partial_path))
        validation = validate_two_tower_export(
            checkpoint_path=checkpoint_path,
            user_tower_path=user_partial_path,
            post_tower_path=post_partial_path,
            expected_model_config=expected_model_config,
        )
        _replace_tower_pair_with_rollback(
            user_partial_path=user_partial_path,
            user_tower_path=user_tower_path,
            post_partial_path=post_partial_path,
            post_tower_path=post_tower_path,
        )
    except Exception:
        user_partial_path.unlink(missing_ok=True)
        post_partial_path.unlink(missing_ok=True)
        raise
    return {
        "user_tower_path": str(user_tower_path),
        "post_tower_path": str(post_tower_path),
        **validation,
    }
