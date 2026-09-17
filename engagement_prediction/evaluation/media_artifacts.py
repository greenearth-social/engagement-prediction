"""Resolve canonical BST runs and their shared validation candidate universe."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from engagement_prediction.data.parquet import scan_parquet_artifact
from engagement_prediction.data.post_liker_users import validate_post_liker_user_vocabulary
from engagement_prediction.data.training_index import (
    MemoryMappedUtf8Table,
    load_index_array,
)
from engagement_prediction.evaluation.artifacts import Stage7Artifact, resolve_stage7_artifact
from engagement_prediction.evaluation.author_mapping import validate_model_author_map
from engagement_prediction.training.model_artifacts import file_sha256


_LINEAGE_STAGES = (
    "00_source_metadata",
    "01_query_selection",
    "02_user_history",
    "03_post_selection",
    "04_negative_selection",
    "05_post_liker_history",
    "06_author_statistics",
    "07_dataset_hydration",
)
_VALIDATION_SPLITS = ("val", "val_unseen_users")


@dataclass(frozen=True)
class ValidationSettings:
    """Saved settings that determine identical validation candidate slates."""

    batch_size: int
    additional_batch_negatives: int | None
    random_seed: int


@dataclass(frozen=True)
class MediaModelArtifact:
    """One canonical BST checkpoint and its original reproducibility files."""

    name: str
    root: Path
    manifest: dict[str, Any]
    model_config: dict[str, Any]
    training_config: dict[str, Any]
    popularity_stats: dict[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible model provenance without loading weights."""

        return {
            "name": self.name,
            "root": str(self.root),
            "manifest": self.manifest,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "popularity_stats": self.popularity_stats,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
        }


def _object(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read required artifact {path}: {exc}") from exc
    return _object(value, str(path))


def _recorded_path(value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must record a nonempty path")
    return Path(value).expanduser().resolve()


def _integer(value: Any, description: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{description} must be an integer >= {minimum}")
    return value


def _lineage(value: Any, description: str, stages: Sequence[str]) -> dict[str, Path]:
    mapping = _object(value, description)
    if set(mapping) != set(stages):
        raise ValueError(f"{description} must record exactly the expected input stages")
    return {stage: _recorded_path(mapping[stage], f"{description}.{stage}") for stage in stages}


def _normalized_training_config(value: Any, description: str) -> dict[str, Any]:
    config = dict(_object(value, description))
    for key in ("stage7_dir", "stage7_bundle"):
        config[key] = _recorded_path(config.get(key), f"{description}.{key}")
    config["lineage"] = _lineage(config.get("lineage"), f"{description}.lineage", _LINEAGE_STAGES)
    return config


def _resolve_model(name: str, path: Path) -> tuple[MediaModelArtifact, ValidationSettings, dict[str, Path]]:
    root = Path(path).expanduser().resolve()
    if not root.is_dir() or root.name.endswith(".partial"):
        raise ValueError(f"Model artifact must be a completed directory: {root}")
    manifest = _read_json(root / "manifest.json")
    expected_manifest = {
        "stage_key": "train_bst_ranker",
        "stage_folder": "08_train_bst_ranker",
        "stage_run_id": root.name,
        "status": "complete",
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Model {name!r} manifest {key} must be {expected!r}")
    lineage = _lineage(manifest.get("inputs"), f"Model {name!r} manifest inputs", _LINEAGE_STAGES)
    model_config = _read_json(root / "model_config.json")
    training_config = _read_json(root / "training_config.json")
    popularity_stats = _read_json(root / "popularity_stats.json")
    summary = _read_json(root / "summary.json")
    normalized_config = _normalized_training_config(training_config, "training_config")
    if normalized_config["lineage"] != lineage:
        raise ValueError(f"Model {name!r} manifest and training_config lineage disagree")
    stage7_dir = normalized_config["stage7_dir"]
    bundle = normalized_config["stage7_bundle"]
    if stage7_dir != lineage["07_dataset_hydration"]:
        raise ValueError(f"Model {name!r} stage7_dir disagrees with its lineage")
    if bundle.parent != stage7_dir or not bundle.name.startswith("hydrated_training_data_") or bundle.name.endswith(".partial"):
        raise ValueError(f"Model {name!r} records an invalid Stage 7 bundle")
    summary_input = _object(summary.get("input"), "summary.input")
    for key, expected in (("dataset_hydration_dir", stage7_dir), ("hydrated_training_data_path", bundle)):
        if _recorded_path(summary_input.get(key), f"summary.input.{key}") != expected:
            raise ValueError(f"Model {name!r} summary input disagrees with training_config")
    if _normalized_training_config(summary.get("parameters"), "summary.parameters") != normalized_config:
        raise ValueError(f"Model {name!r} summary parameters disagree with training_config")
    if summary.get("model") != model_config or summary.get("popularity") != popularity_stats:
        raise ValueError(f"Model {name!r} summary model/popularity disagrees with saved configuration")
    if model_config.get("model_type") != "bst-ranker":
        raise ValueError(f"Model {name!r} must be a canonical BST ranker")
    _integer(model_config.get("max_history_len"), "max_history_len", 1)
    constructor = _object(model_config.get("constructor_args"), "model_config.constructor_args")
    if constructor.get("num_transformer_layers") != 1:
        raise ValueError("Canonical BST requires exactly one transformer layer")
    for flag in ("use_popularity_feature", "use_post_liker_feature"):
        if not isinstance(constructor.get(flag), bool):
            raise ValueError(f"model_config.constructor_args.{flag} must be a boolean")
    if training_config.get("bst_use_post_liker_feature") is not constructor["use_post_liker_feature"]:
        raise ValueError(f"Model {name!r} training_config and model post-liker feature flags disagree")
    if popularity_stats.get("enabled") is not constructor["use_popularity_feature"]:
        raise ValueError(f"Model {name!r} popularity configuration disagrees with its feature flag")
    for argument, stat in (("popularity_log_mean", "log_mean"), ("popularity_log_std", "log_std")):
        if constructor.get(argument) != popularity_stats.get(stat):
            raise ValueError(f"Model {name!r} {argument} disagrees with popularity_stats")
    if constructor["use_post_liker_feature"]:
        _integer(training_config.get("bst_max_post_liker_replay_events_per_post"), "bst_max_post_liker_replay_events_per_post", 1)
    cap = training_config.get("bst_additional_batch_negatives")
    if "bst_additional_batch_negatives" not in training_config:
        raise ValueError("training_config must record bst_additional_batch_negatives")
    if cap is not None:
        _integer(cap, "bst_additional_batch_negatives", 1)
    settings = ValidationSettings(
        batch_size=_integer(training_config.get("eval_batch_size"), "eval_batch_size", 1),
        additional_batch_negatives=cap,
        random_seed=_integer(training_config.get("random_seed"), "random_seed", 0),
    )
    checkpoint_path = root / "checkpoints" / "bst_ranker_best.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model {name!r} best checkpoint is missing: {checkpoint_path}")
    outputs = _object(summary.get("outputs"), "summary.outputs")
    if _recorded_path(outputs.get("checkpoint_path"), "summary.outputs.checkpoint_path") != checkpoint_path:
        raise ValueError(f"Model {name!r} summary records a different best checkpoint")
    return MediaModelArtifact(
        name=name,
        root=root,
        manifest=manifest,
        model_config=model_config,
        training_config=training_config,
        popularity_stats=popularity_stats,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=file_sha256(checkpoint_path),
    ), settings, lineage


def _mapping(path: Path, columns: Sequence[str]) -> pl.DataFrame:
    return scan_parquet_artifact(path).select(list(columns)).sort(columns[0]).collect(engine="streaming")


def _validate_dataset_contract(dataset: Stage7Artifact, model: MediaModelArtifact) -> None:
    config = model.model_config
    constructor = config["constructor_args"]
    index = dataset.loader_index_validation
    if config.get("embedding_model") != dataset.embedding_model or constructor.get("post_embedding_dim") != dataset.embedding_dim:
        raise ValueError(f"Model {model.name!r} embedding contract differs from Stage 7")
    if model.training_config.get("loader_index_format_version") != index["format_version"]:
        raise ValueError(f"Model {model.name!r} loader-index version differs from Stage 7")
    if config.get("author_pad_idx") != 0 or config.get("author_unk_idx") != 1:
        raise ValueError("BST author vocabulary must reserve PAD=0 and UNK=1")
    author_rows = _integer(constructor.get("author_table_num_rows"), "author_table_num_rows", 2)
    if author_rows != index["author_table_num_rows"]:
        raise ValueError(f"Model {model.name!r} author table size differs from Stage 7")
    author_map_path = model.root / "ranker_author_idx.parquet"
    validate_model_author_map(author_map_path, author_table_num_rows=author_rows)
    author_columns = ("author_did", "author_idx")
    if not _mapping(author_map_path, author_columns).equals(_mapping(dataset.bundle_path / "authors", author_columns)):
        raise ValueError(f"Model {model.name!r} author vocabulary differs from Stage 7")
    if not constructor["use_post_liker_feature"]:
        return
    if index["format_version"] < 2:
        raise ValueError("Post-liker BST models require Stage 7 loader-index version >= 2")
    if config.get("post_liker_user_pad_idx") != 0 or config.get("post_liker_user_unk_idx") != 1:
        raise ValueError("BST post-liker vocabulary must reserve PAD=0 and UNK=1")
    user_rows = _integer(constructor.get("post_liker_user_table_num_rows"), "post_liker_user_table_num_rows", 2)
    if user_rows != index["post_liker_user_table_num_rows"]:
        raise ValueError(f"Model {model.name!r} post-liker table size differs from Stage 7")
    parameters = dataset.summary["parameters"]
    vocabulary_path = dataset.bundle_path / "post_liker_users"
    vocabulary = validate_post_liker_user_vocabulary(
        scan_parquet_artifact(vocabulary_path),
        min_training_event_count=_integer(parameters.get("min_post_liker_user_training_event_count"), "min_post_liker_user_training_event_count", 1),
        max_vocabulary_size=_integer(parameters.get("max_post_liker_user_vocabulary_size"), "max_post_liker_user_vocabulary_size", 0),
    )
    if vocabulary["user_table_num_rows"] != user_rows:
        raise ValueError("Stage 7 post-liker vocabulary and index table sizes disagree")
    user_columns = ("liker_did", "liker_idx")
    if not _mapping(model.root / "post_liker_users", user_columns).equals(_mapping(vocabulary_path, user_columns)):
        raise ValueError(f"Model {model.name!r} post-liker vocabulary differs from Stage 7")


def resolve_media_artifacts(
    model_specs: Sequence[tuple[str, Path]],
) -> tuple[Stage7Artifact, tuple[MediaModelArtifact, ...], ValidationSettings]:
    """Require canonical BST runs with the same recorded data and validation slates."""

    if not model_specs:
        raise ValueError("At least one BST model artifact is required")
    names = [name.strip() if isinstance(name, str) else "" for name, _ in model_specs]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Model names must be nonempty and unique")
    resolved = [_resolve_model(name, path) for name, (_, path) in zip(names, model_specs, strict=True)]
    first, settings, lineage = resolved[0]
    bundle_path = _recorded_path(first.training_config["stage7_bundle"], "stage7_bundle")
    for model, model_settings, model_lineage in resolved[1:]:
        differences = [
            f"{stage}: {first.name}={lineage[stage]}, {model.name}={model_lineage[stage]}"
            for stage in _LINEAGE_STAGES
            if model_lineage[stage] != lineage[stage]
        ]
        model_bundle = _recorded_path(model.training_config["stage7_bundle"], "stage7_bundle")
        if model_bundle != bundle_path:
            differences.append(f"stage7_bundle: {first.name}={bundle_path}, {model.name}={model_bundle}")
        if differences:
            raise ValueError(f"Model {model.name!r} input data differs from model {first.name!r}: " + "; ".join(differences))
        if model_settings != settings:
            differences = [
                f"{field}: {first.name}={getattr(settings, field)!r}, {model.name}={getattr(model_settings, field)!r}"
                for field in ("batch_size", "additional_batch_negatives", "random_seed")
                if getattr(settings, field) != getattr(model_settings, field)
            ]
            raise ValueError(f"Model {model.name!r} saved validation settings differ from model {first.name!r}: " + "; ".join(differences))
    dataset = resolve_stage7_artifact(bundle_path)
    ancestors = _lineage(dataset.manifest.get("inputs"), "Stage 7 manifest inputs", _LINEAGE_STAGES[:-1])
    if ancestors != {stage: lineage[stage] for stage in _LINEAGE_STAGES[:-1]}:
        raise ValueError("Stage 7 manifest lineage differs from the BST model lineage")
    for split in _VALIDATION_SPLITS:
        if dataset.split_query_counts.get(split, 0) == 0:
            raise ValueError(f"Stage 7 validation split {split!r} is empty")
    for model, _, _ in resolved:
        _validate_dataset_contract(dataset, model)
    return dataset, tuple(model for model, _, _ in resolved), settings


def collect_candidate_uris(dataset: Stage7Artifact) -> pl.DataFrame:
    """Decode the positive/negative URI union without hydrating history features."""

    metadata = dataset.loader_index_validation["metadata"]
    memberships = np.zeros((2, dataset.embedding_count), dtype=np.bool_)
    for split_index, split in enumerate(_VALIDATION_SPLITS):
        for name in ("positive_emb_indices", "negative_emb_indices"):
            indices = load_index_array(dataset.loader_index_path, name, split, metadata=metadata)
            memberships[split_index, indices] = True
            del indices
    selected = np.flatnonzero(memberships.any(axis=0))
    uri_metadata = metadata["arrow_tables"]["post_uris"]
    with MemoryMappedUtf8Table(
        dataset.loader_index_path / uri_metadata["path"],
        batch_offsets=uri_metadata["batch_offsets"],
    ) as table:
        uris = table.take(selected)
    return pl.DataFrame({
        "at_uri": pl.Series(uris, dtype=pl.String),
        "in_val": memberships[0, selected],
        "in_val_unseen_users": memberships[1, selected],
    }).sort("at_uri")
