"""Resolve canonical Stage 7 data and supported model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from engagement_prediction.data.author_indices import (
    AUTHOR_PAD_IDX,
    AUTHOR_UNK_IDX,
)
from engagement_prediction.data.training_index import (
    FORMAT_VERSION,
    validate_loader_index,
)
from engagement_prediction.evaluation.author_mapping import validate_model_author_map
from engagement_prediction.training.model_artifacts import file_sha256


BST_MODEL_TYPE = "bst-ranker"
TWO_TOWER_MODEL_TYPE = "two-tower"
SUPPORTED_MODEL_TYPES = frozenset({BST_MODEL_TYPE, TWO_TOWER_MODEL_TYPE})
CANONICAL_ARTIFACT_FORMAT = "canonical_stage8"
LEGACY_BST_ARTIFACT_FORMAT = "legacy_stage3_bst"

_BST_SCORE_ARGUMENT_NAMES = (
    "history_embeddings",
    "history_mask",
    "history_time_deltas_hours",
    "candidate_post_embeddings",
    "history_author_indices",
    "candidate_post_author_idx",
    "history_prior_cumulative_likes",
    "candidate_prior_cumulative_likes",
)


@dataclass(frozen=True)
class Stage7Artifact:
    """A validated completed Stage 7 run and its public data bundle."""

    root: Path
    bundle_path: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    loader_index_validation: dict[str, Any]
    embedding_model: str
    embedding_dim: int

    @property
    def loader_index_path(self) -> Path:
        return self.bundle_path / "loader_index"

    @property
    def embedding_count(self) -> int:
        return int(self.loader_index_validation["embedding_count"])

    @property
    def split_query_counts(self) -> dict[str, int]:
        return {
            split: int(counts["query_count"])
            for split, counts in self.loader_index_validation["splits"].items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "bundle_path": str(self.bundle_path),
            "stage_run_id": self.manifest["stage_run_id"],
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "embedding_count": self.embedding_count,
            "loader_index_format_version": int(
                self.loader_index_validation["format_version"]
            ),
            "split_query_counts": self.split_query_counts,
        }


@dataclass(frozen=True)
class ModelArtifact:
    """A validated completed model artifact supported by comparison."""

    name: str
    root: Path
    artifact_format: str
    model_type: str
    manifest: dict[str, Any]
    model_config: dict[str, Any]
    training_config: dict[str, Any]
    script_paths: dict[str, Path]
    script_sha256: dict[str, str]
    author_map_path: Path
    author_map_stats: dict[str, int]
    author_map_allow_extra_columns: bool
    embedding_model: str
    embedding_dim: int
    max_history_len: int
    author_table_num_rows: int
    similarity_temperature: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": str(self.root),
            "stage_run_id": self.manifest["stage_run_id"],
            "artifact_format": self.artifact_format,
            "model_type": self.model_type,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "max_history_len": self.max_history_len,
            "author_table_num_rows": self.author_table_num_rows,
            "similarity_temperature": self.similarity_temperature,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "scripts": {
                key: {
                    "path": str(path),
                    "sha256": self.script_sha256[key],
                    "size_bytes": path.stat().st_size,
                }
                for key, path in self.script_paths.items()
            },
            "author_map": {
                "path": str(self.author_map_path),
                "allow_extra_columns": self.author_map_allow_extra_columns,
                **self.author_map_stats,
            },
        }


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        value = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Could not read {description} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description.capitalize()} at {path} must be a JSON object")
    return value


def _validate_completed_manifest(
    root: Path,
    *,
    expected_stage_key: str,
    expected_stage_folder: str,
) -> dict[str, Any]:
    manifest = _load_json_object(root / "manifest.json", description="completed manifest")
    if "status" in manifest and manifest["status"] != "complete":
        raise ValueError(f"Stage manifest is not complete: {root / 'manifest.json'}")
    expected_values = {
        "stage_key": expected_stage_key,
        "stage_folder": expected_stage_folder,
        "stage_run_id": root.name,
    }
    for key, expected in expected_values.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Stage manifest {key} must be {expected!r}, got {manifest.get(key)!r}: "
                f"{root / 'manifest.json'}"
            )
    if not isinstance(manifest.get("inputs"), dict):
        raise ValueError(f"Stage manifest inputs must be a JSON object: {root / 'manifest.json'}")
    return manifest


def _positive_int(value: Any, *, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return value


def _require_mapping(
    value: Any,
    *,
    description: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _require_nonempty_string(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")
    return value


def _validate_torchscript_method(
    path: Path,
    *,
    method_name: str,
    expected_argument_names: Sequence[str],
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing TorchScript artifact: {path}")
    try:
        module = torch.jit.load(str(path), map_location="cpu")
    except Exception as exc:
        raise ValueError(f"Could not load TorchScript artifact {path}: {exc}") from exc
    try:
        method = getattr(module, method_name)
    except AttributeError as exc:
        raise ValueError(
            f"TorchScript artifact {path} does not export {method_name}"
        ) from exc
    schema = getattr(method, "schema", None)
    if schema is None:
        raise ValueError(f"TorchScript artifact {path} has no schema for {method_name}")
    actual_names = [argument.name for argument in schema.arguments]
    if actual_names and actual_names[0] == "self":
        actual_names = actual_names[1:]
    if actual_names != list(expected_argument_names):
        raise ValueError(
            f"TorchScript artifact {path} has an unexpected {method_name} signature: "
            f"{actual_names}"
        )
    del module


def resolve_stage7_artifact(path: Path) -> Stage7Artifact:
    """Resolve either a Stage 7 run directory or its hydrated bundle."""

    requested_path = Path(path).expanduser().resolve()
    if not requested_path.is_dir():
        raise FileNotFoundError(f"Stage 7 artifact directory does not exist: {requested_path}")
    if requested_path.name.startswith("hydrated_training_data_"):
        if requested_path.name.endswith(".partial"):
            raise ValueError(f"Stage 7 bundle is incomplete: {requested_path}")
        root = requested_path.parent
        bundle_path = requested_path
    else:
        root = requested_path
        candidates = sorted(
            candidate
            for candidate in root.glob("hydrated_training_data_*")
            if candidate.is_dir() and not candidate.name.endswith(".partial")
        )
        if len(candidates) != 1:
            raise ValueError(
                f"Stage 7 directory must contain exactly one completed "
                f"hydrated_training_data_* bundle, found {len(candidates)}: {root}"
            )
        bundle_path = candidates[0].resolve()

    manifest = _validate_completed_manifest(
        root,
        expected_stage_key="dataset_hydration",
        expected_stage_folder="07_dataset_hydration",
    )
    summary = _load_json_object(root / "summary.json", description="Stage 7 summary")
    parameters = _require_mapping(
        summary.get("parameters"), description="Stage 7 summary parameters"
    )
    embedding_model = _require_nonempty_string(
        parameters.get("embedding_model"),
        description="Stage 7 embedding_model",
    )
    embedding_dim = _positive_int(
        parameters.get("embedding_dim"),
        description="Stage 7 embedding_dim",
    )
    try:
        loader_validation = validate_loader_index(bundle_path / "loader_index")
    except Exception as exc:
        raise ValueError(f"Invalid Stage 7 loader index in {bundle_path}: {exc}") from exc
    if int(loader_validation["embedding_dim"]) != embedding_dim:
        raise ValueError("Stage 7 summary and loader index embedding dimensions disagree")
    summary_output = _require_mapping(
        summary.get("outputs"), description="Stage 7 summary outputs"
    ).get("hydrated_training_data_path")
    if summary_output != bundle_path.name:
        raise ValueError(
            "Stage 7 summary does not identify the resolved hydrated training bundle"
        )
    return Stage7Artifact(
        root=root,
        bundle_path=bundle_path,
        manifest=manifest,
        summary=summary,
        loader_index_validation=loader_validation,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )


def _validate_model_config(
    model_config: dict[str, Any],
) -> tuple[str, str, int, int, int, float | None]:
    model_type = model_config.get("model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"model_config.json model_type must be one of "
            f"{sorted(SUPPORTED_MODEL_TYPES)}, got {model_type!r}"
        )
    embedding_model = _require_nonempty_string(
        model_config.get("embedding_model"),
        description="model_config embedding_model",
    )
    max_history_len = _positive_int(
        model_config.get("max_history_len"),
        description="model_config max_history_len",
    )
    if model_config.get("author_pad_idx") != AUTHOR_PAD_IDX:
        raise ValueError("model_config must reserve author PAD=0")
    if model_config.get("author_unk_idx") != AUTHOR_UNK_IDX:
        raise ValueError("model_config must reserve author UNK=1")
    constructor = _require_mapping(
        model_config.get("constructor_args"),
        description="model_config constructor_args",
    )
    embedding_dim = _positive_int(
        constructor.get("post_embedding_dim"),
        description="model_config constructor post_embedding_dim",
    )
    author_table_num_rows = _positive_int(
        constructor.get("author_table_num_rows"),
        description="model_config constructor author_table_num_rows",
    )
    if author_table_num_rows < 2:
        raise ValueError("Model author table must reserve PAD=0 and UNK=1")

    similarity_temperature: float | None = None
    if model_type == BST_MODEL_TYPE:
        if constructor.get("num_transformer_layers") != 1:
            raise ValueError("Canonical BST model must contain exactly one transformer layer")
    else:
        if model_config.get("user_encoder_type") != "cross_attention":
            raise ValueError("Canonical two-tower model must use cross_attention")
        for key in (
            "use_author_embedding_table",
            "use_post_encoder",
            "l2_normalize_embeddings",
        ):
            if model_config.get(key) is not True:
                raise ValueError(f"Canonical two-tower model requires {key}=true")
        if constructor.get("max_history_len") != max_history_len:
            raise ValueError(
                "Two-tower constructor max_history_len disagrees with model_config"
            )
        output_embedding_dim = _positive_int(
            model_config.get("output_embedding_dim"),
            description="two-tower output_embedding_dim",
        )
        if constructor.get("output_embedding_dim") != output_embedding_dim:
            raise ValueError(
                "Two-tower constructor output_embedding_dim disagrees with model_config"
            )
        try:
            similarity_temperature = float(constructor["similarity_temperature"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Two-tower similarity_temperature must be positive") from exc
        if not math.isfinite(similarity_temperature) or similarity_temperature <= 0.0:
            raise ValueError("Two-tower similarity_temperature must be positive")
    return (
        model_type,
        embedding_model,
        embedding_dim,
        max_history_len,
        author_table_num_rows,
        similarity_temperature,
    )


def _resolve_canonical_model_artifact(
    *,
    name: str,
    root: Path,
) -> ModelArtifact:
    """Resolve the canonical Stage 8 contract without legacy concessions."""

    model_config = _load_json_object(
        root / "model_config.json", description="Stage 8 model configuration"
    )
    (
        model_type,
        embedding_model,
        embedding_dim,
        max_history_len,
        author_table_num_rows,
        similarity_temperature,
    ) = _validate_model_config(model_config)
    if model_type == BST_MODEL_TYPE:
        expected_stage_key = "train_bst_ranker"
        expected_stage_folder = "08_train_bst_ranker"
        script_paths = {"ranker": root / "checkpoints" / "ranker.pt"}
        author_map_path = root / "ranker_author_idx.parquet"
        script_methods = {
            "ranker": (
                "score_candidate_matrix",
                _BST_SCORE_ARGUMENT_NAMES,
            )
        }
    else:
        expected_stage_key = "train_two_tower"
        expected_stage_folder = "08_train_two_tower"
        script_paths = {
            "user_tower": root / "checkpoints" / "engagement_user_tower.pt",
            "post_tower": root / "checkpoints" / "engagement_post_tower.pt",
        }
        author_map_path = root / "two_tower_author_idx.parquet"
        script_methods = {
            "user_tower": (
                "forward",
                ("history_embeddings", "history_mask", "history_author_indices"),
            ),
            "post_tower": (
                "forward",
                ("post_embeddings", "post_author_indices"),
            ),
        }

    manifest = _validate_completed_manifest(
        root,
        expected_stage_key=expected_stage_key,
        expected_stage_folder=expected_stage_folder,
    )
    training_config = _load_json_object(
        root / "training_config.json", description="Stage 8 training configuration"
    )
    format_version = training_config.get("loader_index_format_version")
    if format_version != FORMAT_VERSION:
        raise ValueError(
            "Model training configuration uses an unsupported Stage 7 loader-index version"
        )
    stage7_dir = _require_nonempty_string(
        training_config.get("stage7_dir"),
        description="training_config stage7_dir",
    )
    stage7_bundle = _require_nonempty_string(
        training_config.get("stage7_bundle"),
        description="training_config stage7_bundle",
    )
    recorded_stage7 = manifest["inputs"].get("07_dataset_hydration")
    if not isinstance(recorded_stage7, str) or not recorded_stage7.strip():
        raise ValueError("Stage 8 manifest is missing its Stage 7 input")
    if Path(recorded_stage7).expanduser().resolve() != Path(stage7_dir).expanduser().resolve():
        raise ValueError("Stage 8 manifest and training configuration disagree on Stage 7 input")
    recorded_bundle_path = Path(stage7_bundle).expanduser().resolve()
    if (
        recorded_bundle_path.parent != Path(stage7_dir).expanduser().resolve()
        or not recorded_bundle_path.name.startswith("hydrated_training_data_")
        or recorded_bundle_path.name.endswith(".partial")
    ):
        raise ValueError(
            "Stage 8 training configuration records an invalid Stage 7 bundle"
        )

    for key, (method_name, arguments) in script_methods.items():
        _validate_torchscript_method(
            script_paths[key],
            method_name=method_name,
            expected_argument_names=arguments,
        )
    author_map_stats = validate_model_author_map(
        author_map_path,
        author_table_num_rows=author_table_num_rows,
    )
    return ModelArtifact(
        name=name,
        root=root,
        artifact_format=CANONICAL_ARTIFACT_FORMAT,
        model_type=model_type,
        manifest=manifest,
        model_config=model_config,
        training_config=training_config,
        script_paths={key: path.resolve() for key, path in script_paths.items()},
        script_sha256={key: file_sha256(path) for key, path in script_paths.items()},
        author_map_path=author_map_path.resolve(),
        author_map_stats=author_map_stats,
        author_map_allow_extra_columns=False,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        max_history_len=max_history_len,
        author_table_num_rows=author_table_num_rows,
        similarity_temperature=similarity_temperature,
    )


def _resolve_legacy_stage1_root(
    *,
    legacy_root: Path,
    manifest: Mapping[str, Any],
) -> Path | None:
    """Resolve the Stage 1 path recorded by a legacy Stage 3 manifest."""

    recorded = manifest["inputs"].get("01_get_data")
    if recorded is None:
        return None
    if not isinstance(recorded, str) or not recorded.strip():
        raise ValueError("Legacy Stage 3 manifest has an invalid 01_get_data input")
    stage1_root = Path(recorded).expanduser()
    if not stage1_root.is_absolute():
        stage1_root = legacy_root / stage1_root
    return stage1_root.resolve()


def _load_legacy_stage1_summary(stage1_root: Path) -> dict[str, Any]:
    """Validate and load the exact legacy Stage 1 artifact linked by Stage 3."""

    if not stage1_root.is_dir():
        raise FileNotFoundError(
            "Legacy Stage 1 artifact recorded by Stage 3 does not exist: "
            f"{stage1_root}"
        )
    _validate_completed_manifest(
        stage1_root,
        expected_stage_key="get_data",
        expected_stage_folder="01_get_data",
    )
    return _load_json_object(
        stage1_root / "summary.json",
        description="legacy Stage 1 summary",
    )


def _legacy_stage1_embedding_contract(
    summary: Mapping[str, Any],
) -> tuple[str | None, int | None]:
    """Read the legacy Stage 1 embedding identity and vector width."""

    parameters = summary.get("parameters")
    embedding_model: str | None = None
    if parameters is not None:
        parameters = _require_mapping(
            parameters,
            description="legacy Stage 1 summary parameters",
        )
        configured_model = parameters.get("embedding_model")
        if configured_model is not None:
            embedding_model = _require_nonempty_string(
                configured_model,
                description="legacy Stage 1 embedding_model",
            )
    outputs = _require_mapping(
        summary.get("outputs"),
        description="legacy Stage 1 summary outputs",
    )
    output_dimension = outputs.get("embedding_dim")
    embedding_dim = (
        _positive_int(
            output_dimension,
            description="legacy Stage 1 embedding_dim",
        )
        if output_dimension is not None
        else None
    )
    return embedding_model, embedding_dim


def _resolve_legacy_author_map(
    *,
    stage1_root: Path | None,
    stage1_summary: Mapping[str, Any] | None,
    author_map_override: Path | None,
) -> Path:
    """Resolve either an explicit legacy map or Stage 1's exact output."""

    if author_map_override is not None:
        author_map_path = Path(author_map_override).expanduser().resolve()
        if not author_map_path.is_file():
            raise FileNotFoundError(
                f"Legacy author-map override does not exist: {author_map_path}"
            )
        return author_map_path
    if stage1_root is None:
        raise ValueError(
            "Legacy Stage 3 manifest has no 01_get_data input; provide an explicit "
            "legacy author-map override"
        )
    if stage1_summary is None:
        raise RuntimeError("Legacy Stage 1 summary was not loaded")
    outputs = _require_mapping(
        stage1_summary.get("outputs"),
        description="legacy Stage 1 summary outputs",
    )
    recorded_author_map = _require_nonempty_string(
        outputs.get("author_idx_file"),
        description="legacy Stage 1 author_idx_file",
    )
    recorded_path = Path(recorded_author_map).expanduser()
    author_map_path = (
        recorded_path
        if recorded_path.is_absolute()
        else stage1_root / recorded_path
    ).resolve()
    if not author_map_path.is_file():
        raise FileNotFoundError(
            "Legacy Stage 1 summary author_idx_file does not exist: "
            f"{author_map_path}. Supply an explicit legacy author-map override if the "
            "original Stage 1 artifact is unavailable."
        )
    return author_map_path


def _validate_legacy_bst_script(path: Path) -> None:
    """Require the ordinary eight-input legacy BST matrix-scoring contract."""

    try:
        _validate_torchscript_method(
            path,
            method_name="score_candidate_matrix",
            expected_argument_names=_BST_SCORE_ARGUMENT_NAMES,
        )
    except FileNotFoundError:
        raise
    except ValueError as exc:
        raise ValueError(
            "Unsupported legacy BST TorchScript contract. Comparison supports only "
            "the standard eight-argument score_candidate_matrix method; experimental "
            "rankers requiring post-liker, target-user, or other extra features are "
            f"not supported: {path}. Details: {exc}"
        ) from exc


def _resolve_legacy_bst_artifact(
    *,
    name: str,
    root: Path,
    author_map_override: Path | None,
) -> ModelArtifact:
    """Resolve a supported legacy ``03_train`` BST TorchScript artifact."""

    manifest = _validate_completed_manifest(
        root,
        expected_stage_key="train_bst_ranker",
        expected_stage_folder="03_train",
    )
    training_config = _load_json_object(
        root / "training_config.json",
        description="legacy Stage 3 training configuration",
    )
    if training_config.get("model_type") != BST_MODEL_TYPE:
        raise ValueError(
            "Legacy comparison supports only training_config model_type='bst-ranker'"
        )
    if training_config.get("use_author_embedding_table") is not True:
        raise ValueError(
            "Legacy comparison requires a BST trained with the author embedding table"
        )
    embedding_dim = _positive_int(
        training_config.get("post_embedding_dim"),
        description="legacy BST post_embedding_dim",
    )
    max_history_len = _positive_int(
        training_config.get("max_history_len"),
        description="legacy BST max_history_len",
    )
    author_table_num_rows = _positive_int(
        training_config.get("author_table_num_rows"),
        description="legacy BST author_table_num_rows",
    )
    if author_table_num_rows < 2:
        raise ValueError("Legacy BST author table must reserve PAD=0 and UNK=1")
    if training_config.get("author_pad_idx") != AUTHOR_PAD_IDX:
        raise ValueError("Legacy BST must reserve author PAD=0")
    if training_config.get("author_unk_idx") != AUTHOR_UNK_IDX:
        raise ValueError("Legacy BST must reserve author UNK=1")
    ranker_path = root / "checkpoints" / "ranker.pt"
    _validate_legacy_bst_script(ranker_path)

    resolved_config_path = root / "resolved_config.json"
    resolved_config = (
        _load_json_object(
            resolved_config_path,
            description="legacy Stage 3 resolved configuration",
        )
        if resolved_config_path.is_file()
        else {}
    )
    resolved_model_type = resolved_config.get("model_type")
    if resolved_model_type is not None and resolved_model_type != BST_MODEL_TYPE:
        raise ValueError(
            "Legacy resolved configuration and training configuration disagree on "
            "model type"
        )

    stage1_root = _resolve_legacy_stage1_root(
        legacy_root=root,
        manifest=manifest,
    )
    stage1_summary: dict[str, Any] | None = None
    if stage1_root is not None and stage1_root.is_dir():
        stage1_summary = _load_legacy_stage1_summary(stage1_root)
    elif author_map_override is None:
        if stage1_root is None:
            raise ValueError(
                "Legacy Stage 3 manifest has no 01_get_data input; provide an explicit "
                "legacy author-map override"
            )
        raise FileNotFoundError(
            "Legacy Stage 1 artifact recorded by Stage 3 does not exist: "
            f"{stage1_root}. Supply an explicit legacy author-map override if the "
            "original Stage 1 artifact is unavailable."
        )

    stage1_embedding_model: str | None = None
    stage1_embedding_dim: int | None = None
    if stage1_summary is not None:
        (
            stage1_embedding_model,
            stage1_embedding_dim,
        ) = _legacy_stage1_embedding_contract(stage1_summary)
        if stage1_embedding_dim is not None and stage1_embedding_dim != embedding_dim:
            raise ValueError(
                "Legacy Stage 1 and Stage 3 embedding dimensions disagree: "
                f"{stage1_embedding_dim} != {embedding_dim}"
            )

    resolved_embedding_model = resolved_config.get("embedding_model")
    if resolved_embedding_model is not None:
        resolved_embedding_model = _require_nonempty_string(
            resolved_embedding_model,
            description="legacy resolved_config embedding_model",
        )
    if (
        resolved_embedding_model is not None
        and stage1_embedding_model is not None
        and resolved_embedding_model != stage1_embedding_model
    ):
        raise ValueError(
            "Legacy Stage 1 and Stage 3 embedding models disagree: "
            f"{stage1_embedding_model!r} != {resolved_embedding_model!r}"
        )
    embedding_model = resolved_embedding_model or stage1_embedding_model
    if embedding_model is None:
        raise ValueError(
            "Could not determine the legacy model's content-embedding model from "
            "resolved_config.json or its linked Stage 1 summary"
        )

    author_map_path = _resolve_legacy_author_map(
        stage1_root=stage1_root,
        stage1_summary=stage1_summary,
        author_map_override=author_map_override,
    )
    author_map_stats = validate_model_author_map(
        author_map_path,
        author_table_num_rows=author_table_num_rows,
        allow_extra_columns=True,
    )

    model_config = {
        "artifact_format": LEGACY_BST_ARTIFACT_FORMAT,
        "model_type": BST_MODEL_TYPE,
        "embedding_model": embedding_model,
        "max_history_len": max_history_len,
        "author_pad_idx": AUTHOR_PAD_IDX,
        "author_unk_idx": AUTHOR_UNK_IDX,
        "constructor_args": {
            "post_embedding_dim": embedding_dim,
            "author_table_num_rows": author_table_num_rows,
        },
    }
    return ModelArtifact(
        name=name,
        root=root,
        artifact_format=LEGACY_BST_ARTIFACT_FORMAT,
        model_type=BST_MODEL_TYPE,
        manifest=manifest,
        model_config=model_config,
        training_config=training_config,
        script_paths={"ranker": ranker_path.resolve()},
        script_sha256={"ranker": file_sha256(ranker_path)},
        author_map_path=author_map_path,
        author_map_stats=author_map_stats,
        author_map_allow_extra_columns=True,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        max_history_len=max_history_len,
        author_table_num_rows=author_table_num_rows,
        similarity_temperature=None,
    )


def resolve_model_artifact(
    name: str,
    path: Path,
    *,
    author_map_override: Path | None = None,
) -> ModelArtifact:
    """Resolve a canonical Stage 8 model or supported legacy Stage 3 BST."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Model name must be a non-empty string")
    normalized_name = name.strip()
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Model artifact directory does not exist: {root}")
    if (root / "model_config.json").is_file():
        if author_map_override is not None:
            raise ValueError(
                "Author-map overrides are supported only for legacy Stage 3 BST artifacts"
            )
        return _resolve_canonical_model_artifact(name=normalized_name, root=root)
    return _resolve_legacy_bst_artifact(
        name=normalized_name,
        root=root,
        author_map_override=author_map_override,
    )


def validate_comparison_contract(
    dataset: Stage7Artifact,
    models: Sequence[ModelArtifact],
    max_history_len: int | None,
) -> dict[str, int]:
    """Validate cross-artifact compatibility and choose each history width."""

    if len(models) != 2:
        raise ValueError("Model-performance comparison requires exactly two models")
    names = [model.name for model in models]
    if len(set(names)) != 2:
        raise ValueError("Model-performance comparison requires unique model names")
    for model in models:
        if model.embedding_model != dataset.embedding_model:
            raise ValueError(
                f"Model {model.name!r} embedding model {model.embedding_model!r} does not "
                f"match Stage 7 {dataset.embedding_model!r}"
            )
        if model.embedding_dim != dataset.embedding_dim:
            raise ValueError(
                f"Model {model.name!r} embedding dimension {model.embedding_dim} does not "
                f"match Stage 7 {dataset.embedding_dim}"
            )
    if max_history_len is None:
        return {model.name: model.max_history_len for model in models}
    common_history_len = _positive_int(
        max_history_len,
        description="max_history_len override",
    )
    too_short = [
        model.name for model in models if common_history_len > model.max_history_len
    ]
    if too_short:
        details = ", ".join(
            f"{model.name}={model.max_history_len}" for model in models
        )
        raise ValueError(
            f"max_history_len override {common_history_len} exceeds a model maximum: {details}"
        )
    return {model.name: common_history_len for model in models}
