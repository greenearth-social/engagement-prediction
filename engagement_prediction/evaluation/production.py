"""Snapshot the ranker configured on the traffic-serving Cloud Run revision."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
from typing import Any
from urllib.parse import urlsplit

from google.cloud import storage

from engagement_prediction.evaluation.artifacts import (
    BST_MODEL_TYPE,
    ModelArtifact,
    _BST_EXTENDED_SCORE_ARGUMENT_NAMES,
    _BST_SCORE_ARGUMENT_NAMES,
    _load_json_object,
    _require_nonempty_string,
    _validate_model_config,
    _validate_torchscript_method_signatures,
)
from engagement_prediction.evaluation.author_mapping import validate_model_author_map
from engagement_prediction.training.model_artifacts import file_sha256


_REVISION_ENV_NAMES = frozenset({
    "GE_INFERENCE_MODELS",
    "GE_INFERENCE_RANKER_MANIFEST_URI",
    "GE_INFERENCE_RANKER_AUTHOR_MAP_URI",
    "GE_INFERENCE_RANKER_MAX_HISTORY_LEN",
    "GE_INFERENCE_CONTENT_EMBED_DIM",
})


def _describe_cloud_run(
    *, resource: str, name: str, project: str, region: str, projection: str
) -> dict[str, Any]:
    result = subprocess.run(
        [
            "gcloud", "run", resource, "describe", name,
            "--project", project, "--region", region,
            f"--format=json({projection})",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(result.stdout)
    if not isinstance(document, dict):
        raise ValueError(f"Cloud Run {resource} describe did not return an object")
    return document


def _active_revision(service: dict[str, Any]) -> str:
    status = service.get("status", {})
    generation = service.get("metadata", {}).get("generation")
    observed_generation = status.get("observedGeneration")
    if generation is None or str(generation) != str(observed_generation):
        raise ValueError("Production Cloud Run service is still reconciling")
    conditions = {
        condition.get("type"): condition.get("status")
        for condition in status.get("conditions", [])
    }
    if conditions.get("Ready") != "True" or any(
        conditions.get(name, "True") != "True"
        for name in ("ConfigurationsReady", "RoutesReady")
    ):
        raise ValueError("Production Cloud Run service is not ready")
    traffic = [
        target for target in status.get("traffic", [])
        if target.get("percent", 0) > 0
    ]
    if len(traffic) != 1 or traffic[0].get("percent") != 100:
        raise ValueError(
            "Production comparison requires one revision serving 100% of traffic"
        )
    return _require_nonempty_string(
        traffic[0].get("revisionName"), description="traffic-serving revision"
    )


def _model_environment(revision: dict[str, Any]) -> dict[str, str]:
    # Never serialize the full revision: other environment entries can be secrets.
    containers = revision.get("spec", {}).get("containers", [])
    model_containers = [
        container for container in containers
        if any(
            item.get("name") == "GE_INFERENCE_MODELS"
            for item in container.get("env", [])
        )
    ]
    if len(model_containers) != 1:
        raise ValueError("Production revision must have exactly one model container")
    environment = {}
    for item in model_containers[0].get("env", []):
        name = item.get("name")
        if name in _REVISION_ENV_NAMES:
            if name in environment:
                raise ValueError(f"Production revision repeats model setting {name}")
            environment[name] = _require_nonempty_string(
                item.get("value"), description=f"production setting {name}"
            )
    missing = _REVISION_ENV_NAMES.difference(environment)
    if missing:
        raise ValueError(f"Production revision is missing model settings: {sorted(missing)}")
    if "ranker" not in {
        name.strip() for name in environment["GE_INFERENCE_MODELS"].split(",")
    }:
        raise ValueError("The traffic-serving production revision does not load a ranker")
    return environment


def _positive_environment_int(environment: dict[str, str], name: str) -> int:
    try:
        value = int(environment[name])
    except ValueError as exc:
        raise ValueError(f"Production setting {name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"Production setting {name} must be a positive integer")
    return value


def _download_object(
    *, client: storage.Client, uri: str, path: Path
) -> dict[str, Any]:
    parsed = urlsplit(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Production artifacts require a gs:// object URI: {uri}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"Production artifact URI must not contain a query or fragment: {uri}")
    blob = client.bucket(parsed.netloc).blob(parsed.path.lstrip("/"))
    blob.reload()
    generation = int(blob.generation)
    # Pin the observed object version. A concurrent overwrite must fail instead
    # of silently combining artifacts from different versions of a deployment.
    blob.download_to_filename(
        str(path), if_generation_match=generation, checksum="auto"
    )
    return {
        "uri": uri,
        "generation": str(generation),
        "crc32c": blob.crc32c,
        "md5_hash": blob.md5_hash,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "path": str(path),
    }


def snapshot_production_model(
    *,
    project: str,
    region: str,
    service: str,
    output_dir: Path,
    logger: logging.Logger,
) -> tuple[ModelArtifact, dict[str, Any]]:
    """Download and validate one deployed BST baseline without modifying it."""

    service_document = _describe_cloud_run(
        resource="services", name=service, project=project, region=region,
        projection="metadata.generation,status",
    )
    revision_name = _active_revision(service_document)
    revision_document = _describe_cloud_run(
        resource="revisions", name=revision_name, project=project, region=region,
        projection="spec.containers",
    )
    environment = _model_environment(revision_document)
    max_history_len = _positive_environment_int(
        environment, "GE_INFERENCE_RANKER_MAX_HISTORY_LEN"
    )
    deployed_embedding_dim = _positive_environment_int(
        environment, "GE_INFERENCE_CONTENT_EMBED_DIM"
    )
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    client = storage.Client(project=project)
    objects: dict[str, dict[str, Any]] = {}
    manifest_path = root / "ranker_serving_manifest.json"
    objects["manifest"] = _download_object(
        client=client, uri=environment["GE_INFERENCE_RANKER_MANIFEST_URI"],
        path=manifest_path,
    )
    manifest = _load_json_object(manifest_path, description="production serving manifest")
    if manifest.get("ranker_contract_version") not in (None, 1):
        raise ValueError("Unsupported production ranker serving contract")
    if manifest.get("post_liker_feature_enabled", False) is not False:
        raise ValueError("Production comparison does not support post-liker features")
    model_uri = _require_nonempty_string(
        manifest.get("ranker_uri"), description="production ranker URI"
    )
    model_id = _require_nonempty_string(
        manifest.get("ranker_clearml_model_id"), description="production ranker model ID"
    )
    task_id = _require_nonempty_string(
        manifest.get("clearml_task_id"), description="production ClearML task ID"
    )
    suffix = "/models/ranker.pt"
    if not model_uri.endswith(suffix):
        raise ValueError("Production ranker URI must end in /models/ranker.pt")
    task_root = model_uri[:-len(suffix)]
    model_config_path = root / "model_config.json"
    objects["model_config"] = _download_object(
        client=client,
        uri=f"{task_root}/artifacts/bst_model_config/model_config.json",
        path=model_config_path,
    )
    model_config = _load_json_object(
        model_config_path, description="production model configuration"
    )
    (
        model_type, embedding_model, embedding_dim, training_max_history_len,
        author_table_num_rows, similarity_temperature,
    ) = _validate_model_config(model_config)
    if model_type != BST_MODEL_TYPE:
        raise ValueError("Production comparison requires a BST ranker")
    for feature in ("use_post_liker_feature", "use_target_user_feature"):
        if model_config["constructor_args"].get(feature, False) is not False:
            raise ValueError(f"Production comparison does not support {feature}")
    if embedding_dim != deployed_embedding_dim:
        raise ValueError("Production revision and model embedding dimensions disagree")

    author_map_path = root / "ranker_author_idx.parquet"
    objects["author_map"] = _download_object(
        client=client, uri=environment["GE_INFERENCE_RANKER_AUTHOR_MAP_URI"],
        path=author_map_path,
    )
    author_map_stats = validate_model_author_map(
        author_map_path, author_table_num_rows=author_table_num_rows
    )
    # Validate the small contracts and author vocabulary before downloading weights.
    logger.info("Snapshotting production ranker %s from revision %s", model_id, revision_name)
    script_path = root / "ranker.pt"
    objects["ranker"] = _download_object(client=client, uri=model_uri, path=script_path)
    _validate_torchscript_method_signatures(
        script_path, method_name="score_candidate_matrix",
        expected_argument_signatures=(
            _BST_SCORE_ARGUMENT_NAMES, _BST_EXTENDED_SCORE_ARGUMENT_NAMES,
        ),
    )
    artifact = ModelArtifact(
        name="production", root=root, artifact_format="deployed_bst",
        model_type=model_type, manifest=manifest, model_config=model_config,
        training_config={}, script_paths={"ranker": script_path},
        script_sha256={"ranker": objects["ranker"]["sha256"]},
        author_map_path=author_map_path, author_map_stats=author_map_stats,
        author_map_allow_extra_columns=False, embedding_model=embedding_model,
        embedding_dim=embedding_dim, max_history_len=max_history_len,
        author_table_num_rows=author_table_num_rows,
        similarity_temperature=similarity_temperature,
    )
    metadata = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "cloud_run": {
            "project": project, "region": region, "service": service,
            "revision": revision_name,
        },
        "revision_environment": environment,
        "ranker_clearml_model_id": model_id,
        "clearml_task_id": task_id,
        "max_history_len": max_history_len,
        "training_max_history_len": training_max_history_len,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        # Canonical model configs do not carry their training data's date window.
        "training_window": None,
        "objects": objects,
    }
    (root / "production_snapshot.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return artifact, metadata
