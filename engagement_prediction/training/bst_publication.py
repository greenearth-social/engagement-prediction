"""Serving-vocabulary and ClearML publication helpers for the canonical BST."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from engagement_prediction.experiment_tracking import ModelPublicationTracker
from engagement_prediction.training.model_artifacts import write_json_atomically


POST_LIKER_USER_MAP_ARTIFACT_NAME = "post_liker_user_idx_mapping"
POST_LIKER_USER_EMBEDDINGS_ARTIFACT_NAME = "post_liker_user_embeddings"
POST_LIKER_STATE_CONFIG_ARTIFACT_NAME = "post_liker_state_config"


def publish_ranker_to_tracker(
    *,
    tracker: ModelPublicationTracker,
    logger: logging.Logger,
    torchscript_path: Path,
    author_map_path: Path,
    manifest_path: Path,
    post_liker_feature_enabled: bool,
    post_liker_user_map_path: Path | None,
    post_liker_user_embeddings_path: Path | None,
    post_liker_state_config_path: Path | None,
) -> Dict[str, Any]:
    """Best-effort publish a complete ranker serving set to the tracker.

    The serving manifest is a deployment marker. It is not created unless the
    registered model and its matching author vocabulary both reached ClearML.
    """

    companion_paths = {
        POST_LIKER_USER_MAP_ARTIFACT_NAME: post_liker_user_map_path,
        POST_LIKER_USER_EMBEDDINGS_ARTIFACT_NAME: post_liker_user_embeddings_path,
        POST_LIKER_STATE_CONFIG_ARTIFACT_NAME: post_liker_state_config_path,
    }
    if post_liker_feature_enabled:
        missing_paths = [
            name
            for name, path in companion_paths.items()
            if path is None or not Path(path).is_file()
        ]
        if missing_paths:
            raise FileNotFoundError(
                "Feature-enabled BST publication requires all local post-liker "
                f"companions; missing={missing_paths}"
            )
    elif any(path is not None for path in companion_paths.values()):
        raise ValueError(
            "Feature-disabled BST publication must not receive post-liker companions"
        )

    task_id = str(tracker.id or "")
    result: Dict[str, Any] = {
        "status": "not_configured" if not task_id else "incomplete",
        "clearml_task_id": task_id,
        "ranker_clearml_model_id": "",
        "ranker_uri": "",
        "model_registered": False,
        "author_map_uploaded": False,
        "post_liker_feature_enabled": bool(post_liker_feature_enabled),
        "post_liker_user_map_uploaded": False,
        "post_liker_user_embeddings_uploaded": False,
        "post_liker_state_config_uploaded": False,
        "manifest_created": False,
        "manifest_uploaded": False,
        "manifest_path": None,
        "errors": [],
    }
    if not task_id:
        logger.info(
            "Experiment tracker is disabled; keeping local BST serving files only"
        )
        return result

    try:
        model_metadata = tracker.log_artifact(name="ranker", path=torchscript_path)
        model_id = str(model_metadata.get("model_id", "") or "")
        model_uri = str(model_metadata.get("uri", "") or "")
        if not model_id or not model_uri:
            raise RuntimeError("tracker returned an empty ranker model ID or URI")
        if not model_uri.endswith("/models/ranker.pt"):
            raise RuntimeError(
                "tracker returned a ranker URI that does not end in /models/ranker.pt"
            )
        result.update({
            "ranker_clearml_model_id": model_id,
            "ranker_uri": model_uri,
            "model_registered": True,
        })
    except Exception as exc:
        message = f"ClearML ranker model registration failed: {exc}"
        result["errors"].append(message)
        logger.warning(message, exc_info=True)

    try:
        result["author_map_uploaded"] = bool(
            tracker.log_file_artifact("author_idx_mapping", author_map_path)
        )
        if not result["author_map_uploaded"]:
            message = "ClearML did not upload the ranker author_idx_mapping artifact"
            result["errors"].append(message)
            logger.warning(message)
    except Exception as exc:
        message = f"ClearML ranker author-map upload failed: {exc}"
        result["errors"].append(message)
        logger.warning(message, exc_info=True)

    if post_liker_feature_enabled:
        result_keys = {
            POST_LIKER_USER_MAP_ARTIFACT_NAME: "post_liker_user_map_uploaded",
            POST_LIKER_USER_EMBEDDINGS_ARTIFACT_NAME: (
                "post_liker_user_embeddings_uploaded"
            ),
            POST_LIKER_STATE_CONFIG_ARTIFACT_NAME: (
                "post_liker_state_config_uploaded"
            ),
        }
        for artifact_name, path in companion_paths.items():
            result_key = result_keys[artifact_name]
            try:
                result[result_key] = bool(
                    tracker.log_file_artifact(artifact_name, Path(path))
                )
                if not result[result_key]:
                    message = (
                        f"ClearML did not upload the ranker {artifact_name} artifact"
                    )
                    result["errors"].append(message)
                    logger.warning(message)
            except Exception as exc:
                message = f"ClearML ranker {artifact_name} upload failed: {exc}"
                result["errors"].append(message)
                logger.warning(message, exc_info=True)

    required_uploads = [
        result["model_registered"],
        result["author_map_uploaded"],
    ]
    if post_liker_feature_enabled:
        required_uploads.extend([
            result["post_liker_user_map_uploaded"],
            result["post_liker_user_embeddings_uploaded"],
            result["post_liker_state_config_uploaded"],
        ])
    if not all(required_uploads):
        return result

    manifest = {
        "ranker_clearml_model_id": result["ranker_clearml_model_id"],
        "ranker_uri": result["ranker_uri"],
        "clearml_task_id": task_id,
    }
    if post_liker_feature_enabled:
        manifest.update({
            "ranker_contract_version": 2,
            "post_liker_feature_enabled": True,
            "post_liker_user_idx_mapping_artifact_name": (
                POST_LIKER_USER_MAP_ARTIFACT_NAME
            ),
            "post_liker_user_embeddings_artifact_name": (
                POST_LIKER_USER_EMBEDDINGS_ARTIFACT_NAME
            ),
            "post_liker_state_config_artifact_name": (
                POST_LIKER_STATE_CONFIG_ARTIFACT_NAME
            ),
        })
    try:
        write_json_atomically(manifest_path, manifest)
        result["manifest_created"] = True
        result["manifest_path"] = str(manifest_path)
    except Exception as exc:
        message = f"Local ranker serving-manifest write failed: {exc}"
        result["errors"].append(message)
        logger.warning(message, exc_info=True)
        return result
    try:
        result["manifest_uploaded"] = bool(
            tracker.log_file_artifact("ranker_serving_manifest", manifest_path)
        )
        if result["manifest_uploaded"]:
            result["status"] = "complete"
        else:
            message = "ClearML did not upload the ranker serving manifest"
            result["errors"].append(message)
            logger.warning(message)
    except Exception as exc:
        message = f"ClearML ranker serving-manifest upload failed: {exc}"
        result["errors"].append(message)
        logger.warning(message, exc_info=True)
    return result
