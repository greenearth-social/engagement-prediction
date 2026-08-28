"""Serving-vocabulary and ClearML publication for canonical two-tower models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from engagement_prediction.training.model_artifacts import write_json_atomically


def publish_two_tower_to_tracker(
    *,
    tracker: Any,
    logger: logging.Logger,
    user_tower_path: Path,
    post_tower_path: Path,
    author_map_path: Path,
    manifest_path: Path,
    output_embedding_dim: int,
) -> Dict[str, Any]:
    """Best-effort publish a complete two-tower serving set.

    The manifest is the deployment marker, so it is created only after both
    tower OutputModels and the matching author vocabulary reached ClearML.
    """

    task_id = str(getattr(tracker, "id", "") or "") if tracker is not None else ""
    result: Dict[str, Any] = {
        "status": "not_configured" if not task_id else "incomplete",
        "clearml_task_id": task_id,
        "user_tower_clearml_model_id": "",
        "post_tower_clearml_model_id": "",
        "user_tower_uri": "",
        "post_tower_uri": "",
        "user_tower_registered": False,
        "post_tower_registered": False,
        "author_map_uploaded": False,
        "manifest_created": False,
        "manifest_uploaded": False,
        "manifest_path": None,
        "errors": [],
    }
    if not task_id:
        logger.info(
            "Experiment tracker is disabled; keeping local two-tower artifacts only"
        )
        return result

    model_specs = (
        (
            "engagement_user_tower",
            Path(user_tower_path),
            "user_tower_clearml_model_id",
            "user_tower_uri",
            "user_tower_registered",
        ),
        (
            "engagement_post_tower",
            Path(post_tower_path),
            "post_tower_clearml_model_id",
            "post_tower_uri",
            "post_tower_registered",
        ),
    )
    for model_name, path, id_key, uri_key, registered_key in model_specs:
        try:
            metadata = tracker.log_artifact(name=model_name, path=path)
            model_id = str(metadata.get("model_id", "") or "")
            model_uri = str(metadata.get("uri", "") or "")
            if not model_id or not model_uri:
                raise RuntimeError("tracker returned an empty model ID or URI")
            if not model_uri.endswith(f"/models/{path.name}"):
                raise RuntimeError(
                    f"tracker returned a URI that does not end in /models/{path.name}"
                )
            result.update({
                id_key: model_id,
                uri_key: model_uri,
                registered_key: True,
            })
        except Exception as exc:
            message = f"ClearML {model_name} registration failed: {exc}"
            result["errors"].append(message)
            logger.warning(message, exc_info=True)

    try:
        result["author_map_uploaded"] = bool(
            tracker.log_file_artifact("author_idx_mapping", Path(author_map_path))
        )
        if not result["author_map_uploaded"]:
            message = "ClearML did not upload the two-tower author_idx_mapping artifact"
            result["errors"].append(message)
            logger.warning(message)
    except Exception as exc:
        message = f"ClearML two-tower author-map upload failed: {exc}"
        result["errors"].append(message)
        logger.warning(message, exc_info=True)

    if not (
        result["user_tower_registered"]
        and result["post_tower_registered"]
        and result["author_map_uploaded"]
    ):
        return result

    manifest = {
        "user_tower_clearml_model_id": result["user_tower_clearml_model_id"],
        "post_tower_clearml_model_id": result["post_tower_clearml_model_id"],
        "user_tower_uri": result["user_tower_uri"],
        "post_tower_uri": result["post_tower_uri"],
        "output_embedding_dim": int(output_embedding_dim),
        "clearml_task_id": task_id,
        "embedding_space_id": result["post_tower_clearml_model_id"],
    }
    try:
        write_json_atomically(Path(manifest_path), manifest)
        result["manifest_created"] = True
        result["manifest_path"] = str(manifest_path)
    except Exception as exc:
        message = f"Local two-tower serving-manifest write failed: {exc}"
        result["errors"].append(message)
        logger.warning(message, exc_info=True)
        return result
    try:
        result["manifest_uploaded"] = bool(
            tracker.log_file_artifact(
                "two_tower_serving_manifest",
                Path(manifest_path),
            )
        )
        if result["manifest_uploaded"]:
            result["status"] = "complete"
        else:
            message = "ClearML did not upload the two-tower serving manifest"
            result["errors"].append(message)
            logger.warning(message)
    except Exception as exc:
        message = f"ClearML two-tower serving-manifest upload failed: {exc}"
        result["errors"].append(message)
        logger.warning(message, exc_info=True)
    return result
