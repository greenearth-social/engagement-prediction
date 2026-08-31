"""Serving-vocabulary and ClearML publication helpers for the canonical BST."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from engagement_prediction.experiment_tracking import ModelPublicationTracker
from engagement_prediction.training.model_artifacts import write_json_atomically


def publish_ranker_to_tracker(
    *,
    tracker: ModelPublicationTracker,
    logger: logging.Logger,
    torchscript_path: Path,
    author_map_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    """Best-effort publish a complete ranker serving set to the tracker.

    The serving manifest is a deployment marker. It is not created unless the
    registered model and its matching author vocabulary both reached ClearML.
    """

    task_id = str(tracker.id or "")
    result: Dict[str, Any] = {
        "status": "not_configured" if not task_id else "incomplete",
        "clearml_task_id": task_id,
        "ranker_clearml_model_id": "",
        "ranker_uri": "",
        "model_registered": False,
        "author_map_uploaded": False,
        "manifest_created": False,
        "manifest_uploaded": False,
        "manifest_path": None,
        "errors": [],
    }
    if not task_id:
        logger.info(
            "Experiment tracker is disabled; keeping local ranker and author map only"
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

    if not result["model_registered"] or not result["author_map_uploaded"]:
        return result

    manifest = {
        "ranker_clearml_model_id": result["ranker_clearml_model_id"],
        "ranker_uri": result["ranker_uri"],
        "clearml_task_id": task_id,
    }
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
