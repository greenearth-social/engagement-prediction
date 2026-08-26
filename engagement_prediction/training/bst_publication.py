"""Serving-vocabulary and ClearML publication helpers for the canonical BST."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import polars as pl

from engagement_prediction.data.parquet import scan_parquet_artifact


RANKER_AUTHOR_MAP_SCHEMA = {
    "author_did": pl.String,
    "author_idx": pl.UInt32,
}


def _write_serving_manifest_atomically(
    manifest_path: Path,
    manifest: Dict[str, str],
) -> None:
    """Publish the deployment marker without exposing truncated JSON."""

    partial_path = manifest_path.with_name(f"{manifest_path.name}.partial")
    try:
        partial_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        partial_path.replace(manifest_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def write_ranker_author_map(
    *,
    authors_path: Path,
    output_path: Path,
    author_table_num_rows: int,
) -> Dict[str, int]:
    """Write and validate the single-file author map required by serving."""

    if author_table_num_rows < 2:
        raise ValueError("BST author table must reserve PAD=0 and UNK=1")
    authors_lf = scan_parquet_artifact(authors_path)
    source_schema = authors_lf.collect_schema()
    for column, dtype in RANKER_AUTHOR_MAP_SCHEMA.items():
        if source_schema.get(column) != dtype:
            raise ValueError(
                f"Stage 7 authors artifact must contain {column} with dtype {dtype}"
            )

    author_map_lf = authors_lf.select(list(RANKER_AUTHOR_MAP_SCHEMA)).sort("author_did")
    validation = author_map_lf.select(
        pl.len().alias("author_count"),
        pl.col("author_did").null_count().alias("null_did_count"),
        pl.col("author_idx").null_count().alias("null_idx_count"),
        pl.col("author_did").n_unique().alias("unique_did_count"),
        pl.col("author_idx").n_unique().alias("unique_idx_count"),
        pl.col("author_idx").min().alias("min_author_idx"),
        pl.col("author_idx").max().alias("max_author_idx"),
    ).collect().row(0, named=True)
    author_count = int(validation["author_count"])
    if validation["null_did_count"] or validation["null_idx_count"]:
        raise ValueError("Stage 7 author vocabulary contains null keys")
    if int(validation["unique_did_count"]) != author_count:
        raise ValueError("Stage 7 author vocabulary contains duplicate author DIDs")
    if int(validation["unique_idx_count"]) != author_count:
        raise ValueError("Stage 7 author vocabulary contains duplicate author indices")
    if author_table_num_rows != author_count + 2:
        raise ValueError(
            "Stage 7 author vocabulary size does not match the BST author table"
        )
    if author_count:
        if int(validation["min_author_idx"]) != 2:
            raise ValueError("Stage 7 author vocabulary must begin at index 2")
        if int(validation["max_author_idx"]) != author_table_num_rows - 1:
            raise ValueError("Stage 7 author vocabulary indices must be dense")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    try:
        author_map_lf.sink_parquet(
            partial_path,
            compression="zstd",
            maintain_order=True,
            engine="streaming",
        )
        if pl.read_parquet_schema(partial_path) != pl.Schema(RANKER_AUTHOR_MAP_SCHEMA):
            raise ValueError("Published ranker author map has an unexpected schema")
        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return {
        "author_count": author_count,
        "author_table_num_rows": author_table_num_rows,
        "file_size_bytes": output_path.stat().st_size,
    }


def publish_ranker_to_tracker(
    *,
    tracker: Any,
    logger: logging.Logger,
    torchscript_path: Path,
    author_map_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    """Best-effort publish a complete ranker serving set to the tracker.

    The serving manifest is a deployment marker. It is not created unless the
    registered model and its matching author vocabulary both reached ClearML.
    """

    task_id = str(getattr(tracker, "id", "") or "") if tracker is not None else ""
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
        _write_serving_manifest_atomically(manifest_path, manifest)
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
