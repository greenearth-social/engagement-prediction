#!/usr/bin/env python3
"""Compare saved BST checkpoints on validation candidates grouped by media."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence
from urllib.parse import urlsplit
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ops.compare_model_performance import parse_model_argument


MEDIA_COMPARISON_DEFAULTS: dict[str, Any] = {
    "es_url": "https://localhost:9200",
    "metrics_top_ks": [30],
    "es_verify_ssl": False,
    "output_dir": "/mnt/data/dave/outputs/compare",
    "es_index": "posts",
    "es_batch_size": 1000,
    "es_request_timeout": 30.0,
    "device": None,
    "num_dataloader_workers": 4,
    "dataloader_pin_memory": True,
    "dataloader_prefetch_factor": 2,
    "disable_progress": False,
}


class MediaArgumentParser(argparse.ArgumentParser):
    """Validate the standalone interface before creating any output files."""

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        parsed = super().parse_args(args, namespace)
        names = [model.name for model in parsed.model]
        if len(set(names)) != len(names):
            self.error("--model names must be unique")
        if any(k <= 0 for k in parsed.metrics_top_ks):
            self.error("--metrics-top-ks values must be positive")
        if len(set(parsed.metrics_top_ks)) != len(parsed.metrics_top_ks):
            self.error("--metrics-top-ks values must be unique")
        parsed.metrics_top_ks = sorted(parsed.metrics_top_ks)
        if parsed.es_batch_size <= 0:
            self.error("--es-batch-size must be positive")
        if not math.isfinite(parsed.es_request_timeout) or parsed.es_request_timeout <= 0:
            self.error("--es-request-timeout must be finite and positive")
        if parsed.num_dataloader_workers < 0:
            self.error("--num-dataloader-workers must be non-negative")
        if parsed.dataloader_prefetch_factor <= 0:
            self.error("--dataloader-prefetch-factor must be positive")
        try:
            endpoint = urlsplit(parsed.es_url)
            valid_endpoint = (
                endpoint.scheme in {"http", "https"}
                and bool(endpoint.hostname)
                and endpoint.username is None
                and endpoint.password is None
                and not endpoint.query
                and not endpoint.fragment
            )
            endpoint.port
        except ValueError:
            valid_endpoint = False
        if not valid_endpoint:
            self.error("--es-url must be an http(s) URL without credentials, query, or fragment")
        if not parsed.es_index.strip() or any(char in parsed.es_index for char in "/?#"):
            self.error("--es-index must be an index or alias name, not a URL path")
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = MediaArgumentParser(
        description=(
            "Evaluate one or more saved best BST checkpoints on their shared val "
            "and val_unseen_users data, calculating NDCG within each media type."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=parse_model_argument, action="append", required=True,
        metavar="NAME=PATH", help="Named completed canonical Stage 8 BST artifact; repeat to compare",
    )
    defaults = MEDIA_COMPARISON_DEFAULTS
    parser.add_argument("--es-url", default=defaults["es_url"], help="Elasticsearch endpoint")
    parser.add_argument("--es-index", default=defaults["es_index"], help="Elasticsearch index or alias")
    parser.add_argument(
        "--es-verify-ssl", action=argparse.BooleanOptionalAction,
        default=defaults["es_verify_ssl"], help="Verify Elasticsearch TLS certificates",
    )
    parser.add_argument(
        "--es-batch-size", type=int, default=defaults["es_batch_size"],
        help="Unique URIs per Elasticsearch query",
    )
    parser.add_argument(
        "--es-request-timeout", type=float, default=defaults["es_request_timeout"],
        help="Timeout in seconds for each Elasticsearch request",
    )
    parser.add_argument(
        "--metrics-top-ks", type=int, nargs="+", default=list(defaults["metrics_top_ks"]),
        metavar="K", help="NDCG cutoffs (multiple values produce comparison lines)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(defaults["output_dir"]),
        help="Parent directory for unique comparison runs",
    )
    parser.add_argument("--device", default=defaults["device"], help="Torch device; auto selects CUDA or CPU")
    parser.add_argument(
        "--num-dataloader-workers", type=int, default=defaults["num_dataloader_workers"],
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--dataloader-pin-memory", action=argparse.BooleanOptionalAction,
        default=defaults["dataloader_pin_memory"], help="Pin DataLoader memory",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor", type=int, default=defaults["dataloader_prefetch_factor"],
        help="Batches prefetched per DataLoader worker",
    )
    parser.add_argument(
        "--disable-progress", action="store_true", default=defaults["disable_progress"],
        help="Disable progress bars",
    )
    return parser


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_run(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(document), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No report rows generated for {path.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_jsonable(rows))


def _run(args: argparse.Namespace) -> Path:
    # Keep help and argument validation independent of model/data imports.
    from engagement_prediction.evaluation.media_artifacts import (
        collect_candidate_uris,
        resolve_media_artifacts,
    )
    from engagement_prediction.evaluation.media_evaluation import (
        run_media_evaluation,
        write_media_plots,
    )
    from engagement_prediction.evaluation.media_hydration import hydrate_candidate_media
    from engagement_prediction.training.runtime import get_device
    import polars as pl

    output_parent = args.output_dir.expanduser().resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_path = output_parent / run_id
    partial_path = output_parent / f"{run_id}.partial"
    partial_path.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(f"compare_bst_media.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.FileHandler(partial_path / "comparison.log"), logging.StreamHandler()]
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    formatter.converter = time.gmtime
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    document: dict[str, Any] = {
        "tool": "compare_bst_media",
        "run_id": run_id,
        "status": "partial",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "media_definitions": {
            "images": "contains_images=true",
            "videos": "contains_video=true",
            "text_only": "contains_images=false and contains_video=false; includes link cards and quotes",
            "unknown": "missing document or either flag; excluded from all media groups",
        },
        "ranking_semantics": "NDCG within each media group; average over eligible user/hour queries",
    }
    started = time.monotonic()
    try:
        _write_run(partial_path / "run.json", document)
        logger.info("Resolving %d model artifacts and their common validation inputs", len(args.model))
        dataset, models, validation_settings = resolve_media_artifacts(
            [(model.name, model.path) for model in args.model]
        )
        device = get_device(args.device)
        document.update({
            "dataset": dataset.to_dict(),
            "models": [model.to_dict() for model in models],
            "validation_settings": validation_settings,
            "device": device,
        })
        _write_run(partial_path / "run.json", document)
        logger.info("Collecting validation candidate URIs from %s", dataset.bundle_path)
        candidates = collect_candidate_uris(dataset)
        media = hydrate_candidate_media(
            candidates,
            es_url=args.es_url,
            es_index=args.es_index,
            verify_ssl=args.es_verify_ssl,
            request_timeout=args.es_request_timeout,
            batch_size=args.es_batch_size,
            api_key=os.environ.get("GE_ELASTICSEARCH_API_KEY"),
            logger=logger,
        )
        media.write_parquet(partial_path / "media_metadata.parquet")
        media.filter(pl.col("media_status") != "known").write_csv(partial_path / "missing_media.csv")
        document["hydrated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_run(partial_path / "run.json", document)
        result = run_media_evaluation(
            dataset, models, validation_settings, media,
            metrics_top_ks=tuple(args.metrics_top_ks),
            device=device,
            num_workers=args.num_dataloader_workers,
            pin_memory=args.dataloader_pin_memory,
            prefetch_factor=args.dataloader_prefetch_factor,
            disable_progress=args.disable_progress,
            logger=logger,
        )
        _write_rows(partial_path / "metrics.csv", result["metrics"])
        _write_rows(partial_path / "hydration_coverage.csv", result["hydration_coverage"])
        write_media_plots(
            result["metrics"], partial_path,
            model_names=[model.name for model in models],
            metrics_top_ks=args.metrics_top_ks,
        )
        document.update({
            "status": "complete",
            "model_results": result["model_results"],
            "runtime_seconds": time.monotonic() - started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        _write_run(partial_path / "run.json", document)
        logger.info("Comparison completed in %.2f seconds", document["runtime_seconds"])
    except BaseException as exc:
        document.update({
            "status": "failed", "error": str(exc),
            "runtime_seconds": time.monotonic() - started,
        })
        _write_run(partial_path / "run.json", document)
        logger.exception("Comparison failed; partial output retained at %s", partial_path)
        raise
    finally:
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
    partial_path.rename(output_path)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_path = _run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Comparison completed successfully: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
