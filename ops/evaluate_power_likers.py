#!/usr/bin/env python3
"""Evaluate power-liker bias of one trained BST ranker on its own Stage 7 dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Sequence
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


DEFAULT_POOL_SIZE = 10_000
DEFAULT_TOP_K = 100
DEFAULT_MIN_USER_LIKES = 5
DEFAULT_METRICS_TOP_K = 30
DEFAULT_RANDOM_SEED = 42
DEFAULT_BATCH_SIZE = 128  # matches run-all eval_batch_size; in-batch positives are shared candidates
DEFAULT_BST_CANDIDATE_CHUNK_SIZE = 1024
DEFAULT_NUM_DATALOADER_WORKERS = 4
DEFAULT_DATALOADER_PREFETCH_FACTOR = 2
DEFAULT_OUTPUT_DIR = Path("outputs/power_likers")
INFERENCE_BLOB_PREFIX = "bsky_inferences"


class PowerLikerArgumentParser(argparse.ArgumentParser):
    """Validate the standalone interface before creating any output files."""

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        parsed = super().parse_args(args, namespace)
        for name in (
            "pool_size", "top_k", "min_user_likes", "metrics_top_k", "batch_size",
            "bst_candidate_chunk_size", "dataloader_prefetch_factor",
        ):
            if getattr(parsed, name) <= 0:
                self.error(f"--{name.replace('_', '-')} must be positive")
        if parsed.num_dataloader_workers < 0:
            self.error("--num-dataloader-workers must be non-negative")
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = PowerLikerArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="Stage 8 BST run directory; its own Stage 7 dataset is evaluated")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Parent directory for run outputs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE,
                        help=f"Random pool posts scored for every user (default: {DEFAULT_POOL_SIZE})")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Synthetic feed length (default: {DEFAULT_TOP_K})")
    parser.add_argument("--min-user-likes", type=int, default=DEFAULT_MIN_USER_LIKES,
                        help=f"Liked holdout posts with inferences needed per user (default: {DEFAULT_MIN_USER_LIKES})")
    parser.add_argument("--metrics-top-k", type=int, default=DEFAULT_METRICS_TOP_K,
                        help=f"NDCG cutoff for holdout queries (default: {DEFAULT_METRICS_TOP_K})")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help=f"Pool sampling seed (default: {DEFAULT_RANDOM_SEED})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Queries or users per scoring batch (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--bst-candidate-chunk-size", type=int, default=DEFAULT_BST_CANDIDATE_CHUNK_SIZE,
                        help=f"BST candidate chunk size (default: {DEFAULT_BST_CANDIDATE_CHUNK_SIZE})")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (default: CUDA when available)")
    parser.add_argument("--num-dataloader-workers", type=int, default=DEFAULT_NUM_DATALOADER_WORKERS,
                        help=f"DataLoader workers (default: {DEFAULT_NUM_DATALOADER_WORKERS})")
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=DEFAULT_DATALOADER_PREFETCH_FACTOR,
                        help=f"DataLoader prefetch factor (default: {DEFAULT_DATALOADER_PREFETCH_FACTOR})")
    return parser


def _run(args: argparse.Namespace) -> Path:
    # Keep help and argument validation independent of model/data imports.
    from engagement_prediction.data.ingex import list_ingex_parquet_files
    from engagement_prediction.data.source_manifests import load_source_snapshot
    from engagement_prediction.evaluation.artifacts import (
        BST_MODEL_TYPE,
        resolve_model_artifact,
        resolve_stage7_artifact,
    )
    from engagement_prediction.evaluation.power_likers import (
        PowerLikerSettings,
        run_power_liker_evaluation,
    )
    from engagement_prediction.evaluation.scorers import create_model_scorer
    from engagement_prediction.training.runtime import get_device

    output_parent = args.output_dir.expanduser().resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_path = output_parent / run_id
    partial_path = output_parent / f"{run_id}.partial"
    partial_path.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(f"evaluate_power_likers.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.FileHandler(partial_path / "evaluation.log"), logging.StreamHandler()]
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    formatter.converter = time.gmtime
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    try:
        model = resolve_model_artifact("candidate", args.model)
        if model.model_type != BST_MODEL_TYPE:
            raise ValueError("Power-liker evaluation supports BST rankers only")
        dataset = resolve_stage7_artifact(Path(model.training_config["stage7_dir"]))
        # Inferences are not part of the training lineage; list them over the
        # bucket and window of the posts the dataset was built from.
        posts = load_source_snapshot(dataset.bundle_path, manifest_prefix="post_sources_",
                                     expected_blob_prefix="bsky_posts")
        likes = load_source_snapshot(dataset.bundle_path, manifest_prefix="like_sources_",
                                     expected_blob_prefix="bsky_likes")
        inference_paths, _ = list_ingex_parquet_files(
            gcs_bucket=posts.gcs_bucket, blob_prefix=INFERENCE_BLOB_PREFIX, start=posts.start, end=posts.end,
        )
        if not inference_paths:
            raise FileNotFoundError(f"No {INFERENCE_BLOB_PREFIX} exports in gs://{posts.gcs_bucket} for the post window")
        settings = PowerLikerSettings(
            pool_size=args.pool_size,
            top_k=args.top_k,
            min_user_likes=args.min_user_likes,
            metrics_top_k=args.metrics_top_k,
            random_seed=args.random_seed,
            batch_size=args.batch_size,
            device=get_device(args.device),
            num_dataloader_workers=args.num_dataloader_workers,
            dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        )
        logger.info("Starting power-liker evaluation %s: model=%s dataset=%s inference_files=%d device=%s",
                    run_id, model.root, dataset.bundle_path, len(inference_paths), settings.device)
        scorer = create_model_scorer(model, bst_candidate_chunk_size=args.bst_candidate_chunk_size)
        try:
            result = run_power_liker_evaluation(
                scorer, dataset, max_history_len=model.max_history_len,
                inference_paths=inference_paths, like_paths=likes.file_uris,
                like_window=(likes.start, likes.end), settings=settings, logger=logger,
            )
        finally:
            scorer.close()
        summary = {
            "run_id": run_id,
            "model": {
                "root": str(model.root),
                "script_sha256": model.script_sha256,
                "use_popularity_feature": model.model_config["constructor_args"].get("use_popularity_feature"),
            },
            "dataset": str(dataset.bundle_path),
            "inference_source": {"gcs_bucket": posts.gcs_bucket, "blob_prefix": INFERENCE_BLOB_PREFIX,
                                 "start": posts.start.isoformat(), "end": posts.end.isoformat(),
                                 "file_count": len(inference_paths)},
            "settings": asdict(settings),
            **result.summary,
        }
        (partial_path / "power_likers_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        result.ranking_rows.write_parquet(partial_path / "holdout_ranking_rows.parquet")
        result.headline_per_user.write_parquet(partial_path / "headline_per_user_amp.parquet")
        logger.info("Power-liker evaluation completed")
        for handler in handlers:
            handler.close()
        partial_path.rename(output_path)
        return output_path
    except BaseException:
        logger.exception("Power-liker evaluation failed; partial output retained at %s", partial_path)
        for handler in handlers:
            handler.close()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_path = _run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Power-liker evaluation completed successfully: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
