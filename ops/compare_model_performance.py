#!/usr/bin/env python3
"""Compare exactly two TorchScript models on one canonical Stage 7 dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
import time
from typing import Sequence
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


DEFAULT_SPLITS = (
    "val",
    "val_unseen_users",
    "holdout_unseen_users",
    "holdout_seen_users",
)
DEFAULT_BATCH_SIZE = 128
DEFAULT_METRICS_TOP_KS = (30,)
DEFAULT_BST_CANDIDATE_CHUNK_SIZE = 1024
DEFAULT_RANDOM_SEED = 42
DEFAULT_NUM_DATALOADER_WORKERS = 4
DEFAULT_DATALOADER_PIN_MEMORY = True
DEFAULT_DATALOADER_PREFETCH_FACTOR = 2
DEFAULT_MAX_CLASSIFICATION_METRIC_PAIRS = 2_000_000
DEFAULT_OUTPUT_DIR = Path("outputs/comparisons")


@dataclass(frozen=True)
class ModelArgument:
    """One user-supplied ``NAME=PATH`` model reference."""

    name: str
    path: Path


def parse_model_argument(raw: str) -> ModelArgument:
    """Parse a model argument without resolving or inspecting its path."""

    name, separator, raw_path = str(raw).partition("=")
    name = name.strip()
    raw_path = raw_path.strip()
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("model must have format NAME=PATH")
    return ModelArgument(name=name, path=Path(raw_path).expanduser())


parse_model_spec = parse_model_argument


def parse_author_map_argument(raw: str) -> ModelArgument:
    """Parse a model-specific ``NAME=PATH`` author-map override."""

    try:
        return parse_model_argument(raw)
    except argparse.ArgumentTypeError as exc:
        raise argparse.ArgumentTypeError(
            "author map must have format NAME=PATH"
        ) from exc


class ComparisonArgumentParser(argparse.ArgumentParser):
    """Argument parser that enforces the two-model comparison contract."""

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        parsed = super().parse_args(args=args, namespace=namespace)
        model_arguments = list(parsed.model)
        if len(model_arguments) != 2:
            self.error("exactly two --model NAME=PATH arguments are required")
        model_names = [model.name for model in model_arguments]
        if len(set(model_names)) != len(model_names):
            self.error("--model names must be unique")
        author_map_arguments = list(parsed.author_map)
        author_map_names = [author_map.name for author_map in author_map_arguments]
        if len(set(author_map_names)) != len(author_map_names):
            self.error("--author-map names must be unique")
        unknown_author_map_names = sorted(set(author_map_names) - set(model_names))
        if unknown_author_map_names:
            self.error(
                "--author-map names must match a --model name; unknown names: "
                + ", ".join(unknown_author_map_names)
            )
        if parsed.batch_size <= 0:
            self.error("--batch-size must be positive")
        if parsed.additional_batch_negatives is not None and parsed.additional_batch_negatives <= 0:
            self.error("--additional-batch-negatives must be positive")
        if any(k <= 0 for k in parsed.metrics_top_ks):
            self.error("--metrics-top-ks values must be positive")
        if len(set(parsed.metrics_top_ks)) != len(parsed.metrics_top_ks):
            self.error("--metrics-top-ks values must be unique")
        if parsed.bst_candidate_chunk_size <= 0:
            self.error("--bst-candidate-chunk-size must be positive")
        if parsed.num_dataloader_workers < 0:
            self.error("--num-dataloader-workers must be non-negative")
        if parsed.dataloader_prefetch_factor <= 0:
            self.error("--dataloader-prefetch-factor must be positive")
        if parsed.max_history_len is not None and parsed.max_history_len <= 0:
            self.error("--max-history-len must be positive")
        if parsed.max_classification_metric_pairs <= 0:
            self.error("--max-classification-metric-pairs must be positive")
        if len(set(parsed.splits)) != len(parsed.splits):
            self.error("--splits values must be unique")
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = ComparisonArgumentParser(
        description=(
            "Compare exactly two TorchScript models (canonical Stage 8 BST or "
            "two-tower, or standard legacy Stage 3 BST) on deterministic shared "
            "Stage 7 candidate slates."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=(
            "Completed Stage 7 artifact directory or its hydrated_training_data_* bundle"
        ),
    )
    parser.add_argument(
        "--model",
        type=parse_model_argument,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help=(
            "Uniquely named completed canonical Stage 8 artifact or standard "
            "legacy Stage 3 BST artifact; pass exactly twice"
        ),
    )
    parser.add_argument(
        "--author-map",
        type=parse_author_map_argument,
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Optional author-index Parquet override for a named legacy BST model; "
            "repeat for multiple legacy models"
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help=f"Stage 7 splits to evaluate (default: {' '.join(DEFAULT_SPLITS)})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Evaluation batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--additional-batch-negatives",
        type=int,
        help="Shared sampled negative cap per batch (default: all stored negatives)",
    )
    parser.add_argument(
        "--metrics-top-ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_METRICS_TOP_KS),
        metavar="K",
        help="K values for DCG and NDCG (default: 30)",
    )
    parser.add_argument(
        "--bst-candidate-chunk-size",
        type=int,
        default=DEFAULT_BST_CANDIDATE_CHUNK_SIZE,
        help=(
            "Maximum candidates scored per BST TorchScript call "
            f"(default: {DEFAULT_BST_CANDIDATE_CHUNK_SIZE})"
        ),
    )
    parser.add_argument(
        "--device",
        help="Torch device (default: cuda when available, otherwise cpu)",
    )
    parser.add_argument(
        "--num-dataloader-workers",
        type=int,
        default=DEFAULT_NUM_DATALOADER_WORKERS,
        help=f"DataLoader worker processes (default: {DEFAULT_NUM_DATALOADER_WORKERS})",
    )
    parser.add_argument(
        "--dataloader-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DATALOADER_PIN_MEMORY,
        help="Pin DataLoader memory (default: enabled)",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=DEFAULT_DATALOADER_PREFETCH_FACTOR,
        help=(
            "Batches prefetched per DataLoader worker "
            f"(default: {DEFAULT_DATALOADER_PREFETCH_FACTOR})"
        ),
    )
    parser.add_argument(
        "--max-history-len",
        type=int,
        help=(
            "Common positive history-length override; by default each model uses "
            "its configured maximum"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Deterministic evaluation seed (default: {DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        "--max-classification-metric-pairs",
        type=int,
        default=DEFAULT_MAX_CLASSIFICATION_METRIC_PAIRS,
        help=(
            "Maximum deterministically sampled candidate-label pairs for AUC/AP "
            f"(default: {DEFAULT_MAX_CLASSIFICATION_METRIC_PAIRS})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Parent directory for comparison runs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable per-model evaluation progress bars",
    )
    return parser


def generate_comparison_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def _allocate_output_paths(output_parent: Path) -> tuple[str, Path, Path]:
    output_parent = Path(output_parent).expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        run_id = generate_comparison_run_id()
        output_path = output_parent / run_id
        partial_path = output_parent / f"{run_id}.partial"
        if output_path.exists() or partial_path.exists():
            continue
        partial_path.mkdir()
        return run_id, partial_path, output_path
    raise RuntimeError(f"Unable to allocate a comparison run under {output_parent}")


def _configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"compare_model_performance.{uuid.uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    formatter.converter = time.gmtime
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _resolve_device(requested_device: str | None) -> str:
    if requested_device:
        return str(requested_device)
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _run(args: argparse.Namespace) -> Path:
    """Resolve artifacts, run both scorers sequentially, and publish reports."""

    # Imported lazily so ``--help`` remains fast and independent of PyTorch startup.
    from engagement_prediction.evaluation.artifacts import (
        resolve_model_artifact,
        resolve_stage7_artifact,
    )
    from engagement_prediction.evaluation.comparison import (
        ComparisonSettings,
    )
    from engagement_prediction.evaluation.workflow import (
        StandaloneReportContext,
        run_comparison_workflow,
    )

    run_id, partial_path, output_path = _allocate_output_paths(args.output_dir)
    logger = _configure_logger(partial_path / "comparison.log")
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    try:
        device = _resolve_device(args.device)
        dataset = resolve_stage7_artifact(args.dataset)
        author_map_overrides = {
            author_map.name: author_map.path for author_map in args.author_map
        }
        models = tuple(
            resolve_model_artifact(
                model.name,
                model.path,
                author_map_override=author_map_overrides.get(model.name),
            )
            for model in args.model
        )
        settings = ComparisonSettings(
            splits=tuple(args.splits),
            batch_size=args.batch_size,
            additional_batch_negatives=args.additional_batch_negatives,
            metrics_top_ks=tuple(args.metrics_top_ks),
            bst_candidate_chunk_size=args.bst_candidate_chunk_size,
            device=device,
            num_dataloader_workers=args.num_dataloader_workers,
            dataloader_pin_memory=args.dataloader_pin_memory,
            dataloader_prefetch_factor=args.dataloader_prefetch_factor,
            random_seed=args.random_seed,
            max_classification_metric_pairs=args.max_classification_metric_pairs,
            max_history_len=args.max_history_len,
            disable_progress=args.disable_progress,
        )
        logger.info(
            "Starting comparison run %s: dataset=%s models=%s device=%s",
            run_id,
            args.dataset,
            ",".join(model.name for model in args.model),
            device,
        )
        report = run_comparison_workflow(
            dataset=dataset,
            models=models,
            settings=settings,
            output_dir=partial_path,
            report_context=StandaloneReportContext(
                run_id=run_id, started_at=started_at, started_monotonic=started,
            ),
            logger=logger,
        )
        logger.info("Comparison completed successfully in %.5f seconds", report["runtime_seconds"])
        _close_logger(logger)
        partial_path.rename(output_path)
        return output_path
    except BaseException:
        logger.exception("Comparison failed; partial output retained at %s", partial_path)
        _close_logger(logger)
        raise


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
