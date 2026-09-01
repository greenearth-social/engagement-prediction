"""Sequential deterministic comparison of supported TorchScript models."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Sequence

import torch

from engagement_prediction.data.datasets import (
    HydratedBucketedEngagementDataset,
    create_hydrated_data_loader,
)
from engagement_prediction.evaluation.artifacts import (
    ModelArtifact,
    Stage7Artifact,
    validate_comparison_contract,
)
from engagement_prediction.evaluation.author_mapping import (
    temporary_author_index_override,
)
from engagement_prediction.evaluation.scorers import create_model_scorer
from engagement_prediction.training.ranking import evaluate_matrix_scorer


@dataclass(frozen=True)
class ComparisonSettings:
    """Runtime settings supplied explicitly by the standalone CLI."""

    splits: tuple[str, ...]
    batch_size: int
    metrics_top_ks: tuple[int, ...]
    bst_candidate_chunk_size: int
    device: str
    num_dataloader_workers: int
    dataloader_pin_memory: bool
    dataloader_prefetch_factor: int
    random_seed: int
    max_classification_metric_pairs: int
    max_history_len: int | None
    disable_progress: bool


@dataclass(frozen=True)
class ComparisonResult:
    """Aggregate-only result returned to output orchestration."""

    metrics_by_model: dict[str, dict[str, dict[str, Any]]]
    mapping_coverage_by_model: dict[str, dict[str, Any]]
    split_row_counts: dict[str, int]
    skipped_splits: tuple[str, ...]
    history_lengths: dict[str, int]


def _validate_settings(
    settings: ComparisonSettings,
    *,
    available_splits: set[str],
) -> None:
    """Validate standalone CLI settings against the resolved Stage 7 index."""

    if not settings.splits:
        raise ValueError("At least one evaluation split is required")
    if len(set(settings.splits)) != len(settings.splits):
        raise ValueError("Evaluation splits must be unique")
    unknown_splits = sorted(set(settings.splits) - available_splits)
    if unknown_splits:
        raise ValueError(f"Unknown Stage 7 splits: {', '.join(unknown_splits)}")
    if settings.batch_size <= 0:
        raise ValueError("Comparison batch size must be positive")
    if not settings.metrics_top_ks or any(k <= 0 for k in settings.metrics_top_ks):
        raise ValueError("Comparison metric K values must be positive")
    if len(set(settings.metrics_top_ks)) != len(settings.metrics_top_ks):
        raise ValueError("Comparison metric K values must be unique")
    if settings.bst_candidate_chunk_size <= 0:
        raise ValueError("BST candidate chunk size must be positive")
    if settings.num_dataloader_workers < 0:
        raise ValueError("DataLoader worker count must be nonnegative")
    if settings.dataloader_prefetch_factor <= 0:
        raise ValueError("DataLoader prefetch factor must be positive")
    if settings.max_classification_metric_pairs <= 0:
        raise ValueError("Classification metric sample size must be positive")
    try:
        device = torch.device(settings.device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid Torch device: {settings.device!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {settings.device!r} was requested but unavailable")


def _close_datasets(
    datasets: dict[str, HydratedBucketedEngagementDataset],
) -> None:
    """Release process-local mmap and Arrow handles before temp-file cleanup."""

    for dataset in datasets.values():
        dataset.close()


def run_model_comparison(
    *,
    dataset: Stage7Artifact,
    models: Sequence[ModelArtifact],
    settings: ComparisonSettings,
    temporary_dir: Path,
    logger: logging.Logger,
) -> ComparisonResult:
    """Evaluate two models sequentially over identical deterministic slates."""

    history_lengths = validate_comparison_contract(
        dataset,
        models,
        settings.max_history_len,
    )
    split_counts_available = dataset.split_query_counts
    _validate_settings(settings, available_splits=set(split_counts_available))
    split_row_counts = {
        split: split_counts_available[split] for split in settings.splits
    }
    skipped_splits = tuple(
        split for split, count in split_row_counts.items() if count == 0
    )
    for split in skipped_splits:
        logger.warning("Requested Stage 7 split %r is empty; skipping", split)
    evaluated_splits = tuple(
        split for split in settings.splits if split not in skipped_splits
    )
    if not evaluated_splits:
        raise ValueError("All requested Stage 7 splits are empty")

    metrics_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    mapping_coverage_by_model: dict[str, dict[str, Any]] = {}
    # Models are evaluated sequentially so a mixed BST/two-tower comparison
    # never keeps both sets of TorchScript weights resident on the GPU.
    for model in models:
        logger.info(
            "Evaluating model %s (%s, %s) with history length %s",
            model.name,
            model.model_type,
            model.artifact_format,
            history_lengths[model.name],
        )
        scorer = create_model_scorer(
            model,
            bst_candidate_chunk_size=settings.bst_candidate_chunk_size,
        )
        model_datasets: dict[str, HydratedBucketedEngagementDataset] = {}
        loaders: dict[str, Any] = {}
        try:
            with temporary_author_index_override(
                stage7_bundle_path=dataset.bundle_path,
                model_author_map_path=model.author_map_path,
                author_table_num_rows=model.author_table_num_rows,
                embedding_count=dataset.embedding_count,
                temporary_dir=temporary_dir,
                allow_extra_columns=model.author_map_allow_extra_columns,
            ) as author_override:
                try:
                    mapping_coverage_by_model[model.name] = author_override.coverage
                    for split in evaluated_splits:
                        split_dataset = HydratedBucketedEngagementDataset(
                            dataset.bundle_path,
                            split=split,
                            max_history_len=history_lengths[model.name],
                            additional_batch_negatives=None,
                            use_post_liker_feature=False,
                            max_post_liker_replay_events_per_post=None,
                            seed=settings.random_seed,
                            logger=logger,
                            post_author_idx_override_path=author_override.path,
                            author_table_num_rows_override=model.author_table_num_rows,
                        )
                        if len(split_dataset) != split_row_counts[split]:
                            raise RuntimeError(
                                f"Stage 7 split {split!r} changed row count during comparison"
                            )
                        model_datasets[split] = split_dataset
                        loader = create_hydrated_data_loader(
                            split_dataset,
                            batch_size=settings.batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=settings.num_dataloader_workers,
                            pin_memory=settings.dataloader_pin_memory,
                            persistent_workers=False,
                            prefetch_factor=settings.dataloader_prefetch_factor,
                            seed=settings.random_seed,
                            resample_candidates_each_epoch=False,
                            tensor_only=True,
                            tensor_batch_kind=(
                                "two_tower"
                                if model.model_type == "two-tower"
                                else "bst"
                            ),
                        )
                        loader.batch_sampler.set_evaluation_mode(True)
                        loaders[split] = loader

                    model_metrics: dict[str, dict[str, Any]] = {}
                    for split in evaluated_splits:
                        evaluation = evaluate_matrix_scorer(
                            scorer,
                            loaders[split],
                            settings.device,
                            list(settings.metrics_top_ks),
                            max_classification_metric_pairs=(
                                settings.max_classification_metric_pairs
                            ),
                            collect_ranking_rows=False,
                            progress_desc=f"{model.name} {split}",
                            disable_progress=settings.disable_progress,
                        )
                        if evaluation["ranking_rows"]:
                            raise RuntimeError(
                                "Aggregate model comparison unexpectedly produced ranking rows"
                            )
                        split_metrics = dict(evaluation["metrics"])
                        split_metrics.update({
                            "classification_metric_sample_seed": 0,
                            "classification_metric_max_sampled_pair_count": (
                                settings.max_classification_metric_pairs
                            ),
                        })
                        if any("recall" in metric.lower() for metric in split_metrics):
                            raise RuntimeError(
                                "Recall metrics are not part of this comparison"
                            )
                        model_metrics[split] = split_metrics
                        logger.info(
                            "Completed model %s split %s: rank_users=%s pairs=%s sampled=%s",
                            model.name,
                            split,
                            split_metrics["rank_metric_user_count"],
                            split_metrics["classification_metric_pair_count"],
                            split_metrics["classification_metric_sampled_pair_count"],
                        )
                    metrics_by_model[model.name] = model_metrics
                finally:
                    # Release every lazy dataset mapping before the temporary
                    # override file is removed by its context manager.
                    loaders.clear()
                    _close_datasets(model_datasets)
        finally:
            scorer.close()
            if torch.device(settings.device).type == "cuda":
                torch.cuda.empty_cache()

    return ComparisonResult(
        metrics_by_model=metrics_by_model,
        mapping_coverage_by_model=mapping_coverage_by_model,
        split_row_counts=split_row_counts,
        skipped_splits=skipped_splits,
        history_lengths=history_lengths,
    )
