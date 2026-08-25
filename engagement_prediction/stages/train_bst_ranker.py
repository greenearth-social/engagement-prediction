"""Stage 8: train the canonical BST ranker from the Stage 7 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict

import torch

from engagement_prediction.data.datasets import (
    HydratedBucketedEngagementDataset,
    create_hydrated_data_loader,
)
from engagement_prediction.data.parquet import find_artifact_path, scan_parquet_artifact
from engagement_prediction.models.bst_ranker import BSTRanker
from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline.lineage import resolve_recorded_stage_lineage
from engagement_prediction.training.bst_ranker import (
    run_bst_listwise_epoch,
    train_bst_ranker_model,
)
from engagement_prediction.training.popularity import fit_popularity_normalization
from engagement_prediction.training.reporting import write_bst_training_history_plot
from shared.input_data_helpers import AUTHOR_PAD_IDX, AUTHOR_UNK_IDX
from utils.helpers import clear_cuda_memory, get_device, get_stage_logger, set_random_seeds


STAGE_FOLDER = "08_train_bst_ranker"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write deterministic JSON and reject nonportable NaN/Infinity values."""

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _load_stage7_summary(stage7_dir: Path) -> Dict[str, Any]:
    """Load the Stage 7 model-independent dataset metadata."""

    summary_path = stage7_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text())
        parameters = summary["parameters"]
        embedding_model = str(parameters["embedding_model"])
        embedding_dim = int(parameters["embedding_dim"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Stage 7 does not record valid embedding metadata: {summary_path}"
        ) from exc
    if embedding_dim <= 0:
        raise ValueError("Stage 7 embedding_dim must be positive")
    return {
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
    }


def _create_dataset(
    *,
    bundle_path: Path,
    split: str,
    max_history_len: int,
    additional_negatives: int,
    random_seed: int,
    logger: Any,
) -> HydratedBucketedEngagementDataset:
    dataset = HydratedBucketedEngagementDataset(
        bundle_path,
        split=split,
        max_history_len=max_history_len,
        bst_additional_batch_negatives=additional_negatives,
        seed=random_seed,
        logger=logger,
    )
    if len(dataset) == 0:
        raise ValueError(f"Stage 8 requires a nonempty '{split}' split")
    return dataset


def _create_loader(
    *,
    dataset: HydratedBucketedEngagementDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    random_seed: int,
    resample_candidates_each_epoch: bool,
):
    return create_hydrated_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        seed=random_seed,
        resample_candidates_each_epoch=resample_candidates_each_epoch,
    )


def _final_metrics(
    *,
    model: BSTRanker,
    device: str,
    loaders: Dict[str, Any],
    disable_progress: bool,
    gradient_clip_max_norm: float,
    metrics_top_ks: list[int],
    max_train_batches: int | None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate the reloaded best state on deterministic split loaders."""

    results: Dict[str, Dict[str, Any]] = {}
    for split_name, loader in loaders.items():
        _, metrics, _ = run_bst_listwise_epoch(
            train=False,
            split_name=f"Final {split_name}",
            model=model,
            device=device,
            dataloader=loader,
            optimizer=None,
            disable_progress=disable_progress,
            gradient_clip_max_norm=gradient_clip_max_norm,
            metrics_top_ks=metrics_top_ks,
            calc_baseline_metrics=False,
            max_batches=max_train_batches if split_name == "train" else None,
        )
        results[split_name] = metrics
    return results


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    """Train and publish the canonical BST model-training artifact."""

    run_tag = str(args.run_tag or "")
    out_dir = context.new_stage_dir(STAGE_FOLDER, tag=run_tag)
    logger = get_stage_logger("08_TRAIN_BST_RANKER", log_file=out_dir / "stage.log")
    started_at = time.time()

    logger.info("Phase 1/7: resolving and validating Stage 0-7 lineage")
    lineage = resolve_recorded_stage_lineage(
        context,
        terminal_stage_folder="07_dataset_hydration",
        ancestor_stage_folders=(
            "00_source_metadata",
            "01_query_selection",
            "02_user_history",
            "03_post_selection",
            "04_negative_selection",
            "05_post_liker_history",
            "06_author_statistics",
        ),
    )
    stage7_dir = lineage["07_dataset_hydration"]
    bundle_path = find_artifact_path(stage7_dir, "hydrated_training_data_")
    stage7_metadata = _load_stage7_summary(stage7_dir)

    random_seed = int(args.random_seed)
    max_history_len = int(args.max_history_len)
    additional_negatives = int(args.bst_additional_batch_negatives)
    batch_size = int(args.batch_size)
    num_workers = int(args.num_dataloader_workers)
    pin_memory = bool(args.dataloader_pin_memory)
    persistent_workers = bool(args.dataloader_persistent_workers)
    prefetch_factor = int(args.dataloader_prefetch_factor)
    metrics_top_ks = [int(value) for value in args.metrics_top_ks]
    use_popularity_feature = bool(args.bst_use_popularity_feature)
    save_model = not bool(args.no_save_model)
    generate_plots = not bool(args.no_plots)
    disable_progress = bool(args.disable_progress)
    max_train_batches = args.bst_max_train_batches_per_epoch
    if max_train_batches is not None:
        max_train_batches = int(max_train_batches)
    if additional_negatives <= 0:
        raise ValueError("bst_additional_batch_negatives must be positive")
    if num_workers < 0:
        raise ValueError("num_dataloader_workers must be nonnegative")
    if num_workers > 0 and prefetch_factor <= 0:
        raise ValueError("dataloader_prefetch_factor must be positive")

    device = get_device(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device '{device}' was requested but CUDA is unavailable")
    set_random_seeds(random_seed)
    logger.info(
        "Starting native BST training: dataset=%s device=%s history_len=%s "
        "batch_size=%s negatives_per_batch_pool=%s popularity=%s",
        bundle_path,
        device,
        max_history_len,
        batch_size,
        additional_negatives,
        use_popularity_feature,
    )

    logger.info("Phase 2/7: fitting training-only popularity normalization")
    queries_lf = scan_parquet_artifact(bundle_path / "queries")
    positives_lf = scan_parquet_artifact(bundle_path / "query_positives")
    histories_lf = scan_parquet_artifact(bundle_path / "query_histories")
    negatives_lf = scan_parquet_artifact(bundle_path / "hourly_negative_candidates")
    popularity_stats = fit_popularity_normalization(
        queries_lf=queries_lf,
        query_positives_lf=positives_lf,
        query_histories_lf=histories_lf,
        hourly_negative_candidates_lf=negatives_lf,
        enabled=use_popularity_feature,
    )
    logger.info(
        "Popularity normalization: mean=%.6f std=%.6f histories=%s "
        "candidates=%s total=%s",
        popularity_stats.log_mean,
        popularity_stats.log_std,
        f"{popularity_stats.history_observation_count:,}",
        f"{popularity_stats.candidate_observation_count:,}",
        f"{popularity_stats.total_observation_count:,}",
    )

    logger.info("Phase 3/7: loading native train, validation, and unseen-validation datasets")
    datasets = {
        split: _create_dataset(
            bundle_path=bundle_path,
            split=split,
            max_history_len=max_history_len,
            additional_negatives=additional_negatives,
            random_seed=random_seed,
            logger=logger,
        )
        for split in ("train", "val", "val_unseen_users")
    }
    embed_dim = datasets["train"].embed_dim
    author_table_num_rows = datasets["train"].author_table_num_rows
    if embed_dim != stage7_metadata["embedding_dim"]:
        raise ValueError(
            "Stage 7 summary embedding_dim does not match embeddings.npy"
        )
    for split, dataset in datasets.items():
        if dataset.embed_dim != embed_dim:
            raise ValueError(f"Stage 7 split '{split}' has a different embedding dimension")
        if dataset.author_table_num_rows != author_table_num_rows:
            raise ValueError(f"Stage 7 split '{split}' has a different author vocabulary size")

    train_loader = _create_loader(
        dataset=datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        random_seed=random_seed,
        resample_candidates_each_epoch=True,
    )
    val_loader = _create_loader(
        dataset=datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        random_seed=random_seed,
        resample_candidates_each_epoch=False,
    )
    val_unseen_loader = _create_loader(
        dataset=datasets["val_unseen_users"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        random_seed=random_seed,
        resample_candidates_each_epoch=False,
    )

    constructor_args = {
        "post_embedding_dim": embed_dim,
        "author_table_num_rows": author_table_num_rows,
        "author_embedding_dim": int(args.author_embedding_dim),
        "content_projection_dim": int(args.content_projection_dim),
        "author_projection_dim": int(args.author_projection_dim),
        "model_dim": int(args.bst_model_dim),
        "time_embedding_dim": int(args.bst_time_embedding_dim),
        "num_attention_heads": int(args.bst_num_attention_heads),
        "num_transformer_layers": int(args.bst_num_transformer_layers),
        "transformer_ff_dim": int(args.bst_transformer_ff_dim),
        "dropout_rate": float(args.bst_dropout_rate),
        "author_unknown_dropout_rate": float(args.author_unknown_dropout_rate),
        "norm_first": bool(args.bst_norm_first),
        "time_delta_bucket_boundaries_hours": [
            float(value) for value in args.bst_time_delta_bucket_boundaries_hours
        ],
        "prediction_hidden_dims": [int(value) for value in args.prediction_hidden_dims],
        "use_popularity_feature": use_popularity_feature,
        "popularity_projection_dim": int(args.bst_popularity_projection_dim),
        "popularity_log_mean": popularity_stats.log_mean,
        "popularity_log_std": popularity_stats.log_std,
    }
    model_config = {
        "model_type": "bst-ranker",
        "embedding_model": stage7_metadata["embedding_model"],
        "max_history_len": max_history_len,
        "author_pad_idx": AUTHOR_PAD_IDX,
        "author_unk_idx": AUTHOR_UNK_IDX,
        "constructor_args": constructor_args,
    }
    training_config = {
        "stage7_dir": str(stage7_dir),
        "stage7_bundle": str(bundle_path),
        "lineage": {
            stage_folder: str(stage_dir)
            for stage_folder, stage_dir in lineage.items()
        },
        "random_seed": random_seed,
        "batch_size": batch_size,
        "bst_additional_batch_negatives": additional_negatives,
        "bst_max_train_batches_per_epoch": max_train_batches,
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.bst_weight_decay),
        "patience": int(args.patience),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "lr_scheduler_factor": float(args.lr_scheduler_factor),
        "lr_scheduler_patience": int(args.lr_scheduler_patience),
        "gradient_clip_max_norm": float(args.gradient_clip_max_norm),
        "metrics_top_ks": metrics_top_ks,
        "num_dataloader_workers": num_workers,
        "dataloader_pin_memory": pin_memory,
        "dataloader_persistent_workers": persistent_workers,
        "dataloader_prefetch_factor": prefetch_factor,
        "device": device,
        "save_model": save_model,
        "generate_plots": generate_plots,
    }
    popularity_payload = popularity_stats.to_dict()
    model_config_path = out_dir / "model_config.json"
    training_config_path = out_dir / "training_config.json"
    popularity_stats_path = out_dir / "popularity_stats.json"
    _write_json(model_config_path, model_config)
    _write_json(training_config_path, training_config)
    _write_json(popularity_stats_path, popularity_payload)

    authors_source_path = bundle_path / "authors"
    authors_path = out_dir / "authors"
    if not authors_source_path.is_dir():
        raise FileNotFoundError(f"Stage 7 authors artifact is missing: {authors_source_path}")
    shutil.copytree(authors_source_path, authors_path)

    logger.info("Phase 4/7: constructing and training the canonical BST ranker")
    model = BSTRanker(**constructor_args)
    checkpoints_dir = out_dir / "checkpoints" if save_model else None
    checkpoint_metadata = {
        "model_config": model_config,
        "popularity_stats": popularity_payload,
    }
    training_results = train_bst_ranker_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_unseen_loader=val_unseen_loader,
        device=device,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.bst_weight_decay),
        patience=int(args.patience),
        early_stopping_min_delta=float(args.early_stopping_min_delta),
        checkpoints_dir=checkpoints_dir,
        disable_progress=disable_progress,
        lr_scheduler_factor=float(args.lr_scheduler_factor),
        lr_scheduler_patience=int(args.lr_scheduler_patience),
        gradient_clip_max_norm=float(args.gradient_clip_max_norm),
        metrics_top_ks=metrics_top_ks,
        bst_max_train_batches_per_epoch=max_train_batches,
        checkpoint_metadata=checkpoint_metadata,
        experiment_tracker=context.tracker,
        logger=logger,
    )
    trained_model: BSTRanker = training_results["model"]

    logger.info("Phase 5/7: evaluating the reloaded best model deterministically")
    final_train_loader = _create_loader(
        dataset=datasets["train"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        random_seed=random_seed,
        resample_candidates_each_epoch=False,
    )
    final_metrics = _final_metrics(
        model=trained_model,
        device=device,
        loaders={
            "train": final_train_loader,
            "val": val_loader,
            "val_unseen_users": val_unseen_loader,
        },
        disable_progress=disable_progress,
        gradient_clip_max_norm=float(args.gradient_clip_max_norm),
        metrics_top_ks=metrics_top_ks,
        max_train_batches=max_train_batches,
    )

    logger.info("Phase 6/7: writing training results and optional plot")
    runtime_seconds = time.time() - started_at
    result_payload = {
        "primary_metric_name": training_results["primary_metric_name"],
        "best_val_metric": training_results["best_val_metric"],
        "best_val_loss": training_results["best_val_loss"],
        "best_epoch": training_results["best_epoch"],
        "epochs_completed": training_results["epochs_completed"],
        "stopped_early": training_results["stopped_early"],
        "patience_counter": training_results["patience_counter"],
        "baseline_metrics": training_results["baseline_metrics"],
        "final_metrics": final_metrics,
        "training_history": training_results["history"],
        "split_query_counts": {
            split: len(dataset) for split, dataset in datasets.items()
        },
        "runtime_seconds": runtime_seconds,
    }
    training_results_path = out_dir / "training_results.json"
    _write_json(training_results_path, result_payload)

    plot_path = None
    if generate_plots:
        plot_path = out_dir / "training_history.png"
        write_bst_training_history_plot(
            training_results["history"],
            plot_path,
            training_results["best_epoch"],
        )

    checkpoint_path = (
        checkpoints_dir / "bst_ranker_best.pth"
        if checkpoints_dir is not None
        else None
    )
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise RuntimeError("BST training completed without publishing its best checkpoint")

    summary = {
        "parameters": training_config,
        "input": {
            "dataset_hydration_dir": str(stage7_dir),
            "hydrated_training_data_path": str(bundle_path),
        },
        "model": model_config,
        "popularity": popularity_payload,
        "results": {
            key: value
            for key, value in result_payload.items()
            if key not in {"training_history", "final_metrics"}
        },
        "final_metrics": final_metrics,
        "outputs": {
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "model_config_path": model_config_path.name,
            "training_config_path": training_config_path.name,
            "popularity_stats_path": popularity_stats_path.name,
            "training_results_path": training_results_path.name,
            "authors_path": authors_path.name,
            "training_plot_path": plot_path.name if plot_path else None,
        },
        "runtime_seconds": runtime_seconds,
    }
    _write_json(out_dir / "summary.json", summary)

    primary_key = f"ndcg@{metrics_top_ks[0]}"
    stage_info_lines = [
        "stage: train_bst_ranker",
        f"runtime_seconds: {runtime_seconds:.2f}",
        f"dataset_hydration_dir: {stage7_dir}",
        f"device: {device}",
        f"embedding_model: {stage7_metadata['embedding_model']}",
        f"embedding_dim: {embed_dim}",
        f"author_table_num_rows: {author_table_num_rows}",
        f"best_epoch: {training_results['best_epoch']}",
        f"epochs_completed: {training_results['epochs_completed']}",
        f"stopped_early: {training_results['stopped_early']}",
        f"best_val_metric: {training_results['best_val_metric']:.6f}",
        f"popularity_log_mean: {popularity_stats.log_mean:.6f}",
        f"popularity_log_std: {popularity_stats.log_std:.6f}",
    ]
    for split, metrics in final_metrics.items():
        stage_info_lines.extend([
            f"{split}_loss: {metrics['loss']:.6f}",
            f"{split}_{primary_key}: {metrics[primary_key]:.6f}",
        ])
    (out_dir / "stage_info.txt").write_text("\n".join(stage_info_lines) + "\n")

    logger.info("Phase 7/7: attaching reproducibility artifacts to the experiment")
    if context.tracker is not None:
        artifact_paths = {
            "bst_model_config": model_config_path,
            "bst_training_config": training_config_path,
            "bst_popularity_stats": popularity_stats_path,
            "bst_training_results": training_results_path,
        }
        if checkpoint_path is not None:
            artifact_paths["bst_ranker_best_checkpoint"] = checkpoint_path
        if plot_path is not None:
            artifact_paths["bst_training_history_plot"] = plot_path
        for name, path in artifact_paths.items():
            if not context.tracker.log_file_artifact(name, path):
                logger.warning("Experiment tracker did not upload artifact '%s'", name)

    clear_cuda_memory()
    logger.info(
        "BST training completed in %.2fs: best_epoch=%s best_%s=%.6f",
        runtime_seconds,
        training_results["best_epoch"],
        training_results["primary_metric_name"],
        training_results["best_val_metric"],
    )
    return {
        "output_dir": out_dir,
        "artifacts": {
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "model_config_path": str(model_config_path),
            "training_config_path": str(training_config_path),
            "popularity_stats_path": str(popularity_stats_path),
            "training_results_path": str(training_results_path),
            "authors_path": str(authors_path),
            "training_plot_path": str(plot_path) if plot_path else None,
        },
    }
