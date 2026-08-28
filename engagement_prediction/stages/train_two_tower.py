"""Stage 8: train the canonical two-tower model from the Stage 7 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict

import torch

from engagement_prediction.data import training_index
from engagement_prediction.data.author_indices import AUTHOR_PAD_IDX, AUTHOR_UNK_IDX
from engagement_prediction.data.datasets import (
    HydratedBucketedEngagementDataset,
    create_hydrated_data_loader,
)
from engagement_prediction.data.parquet import find_artifact_path
from engagement_prediction.models.two_tower import TwoTowerModel
from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline.lineage import resolve_recorded_stage_lineage
from engagement_prediction.pipeline.logging import get_stage_logger
from engagement_prediction.training.reporting import (
    write_two_tower_training_history_plot,
)
from engagement_prediction.training.two_tower import (
    run_two_tower_listwise_epoch,
    train_two_tower_model,
)
from engagement_prediction.training.two_tower_export import (
    export_two_tower_checkpoint,
    validate_two_tower_export,
)
from engagement_prediction.training.two_tower_publication import (
    publish_two_tower_to_tracker,
)
from engagement_prediction.training.model_artifacts import (
    write_author_map,
    write_json_atomically,
)
from engagement_prediction.training.runtime import (
    clear_cuda_memory,
    get_device,
    set_random_seeds,
)


STAGE_FOLDER = "08_train_two_tower"


def _load_stage7_summary(stage7_dir: Path) -> Dict[str, Any]:
    summary_path = stage7_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text())
        embedding_model = str(summary["parameters"]["embedding_model"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Stage 7 does not record valid embedding-model metadata: {summary_path}"
        ) from exc
    return {"embedding_model": embedding_model}


def _require_loader_index(bundle_path: Path) -> Dict[str, Any]:
    index_path = bundle_path / "loader_index"
    try:
        return training_index.validate_loader_index(index_path)
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Stage 7 loader_index is missing, corrupt, or unsupported; regenerate Stage 7 "
            "with the current dataset-hydration code"
        ) from exc


def _create_dataset(
    *,
    bundle_path: Path,
    split: str,
    max_history_len: int,
    random_seed: int,
    logger: Any,
) -> HydratedBucketedEngagementDataset:
    dataset = HydratedBucketedEngagementDataset(
        bundle_path,
        split=split,
        max_history_len=max_history_len,
        additional_batch_negatives=None,
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
        tensor_only=True,
        tensor_batch_kind="two_tower",
    )


def _final_metrics(
    *,
    model: TwoTowerModel,
    device: str,
    loaders: Dict[str, Any],
    disable_progress: bool,
    gradient_clip_max_norm: float,
    metrics_top_ks: list[int],
) -> Dict[str, Dict[str, Any]]:
    """Evaluate the reloaded best model with deterministic candidate pools."""

    results: Dict[str, Dict[str, Any]] = {}
    for split_name, loader in loaders.items():
        _, metrics, _ = run_two_tower_listwise_epoch(
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
            max_batches=None,
        )
        results[split_name] = metrics
    return results


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    """Train and publish the canonical two-tower Stage 8 artifact."""

    out_dir = context.new_stage_dir(STAGE_FOLDER, tag=str(args.run_tag or ""))
    logger = get_stage_logger("08_TRAIN_TWO_TOWER", log_file=out_dir / "stage.log")
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
    loader_index_validation = _require_loader_index(bundle_path)

    random_seed = int(args.random_seed)
    max_history_len = int(args.max_history_len)
    output_embedding_dim = int(args.output_embedding_dim)
    batch_size = int(args.batch_size)
    eval_batch_size = int(args.eval_batch_size)
    num_workers = int(args.num_dataloader_workers)
    pin_memory = bool(args.dataloader_pin_memory)
    persistent_workers = bool(args.dataloader_persistent_workers)
    prefetch_factor = int(args.dataloader_prefetch_factor)
    metrics_top_ks = [int(value) for value in args.metrics_top_ks]
    generate_plots = not bool(args.no_plots)
    disable_progress = bool(args.disable_progress)
    if output_embedding_dim <= 0:
        raise ValueError("output_embedding_dim must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_dataloader_workers must be nonnegative")
    if num_workers > 0 and prefetch_factor <= 0:
        raise ValueError("dataloader_prefetch_factor must be positive")

    device = get_device(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device '{device}' was requested but CUDA is unavailable")
    clear_cuda_memory()
    set_random_seeds(random_seed)
    logger.info(
        "Starting native two-tower training: dataset=%s device=%s history_len=%s "
        "train_batch_size=%s eval_batch_size=%s output_embedding_dim=%s "
        "candidate_pool=all_hourly_negatives",
        bundle_path,
        device,
        max_history_len,
        batch_size,
        eval_batch_size,
        output_embedding_dim,
    )

    logger.info("Phase 2/7: opening native train and validation datasets")
    datasets = {
        split: _create_dataset(
            bundle_path=bundle_path,
            split=split,
            max_history_len=max_history_len,
            random_seed=random_seed,
            logger=logger,
        )
        for split in ("train", "val", "val_unseen_users")
    }
    embed_dim = datasets["train"].embed_dim
    author_table_num_rows = datasets["train"].author_table_num_rows
    if embed_dim != int(loader_index_validation["embedding_dim"]):
        raise ValueError("Stage 7 dataset embedding dimension does not match loader_index")
    if author_table_num_rows != int(
        loader_index_validation["author_table_num_rows"]
    ):
        raise ValueError(
            "Stage 7 dataset author vocabulary size does not match loader_index"
        )
    for split, dataset in datasets.items():
        if dataset.embed_dim != embed_dim:
            raise ValueError(
                f"Stage 7 split '{split}' has a different embedding dimension"
            )
        if dataset.author_table_num_rows != author_table_num_rows:
            raise ValueError(
                f"Stage 7 split '{split}' has a different author vocabulary size"
            )

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
        batch_size=eval_batch_size,
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
        batch_size=eval_batch_size,
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
        "user_hidden_dim": int(args.user_hidden_dim),
        "post_hidden_dim": int(args.post_hidden_dim),
        "output_embedding_dim": output_embedding_dim,
        "max_history_len": max_history_len,
        "dropout_rate": float(args.dropout_rate_two_tower),
        "author_unknown_dropout_rate": float(args.author_unknown_dropout_rate),
        "similarity_temperature": float(args.similarity_temperature),
    }
    model_config = {
        "model_type": "two-tower",
        "user_encoder_type": "cross_attention",
        "embedding_model": stage7_metadata["embedding_model"],
        "output_embedding_dim": output_embedding_dim,
        "max_history_len": max_history_len,
        "use_author_embedding_table": True,
        "use_post_encoder": True,
        "l2_normalize_embeddings": True,
        "author_pad_idx": AUTHOR_PAD_IDX,
        "author_unk_idx": AUTHOR_UNK_IDX,
        "constructor_args": constructor_args,
    }
    training_config = {
        "stage7_dir": str(stage7_dir),
        "stage7_bundle": str(bundle_path),
        "loader_index_format_version": int(
            loader_index_validation["format_version"]
        ),
        "lineage": {
            stage_folder: str(stage_dir)
            for stage_folder, stage_dir in lineage.items()
        },
        "random_seed": random_seed,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "candidate_pool": "all_hourly_negatives",
        "output_embedding_dim": output_embedding_dim,
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay_two_tower),
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
        "generate_plots": generate_plots,
    }
    model_config_path = out_dir / "model_config.json"
    training_config_path = out_dir / "training_config.json"
    write_json_atomically(model_config_path, model_config)
    write_json_atomically(training_config_path, training_config)

    authors_source_path = bundle_path / "authors"
    authors_path = out_dir / "authors"
    if not authors_source_path.is_dir():
        raise FileNotFoundError(
            f"Stage 7 authors artifact is missing: {authors_source_path}"
        )
    shutil.copytree(authors_source_path, authors_path)

    logger.info(
        "Phase 3/7: training the canonical two-tower model and exporting every new best"
    )
    model = TwoTowerModel(**constructor_args)
    checkpoints_dir = out_dir / "checkpoints"
    user_tower_path = checkpoints_dir / "engagement_user_tower.pt"
    post_tower_path = checkpoints_dir / "engagement_post_tower.pt"
    checkpoint_metadata = {"model_config": model_config}
    torchscript_exports: list[Dict[str, Any]] = []

    def export_best_checkpoint(checkpoint_path: Path) -> None:
        export = export_two_tower_checkpoint(
            checkpoint_path=checkpoint_path,
            user_tower_path=user_tower_path,
            post_tower_path=post_tower_path,
            expected_model_config=model_config,
        )
        torchscript_exports.append(export)
        logger.info(
            "Published local two-tower TorchScript for best epoch %s: "
            "user_bytes=%s post_bytes=%s parity_cases=%s",
            export["best_epoch"],
            export["user_tower"]["size_bytes"],
            export["post_tower"]["size_bytes"],
            export["parity"]["case_count"],
        )

    training_results = train_two_tower_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_unseen_loader=val_unseen_loader,
        device=device,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay_two_tower),
        patience=int(args.patience),
        early_stopping_min_delta=float(args.early_stopping_min_delta),
        checkpoints_dir=checkpoints_dir,
        disable_progress=disable_progress,
        lr_scheduler_factor=float(args.lr_scheduler_factor),
        lr_scheduler_patience=int(args.lr_scheduler_patience),
        gradient_clip_max_norm=float(args.gradient_clip_max_norm),
        metrics_top_ks=metrics_top_ks,
        max_train_batches_per_epoch=None,
        checkpoint_metadata=checkpoint_metadata,
        best_checkpoint_callback=export_best_checkpoint,
        experiment_tracker=context.tracker,
        logger=logger,
    )
    trained_model: TwoTowerModel = training_results["model"]

    logger.info("Phase 4/7: evaluating the reloaded best model deterministically")
    train_loader.batch_sampler.set_evaluation_mode(True)
    final_metrics = _final_metrics(
        model=trained_model,
        device=device,
        loaders={
            "train": train_loader,
            "val": val_loader,
            "val_unseen_users": val_unseen_loader,
        },
        disable_progress=disable_progress,
        gradient_clip_max_norm=float(args.gradient_clip_max_norm),
        metrics_top_ks=metrics_top_ks,
    )

    logger.info("Phase 5/7: validating final checkpoint and serving artifacts")
    checkpoint_path = checkpoints_dir / "two_tower_best.pth"
    if not checkpoint_path.is_file():
        raise RuntimeError(
            "Two-tower training completed without publishing its best checkpoint"
        )
    if not user_tower_path.is_file() or not post_tower_path.is_file():
        raise RuntimeError(
            "Two-tower training completed without publishing both TorchScript towers"
        )
    final_export_validation = validate_two_tower_export(
        checkpoint_path=checkpoint_path,
        user_tower_path=user_tower_path,
        post_tower_path=post_tower_path,
        expected_model_config=model_config,
    )
    if final_export_validation["best_epoch"] != training_results["best_epoch"]:
        raise RuntimeError(
            "Final two-tower checkpoint and TorchScript best epochs disagree"
        )

    author_map_path = out_dir / "two_tower_author_idx.parquet"
    author_map_stats = write_author_map(
        authors_path=authors_path,
        output_path=author_map_path,
        author_table_num_rows=author_table_num_rows,
    )
    plot_path = None
    if generate_plots:
        plot_path = out_dir / "training_history.png"
        write_two_tower_training_history_plot(
            training_results["history"],
            plot_path,
            training_results["best_epoch"],
        )

    logger.info("Phase 6/7: publishing the final serving set to ClearML")
    publication_started_at = time.time()
    serving_manifest_path = checkpoints_dir / "two_tower_serving_manifest.json"
    clearml_publication = publish_two_tower_to_tracker(
        tracker=context.tracker,
        logger=logger,
        user_tower_path=user_tower_path,
        post_tower_path=post_tower_path,
        author_map_path=author_map_path,
        manifest_path=serving_manifest_path,
        output_embedding_dim=output_embedding_dim,
    )
    clearml_publication["runtime_seconds"] = time.time() - publication_started_at
    if clearml_publication["status"] != "complete":
        logger.warning(
            "Two-tower ClearML publication is %s; validated local artifacts remain available",
            clearml_publication["status"],
        )

    logger.info("Phase 7/7: writing results and reproducibility artifacts")
    local_pipeline_runtime_seconds = publication_started_at - started_at
    runtime_seconds = time.time() - started_at
    export_payload = {
        "export_count": len(torchscript_exports),
        "exported_best_epochs": [
            export["best_epoch"] for export in torchscript_exports
        ],
        "exports": [
            {
                key: value
                for key, value in export.items()
                if key not in {"user_tower_path", "post_tower_path"}
            }
            for export in torchscript_exports
        ],
        "final_validation": final_export_validation,
    }
    result_payload = {
        "output_embedding_dim": output_embedding_dim,
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
        "torchscript_export": export_payload,
        "author_map": {"path": str(author_map_path), **author_map_stats},
        "clearml_publication": clearml_publication,
        "local_pipeline_runtime_seconds": local_pipeline_runtime_seconds,
        "runtime_seconds": runtime_seconds,
    }
    training_results_path = out_dir / "training_results.json"
    write_json_atomically(training_results_path, result_payload)

    summary_path = out_dir / "summary.json"
    summary = {
        "parameters": training_config,
        "input": {
            "dataset_hydration_dir": str(stage7_dir),
            "hydrated_training_data_path": str(bundle_path),
        },
        "model": model_config,
        "results": {
            key: value
            for key, value in result_payload.items()
            if key not in {
                "training_history",
                "final_metrics",
                "torchscript_export",
                "author_map",
                "clearml_publication",
            }
        },
        "final_metrics": final_metrics,
        "outputs": {
            "checkpoint_path": str(checkpoint_path),
            "user_tower_path": str(user_tower_path),
            "post_tower_path": str(post_tower_path),
            "author_idx_path": author_map_path.name,
            "serving_manifest_path": (
                str(serving_manifest_path)
                if serving_manifest_path.is_file()
                else None
            ),
            "model_config_path": model_config_path.name,
            "training_config_path": training_config_path.name,
            "training_results_path": training_results_path.name,
            "authors_path": authors_path.name,
            "training_plot_path": plot_path.name if plot_path else None,
        },
        "torchscript_export": export_payload,
        "author_map": result_payload["author_map"],
        "clearml_publication": clearml_publication,
        "runtime_seconds": runtime_seconds,
    }
    write_json_atomically(summary_path, summary)

    primary_key = f"ndcg@{metrics_top_ks[0]}"
    stage_info_path = out_dir / "stage_info.txt"
    stage_info_lines = [
        "stage: train_two_tower",
        f"runtime_seconds: {runtime_seconds:.2f}",
        f"dataset_hydration_dir: {stage7_dir}",
        f"device: {device}",
        f"embedding_model: {stage7_metadata['embedding_model']}",
        f"post_embedding_dim: {embed_dim}",
        f"output_embedding_dim: {output_embedding_dim}",
        f"author_table_num_rows: {author_table_num_rows}",
        f"train_batch_size: {batch_size}",
        f"eval_batch_size: {eval_batch_size}",
        "candidate_pool: all_hourly_negatives",
        f"best_epoch: {training_results['best_epoch']}",
        f"epochs_completed: {training_results['epochs_completed']}",
        f"stopped_early: {training_results['stopped_early']}",
        f"best_val_metric: {training_results['best_val_metric']:.6f}",
        f"torchscript_export_count: {len(torchscript_exports)}",
        f"user_tower_sha256: {final_export_validation['user_tower']['sha256']}",
        f"post_tower_sha256: {final_export_validation['post_tower']['sha256']}",
        f"two_tower_author_count: {author_map_stats['author_count']}",
        f"clearml_publication_status: {clearml_publication['status']}",
        "user_tower_clearml_model_id: "
        f"{clearml_publication['user_tower_clearml_model_id']}",
        "post_tower_clearml_model_id: "
        f"{clearml_publication['post_tower_clearml_model_id']}",
        f"user_tower_uri: {clearml_publication['user_tower_uri']}",
        f"post_tower_uri: {clearml_publication['post_tower_uri']}",
        f"author_map_uploaded: {clearml_publication['author_map_uploaded']}",
        f"serving_manifest_uploaded: {clearml_publication['manifest_uploaded']}",
        "clearml_publication_errors: "
        f"{json.dumps(clearml_publication['errors'], sort_keys=True)}",
    ]
    for split, metrics in final_metrics.items():
        stage_info_lines.extend([
            f"{split}_loss: {metrics['loss']:.6f}",
            f"{split}_{primary_key}: {metrics[primary_key]:.6f}",
        ])
    stage_info_path.write_text("\n".join(stage_info_lines) + "\n")

    if context.tracker is not None and str(getattr(context.tracker, "id", "") or ""):
        artifact_paths = {
            "two_tower_model_config": model_config_path,
            "two_tower_training_config": training_config_path,
            "two_tower_training_results": training_results_path,
            "two_tower_best_checkpoint": checkpoint_path,
            "two_tower_stage_summary": summary_path,
            "two_tower_stage_info": stage_info_path,
        }
        if plot_path is not None:
            artifact_paths["two_tower_training_history_plot"] = plot_path
        for name, path in artifact_paths.items():
            if not context.tracker.log_file_artifact(name, path):
                logger.warning("Experiment tracker did not upload artifact '%s'", name)

    clear_cuda_memory()
    logger.info(
        "Two-tower training completed in %.2fs: best_epoch=%s best_%s=%.6f",
        runtime_seconds,
        training_results["best_epoch"],
        training_results["primary_metric_name"],
        training_results["best_val_metric"],
    )
    return {
        "output_dir": out_dir,
        "artifacts": {
            "checkpoint_path": str(checkpoint_path),
            "user_tower_path": str(user_tower_path),
            "post_tower_path": str(post_tower_path),
            "two_tower_author_idx_path": str(author_map_path),
            "serving_manifest_path": (
                str(serving_manifest_path)
                if serving_manifest_path.is_file()
                else None
            ),
            "model_config_path": str(model_config_path),
            "training_config_path": str(training_config_path),
            "training_results_path": str(training_results_path),
            "authors_path": str(authors_path),
            "training_plot_path": str(plot_path) if plot_path else None,
        },
        "torchscript_export": export_payload,
        "clearml_publication": clearml_publication,
    }
