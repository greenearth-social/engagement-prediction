"""Evaluate saved BST checkpoints on media-filtered validation slates."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from engagement_prediction.data.datasets import (
    HydratedBucketedEngagementDataset,
    create_hydrated_data_loader,
)
from engagement_prediction.evaluation.artifacts import Stage7Artifact
from engagement_prediction.evaluation.media_artifacts import (
    MediaModelArtifact,
    ValidationSettings,
)
from engagement_prediction.training.bst_export import load_bst_checkpoint_model
from engagement_prediction.training.bst_ranker import BSTRankerMatrixScorer
from engagement_prediction.training.ranking import (
    MatrixRankingScorer,
    empty_ndcg_metric_tensor_sums,
    ndcg_metric_tensor_sums_for_batch,
    topk_ranked_labels_for_scores,
)


MEDIA_TYPES = ("images", "videos", "text_only")
VALIDATION_SPLITS = ("val", "val_unseen_users")
MEDIA_STATUSES = ("known", "missing_document", "missing_flags")


class _MediaLookup:
    """Keep one URI index and compact masks shared across all evaluations."""

    def __init__(self, metadata: pl.DataFrame):
        self.positions = {uri: index for index, uri in enumerate(metadata["at_uri"])}
        if len(self.positions) != metadata.height or None in self.positions:
            raise ValueError("Media metadata must have unique, non-null at_uri values")
        if metadata["media_status"].null_count() or not set(
            metadata["media_status"].unique()
        ).issubset(MEDIA_STATUSES):
            raise ValueError("Media metadata contains invalid media_status values")
        self.statuses = {
            status: (metadata["media_status"] == status).to_numpy()
            for status in MEDIA_STATUSES
        }
        known = self.statuses["known"]
        if metadata.filter(pl.col("media_status") == "known").select(
            pl.any_horizontal(
                pl.col("contains_images").is_null(),
                pl.col("contains_video").is_null(),
            ).any()
        ).item():
            raise ValueError("Known media metadata must include both media flags")
        images = metadata["contains_images"].fill_null(False).to_numpy()
        videos = metadata["contains_video"].fill_null(False).to_numpy()
        self.groups = {
            "images": known & images,
            "videos": known & videos,
            "text_only": known & ~images & ~videos,
        }
        self.requested = {
            split: metadata[f"in_{split}"].to_numpy()
            for split in VALIDATION_SPLITS
        }

    def candidate_positions(self, uris: Sequence[str], *, split: str) -> np.ndarray:
        try:
            positions = np.fromiter(
                (self.positions[uri] for uri in uris), dtype=np.int64, count=len(uris)
            )
        except KeyError as exc:
            raise RuntimeError(
                "Evaluated candidate URI was absent from the hydration snapshot"
            ) from exc
        if not self.requested[split][positions].all():
            raise RuntimeError(f"Evaluated candidate URI was not requested for {split}")
        return positions


def _batch_identity(batch: dict[str, Any]) -> bytes:
    """Fingerprint only query, candidate, and relevance identity, not features."""

    labels = batch["label_matrix"].detach().cpu().contiguous()
    if labels.ndim != 2 or tuple(labels.shape) != (
        len(batch["user_id"]), len(batch["candidate_post_id"])
    ):
        raise RuntimeError("Batch identifiers and relevance-label dimensions disagree")
    if not torch.all((labels == 0) | (labels == 1)):
        raise RuntimeError("Validation relevance labels must be binary")
    identifiers = {
        "users": list(batch["user_id"]),
        "query_hour": batch["query_hour"].isoformat(),
        "candidates": list(batch["candidate_post_id"]),
        "shape": list(labels.shape),
    }
    digest = hashlib.sha256(json.dumps(
        identifiers, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))
    digest.update(labels.to(dtype=torch.uint8).numpy().tobytes())
    return digest.digest()


def _media_ndcg_sums(
    scores: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    metrics_top_ks: tuple[int, ...],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Filter columns after scoring and use canonical binary NDCG helpers."""

    group_scores = scores[:, candidate_mask]
    group_labels = labels[:, candidate_mask]
    if group_scores.size(1) == 0:
        return (
            empty_ndcg_metric_tensor_sums(list(metrics_top_ks), device=scores.device),
            torch.zeros((), dtype=torch.int64, device=scores.device),
        )
    ranked_labels = topk_ranked_labels_for_scores(
        group_scores, group_labels, list(metrics_top_ks)
    )
    return ndcg_metric_tensor_sums_for_batch(
        ranked_labels, group_labels.sum(dim=1), list(metrics_top_ks)
    )


def _evaluate_split(
    scorer: MatrixRankingScorer,
    loader: Any,
    lookup: _MediaLookup,
    *,
    model_name: str,
    split: str,
    metrics_top_ks: tuple[int, ...],
    device: str,
    disable_progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Aggregate a split while retaining no individual predictions."""

    sums = {
        media: empty_ndcg_metric_tensor_sums(list(metrics_top_ks), device=device)
        for media in MEDIA_TYPES
    }
    counts = {
        media: dict.fromkeys((
            "eligible_query_count", "candidate_count", "candidate_pair_count",
            "eligible_candidate_pair_count", "positive_count",
        ), 0)
        for media in MEDIA_TYPES
    }
    occurrences = {status: 0 for status in MEDIA_STATUSES}
    pairs = {status: 0 for status in MEDIA_STATUSES}
    positives = {status: 0 for status in MEDIA_STATUSES}
    seen = np.zeros(len(lookup.positions), dtype=np.bool_)
    identity = hashlib.sha256()
    query_count = 0
    batch_count = 0
    with torch.inference_mode():
        for batch in tqdm(
            loader, desc=f"{model_name} {split}", leave=False, disable=disable_progress
        ):
            identity.update(_batch_identity(batch))
            candidate_positions = lookup.candidate_positions(
                batch["candidate_post_id"], split=split
            )
            seen[candidate_positions] = True
            host_labels = batch["label_matrix"]
            row_count = int(host_labels.size(0))
            query_count += row_count
            batch_count += 1
            scores = scorer.score_batch(batch, device).scores
            labels = host_labels.to(device=device, dtype=torch.float32)
            if scores.shape != labels.shape or not torch.isfinite(scores).all():
                raise RuntimeError(
                    f"Model {model_name!r} produced invalid scores on {split}"
                )
            for status, status_mask in lookup.statuses.items():
                mask = status_mask[candidate_positions]
                count = int(mask.sum())
                occurrences[status] += count
                pairs[status] += count * row_count
                positives[status] += int(host_labels[:, mask].sum().item())
            for media, group in lookup.groups.items():
                host_mask = group[candidate_positions]
                mask = torch.as_tensor(host_mask, dtype=torch.bool, device=device)
                batch_sums, eligible = _media_ndcg_sums(
                    scores, labels, mask, metrics_top_ks=metrics_top_ks
                )
                eligible_count = int(eligible.item())
                candidate_count = int(host_mask.sum())
                counts[media]["eligible_query_count"] += eligible_count
                counts[media]["candidate_count"] += candidate_count
                counts[media]["candidate_pair_count"] += candidate_count * row_count
                counts[media]["eligible_candidate_pair_count"] += (
                    candidate_count * eligible_count
                )
                counts[media]["positive_count"] += int(
                    host_labels[:, host_mask].sum().item()
                )
                for key, value in batch_sums.items():
                    sums[media][key] += value

    metric_rows = []
    for media in MEDIA_TYPES:
        eligible_count = counts[media]["eligible_query_count"]
        for k in metrics_top_ks:
            metric_rows.append({
                "model": model_name,
                "split": split,
                "media_type": media,
                "k": k,
                "ndcg": (
                    sums[media][f"ndcg@{k}"].item() / eligible_count
                    if eligible_count else None
                ),
                "total_query_count": query_count,
                **counts[media],
                "skipped_query_count": query_count - eligible_count,
                "scored_unique_uri_count": int((seen & lookup.groups[media]).sum()),
            })
    requested = lookup.requested[split]
    coverage = {
        "model": model_name,
        "split": split,
        "requested_unique_uri_count": int(requested.sum()),
        "scored_unique_uri_count": int(seen.sum()),
        "total_query_count": query_count,
        "candidate_count": sum(occurrences.values()),
        "candidate_pair_count": sum(pairs.values()),
        "positive_count": sum(positives.values()),
    }
    for status, status_mask in lookup.statuses.items():
        coverage.update({
            f"requested_{status}_unique_uri_count": int((requested & status_mask).sum()),
            f"scored_{status}_unique_uri_count": int((seen & status_mask).sum()),
            f"{status}_candidate_count": occurrences[status],
            f"{status}_candidate_pair_count": pairs[status],
            f"{status}_positive_count": positives[status],
        })
    return metric_rows, coverage, {
        "identity_sha256": identity.hexdigest(),
        "batch_count": batch_count,
        "query_count": query_count,
    }


def run_media_evaluation(
    dataset: Stage7Artifact,
    models: Sequence[MediaModelArtifact],
    validation_settings: ValidationSettings,
    media_metadata: pl.DataFrame,
    *,
    metrics_top_ks: tuple[int, ...],
    device: str,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    disable_progress: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Evaluate checkpoints sequentially on identical native validation batches."""

    if not models or len({model.name for model in models}) != len(models):
        raise ValueError("At least one model with unique names is required")
    if not metrics_top_ks or any(k <= 0 for k in metrics_top_ks):
        raise ValueError("Metric K values must be positive")
    if len(set(metrics_top_ks)) != len(metrics_top_ks):
        raise ValueError("Metric K values must be unique")
    for split in VALIDATION_SPLITS:
        if dataset.split_query_counts.get(split, 0) <= 0:
            raise ValueError(f"Validation split {split!r} must be nonempty")
    lookup = _MediaLookup(media_metadata)
    metrics: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    model_results: list[dict[str, Any]] = []
    reference_identities: dict[str, dict[str, Any]] = {}
    for artifact in models:
        logger.info("Loading best checkpoint for %s", artifact.name)
        eager_model, best_epoch = load_bst_checkpoint_model(
            checkpoint_path=artifact.checkpoint_path,
            expected_model_config=artifact.model_config,
            expected_popularity_stats=artifact.popularity_stats,
        )
        scorer = BSTRankerMatrixScorer(eager_model)
        max_history_len = int(artifact.model_config["max_history_len"])
        use_post_liker_feature = bool(
            artifact.model_config["constructor_args"]["use_post_liker_feature"]
        )
        replay_cap = (
            int(artifact.training_config["bst_max_post_liker_replay_events_per_post"])
            if use_post_liker_feature else None
        )
        model_result = {
            "model": artifact.name,
            "best_epoch": best_epoch,
            "max_history_len": max_history_len,
            "use_post_liker_feature": use_post_liker_feature,
            "max_post_liker_replay_events_per_post": replay_cap,
            "splits": {},
        }
        try:
            scorer.prepare_for_eval(device)
            for split in VALIDATION_SPLITS:
                split_dataset = HydratedBucketedEngagementDataset(
                    dataset.bundle_path,
                    split=split,
                    max_history_len=max_history_len,
                    additional_batch_negatives=validation_settings.additional_batch_negatives,
                    use_post_liker_feature=use_post_liker_feature,
                    max_post_liker_replay_events_per_post=replay_cap,
                    seed=validation_settings.random_seed,
                    logger=logger,
                )
                loader = None
                try:
                    if len(split_dataset) != dataset.split_query_counts[split]:
                        raise RuntimeError(f"Validation split {split!r} changed row count")
                    loader = create_hydrated_data_loader(
                        split_dataset,
                        batch_size=validation_settings.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        persistent_workers=False,
                        prefetch_factor=prefetch_factor,
                        seed=validation_settings.random_seed,
                        resample_candidates_each_epoch=False,
                        tensor_only=False,
                        tensor_batch_kind="bst",
                    )
                    loader.batch_sampler.set_evaluation_mode(True)
                    split_metrics, split_coverage, identity = _evaluate_split(
                        scorer, loader, lookup,
                        model_name=artifact.name,
                        split=split,
                        metrics_top_ks=metrics_top_ks,
                        device=device,
                        disable_progress=disable_progress,
                    )
                    if identity["query_count"] != dataset.split_query_counts[split]:
                        raise RuntimeError(f"Incomplete validation evaluation for {split}")
                    if split in reference_identities and identity != reference_identities[split]:
                        raise RuntimeError(
                            "Validation query/candidate/relevance sequence differs "
                            f"between models {models[0].name!r} and {artifact.name!r} "
                            f"on {split}"
                        )
                    reference_identities[split] = identity
                    model_result["splits"][split] = identity
                    metrics.extend(split_metrics)
                    coverage.append(split_coverage)
                    logger.info(
                        "Completed %s %s: %s queries; %s unique candidates",
                        artifact.name, split, identity["query_count"],
                        split_coverage["scored_unique_uri_count"],
                    )
                finally:
                    del loader
                    split_dataset.close()
            model_results.append(model_result)
        finally:
            del scorer, eager_model
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
    return {
        "metrics": metrics,
        "hydration_coverage": coverage,
        "model_results": model_results,
    }


def write_media_plots(
    metrics_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    model_names: Sequence[str],
    metrics_top_ks: Sequence[int],
) -> None:
    """Save one three-panel comparison figure for each validation split."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    labels = {"images": "Images", "videos": "Videos", "text_only": "Text only"}
    markers = ("o", "s", "^", "D", "v", "P", "X")
    ks = sorted(metrics_top_ks)
    for split in VALIDATION_SPLITS:
        figure = Figure(figsize=(14, 4.5), constrained_layout=True)
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 3, sharey=True)
        for axis, media in zip(axes, MEDIA_TYPES):
            has_data = False
            for model_index, model_name in enumerate(model_names):
                values = {
                    row["k"]: row["ndcg"] for row in metrics_rows
                    if row["model"] == model_name and row["split"] == split
                    and row["media_type"] == media
                }
                ys = [values.get(k) for k in ks]
                has_data = has_data or any(value is not None for value in ys)
                axis.plot(
                    ks, [float("nan") if y is None else y for y in ys],
                    label=model_name, marker=markers[model_index % len(markers)],
                    linestyle="-" if len(ks) > 1 else "None",
                )
            if not has_data:
                axis.text(
                    0.5, 0.5, "No eligible queries", ha="center", va="center",
                    transform=axis.transAxes,
                )
            axis.set_title(labels[media])
            axis.set_xlabel("K")
            axis.set_xticks(ks)
            axis.set_ylim(0, 1.02)
            axis.grid(True, alpha=0.25)
            axis.legend()
        axes[0].set_ylabel("NDCG@K")
        figure.suptitle(split)
        figure.savefig(Path(output_dir) / f"{split}_ndcg.png", dpi=150)
        figure.clear()
