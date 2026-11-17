"""Power-liker bias diagnostics for a trained BST ranker.

Ports the synthetic-feed evaluation of the Power Likers paper (Schulz et al.)
to saved Stage 7/8 outputs, and reports holdout ranking quality by like volume.

Synthetic feed. Every holdout user with at least ``min_user_likes`` liked
holdout posts that carry trait inferences receives the ranker's top ``top_k``
posts from one shared pool, scored with the user's longest holdout history.
The pool is ``pool_size`` posts sampled from the random-sourced Stage 4
negatives of all query hours, among posts with an inference row; each post
keeps one of its hours, chosen at random, for its as-of like count. Per user
and trait, in probability units:

    model_amp    = mean(feed) - mean(liked)    # over-serving
    user_pref    = mean(liked) - mean(pool)
    model_excess = mean(feed) - mean(pool)

Like volume. Each user's raw like count over the Stage 1 like snapshot.
"High volume" users are the smallest set of eligible users who give half of
the eligible users' likes (the paper's power-likers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import math
from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl
from scipy.stats import t as student_t
import torch

from engagement_prediction.data.datasets import (
    HydratedBucketedEngagementDataset,
    create_hydrated_data_loader,
)
from engagement_prediction.data.ingex import scan_parquet_files
from engagement_prediction.evaluation.artifacts import Stage7Artifact
from engagement_prediction.training.ranking import MatrixRankingScorer, ranking_rows_for_batch


HOLDOUT_SPLITS = ("holdout_seen_users", "holdout_unseen_users")
HEADLINE_TRAITS = ("sentiment::Negative", "topic::News & Social Concern")
INFERENCE_TEXT_PATH = ("text", "message.commit.record.text")
HISTORY_KEYS = (
    "history_embeddings",
    "history_mask",
    "history_time_deltas_hours",
    "history_author_indices",
    "history_prior_cumulative_likes",
)


@dataclass(frozen=True)
class PowerLikerSettings:
    """Runtime settings supplied explicitly by the standalone CLI."""

    pool_size: int
    top_k: int
    min_user_likes: int
    metrics_top_k: int
    random_seed: int
    batch_size: int
    device: str
    num_dataloader_workers: int
    dataloader_prefetch_factor: int


@dataclass
class HoldoutScan:
    """Per-query metrics, liked post URIs, and the longest history per user."""

    ranking_rows: list[dict[str, Any]] = field(default_factory=list)
    liked: dict[str, set[str]] = field(default_factory=dict)
    histories: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)


@dataclass(frozen=True)
class PowerLikerResult:
    summary: dict[str, Any]
    ranking_rows: pl.DataFrame
    headline_per_user: pl.DataFrame


def scan_holdout(
    scorer: MatrixRankingScorer,
    dataset: Stage7Artifact,
    *,
    max_history_len: int,
    settings: PowerLikerSettings,
    logger: logging.Logger,
) -> HoldoutScan:
    """Score every holdout query against all of its hour's stored candidates."""

    scan = HoldoutScan()
    scorer.prepare_for_eval(settings.device)
    with torch.inference_mode():
        for split in HOLDOUT_SPLITS:
            split_dataset = HydratedBucketedEngagementDataset(
                dataset.bundle_path,
                split=split,
                max_history_len=max_history_len,
                additional_batch_negatives=None,
                use_post_liker_feature=False,
                max_post_liker_replay_events_per_post=None,
                seed=settings.random_seed,
                logger=logger,
            )
            try:
                loader = create_hydrated_data_loader(
                    split_dataset,
                    batch_size=settings.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=settings.num_dataloader_workers,
                    pin_memory=False,
                    persistent_workers=False,
                    prefetch_factor=settings.dataloader_prefetch_factor,
                    seed=settings.random_seed,
                    resample_candidates_each_epoch=False,
                )
                loader.batch_sampler.set_evaluation_mode(True)
                for batch in loader:
                    scores = scorer.score_batch(batch, settings.device).scores
                    labels = batch["label_matrix"].to(dtype=torch.float32)
                    rows = ranking_rows_for_batch(batch, scores, labels, [settings.metrics_top_k])
                    candidates = [str(uri) for uri in batch["candidate_post_id"]]
                    lengths = batch["history_mask"].sum(dim=1)
                    for index, row in enumerate(rows):
                        did = row["did"]
                        row["split"] = split
                        positives = torch.nonzero(labels[index] > 0).flatten().tolist()
                        scan.liked.setdefault(did, set()).update(candidates[j] for j in positives)
                        best = scan.histories.get(did)
                        if best is None or int(lengths[index]) > int(best["history_mask"].sum()):
                            scan.histories[did] = {key: batch[key][index].clone() for key in HISTORY_KEYS}
                    scan.ranking_rows.extend(rows)
            finally:
                split_dataset.close()
            logger.info("Scored %s: %d holdout queries so far", split, len(scan.ranking_rows))
    return scan


def pool_candidates(dataset: Stage7Artifact, random_seed: int) -> pl.DataFrame:
    """Random-sourced Stage 4 negatives of all hours, one row per post at a random hour."""

    frame = (
        pl.scan_parquet(str(dataset.bundle_path / "hourly_negative_candidates" / "**" / "*.parquet"))
        .filter(pl.col("selection_source") == "random")
        .select("subject_uri", "emb_idx", "author_idx", "prior_like_count", "query_hour")
        .sort("subject_uri", "query_hour")
        .collect()
    )
    return (
        frame.sample(fraction=1.0, shuffle=True, seed=random_seed)
        .unique(subset=["subject_uri"], keep="first", maintain_order=True)
        .sort("subject_uri")
    )


def _scan_inferences(inference_paths: Sequence[str], uris: Iterable[str]) -> pl.LazyFrame:
    wanted = pl.Series("at_uri", sorted(set(uris)), dtype=pl.Utf8)
    return scan_parquet_files(inference_paths).filter(pl.col("at_uri").is_in(wanted.implode()))


def inference_uris(inference_paths: Sequence[str], uris: Iterable[str]) -> set[str]:
    """The subset of ``uris`` with a row in the inference exports."""

    frame = _scan_inferences(inference_paths, uris).select("at_uri").unique().collect(engine="streaming")
    return set(frame["at_uri"].to_list())


def load_traits(inference_paths: Sequence[str], uris: Iterable[str]) -> dict[str, dict[str, float]]:
    """Map post URI -> {"group::label": probability} for the post text."""

    frame = (
        _scan_inferences(inference_paths, uris)
        .select("at_uri", "inferences")
        .unique(subset=["at_uri"], keep="first")
        .collect(engine="streaming")
    )
    traits: dict[str, dict[str, float]] = {}
    for uri, raw in frame.iter_rows():
        node: Any = json.loads(raw)
        for key in INFERENCE_TEXT_PATH:
            node = node.get(key) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            continue
        flat = {
            f"{group}::{label}": float(value)
            for group, labels in node.items() if isinstance(labels, dict)
            for label, value in labels.items() if isinstance(value, (int, float))
        }
        if flat:
            traits[uri] = flat
    return traits


def like_counts(like_paths: Sequence[str], start: datetime, end: datetime) -> dict[str, int]:
    """Raw like rows per user inside the half-open snapshot window."""

    frame = (
        scan_parquet_files(like_paths)
        .select("did", "record_created_at")
        .with_columns(pl.col("record_created_at").str.to_datetime(time_zone="UTC", strict=False).alias("t"))
        .filter((pl.col("t") >= start) & (pl.col("t") < end))
        .group_by("did")
        .agg(pl.len().alias("n"))
        .collect(engine="streaming")
    )
    return dict(zip(frame["did"].to_list(), frame["n"].to_list()))


def givers_of_half(user_ids: Sequence[str], volume: dict[str, int]) -> set[str]:
    """Smallest set of top-volume users giving at least half of the group's likes."""

    total = sum(volume.get(user, 0) for user in user_ids)
    high: set[str] = set()
    cumulative = 0
    for user in sorted(user_ids, key=lambda u: volume.get(u, 0), reverse=True):
        if total == 0 or cumulative >= total * 0.5:
            break
        cumulative += volume.get(user, 0)
        high.add(user)
    return high


def synthetic_feeds(
    scorer: MatrixRankingScorer,
    dataset: Stage7Artifact,
    *,
    pool: pl.DataFrame,
    histories: dict[str, dict[str, torch.Tensor]],
    users: Sequence[str],
    max_history_len: int,
    settings: PowerLikerSettings,
) -> dict[str, np.ndarray]:
    """Top-k pool row indices per user."""

    split_dataset = HydratedBucketedEngagementDataset(
        dataset.bundle_path,
        split=HOLDOUT_SPLITS[0],
        max_history_len=max_history_len,
        use_post_liker_feature=False,
        max_post_liker_replay_events_per_post=None,
        seed=settings.random_seed,
        logger=None,
    )
    try:
        emb_idx = pool["emb_idx"].to_numpy().astype(np.int64)
        candidates = {
            "candidate_post_embeddings": torch.from_numpy(np.asarray(split_dataset.embeddings[emb_idx], dtype=np.float32)),
            "candidate_post_author_idx": torch.from_numpy(pool["author_idx"].to_numpy().astype(np.int64)),
            "candidate_prior_cumulative_likes": torch.from_numpy(pool["prior_like_count"].to_numpy().astype(np.float32)),
        }
    finally:
        split_dataset.close()
    k = min(settings.top_k, pool.height)
    feeds: dict[str, np.ndarray] = {}
    scorer.prepare_for_eval(settings.device)
    with torch.inference_mode():
        for start in range(0, len(users), settings.batch_size):
            chunk = users[start:start + settings.batch_size]
            batch: dict[str, Any] = {key: torch.stack([histories[u][key] for u in chunk]) for key in HISTORY_KEYS}
            batch.update(candidates)
            scores = scorer.score_batch(batch, settings.device).scores.float()
            top = torch.topk(scores, k, dim=1).indices.cpu().numpy()
            for index, did in enumerate(chunk):
                feeds[did] = top[index]
    return feeds


def decompose(
    pool_values: np.ndarray,
    liked_means: dict[str, float],
    feed_means: dict[str, float],
    users: Iterable[str],
) -> dict[str, Any] | None:
    """Per-trait decomposition in probability units, as in the paper."""

    pool_values = pool_values[np.isfinite(pool_values)]
    if pool_values.size < 20:
        return None
    pool_mean = float(pool_values.mean())
    pool_sd = float(pool_values.std(ddof=1))
    keep = [u for u in users if u in liked_means and u in feed_means]
    if len(keep) < 10 or pool_sd < 1e-12:
        return None
    liked = np.array([liked_means[u] for u in keep])
    feed = np.array([feed_means[u] for u in keep])

    def stats(values: np.ndarray, name: str) -> dict[str, float]:
        # One-sample t-test against zero.
        mean = float(values.mean())
        se = float(values.std(ddof=1)) / math.sqrt(values.size)
        p = float(2.0 * student_t.sf(abs(mean / se), values.size - 1)) if se > 0 else math.nan
        return {f"mean_{name}": mean, f"se_{name}": se, f"p_{name}": p}

    return {
        "n_users": len(keep), "pool_mean": pool_mean, "pool_sd": pool_sd,
        **stats(feed - liked, "model_amp"), **stats(liked - pool_mean, "user_pref"),
        **stats(feed - pool_mean, "model_excess"),
        "per_user_amp": dict(zip(keep, (feed - liked).tolist())),
    }


def _ranking_summary(frame: pl.DataFrame, metric: str) -> dict[str, Any]:
    per_user = frame.group_by("did").agg(pl.col("auc_roc").mean(), pl.col(metric).mean())
    return {
        "n_queries": frame.height,
        "n_users": per_user.height,
        "query_mean_auc": frame["auc_roc"].mean(),
        f"query_mean_{metric}": frame[metric].mean(),
        "user_mean_auc": per_user["auc_roc"].mean(),
        f"user_mean_{metric}": per_user[metric].mean(),
    }


def run_power_liker_evaluation(
    scorer: MatrixRankingScorer,
    dataset: Stage7Artifact,
    *,
    max_history_len: int,
    inference_paths: Sequence[str],
    like_paths: Sequence[str],
    like_window: tuple[datetime, datetime],
    settings: PowerLikerSettings,
    logger: logging.Logger,
) -> PowerLikerResult:
    """Run the synthetic-feed decomposition and volume-split ranking metrics."""

    logger.info("Scoring holdout queries")
    scan = scan_holdout(scorer, dataset, max_history_len=max_history_len, settings=settings, logger=logger)

    logger.info("Sampling the random pool")
    candidates = pool_candidates(dataset, settings.random_seed)
    available = candidates.filter(
        pl.col("subject_uri").is_in(list(inference_uris(inference_paths, candidates["subject_uri"].to_list())))
    )
    sampled = available.sample(n=min(settings.pool_size, available.height), seed=settings.random_seed, shuffle=True)
    needed = set(sampled["subject_uri"].to_list()).union(*scan.liked.values())
    logger.info("Loading trait inferences for %d posts", len(needed))
    traits = load_traits(inference_paths, needed)
    pool = sampled.filter(pl.col("subject_uri").is_in(list(traits)))
    pool_uris = pool["subject_uri"].to_list()

    liked = {did: [uri for uri in uris if uri in traits] for did, uris in scan.liked.items()}
    eligible = sorted(did for did, uris in liked.items() if len(uris) >= settings.min_user_likes)
    logger.info("Synthetic feeds: %d pool posts, %d eligible users", len(pool_uris), len(eligible))
    feeds = synthetic_feeds(
        scorer, dataset, pool=pool, histories=scan.histories, users=eligible,
        max_history_len=max_history_len, settings=settings,
    )

    volume = like_counts(like_paths, *like_window)
    high = givers_of_half(eligible, volume)

    groups: dict[str, dict[str, Any]] = {}
    per_user_rows: list[dict[str, Any]] = []
    for key in sorted({key for uri in pool_uris for key in traits[uri]}):
        pool_values = np.array([traits[uri].get(key, np.nan) for uri in pool_uris], dtype=np.float64)
        liked_means: dict[str, float] = {}
        feed_means: dict[str, float] = {}
        for did in eligible:
            values = np.array([traits[uri].get(key, np.nan) for uri in liked[did]], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size >= settings.min_user_likes:
                liked_means[did] = float(values.mean())
            feed_values = pool_values[feeds[did]]
            feed_values = feed_values[np.isfinite(feed_values)]
            if feed_values.size:
                feed_means[did] = float(feed_values.mean())
        result = decompose(pool_values, liked_means, feed_means, eligible)
        if result is None:
            continue
        per_user = result.pop("per_user_amp")
        for name, members in (("high_volume", high), ("rest", set(eligible) - high)):
            sub = decompose(pool_values, liked_means, feed_means, sorted(members))
            if sub is not None:
                sub.pop("per_user_amp")
                result[name] = sub
        group, label = key.split("::", 1)
        groups.setdefault(group, {})[label] = result
        if key in HEADLINE_TRAITS:
            per_user_rows.extend({"did": did, "trait": key, "model_amp": amp} for did, amp in per_user.items())

    metric = f"ndcg@{settings.metrics_top_k}"
    queries = pl.DataFrame(scan.ranking_rows).with_columns(
        pl.col("did").replace_strict(volume, default=0, return_dtype=pl.Int64).alias("user_likes_in_window"),
        pl.col("did").is_in(list(high)).alias("high_volume"),
    )
    eligible_queries = queries.filter(pl.col("did").is_in(eligible))
    edges = np.quantile([volume.get(did, 0) for did in eligible], [0.2, 0.4, 0.6, 0.8]).tolist() if eligible else []
    bounds = [-math.inf, *edges, math.inf]
    ranking = {
        "all_holdout_queries": _ranking_summary(queries, metric),
        "eligible_users": _ranking_summary(eligible_queries, metric),
        "high_volume": _ranking_summary(eligible_queries.filter(pl.col("high_volume")), metric),
        "rest": _ranking_summary(eligible_queries.filter(~pl.col("high_volume")), metric),
        "volume_quintile_edges": edges,
        "by_volume_quintile": [
            _ranking_summary(eligible_queries.filter(
                (pl.col("user_likes_in_window") > bounds[q]) & (pl.col("user_likes_in_window") <= bounds[q + 1])
            ), metric)
            for q in range(5)
        ] if eligible else [],
    }
    headline = {
        key: {
            "over_serving_pp": 100 * groups[group][label]["mean_model_amp"],
            "se_pp": 100 * groups[group][label]["se_model_amp"],
            "p": groups[group][label]["p_model_amp"],
            "n_users": groups[group][label]["n_users"],
        }
        for key in HEADLINE_TRAITS
        for group, label in [key.split("::", 1)]
        if label in groups.get(group, {})
    }
    summary = {
        "n_pool_candidates": candidates.height,
        "n_pool_with_inferences": available.height,
        "n_pool": pool.height,
        "n_holdout_users": len(scan.liked),
        "n_users_eligible": len(eligible),
        "n_high_volume": len(high),
        "headline": headline,
        "groups": groups,
        "ranking": ranking,
    }
    logger.info("Headline: %s", json.dumps(headline))
    return PowerLikerResult(
        summary=summary,
        ranking_rows=queries,
        headline_per_user=pl.DataFrame(per_user_rows, schema={"did": pl.Utf8, "trait": pl.Utf8, "model_amp": pl.Float64}),
    )
