"""Power-liker synthetic feeds, decomposition, and holdout scanning tests."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from engagement_prediction.data.datasets_test import _bundle
from engagement_prediction.data.training_index import build_loader_index
from engagement_prediction.evaluation.artifacts import Stage7Artifact
from engagement_prediction.evaluation.power_likers import (
    PowerLikerSettings,
    decompose,
    givers_of_half,
    load_traits,
    pool_candidates,
    run_power_liker_evaluation,
    scan_holdout,
    synthetic_feeds,
)
from engagement_prediction.training.ranking import MatrixBatchScores


UTC = timezone.utc
SETTINGS = PowerLikerSettings(
    pool_size=100, top_k=3, min_user_likes=1, metrics_top_k=30, random_seed=7,
    batch_size=2, device="cpu", num_dataloader_workers=0, dataloader_prefetch_factor=1,
)


class _FirstDimensionScorer:
    """Scores every user's candidates by the first embedding coordinate."""

    def prepare_for_eval(self, device):
        pass

    def score_batch(self, batch, device):
        users = batch["history_mask"].shape[0]
        return MatrixBatchScores(batch["candidate_post_embeddings"][:, 0].expand(users, -1).clone())


def _holdout_artifact(tmp_path: Path) -> Stage7Artifact:
    bundle = _bundle(tmp_path, many_negatives=True)
    queries_path = bundle / "queries" / "part-00000.parquet"
    pl.read_parquet(queries_path).with_columns(
        pl.Series("split", ["holdout_seen_users", "holdout_unseen_users"]),
        pl.Series("user_cohort", ["seen", "unseen"]),
    ).write_parquet(queries_path)
    (bundle / "loader_index").rename(bundle / "original_loader_index")
    build_loader_index(
        posts_path=bundle / "posts", queries_path=bundle / "queries",
        query_positives_path=bundle / "query_positives",
        query_histories_path=bundle / "query_histories",
        hourly_negative_candidates_path=bundle / "hourly_negative_candidates",
        embeddings_path=bundle / "embeddings.npy", authors_path=bundle / "authors",
        indexed_post_liker_events_path=bundle / "indexed_post_liker_events",
        post_liker_users_path=bundle / "post_liker_users",
        output_path=bundle / "loader_index", logger=None,
    )
    return Stage7Artifact(
        root=tmp_path, bundle_path=bundle, manifest={}, summary={},
        loader_index_validation={}, embedding_model="fixture-model", embedding_dim=2,
    )


def _write_inferences(path: Path, negative_by_uri: dict[str, float]) -> str:
    rows = [
        {"at_uri": uri, "inferences": json.dumps(
            {"text": {"message.commit.record.text": {"sentiment": {"Negative": value}}}}
        )}
        for uri, value in negative_by_uri.items()
    ]
    rows.append({"at_uri": "image-only", "inferences": json.dumps({"image": {"x": {"y": 1.0}}})})
    pl.DataFrame(rows).write_parquet(path)
    return str(path)


def test_givers_of_half_is_smallest_top_volume_set():
    volume = {"a": 50, "b": 30, "c": 15, "d": 5}
    assert givers_of_half(["a", "b", "c", "d"], volume) == {"a"}
    assert givers_of_half(["b", "c", "d"], volume) == {"b"}
    assert givers_of_half(["c", "d"], {"c": 1, "d": 1}) == {"c"}
    assert givers_of_half(["x"], {}) == set()


def test_decompose_reports_feed_minus_liked_in_probability_units():
    pool = np.linspace(0.0, 1.0, 21)
    users = [f"u{index}" for index in range(10)]
    liked = {user: 0.40 + 0.002 * index for index, user in enumerate(users)}
    feed = {user: 0.45 + 0.012 * index for index, user in enumerate(users)}
    result = decompose(pool, liked, feed, users)
    assert result is not None
    assert result["n_users"] == 10
    assert result["pool_mean"] == pytest.approx(0.5)
    assert result["mean_model_amp"] == pytest.approx(0.095)
    assert result["mean_user_pref"] == pytest.approx(-0.091)
    assert result["mean_model_excess"] == pytest.approx(0.004)
    assert result["per_user_amp"]["u9"] == pytest.approx(0.14)
    # amp = 0.05 + 0.01 * i; reference values from scipy.stats.ttest_1samp.
    assert result["se_model_amp"] == pytest.approx(0.0302765035 / np.sqrt(10))
    assert result["p_model_amp"] == pytest.approx(3.81772e-06, rel=1e-5)
    assert decompose(pool, liked, feed, users[:9]) is None
    assert decompose(pool[:19], liked, feed, users) is None


def test_load_traits_flattens_text_inferences_and_skips_other_modalities(tmp_path):
    path = _write_inferences(tmp_path / "bsky_inferences_20260101_000000.parquet", {"p1": 0.2, "p2": 0.7})
    traits = load_traits([path], ["p1", "p2", "image-only", "absent"])
    assert traits == {"p1": {"sentiment::Negative": 0.2}, "p2": {"sentiment::Negative": 0.7}}


def test_pool_candidates_keeps_one_random_hour_per_post(tmp_path):
    bundle = tmp_path / "hydrated_training_data_test"
    (bundle / "hourly_negative_candidates").mkdir(parents=True)
    hours = [datetime(2026, 1, 1, hour, tzinfo=UTC) for hour in range(4)]
    pl.DataFrame({
        "query_hour": hours * 2 + [hours[0]],
        "subject_uri": ["a"] * 4 + ["b"] * 4 + ["popular"],
        "selection_source": ["random"] * 8 + ["popular"],
        "emb_idx": [0] * 4 + [1] * 4 + [2],
        "author_idx": [2] * 4 + [3] * 4 + [4],
        "prior_like_count": [0, 1, 2, 3, 10, 11, 12, 13, 99],
    }).write_parquet(bundle / "hourly_negative_candidates" / "part-00000.parquet")
    dataset = Stage7Artifact(
        root=tmp_path, bundle_path=bundle, manifest={}, summary={},
        loader_index_validation={}, embedding_model="fixture-model", embedding_dim=2,
    )
    first = pool_candidates(dataset, random_seed=1)
    assert first["subject_uri"].to_list() == ["a", "b"]
    assert first.equals(pool_candidates(dataset, random_seed=1))
    hours_kept = {
        tuple(pool_candidates(dataset, random_seed=seed)["prior_like_count"].to_list())
        for seed in range(20)
    }
    assert len(hours_kept) > 1


def test_scan_holdout_and_feeds_use_longest_history_and_all_candidates(tmp_path):
    dataset = _holdout_artifact(tmp_path)
    scan = scan_holdout(
        _FirstDimensionScorer(), dataset, max_history_len=3, settings=SETTINGS,
        logger=logging.getLogger(__name__),
    )
    assert sorted((row["did"], row["split"]) for row in scan.ranking_rows) == [
        ("u1", "holdout_seen_users"), ("u2", "holdout_unseen_users"),
    ]
    assert scan.liked == {"u1": {"p1", "p2"}, "u2": {"p2"}}
    assert int(scan.histories["u1"]["history_mask"].sum()) == 3
    assert int(scan.histories["u2"]["history_mask"].sum()) == 0

    pool = pool_candidates(dataset, random_seed=7)
    feeds = synthetic_feeds(
        _FirstDimensionScorer(), dataset, pool=pool, histories=scan.histories,
        users=["u1", "u2"], max_history_len=3, settings=SETTINGS,
    )
    emb_idx = pool["emb_idx"].to_numpy()
    expected = np.argsort(-emb_idx)[:3]
    for user in ("u1", "u2"):
        assert sorted(feeds[user].tolist()) == sorted(expected.tolist())


def test_run_reports_pool_cohort_and_volume_split(tmp_path):
    dataset = _holdout_artifact(tmp_path)
    negatives = {f"n{index}": 0.1 * index for index in range(7)}  # n7 has no inference row
    inference_path = _write_inferences(tmp_path / "bsky_inferences_20260101_000000.parquet", {
        "p1": 0.9, "p2": 0.8, **negatives,
    })
    likes_path = tmp_path / "bsky_likes_20260101_000000.parquet"
    pl.DataFrame({
        "did": ["u1", "u1", "u1", "u2", "u2"],
        "record_created_at": [
            "2026-01-01T09:00:00Z", "2026-01-01T10:00:00Z", "2026-01-01T11:00:00Z",
            "2026-01-01T11:00:00Z", "2025-12-31T23:00:00Z",
        ],
    }).write_parquet(likes_path)
    result = run_power_liker_evaluation(
        _FirstDimensionScorer(), dataset, max_history_len=3,
        inference_paths=[inference_path], like_paths=[str(likes_path)],
        like_window=(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)),
        settings=SETTINGS, logger=logging.getLogger(__name__),
    )
    summary = result.summary
    assert summary["n_pool_candidates"] == 8
    assert summary["n_pool_with_inferences"] == 7
    assert summary["n_pool"] == 7
    assert summary["n_users_eligible"] == 2
    assert summary["n_high_volume"] == 1
    assert summary["headline"] == {}  # too few users to decompose
    assert result.ranking_rows.filter(pl.col("high_volume"))["did"].to_list() == ["u1"]
    assert dict(zip(*result.ranking_rows.select("did", "user_likes_in_window"))) == {"u1": 3, "u2": 1}
    assert summary["ranking"]["eligible_users"]["n_queries"] == 2
    json.dumps(summary)
