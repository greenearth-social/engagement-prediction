"""Media-sliced metrics, native BST validation, and comparison plot tests."""

from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from engagement_prediction.data.datasets import (
    HydratedBucketedEngagementDataset,
    create_hydrated_data_loader,
)
from engagement_prediction.data.datasets_test import _bundle, _write_post_liker_arrays
from engagement_prediction.data.training_index import build_loader_index
from engagement_prediction.evaluation.artifacts import Stage7Artifact
from engagement_prediction.evaluation.media_artifacts import (
    MediaModelArtifact,
    ValidationSettings,
)
from engagement_prediction.evaluation.media_evaluation import (
    _batch_identity,
    _evaluate_split,
    _MediaLookup,
    _media_ndcg_sums,
    run_media_evaluation,
    write_media_plots,
)
from engagement_prediction.evaluation.model_performance_test import _bst_config
from engagement_prediction.models.bst_ranker import BSTRanker
from engagement_prediction.training.bst_export import load_bst_checkpoint_model
from engagement_prediction.training.bst_ranker import BSTRankerMatrixScorer
from engagement_prediction.training.model_artifacts import file_sha256
from engagement_prediction.training.ranking import (
    MatrixBatchScores,
    evaluate_matrix_scorer,
)


class _FixtureScorer:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def prepare_for_eval(self, device):
        pass

    def score_batch(self, batch, device):
        self.calls += 1
        return MatrixBatchScores(self.scores.to(device))


def _metadata(uris, images, videos, statuses):
    return pl.DataFrame({
        "at_uri": uris,
        "in_val": [True] * len(uris),
        "in_val_unseen_users": [True] * len(uris),
        "contains_images": pl.Series(images, dtype=pl.Boolean),
        "contains_video": pl.Series(videos, dtype=pl.Boolean),
        "media_status": statuses,
    })


def _batch(uris, labels):
    return {
        "user_id": [f"user{index}" for index in range(len(labels))],
        "query_hour": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "candidate_post_id": uris,
        "label_matrix": torch.tensor(labels, dtype=torch.float32),
        "history_mask": torch.ones((len(labels), 1), dtype=torch.bool),
    }


def test_media_metrics_filter_before_reranking_and_exclude_unknowns():
    uris = ["image", "video", "both", "text", "missing", "partial"]
    metadata = _metadata(
        [*uris, "unused"], [True, False, True, False, None, True, None],
        [False, True, True, False, None, None, None],
        ["known"] * 4 + ["missing_document", "missing_flags", "missing_flags"],
    )
    batch = _batch(uris, [
        [0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
    ])
    scorer = _FixtureScorer(torch.tensor([[0.9, 0.8, 0.7, 0.6, 1.0, 0.5]] * 3))
    rows, coverage, identity = _evaluate_split(
        scorer, [batch], _MediaLookup(metadata), model_name="model", split="val",
        metrics_top_ks=(1, 2, 30), device="cpu", disable_progress=True,
    )
    values = {(row["media_type"], row["k"]): row for row in rows}
    assert scorer.calls == 1
    assert values["images", 1]["ndcg"] == pytest.approx(0.5)
    assert values["images", 2]["ndcg"] == pytest.approx((1 + 1 / np.log2(3)) / 2)
    assert values["images", 30]["ndcg"] == values["images", 2]["ndcg"]
    assert values["images", 1]["eligible_query_count"] == 2
    assert values["images", 1]["skipped_query_count"] == 1
    assert values["images", 1]["candidate_count"] == 2
    assert values["images", 1]["candidate_pair_count"] == 6
    assert values["images", 1]["eligible_candidate_pair_count"] == 4
    assert values["videos", 1]["ndcg"] == 0
    assert values["videos", 2]["ndcg"] == pytest.approx(1 / np.log2(3))
    assert values["text_only", 1]["ndcg"] == 1
    assert coverage["requested_unique_uri_count"] == 7
    assert coverage["scored_unique_uri_count"] == 6
    assert coverage["requested_missing_flags_unique_uri_count"] == 2
    assert coverage["scored_missing_flags_unique_uri_count"] == 1
    assert coverage["scored_known_unique_uri_count"] == 4
    assert coverage["missing_document_positive_count"] == 1
    assert coverage["missing_flags_positive_count"] == 1
    assert coverage["missing_flags_candidate_pair_count"] == 3
    assert identity["query_count"] == 3


def test_all_candidates_matches_canonical_evaluator_with_multiple_positives():
    batch = _batch(["a", "b", "c"], [[1, 0, 1], [0, 1, 0]])
    scores = torch.tensor([[0.6, 0.8, 0.4], [0.2, 0.9, 0.7]])
    ks = (1, 2, 30)
    sums, count = _media_ndcg_sums(
        scores, batch["label_matrix"], torch.ones(3, dtype=torch.bool),
        metrics_top_ks=ks,
    )
    canonical = evaluate_matrix_scorer(
        _FixtureScorer(scores), [batch], "cpu", list(ks),
        max_classification_metric_pairs=0, disable_progress=True,
    )["metrics"]
    for k in ks:
        assert sums[f"ndcg@{k}"].item() / count.item() == pytest.approx(
            canonical[f"ndcg@{k}"]
        )


@pytest.mark.parametrize("images", [[True, True], [False, False]])
def test_empty_media_or_no_positive_media_produces_blank_metric(images):
    metadata = _metadata(["a", "b"], images, [False, False], ["known", "known"])
    rows, _, _ = _evaluate_split(
        _FixtureScorer(torch.tensor([[0.8, 0.7]])), [_batch(["a", "b"], [[1, 0]])],
        _MediaLookup(metadata), model_name="m", split="val", metrics_top_ks=(30,),
        device="cpu", disable_progress=True,
    )
    videos = next(row for row in rows if row["media_type"] == "videos")
    assert videos["ndcg"] is None
    assert videos["eligible_query_count"] == 0
    assert videos["skipped_query_count"] == 1
    metadata = _metadata(["a", "b"], [False, True], [False, False], ["known", "known"])
    rows, _, _ = _evaluate_split(
        _FixtureScorer(torch.tensor([[0.8, 0.7]])), [_batch(["a", "b"], [[1, 0]])],
        _MediaLookup(metadata), model_name="m", split="val", metrics_top_ks=(30,),
        device="cpu", disable_progress=True,
    )
    images_row = next(row for row in rows if row["media_type"] == "images")
    assert images_row["candidate_count"] == 1
    assert images_row["ndcg"] is None


@pytest.mark.parametrize("change", ["user", "hour", "candidate", "label"])
def test_batch_fingerprint_detects_query_candidate_or_label_changes(change):
    batch = _batch(["a", "b"], [[1, 0]])
    original = _batch_identity(batch)
    if change == "user":
        batch["user_id"] = ["another-user"]
    elif change == "hour":
        batch["query_hour"] = datetime(2026, 1, 2, tzinfo=timezone.utc)
    elif change == "candidate":
        batch["candidate_post_id"] = ["b", "a"]
    else:
        batch["label_matrix"] = torch.tensor([[0.0, 1.0]])
    assert _batch_identity(batch) != original


def _native_validation_artifact(tmp_path: Path) -> Stage7Artifact:
    bundle = _bundle(tmp_path)
    queries_path = bundle / "queries" / "part-00000.parquet"
    pl.read_parquet(queries_path).with_columns(
        pl.Series("split", ["val", "val_unseen_users"]),
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
    _write_post_liker_arrays(bundle, {
        0: [(2, datetime(2026, 1, 1, 10, tzinfo=timezone.utc))],
        1: [(3, datetime(2026, 1, 1, 11, tzinfo=timezone.utc))],
    }, user_table_num_rows=4)
    return Stage7Artifact(
        root=tmp_path, bundle_path=bundle, manifest={}, summary={},
        loader_index_validation={
            "format_version": 2, "embedding_count": 5,
            "splits": {split: {"query_count": 1} for split in ("val", "val_unseen_users")},
        },
        embedding_model="fixture-model", embedding_dim=2,
    )


def _native_model(tmp_path: Path, *, use_post_liker_feature: bool) -> MediaModelArtifact:
    name = "liker" if use_post_liker_feature else "baseline"
    model_config = _bst_config(max_history_len=1 if use_post_liker_feature else 3,
                               author_table_num_rows=7)
    model_config["constructor_args"].update({
        "use_post_liker_feature": use_post_liker_feature,
        "post_liker_user_table_num_rows": 4 if use_post_liker_feature else 2,
        "use_popularity_feature": True,
        "popularity_log_mean": 1.0,
        "popularity_log_std": 2.0,
    })
    popularity = {"enabled": True, "log_mean": 1.0, "log_std": 2.0}
    torch.manual_seed(12)
    model = BSTRanker(**model_config["constructor_args"])
    path = tmp_path / f"{name}.pth"
    torch.save({
        "epoch": 2, "best_epoch": 2, "model_state_dict": model.state_dict(),
        "metadata": {"model_config": model_config, "popularity_stats": popularity},
    }, path)
    return MediaModelArtifact(
        name=name, root=tmp_path, manifest={}, model_config=model_config,
        training_config={"bst_max_post_liker_replay_events_per_post": 2},
        popularity_stats=popularity, checkpoint_path=path, checkpoint_sha256=file_sha256(path),
    )


def _run(dataset, models, metadata):
    return run_media_evaluation(
        dataset, models, ValidationSettings(2, None, 7), metadata,
        metrics_top_ks=(1, 30), device="cpu", num_workers=0, pin_memory=False,
        prefetch_factor=1, disable_progress=True, logger=logging.getLogger(__name__),
    )


@pytest.mark.parametrize("include_liker", [False, True])
def test_native_evaluation_preserves_settings_and_matches_canonical(tmp_path, include_liker):
    dataset = _native_validation_artifact(tmp_path)
    models = [_native_model(tmp_path, use_post_liker_feature=False)]
    if include_liker:
        models.append(_native_model(tmp_path, use_post_liker_feature=True))
    metadata = _metadata(["p1", "p2", "n1"], [True] * 3, [False] * 3, ["known"] * 3)
    result = _run(dataset, models, metadata)
    assert len(result["metrics"]) == len(models) * 2 * 3 * 2
    assert len(result["model_results"]) == len(models)
    json.dumps(result, allow_nan=False)
    for artifact, model_result in zip(models, result["model_results"]):
        assert model_result["best_epoch"] == 2
        assert model_result["max_history_len"] == artifact.model_config["max_history_len"]
        assert model_result["use_post_liker_feature"] is (artifact.name == "liker")
        eager_model, _ = load_bst_checkpoint_model(
            checkpoint_path=artifact.checkpoint_path,
            expected_model_config=artifact.model_config,
            expected_popularity_stats=artifact.popularity_stats,
        )
        split_dataset = HydratedBucketedEngagementDataset(
            dataset.bundle_path, split="val", max_history_len=artifact.model_config["max_history_len"],
            additional_batch_negatives=None, use_post_liker_feature=artifact.name == "liker",
            max_post_liker_replay_events_per_post=2 if artifact.name == "liker" else None,
            seed=7, logger=None,
        )
        try:
            loader = create_hydrated_data_loader(
                split_dataset, batch_size=2, shuffle=False, drop_last=False,
                num_workers=0, pin_memory=False, persistent_workers=False,
                prefetch_factor=1, seed=7, resample_candidates_each_epoch=False,
                tensor_only=False, tensor_batch_kind="bst",
            )
            canonical = evaluate_matrix_scorer(
                BSTRankerMatrixScorer(eager_model), loader, "cpu", [1, 30],
                max_classification_metric_pairs=0, disable_progress=True,
            )["metrics"]
            for row in result["metrics"]:
                if row["model"] == artifact.name and row["split"] == "val" and row["media_type"] == "images":
                    assert row["ndcg"] == pytest.approx(canonical[f"ndcg@{row['k']}"])
        finally:
            split_dataset.close()
    if include_liker:
        first, second = result["model_results"]
        assert first["splits"] == second["splits"]


def test_comparison_rejects_different_validation_sequences(tmp_path, monkeypatch):
    dataset = _native_validation_artifact(tmp_path)
    model = _native_model(tmp_path, use_post_liker_feature=False)
    other = replace(model, name="other")
    metadata = _metadata(["p1", "p2", "n1"], [True] * 3, [False] * 3, ["known"] * 3)
    original_collate = HydratedBucketedEngagementDataset.collate_batch
    batch_count = 0

    def changed_sequence(self, items):
        nonlocal batch_count
        batch = original_collate(self, items)
        batch_count += 1
        if batch_count >= 3:
            batch["user_id"] = [f"changed-{user}" for user in batch["user_id"]]
        return batch

    monkeypatch.setattr(HydratedBucketedEngagementDataset, "collate_batch", changed_sequence)
    with pytest.raises(RuntimeError, match="baseline.*other.*val"):
        _run(dataset, [model, other], metadata)


@pytest.mark.parametrize("ks", [(30,), (1, 10, 30)])
def test_media_plots_write_two_images_with_missing_groups(tmp_path, ks):
    rows = [{
        "model": name, "split": split, "media_type": media, "k": k,
        "ndcg": None if media == "videos" else 0.2 + index / 10,
    } for index, name in enumerate(("a", "b"))
        for split in ("val", "val_unseen_users")
        for media in ("images", "videos", "text_only") for k in ks]
    write_media_plots(rows, tmp_path, model_names=("a", "b"), metrics_top_ks=ks)
    for split in ("val", "val_unseen_users"):
        image = tmp_path / f"{split}_ndcg.png"
        assert image.stat().st_size > 1000
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
