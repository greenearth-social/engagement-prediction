from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch

from engagement_prediction.data import dataset_hydration
from engagement_prediction.data import author_vocabulary
from engagement_prediction.data import training_index
from engagement_prediction.data.datasets import HydratedBucketedEngagementDataset
from engagement_prediction.models.bst_ranker import BSTRanker
from engagement_prediction.pipeline.core import Context
from engagement_prediction.stages import train_bst_ranker


class _RecordingTracker:
    def __init__(self, *, task_id="task-1", manifest_uploaded=True):
        self.id = task_id
        self.manifest_uploaded = manifest_uploaded
        self.scalar_calls = []
        self.file_artifacts = []
        self.model_artifacts = []

    def log_scalar(self, title, series, value, iteration):
        self.scalar_calls.append((title, series, value, iteration))

    def log_file_artifact(self, name, path):
        self.file_artifacts.append((name, Path(path)))
        if name == "ranker_serving_manifest":
            return self.manifest_uploaded
        return True

    def log_artifact(self, name, path):
        self.model_artifacts.append((name, Path(path)))
        return {
            "model_id": "model-1",
            "uri": "gs://models/task/models/ranker.pt",
        }


def _write_dataset(bundle: Path, name: str, frame: pl.DataFrame) -> None:
    path = bundle / name
    path.mkdir()
    frame.write_parquet(path / "part-00000.parquet")


def _build_loader_index(bundle: Path) -> None:
    training_index.build_loader_index(
        posts_path=bundle / "posts",
        queries_path=bundle / "queries",
        query_positives_path=bundle / "query_positives",
        query_histories_path=bundle / "query_histories",
        hourly_negative_candidates_path=bundle / "hourly_negative_candidates",
        embeddings_path=bundle / "embeddings.npy",
        authors_path=bundle / "authors",
        output_path=bundle / "loader_index",
        logger=None,
    )


def _stage7_fixture(tmp_path: Path) -> tuple[Path, Path]:
    stage7_dir = tmp_path / "07_dataset_hydration" / "stage7"
    bundle = stage7_dir / "hydrated_training_data_stage7"
    bundle.mkdir(parents=True)
    embeddings = np.lib.format.open_memmap(
        bundle / "embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(4, 2),
    )
    embeddings[:] = np.array([[1, 0], [0, 1], [2, 0], [0, 2]], dtype=np.float32)
    embeddings.flush()
    del embeddings

    train_hour = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    val_hour = datetime(2026, 1, 1, 13, tzinfo=timezone.utc)
    unseen_hour = datetime(2026, 1, 1, 14, tzinfo=timezone.utc)
    created = datetime(2026, 1, 1, 11, 30, tzinfo=timezone.utc)
    queries = pl.DataFrame({
        "did": ["u1", "u2", "u3", "u4"],
        "query_hour": [train_hour, train_hour, val_hour, unseen_hour],
        "user_cohort": ["seen", "seen", "seen", "unseen"],
        "split": ["train", "train", "val", "val_unseen_users"],
        "positive_count": [1, 1, 1, 1],
    }, schema=dataset_hydration.QUERY_SCHEMA)
    positives = pl.DataFrame({
        "did": ["u1", "u2", "u3", "u4"],
        "query_hour": [train_hour, train_hour, val_hour, unseen_hour],
        "subject_uri": ["p1", "p2", "p1", "p2"],
        "like_created_at": [train_hour, train_hour, val_hour, unseen_hour],
        "emb_idx": [0, 1, 0, 1],
        "post_created_at": [created, created, created, created],
        "author_idx": [2, 3, 2, 3],
        "prior_like_count": [3, 7, 100, 200],
    }, schema=dataset_hydration.QUERY_POSITIVE_SCHEMA)
    histories = pl.DataFrame({
        "did": ["u1", "u2", "u3", "u4"],
        "query_hour": [train_hour, train_hour, val_hour, unseen_hour],
        "history_subject_uris": [["h1", "h1"], [], ["h1"], ["h1"]],
        "history_like_created_ats": [
            [
                datetime(2026, 1, 1, 10, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 9, tzinfo=timezone.utc),
            ],
            [],
            [datetime(2026, 1, 1, 10, tzinfo=timezone.utc)],
            [datetime(2026, 1, 1, 10, tzinfo=timezone.utc)],
        ],
        "history_emb_indices": [[2, 2], [], [2], [2]],
        "history_author_indices": [[4, 4], [], [4], [4]],
        "history_prior_like_counts": [[0, 1], [], [100], [200]],
    }, schema=dataset_hydration.QUERY_HISTORY_SCHEMA)
    negatives = pl.DataFrame({
        "query_hour": [train_hour, val_hour, unseen_hour],
        "subject_uri": ["n1", "n1", "n1"],
        "selection_source": ["random", "random", "random"],
        "emb_idx": [3, 3, 3],
        "post_created_at": [created, created, created],
        "author_idx": [5, 5, 5],
        "prior_like_count": [15, 100, 200],
    }, schema=dataset_hydration.HOURLY_NEGATIVE_SCHEMA)
    _write_dataset(bundle, "queries", queries)
    _write_dataset(bundle, "query_positives", positives)
    _write_dataset(bundle, "query_histories", histories)
    _write_dataset(bundle, "hourly_negative_candidates", negatives)
    _write_dataset(
        bundle,
        "posts",
        pl.DataFrame({
            "subject_uri": ["p1", "p2", "h1", "n1"],
            "emb_idx": [0, 1, 2, 3],
            "post_created_at": [created, created, created, created],
            "author_did": ["a2", "a3", "a4", "a5"],
            "author_idx": [2, 3, 4, 5],
            "is_reply": [False, False, False, False],
            "is_positive": [True, True, False, False],
            "is_history": [False, False, True, False],
            "is_negative": [False, False, False, True],
        }, schema=dataset_hydration.POST_SCHEMA),
    )
    _write_dataset(
        bundle,
        "authors",
        pl.DataFrame({
            "author_did": ["a2", "a3", "a4", "a5"],
            "author_idx": [2, 3, 4, 5],
            "training_feature_count": [1, 1, 1, 1],
            "training_positive_count": [1, 1, 0, 0],
            "training_history_count": [0, 0, 1, 0],
            "training_negative_count": [0, 0, 0, 1],
        }, schema=author_vocabulary.AUTHOR_VOCABULARY_SCHEMA),
    )
    _build_loader_index(bundle)
    (stage7_dir / "summary.json").write_text(json.dumps({
        "parameters": {
            "embedding_model": "fixture-model",
            "embedding_dim": 2,
        }
    }) + "\n")
    return stage7_dir, bundle


def _args(*, save_model: bool, plots: bool) -> SimpleNamespace:
    return SimpleNamespace(
        run_tag=None,
        random_seed=7,
        max_history_len=2,
        bst_additional_batch_negatives=2,
        batch_size=2,
        num_dataloader_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_prefetch_factor=1,
        metrics_top_ks=[1],
        bst_use_popularity_feature=True,
        no_save_model=not save_model,
        no_plots=not plots,
        disable_progress=True,
        bst_max_train_batches_per_epoch=None,
        device="cpu",
        author_embedding_dim=3,
        content_projection_dim=4,
        author_projection_dim=2,
        bst_model_dim=4,
        bst_time_embedding_dim=2,
        bst_num_attention_heads=2,
        bst_num_transformer_layers=1,
        bst_transformer_ff_dim=8,
        bst_dropout_rate=0.0,
        author_unknown_dropout_rate=0.0,
        bst_norm_first=False,
        bst_time_delta_bucket_boundaries_hours=[1.0, 6.0, 24.0],
        prediction_hidden_dims=[4],
        bst_popularity_projection_dim=2,
        epochs=1,
        learning_rate=1.0e-3,
        bst_weight_decay=0.0,
        patience=2,
        early_stopping_min_delta=0.1,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=1,
        gradient_clip_max_norm=1.0,
    )


def _context(tmp_path: Path, tracker: _RecordingTracker) -> Context:
    run_dir = tmp_path / "runs" / "run"
    run_dir.mkdir(parents=True)
    return Context(
        run_dir=run_dir,
        artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "runs",
        use_latest=True,
        tracker=tracker,
    )


def test_stage8_trains_native_dataset_and_publishes_reloadable_checkpoint(
    tmp_path,
    monkeypatch,
):
    stage7_dir, bundle = _stage7_fixture(tmp_path)
    monkeypatch.setattr(
        train_bst_ranker,
        "resolve_recorded_stage_lineage",
        lambda *args, **kwargs: {"07_dataset_hydration": stage7_dir},
    )
    tracker = _RecordingTracker()
    create_loader = train_bst_ranker._create_loader
    loader_calls = []

    def record_loader(**kwargs):
        loader_calls.append(kwargs)
        return create_loader(**kwargs)

    monkeypatch.setattr(train_bst_ranker, "_create_loader", record_loader)

    result = train_bst_ranker.run(
        _context(tmp_path, tracker),
        _args(save_model=True, plots=True),
    )

    output_dir = Path(result["output_dir"])
    checkpoint_path = output_dir / "checkpoints" / "bst_ranker_best.pth"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_config = json.loads((output_dir / "model_config.json").read_text())
    popularity = json.loads((output_dir / "popularity_stats.json").read_text())
    training_results = json.loads((output_dir / "training_results.json").read_text())
    summary = json.loads((output_dir / "summary.json").read_text())
    assert popularity["history_observation_count"] == 2
    assert popularity["candidate_observation_count"] == 3
    assert checkpoint["metadata"]["model_config"] == model_config
    assert checkpoint["metadata"]["popularity_stats"] == popularity
    assert training_results["best_epoch"] == 1
    assert set(training_results["final_metrics"]) == {
        "train",
        "val",
        "val_unseen_users",
    }
    assert not any(
        "recall" in key
        for metrics in training_results["final_metrics"].values()
        for key in metrics
    )
    assert "mean_average_precision" not in json.dumps(training_results)
    assert "mean_average_precision" not in json.dumps(checkpoint["history"])
    assert "mean_average_precision" not in json.dumps(checkpoint["baseline_metrics"])
    assert (output_dir / "training_history.png").stat().st_size > 0
    assert list((output_dir / "authors").glob("*.parquet"))
    torchscript_path = output_dir / "checkpoints" / "ranker.pt"
    author_map_path = output_dir / "ranker_author_idx.parquet"
    manifest_path = output_dir / "checkpoints" / "ranker_serving_manifest.json"
    assert torchscript_path.stat().st_size > 0
    assert pl.read_parquet(author_map_path).select(
        "author_did", "author_idx"
    ).to_dicts() == [
        {"author_did": "a2", "author_idx": 2},
        {"author_did": "a3", "author_idx": 3},
        {"author_did": "a4", "author_idx": 4},
        {"author_did": "a5", "author_idx": 5},
    ]
    assert json.loads(manifest_path.read_text()) == {
        "ranker_clearml_model_id": "model-1",
        "ranker_uri": "gs://models/task/models/ranker.pt",
        "clearml_task_id": "task-1",
    }
    assert training_results["torchscript_export"]["export_count"] == 1
    assert training_results["torchscript_export"]["exported_best_epochs"] == [1]
    assert "path" not in training_results["torchscript_export"]["exports"][0]
    assert training_results["clearml_publication"]["status"] == "complete"
    assert summary["outputs"]["torchscript_path"] == str(torchscript_path)
    assert result["artifacts"]["torchscript_path"] == str(torchscript_path)
    assert result["artifacts"]["ranker_author_idx_path"] == str(author_map_path)
    assert result["artifacts"]["serving_manifest_path"] == str(manifest_path)

    reloaded_a = BSTRanker(**model_config["constructor_args"])
    reloaded_a.load_state_dict(checkpoint["model_state_dict"])
    dataset = HydratedBucketedEngagementDataset(
        bundle,
        split="train",
        max_history_len=2,
        bst_additional_batch_negatives=2,
        seed=7,
        logger=None,
    )
    batch = dataset.collate_batch([dataset[0], dataset[1]])
    scripted_model = torch.jit.load(str(torchscript_path)).eval()
    with torch.inference_mode():
        scores_a = reloaded_a.score_candidate_matrix(
            history_embeddings=batch["history_embeddings"],
            history_mask=batch["history_mask"],
            history_time_deltas_hours=batch["history_time_deltas_hours"],
            candidate_post_embeddings=batch["candidate_post_embeddings"],
            history_author_indices=batch["history_author_indices"],
            candidate_post_author_idx=batch["candidate_post_author_idx"],
            history_prior_cumulative_likes=batch["history_prior_cumulative_likes"],
            candidate_prior_cumulative_likes=batch["candidate_prior_cumulative_likes"],
        )
        scores_b = scripted_model.score_candidate_matrix(
            batch["history_embeddings"],
            batch["history_mask"],
            batch["history_time_deltas_hours"],
            batch["candidate_post_embeddings"],
            batch["history_author_indices"],
            batch["candidate_post_author_idx"],
            batch["history_prior_cumulative_likes"],
            batch["candidate_prior_cumulative_likes"],
        )
    assert torch.equal(scores_a, scores_b)
    for series in (
        "Train NDCG@1",
        "Validation NDCG@1",
        "Validation Unseen Users NDCG@1",
    ):
        calls = [
            call
            for call in tracker.scalar_calls
            if call[0] == "NDCG@1" and call[1] == series
        ]
        assert [call[3] for call in calls] == [0, 1]
    assert not any(
        "MAP" in title or "MAP" in series
        for title, series, _, _ in tracker.scalar_calls
    )
    assert tracker.model_artifacts == [("ranker", torchscript_path)]
    assert {
        "author_idx_mapping",
        "ranker_serving_manifest",
        "bst_ranker_best_checkpoint",
    }.issubset({name for name, _ in tracker.file_artifacts})
    assert len(loader_calls) == 3


def test_stage8_ignores_no_save_model_and_respects_no_plots(tmp_path, monkeypatch):
    stage7_dir, _ = _stage7_fixture(tmp_path)
    monkeypatch.setattr(
        train_bst_ranker,
        "resolve_recorded_stage_lineage",
        lambda *args, **kwargs: {"07_dataset_hydration": stage7_dir},
    )

    tracker = _RecordingTracker(task_id="")
    result = train_bst_ranker.run(
        _context(tmp_path, tracker),
        _args(save_model=False, plots=False),
    )

    output_dir = Path(result["output_dir"])
    assert (output_dir / "checkpoints" / "bst_ranker_best.pth").is_file()
    assert (output_dir / "checkpoints" / "ranker.pt").is_file()
    assert (output_dir / "ranker_author_idx.parquet").is_file()
    assert result["artifacts"]["serving_manifest_path"] is None
    training_config = json.loads(
        (output_dir / "training_config.json").read_text()
    )
    assert "save_model" not in training_config
    assert "no_save_model_requested_but_ignored" not in training_config
    assert not (output_dir / "training_history.png").exists()
    assert tracker.model_artifacts == []
    assert tracker.file_artifacts == []
    training_results = json.loads((output_dir / "training_results.json").read_text())
    summary = json.loads((output_dir / "summary.json").read_text())
    assert training_results["clearml_publication"]["status"] == "not_configured"
    assert summary["clearml_publication"]["status"] == "not_configured"
    assert "clearml_publication_status: not_configured" in (
        output_dir / "stage_info.txt"
    ).read_text()


def test_stage8_keeps_local_manifest_when_manifest_upload_fails(tmp_path, monkeypatch):
    stage7_dir, _ = _stage7_fixture(tmp_path)
    monkeypatch.setattr(
        train_bst_ranker,
        "resolve_recorded_stage_lineage",
        lambda *args, **kwargs: {"07_dataset_hydration": stage7_dir},
    )
    tracker = _RecordingTracker(manifest_uploaded=False)

    result = train_bst_ranker.run(
        _context(tmp_path, tracker),
        _args(save_model=True, plots=False),
    )

    output_dir = Path(result["output_dir"])
    manifest_path = output_dir / "checkpoints" / "ranker_serving_manifest.json"
    assert manifest_path.is_file()
    assert result["artifacts"]["serving_manifest_path"] == str(manifest_path)
    assert result["clearml_publication"]["status"] == "incomplete"
    assert result["clearml_publication"]["manifest_uploaded"] is False
    training_results = json.loads((output_dir / "training_results.json").read_text())
    summary = json.loads((output_dir / "summary.json").read_text())
    assert training_results["clearml_publication"]["status"] == "incomplete"
    assert summary["clearml_publication"]["status"] == "incomplete"
    assert "clearml_publication_status: incomplete" in (
        output_dir / "stage_info.txt"
    ).read_text()


def test_stage8_requires_every_training_split(tmp_path, monkeypatch):
    stage7_dir, bundle = _stage7_fixture(tmp_path)
    for artifact_name in ("queries", "query_positives", "query_histories"):
        artifact_path = bundle / artifact_name / "part-00000.parquet"
        pl.read_parquet(artifact_path).filter(
            pl.col("did") != "u4"
        ).write_parquet(artifact_path)
    negatives_path = bundle / "hourly_negative_candidates" / "part-00000.parquet"
    pl.read_parquet(negatives_path).filter(
        pl.col("query_hour") != datetime(2026, 1, 1, 14, tzinfo=timezone.utc)
    ).write_parquet(negatives_path)
    shutil.rmtree(bundle / "loader_index")
    _build_loader_index(bundle)
    monkeypatch.setattr(
        train_bst_ranker,
        "resolve_recorded_stage_lineage",
        lambda *args, **kwargs: {"07_dataset_hydration": stage7_dir},
    )

    with pytest.raises(ValueError, match="nonempty 'val_unseen_users'"):
        train_bst_ranker.run(
            _context(tmp_path, _RecordingTracker()),
            _args(save_model=False, plots=False),
        )


def test_stage8_rejects_stage7_without_loader_index(tmp_path, monkeypatch):
    stage7_dir, bundle = _stage7_fixture(tmp_path)
    shutil.rmtree(bundle / "loader_index")
    monkeypatch.setattr(
        train_bst_ranker,
        "resolve_recorded_stage_lineage",
        lambda *args, **kwargs: {"07_dataset_hydration": stage7_dir},
    )

    with pytest.raises(ValueError, match="regenerate Stage 7"):
        train_bst_ranker.run(
            _context(tmp_path, _RecordingTracker()),
            _args(save_model=False, plots=False),
        )


def test_stage8_rejects_unsupported_loader_index_format(tmp_path, monkeypatch):
    stage7_dir, bundle = _stage7_fixture(tmp_path)
    format_path = bundle / "loader_index" / "format.json"
    metadata = json.loads(format_path.read_text())
    metadata["format_version"] = 0
    format_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(
        train_bst_ranker,
        "resolve_recorded_stage_lineage",
        lambda *args, **kwargs: {"07_dataset_hydration": stage7_dir},
    )

    with pytest.raises(ValueError, match="regenerate Stage 7"):
        train_bst_ranker.run(
            _context(tmp_path, _RecordingTracker()),
            _args(save_model=False, plots=False),
        )
