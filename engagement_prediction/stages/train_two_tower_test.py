from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import polars as pl
import pytest
import torch

from engagement_prediction.data.datasets import HydratedBucketedEngagementDataset
from engagement_prediction.models.two_tower import TwoTowerModel
from engagement_prediction.pipeline.core import Context
from engagement_prediction.stages import train_two_tower
from engagement_prediction.stages.train_bst_ranker_test import _stage7_fixture


class _RecordingTracker:
    def __init__(self, *, task_id="task-1"):
        self.id = task_id
        self.scalar_calls = []
        self.histogram_calls = []
        self.model_artifacts = []
        self.file_artifacts = []

    def log_scalar(self, title, series, value, iteration):
        self.scalar_calls.append((title, series, value, iteration))

    def log_histogram(
        self,
        title,
        series,
        values,
        iteration=0,
        xlabels=None,
        xaxis=None,
        yaxis=None,
        labels=None,
        mode=None,
    ):
        self.histogram_calls.append({
            "title": title,
            "series": series,
            "values": values,
            "iteration": iteration,
            "xlabels": xlabels,
            "xaxis": xaxis,
            "yaxis": yaxis,
            "labels": labels,
            "mode": mode,
        })

    def log_artifact(self, name, path):
        path = Path(path)
        self.model_artifacts.append((name, path))
        return {
            "model_id": f"{name}-id",
            "uri": f"gs://models/task/models/{path.name}",
        }

    def log_file_artifact(self, name, path):
        self.file_artifacts.append((name, Path(path)))
        return True


def _args(*, plots: bool = False, output_embedding_dim: int = 3):
    return SimpleNamespace(
        run_tag=None,
        random_seed=7,
        max_history_len=2,
        output_embedding_dim=output_embedding_dim,
        batch_size=2,
        eval_batch_size=3,
        num_dataloader_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_prefetch_factor=1,
        metrics_top_ks=[1],
        no_plots=not plots,
        no_save_model=True,
        disable_progress=True,
        device="cpu",
        author_embedding_dim=3,
        content_projection_dim=4,
        author_projection_dim=2,
        user_hidden_dim=4,
        post_hidden_dim=5,
        dropout_rate_two_tower=0.0,
        author_unknown_dropout_rate=0.0,
        similarity_temperature=0.5,
        epochs=1,
        learning_rate=1.0e-3,
        weight_decay_two_tower=0.0,
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


def test_stage8_trains_native_two_tower_and_publishes_serving_artifacts(
    tmp_path,
    monkeypatch,
):
    stage7_dir, bundle = _stage7_fixture(tmp_path)
    monkeypatch.setattr(
        train_two_tower,
        "resolve_recorded_stage_lineage",
        lambda *args, **kwargs: {"07_dataset_hydration": stage7_dir},
    )
    tracker = _RecordingTracker()
    create_loader = train_two_tower._create_loader
    loader_calls = []

    def record_loader(**kwargs):
        loader_calls.append(kwargs)
        return create_loader(**kwargs)

    monkeypatch.setattr(train_two_tower, "_create_loader", record_loader)
    result = train_two_tower.run(
        _context(tmp_path, tracker),
        _args(plots=True, output_embedding_dim=3),
    )

    output_dir = Path(result["output_dir"])
    checkpoint_path = output_dir / "checkpoints" / "two_tower_best.pth"
    user_tower_path = output_dir / "checkpoints" / "engagement_user_tower.pt"
    post_tower_path = output_dir / "checkpoints" / "engagement_post_tower.pt"
    author_map_path = output_dir / "two_tower_author_idx.parquet"
    manifest_path = output_dir / "checkpoints" / "two_tower_serving_manifest.json"
    model_config = json.loads((output_dir / "model_config.json").read_text())
    training_config = json.loads((output_dir / "training_config.json").read_text())
    training_results = json.loads((output_dir / "training_results.json").read_text())
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    assert model_config["output_embedding_dim"] == 3
    assert model_config["user_encoder_type"] == "cross_attention"
    assert model_config["l2_normalize_embeddings"] is True
    assert "shared_dim" not in json.dumps(model_config)
    assert training_config["output_embedding_dim"] == 3
    assert training_config["candidate_pool"] == "all_hourly_negatives"
    assert training_config["batch_size"] == 2
    assert training_config["eval_batch_size"] == 3
    assert "save_model" not in training_config
    assert training_results["output_embedding_dim"] == 3
    assert checkpoint["metadata"]["model_config"] == model_config
    assert checkpoint["output_embedding_dim"] == 3
    assert training_results["best_epoch"] == 1
    assert set(training_results["final_metrics"]) == {
        "train",
        "val",
        "val_unseen_users",
    }
    assert "mean_average_precision" not in json.dumps(training_results)
    assert "recall" not in json.dumps(training_results).lower()
    assert user_tower_path.stat().st_size > 0
    assert post_tower_path.stat().st_size > 0
    assert (output_dir / "training_history.png").stat().st_size > 0
    assert pl.read_parquet(author_map_path).columns == ["author_did", "author_idx"]
    assert json.loads(manifest_path.read_text()) == {
        "user_tower_clearml_model_id": "engagement_user_tower-id",
        "post_tower_clearml_model_id": "engagement_post_tower-id",
        "user_tower_uri": "gs://models/task/models/engagement_user_tower.pt",
        "post_tower_uri": "gs://models/task/models/engagement_post_tower.pt",
        "output_embedding_dim": 3,
        "clearml_task_id": "task-1",
        "embedding_space_id": "engagement_post_tower-id",
    }
    stage_info = (output_dir / "stage_info.txt").read_text()
    assert "user_tower_clearml_model_id: engagement_user_tower-id" in stage_info
    assert "post_tower_clearml_model_id: engagement_post_tower-id" in stage_info
    assert "author_map_uploaded: True" in stage_info
    assert "serving_manifest_uploaded: True" in stage_info
    assert [name for name, _ in tracker.model_artifacts] == [
        "engagement_user_tower",
        "engagement_post_tower",
    ]
    assert len(tracker.histogram_calls) == 1
    histogram_call = dict(tracker.histogram_calls[0])
    histogram_values = histogram_call.pop("values")
    assert histogram_call == {
        "title": "Random Baseline NDCG@1",
        "series": "Random Baseline",
        "iteration": 0,
        "xlabels": ["Train", "Validation", "Validation Unseen Users"],
        "xaxis": "Split",
        "yaxis": "NDCG@1",
        "labels": ["All observations", "Zero-history only"],
        "mode": "group",
    }
    assert len(histogram_values) == 2
    assert [len(values) for values in histogram_values] == [3, 3]
    assert histogram_values[0] == pytest.approx([1 / 3, 1 / 2, 1 / 2])
    assert histogram_values[1] == pytest.approx([1 / 3, 0.0, 0.0])
    assert [call[3] for call in tracker.scalar_calls if call[1] == "Train NDCG@1"] == [1]
    assert not any(call[3] == 0 for call in tracker.scalar_calls)
    assert [call["batch_size"] for call in loader_calls] == [2, 3, 3]
    assert [call["shuffle"] for call in loader_calls] == [True, False, False]
    assert [call["resample_candidates_each_epoch"] for call in loader_calls] == [
        True,
        False,
        False,
    ]

    eager_model = TwoTowerModel(**model_config["constructor_args"])
    eager_model.load_state_dict(checkpoint["model_state_dict"])
    eager_model.eval()
    dataset = HydratedBucketedEngagementDataset(
        bundle,
        split="train",
        max_history_len=2,
        additional_batch_negatives=None,
        seed=7,
        logger=None,
    )
    batch = dataset.collate_two_tower_batch([dataset[0], dataset[1]])
    scripted_user = torch.jit.load(str(user_tower_path)).eval()
    scripted_post = torch.jit.load(str(post_tower_path)).eval()
    with torch.inference_mode():
        eager_users = eager_model.encode_user(
            batch["history_embeddings"],
            batch["history_mask"],
            batch["history_author_indices"],
        )
        eager_posts = eager_model.encode_post(
            batch["candidate_post_embeddings"],
            batch["candidate_post_author_idx"],
        )
        assert torch.equal(
            eager_users,
            scripted_user(
                batch["history_embeddings"],
                batch["history_mask"],
                batch["history_author_indices"],
            ),
        )
        assert torch.equal(
            eager_posts,
            scripted_post(
                batch["candidate_post_embeddings"],
                batch["candidate_post_author_idx"],
            ),
        )
    assert result["artifacts"]["checkpoint_path"] == str(checkpoint_path)
    assert result["artifacts"]["serving_manifest_path"] == str(manifest_path)


def test_stage8_saves_locally_when_tracker_is_disabled(tmp_path, monkeypatch):
    stage7_dir, _ = _stage7_fixture(tmp_path)
    monkeypatch.setattr(
        train_two_tower,
        "resolve_recorded_stage_lineage",
        lambda *args, **kwargs: {"07_dataset_hydration": stage7_dir},
    )

    result = train_two_tower.run(
        _context(tmp_path, _RecordingTracker(task_id="")),
        _args(),
    )

    output_dir = Path(result["output_dir"])
    assert (output_dir / "checkpoints" / "two_tower_best.pth").is_file()
    assert (output_dir / "checkpoints" / "engagement_user_tower.pt").is_file()
    assert (output_dir / "checkpoints" / "engagement_post_tower.pt").is_file()
    assert (output_dir / "two_tower_author_idx.parquet").is_file()
    assert result["artifacts"]["serving_manifest_path"] is None
    assert result["clearml_publication"]["status"] == "not_configured"


def test_stage8_rejects_stage7_without_loader_index(tmp_path, monkeypatch):
    stage7_dir, bundle = _stage7_fixture(tmp_path)
    shutil.rmtree(bundle / "loader_index")
    monkeypatch.setattr(
        train_two_tower,
        "resolve_recorded_stage_lineage",
        lambda *args, **kwargs: {"07_dataset_hydration": stage7_dir},
    )

    with pytest.raises(ValueError, match="regenerate Stage 7"):
        train_two_tower.run(
            _context(tmp_path, _RecordingTracker()),
            _args(),
        )
