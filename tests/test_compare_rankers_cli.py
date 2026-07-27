import json
import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

import cli
import compare


def _make_stage_output(
    artifacts_dir: Path,
    stage_folder: str,
    stage_run_id: str,
    *,
    inputs=None,
) -> Path:
    out_dir = artifacts_dir / stage_folder / stage_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps({
        "stage_folder": stage_folder,
        "stage_run_id": stage_run_id,
        "inputs": inputs or {},
    }) + "\n")
    return out_dir


def _compare_checkpoint_config(
    max_history_len=7,
    use_author_embedding_table=True,
    *,
    model_type="two_tower",
    use_post_liker_user_pooling=False,
    max_post_liker_replay_events_per_post=13,
):
    config = {
        "model_type": model_type,
        "max_history_len": max_history_len,
        "use_author_embedding_table": use_author_embedding_table,
        "author_embedding_dim": 2,
        "content_projection_dim": 4,
        "author_projection_dim": 2,
        "author_table_num_rows": 8,
        "author_unknown_dropout_rate": 0.0,
    }
    if model_type == "bst-ranker":
        config.update({
            "bst_use_post_liker_user_pooling": use_post_liker_user_pooling,
            "bst_max_post_liker_replay_events_per_post": max_post_liker_replay_events_per_post,
        })
    return config


def _write_compare_checkpoint(path: Path, config: dict) -> None:
    torch.save({"model_state_dict": {}, "config": config}, path)


def test_compare_rankers_parser_accepts_repeated_models():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "compare-rankers",
        "--model", "tt:two-tower:/tmp/two_tower.pth",
        "--model", "bst:bst-ranker:/tmp/bst.pth",
        "--splits", "val", "holdout_unseen_users",
        "--bst-candidate-chunk-size", "512",
        "--bst-max-post-liker-replay-events-per-post", "19",
    ])

    assert raw.command == "compare-rankers"
    assert raw.model == [
        "tt:two-tower:/tmp/two_tower.pth",
        "bst:bst-ranker:/tmp/bst.pth",
    ]
    assert raw.splits == ["val", "holdout_unseen_users"]
    assert raw.bst_candidate_chunk_size == 512
    assert raw.bst_max_post_liker_replay_events_per_post == 19


def test_implicit_run_all_parser_still_defaults_to_run_all():
    parser = cli.build_parser()
    raw = parser.parse_args(["--epochs", "3"])
    merged = cli._merge_args_with_config(raw)

    assert merged.command == "run-all"
    assert merged.epochs == 3


@pytest.mark.parametrize(
    "raw_spec",
    [
        "missing_parts",
        ":two-tower:/tmp/model.pth",
        "name:unknown:/tmp/model.pth",
        "name:two-tower:",
    ],
)
def test_parse_compare_model_spec_rejects_invalid_specs(raw_spec):
    with pytest.raises(ValueError):
        compare._parse_compare_model_spec(raw_spec)


def test_compare_model_spec_preserves_absolute_checkpoint_path(tmp_path):
    checkpoint_path = tmp_path / "two_tower.pth"
    checkpoint_path.write_bytes(b"checkpoint")

    spec = compare._parse_compare_model_spec(f"tt:two-tower:{checkpoint_path}")
    resolved = compare._resolve_compare_checkpoint_path(spec["checkpoint_path"])

    assert spec["checkpoint_path"] == str(checkpoint_path)
    assert resolved == checkpoint_path.resolve()


def test_compare_max_history_len_comes_from_model_configs():
    specs = [
        {"name": "tt", "checkpoint_path": "/tmp/two_tower.pth"},
        {"name": "bst", "checkpoint_path": "/tmp/bst.pth"},
    ]
    configs = {
        "tt": _compare_checkpoint_config(max_history_len=11),
        "bst": _compare_checkpoint_config(max_history_len=11),
    }

    resolved = compare._resolve_compare_max_history_len(
        argparse.Namespace(),
        model_specs=specs,
        model_configs=configs,
    )

    assert resolved == 11


def test_compare_max_history_len_fails_when_missing_without_cli_override():
    specs = [{"name": "tt", "checkpoint_path": "/tmp/two_tower.pth"}]
    configs = {"tt": {"use_author_embedding_table": True}}

    with pytest.raises(ValueError, match="max_history_len"):
        compare._resolve_compare_max_history_len(
            argparse.Namespace(),
            model_specs=specs,
            model_configs=configs,
        )


def test_compare_max_history_len_fails_when_model_configs_disagree():
    specs = [
        {"name": "tt", "checkpoint_path": "/tmp/two_tower.pth"},
        {"name": "bst", "checkpoint_path": "/tmp/bst.pth"},
    ]
    configs = {
        "tt": _compare_checkpoint_config(max_history_len=7),
        "bst": _compare_checkpoint_config(max_history_len=9),
    }

    with pytest.raises(ValueError, match="matching max_history_len"):
        compare._resolve_compare_max_history_len(
            argparse.Namespace(),
            model_specs=specs,
            model_configs=configs,
        )


def test_compare_max_history_len_cli_override_allows_config_disagreement():
    specs = [
        {"name": "tt", "checkpoint_path": "/tmp/two_tower.pth"},
        {"name": "bst", "checkpoint_path": "/tmp/bst.pth"},
    ]
    configs = {
        "tt": _compare_checkpoint_config(max_history_len=7),
        "bst": _compare_checkpoint_config(max_history_len=9),
    }

    resolved = compare._resolve_compare_max_history_len(
        argparse.Namespace(max_history_len=12),
        model_specs=specs,
        model_configs=configs,
    )

    assert resolved == 12


def test_compare_post_liker_replay_cap_comes_from_enabled_model_configs():
    specs = [
        {"name": "tt", "model_type": "two-tower", "checkpoint_path": "/tmp/two_tower.pth"},
        {"name": "bst", "model_type": "bst-ranker", "checkpoint_path": "/tmp/bst.pth"},
    ]
    configs = {
        "tt": _compare_checkpoint_config(),
        "bst": _compare_checkpoint_config(
            model_type="bst-ranker",
            use_post_liker_user_pooling=True,
            max_post_liker_replay_events_per_post=17,
        ),
    }

    resolved = compare._resolve_compare_max_post_liker_replay_events_per_post(
        argparse.Namespace(),
        model_specs=specs,
        model_configs=configs,
    )

    assert resolved == 17


def test_compare_post_liker_replay_cap_is_none_when_feature_is_disabled():
    specs = [
        {"name": "bst", "model_type": "bst-ranker", "checkpoint_path": "/tmp/bst.pth"},
    ]
    configs = {
        "bst": _compare_checkpoint_config(
            model_type="bst-ranker",
            use_post_liker_user_pooling=False,
        ),
    }

    resolved = compare._resolve_compare_max_post_liker_replay_events_per_post(
        argparse.Namespace(),
        model_specs=specs,
        model_configs=configs,
    )

    assert resolved is None


def test_compare_post_liker_replay_cap_requires_matching_enabled_models():
    specs = [
        {"name": "left", "model_type": "bst-ranker", "checkpoint_path": "/tmp/left.pth"},
        {"name": "right", "model_type": "bst-ranker", "checkpoint_path": "/tmp/right.pth"},
    ]
    configs = {
        "left": _compare_checkpoint_config(
            model_type="bst-ranker",
            use_post_liker_user_pooling=True,
            max_post_liker_replay_events_per_post=17,
        ),
        "right": _compare_checkpoint_config(
            model_type="bst-ranker",
            use_post_liker_user_pooling=True,
            max_post_liker_replay_events_per_post=23,
        ),
    }

    with pytest.raises(
        ValueError,
        match="requires matching bst_max_post_liker_replay_events_per_post",
    ):
        compare._resolve_compare_max_post_liker_replay_events_per_post(
            argparse.Namespace(),
            model_specs=specs,
            model_configs=configs,
        )


def test_compare_post_liker_replay_cap_cli_override_allows_config_disagreement():
    specs = [
        {"name": "left", "model_type": "bst-ranker", "checkpoint_path": "/tmp/left.pth"},
        {"name": "right", "model_type": "bst-ranker", "checkpoint_path": "/tmp/right.pth"},
    ]
    configs = {
        "left": _compare_checkpoint_config(
            model_type="bst-ranker",
            use_post_liker_user_pooling=True,
            max_post_liker_replay_events_per_post=17,
        ),
        "right": _compare_checkpoint_config(
            model_type="bst-ranker",
            use_post_liker_user_pooling=True,
            max_post_liker_replay_events_per_post=23,
        ),
    }

    resolved = compare._resolve_compare_max_post_liker_replay_events_per_post(
        argparse.Namespace(bst_max_post_liker_replay_events_per_post=19),
        model_specs=specs,
        model_configs=configs,
    )

    assert resolved == 19


def test_compare_rankers_requires_author_embedding_config():
    spec = {"name": "tt", "checkpoint_path": "/tmp/two_tower.pth"}

    with pytest.raises(ValueError, match="use_author_embedding_table=True"):
        compare._validate_compare_author_config(
            spec,
            _compare_checkpoint_config(use_author_embedding_table=False),
        )


def test_compare_rankers_requires_bst_projection_config():
    spec = {"name": "bst", "model_type": "bst-ranker", "checkpoint_path": "/tmp/bst.pth"}
    config = _compare_checkpoint_config()
    del config["content_projection_dim"]

    with pytest.raises(ValueError, match="content_projection_dim"):
        compare._validate_compare_bst_config(spec, config)


def test_compare_rankers_rejects_config(tmp_path):
    parser = cli.build_parser()
    raw = parser.parse_args([
        "compare-rankers",
        "--config", str(tmp_path / "config.yml"),
        "--model", "tt:two-tower:/tmp/two_tower.pth",
    ])

    with pytest.raises(SystemExit, match="config"):
        cli.cmd_compare_rankers(raw)


@pytest.mark.parametrize("use_post_liker_user_pooling", [False, True])
def test_compare_rankers_evaluates_models_and_writes_metrics(
    tmp_path,
    monkeypatch,
    use_post_liker_user_pooling,
):
    output_root = tmp_path / "outputs"
    artifacts_dir = output_root / "artifacts"
    get_data_dir = _make_stage_output(artifacts_dir, "01_get_data", "20260101_000000_get")
    history_dir = _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260102_000000_history",
        inputs={"01_get_data": str(get_data_dir.resolve())},
    )
    two_tower_checkpoint = tmp_path / "two_tower.pth"
    bst_checkpoint = tmp_path / "bst.pth"
    _write_compare_checkpoint(two_tower_checkpoint, _compare_checkpoint_config(max_history_len=7))
    _write_compare_checkpoint(
        bst_checkpoint,
        _compare_checkpoint_config(
            max_history_len=9,
            model_type="bst-ranker",
            use_post_liker_user_pooling=use_post_liker_user_pooling,
        ),
    )

    import utils.dataloaders as dataloaders
    import utils.matrix_ranking as matrix_ranking
    import utils.ranking_adapters as ranking_adapters

    created_datasets = []
    eval_calls = []
    post_liker_artifact_calls = []
    post_liker_events_df = object()
    post_liker_user_idx_df = object()
    post_liker_event_lookup = object()

    def fake_load_bucketed_training_data(context, logger=None, require_target_hour_history_popularity=False):
        assert require_target_hour_history_popularity is False
        return (
            np.zeros((4, 2), dtype=np.float32),
            object(),
            object(),
            object(),
            object(),
            2,
        )

    def fake_load_post_liker_event_artifacts(context, logger=None):
        post_liker_artifact_calls.append(context)
        return post_liker_events_df, post_liker_user_idx_df

    class FakePostLikerEventLookup:
        @classmethod
        def from_dataframe(cls, events_df):
            assert events_df is post_liker_events_df
            return post_liker_event_lookup

    class FakeBucketedDataset:
        def __init__(
            self,
            embeddings_mmap,
            likes_core_df,
            posts_core_df,
            history_df,
            split,
            max_history_len,
            embed_dim,
            use_author_embedding_table,
            use_popularity_feature,
            use_post_liker_user_pooling,
            post_liker_event_lookup,
            post_liker_user_idx_df,
            max_post_liker_replay_events_per_post,
            logger,
        ):
            created_datasets.append({
                "split": split,
                "use_author_embedding_table": use_author_embedding_table,
                "use_popularity_feature": use_popularity_feature,
                "use_post_liker_user_pooling": use_post_liker_user_pooling,
                "post_liker_event_lookup": post_liker_event_lookup,
                "post_liker_user_idx_df": post_liker_user_idx_df,
                "max_post_liker_replay_events_per_post": max_post_liker_replay_events_per_post,
                "max_history_len": max_history_len,
                "embed_dim": embed_dim,
            })
            self.split = split
            self.row_indices_by_bucket = {} if split == "empty" else {split: [0]}

        def __len__(self):
            return sum(len(row_indices) for row_indices in self.row_indices_by_bucket.values())

        def __getitem__(self, idx):
            return {"row_idx": idx}

        def collate_batch(self, items):
            return {"label_matrix": torch.tensor([[1.0, 0.0]], dtype=torch.float32)}

    class FakeTwoTowerAdapter:
        def __init__(self, checkpoint_path):
            self.checkpoint_path = checkpoint_path

    class FakeBstAdapter:
        def __init__(self, checkpoint_path, candidate_chunk_size):
            self.checkpoint_path = checkpoint_path
            self.candidate_chunk_size = candidate_chunk_size

    def fake_evaluate_matrix_scorer(
        adapter,
        data_loader,
        device,
        metrics_top_ks,
        collect_ranking_rows=False,
        progress_desc=None,
        disable_progress=True,
    ):
        batch = next(iter(data_loader))
        assert batch["label_matrix"].shape == (1, 2)
        eval_calls.append({
            "adapter": adapter,
            "device": device,
            "metrics_top_ks": metrics_top_ks,
            "progress_desc": progress_desc,
        })
        return {
            "metrics": {
                "auc_roc": 0.5,
                "ndcg@30": 0.75,
                "classification_metric_sampled": False,
                "loss": None,
            },
            "ranking_rows": [],
        }

    monkeypatch.setattr(dataloaders, "load_bucketed_training_data", fake_load_bucketed_training_data)
    monkeypatch.setattr(dataloaders, "load_post_liker_event_artifacts", fake_load_post_liker_event_artifacts)
    monkeypatch.setattr(dataloaders, "PostLikerEventLookup", FakePostLikerEventLookup)
    monkeypatch.setattr(dataloaders, "BucketedEngagementDataset", FakeBucketedDataset)
    monkeypatch.setattr(ranking_adapters, "TwoTowerPthAdapter", FakeTwoTowerAdapter)
    monkeypatch.setattr(ranking_adapters, "BstPthAdapter", FakeBstAdapter)
    monkeypatch.setattr(matrix_ranking, "evaluate_matrix_scorer", fake_evaluate_matrix_scorer)

    parser = cli.build_parser()
    raw = parser.parse_args([
        "compare-rankers",
        "--output-dir", str(output_root),
        "--prior-01-get-data", str(get_data_dir),
        "--prior-02-user-history", str(history_dir),
        "--model", f"tt:two-tower:{two_tower_checkpoint}",
        "--model", f"bst:bst-ranker:{bst_checkpoint}",
        "--splits", "val", "empty",
        "--metrics-top-ks", "30",
        "--batch-size", "2",
        "--max-history-len", "5",
        "--num-dataloader-workers", "0",
        "--device", "cpu",
        "--bst-candidate-chunk-size", "17",
    ])

    assert cli.cmd_compare_rankers(raw) == 0

    assert len(created_datasets) == 2
    assert [dataset["split"] for dataset in created_datasets] == ["val", "empty"]
    assert all(dataset["use_author_embedding_table"] is True for dataset in created_datasets)
    assert all(dataset["use_popularity_feature"] is False for dataset in created_datasets)
    assert all(
        dataset["use_post_liker_user_pooling"] is use_post_liker_user_pooling
        for dataset in created_datasets
    )
    assert all(dataset["max_history_len"] == 5 for dataset in created_datasets)
    assert all(dataset["embed_dim"] == 2 for dataset in created_datasets)
    if use_post_liker_user_pooling:
        assert len(post_liker_artifact_calls) == 1
        assert all(
            dataset["post_liker_event_lookup"] is post_liker_event_lookup
            for dataset in created_datasets
        )
        assert all(
            dataset["post_liker_user_idx_df"] is post_liker_user_idx_df
            for dataset in created_datasets
        )
        assert all(
            dataset["max_post_liker_replay_events_per_post"] == 13
            for dataset in created_datasets
        )
    else:
        assert post_liker_artifact_calls == []
        assert all(
            dataset["post_liker_event_lookup"] is None
            for dataset in created_datasets
        )
        assert all(
            dataset["post_liker_user_idx_df"] is None
            for dataset in created_datasets
        )
        assert all(
            dataset["max_post_liker_replay_events_per_post"] is None
            for dataset in created_datasets
        )
    assert len(eval_calls) == 2
    assert all(call["device"] == "cpu" for call in eval_calls)
    assert isinstance(eval_calls[0]["adapter"], FakeTwoTowerAdapter)
    assert isinstance(eval_calls[1]["adapter"], FakeBstAdapter)
    assert eval_calls[1]["adapter"].candidate_chunk_size == 17

    compare_dirs = list((artifacts_dir / "compare_rankers").iterdir())
    assert len(compare_dirs) == 1
    out_dir = compare_dirs[0]
    metrics_summary = json.loads((out_dir / "metrics.json").read_text())
    assert metrics_summary["skipped_splits"] == ["empty"]
    assert metrics_summary["max_history_len"] == 5
    assert metrics_summary["bst_use_post_liker_user_pooling"] is use_post_liker_user_pooling
    assert metrics_summary["bst_max_post_liker_replay_events_per_post"] == (
        13 if use_post_liker_user_pooling else None
    )
    assert set(metrics_summary["metrics"].keys()) == {"tt", "bst"}
    assert set(metrics_summary["metrics"]["tt"].keys()) == {"val"}
    assert (out_dir / "metrics.csv").read_text().startswith("model_name,model_type,checkpoint_path,split,metric,value\n")
    assert json.loads((out_dir / "model_specs.json").read_text())[0]["name"] == "tt"
    assert "stage: compare_rankers" in (out_dir / "stage_info.txt").read_text()
