"""Artifact and candidate-universe contracts for BST media comparisons."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import polars as pl
import pytest
import torch

from engagement_prediction.data import post_liker_users, training_index
from engagement_prediction.evaluation.media_artifacts import (
    ValidationSettings,
    collect_candidate_uris,
    resolve_media_artifacts,
)
from engagement_prediction.evaluation.model_performance_test import _bst_config
from engagement_prediction.models.bst_ranker import BSTRanker
from engagement_prediction.training.bst_export import load_bst_checkpoint_model
from engagement_prediction.training.bst_ranker_test import _native_bundle


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture
def stage7(tmp_path):
    root = tmp_path / "stage7"
    root.mkdir()
    bundle = _native_bundle(root)
    queries_path = bundle / "queries" / "part-00000.parquet"
    pl.read_parquet(queries_path).with_columns(
        pl.Series("split", ["val", "val_unseen_users"])
    ).write_parquet(queries_path)
    pl.DataFrame({
        "liker_did": ["liker-a", "liker-b"],
        "liker_idx": [2, 3],
        "training_event_count": [1, 1],
    }, schema=post_liker_users.POST_LIKER_USER_VOCABULARY_SCHEMA).write_parquet(
        bundle / "post_liker_users" / "part-00000.parquet"
    )
    shutil.rmtree(bundle / "loader_index")
    training_index.build_loader_index(
        posts_path=bundle / "posts",
        queries_path=bundle / "queries",
        query_positives_path=bundle / "query_positives",
        query_histories_path=bundle / "query_histories",
        hourly_negative_candidates_path=bundle / "hourly_negative_candidates",
        embeddings_path=bundle / "embeddings.npy",
        authors_path=bundle / "authors",
        indexed_post_liker_events_path=bundle / "indexed_post_liker_events",
        post_liker_users_path=bundle / "post_liker_users",
        output_path=bundle / "loader_index",
        logger=None,
    )
    stages = (
        "00_source_metadata", "01_query_selection", "02_user_history",
        "03_post_selection", "04_negative_selection", "05_post_liker_history",
        "06_author_statistics",
    )
    lineage = {stage: str(tmp_path / stage / "input-run") for stage in stages}
    _write_json(root / "manifest.json", {
        "stage_key": "dataset_hydration",
        "stage_folder": "07_dataset_hydration",
        "stage_run_id": root.name,
        "status": "complete",
        "inputs": lineage,
    })
    _write_json(root / "summary.json", {
        "parameters": {
            "embedding_model": "fixture-model",
            "embedding_dim": 2,
            "min_post_liker_user_training_event_count": 1,
            "max_post_liker_user_vocabulary_size": 2,
        },
        "outputs": {"hydrated_training_data_path": bundle.name},
    })
    return root, bundle, {**lineage, "07_dataset_hydration": str(root)}


def _model(path: Path, stage7, *, use_post_liker_feature: bool) -> Path:
    stage7_root, bundle, lineage = stage7
    path.mkdir()
    checkpoints = path / "checkpoints"
    checkpoints.mkdir()
    config = _bst_config(max_history_len=2, author_table_num_rows=6)
    config.update(post_liker_user_pad_idx=0, post_liker_user_unk_idx=1)
    config["constructor_args"]["use_post_liker_feature"] = use_post_liker_feature
    config["constructor_args"]["post_liker_user_table_num_rows"] = 4 if use_post_liker_feature else 2
    training_config = {
        "lineage": lineage,
        "stage7_dir": str(stage7_root),
        "stage7_bundle": str(bundle),
        "loader_index_format_version": training_index.FORMAT_VERSION,
        "eval_batch_size": 2,
        "bst_additional_batch_negatives": 1,
        "random_seed": 42,
        "bst_use_post_liker_feature": use_post_liker_feature,
        "bst_max_post_liker_replay_events_per_post": 2,
    }
    popularity = {"enabled": False, "log_mean": 0.0, "log_std": 1.0}
    checkpoint_path = checkpoints / "bst_ranker_best.pth"
    model = BSTRanker(**config["constructor_args"])
    torch.save({
        "best_epoch": 2,
        "epoch": 2,
        "model_state_dict": model.state_dict(),
        "metadata": {"model_config": config, "popularity_stats": popularity},
    }, checkpoint_path)
    _write_json(path / "manifest.json", {
        "stage_key": "train_bst_ranker",
        "stage_folder": "08_train_bst_ranker",
        "stage_run_id": path.name,
        "status": "complete",
        "inputs": lineage,
    })
    _write_json(path / "model_config.json", config)
    _write_json(path / "training_config.json", training_config)
    _write_json(path / "popularity_stats.json", popularity)
    _write_json(path / "summary.json", {
        "input": {
            "dataset_hydration_dir": str(stage7_root),
            "hydrated_training_data_path": str(bundle),
        },
        "parameters": training_config,
        "model": config,
        "popularity": popularity,
        "outputs": {"checkpoint_path": str(checkpoint_path)},
    })
    pl.read_parquet(bundle / "authors" / "part-00000.parquet").select(
        "author_did", "author_idx"
    ).write_parquet(path / "ranker_author_idx.parquet")
    if use_post_liker_feature:
        shutil.copytree(bundle / "post_liker_users", path / "post_liker_users")
    return path


@pytest.mark.parametrize("model_count", [1, 2, 3])
def test_resolve_one_or_more_canonical_bst_models(tmp_path, stage7, model_count):
    specs = [
        (f"model-{i}", _model(tmp_path / f"model-{i}", stage7, use_post_liker_feature=i % 2 == 0))
        for i in range(model_count)
    ]
    dataset, models, settings = resolve_media_artifacts(specs)
    assert dataset.bundle_path == stage7[1]
    assert [model.name for model in models] == [name for name, _ in specs]
    assert settings == ValidationSettings(batch_size=2, additional_batch_negatives=1, random_seed=42)
    assert all(len(model.checkpoint_sha256) == 64 for model in models)
    assert models[0].to_dict()["checkpoint_path"].endswith("bst_ranker_best.pth")
    eager, best_epoch = load_bst_checkpoint_model(
        checkpoint_path=models[0].checkpoint_path,
        expected_model_config=models[0].model_config,
        expected_popularity_stats=models[0].popularity_stats,
    )
    assert best_epoch == 2
    assert eager.use_post_liker_feature is True


def test_candidate_union_covers_both_splits_and_excludes_history(tmp_path, stage7, monkeypatch):
    model_path = _model(tmp_path / "model", stage7, use_post_liker_feature=False)
    dataset, _, _ = resolve_media_artifacts([("model", model_path)])
    # Enumeration remains independent of the feature embedding file.
    (stage7[1] / "embeddings.npy").unlink()
    actual = collect_candidate_uris(dataset)
    assert actual.to_dicts() == [
        {"at_uri": "n1", "in_val": True, "in_val_unseen_users": True},
        {"at_uri": "p1", "in_val": True, "in_val_unseen_users": False},
        {"at_uri": "p2", "in_val": False, "in_val_unseen_users": True},
    ]
    assert actual.schema == pl.Schema({"at_uri": pl.String, "in_val": pl.Boolean, "in_val_unseen_users": pl.Boolean})


@pytest.mark.parametrize("field,value", [
    ("eval_batch_size", 3), ("bst_additional_batch_negatives", 2), ("random_seed", 43),
])
def test_rejects_different_validation_slates(tmp_path, stage7, field, value):
    first = _model(tmp_path / "first", stage7, use_post_liker_feature=False)
    second = _model(tmp_path / "second", stage7, use_post_liker_feature=False)
    training = _read_json(second / "training_config.json")
    training[field] = value
    _write_json(second / "training_config.json", training)
    summary = _read_json(second / "summary.json")
    summary["parameters"] = training
    _write_json(second / "summary.json", summary)
    with pytest.raises(ValueError, match="saved validation"):
        resolve_media_artifacts([("first", first), ("second", second)])


def test_rejects_different_ancestor_even_with_same_stage7(tmp_path, stage7):
    first = _model(tmp_path / "first", stage7, use_post_liker_feature=False)
    second = _model(tmp_path / "second", stage7, use_post_liker_feature=False)
    for filename, nesting in (("manifest.json", ["inputs"]), ("training_config.json", ["lineage"]), ("summary.json", ["parameters", "lineage"])):
        document = _read_json(second / filename)
        mapping = document
        for key in nesting:
            mapping = mapping[key]
        mapping["01_query_selection"] = str(tmp_path / "other-selection")
        _write_json(second / filename, document)
    with pytest.raises(ValueError, match="input data differs"):
        resolve_media_artifacts([("first", first), ("second", second)])


@pytest.mark.parametrize("field", ["stage7_dir", "stage7_bundle", "lineage", "eval_batch_size"])
def test_rejects_summary_configuration_disagreement(tmp_path, stage7, field):
    path = _model(tmp_path / "model", stage7, use_post_liker_feature=False)
    summary = _read_json(path / "summary.json")
    summary["parameters"][field] = {} if field == "lineage" else 17 if field == "eval_batch_size" else str(tmp_path / "wrong")
    _write_json(path / "summary.json", summary)
    with pytest.raises(ValueError, match="summary"):
        resolve_media_artifacts([("model", path)])


def test_resolves_symlink_input_aliases(tmp_path, stage7):
    path = _model(tmp_path / "model", stage7, use_post_liker_feature=False)
    alias = tmp_path / "stage7-alias"
    alias.symlink_to(stage7[0], target_is_directory=True)
    training = _read_json(path / "training_config.json")
    training["stage7_dir"] = str(alias)
    training["stage7_bundle"] = str(alias / stage7[1].name)
    training["lineage"]["07_dataset_hydration"] = str(alias)
    _write_json(path / "training_config.json", training)
    dataset, _, _ = resolve_media_artifacts([("model", path)])
    assert dataset.root == stage7[0]


@pytest.mark.parametrize("kind", ["authors", "post_liker_users"])
def test_rejects_same_size_different_vocabularies(tmp_path, stage7, kind):
    path = _model(tmp_path / "model", stage7, use_post_liker_feature=True)
    if kind == "authors":
        map_path = path / "ranker_author_idx.parquet"
        index_column = "author_idx"
    else:
        map_path = path / "post_liker_users" / "part-00000.parquet"
        index_column = "liker_idx"
    mapping = pl.read_parquet(map_path)
    mapping.with_columns(pl.col(index_column).reverse()).write_parquet(map_path)
    with pytest.raises(ValueError, match="vocabulary differs"):
        resolve_media_artifacts([("model", path)])


@pytest.mark.parametrize("failure", ["incomplete", "legacy", "missing_checkpoint", "bad_json", "embedding"])
def test_rejects_invalid_model_artifacts(tmp_path, stage7, failure):
    path = _model(tmp_path / "model", stage7, use_post_liker_feature=False)
    if failure in {"incomplete", "legacy"}:
        manifest = _read_json(path / "manifest.json")
        manifest["status" if failure == "incomplete" else "stage_folder"] = "incomplete" if failure == "incomplete" else "03_train"
        _write_json(path / "manifest.json", manifest)
    elif failure == "missing_checkpoint":
        (path / "checkpoints" / "bst_ranker_best.pth").unlink()
    elif failure == "bad_json":
        (path / "training_config.json").write_text("not json")
    else:
        config = _read_json(path / "model_config.json")
        config["embedding_model"] = "different-embeddings"
        _write_json(path / "model_config.json", config)
        summary = _read_json(path / "summary.json")
        summary["model"] = config
        _write_json(path / "summary.json", summary)
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_media_artifacts([("model", path)])


def test_rejects_stage7_lineage_disagreement(tmp_path, stage7):
    path = _model(tmp_path / "model", stage7, use_post_liker_feature=False)
    manifest_path = stage7[0] / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["inputs"]["02_user_history"] = str(tmp_path / "different-history")
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Stage 7 manifest lineage"):
        resolve_media_artifacts([("model", path)])


def test_requires_unique_nonempty_model_names():
    for specs in ([], [("", Path("unused"))], [("same", Path("one")), ("same", Path("two"))]):
        with pytest.raises(ValueError):
            resolve_media_artifacts(specs)
