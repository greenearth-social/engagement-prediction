from __future__ import annotations

import copy
import csv
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import polars as pl
import pytest
import torch

from engagement_prediction.data.datasets_test import _bundle
from engagement_prediction.data.training_index import FORMAT_VERSION
from engagement_prediction.evaluation.artifacts import (
    LEGACY_BST_ARTIFACT_FORMAT,
    ModelArtifact,
    resolve_model_artifact,
    resolve_stage7_artifact,
    validate_comparison_contract,
)
from engagement_prediction.evaluation.author_mapping import (
    build_author_index_override,
    temporary_author_index_override,
)
from engagement_prediction.evaluation.comparison import (
    ComparisonSettings,
    run_model_comparison,
)
from engagement_prediction.evaluation.reporting import (
    build_metric_deltas,
    round_output_floats,
    write_metric_deltas_csv,
    write_metrics_csv,
)
from engagement_prediction.evaluation.scorers import (
    BSTTorchScriptScorer,
    TwoTowerTorchScriptScorer,
)
from engagement_prediction.models.bst_ranker import BSTRanker
from engagement_prediction.models.two_tower import TwoTowerModel
from engagement_prediction.training.ranking import (
    MatrixBatchScores,
    evaluate_matrix_scorer,
)


def _write_manifest(path: Path, *, stage_key: str, stage_folder: str, inputs: dict) -> None:
    (path / "manifest.json").write_text(json.dumps({
        "stage_key": stage_key,
        "stage_folder": stage_folder,
        "stage_run_id": path.name,
        "inputs": inputs,
    }) + "\n")


def _author_map(path: Path, author_dids: list[str]) -> Path:
    pl.DataFrame({
        "author_did": author_dids,
        "author_idx": pl.Series(range(2, len(author_dids) + 2), dtype=pl.UInt32),
    }).write_parquet(path)
    return path


def _bst_config(*, max_history_len: int, author_table_num_rows: int) -> dict:
    return {
        "model_type": "bst-ranker",
        "embedding_model": "fixture-model",
        "max_history_len": max_history_len,
        "author_pad_idx": 0,
        "author_unk_idx": 1,
        "constructor_args": {
            "post_embedding_dim": 2,
            "author_table_num_rows": author_table_num_rows,
            "author_embedding_dim": 2,
            "content_projection_dim": 2,
            "author_projection_dim": 2,
            "model_dim": 4,
            "time_embedding_dim": 2,
            "num_attention_heads": 2,
            "num_transformer_layers": 1,
            "transformer_ff_dim": 8,
            "dropout_rate": 0.0,
            "author_unknown_dropout_rate": 0.0,
            "norm_first": False,
            "time_delta_bucket_boundaries_hours": [1.0, 24.0],
            "prediction_hidden_dims": [4],
            "use_popularity_feature": False,
            "popularity_projection_dim": 2,
            "popularity_log_mean": 0.0,
            "popularity_log_std": 1.0,
        },
    }


def _two_tower_config(*, max_history_len: int, author_table_num_rows: int) -> dict:
    constructor = {
        "post_embedding_dim": 2,
        "author_table_num_rows": author_table_num_rows,
        "author_embedding_dim": 2,
        "content_projection_dim": 3,
        "author_projection_dim": 2,
        "user_hidden_dim": 4,
        "post_hidden_dim": 4,
        "output_embedding_dim": 3,
        "max_history_len": max_history_len,
        "dropout_rate": 0.0,
        "author_unknown_dropout_rate": 0.0,
        "similarity_temperature": 0.5,
    }
    return {
        "model_type": "two-tower",
        "user_encoder_type": "cross_attention",
        "embedding_model": "fixture-model",
        "output_embedding_dim": 3,
        "max_history_len": max_history_len,
        "use_author_embedding_table": True,
        "use_post_encoder": True,
        "l2_normalize_embeddings": True,
        "author_pad_idx": 0,
        "author_unk_idx": 1,
        "constructor_args": constructor,
    }


def _write_model_artifact(
    path: Path,
    *,
    model_type: str,
    max_history_len: int,
    author_dids: list[str],
    stage7_dir: Path,
) -> Path:
    path.mkdir(parents=True)
    checkpoints = path / "checkpoints"
    checkpoints.mkdir()
    author_table_num_rows = len(author_dids) + 2
    if model_type == "bst-ranker":
        config = _bst_config(
            max_history_len=max_history_len,
            author_table_num_rows=author_table_num_rows,
        )
        torch.jit.script(BSTRanker(**config["constructor_args"])).save(
            str(checkpoints / "ranker.pt")
        )
        _author_map(path / "ranker_author_idx.parquet", author_dids)
        stage_key = "train_bst_ranker"
        stage_folder = "08_train_bst_ranker"
    else:
        config = _two_tower_config(
            max_history_len=max_history_len,
            author_table_num_rows=author_table_num_rows,
        )
        model = TwoTowerModel(**config["constructor_args"])
        torch.jit.script(copy.deepcopy(model.user_tower)).save(
            str(checkpoints / "engagement_user_tower.pt")
        )
        torch.jit.script(copy.deepcopy(model.post_tower)).save(
            str(checkpoints / "engagement_post_tower.pt")
        )
        _author_map(path / "two_tower_author_idx.parquet", author_dids)
        stage_key = "train_two_tower"
        stage_folder = "08_train_two_tower"
    (path / "model_config.json").write_text(json.dumps(config) + "\n")
    (path / "training_config.json").write_text(json.dumps({
        "stage7_dir": str(stage7_dir),
        "stage7_bundle": str(stage7_dir / "hydrated_training_data_fixture"),
        "loader_index_format_version": FORMAT_VERSION,
    }) + "\n")
    _write_manifest(
        path,
        stage_key=stage_key,
        stage_folder=stage_folder,
        inputs={"07_dataset_hydration": str(stage7_dir)},
    )
    return path


def _write_legacy_stage1_artifact(
    path: Path,
    *,
    author_dids: list[str],
) -> tuple[Path, Path]:
    path.mkdir(parents=True)
    author_map_path = path / "author_idx.parquet"
    pl.DataFrame({
        "author_did": author_dids,
        "author_train_count": pl.Series(
            [100] * len(author_dids),
            dtype=pl.UInt32,
        ),
        "author_idx": pl.Series(
            range(2, len(author_dids) + 2),
            dtype=pl.UInt32,
        ),
    }).write_parquet(author_map_path)
    (path / "summary.json").write_text(json.dumps({
        "parameters": {"embedding_model": "fixture-model"},
        "outputs": {
            "embedding_dim": 2,
            "author_idx_file": author_map_path.name,
        },
    }) + "\n")
    _write_manifest(
        path,
        stage_key="get_data",
        stage_folder="01_get_data",
        inputs={},
    )
    return path, author_map_path


def _write_legacy_bst_artifact(
    path: Path,
    *,
    author_dids: list[str],
    stage1_path: Path | None,
    script_module: torch.nn.Module | None = None,
) -> Path:
    path.mkdir(parents=True)
    checkpoints = path / "checkpoints"
    checkpoints.mkdir()
    author_table_num_rows = len(author_dids) + 2
    config = _bst_config(
        max_history_len=2,
        author_table_num_rows=author_table_num_rows,
    )
    module = script_module or BSTRanker(**config["constructor_args"])
    torch.jit.script(module).save(str(checkpoints / "ranker.pt"))
    (path / "training_config.json").write_text(json.dumps({
        "model_type": "bst-ranker",
        "use_author_embedding_table": True,
        "post_embedding_dim": 2,
        "max_history_len": 2,
        "author_table_num_rows": author_table_num_rows,
        "author_pad_idx": 0,
        "author_unk_idx": 1,
    }) + "\n")
    (path / "resolved_config.json").write_text(json.dumps({
        "model_type": "bst-ranker",
        "embedding_model": "fixture-model",
    }) + "\n")
    _write_manifest(
        path,
        stage_key="train_bst_ranker",
        stage_folder="03_train",
        inputs=(
            {"01_get_data": str(stage1_path)}
            if stage1_path is not None
            else {}
        ),
    )
    return path


def _write_stage7_artifact(tmp_path: Path) -> tuple[Path, Path]:
    stage7 = tmp_path / "stage7"
    stage7.mkdir()
    bundle = _bundle(stage7)
    # The dataset helper names the bundle deterministically and constructs the
    # full canonical loader index, including empty canonical splits.
    summary = {
        "parameters": {
            "embedding_model": "fixture-model",
            "embedding_dim": 2,
        },
        "outputs": {"hydrated_training_data_path": bundle.name},
    }
    (stage7 / "summary.json").write_text(json.dumps(summary) + "\n")
    _write_manifest(
        stage7,
        stage_key="dataset_hydration",
        stage_folder="07_dataset_hydration",
        inputs={},
    )
    return stage7, bundle


class _LegacyBSTWithExtraArgument(torch.nn.Module):
    @torch.jit.export
    def score_candidate_matrix(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        candidate_post_author_idx: torch.Tensor,
        history_prior_cumulative_likes: Optional[torch.Tensor] = None,
        candidate_prior_cumulative_likes: Optional[torch.Tensor] = None,
        target_user_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del history_mask, history_time_deltas_hours, history_author_indices
        del candidate_post_author_idx, history_prior_cumulative_likes
        del candidate_prior_cumulative_likes, target_user_embeddings
        users = history_embeddings.sum(dim=(1, 2)).unsqueeze(1)
        candidates = candidate_post_embeddings.sum(dim=1).unsqueeze(0)
        return users + candidates


def test_resolves_stage7_directory_and_direct_bundle(tmp_path):
    stage7, bundle = _write_stage7_artifact(tmp_path)

    from_directory = resolve_stage7_artifact(stage7)
    from_bundle = resolve_stage7_artifact(bundle)

    assert from_directory.bundle_path == bundle.resolve()
    assert from_bundle.root == stage7.resolve()
    assert from_bundle.embedding_dim == 2
    assert from_bundle.split_query_counts["train"] == 2


def test_resolves_bst_two_tower_and_mixed_artifacts(tmp_path):
    stage7, _ = _write_stage7_artifact(tmp_path)
    bst_path = _write_model_artifact(
        tmp_path / "bst",
        model_type="bst-ranker",
        max_history_len=2,
        author_dids=["author0", "author1"],
        stage7_dir=stage7,
    )
    two_tower_path = _write_model_artifact(
        tmp_path / "two-tower",
        model_type="two-tower",
        max_history_len=3,
        author_dids=["author0", "author1"],
        stage7_dir=stage7,
    )

    bst = resolve_model_artifact("baseline", bst_path)
    two_tower = resolve_model_artifact("candidate", two_tower_path)

    assert bst.model_type == "bst-ranker"
    assert set(bst.script_sha256) == {"ranker"}
    assert two_tower.model_type == "two-tower"
    assert set(two_tower.script_sha256) == {"user_tower", "post_tower"}
    dataset = resolve_stage7_artifact(stage7)
    assert validate_comparison_contract(dataset, (bst, two_tower), None) == {
        "baseline": 2,
        "candidate": 3,
    }
    assert validate_comparison_contract(dataset, (bst, two_tower), 2) == {
        "baseline": 2,
        "candidate": 2,
    }
    with pytest.raises(ValueError, match="exceeds"):
        validate_comparison_contract(dataset, (bst, two_tower), 3)


def test_resolves_legacy_bst_from_linked_stage1_summary_and_extra_column_map(
    tmp_path,
):
    author_dids = ["author-a", "author-b"]
    stage1_path, author_map_path = _write_legacy_stage1_artifact(
        tmp_path / "legacy-stage1",
        author_dids=author_dids,
    )
    model_path = _write_legacy_bst_artifact(
        tmp_path / "legacy-stage3",
        author_dids=author_dids,
        stage1_path=stage1_path,
    )

    model = resolve_model_artifact("legacy", model_path)

    assert model.artifact_format == LEGACY_BST_ARTIFACT_FORMAT
    assert model.model_type == "bst-ranker"
    assert model.embedding_model == "fixture-model"
    assert model.embedding_dim == 2
    assert model.max_history_len == 2
    assert model.author_map_path == author_map_path.resolve()
    assert model.author_map_allow_extra_columns is True
    assert model.author_map_stats["author_count"] == len(author_dids)
    assert model.model_config["constructor_args"] == {
        "post_embedding_dim": 2,
        "author_table_num_rows": len(author_dids) + 2,
    }
    assert model.to_dict()["author_map"]["allow_extra_columns"] is True


def test_resolves_legacy_bst_with_explicit_author_map_when_stage1_is_missing(
    tmp_path,
):
    author_dids = ["author-a", "author-b"]
    missing_stage1_path = tmp_path / "missing-stage1"
    model_path = _write_legacy_bst_artifact(
        tmp_path / "legacy-stage3",
        author_dids=author_dids,
        stage1_path=missing_stage1_path,
    )
    author_map_override = _author_map(
        tmp_path / "legacy-author-override.parquet",
        author_dids,
    )

    with pytest.raises(FileNotFoundError, match="explicit legacy author-map override"):
        resolve_model_artifact("legacy", model_path)

    model = resolve_model_artifact(
        "legacy",
        model_path,
        author_map_override=author_map_override,
    )

    assert model.author_map_path == author_map_override.resolve()
    assert model.embedding_model == "fixture-model"


def test_model_resolution_rejects_author_map_override_for_canonical_artifact(
    tmp_path,
):
    stage7, _ = _write_stage7_artifact(tmp_path)
    model_path = _write_model_artifact(
        tmp_path / "canonical-bst",
        model_type="bst-ranker",
        max_history_len=2,
        author_dids=["author0"],
        stage7_dir=stage7,
    )

    with pytest.raises(ValueError, match="only for legacy"):
        resolve_model_artifact(
            "canonical",
            model_path,
            author_map_override=model_path / "ranker_author_idx.parquet",
        )


def test_model_resolution_rejects_legacy_bst_with_extra_script_argument(tmp_path):
    author_dids = ["author-a"]
    stage1_path, _ = _write_legacy_stage1_artifact(
        tmp_path / "legacy-stage1",
        author_dids=author_dids,
    )
    model_path = _write_legacy_bst_artifact(
        tmp_path / "legacy-stage3",
        author_dids=author_dids,
        stage1_path=stage1_path,
        script_module=_LegacyBSTWithExtraArgument(),
    )

    with pytest.raises(ValueError, match="standard eight-argument"):
        resolve_model_artifact("experimental", model_path)


def test_mixed_legacy_bst_and_canonical_two_tower_compare_on_stage7(tmp_path):
    stage7_path, _ = _write_stage7_artifact(tmp_path)
    dataset = resolve_stage7_artifact(stage7_path)
    author_dids = [f"author{index}" for index in range(dataset.embedding_count)]
    stage1_path, _ = _write_legacy_stage1_artifact(
        tmp_path / "legacy-stage1",
        author_dids=author_dids,
    )
    legacy = resolve_model_artifact(
        "legacy",
        _write_legacy_bst_artifact(
            tmp_path / "legacy-stage3",
            author_dids=author_dids,
            stage1_path=stage1_path,
        ),
    )
    two_tower = resolve_model_artifact(
        "canonical",
        _write_model_artifact(
            tmp_path / "two-tower",
            model_type="two-tower",
            max_history_len=3,
            author_dids=author_dids,
            stage7_dir=stage7_path,
        ),
    )

    assert validate_comparison_contract(dataset, (legacy, two_tower), None) == {
        "legacy": 2,
        "canonical": 3,
    }
    temporary_dir = tmp_path / "mixed-mappings"
    result = run_model_comparison(
        dataset=dataset,
        models=(legacy, two_tower),
        settings=ComparisonSettings(
            splits=("train",),
            batch_size=1,
            metrics_top_ks=(2,),
            bst_candidate_chunk_size=2,
            device="cpu",
            num_dataloader_workers=0,
            dataloader_pin_memory=False,
            dataloader_prefetch_factor=1,
            random_seed=7,
            max_classification_metric_pairs=10,
            max_history_len=None,
            disable_progress=True,
        ),
        temporary_dir=temporary_dir,
        logger=logging.getLogger("mixed-legacy-comparison-test"),
    )

    assert set(result.metrics_by_model) == {"legacy", "canonical"}
    assert result.mapping_coverage_by_model["legacy"]["model_known_post_count"] == (
        dataset.embedding_count
    )
    assert not list(temporary_dir.iterdir())


@pytest.mark.parametrize("broken_file", ["manifest.json", "model_config.json"])
def test_model_resolution_rejects_malformed_json(tmp_path, broken_file):
    stage7, _ = _write_stage7_artifact(tmp_path)
    model_path = _write_model_artifact(
        tmp_path / "bst",
        model_type="bst-ranker",
        max_history_len=2,
        author_dids=["author0"],
        stage7_dir=stage7,
    )
    (model_path / broken_file).write_text("not-json")

    with pytest.raises(ValueError):
        resolve_model_artifact("bad", model_path)


def test_model_resolution_rejects_incomplete_manifest_and_mismatched_stage7_bundle(
    tmp_path,
):
    stage7, _ = _write_stage7_artifact(tmp_path)
    model_path = _write_model_artifact(
        tmp_path / "bst",
        model_type="bst-ranker",
        max_history_len=2,
        author_dids=["author0"],
        stage7_dir=stage7,
    )
    manifest = json.loads((model_path / "manifest.json").read_text())
    manifest["status"] = None
    (model_path / "manifest.json").write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="not complete"):
        resolve_model_artifact("bad", model_path)

    manifest.pop("status")
    (model_path / "manifest.json").write_text(json.dumps(manifest) + "\n")
    training_config = json.loads((model_path / "training_config.json").read_text())
    training_config["stage7_bundle"] = str(tmp_path / "unrelated" / "bundle")
    (model_path / "training_config.json").write_text(
        json.dumps(training_config) + "\n"
    )
    with pytest.raises(ValueError, match="invalid Stage 7 bundle"):
        resolve_model_artifact("bad", model_path)


def test_model_resolution_requires_canonical_loader_index_version(tmp_path):
    stage7, _ = _write_stage7_artifact(tmp_path)
    model_path = _write_model_artifact(
        tmp_path / "bst",
        model_type="bst-ranker",
        max_history_len=2,
        author_dids=["author0"],
        stage7_dir=stage7,
    )
    training_config = json.loads((model_path / "training_config.json").read_text())
    training_config.pop("loader_index_format_version")
    (model_path / "training_config.json").write_text(
        json.dumps(training_config) + "\n"
    )

    with pytest.raises(ValueError, match="loader-index version"):
        resolve_model_artifact("bad", model_path)


def test_model_resolution_rejects_malformed_script_and_author_map(tmp_path):
    stage7, _ = _write_stage7_artifact(tmp_path)
    model_path = _write_model_artifact(
        tmp_path / "bst",
        model_type="bst-ranker",
        max_history_len=2,
        author_dids=["author0"],
        stage7_dir=stage7,
    )
    (model_path / "checkpoints" / "ranker.pt").write_text("not-torchscript")
    with pytest.raises(ValueError, match="TorchScript"):
        resolve_model_artifact("bad", model_path)

    model_path = _write_model_artifact(
        tmp_path / "bst-map",
        model_type="bst-ranker",
        max_history_len=2,
        author_dids=["author0"],
        stage7_dir=stage7,
    )
    pl.DataFrame({"wrong": [1]}).write_parquet(
        model_path / "ranker_author_idx.parquet"
    )
    with pytest.raises(ValueError, match="schema"):
        resolve_model_artifact("bad", model_path)


def test_author_remap_cross_dataset_unk_and_complete_cleanup(tmp_path):
    bundle = tmp_path / "hydrated_training_data_mapping"
    posts = bundle / "posts"
    posts.mkdir(parents=True)
    pl.DataFrame({
        "emb_idx": pl.Series([2, 0], dtype=pl.UInt32),
        "author_did": ["known-b", "missing"],
    }).write_parquet(posts / "part-00000.parquet")
    pl.DataFrame({
        "emb_idx": pl.Series([1], dtype=pl.UInt32),
        "author_did": ["known-a"],
    }).write_parquet(posts / "part-00001.parquet")
    model_map = _author_map(tmp_path / "authors.parquet", ["known-a", "known-b"])

    output_path = tmp_path / "mapped.npy"
    coverage = build_author_index_override(
        stage7_bundle_path=bundle,
        model_author_map_path=model_map,
        author_table_num_rows=4,
        embedding_count=3,
        output_path=output_path,
    )

    mapped = np.load(output_path, mmap_mode="r")
    assert mapped.dtype.str == np.dtype("<u4").str
    assert mapped.tolist() == [1, 2, 3]
    assert coverage["model_known_post_count"] == 2
    assert coverage["model_unknown_post_count"] == 1
    del mapped

    with temporary_author_index_override(
        stage7_bundle_path=bundle,
        model_author_map_path=model_map,
        author_table_num_rows=4,
        embedding_count=3,
        temporary_dir=tmp_path / "temporary",
    ) as temporary:
        temporary_path = temporary.path
        assert temporary_path.exists()
    assert not temporary_path.exists()


@pytest.mark.parametrize(
    ("emb_indices", "message"),
    [([0, 0, 1], "duplicate"), ([0, 2], "exactly one row")],
)
def test_author_remap_rejects_duplicate_or_missing_coverage(
    tmp_path,
    emb_indices,
    message,
):
    bundle = tmp_path / "bundle"
    posts = bundle / "posts"
    posts.mkdir(parents=True)
    pl.DataFrame({
        "emb_idx": pl.Series(emb_indices, dtype=pl.UInt32),
        "author_did": ["a"] * len(emb_indices),
    }).write_parquet(posts / "part.parquet")
    model_map = _author_map(tmp_path / "authors.parquet", ["a"])

    with pytest.raises(ValueError, match=message):
        build_author_index_override(
            stage7_bundle_path=bundle,
            model_author_map_path=model_map,
            author_table_num_rows=3,
            embedding_count=3,
            output_path=tmp_path / "mapped.npy",
        )
    assert not (tmp_path / "mapped.npy").exists()


class _ScriptedBST(torch.nn.Module):
    @torch.jit.export
    def score_candidate_matrix(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        candidate_post_author_idx: torch.Tensor,
        history_prior_cumulative_likes: Optional[torch.Tensor] = None,
        candidate_prior_cumulative_likes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del history_mask, history_time_deltas_hours, history_author_indices
        del candidate_post_author_idx, history_prior_cumulative_likes
        del candidate_prior_cumulative_likes
        users = history_embeddings.sum(dim=(1, 2)).unsqueeze(1)
        candidates = candidate_post_embeddings.sum(dim=1).unsqueeze(0)
        return users + candidates


class _ScriptedUserTower(torch.nn.Module):
    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_author_indices: torch.Tensor,
    ) -> torch.Tensor:
        del history_mask, history_author_indices
        return history_embeddings.sum(dim=1)


class _ScriptedPostTower(torch.nn.Module):
    def forward(
        self,
        post_embeddings: torch.Tensor,
        post_author_indices: torch.Tensor,
    ) -> torch.Tensor:
        del post_author_indices
        return post_embeddings


def _score_batch() -> dict[str, torch.Tensor]:
    return {
        "history_embeddings": torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
        "history_mask": torch.tensor([[True], [True]]),
        "history_time_deltas_hours": torch.zeros(2, 1),
        "history_author_indices": torch.tensor([[2], [3]]),
        "history_prior_cumulative_likes": torch.zeros(2, 1),
        "candidate_post_embeddings": torch.tensor([
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 1.0],
            [2.0, 5.0],
            [4.0, 6.0],
        ]),
        "candidate_post_author_idx": torch.tensor([2, 3, 2, 3, 2]),
        "candidate_prior_cumulative_likes": torch.zeros(5),
        "label_matrix": torch.zeros(2, 5),
    }


def test_bst_chunking_matches_full_torchscript_scores(tmp_path):
    path = tmp_path / "ranker.pt"
    scripted = torch.jit.script(_ScriptedBST())
    scripted.save(str(path))
    batch = _score_batch()
    scorer = BSTTorchScriptScorer(path, candidate_chunk_size=2)
    scorer.prepare_for_eval("cpu")

    actual = scorer.score_batch(batch, "cpu").scores
    expected = scripted.score_candidate_matrix(
        batch["history_embeddings"],
        batch["history_mask"],
        batch["history_time_deltas_hours"],
        batch["candidate_post_embeddings"],
        batch["history_author_indices"],
        batch["candidate_post_author_idx"],
        batch["history_prior_cumulative_likes"],
        batch["candidate_prior_cumulative_likes"],
    )
    torch.testing.assert_close(actual, expected)


def test_two_tower_scorer_matches_temperature_scaled_dot_product(tmp_path):
    user_path = tmp_path / "user.pt"
    post_path = tmp_path / "post.pt"
    user = torch.jit.script(_ScriptedUserTower())
    post = torch.jit.script(_ScriptedPostTower())
    user.save(str(user_path))
    post.save(str(post_path))
    batch = _score_batch()
    scorer = TwoTowerTorchScriptScorer(
        user_path,
        post_path,
        similarity_temperature=0.5,
        output_embedding_dim=2,
    )
    scorer.prepare_for_eval("cpu")

    actual = scorer.score_batch(batch, "cpu").scores
    expected = user(
        batch["history_embeddings"],
        batch["history_mask"],
        batch["history_author_indices"],
    ) @ post(
        batch["candidate_post_embeddings"],
        batch["candidate_post_author_idx"],
    ).T / 0.5
    torch.testing.assert_close(actual, expected)


class _FixedScorer:
    def __init__(self, scores: torch.Tensor):
        self.scores = scores

    def prepare_for_eval(self, device: str) -> None:
        pass

    def score_batch(self, batch, device: str) -> MatrixBatchScores:
        return MatrixBatchScores(self.scores)


def test_legacy_metrics_are_complete_deterministic_and_have_no_recall():
    batch = {
        "label_matrix": torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32),
        "history_mask": torch.tensor([[False], [True]]),
    }
    scores = torch.tensor([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]])

    first = evaluate_matrix_scorer(
        _FixedScorer(scores),
        [batch],
        "cpu",
        [2],
        max_classification_metric_pairs=3,
    )["metrics"]
    second = evaluate_matrix_scorer(
        _FixedScorer(scores),
        [batch],
        "cpu",
        [2],
        max_classification_metric_pairs=3,
    )["metrics"]

    for metric in (
        "dcg@2",
        "ndcg@2",
        "mean_average_precision",
        "zero_history_dcg@2",
        "zero_history_ndcg@2",
        "zero_history_mean_average_precision",
        "zero_history_rank_metric_user_count",
        "auc_roc",
        "classification_average_precision",
        "classification_metric_sampled_pair_count",
        "classification_metric_sampled",
    ):
        assert metric in first
    assert first == second
    assert not any("recall" in metric.lower() for metric in first)

    all_negative = evaluate_matrix_scorer(
        _FixedScorer(torch.tensor([[1.0, 2.0, 3.0]])),
        [{
            "label_matrix": torch.zeros(1, 3),
            "history_mask": torch.zeros(1, 1, dtype=torch.bool),
        }],
        "cpu",
        [2],
        max_classification_metric_pairs=3,
    )["metrics"]
    assert all_negative["rank_metric_user_count"] == 0
    assert all_negative["mean_average_precision"] == 0.0
    assert all_negative["auc_roc"] is None
    assert all_negative["classification_average_precision"] is None


def _model_stub(name: str, model_type: str, root: Path) -> SimpleNamespace:
    return SimpleNamespace(name=name, model_type=model_type, root=root)


def test_round_output_floats_preserves_non_float_result_values():
    rounded = round_output_floats({
        "metric": 0.1234567,
        "numpy_metric": np.float32(0.7654321),
        "negative_zero": -0.000001,
        "count": 10,
        "sampled": True,
        "missing": None,
        "nested": [0.3333333],
    })

    assert rounded == {
        "metric": 0.12346,
        "numpy_metric": 0.76543,
        "negative_zero": 0.0,
        "count": 10,
        "sampled": True,
        "missing": None,
        "nested": [0.33333],
    }


def test_metrics_and_model_b_minus_a_delta_csvs(tmp_path):
    metrics = {
        "a": {"val": {
            "ndcg@30": 0.25123456,
            "zero_history_ndcg@30": 0.100006,
            "classification_metric_pair_count": 10,
            "recall@30": 0.9,
            "auc_roc": None,
        }},
        "b": {"val": {
            "ndcg@30": 0.40123456,
            "zero_history_ndcg@30": 0.050004,
            "classification_metric_pair_count": 10,
            "recall@30": 0.8,
            "auc_roc": None,
        }},
    }
    rows = build_metric_deltas(
        model_a_name="a",
        model_b_name="b",
        metrics_by_model=metrics,
    )
    by_metric = {row["metric"]: row for row in rows}
    assert by_metric["ndcg@30"]["delta_model_b_minus_model_a"] == pytest.approx(0.15)
    assert by_metric["zero_history_ndcg@30"]["delta_model_b_minus_model_a"] == pytest.approx(-0.050002)
    assert by_metric["auc_roc"]["delta_model_b_minus_model_a"] is None
    assert "classification_metric_pair_count" not in by_metric
    assert "recall@30" not in by_metric

    models = (
        _model_stub("a", "bst-ranker", tmp_path / "a"),
        _model_stub("b", "two-tower", tmp_path / "b"),
    )
    write_metrics_csv(tmp_path / "metrics.csv", models=models, metrics_by_model=metrics)
    write_metric_deltas_csv(
        tmp_path / "deltas.csv",
        model_a_name="a",
        model_b_name="b",
        metrics_by_model=metrics,
    )
    metric_csv = list(csv.DictReader((tmp_path / "metrics.csv").open()))
    assert len(metric_csv) == 10
    metric_csv_by_key = {
        (row["model_name"], row["metric"]): row["value"]
        for row in metric_csv
    }
    assert metric_csv_by_key[("a", "ndcg@30")] == "0.25123"
    assert metric_csv_by_key[("a", "zero_history_ndcg@30")] == "0.10001"
    assert metric_csv_by_key[("a", "classification_metric_pair_count")] == "10"
    assert metric_csv_by_key[("a", "auc_roc")] == ""
    delta_csv = list(csv.DictReader((tmp_path / "deltas.csv").open()))
    assert {row["metric"] for row in delta_csv} == {
        "auc_roc",
        "ndcg@30",
        "zero_history_ndcg@30",
    }
    delta_csv_by_metric = {row["metric"]: row for row in delta_csv}
    assert delta_csv_by_metric["ndcg@30"] == {
        "model_a_name": "a",
        "model_b_name": "b",
        "split": "val",
        "metric": "ndcg@30",
        "model_a_value": "0.25123",
        "model_b_value": "0.40123",
        "delta_model_b_minus_model_a": "0.15000",
    }
    assert delta_csv_by_metric["zero_history_ndcg@30"][
        "delta_model_b_minus_model_a"
    ] == "-0.05001"


@pytest.mark.parametrize("num_workers", [0, 2])
def test_comparison_keeps_shared_batches_and_override_alive(
    tmp_path,
    monkeypatch,
    num_workers,
):
    stage7_path, _ = _write_stage7_artifact(tmp_path)
    dataset = resolve_stage7_artifact(stage7_path)
    authors = [f"author{index}" for index in range(dataset.embedding_count)]
    models = (
        resolve_model_artifact(
            "a",
            _write_model_artifact(
                tmp_path / "bst",
                model_type="bst-ranker",
                max_history_len=2,
                author_dids=authors,
                stage7_dir=stage7_path,
            ),
        ),
        resolve_model_artifact(
            "b",
            _write_model_artifact(
                tmp_path / "tt",
                model_type="two-tower",
                max_history_len=3,
                author_dids=authors,
                stage7_dir=stage7_path,
            ),
        ),
    )
    observed: list[list[tuple[torch.Tensor, torch.Tensor]]] = []

    class _NoopScorer:
        def close(self):
            pass

    monkeypatch.setattr(
        "engagement_prediction.evaluation.comparison.create_model_scorer",
        lambda *args, **kwargs: _NoopScorer(),
    )

    def capture(_scorer, loader, *_args, **_kwargs):
        batches = []
        for batch in loader:
            batches.append((
                batch["candidate_post_embeddings"].clone(),
                batch["label_matrix"].clone(),
            ))
        observed.append(batches)
        return {
            "metrics": {
                "rank_metric_user_count": sum(
                    batch_labels.size(0) for _, batch_labels in batches
                ),
                "classification_metric_pair_count": sum(
                    batch_labels.numel() for _, batch_labels in batches
                ),
                "classification_metric_sampled_pair_count": 1,
                "ndcg@30": 0.5,
            },
            "ranking_rows": [],
        }

    monkeypatch.setattr(
        "engagement_prediction.evaluation.comparison.evaluate_matrix_scorer",
        capture,
    )
    temporary_dir = tmp_path / "mappings"
    result = run_model_comparison(
        dataset=dataset,
        models=models,
        settings=ComparisonSettings(
            splits=("train", "holdout_seen_users"),
            batch_size=1,
            metrics_top_ks=(30,),
            bst_candidate_chunk_size=2,
            device="cpu",
            num_dataloader_workers=num_workers,
            dataloader_pin_memory=False,
            dataloader_prefetch_factor=1,
            random_seed=7,
            max_classification_metric_pairs=10,
            max_history_len=None,
            disable_progress=True,
        ),
        temporary_dir=temporary_dir,
        logger=logging.getLogger("comparison-test"),
    )

    assert result.history_lengths == {"a": 2, "b": 3}
    assert result.skipped_splits == ("holdout_seen_users",)
    assert len(observed) == 2
    assert len(observed[0]) == len(observed[1])
    for first, second in zip(observed[0], observed[1]):
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])
    assert not list(temporary_dir.iterdir())


def test_comparison_fails_when_all_requested_splits_are_empty(tmp_path):
    stage7_path, _ = _write_stage7_artifact(tmp_path)
    dataset = resolve_stage7_artifact(stage7_path)
    dummy_model = SimpleNamespace(
        name="a",
        max_history_len=2,
        embedding_model=dataset.embedding_model,
        embedding_dim=dataset.embedding_dim,
    )
    other_model = SimpleNamespace(
        name="b",
        max_history_len=2,
        embedding_model=dataset.embedding_model,
        embedding_dim=dataset.embedding_dim,
    )

    with pytest.raises(ValueError, match="All requested"):
        run_model_comparison(
            dataset=dataset,
            models=(dummy_model, other_model),
            settings=ComparisonSettings(
                splits=("holdout_seen_users",),
                batch_size=1,
                metrics_top_ks=(30,),
                bst_candidate_chunk_size=2,
                device="cpu",
                num_dataloader_workers=0,
                dataloader_pin_memory=False,
                dataloader_prefetch_factor=1,
                random_seed=7,
                max_classification_metric_pairs=10,
                max_history_len=None,
                disable_progress=True,
            ),
            temporary_dir=tmp_path / "mappings",
            logger=logging.getLogger("comparison-empty-test"),
        )
