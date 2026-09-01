"""Tests for canonical BST TorchScript export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from engagement_prediction.models.bst_ranker import BSTRanker
from engagement_prediction.training import bst_export
from engagement_prediction.training.bst_export import (
    export_bst_ranker_checkpoint,
    export_post_liker_serving_artifacts,
    validate_bst_ranker_export,
)


def _model_config(
    *,
    use_popularity_feature: bool,
    use_post_liker_feature: bool = False,
) -> dict:
    return {
        "model_type": "bst-ranker",
        "embedding_model": "fixture",
        "max_history_len": 3,
        "author_pad_idx": 0,
        "author_unk_idx": 1,
        "constructor_args": {
            "post_embedding_dim": 4,
            "author_table_num_rows": 8,
            "author_embedding_dim": 3,
            "content_projection_dim": 6,
            "author_projection_dim": 4,
            "model_dim": 5,
            "time_embedding_dim": 3,
            "num_attention_heads": 2,
            "num_transformer_layers": 1,
            "transformer_ff_dim": 16,
            "dropout_rate": 0.0,
            "author_unknown_dropout_rate": 0.0,
            "norm_first": False,
            "time_delta_bucket_boundaries_hours": [1.0, 6.0, 24.0],
            "prediction_hidden_dims": [8],
            "use_popularity_feature": use_popularity_feature,
            "popularity_projection_dim": 2,
            "popularity_log_mean": 1.25 if use_popularity_feature else 0.0,
            "popularity_log_std": 2.5 if use_popularity_feature else 1.0,
            "use_post_liker_feature": use_post_liker_feature,
            "post_liker_user_table_num_rows": 4 if use_post_liker_feature else 2,
            "post_liker_user_embedding_dim": 3,
            "post_liker_projection_dim": 2,
            "post_liker_pooling_tau_hours": 24.0,
            "post_liker_user_unknown_dropout_rate": 0.0,
        },
    }


def _popularity_stats(*, use_popularity_feature: bool) -> dict:
    return {
        "enabled": use_popularity_feature,
        "log_mean": 1.25 if use_popularity_feature else 0.0,
        "log_std": 2.5 if use_popularity_feature else 1.0,
        "history_observation_count": 11 if use_popularity_feature else 0,
        "candidate_observation_count": 7 if use_popularity_feature else 0,
        "total_observation_count": 18 if use_popularity_feature else 0,
    }


def _write_checkpoint(
    path: Path,
    *,
    use_popularity_feature: bool,
    model_config: dict | None = None,
    state_dict: dict | None = None,
    use_post_liker_feature: bool = False,
) -> tuple[dict, dict]:
    model_config = model_config or _model_config(
        use_popularity_feature=use_popularity_feature,
        use_post_liker_feature=use_post_liker_feature,
    )
    popularity_stats = _popularity_stats(
        use_popularity_feature=use_popularity_feature
    )
    torch.manual_seed(123)
    model = BSTRanker(**model_config["constructor_args"])
    torch.save(
        {
            "epoch": 3,
            "best_epoch": 3,
            "model_state_dict": model.state_dict() if state_dict is None else state_dict,
            "metadata": {
                "model_config": model_config,
                "popularity_stats": popularity_stats,
            },
        },
        path,
    )
    return model_config, popularity_stats


@pytest.mark.parametrize("use_popularity_feature", [False, True])
def test_export_scripts_reloads_and_validates_all_serving_paths(
    tmp_path,
    use_popularity_feature,
):
    checkpoint_path = tmp_path / "bst_ranker_best.pth"
    output_path = tmp_path / "ranker.pt"
    model_config, popularity_stats = _write_checkpoint(
        checkpoint_path,
        use_popularity_feature=use_popularity_feature,
    )

    export = export_bst_ranker_checkpoint(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        expected_model_config=model_config,
        expected_popularity_stats=popularity_stats,
    )

    assert export["path"] == str(output_path)
    assert export["best_epoch"] == 3
    assert export["size_bytes"] == output_path.stat().st_size
    assert len(export["sha256"]) == 64
    assert export["parity"] == {
        "case_count": 2,
        "all_exact": True,
        "cases": [
            {
                "case": "normal_and_all_masked",
                "shape": [2, 3],
                "exact_match": True,
                "finite": True,
            },
            {
                "case": "zero_length_history",
                "shape": [2, 3],
                "exact_match": True,
                "finite": True,
            },
        ],
    }
    assert not (tmp_path / "ranker.pt.partial").exists()
    loaded = torch.jit.load(str(output_path)).eval()
    assert callable(getattr(loaded, "score_candidate_matrix", None))

    final_validation = validate_bst_ranker_export(
        checkpoint_path=checkpoint_path,
        scripted_model_path=output_path,
        expected_model_config=model_config,
        expected_popularity_stats=popularity_stats,
    )
    assert final_validation == {
        key: value for key, value in export.items() if key != "path"
    }


def test_export_rejects_checkpoint_model_config_mismatch(tmp_path):
    checkpoint_path = tmp_path / "bst_ranker_best.pth"
    output_path = tmp_path / "ranker.pt"
    model_config, popularity_stats = _write_checkpoint(
        checkpoint_path,
        use_popularity_feature=False,
    )
    expected_model_config = {
        **model_config,
        "embedding_model": "different-model",
    }

    with pytest.raises(ValueError, match="model_config does not match"):
        export_bst_ranker_checkpoint(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            expected_model_config=expected_model_config,
            expected_popularity_stats=popularity_stats,
        )

    assert not output_path.exists()
    assert not (tmp_path / "ranker.pt.partial").exists()


def test_export_rejects_checkpoint_popularity_mismatch(tmp_path):
    checkpoint_path = tmp_path / "bst_ranker_best.pth"
    model_config, popularity_stats = _write_checkpoint(
        checkpoint_path,
        use_popularity_feature=True,
    )

    with pytest.raises(ValueError, match="popularity_stats do not match"):
        export_bst_ranker_checkpoint(
            checkpoint_path=checkpoint_path,
            output_path=tmp_path / "ranker.pt",
            expected_model_config=model_config,
            expected_popularity_stats={**popularity_stats, "log_mean": 99.0},
        )


def test_export_rejects_corrupt_checkpoint_state_dict(tmp_path):
    checkpoint_path = tmp_path / "bst_ranker_best.pth"
    model_config, popularity_stats = _write_checkpoint(
        checkpoint_path,
        use_popularity_feature=False,
        state_dict={"not_a_model_weight": torch.ones(1)},
    )

    with pytest.raises(ValueError, match="cannot reconstruct"):
        export_bst_ranker_checkpoint(
            checkpoint_path=checkpoint_path,
            output_path=tmp_path / "ranker.pt",
            expected_model_config=model_config,
            expected_popularity_stats=popularity_stats,
        )


def test_export_does_not_replace_existing_model_when_validation_fails(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = tmp_path / "bst_ranker_best.pth"
    output_path = tmp_path / "ranker.pt"
    output_path.write_bytes(b"existing validated model")
    model_config, popularity_stats = _write_checkpoint(
        checkpoint_path,
        use_popularity_feature=False,
    )

    def fail_validation(**kwargs):
        raise RuntimeError("injected parity failure")

    monkeypatch.setattr(bst_export, "validate_bst_ranker_export", fail_validation)
    with pytest.raises(RuntimeError, match="injected parity failure"):
        export_bst_ranker_checkpoint(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            expected_model_config=model_config,
            expected_popularity_stats=popularity_stats,
        )

    assert output_path.read_bytes() == b"existing validated model"
    assert (tmp_path / "ranker.pt.partial").is_file()


def test_validator_detects_checkpoint_weights_changed_after_export(tmp_path):
    checkpoint_path = tmp_path / "bst_ranker_best.pth"
    output_path = tmp_path / "ranker.pt"
    model_config, popularity_stats = _write_checkpoint(
        checkpoint_path,
        use_popularity_feature=False,
    )
    export_bst_ranker_checkpoint(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        expected_model_config=model_config,
        expected_popularity_stats=popularity_stats,
    )

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    first_weight = next(iter(checkpoint["model_state_dict"]))
    checkpoint["model_state_dict"][first_weight] = (
        checkpoint["model_state_dict"][first_weight] + 1.0
    )
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="not exactly equal"):
        validate_bst_ranker_export(
            checkpoint_path=checkpoint_path,
            scripted_model_path=output_path,
            expected_model_config=model_config,
            expected_popularity_stats=popularity_stats,
        )


def test_feature_enabled_export_validates_events_vectors_lookup_and_companions(
    tmp_path,
):
    checkpoint_path = tmp_path / "bst_ranker_best.pth"
    output_path = tmp_path / "ranker.pt"
    model_config, popularity_stats = _write_checkpoint(
        checkpoint_path,
        use_popularity_feature=True,
        use_post_liker_feature=True,
    )

    export = export_bst_ranker_checkpoint(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        expected_model_config=model_config,
        expected_popularity_stats=popularity_stats,
    )

    assert export["parity"]["case_count"] == 4
    assert export["parity"]["all_exact"] is True
    assert export["parity"]["post_liker_lookup"] == {
        "row_count": 3,
        "exact_match": True,
        "finite": True,
    }
    assert all(
        case["event_and_prepooled_exact_match"]
        for case in export["parity"]["cases"]
    )

    vocabulary_path = tmp_path / "post_liker_users"
    vocabulary_path.mkdir()
    pl.DataFrame({
        "liker_did": ["did:plc:a", "did:plc:z"],
        "liker_idx": pl.Series([2, 3], dtype=pl.UInt32),
        "training_event_count": pl.Series([9, 3], dtype=pl.UInt64),
    }).write_parquet(vocabulary_path / "part-00000.parquet")
    user_map_path = tmp_path / "ranker_liker_user_idx.parquet"
    embeddings_path = tmp_path / "ranker_liker_user_embeddings.npy"
    state_config_path = tmp_path / "post_liker_state_config.json"

    companions = export_post_liker_serving_artifacts(
        checkpoint_path=checkpoint_path,
        scripted_model_path=output_path,
        expected_model_config=model_config,
        expected_popularity_stats=popularity_stats,
        vocabulary_path=vocabulary_path,
        user_map_output_path=user_map_path,
        embeddings_output_path=embeddings_path,
        state_config_output_path=state_config_path,
        max_replay_events_per_post=128,
    )

    assert companions["best_epoch"] == 3
    assert companions["embeddings_shape"] == [4, 3]
    assert companions["checkpoint_script_lookup_exact_match"] is True
    assert pl.read_parquet(user_map_path).to_dict(as_series=False) == {
        "liker_did": ["did:plc:a", "did:plc:z"],
        "liker_idx": [2, 3],
    }
    table = np.load(embeddings_path, allow_pickle=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_table = checkpoint["model_state_dict"][
        "post_liker_user_pooler.user_embedding.weight"
    ].numpy()
    assert np.array_equal(table, expected_table)
    state_config = json.loads(state_config_path.read_text())
    assert state_config["ranker_contract_version"] == 2
    assert state_config["post_liker_feature_enabled"] is True
    assert state_config["post_liker_user_pad_idx"] == 0
    assert state_config["post_liker_user_unk_idx"] == 1
    assert state_config["post_liker_pooling_tau_hours"] == 24.0
    assert state_config["max_post_liker_replay_events_per_post"] == 128
    assert state_config["incremental_state"]["stored_fields"] == [
        "pooled_embedding_mean",
        "decayed_weight",
        "reference_timestamp",
        "liker_embedding_model_version",
    ]
