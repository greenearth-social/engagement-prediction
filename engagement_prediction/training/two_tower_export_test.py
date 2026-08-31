from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engagement_prediction.models.two_tower import TwoTowerModel
from engagement_prediction.training import two_tower_export


def _model_config(output_embedding_dim: int = 4) -> dict:
    constructor_args = {
        "post_embedding_dim": 3,
        "author_table_num_rows": 6,
        "author_embedding_dim": 2,
        "content_projection_dim": 4,
        "author_projection_dim": 2,
        "user_hidden_dim": 5,
        "post_hidden_dim": 6,
        "output_embedding_dim": output_embedding_dim,
        "max_history_len": 3,
        "dropout_rate": 0.0,
        "author_unknown_dropout_rate": 0.0,
        "similarity_temperature": 0.5,
    }
    return {
        "model_type": "two-tower",
        "user_encoder_type": "cross_attention",
        "output_embedding_dim": output_embedding_dim,
        "constructor_args": constructor_args,
    }


def _checkpoint(path: Path, config: dict, *, best_epoch: int = 2) -> Path:
    model = TwoTowerModel(**config["constructor_args"])
    torch.save({
        "epoch": best_epoch,
        "best_epoch": best_epoch,
        "output_embedding_dim": config["output_embedding_dim"],
        "model_state_dict": model.state_dict(),
        "metadata": {"model_config": config},
    }, path)
    return path


@pytest.mark.parametrize("output_embedding_dim", [3, 8])
def test_export_scripts_both_towers_with_exact_parity(
    tmp_path,
    output_embedding_dim,
):
    config = _model_config(output_embedding_dim)
    checkpoint_path = _checkpoint(tmp_path / "two_tower_best.pth", config)
    user_path = tmp_path / "engagement_user_tower.pt"
    post_path = tmp_path / "engagement_post_tower.pt"

    result = two_tower_export.export_two_tower_checkpoint(
        checkpoint_path=checkpoint_path,
        user_tower_path=user_path,
        post_tower_path=post_path,
        expected_model_config=config,
    )

    assert result["best_epoch"] == 2
    assert result["output_embedding_dim"] == output_embedding_dim
    assert result["parity"]["case_count"] == 2
    assert [case["case"] for case in result["parity"]["cases"]] == [
        "ordinary",
        "all_masked",
    ]
    assert result["user_tower"]["size_bytes"] == user_path.stat().st_size
    assert result["post_tower"]["size_bytes"] == post_path.stat().st_size
    assert not list(tmp_path.glob("*.partial"))
    torch.jit.load(str(user_path))
    torch.jit.load(str(post_path))


def test_export_rejects_mismatched_model_configuration(tmp_path):
    config = _model_config()
    checkpoint_path = _checkpoint(tmp_path / "two_tower_best.pth", config)
    expected = _model_config()
    expected["constructor_args"]["similarity_temperature"] = 0.25

    with pytest.raises(ValueError, match="does not match"):
        two_tower_export.export_two_tower_checkpoint(
            checkpoint_path=checkpoint_path,
            user_tower_path=tmp_path / "engagement_user_tower.pt",
            post_tower_path=tmp_path / "engagement_post_tower.pt",
            expected_model_config=expected,
        )


def test_export_failure_does_not_publish_partial_towers(tmp_path, monkeypatch):
    config = _model_config()
    checkpoint_path = _checkpoint(tmp_path / "two_tower_best.pth", config)

    def fail_validation(**_kwargs):
        raise RuntimeError("parity failed")

    monkeypatch.setattr(
        two_tower_export,
        "validate_two_tower_export",
        fail_validation,
    )
    user_path = tmp_path / "engagement_user_tower.pt"
    post_path = tmp_path / "engagement_post_tower.pt"
    with pytest.raises(RuntimeError, match="parity failed"):
        two_tower_export.export_two_tower_checkpoint(
            checkpoint_path=checkpoint_path,
            user_tower_path=user_path,
            post_tower_path=post_path,
            expected_model_config=config,
        )

    assert not user_path.exists()
    assert not post_path.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_export_restores_prior_pair_when_second_publish_rename_fails(
    tmp_path,
    monkeypatch,
):
    config = _model_config()
    checkpoint_path = _checkpoint(tmp_path / "two_tower_best.pth", config)
    user_path = tmp_path / "engagement_user_tower.pt"
    post_path = tmp_path / "engagement_post_tower.pt"
    user_path.write_bytes(b"prior-user")
    post_path.write_bytes(b"prior-post")
    post_partial_path = post_path.with_name(f"{post_path.name}.partial")
    original_replace = Path.replace

    def fail_second_publish(source, target):
        if Path(source) == post_partial_path and Path(target) == post_path:
            raise OSError("post publish failed")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_publish)

    with pytest.raises(OSError, match="post publish failed"):
        two_tower_export.export_two_tower_checkpoint(
            checkpoint_path=checkpoint_path,
            user_tower_path=user_path,
            post_tower_path=post_path,
            expected_model_config=config,
        )

    assert user_path.read_bytes() == b"prior-user"
    assert post_path.read_bytes() == b"prior-post"
    assert not list(tmp_path.glob("*.partial"))
    assert not list(tmp_path.glob("*.previous"))
