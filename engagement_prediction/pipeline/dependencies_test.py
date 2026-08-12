import json
from pathlib import Path

import pytest

from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline.dependencies import (
    get_stage_folder_to_keys,
    get_stage_input_folders,
    resolve_stage_dependencies_for_run,
    validate_explicit_prior_pin_consistency,
    validate_legacy_training_inputs,
)


def _make_stage_output(
    artifacts_dir: Path,
    stage_folder: str,
    stage_run_id: str,
    *,
    inputs=None,
) -> Path:
    out_dir = artifacts_dir / stage_folder / stage_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage_folder": stage_folder,
        "stage_run_id": stage_run_id,
        "inputs": inputs or {},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest) + "\n")
    return out_dir


def test_get_stage_folder_to_keys_is_derived_from_registry():
    assert get_stage_folder_to_keys() == {
        "01_query_selection": ("query_selection",),
        "02_user_history": ("user_history",),
        "03_train": ("train_mlp", "train_two_tower", "train_bst_ranker"),
        "04_evaluate": ("evaluate",),
    }


def test_get_stage_input_folders_is_derived_from_stage_order():
    assert get_stage_input_folders() == {
        "01_query_selection": [],
        "02_user_history": ["01_query_selection"],
        "03_train": [],
        "04_evaluate": [],
    }


def test_resolve_stage_dependencies_for_user_history_selects_latest_query_artifact(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    _make_stage_output(artifacts_dir, "01_query_selection", "20260101_000000_old")
    query_new = _make_stage_output(artifacts_dir, "01_query_selection", "20260105_000000_new")

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", use_latest=True)

    resolved = resolve_stage_dependencies_for_run(
        ctx=ctx,
        consumer_stage_folder="02_user_history",
    )

    assert resolved == {"01_query_selection": query_new.resolve()}


def test_validate_explicit_prior_pin_consistency_rejects_misaligned_query_history(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    query_old = _make_stage_output(artifacts_dir, "01_query_selection", "20260101_000000_old")
    query_new = _make_stage_output(artifacts_dir, "01_query_selection", "20260104_000000_new")
    user_history = _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260105_000000_history",
        inputs={"01_query_selection": str(query_new)},
    )

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", use_latest=True)
    ctx.prior_outputs["01_query_selection"] = query_old
    ctx.prior_outputs["02_user_history"] = user_history

    with pytest.raises(ValueError, match="Explicit prior pins are inconsistent"):
        validate_explicit_prior_pin_consistency(ctx)


def test_validate_legacy_training_inputs_accepts_aligned_explicit_artifacts(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    get_data = _make_stage_output(artifacts_dir, "01_get_data", "20260101_000000_get")
    user_history = _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260102_000000_history",
        inputs={"01_get_data": str(get_data)},
    )
    (user_history / "history_posts_test.parquet").touch()

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", use_latest=True)
    ctx.prior_outputs["01_get_data"] = get_data
    ctx.prior_outputs["02_user_history"] = user_history

    assert validate_legacy_training_inputs(ctx) == {
        "01_get_data": get_data.resolve(),
        "02_user_history": user_history.resolve(),
    }


def test_validate_legacy_training_inputs_rejects_new_or_misaligned_history(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    get_data = _make_stage_output(artifacts_dir, "01_get_data", "20260101_000000_get")
    other_get_data = _make_stage_output(artifacts_dir, "01_get_data", "20260102_000000_other")
    user_history = _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260103_000000_history",
        inputs={"01_get_data": str(other_get_data)},
    )
    (user_history / "history_posts_test.parquet").touch()

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", use_latest=True)
    ctx.prior_outputs["01_get_data"] = get_data
    ctx.prior_outputs["02_user_history"] = user_history
    with pytest.raises(ValueError, match="was built from"):
        validate_legacy_training_inputs(ctx)

    query_selection = _make_stage_output(
        artifacts_dir,
        "01_query_selection",
        "20260104_000000_query",
    )
    new_history = _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260105_000000_newhistory",
        inputs={"01_query_selection": str(query_selection)},
    )
    ctx.prior_outputs["02_user_history"] = new_history
    with pytest.raises(ValueError, match="New query-history artifacts"):
        validate_legacy_training_inputs(ctx)
