import json
from pathlib import Path

import pytest

import cli
from utils.pipeline.core import Context


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


def test_resolve_prior_spec_resolves_stage_run_id(tmp_path):
    output_root = Path(tmp_path) / "out"
    artifacts_dir = output_root / "artifacts"
    stage_folder = "02_target_posts"
    stage_run_id = "20260102_000000_abcd1234"
    target = artifacts_dir / stage_folder / stage_run_id
    target.mkdir(parents=True)

    resolved = cli._resolve_prior_spec(
        stage_run_id,
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder=stage_folder,
    )

    assert resolved == target.resolve()


def test_resolve_prior_spec_resolves_relative_path_against_output_root(tmp_path):
    output_root = Path(tmp_path) / "out"
    artifacts_dir = output_root / "artifacts"
    stage_folder = "03_user_history"
    p = output_root / "some" / "custom_prior"
    p.mkdir(parents=True)

    resolved = cli._resolve_prior_spec(
        "some/custom_prior",
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder=stage_folder,
    )

    assert resolved == p.resolve()


def test_resolve_prior_spec_raises_if_missing(tmp_path):
    output_root = Path(tmp_path) / "out"
    artifacts_dir = output_root / "artifacts"
    with pytest.raises(FileNotFoundError):
        cli._resolve_prior_spec(
            "does_not_exist",
            output_root=output_root,
            artifacts_dir=artifacts_dir,
            stage_folder="01_get_data",
        )


def test_resolve_stage_dependencies_for_train_follows_latest_downstream_lineage(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    get_data_old = _make_stage_output(artifacts_dir, "01_get_data", "20260101_000000_oldget")
    get_data_new = _make_stage_output(artifacts_dir, "01_get_data", "20260105_000000_newget")
    target_posts_old = _make_stage_output(
        artifacts_dir,
        "02_target_posts",
        "20260102_000000_oldtarget",
        inputs={"01_get_data": str(get_data_old)},
    )
    _make_stage_output(
        artifacts_dir,
        "02_target_posts",
        "20260106_000000_newtarget",
        inputs={"01_get_data": str(get_data_new)},
    )
    user_history_old = _make_stage_output(
        artifacts_dir,
        "03_user_history",
        "20260103_000000_oldhistory",
        inputs={
            "01_get_data": str(get_data_old),
            "02_target_posts": str(target_posts_old),
        },
    )

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", use_latest=True)

    resolved = cli._resolve_stage_dependencies_for_run(
        ctx=ctx,
        consumer_stage_folder="04_train",
    )

    assert resolved == {
        "01_get_data": get_data_old.resolve(),
        "02_target_posts": target_posts_old.resolve(),
        "03_user_history": user_history_old.resolve(),
    }


def test_resolve_stage_dependencies_raises_on_misaligned_explicit_pins(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    get_data_old = _make_stage_output(artifacts_dir, "01_get_data", "20260101_000000_oldget")
    get_data_new = _make_stage_output(artifacts_dir, "01_get_data", "20260104_000000_newget")
    target_posts_old = _make_stage_output(
        artifacts_dir,
        "02_target_posts",
        "20260102_000000_oldtarget",
        inputs={"01_get_data": str(get_data_old)},
    )
    target_posts_new = _make_stage_output(
        artifacts_dir,
        "02_target_posts",
        "20260105_000000_newtarget",
        inputs={"01_get_data": str(get_data_new)},
    )
    user_history_new = _make_stage_output(
        artifacts_dir,
        "03_user_history",
        "20260106_000000_newhistory",
        inputs={
            "01_get_data": str(get_data_new),
            "02_target_posts": str(target_posts_new),
        },
    )

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", use_latest=True)
    ctx.prior_outputs["02_target_posts"] = target_posts_old
    ctx.prior_outputs["03_user_history"] = user_history_new

    with pytest.raises(ValueError, match="Misaligned inputs for stage '04_train'"):
        cli._resolve_stage_dependencies_for_run(
            ctx=ctx,
            consumer_stage_folder="04_train",
        )


def test_resolve_stage_dependencies_for_evaluate_infers_inputs_from_train_manifest(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    get_data_old = _make_stage_output(artifacts_dir, "01_get_data", "20260101_000000_oldget")
    _make_stage_output(artifacts_dir, "01_get_data", "20260109_000000_newget")
    target_posts_old = _make_stage_output(
        artifacts_dir,
        "02_target_posts",
        "20260102_000000_oldtarget",
        inputs={"01_get_data": str(get_data_old)},
    )
    user_history_old = _make_stage_output(
        artifacts_dir,
        "03_user_history",
        "20260103_000000_oldhistory",
        inputs={
            "01_get_data": str(get_data_old),
            "02_target_posts": str(target_posts_old),
        },
    )
    train_old = _make_stage_output(
        artifacts_dir,
        "04_train",
        "20260104_000000_oldtrain",
        inputs={
            "01_get_data": str(get_data_old),
            "02_target_posts": str(target_posts_old),
            "03_user_history": str(user_history_old),
        },
    )

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", use_latest=True)

    resolved = cli._resolve_stage_dependencies_for_run(
        ctx=ctx,
        consumer_stage_folder="05_evaluate",
    )

    assert resolved == {
        "01_get_data": get_data_old.resolve(),
        "02_target_posts": target_posts_old.resolve(),
        "03_user_history": user_history_old.resolve(),
        "04_train": train_old.resolve(),
    }


def test_validate_explicit_prior_pin_consistency_raises_on_misaligned_stage1_stage2_pins(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    get_data_old = _make_stage_output(artifacts_dir, "01_get_data", "20260101_000000_oldget")
    get_data_new = _make_stage_output(artifacts_dir, "01_get_data", "20260104_000000_newget")
    target_posts_new = _make_stage_output(
        artifacts_dir,
        "02_target_posts",
        "20260105_000000_newtarget",
        inputs={"01_get_data": str(get_data_new)},
    )

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", use_latest=True)
    ctx.prior_outputs["01_get_data"] = get_data_old
    ctx.prior_outputs["02_target_posts"] = target_posts_new

    with pytest.raises(ValueError, match="Explicit prior pins are inconsistent"):
        cli._validate_explicit_prior_pin_consistency(ctx)
