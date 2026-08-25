from pathlib import Path

import pytest

import cli


def test_resolve_prior_spec_resolves_stage_run_id(tmp_path):
    output_root = Path(tmp_path) / "out"
    artifacts_dir = output_root / "artifacts"
    stage_folder = "02_user_history"
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
    stage_folder = "02_user_history"
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


def test_resolve_prior_post_selection_stage_run_id(tmp_path):
    output_root = Path(tmp_path) / "out"
    artifacts_dir = output_root / "artifacts"
    target = artifacts_dir / "03_post_selection" / "20260103_000000_posts"
    target.mkdir(parents=True)

    resolved = cli._resolve_prior_spec(
        target.name,
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="03_post_selection",
    )

    assert resolved == target.resolve()


def test_resolve_prior_negative_selection_stage_run_id(tmp_path):
    output_root = Path(tmp_path) / "out"
    artifacts_dir = output_root / "artifacts"
    target = artifacts_dir / "04_negative_selection" / "20260104_000000_negatives"
    target.mkdir(parents=True)

    resolved = cli._resolve_prior_spec(
        target.name,
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="04_negative_selection",
    )

    assert resolved == target.resolve()


def test_resolve_prior_post_liker_history_stage_run_id(tmp_path):
    output_root = Path(tmp_path) / "out"
    artifacts_dir = output_root / "artifacts"
    target = artifacts_dir / "05_post_liker_history" / "20260105_000000_likers"
    target.mkdir(parents=True)

    resolved = cli._resolve_prior_spec(
        target.name,
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="05_post_liker_history",
    )

    assert resolved == target.resolve()


def test_resolve_prior_author_statistics_stage_run_id(tmp_path):
    output_root = Path(tmp_path) / "out"
    artifacts_dir = output_root / "artifacts"
    target = artifacts_dir / "06_author_statistics" / "20260106_000000_authors"
    target.mkdir(parents=True)

    resolved = cli._resolve_prior_spec(
        target.name,
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="06_author_statistics",
    )

    assert resolved == target.resolve()


def test_resolve_prior_dataset_hydration_stage_run_id(tmp_path):
    output_root = Path(tmp_path) / "out"
    artifacts_dir = output_root / "artifacts"
    target = artifacts_dir / "07_dataset_hydration" / "20260107_000000_hydrated"
    target.mkdir(parents=True)

    resolved = cli._resolve_prior_spec(
        target.name,
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="07_dataset_hydration",
    )

    assert resolved == target.resolve()
