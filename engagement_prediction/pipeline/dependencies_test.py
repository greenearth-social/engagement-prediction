import json
from pathlib import Path

import pytest

from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline.dependencies import (
    get_stage_folder_to_keys,
    get_stage_input_folders,
    pin_lineage_aligned_inputs,
    resolve_stage_dependencies_for_run,
    validate_explicit_prior_pin_consistency,
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
        "00_source_metadata": ("source_metadata",),
        "01_query_selection": ("query_selection",),
        "02_user_history": ("user_history",),
        "03_post_selection": ("post_selection",),
        "04_negative_selection": ("negative_selection",),
        "05_post_liker_history": ("post_liker_history",),
        "06_author_statistics": ("author_statistics",),
        "07_dataset_hydration": ("dataset_hydration",),
        "08_train_bst_ranker": ("train_bst_ranker",),
        "08_train_two_tower": ("train_two_tower",),
    }


def test_get_stage_input_folders_is_derived_from_stage_order():
    assert get_stage_input_folders() == {
        "00_source_metadata": [],
        "01_query_selection": ["00_source_metadata"],
        "02_user_history": ["01_query_selection"],
        "03_post_selection": ["02_user_history"],
        "04_negative_selection": ["03_post_selection"],
        "05_post_liker_history": ["04_negative_selection"],
        "06_author_statistics": ["05_post_liker_history"],
        "07_dataset_hydration": ["06_author_statistics"],
        "08_train_bst_ranker": ["07_dataset_hydration"],
        "08_train_two_tower": ["07_dataset_hydration"],
    }


def test_resolve_stage_dependencies_for_user_history_selects_latest_query_artifact(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    source = _make_stage_output(artifacts_dir, "00_source_metadata", "20260101_000000_source")
    _make_stage_output(
        artifacts_dir,
        "01_query_selection",
        "20260101_000000_old",
        inputs={"00_source_metadata": str(source)},
    )
    query_new = _make_stage_output(
        artifacts_dir,
        "01_query_selection",
        "20260105_000000_new",
        inputs={"00_source_metadata": str(source)},
    )

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", use_latest=True)

    resolved = resolve_stage_dependencies_for_run(
        ctx=ctx,
        consumer_stage_folder="02_user_history",
    )

    assert resolved == {
        "00_source_metadata": source.resolve(),
        "01_query_selection": query_new.resolve(),
    }


def test_resolve_post_selection_from_history_pin_infers_query_ancestor(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    source = _make_stage_output(artifacts_dir, "00_source_metadata", "20260100_000000_source")
    query_selection = _make_stage_output(
        artifacts_dir,
        "01_query_selection",
        "20260101_000000_query",
        inputs={"00_source_metadata": str(source)},
    )
    user_history = _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260102_000000_history",
        inputs={
            "00_source_metadata": str(source),
            "01_query_selection": str(query_selection),
        },
    )
    ctx = Context(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        runs_dir=Path(tmp_path) / "runs",
        use_latest=True,
    )
    ctx.prior_outputs["02_user_history"] = user_history

    resolved = resolve_stage_dependencies_for_run(
        ctx=ctx,
        consumer_stage_folder="03_post_selection",
    )

    assert resolved == {
        "00_source_metadata": source.resolve(),
        "01_query_selection": query_selection.resolve(),
        "02_user_history": user_history.resolve(),
    }


def test_resolve_negative_selection_from_post_pin_infers_all_ancestors(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    source = _make_stage_output(artifacts_dir, "00_source_metadata", "20260100_000000_source")
    query_selection = _make_stage_output(
        artifacts_dir,
        "01_query_selection",
        "20260101_000000_query",
        inputs={"00_source_metadata": str(source)},
    )
    user_history = _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260102_000000_history",
        inputs={
            "00_source_metadata": str(source),
            "01_query_selection": str(query_selection),
        },
    )
    post_selection = _make_stage_output(
        artifacts_dir,
        "03_post_selection",
        "20260103_000000_posts",
        inputs={
            "00_source_metadata": str(source),
            "01_query_selection": str(query_selection),
            "02_user_history": str(user_history),
        },
    )
    ctx = Context(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        runs_dir=Path(tmp_path) / "runs",
        use_latest=True,
    )
    ctx.prior_outputs["03_post_selection"] = post_selection

    resolved = resolve_stage_dependencies_for_run(
        ctx=ctx,
        consumer_stage_folder="04_negative_selection",
    )

    assert resolved == {
        "00_source_metadata": source.resolve(),
        "01_query_selection": query_selection.resolve(),
        "02_user_history": user_history.resolve(),
        "03_post_selection": post_selection.resolve(),
    }


def test_resolve_post_liker_history_from_negative_pin_infers_all_ancestors(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    source = _make_stage_output(artifacts_dir, "00_source_metadata", "20260100_000000_source")
    query_selection = _make_stage_output(
        artifacts_dir,
        "01_query_selection",
        "20260101_000000_query",
        inputs={"00_source_metadata": str(source)},
    )
    user_history = _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260102_000000_history",
        inputs={
            "00_source_metadata": str(source),
            "01_query_selection": str(query_selection),
        },
    )
    post_selection = _make_stage_output(
        artifacts_dir,
        "03_post_selection",
        "20260103_000000_posts",
        inputs={
            "00_source_metadata": str(source),
            "01_query_selection": str(query_selection),
            "02_user_history": str(user_history),
        },
    )
    negative_selection = _make_stage_output(
        artifacts_dir,
        "04_negative_selection",
        "20260104_000000_negatives",
        inputs={
            "00_source_metadata": str(source),
            "01_query_selection": str(query_selection),
            "02_user_history": str(user_history),
            "03_post_selection": str(post_selection),
        },
    )
    ctx = Context(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        runs_dir=Path(tmp_path) / "runs",
        use_latest=True,
    )
    ctx.prior_outputs["04_negative_selection"] = negative_selection

    resolved = resolve_stage_dependencies_for_run(
        ctx=ctx,
        consumer_stage_folder="05_post_liker_history",
    )

    assert resolved == {
        "00_source_metadata": source.resolve(),
        "01_query_selection": query_selection.resolve(),
        "02_user_history": user_history.resolve(),
        "03_post_selection": post_selection.resolve(),
        "04_negative_selection": negative_selection.resolve(),
    }


def test_resolve_author_statistics_from_post_liker_pin_infers_all_ancestors(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    source = _make_stage_output(artifacts_dir, "00_source_metadata", "20260100_000000_source")
    query_selection = _make_stage_output(
        artifacts_dir,
        "01_query_selection",
        "20260101_000000_query",
        inputs={"00_source_metadata": str(source)},
    )
    user_history = _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260102_000000_history",
        inputs={
            "00_source_metadata": str(source),
            "01_query_selection": str(query_selection),
        },
    )
    post_selection = _make_stage_output(
        artifacts_dir,
        "03_post_selection",
        "20260103_000000_posts",
        inputs={
            "00_source_metadata": str(source),
            "01_query_selection": str(query_selection),
            "02_user_history": str(user_history),
        },
    )
    negative_selection = _make_stage_output(
        artifacts_dir,
        "04_negative_selection",
        "20260104_000000_negatives",
        inputs={
            "00_source_metadata": str(source),
            "01_query_selection": str(query_selection),
            "02_user_history": str(user_history),
            "03_post_selection": str(post_selection),
        },
    )
    post_liker_history = _make_stage_output(
        artifacts_dir,
        "05_post_liker_history",
        "20260105_000000_likers",
        inputs={
            "00_source_metadata": str(source),
            "01_query_selection": str(query_selection),
            "02_user_history": str(user_history),
            "03_post_selection": str(post_selection),
            "04_negative_selection": str(negative_selection),
        },
    )
    ctx = Context(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        runs_dir=Path(tmp_path) / "runs",
        use_latest=True,
    )
    ctx.prior_outputs["05_post_liker_history"] = post_liker_history

    resolved = resolve_stage_dependencies_for_run(
        ctx=ctx,
        consumer_stage_folder="06_author_statistics",
    )

    assert resolved == {
        "00_source_metadata": source.resolve(),
        "01_query_selection": query_selection.resolve(),
        "02_user_history": user_history.resolve(),
        "03_post_selection": post_selection.resolve(),
        "04_negative_selection": negative_selection.resolve(),
        "05_post_liker_history": post_liker_history.resolve(),
    }


def test_resolve_dataset_hydration_from_author_pin_infers_all_ancestors(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    source = _make_stage_output(artifacts_dir, "00_source_metadata", "20260100_000000_source")
    query_selection = _make_stage_output(
        artifacts_dir,
        "01_query_selection",
        "20260101_000000_query",
        inputs={"00_source_metadata": str(source)},
    )
    user_history = _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260102_000000_history",
        inputs={
            "00_source_metadata": str(source),
            "01_query_selection": str(query_selection),
        },
    )
    post_selection = _make_stage_output(
        artifacts_dir,
        "03_post_selection",
        "20260103_000000_posts",
        inputs={
            "00_source_metadata": str(source),
            "01_query_selection": str(query_selection),
            "02_user_history": str(user_history),
        },
    )
    negative_selection = _make_stage_output(
        artifacts_dir,
        "04_negative_selection",
        "20260104_000000_negatives",
        inputs={
            "00_source_metadata": str(source),
            "01_query_selection": str(query_selection),
            "02_user_history": str(user_history),
            "03_post_selection": str(post_selection),
        },
    )
    post_liker_history = _make_stage_output(
        artifacts_dir,
        "05_post_liker_history",
        "20260105_000000_likers",
        inputs={
            "00_source_metadata": str(source),
            "01_query_selection": str(query_selection),
            "02_user_history": str(user_history),
            "03_post_selection": str(post_selection),
            "04_negative_selection": str(negative_selection),
        },
    )
    author_statistics = _make_stage_output(
        artifacts_dir,
        "06_author_statistics",
        "20260106_000000_authors",
        inputs={
            "00_source_metadata": str(source),
            "01_query_selection": str(query_selection),
            "02_user_history": str(user_history),
            "03_post_selection": str(post_selection),
            "04_negative_selection": str(negative_selection),
            "05_post_liker_history": str(post_liker_history),
        },
    )
    ctx = Context(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        runs_dir=Path(tmp_path) / "runs",
        use_latest=True,
    )
    ctx.prior_outputs["06_author_statistics"] = author_statistics

    resolved = resolve_stage_dependencies_for_run(
        ctx=ctx,
        consumer_stage_folder="07_dataset_hydration",
    )

    assert resolved == {
        "00_source_metadata": source.resolve(),
        "01_query_selection": query_selection.resolve(),
        "02_user_history": user_history.resolve(),
        "03_post_selection": post_selection.resolve(),
        "04_negative_selection": negative_selection.resolve(),
        "05_post_liker_history": post_liker_history.resolve(),
        "06_author_statistics": author_statistics.resolve(),
    }


@pytest.mark.parametrize(
    "consumer_stage_folder",
    ["08_train_bst_ranker", "08_train_two_tower"],
)
def test_resolve_native_training_from_stage7_pin_infers_all_ancestors(
    tmp_path,
    consumer_stage_folder,
):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    folders = [
        "00_source_metadata",
        "01_query_selection",
        "02_user_history",
        "03_post_selection",
        "04_negative_selection",
        "05_post_liker_history",
        "06_author_statistics",
        "07_dataset_hydration",
    ]
    artifacts = {}
    for index, folder in enumerate(folders):
        artifacts[folder] = _make_stage_output(
            artifacts_dir,
            folder,
            f"2026010{index}_000000_stage{index}",
            inputs={
                ancestor: str(artifacts[ancestor])
                for ancestor in folders[:index]
            },
        )
    ctx = Context(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        runs_dir=Path(tmp_path) / "runs",
        use_latest=True,
    )
    ctx.prior_outputs["07_dataset_hydration"] = artifacts["07_dataset_hydration"]

    resolved = resolve_stage_dependencies_for_run(
        ctx=ctx,
        consumer_stage_folder=consumer_stage_folder,
    )

    assert resolved == {
        folder: artifacts[folder].resolve()
        for folder in folders
    }

    mismatched_author = _make_stage_output(
        artifacts_dir,
        "06_author_statistics",
        "20260109_000000_mismatched",
        inputs={
            ancestor: str(artifacts[ancestor])
            for ancestor in folders[:6]
        },
    )
    ctx.prior_outputs["06_author_statistics"] = mismatched_author
    with pytest.raises(ValueError, match="Explicit prior pins are inconsistent"):
        validate_explicit_prior_pin_consistency(ctx)


def test_pin_lineage_aligned_inputs_allows_native_two_tower_training(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    folders = [
        "00_source_metadata",
        "01_query_selection",
        "02_user_history",
        "03_post_selection",
        "04_negative_selection",
        "05_post_liker_history",
        "06_author_statistics",
        "07_dataset_hydration",
    ]
    artifacts = {}
    for index, folder in enumerate(folders):
        artifacts[folder] = _make_stage_output(
            artifacts_dir,
            folder,
            f"2026010{index}_000000_stage{index}",
            inputs={
                ancestor: str(artifacts[ancestor])
                for ancestor in folders[:index]
            },
        )
    ctx = Context(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        runs_dir=Path(tmp_path) / "runs",
        use_latest=True,
    )
    ctx.prior_outputs["07_dataset_hydration"] = artifacts["07_dataset_hydration"]

    pin_lineage_aligned_inputs(
        ctx,
        "train_two_tower",
        {"train_two_tower": "08_train_two_tower"},
    )

    assert ctx.prior_outputs == {
        folder: artifacts[folder].resolve()
        for folder in folders
    }


def test_validate_explicit_prior_pin_consistency_rejects_misaligned_query_history(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    source = _make_stage_output(artifacts_dir, "00_source_metadata", "20260100_000000_source")
    query_old = _make_stage_output(
        artifacts_dir,
        "01_query_selection",
        "20260101_000000_old",
        inputs={"00_source_metadata": str(source)},
    )
    query_new = _make_stage_output(
        artifacts_dir,
        "01_query_selection",
        "20260104_000000_new",
        inputs={"00_source_metadata": str(source)},
    )
    user_history = _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260105_000000_history",
        inputs={
            "00_source_metadata": str(source),
            "01_query_selection": str(query_new),
        },
    )

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", use_latest=True)
    ctx.prior_outputs["01_query_selection"] = query_old
    ctx.prior_outputs["02_user_history"] = user_history

    with pytest.raises(ValueError, match="Explicit prior pins are inconsistent"):
        validate_explicit_prior_pin_consistency(ctx)
