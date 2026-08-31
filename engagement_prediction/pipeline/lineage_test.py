import json
from pathlib import Path

import pytest

from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline.lineage import resolve_recorded_stage_lineage


def _write_stage(path: Path, stage_folder: str, inputs: dict[str, Path]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps({
        "stage_key": stage_folder,
        "stage_folder": stage_folder,
        "inputs": {key: str(value.resolve()) for key, value in inputs.items()},
    }) + "\n")
    return path


def _lineage(tmp_path: Path) -> tuple[Path, Path, Path]:
    stage1 = _write_stage(tmp_path / "01_query_selection" / "one", "01_query_selection", {})
    stage2 = _write_stage(
        tmp_path / "02_user_history" / "two",
        "02_user_history",
        {"01_query_selection": stage1},
    )
    stage3 = _write_stage(
        tmp_path / "03_post_selection" / "three",
        "03_post_selection",
        {"01_query_selection": stage1, "02_user_history": stage2},
    )
    return stage1, stage2, stage3


def _context(tmp_path: Path, stage3: Path) -> Context:
    context = Context(
        run_dir=tmp_path / "runs" / "run",
        artifacts_dir=tmp_path,
        runs_dir=tmp_path / "runs",
    )
    context.prior_outputs["03_post_selection"] = stage3
    return context


def test_resolves_and_records_complete_transitive_lineage(tmp_path):
    stage1, stage2, stage3 = _lineage(tmp_path)
    context = _context(tmp_path, stage3)

    resolved = resolve_recorded_stage_lineage(
        context,
        terminal_stage_folder="03_post_selection",
        ancestor_stage_folders=("01_query_selection", "02_user_history"),
    )

    assert resolved == {
        "01_query_selection": stage1.resolve(),
        "02_user_history": stage2.resolve(),
        "03_post_selection": stage3.resolve(),
    }
    assert context.get_active_stage_inputs() == resolved


def test_rejects_explicit_pin_outside_terminal_lineage(tmp_path):
    _, _, stage3 = _lineage(tmp_path)
    context = _context(tmp_path, stage3)
    other_stage1 = _write_stage(
        tmp_path / "01_query_selection" / "other",
        "01_query_selection",
        {},
    )
    context.prior_outputs["01_query_selection"] = other_stage1

    with pytest.raises(ValueError, match="does not match Stage 3 lineage"):
        resolve_recorded_stage_lineage(
            context,
            terminal_stage_folder="03_post_selection",
            ancestor_stage_folders=("01_query_selection", "02_user_history"),
        )


def test_rejects_inconsistent_intermediate_manifest(tmp_path):
    stage1, stage2, stage3 = _lineage(tmp_path)
    other_stage1 = _write_stage(
        tmp_path / "01_query_selection" / "other",
        "01_query_selection",
        {},
    )
    _write_stage(stage2, "02_user_history", {"01_query_selection": other_stage1})

    with pytest.raises(ValueError, match="Stage 3 lineage is invalid"):
        resolve_recorded_stage_lineage(
            _context(tmp_path, stage3),
            terminal_stage_folder="03_post_selection",
            ancestor_stage_folders=("01_query_selection", "02_user_history"),
        )

    assert stage1 != other_stage1


def test_rejects_artifact_without_stage_zero_lineage_with_rerun_guidance(tmp_path):
    stage1 = _write_stage(
        tmp_path / "01_query_selection" / "old",
        "01_query_selection",
        {},
    )
    context = Context(
        run_dir=tmp_path / "runs" / "run",
        artifacts_dir=tmp_path,
        runs_dir=tmp_path / "runs",
    )
    context.prior_outputs["01_query_selection"] = stage1

    with pytest.raises(
        ValueError,
        match="predates the required Stage 00 lineage.*rerun.*source_metadata",
    ):
        resolve_recorded_stage_lineage(
            context,
            terminal_stage_folder="01_query_selection",
            ancestor_stage_folders=("00_source_metadata",),
        )
