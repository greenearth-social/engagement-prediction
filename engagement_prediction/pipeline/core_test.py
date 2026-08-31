import os
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pytest

import engagement_prediction.pipeline.core as pipeline_core
from engagement_prediction.pipeline.core import Context, select_prior_output, list_stage_outputs


def test_generate_run_timestamp_uses_los_angeles_time(monkeypatch):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            fixed_utc = datetime(2026, 5, 11, 1, 30, 0, tzinfo=timezone.utc)
            return fixed_utc.astimezone(tz) if tz is not None else fixed_utc.replace(tzinfo=None)

    monkeypatch.setattr(pipeline_core, "datetime", FixedDateTime)

    assert pipeline_core.generate_run_timestamp() == "20260510_183000"


def test_pipeline_core_resolves_repository_root():
    repo_root = Path(__file__).resolve().parents[2]

    assert pipeline_core.ROOT == repo_root
    assert pipeline_core.DEFAULT_ARTIFACTS_DIR == repo_root / "artifacts"
    assert pipeline_core.DEFAULT_RUNS_DIR == repo_root / "runs"


def test_select_prior_output_prefers_latest(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    older = artifacts_dir / "02_user_history" / "20240101_000000"
    newer = artifacts_dir / "02_user_history" / "20240102_000000"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "manifest.json").write_text("{}\n")
    (newer / "manifest.json").write_text("{}\n")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    chosen = select_prior_output(artifacts_dir=artifacts_dir, stage_folder="02_user_history")

    assert chosen == newer


def test_select_prior_output_honors_explicit_prior_path(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    explicit = Path(tmp_path) / "custom_prior"
    other = artifacts_dir / "03_train" / "20240101_000000"
    explicit.mkdir(parents=True)
    other.mkdir(parents=True)

    chosen = select_prior_output(artifacts_dir=artifacts_dir, stage_folder="03_train", prior_path=explicit)

    assert chosen == explicit


def test_list_stage_outputs_sorts_by_timestamp_then_mtime(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    stage_folder = "02_user_history"
    base = artifacts_dir / stage_folder
    base.mkdir(parents=True, exist_ok=True)

    # Newest timestamp should win even if its mtime is older.
    older_ts_newer_mtime = base / "20240101_000000_abcd1234"
    newer_ts_older_mtime = base / "20240102_000000_zzzz9999"
    older_ts_newer_mtime.mkdir(parents=True, exist_ok=True)
    newer_ts_older_mtime.mkdir(parents=True, exist_ok=True)
    os.utime(older_ts_newer_mtime, (200, 200))
    os.utime(newer_ts_older_mtime, (100, 100))

    # Same timestamp: tie-break by mtime.
    same_ts_older_mtime = base / "20240103_000000_tag_11111111"
    same_ts_newer_mtime = base / "20240103_000000_tag_22222222"
    same_ts_older_mtime.mkdir(parents=True, exist_ok=True)
    same_ts_newer_mtime.mkdir(parents=True, exist_ok=True)
    for output_dir in (
        older_ts_newer_mtime,
        newer_ts_older_mtime,
        same_ts_older_mtime,
        same_ts_newer_mtime,
    ):
        (output_dir / "manifest.json").write_text("{}\n")
    os.utime(same_ts_older_mtime, (10, 10))
    os.utime(same_ts_newer_mtime, (20, 20))

    outs = list_stage_outputs(artifacts_dir=artifacts_dir, stage_folder=stage_folder)
    assert outs[0] == same_ts_newer_mtime
    assert outs[1] == same_ts_older_mtime
    assert outs[2] == newer_ts_older_mtime
    assert outs[3] == older_ts_newer_mtime


def test_list_stage_outputs_ignores_incomplete_and_malformed_artifacts(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    base = artifacts_dir / "00_source_metadata"
    completed = base / "20240101_000000_complete"
    missing_manifest = base / "20240104_000000_missing"
    partial_status = base / "20240103_000000_partial"
    malformed_manifest = base / "20240102_000000_malformed"
    for output_dir in (
        completed,
        missing_manifest,
        partial_status,
        malformed_manifest,
    ):
        output_dir.mkdir(parents=True)
    (completed / "manifest.json").write_text('{"status": "complete"}\n')
    (partial_status / "manifest.json").write_text('{"status": "partial"}\n')
    (malformed_manifest / "manifest.json").write_text("{not-json\n")

    assert list_stage_outputs(artifacts_dir, "00_source_metadata") == [completed]
    assert select_prior_output(
        artifacts_dir=artifacts_dir,
        stage_folder="00_source_metadata",
    ) == completed


def test_select_prior_output_keeps_explicit_incomplete_path_semantics(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    explicit = artifacts_dir / "00_source_metadata" / "20240101_000000_partial"
    explicit.mkdir(parents=True)

    assert select_prior_output(
        artifacts_dir=artifacts_dir,
        stage_folder="00_source_metadata",
        prior_path=explicit,
    ) == explicit


def test_stage_metadata_json_includes_nulls(tmp_path):
    run_dir = Path(tmp_path) / "runs" / "run1"
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", pipeline_run_id="run1")
    ctx.begin_stage("user_history", "02_user_history")
    out_dir = ctx.new_stage_dir(tag="test")

    args = argparse.Namespace(foo=None, bar="baz")
    ctx.finalize_stage(stage_key="user_history", stage_folder="02_user_history", output_dir=out_dir, args=args, argv=None)

    manifest_path = out_dir / "manifest.json"
    resolved_config_path = out_dir / "resolved_config.json"
    assert manifest_path.exists()
    assert resolved_config_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert "argv" in manifest
    assert manifest["argv"] is None
    assert manifest["status"] == "complete"

    resolved = json.loads(resolved_config_path.read_text())
    assert "foo" in resolved
    assert resolved["foo"] is None


def test_write_partial_stage_manifest_records_active_inputs(tmp_path):
    run_dir = Path(tmp_path) / "runs" / "run1"
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    prior_dir = artifacts_dir / "01_get_data" / "20260101_000000_abcd1234"
    prior_dir.mkdir(parents=True, exist_ok=True)

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", pipeline_run_id="run1")
    ctx.begin_stage("train_two_tower", "03_train")
    ctx.record_prior_input("01_get_data", prior_dir)
    out_dir = ctx.new_stage_dir(tag="test")

    manifest_path = ctx.write_partial_stage_manifest(output_dir=out_dir, argv=["pipeline", "all"])

    assert manifest_path == out_dir / "manifest.partial.json"
    assert not (out_dir / "manifest.json").exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "partial"
    assert manifest["stage_key"] == "train_two_tower"
    assert manifest["stage_folder"] == "03_train"
    assert manifest["argv"] == ["pipeline", "all"]
    assert manifest["inputs"] == {"01_get_data": str(prior_dir.resolve())}


def test_finalize_stage_appends_prior_inputs_to_stage_info(tmp_path):
    run_dir = Path(tmp_path) / "runs" / "run1"
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    prior_dir = artifacts_dir / "01_get_data" / "20260101_000000_abcd1234"
    prior_dir.mkdir(parents=True, exist_ok=True)

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", pipeline_run_id="run1")
    ctx.begin_stage("user_history", "02_user_history")
    ctx.record_prior_input("01_get_data", prior_dir)
    out_dir = ctx.new_stage_dir(tag="test")
    (out_dir / "stage_info.txt").write_text("stage: user_history\n")

    args = argparse.Namespace()
    ctx.finalize_stage(stage_key="user_history", stage_folder="02_user_history", output_dir=out_dir, args=args, argv=None)

    stage_info = (out_dir / "stage_info.txt").read_text()
    assert "prior_inputs: 1" in stage_info
    assert f"prior_input_01_get_data: {prior_dir.resolve()}" in stage_info


@pytest.mark.parametrize("failure_point", ["stage_info", "lineage"])
def test_finalize_stage_does_not_publish_completion_markers_after_metadata_failure(
    tmp_path,
    monkeypatch,
    failure_point,
):
    run_dir = Path(tmp_path) / "runs" / "run1"
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir.mkdir(parents=True)
    ctx = Context(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        runs_dir=Path(tmp_path) / "runs",
        pipeline_run_id="run1",
    )
    ctx.begin_stage("source_metadata", "00_source_metadata")
    out_dir = ctx.new_stage_dir(tag="test")

    method_name = (
        "_append_stage_info_inputs"
        if failure_point == "stage_info"
        else "_update_lineage"
    )

    def fail(*args, **kwargs):
        raise OSError(f"{failure_point} failed")

    monkeypatch.setattr(ctx, method_name, fail)

    with pytest.raises(OSError, match=f"{failure_point} failed"):
        ctx.finalize_stage(
            stage_key="source_metadata",
            stage_folder="00_source_metadata",
            output_dir=out_dir,
            args=argparse.Namespace(),
        )

    assert not (out_dir / "manifest.json").exists()
    assert not (run_dir / "00_source_metadata").exists()


def test_finalize_stage_rolls_back_symlink_when_manifest_publication_fails(
    tmp_path,
    monkeypatch,
):
    run_dir = Path(tmp_path) / "runs" / "run1"
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir.mkdir(parents=True)
    ctx = Context(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        runs_dir=Path(tmp_path) / "runs",
        pipeline_run_id="run1",
    )
    ctx.begin_stage("source_metadata", "00_source_metadata")
    out_dir = ctx.new_stage_dir(tag="test")
    original_replace = Path.replace

    def fail_manifest_replace(path, target):
        if Path(target).name == "manifest.json":
            raise OSError("manifest publication failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="manifest publication failed"):
        ctx.finalize_stage(
            stage_key="source_metadata",
            stage_folder="00_source_metadata",
            output_dir=out_dir,
            args=argparse.Namespace(),
        )

    assert not (out_dir / "manifest.json").exists()
    assert not list(out_dir.glob(".manifest.json.*.partial"))
    assert not (run_dir / "00_source_metadata").exists()


def test_new_stage_dir_rejects_mismatched_stage_folder_when_active(tmp_path):
    run_dir = Path(tmp_path) / "runs" / "run1"
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", pipeline_run_id="run1")
    ctx.begin_stage("user_history", "02_user_history")

    try:
        ctx.new_stage_dir("03_train")
        assert False, "Expected ValueError for mismatched stage folder"
    except ValueError as e:
        assert "mismatch" in str(e).lower()
