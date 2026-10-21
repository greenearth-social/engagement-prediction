"""Resumed date windows must follow the exact upstream artifacts they reuse."""

import json
import os
import subprocess
from datetime import datetime, timezone

import pytest

import cli
from engagement_prediction.experiment_tracking.base import NoOpExperimentTracker


def _args(output_root, *argv):
    return cli._merge_args_with_config(cli.build_parser().parse_args([
        "--output-dir", str(output_root), *argv,
    ]))


def _write_stage(output_root, folder, run_id, inputs):
    stage_dir = output_root / "artifacts" / folder / run_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "manifest.json").write_text(json.dumps({
        "stage_folder": folder,
        "status": "complete",
        "inputs": {key: str(path) for key, path in inputs.items()},
    }))
    return stage_dir


def _source(output_root, run_id, posts_end):
    source = _write_stage(output_root, "00_source_metadata", run_id, {})
    (source / "summary.json").write_text(json.dumps({
        "posts_start": "2026-09-09T00:00:00+00:00",
        "posts_end": posts_end,
    }))
    return source


def _chain(output_root, run_id="20261016_000001_old", posts_end="2026-10-16T00:00:00+00:00"):
    chain = {"00_source_metadata": _source(output_root, run_id, posts_end)}
    for folder in ("01_query_selection", "02_user_history"):
        chain[folder] = _write_stage(output_root, folder, run_id, chain)
    return chain


@pytest.mark.parametrize("start,stop,expected_folders", [
    ("query_selection", "query_selection", ("00_source_metadata",)),
    ("user_history", "post_selection", ("00_source_metadata", "01_query_selection")),
    ("post_selection", "post_selection", ("00_source_metadata", "01_query_selection", "02_user_history")),
])
def test_resume_inherits_source_anchor_and_pins_selected_lineage(tmp_path, start, stop, expected_folders):
    chain = _chain(tmp_path)
    raw = _args(tmp_path, "--start-from", start, "--stop-after", stop)
    before = set(tmp_path.rglob("*"))

    frozen = cli._freeze_posts_end(raw)

    assert frozen is not raw
    assert raw.posts_end is None
    assert frozen.posts_end == "2026-10-16T00:00:00+00:00"
    for folder in expected_folders:
        assert getattr(frozen, f"prior_{folder}") == str(chain[folder])
        assert getattr(raw, f"prior_{folder}") is None
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("pin_style", ["latest", "run_id", "relative_path", "absolute_path"])
def test_post_selection_uses_its_own_stage_zero_ancestor(tmp_path, pin_style):
    chain = _chain(tmp_path)
    # A more recently built source artifact belongs to a different data window.
    _source(tmp_path, "20261020_000001_other", "2026-10-20T00:00:00+00:00")
    stage2 = chain["02_user_history"]
    argv = ["--start-from", "post_selection", "--stop-after", "post_selection"]
    if pin_style != "latest":
        spec = {
            "run_id": stage2.name,
            "relative_path": str(stage2.relative_to(tmp_path)),
            "absolute_path": str(stage2),
        }[pin_style]
        argv += ["--prior-02-user-history", spec]

    resolved = cli._resolve_date_args(_args(tmp_path, *argv))

    assert resolved.posts_end == "2026-10-16T00:00:00+00:00"
    assert resolved.posts_start == "2026-09-09T00:00:00+00:00"
    for folder, path in chain.items():
        assert getattr(resolved, f"prior_{folder}") == str(path)


def test_resume_retains_configured_offsets_and_their_validation(tmp_path):
    _chain(tmp_path)
    raw = _args(tmp_path, "--start-from", "query_selection", "--stop-after", "query_selection",
                "--posts-start-days-before-end", "30", "--val-start-days-before-end", "6")

    resolved = cli._resolve_date_args(raw)

    assert resolved.posts_end == "2026-10-16T00:00:00+00:00"
    assert resolved.posts_start == "2026-09-16T00:00:00+00:00"
    assert resolved.train_start == "2026-09-23T00:00:00+00:00"
    assert resolved.val_start == "2026-10-10T00:00:00+00:00"
    assert resolved.holdout_start == "2026-10-13T00:00:00+00:00"
    raw.val_start_days_before_end = 40
    with pytest.raises(ValueError, match="posts_start <= train_start"):
        cli._resolve_date_args(raw)


@pytest.mark.parametrize("anchor_source", ["config", "cli"])
def test_explicit_anchor_does_not_require_artifact_discovery(tmp_path, anchor_source):
    config = tmp_path / "inputs.json"
    config.write_text(json.dumps({"posts_end": "2026-10-19"}))
    argv = ["--start-from", "post_selection", "--stop-after", "post_selection", "--config", str(config)]
    expected = "2026-10-19T00:00:00+00:00"
    if anchor_source == "cli":
        argv += ["--posts-end", "2026-10-17"]
        expected = "2026-10-17T00:00:00+00:00"
    output_root = tmp_path / "no-artifacts"

    resolved = cli._resolve_date_args(_args(output_root, *argv))

    assert resolved.posts_end == expected
    assert resolved.prior_00_source_metadata is None
    assert not output_root.exists()


def test_fresh_source_run_uses_today_even_when_old_artifacts_exist(tmp_path, monkeypatch):
    _chain(tmp_path)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz == timezone.utc
            return cls(2026, 10, 21, 23, 59, tzinfo=timezone.utc)

    monkeypatch.setattr(cli, "datetime", FrozenDatetime)

    frozen = cli._freeze_posts_end(_args(tmp_path, "--stop-after", "source_metadata"))

    assert frozen.posts_end == "2026-10-21T00:00:00+00:00"
    assert frozen.prior_00_source_metadata is None


@pytest.mark.parametrize("stage", ["user_history", "negative_selection", "dataset_hydration", "train"])
def test_artifact_only_run_does_not_discover_an_unused_anchor(tmp_path, stage):
    resolved = cli._resolve_date_args(_args(tmp_path, "--start-from", stage, "--stop-after", stage))

    assert resolved.posts_end is None
    assert resolved.posts_start is None
    assert resolved.prior_00_source_metadata is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("bad_summary", [None, "{broken json", "{}", '{"posts_end": "invalid"}'])
def test_missing_or_invalid_source_anchor_fails_instead_of_using_today(tmp_path, bad_summary):
    chain = _chain(tmp_path)
    summary = chain["00_source_metadata"] / "summary.json"
    if bad_summary is None:
        summary.unlink()
    else:
        summary.write_text(bad_summary)

    with pytest.raises((ValueError, OSError)):
        cli._freeze_posts_end(_args(tmp_path, "--start-from", "post_selection", "--stop-after", "post_selection"))


def test_conflicting_prior_pins_fail_before_dates_are_inherited(tmp_path):
    chain = _chain(tmp_path)
    other = _source(tmp_path, "20261020_000001_other", "2026-10-20T00:00:00+00:00")

    with pytest.raises(ValueError, match="inconsistent|Misaligned|does not match"):
        cli._freeze_posts_end(_args(
            tmp_path, "--start-from", "post_selection", "--stop-after", "post_selection",
            "--prior-00-source-metadata", str(other),
            "--prior-02-user-history", str(chain["02_user_history"]),
        ))


def test_missing_stage_zero_lineage_does_not_pick_unrelated_latest_source(tmp_path):
    chain = _chain(tmp_path)
    (chain["01_query_selection"] / "manifest.json").write_text(json.dumps({"inputs": {}}))

    with pytest.raises(ValueError, match="00_source_metadata|Stage 00"):
        cli._freeze_posts_end(_args(tmp_path, "--start-from", "post_selection", "--stop-after", "post_selection"))


@pytest.mark.parametrize("background", [False, True])
def test_resume_dry_run_resolves_dates_without_writes_tracker_or_child(tmp_path, monkeypatch, capsys, background):
    chain = _chain(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    before_paths = set(tmp_path.rglob("*"))
    monkeypatch.setattr(cli, "build_experiment_tracker", lambda *a, **k: pytest.fail("dry-run created tracker"))
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: pytest.fail("dry-run launched child"))
    monkeypatch.setattr(cli.reg, "run_stage", lambda *a, **k: pytest.fail("dry-run executed stage"))
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("dry-run prompted for input"))
    argv = ["--output-dir", str(tmp_path), "--start-from", "post_selection", "--stop-after", "post_selection", "--dry-run", "--pick-prior"]
    if background:
        argv.append("--background")

    assert cli.main(argv) == 0

    preview = json.loads(capsys.readouterr().out)
    assert preview["posts_end"] == "2026-10-16T00:00:00+00:00"
    assert preview["posts_start"] == "2026-09-09T00:00:00+00:00"
    assert preview["prior_02_user_history"] == str(chain["02_user_history"])
    assert set(tmp_path.rglob("*")) == before_paths
    assert {path: path.read_bytes() for path in before} == before


def test_background_config_freezes_anchor_and_inputs_before_new_artifacts_arrive(tmp_path, monkeypatch, capsys):
    chain = _chain(tmp_path)
    monkeypatch.delenv("ENGAGEMENT_PIPELINE_RUN_ID", raising=False)
    monkeypatch.setenv("ENGAGEMENT_RUN_TIMESTAMP", "20261021_000001")
    monkeypatch.setattr(cli, "build_experiment_tracker", lambda *a, **k: pytest.fail("parent created tracker"))
    launches = []

    def launch(command, **kwargs):
        launches.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="12345\n", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", launch)
    assert cli.main([
        "--output-dir", str(tmp_path), "--start-from", "post_selection",
        "--stop-after", "post_selection", "--background",
    ]) == 0
    assert len(launches) == 1
    paths = [path for path in (tmp_path / "runs").glob("*/run-all.resolved-config.json") if not path.parent.is_symlink()]
    assert len(paths) == 1
    saved = json.loads(paths[0].read_text())
    assert saved["posts_end"] == "2026-10-16T00:00:00+00:00"
    for folder, path in chain.items():
        assert saved[f"prior_{folder}"] == str(path)
    assert not (set(saved) & (set(cli.DATE_FIELDS) - {"posts_end"}))

    _chain(tmp_path, "20261021_000001_new", "2026-10-21T00:00:00+00:00")
    capsys.readouterr()
    assert cli.main(["--config", str(paths[0]), "--dry-run"]) == 0
    child_preview = json.loads(capsys.readouterr().out)
    assert child_preview["posts_end"] == saved["posts_end"]
    for folder in chain:
        assert child_preview[f"prior_{folder}"] == saved[f"prior_{folder}"]


def test_foreground_stage_receives_the_same_lineage_and_anchor_saved_for_replay(tmp_path, monkeypatch):
    chain = _chain(tmp_path)
    monkeypatch.delenv("ENGAGEMENT_PIPELINE_RUN_ID", raising=False)
    monkeypatch.setattr(cli, "build_experiment_tracker", lambda *a, **k: NoOpExperimentTracker())
    received = {}

    def run_stage(stage, context, args):
        assert stage == "post_selection"
        assert context.prior_outputs == chain
        assert args.posts_end == "2026-10-16T00:00:00+00:00"
        assert args.posts_start == "2026-09-09T00:00:00+00:00"
        received["config"] = context.run_dir / "run-all.resolved-config.json"
        return {}

    monkeypatch.setattr(cli.reg, "run_stage", run_stage)
    assert cli.main([
        "--output-dir", str(tmp_path), "--start-from", "post_selection",
        "--stop-after", "post_selection", "--no-disk-checks",
    ]) == 0

    saved = json.loads(received["config"].read_text())
    assert saved["posts_end"] == "2026-10-16T00:00:00+00:00"
    for folder, path in chain.items():
        assert saved[f"prior_{folder}"] == str(path)


def test_remote_output_and_prior_overrides_are_applied_before_anchor_inference(tmp_path, monkeypatch):
    initial_root = tmp_path / "initial"
    remote_root = tmp_path / "remote"
    old_chain = _chain(remote_root)
    _chain(remote_root, "20261021_000001_new", "2026-10-21T00:00:00+00:00")
    monkeypatch.delenv("ENGAGEMENT_PIPELINE_RUN_ID", raising=False)

    class RemoteTracker(NoOpExperimentTracker):
        def connect_args(self, args, name=None):
            assert args.posts_end is None
            args.output_dir = str(remote_root)
            args.prior_02_user_history = str(old_chain["02_user_history"])
            return args

    monkeypatch.setattr(cli, "build_experiment_tracker", lambda *a, **k: RemoteTracker())
    observed = []

    def run_stage(stage, context, args):
        observed.append((args.posts_end, context.prior_outputs.copy()))
        return {}

    monkeypatch.setattr(cli.reg, "run_stage", run_stage)
    assert cli.main([
        "--output-dir", str(initial_root), "--start-from", "post_selection",
        "--stop-after", "post_selection", "--no-disk-checks",
    ]) == 0

    assert observed == [("2026-10-16T00:00:00+00:00", old_chain)]
    assert not initial_root.exists()


def test_interactively_selected_prior_determines_the_inherited_anchor(tmp_path, monkeypatch):
    old_chain = _chain(tmp_path)
    new_chain = _chain(tmp_path, "20261021_000001_new", "2026-10-21T00:00:00+00:00")
    # The interactive chooser orders by mtime; make its displayed order explicit.
    os.utime(old_chain["02_user_history"], (1000, 1000))
    os.utime(new_chain["02_user_history"], (2000, 2000))
    monkeypatch.delenv("ENGAGEMENT_PIPELINE_RUN_ID", raising=False)
    monkeypatch.setattr(cli, "build_experiment_tracker", lambda *a, **k: NoOpExperimentTracker())
    prompts = []

    def select_older(prompt):
        prompts.append(prompt)
        return "1"

    monkeypatch.setattr("builtins.input", select_older)
    observed = []

    def run_stage(stage, context, args):
        observed.append((args.posts_end, context.prior_outputs.copy()))
        return {}

    monkeypatch.setattr(cli.reg, "run_stage", run_stage)
    assert cli.main([
        "--output-dir", str(tmp_path), "--start-from", "post_selection",
        "--stop-after", "post_selection", "--no-disk-checks", "--pick-prior",
    ]) == 0

    assert len(prompts) == 1
    assert observed == [("2026-10-16T00:00:00+00:00", old_chain)]


@pytest.mark.parametrize("failure", ["missing_artifacts", "missing_summary"])
def test_foreground_anchor_resolution_failure_marks_task_failed_and_closes(tmp_path, monkeypatch, failure):
    if failure == "missing_summary":
        chain = _chain(tmp_path)
        (chain["00_source_metadata"] / "summary.json").unlink()
    before = set(tmp_path.rglob("*"))
    events = []

    class TrackingFailure(NoOpExperimentTracker):
        def mark_failed(self, reason):
            assert reason
            events.append("failed")

        def close(self):
            events.append("closed")

    monkeypatch.setattr(cli, "build_experiment_tracker", lambda *a, **k: TrackingFailure())
    monkeypatch.setattr(cli.reg, "run_stage", lambda *a, **k: pytest.fail("invalid lineage executed stage"))

    with pytest.raises((ValueError, OSError)):
        cli.main([
            "--output-dir", str(tmp_path), "--start-from", "post_selection",
            "--stop-after", "post_selection",
        ])

    assert events == ["failed", "closed"]
    assert set(tmp_path.rglob("*")) == before
