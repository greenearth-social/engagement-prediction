"""Pipeline sequencing and failure behavior for the optional production comparison."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli
from engagement_prediction.experiment_tracking import NoOpExperimentTracker


def comparison_args(tmp_path, *extra):
    return cli._merge_args_with_config(cli.build_parser().parse_args([
        "--output-dir", str(tmp_path), "--compare-to-production",
        "--production-gcp-project", "test-project",
        "--production-gcp-region", "test-region",
        "--production-cloud-run-service", "test-service", *extra,
    ]))


@pytest.mark.parametrize(("overrides", "message"), [
    ({"compare_to_production": "true"}, "must be a boolean"),
    ({"model_type": "two-tower"}, "BST rankers only"),
    ({"bst_use_post_liker_feature": True}, "post-liker"),
    ({"stop_after": "dataset_hydration"}, "includes training"),
    ({"holdout_start_days_before_end": None}, "holdout to be enabled"),
    ({"production_gcp_project": None}, "production_gcp_project is required"),
    ({"production_gcp_region": " "}, "production_gcp_region is required"),
    ({"production_cloud_run_service": None}, "production_cloud_run_service is required"),
    ({"production_comparison_candidate_chunk_size": 0}, "positive integer"),
    ({"production_comparison_max_classification_metric_pairs": True}, "positive integer"),
])
def test_invalid_comparison_configuration_fails_before_execution(tmp_path, overrides, message):
    args = comparison_args(tmp_path)
    config = {key: getattr(args, key) for key in cli.DEFAULTS}
    config.update(overrides)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=message):
        cli._merge_args_with_config(cli.build_parser().parse_args(["--config", str(path)]))


def test_comparison_dry_run_does_not_contact_production_or_create_outputs(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "absent"
    args = comparison_args(output_dir, "--dry-run", "--posts-end", "2026-10-20")
    monkeypatch.setattr(cli, "_prepare_production_comparison", lambda *a: pytest.fail("network"))
    monkeypatch.setattr(cli, "build_experiment_tracker", lambda *a, **k: pytest.fail("ClearML"))

    assert cli.cmd_run_all(args) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["compare_to_production"] is True
    assert preview["holdout_start"] == "2026-10-17T00:00:00+00:00"
    assert not output_dir.exists()


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    from engagement_prediction.evaluation import production

    events = []

    class Tracker(NoOpExperimentTracker):
        def close(self):
            events.append("close")

        def mark_failed(self, reason):
            assert "close" not in events
            events.append("failed")

        def log_params(self, params, name=None):
            if name == "Args":
                assert params == {"max_history_posts_per_query": 64}

    context = cli.Context(
        run_dir=tmp_path / "runs" / "run", artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "runs", tracker=Tracker(),
    )
    context.run_dir.mkdir(parents=True)
    candidate_dir = context.artifacts_dir / "08_train_bst_ranker" / "candidate"
    baseline = SimpleNamespace(max_history_len=64, embedding_model="all_MiniLM_L12_v2")

    def snapshot(**kwargs):
        events.append("snapshot")
        assert kwargs["project"] == "test-project"
        assert kwargs["service"] == "test-service"
        assert kwargs["output_dir"] == context.run_dir / "production_comparison" / "production_snapshot"
        return baseline, {"revision": "production-at-start"}

    def validate_dataset(args, ctx, production_model):
        events.append("validate holdout")
        assert production_model is baseline

    def stage(key, ctx, args):
        events.append(key)
        assert args.max_history_posts_per_query == 64
        assert args.max_history_len == 32
        if key == "train_bst_ranker":
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "manifest.json").write_text("{}")
        return {"output_dir": candidate_dir}

    def compare(**kwargs):
        events.append("compare")
        assert kwargs["candidate_dir"] == candidate_dir
        assert (candidate_dir / "manifest.json").exists()
        assert kwargs["production_metadata"] == {"revision": "production-at-start"}
        assert kwargs["production_model"] is baseline

    monkeypatch.setattr(production, "snapshot_production_model", snapshot)
    monkeypatch.setattr(cli, "_validate_production_comparison_dataset", validate_dataset)
    monkeypatch.setattr(cli, "_run_production_comparison", compare)
    monkeypatch.setattr(cli, "pin_lineage_aligned_inputs", lambda *a: None)
    monkeypatch.setattr(cli.reg, "run_stage", stage)
    return SimpleNamespace(context=context, events=events, candidate_dir=candidate_dir, baseline=baseline)


@pytest.mark.parametrize("start", [None, "train"])
def test_pipeline_snapshots_before_data_and_compares_finalized_candidate(tmp_path, pipeline, start):
    args = comparison_args(tmp_path, *(["--start-from", start] if start else []))
    assert cli.cmd__run_all_exec(args, pipeline.context) == 0
    stages = cli._get_stage_order_for_model_type("train_bst_ranker") if not start else ["train_bst_ranker"]
    assert pipeline.events == ["snapshot", *stages[:-1], "validate holdout", stages[-1], "compare", "close"]
    resolved = json.loads((pipeline.context.run_dir / "run-all.resolved-config.json").read_text())
    assert resolved["max_history_posts_per_query"] == 64
    assert resolved["max_history_len"] == 32


@pytest.mark.parametrize(("failure", "phase"), [
    ("snapshot", "production snapshot"),
    ("validate holdout", "holdout validation"),
    ("compare", "evaluation and reporting"),
])
def test_comparison_failures_mark_task_failed_and_preserve_candidate(tmp_path, monkeypatch, pipeline, failure, phase):
    args = comparison_args(tmp_path, "--start-from", "train")

    def fail(*a, **k):
        pipeline.events.append(failure)
        raise ValueError("test comparison failure")

    target = {
        "snapshot": "_prepare_production_comparison",
        "validate holdout": "_validate_production_comparison_dataset",
        "compare": "_run_production_comparison",
    }[failure]
    monkeypatch.setattr(cli, target, fail)
    with pytest.raises(ValueError, match="test comparison failure"):
        cli.cmd__run_all_exec(args, pipeline.context)

    report = json.loads((pipeline.context.run_dir / "production_comparison" / "failure.json").read_text())
    assert report["status"] == "failed"
    assert report["phase"] == phase
    assert pipeline.events[-2:] == ["failed", "close"]
    assert pipeline.candidate_dir.exists() == (failure == "compare")
    assert ("train_bst_ranker" in pipeline.events) == (failure == "compare")


def test_remote_override_revalidates_comparison_before_snapshot(tmp_path, pipeline):
    args = comparison_args(tmp_path)
    args.bst_use_post_liker_feature = True
    with pytest.raises(ValueError, match="post-liker"):
        cli.cmd__run_all_exec(args, pipeline.context)
    assert pipeline.events == ["failed", "close"]


def test_disabled_comparison_keeps_ordinary_pipeline_behavior(tmp_path, monkeypatch):
    args = cli._merge_args_with_config(cli.build_parser().parse_args([
        "--output-dir", str(tmp_path), "--start-from", "train",
    ]))
    context = cli.Context(run_dir=tmp_path / "run", artifacts_dir=tmp_path / "artifacts")
    monkeypatch.setattr(cli, "_prepare_production_comparison", lambda *a: pytest.fail("snapshot"))
    monkeypatch.setattr(cli, "_validate_production_comparison_dataset", lambda *a: pytest.fail("validation"))
    monkeypatch.setattr(cli, "_run_production_comparison", lambda **k: pytest.fail("comparison"))
    monkeypatch.setattr(cli, "pin_lineage_aligned_inputs", lambda *a: None)
    monkeypatch.setattr(cli.reg, "run_stage", lambda *a: None)

    assert cli.cmd__run_all_exec(args, context) == 0
    assert not (context.run_dir / "production_comparison").exists()
