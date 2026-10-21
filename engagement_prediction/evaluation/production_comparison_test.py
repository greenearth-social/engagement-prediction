from dataclasses import replace
import json
import logging
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from engagement_prediction.evaluation.artifacts import ModelArtifact, Stage7Artifact
from engagement_prediction.evaluation.comparison import ComparisonResult
from engagement_prediction.evaluation import comparison as comparison_engine
from engagement_prediction.evaluation import production_comparison as comparison
from engagement_prediction.experiment_tracking import NoOpExperimentTracker


def _model(name, root, history_limit, training_config):
    root.mkdir()
    script_path = root / "ranker.pt"
    script_path.write_bytes(b"fixture")
    return ModelArtifact(
        name=name, root=root, artifact_format="canonical_stage8", model_type="bst-ranker",
        manifest={"stage_run_id": root.name}, model_config={}, training_config=training_config,
        script_paths={"ranker": script_path}, script_sha256={"ranker": "fixture-sha256"},
        author_map_path=root / "ranker_author_idx.parquet", author_map_stats={},
        author_map_allow_extra_columns=False, embedding_model="fixture", embedding_dim=2,
        max_history_len=history_limit, author_table_num_rows=4, similarity_temperature=None,
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    stage1 = tmp_path / "stage1"
    stage2 = tmp_path / "stage2"
    stage7 = tmp_path / "stage7"
    for path in (stage1, stage2, stage7):
        path.mkdir()
    (stage1 / "summary.json").write_text(json.dumps({
        "posts_start": "2026-09-09", "posts_end": "2026-10-16",
        "parameters": {
            "train_start": "2026-09-16", "val_start": "2026-10-08",
            "holdout_start": "2026-10-13", "holdout_end": None,
        },
    }))
    (stage2 / "summary.json").write_text(json.dumps({"parameters": {
        "max_history_posts_per_query": 64,
    }}))
    dataset = Stage7Artifact(
        root=stage7, bundle_path=stage7 / "bundle",
        manifest={"stage_run_id": "stage7", "inputs": {
            "01_query_selection": str(stage1), "02_user_history": str(stage2),
        }}, summary={"parameters": {}, "input": {}},
        loader_index_validation={"embedding_count": 100, "format_version": 2, "splits": {
            "holdout_seen_users": {"query_count": 10},
            "holdout_unseen_users": {"query_count": 5},
        }}, embedding_model="fixture", embedding_dim=2,
    )
    candidate = _model("candidate", tmp_path / "candidate", 32, {
        "stage7_dir": str(stage7), "eval_batch_size": 8,
        "random_seed": 42, "bst_additional_batch_negatives": 4,
        "metrics_top_ks": [30], "device": "cpu",
    })
    production = _model("production", tmp_path / "production", 64, {})
    metrics = {
        "production": {split: {"ndcg@30": 0.5} for split in comparison.HOLDOUT_SPLITS},
        "candidate": {
            "holdout_seen_users": {"ndcg@30": 0.5000001},
            "holdout_unseen_users": {"ndcg@30": 0.4},
        },
    }
    result = ComparisonResult(
        metrics_by_model=metrics,
        mapping_coverage_by_model={name: {
            "stage7_post_count": 100, "model_unknown_post_count": 10,
        } for name in metrics},
        split_row_counts={"holdout_seen_users": 10, "holdout_unseen_users": 5},
        skipped_splits=(), history_lengths={"candidate": 32, "production": 64},
    )
    resolve_candidate = Mock(return_value=candidate)
    resolve_dataset = Mock(return_value=dataset)
    evaluate = Mock(return_value=result)
    monkeypatch.setattr(comparison, "resolve_model_artifact", resolve_candidate)
    monkeypatch.setattr(comparison, "resolve_stage7_artifact", resolve_dataset)
    monkeypatch.setattr(comparison_engine, "run_model_comparison", evaluate)
    args = SimpleNamespace(
        metrics_top_ks=[30], production_comparison_candidate_chunk_size=1024,
        production_comparison_max_classification_metric_pairs=2000,
        device="cpu", num_dataloader_workers=0, dataloader_pin_memory=False,
        dataloader_prefetch_factor=2,
        # Evaluation must prefer the saved training values over these current arguments.
        eval_batch_size=99, random_seed=123, bst_additional_batch_negatives=88,
        holdout_start="2000-01-01", holdout_end="2000-01-02",
    )
    tracker = Mock(id="candidate-task")
    tracker.log_file_artifact.return_value = True
    kwargs = {
        "candidate_dir": candidate.root, "production_model": production,
        "production_metadata": {
            "ranker_clearml_model_id": "production-model", "cloud_run": {"revision": "revision-1"},
            "training_window": None,
        },
        "args": args, "output_dir": tmp_path / "comparison", "tracker": tracker,
        "logger": logging.getLogger(__name__),
    }
    return SimpleNamespace(
        kwargs=kwargs, candidate=candidate, production=production, dataset=dataset,
        stage1=stage1, stage2=stage2, result=result,
        resolve_candidate=resolve_candidate, resolve_dataset=resolve_dataset, evaluate=evaluate,
    )


def test_uses_exact_candidate_lineage_and_saved_evaluation_settings(setup):
    setup.kwargs["args"].metrics_top_ks = [10]
    report = comparison.run_production_comparison(**setup.kwargs)

    setup.resolve_candidate.assert_called_once_with("candidate", setup.candidate.root)
    setup.resolve_dataset.assert_called_once_with(Path(setup.candidate.training_config["stage7_dir"]))
    passed = setup.evaluate.call_args.kwargs
    assert passed["models"] == (setup.production, setup.candidate)
    assert passed["dataset"] is setup.dataset
    settings = passed["settings"]
    assert settings.splits == comparison.HOLDOUT_SPLITS
    assert settings.batch_size == 8 and settings.random_seed == 42
    assert settings.additional_batch_negatives == 4
    assert settings.metrics_top_ks == (30,)
    assert settings.max_history_len is None
    assert settings.bst_candidate_chunk_size == 1024 and settings.device == "cpu"
    assert report["dates"]["holdout_start"] == "2026-10-13T00:00:00+00:00"
    assert report["dates"]["holdout_end"] == "2026-10-16T00:00:00+00:00"
    assert report["dates"]["posts_start"] == "2026-09-09T00:00:00+00:00"
    assert report["dates"]["posts_end"] == "2026-10-16T00:00:00+00:00"
    assert report["history_extraction_limit"] == 64
    assert not passed["temporary_dir"].exists()


def test_null_device_uses_training_device_selection_behavior(setup, monkeypatch):
    import torch

    setup.kwargs["args"].device = None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    comparison.run_production_comparison(**setup.kwargs)

    assert setup.evaluate.call_args.kwargs["settings"].device == "cpu"


def test_writes_full_precision_metrics_and_review_with_provenance(setup):
    report = comparison.run_production_comparison(**setup.kwargs)
    output = setup.kwargs["output_dir"]
    saved = json.loads((output / "metrics.json").read_text())

    assert saved["status"] == "complete"
    assert saved["production"]["cloud_run"]["revision"] == "revision-1"
    assert saved["models"][0]["scripts"]["ranker"]["sha256"] == "fixture-sha256"
    assert saved["metrics"]["candidate"]["holdout_seen_users"]["ndcg@30"] == 0.5000001
    primary = saved["primary_comparison"]
    assert primary["holdout_seen_users"]["direction"] == "higher"
    assert primary["holdout_seen_users"]["delta"] > 0
    assert primary["holdout_unseen_users"]["direction"] == "lower"
    review = (output / "review.md").read_text()
    assert "production-model" in review and "revision-1" in review
    assert "training dates are unknown" in review
    assert "10 / 100" in review
    assert "no model was deployed" in review
    assert {path.name for path in output.iterdir()} == {
        "metrics.json", "metrics.csv", "metric_deltas.csv", "review.md",
    }
    assert report["primary_comparison"] == saved["primary_comparison"]


def test_uploads_named_reports_and_logs_primary_metric_scalars(setup):
    comparison.run_production_comparison(**setup.kwargs)

    tracker = setup.kwargs["tracker"]
    assert {call.args[0] for call in tracker.log_file_artifact.call_args_list} == {
        "production_comparison_metrics_json", "production_comparison_metrics_csv",
        "production_comparison_metric_deltas_csv", "production_comparison_review_md",
    }
    assert tracker.log_scalar.call_count == 6
    assert {call.kwargs["series"] for call in tracker.log_scalar.call_args_list} == {
        "production", "candidate", "delta",
    }


def test_noop_tracker_keeps_local_reports(setup):
    setup.kwargs["tracker"] = NoOpExperimentTracker()

    assert comparison.run_production_comparison(**setup.kwargs)["status"] == "complete"
    assert (setup.kwargs["output_dir"] / "review.md").is_file()


@pytest.mark.parametrize("failure", [False, RuntimeError("upload failed")])
def test_report_upload_failure_preserves_local_files_and_raises(setup, failure):
    tracker = setup.kwargs["tracker"]
    if isinstance(failure, Exception):
        tracker.log_file_artifact.side_effect = failure
    else:
        tracker.log_file_artifact.return_value = failure

    with pytest.raises(RuntimeError):
        comparison.run_production_comparison(**setup.kwargs)

    output = setup.kwargs["output_dir"]
    assert (output / "review.md").is_file() and (output / "metrics.json").is_file()


@pytest.mark.parametrize("split", comparison.HOLDOUT_SPLITS)
def test_empty_holdout_prevents_evaluation(setup, split):
    setup.dataset.loader_index_validation["splits"][split]["query_count"] = 0

    with pytest.raises(ValueError, match=f"nonempty {split}"):
        comparison.run_production_comparison(**setup.kwargs)
    setup.evaluate.assert_not_called()


@pytest.mark.parametrize("start,end", [(None, None), ("2026-10-17", "2026-10-16")])
def test_unconfigured_or_invalid_holdout_prevents_evaluation(setup, start, end):
    path = setup.stage1 / "summary.json"
    document = json.loads(path.read_text())
    document["parameters"].update(holdout_start=start, holdout_end=end)
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="configured, nonempty holdout"):
        comparison.run_production_comparison(**setup.kwargs)
    setup.evaluate.assert_not_called()


def test_history_capacity_checks_source_extraction_not_candidate_model_limit(setup):
    (setup.stage2 / "summary.json").write_text(json.dumps({"parameters": {
        "max_history_posts_per_query": 32,
    }}))

    with pytest.raises(ValueError, match="extracted with limit 32.*require 64"):
        comparison.run_production_comparison(**setup.kwargs)
    setup.evaluate.assert_not_called()


def test_history_capacity_accepts_recorded_stage7_limit_without_stage2(setup):
    setup.dataset.summary["parameters"]["max_history_posts_per_query"] = 128
    (setup.stage2 / "summary.json").unlink()

    assert comparison.validate_history_capacity(setup.dataset, 64) == 128


def test_missing_history_lineage_cannot_silently_truncate_production(setup):
    setup.dataset.manifest["inputs"].pop("02_user_history")

    with pytest.raises(ValueError, match="02_user_history"):
        comparison.run_production_comparison(**setup.kwargs)
    setup.evaluate.assert_not_called()


def test_can_follow_stage7_summary_lineage_when_manifest_omits_it(setup):
    setup.dataset.manifest["inputs"].clear()
    setup.dataset.summary["input"].update(
        query_selection_dir=str(setup.stage1), user_history_dir=str(setup.stage2),
    )

    assert comparison.run_production_comparison(**setup.kwargs)["status"] == "complete"


def test_known_production_dates_are_reported_without_claiming_nonoverlap(setup):
    setup.kwargs["production_metadata"]["training_window"] = {
        "train_start": "2026-10-01", "val_start": "2026-10-15",
    }

    comparison.run_production_comparison(**setup.kwargs)

    review = (setup.kwargs["output_dir"] / "review.md").read_text()
    assert "2026-10-15" in review and "Review that window for overlap" in review


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_missing_primary_metric_does_not_produce_success_report(setup, value):
    setup.result.metrics_by_model["candidate"]["holdout_seen_users"]["ndcg@30"] = value

    with pytest.raises(ValueError, match="finite ndcg@30"):
        comparison.run_production_comparison(**setup.kwargs)
    assert not (setup.kwargs["output_dir"] / "metrics.json").exists()


def test_comparison_failure_propagates_without_success_files(setup):
    setup.evaluate.side_effect = RuntimeError("scoring failed")

    with pytest.raises(RuntimeError, match="scoring failed"):
        comparison.run_production_comparison(**setup.kwargs)
    assert not (setup.kwargs["output_dir"] / "metrics.json").exists()


def test_non_bst_baseline_is_rejected(setup):
    setup.kwargs["production_model"] = replace(setup.production, model_type="two-tower")

    with pytest.raises(ValueError, match="BST rankers only"):
        comparison.run_production_comparison(**setup.kwargs)
    setup.evaluate.assert_not_called()


def test_compares_real_tiny_torchscript_models_on_both_holdouts(tmp_path):
    import polars as pl

    from engagement_prediction.data.training_index import build_loader_index
    from engagement_prediction.evaluation.model_performance_test import (
        _write_model_artifact,
        _write_stage7_artifact,
    )

    stage7, bundle = _write_stage7_artifact(tmp_path)
    # Reuse the canonical two-query fixture, putting one query in each holdout.
    queries_path = next((bundle / "queries").glob("*.parquet"))
    queries = pl.read_parquet(queries_path).with_columns(
        pl.Series("split", list(comparison.HOLDOUT_SPLITS)),
    )
    queries.write_parquet(queries_path)
    shutil.rmtree(bundle / "loader_index")
    build_loader_index(
        posts_path=bundle / "posts", queries_path=bundle / "queries",
        query_positives_path=bundle / "query_positives",
        query_histories_path=bundle / "query_histories",
        hourly_negative_candidates_path=bundle / "hourly_negative_candidates",
        embeddings_path=bundle / "embeddings.npy", authors_path=bundle / "authors",
        indexed_post_liker_events_path=bundle / "indexed_post_liker_events",
        post_liker_users_path=bundle / "post_liker_users",
        output_path=bundle / "loader_index", logger=None,
    )
    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    (stage1 / "summary.json").write_text(json.dumps({
        "posts_start": "2025-12-01", "posts_end": "2026-01-02",
        "parameters": {"holdout_start": "2026-01-01", "holdout_end": None},
    }))
    summary_path = stage7 / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["parameters"]["max_history_posts_per_query"] = 3
    summary["input"] = {"query_selection_dir": str(stage1)}
    summary_path.write_text(json.dumps(summary))
    candidate_dir = _write_model_artifact(
        tmp_path / "candidate", model_type="bst-ranker", max_history_len=2,
        author_dids=["author0", "author1"], stage7_dir=stage7,
    )
    production_dir = _write_model_artifact(
        tmp_path / "production", model_type="bst-ranker", max_history_len=3,
        author_dids=["author1", "author2"], stage7_dir=stage7,
    )
    config_path = candidate_dir / "training_config.json"
    config = json.loads(config_path.read_text())
    config.update(eval_batch_size=2, random_seed=42, bst_additional_batch_negatives=1, metrics_top_ks=[30])
    config_path.write_text(json.dumps(config))
    args = SimpleNamespace(
        device="cpu", metrics_top_ks=[30], production_comparison_candidate_chunk_size=2,
        production_comparison_max_classification_metric_pairs=1000,
        num_dataloader_workers=0, dataloader_pin_memory=False, dataloader_prefetch_factor=2,
    )

    report = comparison.run_production_comparison(
        candidate_dir=candidate_dir,
        production_model=comparison.resolve_model_artifact("production", production_dir),
        production_metadata={"ranker_clearml_model_id": "fixture-production", "training_window": None},
        args=args, output_dir=tmp_path / "review", tracker=NoOpExperimentTracker(),
        logger=logging.getLogger(__name__),
    )

    assert report["status"] == "complete"
    assert report["split_row_counts"] == {split: 1 for split in comparison.HOLDOUT_SPLITS}
    assert report["history_lengths"] == {"production": 3, "candidate": 2}
    assert set(report["metrics"]) == {"production", "candidate"}
    assert (tmp_path / "review" / "review.md").is_file()
