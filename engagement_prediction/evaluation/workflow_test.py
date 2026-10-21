"""Compatibility checks for the shared comparison orchestration."""

import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from engagement_prediction.evaluation import comparison, workflow
from engagement_prediction.evaluation.artifacts import ModelArtifact, Stage7Artifact
from engagement_prediction.evaluation.comparison import ComparisonResult, ComparisonSettings


@pytest.fixture
def resolved_comparison(tmp_path, monkeypatch):
    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    (stage1 / "summary.json").write_text(json.dumps({
        "posts_start": "2026-09-09", "posts_end": "2026-10-16",
        "parameters": {
            "train_start": "2026-09-16", "val_start": "2026-10-08",
            "holdout_start": "2026-10-13", "holdout_end": None,
        },
    }))
    dataset = Stage7Artifact(
        root=tmp_path / "stage7", bundle_path=tmp_path / "stage7" / "bundle",
        manifest={"stage_run_id": "stage7", "inputs": {"01_query_selection": str(stage1)}},
        summary={"parameters": {"max_history_posts_per_query": 64}},
        loader_index_validation={"embedding_count": 100, "format_version": 2, "splits": {
            "holdout_seen_users": {"query_count": 10},
            "holdout_unseen_users": {"query_count": 5},
        }},
        embedding_model="fixture", embedding_dim=2,
    )
    models = []
    for name, history_len in (("production", 64), ("candidate", 32)):
        root = tmp_path / name
        root.mkdir()
        script = root / "ranker.pt"
        script.write_bytes(b"fixture")
        models.append(ModelArtifact(
            name=name, root=root, artifact_format="canonical_stage8", model_type="bst-ranker",
            manifest={"stage_run_id": name},
            model_config={"unrounded_metadata": 0.123456789}, training_config={},
            script_paths={"ranker": script}, script_sha256={"ranker": "fixture-sha256"},
            author_map_path=root / "ranker_author_idx.parquet", author_map_stats={},
            author_map_allow_extra_columns=False, embedding_model="fixture", embedding_dim=2,
            max_history_len=history_len, author_table_num_rows=4, similarity_temperature=None,
        ))
    settings = ComparisonSettings(
        splits=("holdout_seen_users", "holdout_unseen_users"), batch_size=8,
        additional_batch_negatives=4, metrics_top_ks=(30,), bst_candidate_chunk_size=1024,
        device="cpu", num_dataloader_workers=0, dataloader_pin_memory=False,
        dataloader_prefetch_factor=2, random_seed=42, max_classification_metric_pairs=2000,
        max_history_len=None, disable_progress=True,
    )
    result = ComparisonResult(
        metrics_by_model={
            "production": {split: {"ndcg@30": 0.5} for split in settings.splits},
            "candidate": {
                "holdout_seen_users": {"ndcg@30": 0.5000001},
                "holdout_unseen_users": {"ndcg@30": 0.412345678},
            },
        },
        mapping_coverage_by_model={model.name: {
            "stage7_post_count": 100, "model_unknown_post_count": 10,
            "model_known_post_fraction": 0.123456789,
        } for model in models},
        split_row_counts={"holdout_seen_users": 10, "holdout_unseen_users": 5},
        skipped_splits=(), history_lengths={"production": 64, "candidate": 32},
    )
    evaluate = Mock(return_value=result)
    monkeypatch.setattr(comparison, "run_model_comparison", evaluate)
    metadata = {
        "ranker_clearml_model_id": "baseline-model",
        "cloud_run": {"revision": "baseline-revision"}, "training_window": None,
    }
    return SimpleNamespace(
        dataset=dataset, models=tuple(models), settings=settings, result=result,
        evaluate=evaluate, metadata=metadata, stage1=stage1,
    )


def _standalone_context():
    return workflow.StandaloneReportContext(
        run_id="comparison-fixture",
        started_at=datetime(2026, 10, 20, tzinfo=timezone.utc),
        started_monotonic=time.monotonic(),
    )


def _run(fixture, output_dir, context):
    return workflow.run_comparison_workflow(
        dataset=fixture.dataset, models=fixture.models, settings=fixture.settings,
        output_dir=output_dir, report_context=context, logger=logging.getLogger(__name__),
    )


def test_report_profiles_preserve_existing_schemas_precision_and_files(resolved_comparison, tmp_path):
    fixture = resolved_comparison
    standalone_dir, production_dir = tmp_path / "standalone", tmp_path / "automated"
    standalone = _run(fixture, standalone_dir, _standalone_context())
    production = _run(
        fixture, production_dir, workflow.ProductionReportContext(metadata=fixture.metadata),
    )

    assert set(standalone) == {
        "run_id", "started_at_utc", "runtime_seconds", "settings", "dataset", "models",
        "history_lengths", "split_row_counts", "skipped_splits", "mapping_coverage", "metrics",
    }
    assert set(production) == {
        "status", "created_at", "output_dir", "production", "models", "dataset", "dates",
        "settings", "history_extraction_limit", "history_lengths", "split_row_counts",
        "mapping_coverage", "metrics", "metric_deltas", "primary_metric", "primary_comparison",
        "report_paths",
    }
    assert {path.name for path in standalone_dir.iterdir()} == {
        "metrics.json", "metrics.csv", "metric_deltas.csv", "model_specs.json", "stage_info.txt",
    }
    assert {path.name for path in production_dir.iterdir()} == {
        "metrics.json", "metrics.csv", "metric_deltas.csv", "review.md",
    }
    for output_dir in (standalone_dir, production_dir):
        saved = json.loads((output_dir / "metrics.json").read_text())
        assert saved["models"][0]["model_config"]["unrounded_metadata"] == 0.123456789
        assert saved["settings"]["additional_batch_negatives"] == 4
        assert saved["history_lengths"] == {"production": 64, "candidate": 32}
        with (output_dir / "metric_deltas.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        seen_row = next(row for row in rows if row["split"] == "holdout_seen_users")
        # Both original CSV formats compute differences after rounding operands.
        assert seen_row["delta_model_b_minus_model_a"] == "0.00000"

    assert standalone["metrics"]["candidate"]["holdout_seen_users"]["ndcg@30"] == 0.5
    assert standalone["metrics"]["candidate"]["holdout_unseen_users"]["ndcg@30"] == 0.41235
    assert standalone["mapping_coverage"]["candidate"]["model_known_post_fraction"] == 0.12346
    assert production["metrics"] == fixture.result.metrics_by_model
    assert production["mapping_coverage"] == fixture.result.mapping_coverage_by_model
    primary = production["primary_comparison"]["holdout_seen_users"]
    assert primary["direction"] == "higher" and primary["delta"] > 0
    assert production["dates"]["holdout_end"] == "2026-10-16T00:00:00+00:00"
    assert production["history_extraction_limit"] == 64
    assert production["production"] == fixture.metadata
    assert "baseline-revision" in (production_dir / "review.md").read_text()
    assert "training dates are unknown" in (production_dir / "review.md").read_text()
    specs = json.loads((standalone_dir / "model_specs.json").read_text())
    assert set(specs) == {"models", "history_lengths", "mapping_coverage"}
    assert specs["models"] == standalone["models"]
    assert specs["mapping_coverage"] == standalone["mapping_coverage"]
    info = (standalone_dir / "stage_info.txt").read_text()
    assert "run_id: comparison-fixture\n" in info
    assert "additional_batch_negatives: 4\n" in info
    assert "skipped_splits: none\n" in info

    for call in fixture.evaluate.call_args_list:
        assert call.kwargs["dataset"] is fixture.dataset
        assert call.kwargs["models"] == fixture.models
        assert call.kwargs["settings"] is fixture.settings
        assert not call.kwargs["temporary_dir"].exists()


@pytest.mark.parametrize("problem,match", [
    ("insufficient_history", "extracted with limit 32.*require 64"),
    ("missing_dates", "01_query_selection"),
    ("empty_holdout", "nonempty holdout_seen_users"),
])
def test_production_requirements_do_not_become_standalone_requirements(
    resolved_comparison, tmp_path, problem, match,
):
    fixture = resolved_comparison
    if problem == "insufficient_history":
        fixture.dataset.summary["parameters"]["max_history_posts_per_query"] = 32
    elif problem == "missing_dates":
        fixture.dataset.manifest["inputs"].clear()
    else:
        fixture.dataset.loader_index_validation["splits"]["holdout_seen_users"]["query_count"] = 0
        fixture.evaluate.return_value = replace(
            fixture.result, skipped_splits=("holdout_seen_users",),
            split_row_counts={"holdout_seen_users": 0, "holdout_unseen_users": 5},
            metrics_by_model={
                model.name: {"holdout_unseen_users": fixture.result.metrics_by_model[model.name]["holdout_unseen_users"]}
                for model in fixture.models
            },
        )

    with pytest.raises(ValueError, match=match):
        _run(fixture, tmp_path / "production", workflow.ProductionReportContext(metadata=fixture.metadata))
    fixture.evaluate.assert_not_called()

    report = _run(fixture, tmp_path / "standalone", _standalone_context())
    fixture.evaluate.assert_called_once()
    if problem == "empty_holdout":
        assert tuple(report["skipped_splits"]) == ("holdout_seen_users",)


@pytest.mark.parametrize("profile", ["standalone", "production"])
@pytest.mark.parametrize("error", [RuntimeError("scoring failed"), KeyboardInterrupt()])
def test_scoring_failure_cleans_temporary_files_without_success_reports(
    resolved_comparison, tmp_path, profile, error,
):
    fixture = resolved_comparison
    observed = []

    def fail(**kwargs):
        temporary_dir = kwargs["temporary_dir"]
        assert temporary_dir.is_dir()
        (temporary_dir / "mapping.npy").write_bytes(b"temporary mapping")
        observed.append(temporary_dir)
        raise error

    fixture.evaluate.side_effect = fail
    context = (
        _standalone_context() if profile == "standalone"
        else workflow.ProductionReportContext(metadata=fixture.metadata)
    )
    output_dir = tmp_path / "failed"
    with pytest.raises(type(error)):
        _run(fixture, output_dir, context)

    assert observed and all(not path.exists() for path in observed)
    assert not (output_dir / "metrics.json").exists()
    assert not (output_dir / "review.md").exists()


def test_standalone_library_matches_direct_torchscript_evaluation(tmp_path):
    from engagement_prediction.evaluation.artifacts import resolve_model_artifact, resolve_stage7_artifact
    from engagement_prediction.evaluation.model_performance_test import (
        _write_model_artifact, _write_stage7_artifact,
    )
    from engagement_prediction.evaluation.reporting import round_output_floats

    stage7_path, _ = _write_stage7_artifact(tmp_path)
    dataset = resolve_stage7_artifact(stage7_path)
    models = tuple(
        resolve_model_artifact(name, _write_model_artifact(
            tmp_path / name, model_type=model_type, max_history_len=history_len,
            author_dids=["author0", "author1"], stage7_dir=stage7_path,
        ))
        for name, model_type, history_len in (
            ("baseline", "bst-ranker", 2), ("candidate", "two-tower", 3),
        )
    )
    settings = ComparisonSettings(
        splits=("train", "holdout_seen_users"), batch_size=1, additional_batch_negatives=1,
        metrics_top_ks=(2,), bst_candidate_chunk_size=2, device="cpu", num_dataloader_workers=0,
        dataloader_pin_memory=False, dataloader_prefetch_factor=2, random_seed=7,
        max_classification_metric_pairs=10, max_history_len=None, disable_progress=True,
    )
    logger = logging.getLogger(__name__)
    original = comparison.run_model_comparison(
        dataset=dataset, models=models, settings=settings,
        temporary_dir=tmp_path / "direct-mappings", logger=logger,
    )

    report = workflow.run_comparison_workflow(
        dataset=dataset, models=models, settings=settings,
        output_dir=tmp_path / "shared", report_context=_standalone_context(), logger=logger,
    )

    assert report["metrics"] == round_output_floats(original.metrics_by_model)
    assert report["mapping_coverage"] == round_output_floats(original.mapping_coverage_by_model)
    assert report["history_lengths"] == original.history_lengths == {"baseline": 2, "candidate": 3}
    assert report["split_row_counts"] == original.split_row_counts
    assert tuple(report["skipped_splits"]) == original.skipped_splits == ("holdout_seen_users",)
