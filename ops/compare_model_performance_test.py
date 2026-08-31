"""Tests for the standalone model-performance comparison CLI."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import pytest

import cli
from ops import compare_model_performance


def _required_arguments() -> list[str]:
    return [
        "--dataset",
        "/artifacts/07_dataset_hydration/stage7",
        "--model",
        "baseline=/artifacts/08_train_bst_ranker/baseline",
        "--model",
        "candidate=/artifacts/08_train_two_tower/candidate",
    ]


def test_parser_defaults_match_standalone_comparison_contract():
    args = compare_model_performance.build_parser().parse_args(_required_arguments())

    assert args.dataset == Path("/artifacts/07_dataset_hydration/stage7")
    assert args.model == [
        compare_model_performance.ModelArgument(
            "baseline",
            Path("/artifacts/08_train_bst_ranker/baseline"),
        ),
        compare_model_performance.ModelArgument(
            "candidate",
            Path("/artifacts/08_train_two_tower/candidate"),
        ),
    ]
    assert args.author_map == []
    assert args.splits == list(compare_model_performance.DEFAULT_SPLITS)
    assert args.batch_size == 128
    assert args.metrics_top_ks == [30]
    assert args.bst_candidate_chunk_size == 1024
    assert args.device is None
    assert args.num_dataloader_workers == 4
    assert args.dataloader_pin_memory is True
    assert args.dataloader_prefetch_factor == 2
    assert args.max_history_len is None
    assert args.output_dir == Path("outputs/comparisons")


@pytest.mark.parametrize("model_count", [0, 1, 3])
def test_parser_requires_exactly_two_models(model_count):
    argv = ["--dataset", "/stage7"]
    for index in range(model_count):
        argv.extend(["--model", f"model-{index}=/stage8/{index}"])

    with pytest.raises(SystemExit):
        compare_model_performance.build_parser().parse_args(argv)


def test_parser_requires_unique_model_names():
    with pytest.raises(SystemExit, match="2"):
        compare_model_performance.build_parser().parse_args([
            "--dataset",
            "/stage7",
            "--model",
            "same=/stage8/a",
            "--model",
            "same=/stage8/b",
        ])


@pytest.mark.parametrize(
    "raw",
    [
        "missing-separator",
        "=/stage8/model",
        "name=",
        "   =   ",
    ],
)
def test_model_argument_rejects_invalid_name_path_syntax(raw):
    with pytest.raises(argparse.ArgumentTypeError, match="NAME=PATH"):
        compare_model_performance.parse_model_argument(raw)


def test_model_argument_splits_only_the_first_equals_character():
    parsed = compare_model_performance.parse_model_argument(
        " candidate = /stage8/run=name "
    )

    assert parsed == compare_model_performance.ModelArgument(
        "candidate",
        Path("/stage8/run=name"),
    )


def test_parser_accepts_model_specific_author_map_overrides():
    args = compare_model_performance.build_parser().parse_args([
        *_required_arguments(),
        "--author-map",
        "baseline=/legacy/author_idx.parquet",
        "--author-map",
        "candidate=/canonical/two_tower_author_idx.parquet",
    ])

    assert args.author_map == [
        compare_model_performance.ModelArgument(
            "baseline",
            Path("/legacy/author_idx.parquet"),
        ),
        compare_model_performance.ModelArgument(
            "candidate",
            Path("/canonical/two_tower_author_idx.parquet"),
        ),
    ]


@pytest.mark.parametrize(
    "author_maps",
    [
        ["unknown=/legacy/author_idx.parquet"],
        [
            "baseline=/legacy/author_idx-a.parquet",
            "baseline=/legacy/author_idx-b.parquet",
        ],
    ],
)
def test_parser_rejects_unknown_or_duplicate_author_map_names(author_maps):
    argv = _required_arguments()
    for author_map in author_maps:
        argv.extend(["--author-map", author_map])

    with pytest.raises(SystemExit):
        compare_model_performance.build_parser().parse_args(argv)


def test_author_map_argument_rejects_invalid_name_path_syntax():
    with pytest.raises(argparse.ArgumentTypeError, match="author map.*NAME=PATH"):
        compare_model_performance.parse_author_map_argument("missing-separator")


def test_cli_rejects_removed_compare_rankers_command_and_options():
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["compare-rankers"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "baseline=/stage8"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--splits", "val"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--bst-candidate-chunk-size", "1024"])


def test_output_allocator_creates_only_partial_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        compare_model_performance,
        "generate_comparison_run_id",
        lambda: "comparison-test",
    )

    run_id, partial_path, output_path = (
        compare_model_performance._allocate_output_paths(tmp_path)
    )

    assert run_id == "comparison-test"
    assert partial_path == tmp_path / "comparison-test.partial"
    assert partial_path.is_dir()
    assert output_path == tmp_path / "comparison-test"
    assert not output_path.exists()


@dataclass(frozen=True)
class _FakeDatasetArtifact:
    root: Path

    @property
    def bundle_path(self):
        return self.root / "hydrated_training_data_test"

    def to_dict(self):
        return {"root": str(self.root), "embedding_model": "test-model"}


@dataclass(frozen=True)
class _FakeModelArtifact:
    name: str
    root: Path
    model_type: str
    artifact_format: str = "canonical_stage8"

    def to_dict(self):
        return {
            "name": self.name,
            "root": str(self.root),
            "model_type": self.model_type,
            "artifact_format": self.artifact_format,
            "model_config": {"similarity_temperature": 0.123456789},
        }


@dataclass(frozen=True)
class _FakeComparisonResult:
    metrics_by_model: dict
    mapping_coverage_by_model: dict
    split_row_counts: dict
    skipped_splits: tuple
    history_lengths: dict


def _patch_comparison_api(monkeypatch, *, run_model_comparison):
    from engagement_prediction.evaluation import artifacts, comparison, reporting

    dataset = _FakeDatasetArtifact(Path("/resolved/stage7"))
    models = (
        _FakeModelArtifact("baseline", Path("/resolved/baseline"), "bst-ranker"),
        _FakeModelArtifact("candidate", Path("/resolved/candidate"), "two-tower"),
    )
    monkeypatch.setattr(artifacts, "resolve_stage7_artifact", lambda path: dataset)
    monkeypatch.setattr(
        artifacts,
        "resolve_model_artifact",
        lambda name, path, *, author_map_override=None: (
            models[0] if name == "baseline" else models[1]
        ),
    )
    monkeypatch.setattr(
        artifacts,
        "validate_comparison_contract",
        lambda resolved_dataset, resolved_models, max_history_len: {
            "baseline": 20,
            "candidate": 16,
        },
    )
    monkeypatch.setattr(comparison, "ComparisonSettings", lambda **kwargs: kwargs)
    monkeypatch.setattr(comparison, "run_model_comparison", run_model_comparison)

    def write_metrics_csv(path, **kwargs):
        path.write_text("model_name,split,metric,value\n")

    def write_metric_deltas_csv(path, **kwargs):
        path.write_text("model_a,model_b,split,metric,delta\n")

    monkeypatch.setattr(reporting, "write_metrics_csv", write_metrics_csv)
    monkeypatch.setattr(reporting, "write_metric_deltas_csv", write_metric_deltas_csv)
    monkeypatch.setattr(compare_model_performance, "_resolve_device", lambda value: "cpu")
    monkeypatch.setattr(
        compare_model_performance,
        "generate_comparison_run_id",
        lambda: "comparison-test",
    )
    return dataset, models


def test_run_publishes_complete_output_atomically_and_cleans_temporary_files(
    tmp_path,
    monkeypatch,
):
    observed_temporary_dirs = []

    def fake_run_model_comparison(**kwargs):
        observed_temporary_dirs.append(kwargs["temporary_dir"])
        assert kwargs["temporary_dir"].is_dir()
        return _FakeComparisonResult(
            metrics_by_model={
                "baseline": {"val": {"ndcg@30": 0.41234567}},
                "candidate": {"val": {"ndcg@30": 0.51234567}},
            },
            mapping_coverage_by_model={
                "baseline": {
                    "model_unknown_post_count": 1,
                    "model_known_post_fraction": 0.12345678,
                },
                "candidate": {
                    "model_unknown_post_count": 2,
                    "model_known_post_fraction": 0.98765432,
                },
            },
            split_row_counts={"val": 3},
            skipped_splits=(),
            history_lengths={"baseline": 20, "candidate": 16},
        )

    _patch_comparison_api(
        monkeypatch,
        run_model_comparison=fake_run_model_comparison,
    )
    args = compare_model_performance.build_parser().parse_args([
        *_required_arguments(),
        "--output-dir",
        str(tmp_path),
    ])

    output_path = compare_model_performance._run(args)

    assert output_path == tmp_path / "comparison-test"
    assert not (tmp_path / "comparison-test.partial").exists()
    assert set(path.name for path in output_path.iterdir()) == {
        "comparison.log",
        "metric_deltas.csv",
        "metrics.csv",
        "metrics.json",
        "model_specs.json",
        "stage_info.txt",
    }
    assert observed_temporary_dirs
    assert all(not path.exists() for path in observed_temporary_dirs)
    metrics = json.loads((output_path / "metrics.json").read_text())
    assert metrics["metrics"]["baseline"]["val"]["ndcg@30"] == 0.41235
    assert metrics["metrics"]["candidate"]["val"]["ndcg@30"] == 0.51235
    assert metrics["mapping_coverage"]["baseline"] == {
        "model_unknown_post_count": 1,
        "model_known_post_fraction": 0.12346,
    }
    assert metrics["runtime_seconds"] == round(metrics["runtime_seconds"], 5)
    assert metrics["skipped_splits"] == []
    model_specs = json.loads((output_path / "model_specs.json").read_text())
    assert model_specs["mapping_coverage"]["candidate"][
        "model_known_post_fraction"
    ] == 0.98765
    assert model_specs["models"][0]["model_config"][
        "similarity_temperature"
    ] == 0.123456789
    runtime_line = next(
        line
        for line in (output_path / "stage_info.txt").read_text().splitlines()
        if line.startswith("runtime_seconds: ")
    )
    assert len(runtime_line.rsplit(".", 1)[1]) == 5
    completion_line = next(
        line
        for line in (output_path / "comparison.log").read_text().splitlines()
        if "Comparison completed successfully in " in line
    )
    logged_runtime = completion_line.split(
        "Comparison completed successfully in ", 1
    )[1].split(" seconds", 1)[0]
    assert len(logged_runtime.rsplit(".", 1)[1]) == 5


def test_run_passes_matching_author_map_overrides_to_artifact_resolution(
    tmp_path,
    monkeypatch,
):
    from engagement_prediction.evaluation import artifacts, comparison, reporting

    dataset = _FakeDatasetArtifact(Path("/resolved/stage7"))
    models = (
        _FakeModelArtifact("baseline", Path("/resolved/baseline"), "bst-ranker"),
        _FakeModelArtifact("candidate", Path("/resolved/candidate"), "two-tower"),
    )
    observed = []
    monkeypatch.setattr(artifacts, "resolve_stage7_artifact", lambda path: dataset)

    def resolve_model(name, path, *, author_map_override=None):
        observed.append((name, path, author_map_override))
        return models[0] if name == "baseline" else models[1]

    monkeypatch.setattr(artifacts, "resolve_model_artifact", resolve_model)
    monkeypatch.setattr(
        artifacts,
        "validate_comparison_contract",
        lambda resolved_dataset, resolved_models, max_history_len: {
            "baseline": 20,
            "candidate": 16,
        },
    )
    monkeypatch.setattr(comparison, "ComparisonSettings", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        comparison,
        "run_model_comparison",
        lambda **kwargs: _FakeComparisonResult(
            metrics_by_model={"baseline": {}, "candidate": {}},
            mapping_coverage_by_model={"baseline": {}, "candidate": {}},
            split_row_counts={},
            skipped_splits=(),
            history_lengths={"baseline": 20, "candidate": 16},
        ),
    )
    monkeypatch.setattr(
        reporting,
        "write_metrics_csv",
        lambda path, **kwargs: path.write_text("model_name,split,metric,value\n"),
    )
    monkeypatch.setattr(
        reporting,
        "write_metric_deltas_csv",
        lambda path, **kwargs: path.write_text(
            "model_a,model_b,split,metric,delta\n"
        ),
    )
    monkeypatch.setattr(compare_model_performance, "_resolve_device", lambda value: "cpu")
    monkeypatch.setattr(
        compare_model_performance,
        "generate_comparison_run_id",
        lambda: "comparison-test",
    )
    args = compare_model_performance.build_parser().parse_args([
        *_required_arguments(),
        "--author-map",
        "baseline=/legacy/author_idx.parquet",
        "--output-dir",
        str(tmp_path),
    ])

    compare_model_performance._run(args)

    assert observed == [
        (
            "baseline",
            Path("/artifacts/08_train_bst_ranker/baseline"),
            Path("/legacy/author_idx.parquet"),
        ),
        (
            "candidate",
            Path("/artifacts/08_train_two_tower/candidate"),
            None,
        ),
    ]


def test_run_retains_partial_output_and_removes_temporary_files_on_failure(
    tmp_path,
    monkeypatch,
):
    observed_temporary_dirs = []

    def fail_comparison(**kwargs):
        observed_temporary_dirs.append(kwargs["temporary_dir"])
        raise RuntimeError("test comparison failure")

    _patch_comparison_api(monkeypatch, run_model_comparison=fail_comparison)
    args = compare_model_performance.build_parser().parse_args([
        *_required_arguments(),
        "--output-dir",
        str(tmp_path),
    ])

    with pytest.raises(RuntimeError, match="test comparison failure"):
        compare_model_performance._run(args)

    partial_path = tmp_path / "comparison-test.partial"
    assert partial_path.is_dir()
    assert not (tmp_path / "comparison-test").exists()
    assert (partial_path / "comparison.log").is_file()
    assert "test comparison failure" in (
        partial_path / "comparison.log"
    ).read_text()
    assert observed_temporary_dirs
    assert all(not path.exists() for path in observed_temporary_dirs)
