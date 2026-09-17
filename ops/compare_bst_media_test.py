"""CLI defaults, validation, and report publication for BST media comparison."""

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from ops import compare_bst_media
from engagement_prediction.evaluation.media_artifacts_test import stage7


def test_defaults_and_one_or_more_named_models():
    parser = compare_bst_media.build_parser()
    for count in (1, 2, 3):
        arguments = [value for i in range(count) for value in ("--model", f"m{i}=/models/{i}")]
        args = parser.parse_args(arguments)
        assert len(args.model) == count
        assert args.es_url == "https://localhost:9202"
        assert args.metrics_top_ks == [30]
        assert args.es_verify_ssl is False
        assert args.output_dir == Path("/mnt/data/dave/outputs/compare")
        assert args.es_index == "posts"
        assert args.es_batch_size == 1000
        assert args.es_request_timeout == 30.0
        assert args.device is None
        assert args.num_dataloader_workers == 4


def test_cli_overrides():
    args = compare_bst_media.build_parser().parse_args([
        "--model", "a=/a", "--es-url", "http://127.0.0.1:9201",
        "--metrics-top-ks", "30", "1", "10", "--es-verify-ssl",
        "--output-dir", "/tmp/media", "--es-index", "posts,replies",
        "--es-batch-size", "8", "--es-request-timeout", "1.5",
        "--device", "cpu", "--num-dataloader-workers", "0",
        "--no-dataloader-pin-memory", "--dataloader-prefetch-factor", "1",
        "--disable-progress",
    ])
    assert args.es_url == "http://127.0.0.1:9201"
    assert args.metrics_top_ks == [1, 10, 30]
    assert args.es_verify_ssl is True
    assert args.output_dir == Path("/tmp/media")
    assert args.es_index == "posts,replies"
    assert args.es_batch_size == 8
    assert args.es_request_timeout == 1.5
    assert args.device == "cpu"
    assert args.num_dataloader_workers == 0
    assert args.dataloader_pin_memory is False
    assert args.dataloader_prefetch_factor == 1
    assert args.disable_progress is True


@pytest.mark.parametrize("extra", [
    ["--model", "a=/duplicate"],
    ["--metrics-top-ks", "0"],
    ["--metrics-top-ks", "1", "1"],
    ["--es-batch-size", "0"],
    ["--es-request-timeout", "nan"],
    ["--es-request-timeout", "0"],
    ["--num-dataloader-workers", "-1"],
    ["--dataloader-prefetch-factor", "0"],
    ["--es-url", "localhost:9202"],
    ["--es-url", "https://user:secret@localhost:9202"],
    ["--es-url", "https://localhost:9202?api_key=secret"],
    ["--es-index", "posts/_search"],
])
def test_invalid_arguments(extra):
    with pytest.raises(SystemExit):
        compare_bst_media.build_parser().parse_args(["--model", "a=/a", *extra])


def test_models_are_required():
    with pytest.raises(SystemExit):
        compare_bst_media.build_parser().parse_args([])


@dataclass(frozen=True)
class _Settings:
    batch_size: int
    additional_batch_negatives: int
    random_seed: int


def _patch_pipeline(monkeypatch, *, failure):
    from engagement_prediction.evaluation import media_artifacts, media_evaluation, media_hydration

    dataset = SimpleNamespace(
        bundle_path=Path("/shared/bundle"),
        to_dict=lambda: {"bundle_path": "/shared/bundle"},
    )
    model = SimpleNamespace(name="a", to_dict=lambda: {"name": "a", "checkpoint": "/a/best.pth"})
    settings = _Settings(128, 64, 42)
    candidates = pl.DataFrame({
        "at_uri": ["known", "missing"],
        "in_val": [True, True], "in_val_unseen_users": [True, False],
    })
    media = candidates.with_columns(
        pl.Series("contains_images", [True, None], dtype=pl.Boolean),
        pl.Series("contains_video", [False, None], dtype=pl.Boolean),
        pl.Series("media_status", ["known", "missing_document"]),
    )
    observed = []

    def resolve(specs):
        observed.append("resolve")
        assert specs == [("a", Path("/a"))]
        if failure == "resolve":
            raise ValueError("Different input directories")
        return dataset, (model,), settings

    def hydrate(frame, **kwargs):
        observed.append("hydrate")
        assert frame.equals(candidates)
        assert kwargs["es_url"] == "https://localhost:9202"
        assert kwargs["verify_ssl"] is False
        assert kwargs["api_key"] == "test-private-key"
        if failure == "hydrate":
            raise RuntimeError("Elasticsearch connection failed at https://localhost:9202")
        return media

    def evaluate(actual_dataset, actual_models, actual_settings, actual_media, **kwargs):
        observed.append("evaluate")
        assert actual_dataset is dataset
        assert actual_models == (model,)
        assert actual_settings == settings
        assert actual_media.equals(media)
        assert kwargs["device"] == "cpu"
        return {
            "metrics": [
                {"model": "a", "split": split, "media_type": group, "k": k,
                 "ndcg": 0.5 if group == "images" else None, "eligible_query_count": 1 if group == "images" else 0}
                for split in ("val", "val_unseen_users")
                for group in ("images", "videos", "text_only")
                for k in kwargs["metrics_top_ks"]
            ],
            "hydration_coverage": [{"split": "val", "requested_unique_uri_count": 2, "missing_document_count": 1}],
            "model_results": [{"model": "a", "best_epoch": 7}],
        }

    monkeypatch.setattr(media_artifacts, "resolve_media_artifacts", resolve)
    monkeypatch.setattr(media_artifacts, "collect_candidate_uris", lambda _: candidates)
    monkeypatch.setattr(media_hydration, "hydrate_candidate_media", hydrate)
    monkeypatch.setattr(media_evaluation, "run_media_evaluation", evaluate)
    monkeypatch.setenv("GE_ELASTICSEARCH_API_KEY", "test-private-key")
    return observed


@pytest.mark.parametrize("ks", [[30], [1, 10, 30]])
def test_report_publication_with_real_csv_parquet_and_plots(tmp_path, monkeypatch, ks):
    observed = _patch_pipeline(monkeypatch, failure=None)
    args = compare_bst_media.build_parser().parse_args([
        "--model", "a=/a", "--output-dir", str(tmp_path), "--device", "cpu",
        "--metrics-top-ks", *map(str, ks),
    ])
    output = compare_bst_media._run(args)
    assert observed == ["resolve", "hydrate", "evaluate"]
    assert output.parent == tmp_path
    assert not output.name.endswith(".partial")
    assert not list(tmp_path.glob("*.partial"))
    assert {p.name for p in output.iterdir()} == {
        "run.json", "comparison.log", "metrics.csv", "hydration_coverage.csv",
        "missing_media.csv", "media_metadata.parquet", "val_ndcg.png", "val_unseen_users_ndcg.png",
    }
    run_text = (output / "run.json").read_text()
    run = json.loads(run_text)
    assert run["status"] == "complete"
    assert run["validation_settings"] == {"batch_size": 128, "additional_batch_negatives": 64, "random_seed": 42}
    assert "test-private-key" not in run_text
    assert "test-private-key" not in (output / "comparison.log").read_text()
    assert pl.read_csv(output / "metrics.csv").height == 6 * len(ks)
    assert pl.read_csv(output / "metrics.csv")["ndcg"].null_count() == 4 * len(ks)
    assert pl.read_csv(output / "missing_media.csv")["at_uri"].to_list() == ["missing"]
    assert pl.read_parquet(output / "media_metadata.parquet").height == 2
    for name in ("val_ndcg.png", "val_unseen_users_ndcg.png"):
        assert (output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("failure, expected_calls", [
    ("resolve", ["resolve"]),
    ("hydrate", ["resolve", "hydrate"]),
])
def test_failure_keeps_partial_report_and_never_scores(tmp_path, monkeypatch, failure, expected_calls):
    observed = _patch_pipeline(monkeypatch, failure=failure)
    result = compare_bst_media.main([
        "--model", "a=/a", "--output-dir", str(tmp_path), "--device", "cpu",
    ])
    assert result == 1
    assert observed == expected_calls
    partial = list(tmp_path.iterdir())
    assert len(partial) == 1
    assert partial[0].name.endswith(".partial")
    assert json.loads((partial[0] / "run.json").read_text())["status"] == "failed"
    assert not (partial[0] / "metrics.csv").exists()


@pytest.mark.parametrize("model_count", [1, 2])
def test_full_cli_native_checkpoints_and_shared_es_hydration(tmp_path, monkeypatch, stage7, model_count):
    import requests
    from engagement_prediction.evaluation.media_artifacts_test import _model
    from engagement_prediction.evaluation.media_hydration_test import FakeResponse, _search

    calls = []

    class Session:
        def __init__(self):
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            body = kwargs["json"]
            if method == "DELETE":
                return FakeResponse({"succeeded": True})
            if "match_none" in body["query"]:
                return _search()
            return _search(*[
                {"at_uri": uri, "contains_images": i % 3 == 0, "contains_video": i % 3 == 1}
                for i, uri in enumerate(body["query"]["terms"]["at_uri"])
            ], scroll_id="native-test")

    monkeypatch.setattr(requests, "Session", Session)
    monkeypatch.setenv("GE_ELASTICSEARCH_API_KEY", "integration-secret")
    output_parent = tmp_path / "output"
    argv = [
        "--output-dir", str(output_parent), "--device", "cpu",
        "--num-dataloader-workers", "0", "--no-dataloader-pin-memory", "--disable-progress",
    ]
    for index in range(model_count):
        model_path = _model(tmp_path / f"model-{index}", stage7, use_post_liker_feature=index == 0)
        argv.extend(["--model", f"model-{index}={model_path}"])
    assert compare_bst_media.main(argv) == 0
    output, = output_parent.iterdir()
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "complete"
    assert len(run["model_results"]) == model_count
    assert all(result["best_epoch"] == 2 for result in run["model_results"])
    assert run["model_results"][0]["use_post_liker_feature"] is True
    metrics = pl.read_csv(output / "metrics.csv")
    assert metrics.height == model_count * 6
    assert set(metrics["split"]) == {"val", "val_unseen_users"}
    assert set(metrics["media_type"]) == {"images", "videos", "text_only"}
    assert metrics["ndcg"].drop_nulls().is_between(0, 1).all()
    assert len(calls) == 3  # One preflight, one shared lookup, one scroll cleanup.
    assert (output / "missing_media.csv").read_text().count("\n") == 1
    assert "integration-secret" not in (output / "run.json").read_text()
    if model_count == 2:
        assert run["model_results"][0]["splits"] == run["model_results"][1]["splits"]
