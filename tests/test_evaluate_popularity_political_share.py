import asyncio
import csv
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import requests


@pytest.fixture(scope="module")
def popularity_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "ops/evaluate_popularity_political_share.py"
    spec = importlib.util.spec_from_file_location(
        "evaluate_popularity_political_share", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_popularity_political_share"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _inference_json(politics=None, news=None):
    record = {}
    if politics is not None:
        record["text_arbitrary"] = {"Politics": politics}
    if news is not None:
        record["topic"] = {"News & Social Concern": news}
    return json.dumps({"text": {"message.commit.record.text": record}})


def _uri(name):
    return f"at://did:plc:{name}/app.bsky.feed.post/{name}"


def _args(module, output_dir, **overrides):
    values = {
        "model_a_dir": Path("model-a"),
        "model_b_dir": Path("model-b"),
        "model_a_get_data_dir": None,
        "model_b_get_data_dir": None,
        "small_num_candidates": 100,
        "large_num_candidates": 1000,
        "top_k": 50,
        "max_age_hours": 168,
        "api_timeout_seconds": 30.0,
        "gcs_bucket": "bucket",
        "inference_prefix": "bsky_inferences",
        "political_threshold": 0.95,
        "embedding_model": "all_MiniLM_L12_v2",
        "elasticsearch_url": "http://localhost:9200",
        "elasticsearch_api_key": "es-secret",
        "elasticsearch_index": "posts_recent",
        "elasticsearch_batch_size": 1000,
        "elasticsearch_timeout_seconds": 60.0,
        "elasticsearch_insecure": True,
        "device": "cpu",
        "output_dir": output_dir,
    }
    values.update(overrides)
    return module.argparse.Namespace(**values)


def _patch_fake_models(module, monkeypatch):
    def load_model_bundle(label, model_dir, get_data_dir, device):
        return SimpleNamespace(
            label=label,
            model_dir=Path(model_dir),
            author_idx_path=None,
            use_popularity_feature=True,
        )

    monkeypatch.setattr(
        module.bst_comparison,
        "load_model_bundle",
        load_model_bundle,
    )
    monkeypatch.setattr(
        module.bst_comparison,
        "validate_model_compatibility",
        lambda model_a, model_b: 2,
    )


class FakeResponse:
    def __init__(self, payload=None, status_code=200, json_error=None):
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def post(self, url, headers, json, timeout):
        self.calls.append({
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        })
        if self.error is not None:
            raise self.error
        if isinstance(self.response, list):
            return self.response[len(self.calls) - 1]
        return self.response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_cli_defaults(popularity_module):
    args = popularity_module.build_arg_parser().parse_args(["model-a", "model-b"])

    assert args.model_a_dir == Path("model-a")
    assert args.model_b_dir == Path("model-b")
    assert args.small_num_candidates == 100
    assert args.large_num_candidates == 1000
    assert args.top_k == 50
    assert not hasattr(args, "num_candidates")
    assert args.max_age_hours == 168
    assert args.api_timeout_seconds == 30.0
    assert args.gcs_bucket == popularity_module.DEFAULT_GCS_BUCKET
    assert args.inference_prefix == "bsky_inferences"
    assert args.political_threshold == 0.95
    assert args.embedding_model == "all_MiniLM_L12_v2"
    assert args.elasticsearch_index == "posts_recent"
    assert args.elasticsearch_batch_size == 1000
    assert args.elasticsearch_timeout_seconds == 60.0
    assert args.elasticsearch_insecure is True
    assert args.device == "cpu"
    assert args.output_dir is None

    secure = popularity_module.build_arg_parser().parse_args(
        ["model-a", "model-b", "--elasticsearch-secure"]
    )
    assert secure.elasticsearch_insecure is False

    configured = popularity_module.build_arg_parser().parse_args([
        "model-a",
        "model-b",
        "--model-a-get-data-dir",
        "data-a",
        "--model-b-get-data-dir",
        "data-b",
        "--small-num-candidates",
        "25",
        "--large-num-candidates",
        "250",
        "--top-k",
        "10",
        "--political-threshold",
        "0.8",
        "--elasticsearch-url",
        "http://localhost:19200",
        "--elasticsearch-index",
        "custom-posts",
        "--elasticsearch-batch-size",
        "50",
        "--elasticsearch-timeout-seconds",
        "12.5",
    ])
    assert configured.model_a_get_data_dir == Path("data-a")
    assert configured.model_b_get_data_dir == Path("data-b")
    assert configured.small_num_candidates == 25
    assert configured.large_num_candidates == 250
    assert configured.top_k == 10
    assert configured.political_threshold == 0.8
    assert configured.elasticsearch_url == "http://localhost:19200"
    assert configured.elasticsearch_index == "custom-posts"
    assert configured.elasticsearch_batch_size == 50
    assert configured.elasticsearch_timeout_seconds == 12.5

    with pytest.raises(SystemExit):
        popularity_module.build_arg_parser().parse_args(
            ["model-a", "model-b", "--num-candidates", "10"]
        )


def test_api_settings_default_override_validation_and_sanitization(popularity_module):
    assert popularity_module.resolve_api_settings({"GE_API_KEY": "secret"}) == (
        "https://api.greenearth.social",
        "secret",
    )
    assert popularity_module.resolve_api_settings({
        "GE_API_URL": "localhost:8000/",
        "GE_API_KEY": " secret ",
    }) == ("https://localhost:8000", "secret")
    assert popularity_module.sanitize_api_url(
        "https://user:password@example.com:8443/base?token=secret"
    ) == "https://example.com:8443/base"

    with pytest.raises(ValueError, match="GE_API_KEY is required"):
        popularity_module.resolve_api_settings({})
    with pytest.raises(ValueError, match="GE_API_URL must not be empty"):
        popularity_module.resolve_api_settings({
            "GE_API_URL": "  ",
            "GE_API_KEY": "secret",
        })


@pytest.mark.parametrize(
    ("small_num_candidates", "large_num_candidates"),
    [(0, 1000), (100, 1001)],
)
def test_candidate_count_validation(
    popularity_module, small_num_candidates, large_num_candidates
):
    with pytest.raises(ValueError, match="between 1 and 1000"):
        popularity_module.validate_parameters(
            small_num_candidates,
            large_num_candidates,
            50,
            168,
            30.0,
            0.95,
            1000,
            60.0,
        )


def test_other_parameter_validation(popularity_module):
    with pytest.raises(ValueError, match="less than or equal"):
        popularity_module.validate_parameters(101, 100, 50, 168, 30.0, 0.95, 1000, 60.0)
    with pytest.raises(ValueError, match="top-k must be positive"):
        popularity_module.validate_parameters(100, 1000, 0, 168, 30.0, 0.95, 1000, 60.0)
    with pytest.raises(ValueError, match="must be one of"):
        popularity_module.validate_parameters(100, 1000, 50, 100, 30.0, 0.95, 1000, 60.0)
    with pytest.raises(ValueError, match="must be positive"):
        popularity_module.validate_parameters(100, 1000, 50, 168, 0.0, 0.95, 1000, 60.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        popularity_module.validate_parameters(100, 1000, 50, 168, 30.0, 1.1, 1000, 60.0)
    with pytest.raises(ValueError, match="batch-size must be positive"):
        popularity_module.validate_parameters(100, 1000, 50, 168, 30.0, 0.95, 0, 60.0)
    with pytest.raises(ValueError, match="timeout-seconds must be positive"):
        popularity_module.validate_parameters(100, 1000, 50, 168, 30.0, 0.95, 1000, 0.0)


def test_inference_window_adds_six_days_to_both_ends(popularity_module):
    request_time = datetime(2026, 8, 12, 15, 30, tzinfo=timezone.utc)

    window = popularity_module.derive_inference_window(request_time, 168)

    assert window.start == datetime(2026, 7, 30, 15, 30, tzinfo=timezone.utc)
    assert window.end == datetime(2026, 8, 18, 15, 30, tzinfo=timezone.utc)


def test_inference_file_listing_uses_half_open_buffered_window(popularity_module):
    class Blob:
        def __init__(self, name):
            self.name = name

    class Client:
        def __init__(self):
            self.calls = []

        def list_blobs(self, bucket, prefix):
            self.calls.append((bucket, prefix))
            return [
                Blob("bsky_inferences_20260730_152959.parquet"),
                Blob("bsky_inferences_20260730_153000.parquet"),
                Blob("bsky_inferences_20260818_152959.parquet"),
                Blob("bsky_inferences_20260818_153000.parquet"),
                Blob("other_20260801_000000.parquet"),
                Blob("bsky_inferences_invalid.parquet"),
            ]

    client = Client()
    window = popularity_module.derive_inference_window(
        datetime(2026, 8, 12, 15, 30, tzinfo=timezone.utc), 168
    )

    paths = popularity_module.list_inference_parquet_paths(
        client, "bucket", "bsky_inferences", window
    )

    assert client.calls == [("bucket", "bsky_inferences")]
    assert paths == [
        "gs://bucket/bsky_inferences_20260730_153000.parquet",
        "gs://bucket/bsky_inferences_20260818_152959.parquet",
    ]


def test_api_request_uses_popularity_only_api_key_and_configured_count(popularity_module):
    session = FakeSession(FakeResponse({
        "candidates": [{"at_uri": "at://post/one"}],
    }))

    candidates = popularity_module.fetch_popularity_candidates(
        session,
        "https://api.greenearth.social",
        "top-secret",
        250,
        72,
        12.5,
    )

    assert candidates == [{"at_uri": "at://post/one"}]
    assert session.calls == [{
        "url": "https://api.greenearth.social/candidates/generate",
        "headers": {
            "X-API-Key": "top-secret",
            "Content-Type": "application/json",
        },
        "json": {
            "generators": [{"name": "popularity", "weight": 1.0}],
            "user_did": popularity_module.DEFAULT_USER_DID,
            "num_candidates": 250,
            "video_only": False,
            "max_age_hours": 72,
            "exclude_uris": [],
        },
        "timeout": 12.5,
    }]


def test_api_failures_are_clear_and_do_not_expose_key(popularity_module):
    cases = [
        (FakeSession(error=requests.Timeout()), "timed out"),
        (FakeSession(error=requests.ConnectionError()), "request failed"),
        (FakeSession(FakeResponse({}, status_code=401)), "HTTP 401"),
        (FakeSession(FakeResponse(json_error=ValueError("bad"))), "invalid JSON"),
        (FakeSession(FakeResponse({})), "missing a candidates list"),
        (FakeSession(FakeResponse({"candidates": ["bad"]})), "must be JSON objects"),
    ]
    for session, error_text in cases:
        with pytest.raises(RuntimeError, match=error_text) as exc_info:
            popularity_module.fetch_popularity_candidates(
                session,
                "https://api.greenearth.social",
                "top-secret",
                100,
                168,
                30.0,
            )
        assert "top-secret" not in str(exc_info.value)


def test_at_uri_to_url_accepts_posts_and_rejects_malformed_uris(popularity_module):
    at_uri = "at://did:plc:alice/app.bsky.feed.post/3abc?ignored=true"

    assert popularity_module.at_uri_to_url(at_uri) == (
        "https://bsky.app/profile/did:plc:alice/post/3abc"
    )
    with pytest.raises(ValueError, match="Could not parse AT URI"):
        popularity_module.at_uri_to_url("at://did:plc:alice/app.bsky.feed.like/3abc")


def test_candidate_frame_preserves_api_rank_and_diagnoses_bad_uris(popularity_module):
    first_uri = _uri("one")
    second_uri = _uri("two")
    candidates = [
        {
            "at_uri": first_uri,
            "score": 3,
            "author_did": "did:one",
            "content": "one",
            "like_count": 12,
            "generator_name": "popularity",
        },
        {"at_uri": None, "score": 2},
        {"at_uri": first_uri, "score": 1},
        {
            "at_uri": second_uri,
            "score": float("nan"),
            "like_count": "9",
        },
    ]

    frame, diagnostics = popularity_module.build_candidate_frame(
        candidates, 100, "small"
    )

    assert frame["candidate_pool"].to_list() == ["small", "small"]
    assert frame["requested_pool_size"].to_list() == [100, 100]
    assert frame["candidate_rank"].to_list() == [1, 4]
    assert frame["at_uri"].to_list() == [first_uri, second_uri]
    assert frame["url"].to_list() == [
        "https://bsky.app/profile/did:plc:one/post/one",
        "https://bsky.app/profile/did:plc:two/post/two",
    ]
    assert frame["popularity_score"].to_list() == [3.0, None]
    assert frame["like_count"].to_list() == [12, None]
    assert diagnostics.requested_candidates == 100
    assert diagnostics.api_candidates_returned == 4
    assert diagnostics.unique_candidate_uris == 2
    assert diagnostics.candidates_missing_at_uri == 1
    assert diagnostics.duplicate_candidate_uris == 1


def test_inference_scores_filter_uris_before_decode_and_use_latest_row(
    popularity_module, tmp_path
):
    path = tmp_path / "inferences.parquet"
    pl.DataFrame({
        "at_uri": [
            "political",
            "one-low",
            "missing-field",
            "changed",
            "changed",
            "outside-candidate-set",
        ],
        "indexed_at": [
            "2026-08-01T00:00:00Z",
            "2026-08-01T00:00:00Z",
            "2026-08-01T00:00:00Z",
            "2026-08-01T00:00:00Z",
            "2026-08-02T00:00:00Z",
            "2026-08-01T00:00:00Z",
        ],
        "inferences": [
            _inference_json(0.8, 0.8),
            _inference_json(0.9, 0.79),
            _inference_json(0.95, None),
            _inference_json(0.2, 0.2),
            _inference_json(0.91, 0.92),
            "not valid JSON",
        ],
    }).write_parquet(path)

    scores = popularity_module.build_candidate_inference_scores(
        [str(path)],
        ["political", "one-low", "missing-field", "changed", "not-found"],
        0.8,
    )

    by_uri = {row["at_uri"]: row for row in scores.iter_rows(named=True)}
    assert set(by_uri) == {"political", "one-low", "missing-field", "changed"}
    assert by_uri["political"]["news_social_concern_score"] == 0.8
    assert by_uri["one-low"]["news_social_concern_score"] == 0.79
    assert by_uri["missing-field"]["news_social_concern_score"] is None
    assert by_uri["changed"]["news_social_concern_score"] == 0.92
    assert by_uri["changed"]["inference_indexed_at"] == "2026-08-02T00:00:00Z"
    assert by_uri["political"]["is_political"] is True
    assert by_uri["one-low"]["is_political"] is False
    assert by_uri["missing-field"]["is_political"] is None
    assert by_uri["changed"]["is_political"] is True
    assert "politics_score" not in scores.columns


def test_attach_scores_retains_unknown_candidates_and_candidate_order(popularity_module):
    political_uri = _uri("political")
    non_political_uri = _uri("non-political")
    unknown_uri = _uri("unknown")
    candidates, _ = popularity_module.build_candidate_frame(
        [
            {"at_uri": political_uri, "generator_name": "popularity"},
            {"at_uri": non_political_uri, "generator_name": "popularity"},
            {"at_uri": unknown_uri, "generator_name": "popularity"},
        ],
        3,
        "small",
    )
    scores = pl.DataFrame(
        [
            {
                "at_uri": political_uri,
                "news_social_concern_score": 0.9,
                "inference_indexed_at": "2026-08-01T00:00:00Z",
                "is_political": True,
            },
            {
                "at_uri": non_political_uri,
                "news_social_concern_score": 0.2,
                "inference_indexed_at": "2026-08-01T00:00:00Z",
                "is_political": False,
            },
        ],
        schema=popularity_module.INFERENCE_SCORE_SCHEMA,
    )

    output = popularity_module.attach_results(
        candidates,
        scores,
        pl.DataFrame(schema=popularity_module.MODEL_SCORE_SCHEMA),
    )

    assert output["at_uri"].to_list() == [
        political_uri,
        non_political_uri,
        unknown_uri,
    ]
    assert output["news_social_concern_score"].to_list() == [0.9, 0.2, None]
    assert output["inference_indexed_at"].to_list() == [
        "2026-08-01T00:00:00Z",
        "2026-08-01T00:00:00Z",
        None,
    ]
    assert output["is_political"].to_list() == [True, False, None]
    assert output["model_scored"].to_list() == [False, False, False]
    assert output["scoring_status"].to_list() == [
        "unscored",
        "unscored",
        "unscored",
    ]


def test_score_distribution_reports_quantiles_missing_and_non_finite(popularity_module):
    distribution = popularity_module.score_distribution(
        [0.0, 0.25, 0.5, 0.75, 1.0, None, float("nan")]
    )

    assert distribution["total_candidates"] == 7
    assert distribution["finite_score_count"] == 5
    assert distribution["missing_score_count"] == 1
    assert distribution["non_finite_score_count"] == 1
    assert distribution["mean"] == pytest.approx(0.5)
    assert distribution["population_stddev"] == pytest.approx(2 ** 0.5 / 4)
    assert distribution["min"] == 0.0
    assert distribution["p10"] == pytest.approx(0.1)
    assert distribution["p25"] == 0.25
    assert distribution["median"] == 0.5
    assert distribution["p75"] == 0.75
    assert distribution["p90"] == pytest.approx(0.9)
    assert distribution["p95"] == pytest.approx(0.95)
    assert distribution["p99"] == pytest.approx(0.99)
    assert distribution["max"] == 1.0


def test_inference_score_summary(popularity_module):
    frame = pl.DataFrame({
        "candidate_rank": range(1, 26),
        "inference_indexed_at": ["2026-08-01T00:00:00Z"] * 20 + [None] * 5,
        "news_social_concern_score": [0.8] * 5 + [0.2] * 15 + [None] * 5,
        "is_political": [True] * 5 + [False] * 15 + [None] * 5,
    })

    overall = popularity_module.inference_score_summary(frame)
    assert overall["total_candidates"] == 25
    assert overall["matched_inference_candidates"] == 20
    assert overall["candidates_without_inference"] == 5
    assert overall["inference_coverage"] == 0.8
    assert overall["political_candidates"] == 5
    assert overall["political_share_known_labels"] == 0.25
    assert overall["political_share_all_candidates_lower_bound"] == 0.2
    assert overall["news_social_concern_score"]["mean"] == pytest.approx(0.35)


def test_zero_coverage_and_empty_score_summaries(popularity_module):
    unknown = pl.DataFrame({
        "inference_indexed_at": pl.Series([None, None], dtype=pl.String),
        "news_social_concern_score": pl.Series([None, None], dtype=pl.Float64),
        "is_political": pl.Series([None, None], dtype=pl.Boolean),
    })
    empty = unknown.head(0)

    unknown_summary = popularity_module.inference_score_summary(unknown)
    empty_summary = popularity_module.inference_score_summary(empty)
    assert unknown_summary["total_candidates"] == 2
    assert unknown_summary["matched_inference_candidates"] == 0
    assert unknown_summary["candidates_without_inference"] == 2
    assert unknown_summary["inference_coverage"] == 0.0
    assert unknown_summary["news_social_concern_score"]["finite_score_count"] == 0
    assert unknown_summary["news_social_concern_score"]["mean"] is None
    assert empty_summary["inference_coverage"] is None


def test_pool_local_ranks_and_eight_split_metrics(popularity_module):
    uris = {name: _uri(name) for name in "abcdef"}
    small, _ = popularity_module.build_candidate_frame(
        [
            {"at_uri": uris["a"], "score": 1.0},
            {"at_uri": uris["b"], "score": 5.0},
            {"at_uri": uris["c"], "score": 4.0},
            {"at_uri": uris["d"], "score": 3.0},
            {"at_uri": uris["e"], "score": 2.0},
        ],
        5,
        "small",
    )
    large, _ = popularity_module.build_candidate_frame(
        [
            {"at_uri": uris["a"], "score": 10.0},
            {"at_uri": uris["b"], "score": 7.0},
            {"at_uri": uris["c"], "score": 8.0},
            {"at_uri": uris["f"], "score": 9.0},
        ],
        9,
        "large",
    )
    candidates = pl.concat([small, large], how="vertical")
    inference_scores = pl.DataFrame(
        {
            "at_uri": [
                uris["a"],
                uris["b"],
                uris["d"],
                uris["e"],
                uris["f"],
            ],
            "news_social_concern_score": [1.0, 0.2, 0.8, 0.0, 0.6],
            "inference_indexed_at": ["2026-08-01T00:00:00Z"] * 5,
            "is_political": [True, False, True, False, True],
        },
        schema=popularity_module.INFERENCE_SCORE_SCHEMA,
    )
    logits = {
        "a": (5.0, 1.0),
        "b": (4.0, 2.0),
        "d": (3.0, 3.0),
        "e": (2.0, 4.0),
        "f": (1.0, 5.0),
    }
    model_rows = []
    for index, (name, (model_a_logit, model_b_logit)) in enumerate(logits.items()):
        model_rows.append({
            "at_uri": uris[name],
            "scoring_author_did": f"did:plc:es-{index}",
            "created_at": "2026-08-01T00:00:00Z",
            "elasticsearch_indexed_at": "2026-08-01T00:01:00Z",
            "scoring_content": f"es-{index}",
            "current_like_count": index,
            "model_a_logit": model_a_logit,
            "model_b_logit": model_b_logit,
        })
    output = popularity_module.attach_results(
        candidates,
        inference_scores,
        popularity_module.build_model_scores_frame(model_rows),
    )
    metrics = popularity_module.build_split_metrics(output, top_k=3)

    assert output["candidate_pool"].to_list() == ["small"] * 5 + ["large"] * 4
    small_output = output.filter(pl.col("candidate_pool") == "small")
    large_output = output.filter(pl.col("candidate_pool") == "large")
    assert small_output["candidate_score_rank"].to_list() == [5, 1, 2, 3, 4]
    assert large_output["candidate_score_rank"].to_list() == [1, 4, 3, 2]
    assert small_output["model_a_rank"].to_list() == [1, 2, None, 3, 4]
    assert large_output["model_a_rank"].to_list() == [1, 2, None, 3]
    assert len(metrics) == 8
    assert [row["split_key"] for row in metrics] == list(
        popularity_module.SPLIT_KEYS
    ) * 2
    assert [row["candidate_pool"] for row in metrics] == ["small"] * 4 + [
        "large"
    ] * 4

    by_key = {
        (row["candidate_pool"], row["split_key"]): row for row in metrics
    }
    small_full = by_key[("small", "full_pool")]
    small_candidate = by_key[("small", "candidate_score")]
    small_model_a = by_key[("small", "model_a")]
    small_model_b = by_key[("small", "model_b")]
    assert small_full["selected_count"] == 5
    assert small_full["known_label_count"] == 4
    assert small_full["political_share"] == 0.5
    assert small_full["ndcg"] is None
    assert small_candidate["selected_count"] == 3
    assert small_candidate["known_label_count"] == 2
    assert small_candidate["political_share"] == 0.5
    assert small_candidate["news_social_concern_mean"] == pytest.approx(0.5)
    assert small_candidate["news_social_concern_median"] == pytest.approx(0.5)
    assert small_candidate["ndcg"] == pytest.approx(
        popularity_module.graded_ndcg([0.2, 0.8, 0.0, 1.0], 3)
    )
    assert small_candidate["ndcg_known_label_count"] == 4
    assert small_candidate["ndcg_actual_cutoff"] == 3
    assert small_model_a["selected_count"] == 3
    assert small_model_a["political_share"] == pytest.approx(2 / 3)
    assert small_model_a["ndcg"] == pytest.approx(
        popularity_module.graded_ndcg([1.0, 0.2, 0.8, 0.0], 3)
    )
    assert small_model_b["political_share"] == pytest.approx(1 / 3)

    large_model_a = by_key[("large", "model_a")]
    assert large_model_a["ranking_population_count"] == 3
    assert large_model_a["selected_count"] == 3
    assert large_model_a["ndcg_actual_cutoff"] == 3

    capped = popularity_module.build_split_metrics(output, top_k=50)
    capped_by_key = {
        (row["candidate_pool"], row["split_key"]): row for row in capped
    }
    assert capped_by_key[("small", "candidate_score")]["selected_count"] == 5
    assert capped_by_key[("small", "model_a")]["selected_count"] == 4
    assert capped_by_key[("large", "model_b")]["selected_count"] == 3


def test_model_rank_ties_are_ordered_by_at_uri(popularity_module):
    high_uri = _uri("z")
    low_uri = _uri("a")
    rows = [
        {
            "at_uri": uri,
            "scoring_author_did": None,
            "created_at": None,
            "elasticsearch_indexed_at": None,
            "scoring_content": "content",
            "current_like_count": 0,
            "model_a_logit": 1.0,
            "model_b_logit": 1.0,
        }
        for uri in (high_uri, low_uri)
    ]

    candidates, _ = popularity_module.build_candidate_frame(
        [
            {"at_uri": high_uri, "score": 1.0},
            {"at_uri": low_uri, "score": 1.0},
        ],
        2,
        "small",
    )
    output = popularity_module.attach_results(
        candidates,
        pl.DataFrame(schema=popularity_module.INFERENCE_SCORE_SCHEMA),
        popularity_module.build_model_scores_frame(rows),
    )

    assert output["candidate_score_rank"].to_list() == [2, 1]
    assert output["model_a_rank"].to_list() == [2, 1]
    assert output["model_b_rank"].to_list() == [2, 1]


def test_default_output_directory(popularity_module, monkeypatch, tmp_path):
    monkeypatch.setattr(popularity_module, "DEFAULT_OUTPUT_ROOT", tmp_path)

    output_dir = popularity_module.create_output_dir(
        None,
        datetime(2026, 8, 12, 18, 30, 45, tzinfo=timezone.utc),
    )

    assert output_dir == tmp_path / "popularity_political_20260812_183045"
    assert output_dir.is_dir()


def test_empty_api_result_writes_empty_artifacts_without_listing_gcs(
    popularity_module, monkeypatch, tmp_path
):
    _patch_fake_models(popularity_module, monkeypatch)
    session = FakeSession(FakeResponse({"candidates": []}))
    monkeypatch.setattr(popularity_module.requests, "Session", lambda: session)

    class StorageClient:
        def __init__(self):
            raise AssertionError("GCS should not be accessed for an empty candidate result")

    monkeypatch.setattr(popularity_module.storage, "Client", StorageClient)
    args = _args(popularity_module, tmp_path)
    request_time = datetime(2026, 8, 12, 18, 30, tzinfo=timezone.utc)

    output_dir = asyncio.run(
        popularity_module.run(
            args,
            {"GE_API_KEY": "top-secret"},
            request_time=request_time,
        )
    )

    frame = pl.read_parquet(
        output_dir / "popularity_candidates_with_inference_scores.parquet"
    )
    summary_text = (output_dir / "summary.json").read_text()
    summary = json.loads(summary_text)
    assert frame.schema == pl.Schema(popularity_module.OUTPUT_SCHEMA)
    assert frame.is_empty()
    assert [call["json"]["num_candidates"] for call in session.calls] == [100, 1000]
    assert summary["api_candidates"]["small"]["api_candidates_returned"] == 0
    assert summary["api_candidates"]["large"]["api_candidates_returned"] == 0
    assert summary["inference_export_window"]["files_scanned"] == 0
    assert summary["pool_summaries"]["small"]["matched_inference_candidates"] == 0
    assert summary["pool_summaries"]["large"]["news_social_concern_score"][
        "mean"
    ] is None
    assert len(summary["split_metrics"]) == 8
    assert (output_dir / "split_comparison.csv").is_file()
    assert (output_dir / "split_comparison.png").stat().st_size > 0
    assert summary["run_config"]["political_score_field"] == (
        "topic.News & Social Concern"
    )
    assert "top-secret" not in summary_text


def test_nonempty_api_result_with_no_inference_files_is_all_unknown(
    popularity_module, monkeypatch, tmp_path
):
    _patch_fake_models(popularity_module, monkeypatch)
    at_uri = _uri("one")
    session = FakeSession(FakeResponse({
        "candidates": [
            {
                "at_uri": at_uri,
                "score": 1.2,
                "generator_name": "popularity",
            }
        ],
    }))
    monkeypatch.setattr(popularity_module.requests, "Session", lambda: session)

    class StorageClient:
        def list_blobs(self, bucket, prefix):
            return []

    monkeypatch.setattr(popularity_module.storage, "Client", StorageClient)
    async def fetch_elasticsearch_batch(*args, **kwargs):
        diagnostics = args[-1]
        diagnostics.requested_uris += 1
        diagnostics.missing_documents += 1
        return []

    monkeypatch.setattr(
        popularity_module.bst_comparison,
        "fetch_elasticsearch_batch",
        fetch_elasticsearch_batch,
    )
    args = _args(popularity_module, tmp_path)

    output_dir = asyncio.run(
        popularity_module.run(
            args,
            {"GE_API_KEY": "top-secret"},
            request_time=datetime(2026, 8, 12, 18, 30, tzinfo=timezone.utc),
        )
    )

    frame = pl.read_parquet(
        output_dir / "popularity_candidates_with_inference_scores.parquet"
    )
    summary = json.loads((output_dir / "summary.json").read_text())
    assert frame["candidate_pool"].to_list() == ["small", "large"]
    assert frame["news_social_concern_score"].to_list() == [None, None]
    assert frame["model_scored"].to_list() == [False, False]
    assert frame["model_a_rank"].to_list() == [None, None]
    assert summary["pool_summaries"]["small"]["candidates_without_inference"] == 1
    assert summary["pool_summaries"]["large"]["news_social_concern_score"][
        "finite_score_count"
    ] == 0


def test_run_retains_unscored_candidates_and_writes_complete_comparison(
    popularity_module, monkeypatch, tmp_path, capsys
):
    _patch_fake_models(popularity_module, monkeypatch)
    uris = [_uri(name) for name in ("political", "missing-es", "unknown")]
    small_candidates = [
        {
            "at_uri": uris[0],
            "score": 2.123456,
            "content": "small-api-0",
            "author_did": "did:plc:small-api-0",
            "like_count": 0,
            "generator_name": "popularity",
        },
        {
            "at_uri": uris[1],
            "score": 1.0,
            "content": "small-api-1",
            "author_did": "did:plc:small-api-1",
            "like_count": 1,
            "generator_name": "popularity",
        },
    ]
    large_candidates = [
        {
            "at_uri": uri,
            "score": 3.123456 - index,
            "content": f"large-api-{index}",
            "author_did": f"did:plc:large-api-{index}",
            "like_count": index,
            "generator_name": "popularity",
        }
        for index, uri in enumerate(uris)
    ]
    session = FakeSession([
        FakeResponse({"candidates": small_candidates}),
        FakeResponse({"candidates": large_candidates}),
    ])
    monkeypatch.setattr(popularity_module.requests, "Session", lambda: session)

    inference_path = tmp_path / "inferences.parquet"
    pl.DataFrame({
        "at_uri": uris[:2],
        "indexed_at": ["2026-08-01T00:00:00Z"] * 2,
        "inferences": [
            _inference_json(news=0.987654),
            _inference_json(news=0.123456),
        ],
    }).write_parquet(inference_path)
    monkeypatch.setattr(popularity_module.storage, "Client", lambda: object())
    monkeypatch.setattr(
        popularity_module,
        "list_inference_parquet_paths",
        lambda client, bucket, prefix, window: [str(inference_path)],
    )

    hydrated_requests = []

    async def fetch_elasticsearch_batch(
        client,
        elasticsearch_url,
        elasticsearch_index,
        api_key,
        at_uris,
        embedding_field,
        expected_dim,
        diagnostics,
    ):
        hydrated_requests.append(list(at_uris))
        diagnostics.elasticsearch_batches += 1
        diagnostics.requested_uris += len(at_uris)
        diagnostics.documents_found += 2
        diagnostics.missing_documents += 1
        diagnostics.hydrated_posts += 2
        return [
            popularity_module.bst_comparison.HydratedPost(
                at_uri=uri,
                author_did=f"did:plc:es-{index}",
                created_at="2026-08-01T00:00:00Z",
                indexed_at="2026-08-01T00:01:00Z",
                content=f"es-{index}",
                like_count=10 + index,
                embedding=[float(index), 1.0],
            )
            for index, uri in ((0, uris[0]), (2, uris[2]))
        ]

    def score_model_batch(model, candidate_embeddings, candidates, device):
        return np.asarray([3.0, 1.0] if model.label == "A" else [1.0, 3.0])

    monkeypatch.setattr(
        popularity_module.bst_comparison,
        "fetch_elasticsearch_batch",
        fetch_elasticsearch_batch,
    )
    monkeypatch.setattr(
        popularity_module.bst_comparison,
        "score_model_batch",
        score_model_batch,
    )
    output_dir = tmp_path / "output"
    args = _args(
        popularity_module,
        output_dir,
        small_num_candidates=2,
        large_num_candidates=3,
        top_k=2,
    )

    actual_output_dir = asyncio.run(
        popularity_module.run(
            args,
            {"GE_API_KEY": "api-secret"},
            request_time=datetime(2026, 8, 12, 18, 30, tzinfo=timezone.utc),
        )
    )

    frame = pl.read_parquet(
        actual_output_dir / "popularity_candidates_with_inference_scores.parquet"
    )
    summary_text = (actual_output_dir / "summary.json").read_text()
    summary = json.loads(summary_text)
    assert [call["json"]["num_candidates"] for call in session.calls] == [2, 3]
    assert hydrated_requests == [uris]
    assert frame["candidate_pool"].to_list() == ["small", "small"] + [
        "large"
    ] * 3
    assert frame["at_uri"].to_list() == uris[:2] + uris
    assert frame["content"].to_list() == [
        "es-0",
        "small-api-1",
        "es-0",
        "large-api-1",
        "es-2",
    ]
    assert frame["current_like_count"].to_list() == [10, None, 10, None, 12]
    assert frame["is_political"].to_list() == [True, False, True, False, None]
    assert frame["model_scored"].to_list() == [True, False, True, False, True]
    assert frame["model_a_rank"].to_list() == [1, None, 1, None, 2]
    assert frame["model_b_rank"].to_list() == [1, None, 2, None, 1]
    assert frame["popularity_score"].to_list()[2] == pytest.approx(3.123456)
    assert summary["retrieval_diagnostics"]["scored_posts"] == 2
    assert len(summary["split_metrics"]) == 8
    assert summary["run_config"]["small_num_candidates"] == 2
    assert summary["run_config"]["large_num_candidates"] == 3
    assert summary["run_config"]["top_k"] == 2
    small_full = summary["split_metrics"][0]
    assert small_full["news_social_concern_mean"] == pytest.approx(0.555555)

    csv_path = actual_output_dir / "split_comparison.csv"
    with csv_path.open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 8
    assert csv_rows[0]["Political share"] == "50.00%"
    assert csv_rows[0]["NDCG"] == "n/a"
    assert csv_rows[0]["Mean"] == "0.56"
    assert csv_rows[0]["Median"] == "0.56"
    png_path = actual_output_dir / "split_comparison.png"
    assert png_path.is_file()
    assert png_path.stat().st_size > 0

    console = capsys.readouterr().out
    assert "50.00%" in console
    assert "0.56" in console
    assert "Top 10" not in console
    assert "api-secret" not in summary_text
    assert "es-secret" not in summary_text
