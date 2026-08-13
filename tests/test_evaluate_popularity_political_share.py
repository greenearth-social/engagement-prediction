import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

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


def _args(module, output_dir, **overrides):
    values = {
        "num_candidates": 100,
        "max_age_hours": 168,
        "api_timeout_seconds": 30.0,
        "gcs_bucket": "bucket",
        "inference_prefix": "bsky_inferences",
        "output_dir": output_dir,
    }
    values.update(overrides)
    return module.argparse.Namespace(**values)


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
        return self.response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_cli_defaults(popularity_module):
    args = popularity_module.build_arg_parser().parse_args([])

    assert args.num_candidates == 100
    assert args.max_age_hours == 168
    assert args.api_timeout_seconds == 30.0
    assert args.gcs_bucket == popularity_module.DEFAULT_GCS_BUCKET
    assert args.inference_prefix == "bsky_inferences"
    assert args.output_dir is None


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


@pytest.mark.parametrize("num_candidates", [0, 1001])
def test_candidate_count_validation(popularity_module, num_candidates):
    with pytest.raises(ValueError, match="between 1 and 1000"):
        popularity_module.validate_parameters(num_candidates, 168, 30.0)


def test_other_parameter_validation(popularity_module):
    with pytest.raises(ValueError, match="must be one of"):
        popularity_module.validate_parameters(100, 100, 30.0)
    with pytest.raises(ValueError, match="must be positive"):
        popularity_module.validate_parameters(100, 168, 0.0)


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


def test_candidate_frame_preserves_api_rank_and_diagnoses_bad_uris(popularity_module):
    candidates = [
        {
            "at_uri": "at://post/one",
            "score": 3,
            "author_did": "did:one",
            "content": "one",
            "like_count": 12,
            "generator_name": "popularity",
        },
        {"at_uri": None, "score": 2},
        {"at_uri": "at://post/one", "score": 1},
        {
            "at_uri": "at://post/two",
            "score": float("nan"),
            "like_count": "9",
        },
    ]

    frame, diagnostics = popularity_module.build_candidate_frame(candidates, 100)

    assert frame["candidate_rank"].to_list() == [1, 4]
    assert frame["at_uri"].to_list() == ["at://post/one", "at://post/two"]
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
    )

    by_uri = {row["at_uri"]: row for row in scores.iter_rows(named=True)}
    assert set(by_uri) == {"political", "one-low", "missing-field", "changed"}
    assert by_uri["political"]["politics_score"] == 0.8
    assert by_uri["political"]["news_social_concern_score"] == 0.8
    assert by_uri["one-low"]["politics_score"] == 0.9
    assert by_uri["one-low"]["news_social_concern_score"] == 0.79
    assert by_uri["missing-field"]["politics_score"] == 0.95
    assert by_uri["missing-field"]["news_social_concern_score"] is None
    assert by_uri["changed"]["politics_score"] == 0.91
    assert by_uri["changed"]["news_social_concern_score"] == 0.92
    assert by_uri["changed"]["inference_indexed_at"] == "2026-08-02T00:00:00Z"


def test_attach_scores_retains_unknown_candidates_and_candidate_order(popularity_module):
    candidates, _ = popularity_module.build_candidate_frame(
        [
            {"at_uri": "political", "generator_name": "popularity"},
            {"at_uri": "non-political", "generator_name": "popularity"},
            {"at_uri": "unknown", "generator_name": "popularity"},
        ],
        3,
    )
    scores = pl.DataFrame(
        [
            {
                "at_uri": "political",
                "politics_score": 0.9,
                "news_social_concern_score": 0.9,
                "inference_indexed_at": "2026-08-01T00:00:00Z",
            },
            {
                "at_uri": "non-political",
                "politics_score": 0.9,
                "news_social_concern_score": 0.2,
                "inference_indexed_at": "2026-08-01T00:00:00Z",
            },
        ],
        schema=popularity_module.INFERENCE_SCORE_SCHEMA,
    )

    output = popularity_module.attach_inference_scores(candidates, scores)

    assert output["at_uri"].to_list() == [
        "political",
        "non-political",
        "unknown",
    ]
    assert output["politics_score"].to_list() == [0.9, 0.9, None]
    assert output["news_social_concern_score"].to_list() == [0.9, 0.2, None]
    assert output["inference_indexed_at"].to_list() == [
        "2026-08-01T00:00:00Z",
        "2026-08-01T00:00:00Z",
        None,
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


def test_joint_distribution_reports_relationship_and_bivariate_bins(popularity_module):
    joint = popularity_module.joint_score_distribution(
        [-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, None],
        [1.1, 0.9, 0.7, 0.5, 0.3, 0.1, 1.0, -0.1, 0.5],
    )

    assert joint["finite_pair_count"] == 8
    assert joint["pearson_correlation"] == pytest.approx(-0.673202614379085)
    assert joint["population_covariance"] == pytest.approx(-0.11265625)
    assert joint["mean_politics_minus_news_social_concern"] == pytest.approx(0.0)
    assert joint["mean_absolute_score_difference"] == pytest.approx(0.6)
    assert joint["minimum_of_two_scores"]["median"] == pytest.approx(0.2)
    assert joint["maximum_of_two_scores"]["median"] == pytest.approx(0.9)
    histogram = joint["bivariate_histogram"]
    assert sum(sum(row) for row in histogram["counts"]) == 8
    assert histogram["counts"][0][-1] == 1
    assert histogram["counts"][-1][0] == 1
    assert histogram["counts"][-2][-2] == 1
    assert sum(sum(row) for row in histogram["shares_of_finite_pairs"]) == 1.0


def test_joint_histogram_bin_boundaries(popularity_module):
    values = [-0.1, 0.0, 0.199, 0.2, 0.399, 0.4, 0.599, 0.6, 0.799, 0.8, 1.0, 1.1]

    joint = popularity_module.joint_score_distribution(values, values)

    counts = joint["bivariate_histogram"]["counts"]
    assert [counts[index][index] for index in range(7)] == [1, 2, 2, 2, 2, 2, 1]
    assert sum(sum(row) for row in counts) == len(values)


def test_inference_score_and_top_cutoff_summaries(popularity_module):
    frame = pl.DataFrame({
        "candidate_rank": range(1, 26),
        "inference_indexed_at": ["2026-08-01T00:00:00Z"] * 20 + [None] * 5,
        "politics_score": [0.9] * 5 + [0.1] * 15 + [None] * 5,
        "news_social_concern_score": [0.8] * 5 + [0.2] * 15 + [None] * 5,
    })

    overall = popularity_module.inference_score_summary(frame)
    top = popularity_module.top_cutoff_summaries(frame)

    assert overall["total_candidates"] == 25
    assert overall["matched_inference_candidates"] == 20
    assert overall["candidates_without_inference"] == 5
    assert overall["inference_coverage"] == 0.8
    assert overall["politics_score"]["finite_score_count"] == 20
    assert overall["politics_score"]["mean"] == pytest.approx(0.3)
    assert overall["news_social_concern_score"]["mean"] == pytest.approx(0.35)
    assert overall["joint_distribution"]["finite_pair_count"] == 20
    assert top["10"]["actual_cutoff"] == 10
    assert top["10"]["matched_inference_candidates"] == 10
    assert top["10"]["politics_score"]["mean"] == pytest.approx(0.5)
    assert top["10"]["news_social_concern_score"]["mean"] == pytest.approx(0.5)
    assert top["100"]["actual_cutoff"] == 25
    assert top["100"]["politics_score"]["mean"] == pytest.approx(0.3)


def test_zero_coverage_and_empty_score_summaries(popularity_module):
    unknown = pl.DataFrame({
        "inference_indexed_at": pl.Series([None, None], dtype=pl.String),
        "politics_score": pl.Series([None, None], dtype=pl.Float64),
        "news_social_concern_score": pl.Series([None, None], dtype=pl.Float64),
    })
    empty = unknown.head(0)

    unknown_summary = popularity_module.inference_score_summary(unknown)
    empty_summary = popularity_module.inference_score_summary(empty)
    assert unknown_summary["total_candidates"] == 2
    assert unknown_summary["matched_inference_candidates"] == 0
    assert unknown_summary["candidates_without_inference"] == 2
    assert unknown_summary["inference_coverage"] == 0.0
    assert unknown_summary["politics_score"]["finite_score_count"] == 0
    assert unknown_summary["politics_score"]["mean"] is None
    assert unknown_summary["joint_distribution"]["finite_pair_count"] == 0
    assert unknown_summary["joint_distribution"]["pearson_correlation"] is None
    assert empty_summary["inference_coverage"] is None


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
    session = FakeSession(FakeResponse({"candidates": []}))
    monkeypatch.setattr(popularity_module.requests, "Session", lambda: session)

    class StorageClient:
        def __init__(self):
            raise AssertionError("GCS should not be accessed for an empty candidate result")

    monkeypatch.setattr(popularity_module.storage, "Client", StorageClient)
    args = _args(popularity_module, tmp_path)
    request_time = datetime(2026, 8, 12, 18, 30, tzinfo=timezone.utc)

    output_dir = popularity_module.run(
        args,
        {"GE_API_KEY": "top-secret"},
        request_time=request_time,
    )

    frame = pl.read_parquet(
        output_dir / "popularity_candidates_with_inference_scores.parquet"
    )
    summary_text = (output_dir / "summary.json").read_text()
    summary = json.loads(summary_text)
    assert frame.schema == pl.Schema(popularity_module.OUTPUT_SCHEMA)
    assert frame.is_empty()
    assert summary["api_candidates"]["api_candidates_returned"] == 0
    assert summary["inference_export_window"]["files_scanned"] == 0
    assert summary["overall"]["matched_inference_candidates"] == 0
    assert summary["overall"]["politics_score"]["mean"] is None
    assert "top-secret" not in summary_text


def test_nonempty_api_result_with_no_inference_files_is_all_unknown(
    popularity_module, monkeypatch, tmp_path
):
    session = FakeSession(FakeResponse({
        "candidates": [
            {
                "at_uri": "at://post/one",
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
    args = _args(popularity_module, tmp_path)

    output_dir = popularity_module.run(
        args,
        {"GE_API_KEY": "top-secret"},
        request_time=datetime(2026, 8, 12, 18, 30, tzinfo=timezone.utc),
    )

    frame = pl.read_parquet(
        output_dir / "popularity_candidates_with_inference_scores.parquet"
    )
    summary = json.loads((output_dir / "summary.json").read_text())
    assert frame["politics_score"].to_list() == [None]
    assert frame["news_social_concern_score"].to_list() == [None]
    assert summary["overall"]["candidates_without_inference"] == 1
    assert summary["overall"]["politics_score"]["finite_score_count"] == 0
    assert summary["overall"]["joint_distribution"]["finite_pair_count"] == 0
