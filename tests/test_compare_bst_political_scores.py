import argparse
import asyncio
import importlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import polars as pl
import pytest
import torch


@pytest.fixture(scope="module")
def comparison_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "ops/compare_bst_political_scores.py"
    spec = importlib.util.spec_from_file_location(
        "compare_bst_political_scores", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_bst_political_scores"] = module
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


def _model_bundle(module, label, ranker, author_map=None):
    max_history_len = 3
    embed_dim = 2
    return module.ModelBundle(
        label=label,
        model_dir=Path(f"/tmp/model-{label.lower()}"),
        training_config={},
        ranker=ranker,
        author_idx_by_did=author_map or {},
        author_idx_path=None,
        post_embedding_dim=embed_dim,
        max_history_len=max_history_len,
        use_popularity_feature=True,
        history_embeddings=torch.zeros((1, max_history_len, embed_dim)),
        history_mask=torch.zeros((1, max_history_len), dtype=torch.bool),
        history_time_deltas_hours=torch.zeros((1, max_history_len)),
        history_author_indices=torch.zeros((1, max_history_len), dtype=torch.long),
        history_prior_cumulative_likes=torch.zeros((1, max_history_len)),
    )


def _hydrated_post(module, at_uri, author_did, like_count, embedding):
    return module.HydratedPost(
        at_uri=at_uri,
        author_did=author_did,
        created_at="2026-08-01T00:00:00Z",
        indexed_at="2026-08-01T00:01:00Z",
        content="content",
        like_count=like_count,
        embedding=embedding,
    )


def test_cli_uses_positional_model_directories_and_expected_defaults(comparison_module):
    parser = comparison_module.build_arg_parser()
    args = parser.parse_args(
        [
            "model-a",
            "model-b",
            "--start-date",
            "2026-08-07",
            "--end-date",
            "2026-08-10",
        ]
    )

    assert args.model_a_dir == Path("model-a")
    assert args.model_b_dir == Path("model-b")
    assert args.start_date == "2026-08-07"
    assert args.end_date == "2026-08-10"
    assert args.political_threshold == 0.95
    assert args.class_sample_size == 10_000
    assert args.random_seed == 42
    assert args.elasticsearch_index == "posts_recent"
    assert args.elasticsearch_batch_size == 1000
    assert args.elasticsearch_insecure is True

    secure_args = parser.parse_args(
        [
            "model-a",
            "model-b",
            "--start-date",
            "2026-08-07",
            "--end-date",
            "2026-08-10",
            "--elasticsearch-secure",
        ]
    )
    assert secure_args.elasticsearch_insecure is False

    with pytest.raises(SystemExit):
        parser.parse_args(["model-a", "model-b"])


def test_default_output_directory_includes_bst_prefix_and_timestamp(
    comparison_module, monkeypatch, tmp_path
):
    monkeypatch.setattr(comparison_module, "DEFAULT_OUTPUT_ROOT", tmp_path)
    evaluated_at = datetime(2026, 8, 10, 12, 34, 56, tzinfo=timezone.utc)

    output_dir = comparison_module._create_output_dir(None, evaluated_at)

    assert output_dir == tmp_path / "bst_political_20260810_123456"
    assert output_dir.is_dir()


def test_elasticsearch_cli_values_override_environment_and_url_is_sanitized(
    comparison_module,
):
    url, api_key = comparison_module.resolve_elasticsearch_settings(
        "https://cli-user:cli-pass@example.com:9200/base?token=value",
        "cli-secret",
        {
            "GE_ELASTICSEARCH_URL": "https://env.example.com",
            "GE_ELASTICSEARCH_API_KEY": "env-secret",
        },
    )

    assert url == "https://cli-user:cli-pass@example.com:9200/base?token=value"
    assert api_key == "cli-secret"
    assert comparison_module.sanitize_elasticsearch_url(url) == "https://example.com:9200/base"


def test_elasticsearch_settings_use_environment_and_require_both_values(
    comparison_module,
):
    assert comparison_module.resolve_elasticsearch_settings(
        None,
        None,
        {
            "GE_ELASTICSEARCH_URL": "localhost:9200",
            "GE_ELASTICSEARCH_API_KEY": "env-secret",
        },
    ) == ("https://localhost:9200", "env-secret")

    with pytest.raises(ValueError, match="Elasticsearch URL is required"):
        comparison_module.resolve_elasticsearch_settings(
            None, None, {"GE_ELASTICSEARCH_API_KEY": "secret"}
        )
    with pytest.raises(ValueError, match="Elasticsearch API key is required"):
        comparison_module.resolve_elasticsearch_settings(
            None, None, {"GE_ELASTICSEARCH_URL": "https://localhost:9200"}
        )


def test_date_window_uses_explicit_utc_dates_and_validates_input(comparison_module):
    window = comparison_module.resolve_date_window("2026-08-07", "2026-08-10")
    assert window.start == datetime(2026, 8, 7, tzinfo=timezone.utc)
    assert window.end == datetime(2026, 8, 10, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="--start-date must use YYYY-MM-DD"):
        comparison_module.resolve_date_window("08/07/2026", "2026-08-10")
    with pytest.raises(ValueError, match="--start-date must use YYYY-MM-DD"):
        comparison_module.resolve_date_window("20260807", "2026-08-10")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        comparison_module.resolve_date_window("2026-08-07", "08/10/2026")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        comparison_module.resolve_date_window("2026-08-07", "20260810")
    with pytest.raises(ValueError, match="must be earlier"):
        comparison_module.resolve_date_window("2026-08-10", "2026-08-10")
    with pytest.raises(ValueError, match="must be earlier"):
        comparison_module.resolve_date_window("2026-08-11", "2026-08-10")


def test_inference_file_listing_obeys_export_window(comparison_module):
    class Blob:
        def __init__(self, name):
            self.name = name

    class Client:
        def __init__(self):
            self.calls = []

        def list_blobs(self, bucket, prefix):
            self.calls.append((bucket, prefix))
            return [
                Blob("bsky_inferences_20260806_235959.parquet"),
                Blob("bsky_inferences_20260807_000000.parquet"),
                Blob("bsky_inferences_20260809_235959.parquet"),
                Blob("bsky_inferences_20260810_000000.parquet"),
                Blob("bsky_inferences_bad.parquet"),
            ]

    client = Client()
    window = comparison_module.resolve_date_window("2026-08-07", "2026-08-10")
    paths = comparison_module.list_inference_parquet_paths(
        client, "bucket", "bsky_inferences", window
    )

    assert client.calls == [("bucket", "bsky_inferences")]
    assert paths == [
        "gs://bucket/bsky_inferences_20260807_000000.parquet",
        "gs://bucket/bsky_inferences_20260809_235959.parquet",
    ]


def test_inference_file_listing_fails_when_window_is_empty(comparison_module):
    class Client:
        def list_blobs(self, bucket, prefix):
            return []

    window = comparison_module.resolve_date_window("2026-08-07", "2026-08-10")
    with pytest.raises(FileNotFoundError, match="No inference parquet files"):
        comparison_module.list_inference_parquet_paths(
            Client(), "bucket", "bsky_inferences", window
        )


def test_political_extraction_uses_news_threshold_missing_as_zero_and_latest_row(
    comparison_module, tmp_path
):
    path = tmp_path / "inferences.parquet"
    pl.DataFrame(
        {
            "at_uri": [
                "news-boundary",
                "politics-only",
                "missing-field",
                "became-nonpolitical",
                "became-nonpolitical",
                "became-political",
                "became-political",
                "null-indexed-at",
                "null-indexed-at",
            ],
            "indexed_at": [
                "2026-08-01T00:00:00Z",
                "2026-08-01T00:00:00Z",
                "2026-08-01T00:00:00Z",
                "2026-08-01T00:00:00Z",
                "2026-08-02T00:00:00Z",
                "2026-08-01T00:00:00+00:00",
                "2026-08-02T00:00:00+00:00",
                None,
                "2026-08-03T00:00:00Z",
            ],
            "inferences": [
                _inference_json(0.1, 0.8),
                _inference_json(0.9, 0.79),
                _inference_json(0.9, None),
                _inference_json(0.95, 0.95),
                _inference_json(0.4, 0.4),
                _inference_json(0.4, 0.4),
                _inference_json(0.91, 0.92),
                _inference_json(0.99, 0.99),
                _inference_json(0.85, 0.86),
            ],
        }
    ).write_parquet(path)

    result, stats = comparison_module.build_evaluation_posts_df(
        [str(path)], 0.8, 10, 42
    )

    political_result = result.filter(pl.col("is_political"))
    assert political_result["at_uri"].to_list() == [
        "became-political",
        "news-boundary",
        "null-indexed-at",
    ]
    by_uri = {row["at_uri"]: row for row in result.iter_rows(named=True)}
    assert by_uri["news-boundary"]["news_social_concern_score"] == 0.8
    assert by_uri["became-political"]["news_social_concern_score"] == 0.92
    assert by_uri["null-indexed-at"]["news_social_concern_score"] == 0.86
    assert "politics_score" not in result.columns
    assert political_result.height == 3
    assert result.filter(~pl.col("is_political")).height == 3
    assert stats == {
        "unique_inference_uris": 6,
        "class_sample_size_requested": 10,
        "class_sample_size_selected": 3,
        "political_uris_available": 3,
        "political_uris_selected": 3,
        "non_political_uris_available": 3,
        "non_political_uris_selected": 3,
        "evaluation_uris": 6,
    }


def test_class_sampling_is_balanced_and_deterministic(
    comparison_module, tmp_path
):
    path = tmp_path / "sampling_inferences.parquet"
    political_uris = [f"political-{index}" for index in range(20)]
    non_political_uris = [f"non-political-{index}" for index in range(20)]
    pl.DataFrame(
        {
            "at_uri": political_uris + non_political_uris,
            "indexed_at": ["2026-08-01T00:00:00Z"] * 40,
            "inferences": (
                [_inference_json(0.9, 0.9)] * 20
                + [_inference_json(0.2, 0.3)] * 20
            ),
        }
    ).write_parquet(path)

    first, first_stats = comparison_module.build_evaluation_posts_df(
        [str(path)], 0.8, 10, 123
    )
    second, second_stats = comparison_module.build_evaluation_posts_df(
        [str(path)], 0.8, 10, 123
    )

    assert first.to_dicts() == second.to_dicts()
    assert first_stats == second_stats == {
        "unique_inference_uris": 40,
        "class_sample_size_requested": 10,
        "class_sample_size_selected": 10,
        "political_uris_available": 20,
        "political_uris_selected": 10,
        "non_political_uris_available": 20,
        "non_political_uris_selected": 10,
        "evaluation_uris": 20,
    }
    assert first.filter(pl.col("is_political")).height == 10
    assert first.filter(~pl.col("is_political")).height == 10

    with pytest.raises(ValueError, match="must be non-negative"):
        comparison_module.build_evaluation_posts_df([str(path)], 0.8, -1, 123)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.request = httpx.Request("POST", "https://example.com/posts/_search")

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            response = httpx.Response(self.status_code, request=self.request)
            raise httpx.HTTPStatusError(
                "failed", request=self.request, response=response
            )


class FakeAsyncClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def post(self, url, json, headers):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self.response


def test_elasticsearch_batch_is_narrow_ordered_and_reports_bad_hits(
    comparison_module,
):
    def hit(uri, vector, **source):
        fields = {} if vector is None else {"embeddings.all_MiniLM_L12_v2": [vector]}
        return {
            "_index": source.pop("_index", "posts-1"),
            "_id": source.pop("_id", uri),
            "_source": {"at_uri": uri, "content": "text", **source},
            "fields": fields,
        }

    response = FakeResponse(
        {
            "hits": {
                "hits": [
                    hit(
                        "valid-duplicate",
                        [0.1, 0.2],
                        indexed_at="2026-08-01T00:00:00Z",
                        author_did="did:old",
                        like_count=1,
                    ),
                    hit("bad-vector", [0.1], author_did="did:bad", like_count=2),
                    hit("no-vector", None, author_did="did:none", like_count=3),
                    hit(
                        "valid-duplicate",
                        [0.3, 0.4],
                        indexed_at="2026-08-02T00:00:00Z",
                        author_did="did:new",
                        like_count=4,
                    ),
                    hit("missing-metadata", [0.5, 0.6]),
                    hit("invalid-like", [0.7, 0.8], author_did="did:x", like_count="8"),
                    hit("blank-content", [0.9, 1.0], content="   ", like_count=9),
                ]
            }
        }
    )
    client = FakeAsyncClient(response)
    diagnostics = comparison_module.RetrievalDiagnostics()
    requested = [
        "valid-duplicate",
        "bad-vector",
        "no-vector",
        "missing-metadata",
        "invalid-like",
        "blank-content",
        "missing-document",
    ]

    posts = asyncio.run(
        comparison_module.fetch_elasticsearch_batch(
            client,
            "https://example.com",
            "posts_recent",
            "top-secret",
            requested,
            "embeddings.all_MiniLM_L12_v2",
            2,
            diagnostics,
        )
    )

    assert [post.at_uri for post in posts] == [
        "valid-duplicate",
        "missing-metadata",
        "invalid-like",
    ]
    assert posts[0].author_did == "did:new"
    assert posts[0].like_count == 4
    assert posts[1].author_did is None
    assert posts[1].like_count == 0
    assert posts[2].like_count == 0
    assert diagnostics.elasticsearch_batches == 1
    assert diagnostics.requested_uris == 7
    assert diagnostics.documents_found == 6
    assert diagnostics.missing_documents == 1
    assert diagnostics.duplicate_hits == 1
    assert diagnostics.missing_or_blank_content == 1
    assert diagnostics.missing_embeddings == 1
    assert diagnostics.malformed_embeddings == 1
    assert diagnostics.missing_authors == 1
    assert diagnostics.missing_like_counts == 1
    assert diagnostics.invalid_like_counts == 1
    assert diagnostics.hydrated_posts == 3

    call = client.calls[0]
    assert call["url"] == "https://example.com/posts_recent/_search"
    assert call["headers"]["Authorization"] == "ApiKey top-secret"
    assert call["json"]["query"] == {"terms": {"at_uri": requested}}
    assert call["json"]["size"] == len(requested)
    assert call["json"]["docvalue_fields"] == ["embeddings.all_MiniLM_L12_v2"]
    assert call["json"]["_source"] == [
        "at_uri",
        "author_did",
        "created_at",
        "indexed_at",
        "content",
        "like_count",
    ]


def test_elasticsearch_http_failure_does_not_include_api_key(comparison_module):
    client = FakeAsyncClient(FakeResponse({}, status_code=401))

    with pytest.raises(RuntimeError, match="HTTP 401") as exc_info:
        asyncio.run(
            comparison_module.search_elasticsearch(
                client,
                "https://example.com",
                "posts_recent",
                "top-secret",
                {"query": {"match_all": {}}},
            )
        )
    assert "top-secret" not in str(exc_info.value)


def test_matrix_scoring_uses_empty_history_model_author_map_and_current_likes(
    comparison_module,
):
    class Ranker:
        def __init__(self):
            self.calls = []

        def score_candidate_matrix(self, *args):
            self.calls.append(args)
            return args[3].sum(dim=1).unsqueeze(0) + args[7].unsqueeze(0)

    ranker = Ranker()
    model = _model_bundle(
        comparison_module,
        "A",
        ranker,
        {"did:known": 7},
    )
    candidates = [
        _hydrated_post(comparison_module, "one", "did:known", 10, [1.0, 2.0]),
        _hydrated_post(comparison_module, "two", "did:unknown", 20, [3.0, 4.0]),
        _hydrated_post(comparison_module, "three", None, 30, [5.0, 6.0]),
    ]
    embeddings = torch.tensor([post.embedding for post in candidates])

    scores = comparison_module.score_model_batch(
        model, embeddings, candidates, torch.device("cpu")
    )

    assert scores.tolist() == [13.0, 27.0, 41.0]
    call = ranker.calls[0]
    assert len(call) == 8
    assert call[0].shape == (1, 3, 2)
    assert not call[1].any()
    assert torch.equal(call[2], torch.zeros((1, 3)))
    assert call[4].tolist() == [[0, 0, 0]]
    assert call[5].tolist() == [7, 1, 1]
    assert call[6].tolist() == [[0.0, 0.0, 0.0]]
    assert call[7].tolist() == [10.0, 20.0, 30.0]


def test_score_frame_uses_global_normalization_and_neutral_delta_labels(
    comparison_module,
):
    rows = []
    for at_uri, score_a, score_b, is_political in [
        ("one", 1.0, 3.0, True),
        ("two", 2.0, 2.0, False),
        ("three", 3.0, 1.0, True),
    ]:
        rows.append(
            {
                "at_uri": at_uri,
                "author_did": "did:a",
                "created_at": None,
                "elasticsearch_indexed_at": None,
                "content": "content",
                "current_like_count": 1,
                "news_social_concern_score": 0.9,
                "inference_indexed_at": "2026-08-01T00:00:00Z",
                "is_political": is_political,
                "model_a_logit": score_a,
                "model_b_logit": score_b,
            }
        )

    frame = comparison_module.build_scores_frame(rows)

    assert frame["model_a_normalized_score"].to_list() == [0.0, 0.5, 1.0]
    assert frame["model_b_normalized_score"].to_list() == [1.0, 0.5, 0.0]
    assert frame["model_b_minus_model_a"].to_list() == [2.0, 0.0, -2.0]
    assert frame["model_a_rank"].to_list() == [3, 2, 1]
    assert frame["model_b_rank"].to_list() == [1, 2, 3]
    assert frame["model_a_rank_percentile"].to_list() == [1.0, 0.5, 0.0]
    assert frame["model_b_rank_percentile"].to_list() == [0.0, 0.5, 1.0]
    assert frame["model_b_minus_model_a_rank"].to_list() == [-2, 0, 2]
    assert frame["higher_scoring_model"].to_list() == ["B", "tie", "A"]
    assert comparison_module.normalize_logits([5.0, 5.0]).tolist() == [0.5, 0.5]

    ranking = comparison_module.model_ranking_summary(
        frame, "model_a_logit", "model_a_rank", "model_a_rank_percentile"
    )
    assert ranking["political_average_rank"] == 2.0
    assert ranking["non_political_average_rank"] == 2.0
    assert ranking["political_vs_non_political_roc_auc"] == 0.5
    assert ranking["top_cutoffs"]["100"] == {
        "requested_posts": 100,
        "actual_posts": 3,
        "political_posts": 2,
        "non_political_posts": 1,
        "political_share": pytest.approx(2 / 3),
        "political_recall": 1.0,
        "political_share_lift_vs_evaluation_population": 1.0,
    }

    rank_shift = comparison_module.political_rank_shift_summary(frame)
    assert rank_shift["political_posts_ranked_higher_by_model_b"] == 1
    assert rank_shift["political_posts_ranked_lower_by_model_b"] == 1
    assert rank_shift["political_posts_with_unchanged_rank"] == 0


def test_ranking_summary_reports_top_100_political_share_and_auc(comparison_module):
    rows = []
    for index in range(120):
        is_political = index < 20
        rows.append(
            {
                "at_uri": f"post-{index:03d}",
                "author_did": "did:a",
                "created_at": None,
                "elasticsearch_indexed_at": None,
                "content": "content",
                "current_like_count": 1,
                "news_social_concern_score": 0.9 if is_political else 0.1,
                "inference_indexed_at": "2026-08-01T00:00:00Z",
                "is_political": is_political,
                "model_a_logit": 1000.0 - index if is_political else 500.0 - index,
                "model_b_logit": -1000.0 - index if is_political else 500.0 - index,
            }
        )
    frame = comparison_module.build_scores_frame(rows)

    model_a = comparison_module.model_ranking_summary(
        frame, "model_a_logit", "model_a_rank", "model_a_rank_percentile"
    )
    model_b = comparison_module.model_ranking_summary(
        frame, "model_b_logit", "model_b_rank", "model_b_rank_percentile"
    )

    assert model_a["political_average_rank"] == 10.5
    assert model_b["political_average_rank"] == 110.5
    assert model_a["political_vs_non_political_roc_auc"] == 1.0
    assert model_b["political_vs_non_political_roc_auc"] == 0.0
    assert model_a["top_cutoffs"]["100"]["political_posts"] == 20
    assert model_a["top_cutoffs"]["100"]["political_share"] == 0.2
    assert model_a["top_cutoffs"]["100"]["political_recall"] == 1.0
    assert model_a["top_cutoffs"]["100"][
        "political_share_lift_vs_evaluation_population"
    ] == pytest.approx(1.2)
    assert model_b["top_cutoffs"]["100"]["political_posts"] == 0
    assert model_b["top_cutoffs"]["100"]["political_share"] == 0.0


def test_strict_model_loading_requires_ranker_pt_and_exported_matrix_method(
    comparison_module, tmp_path
):
    model_dir = tmp_path / "model"
    checkpoints_dir = model_dir / "checkpoints"
    get_data_dir = tmp_path / "get-data"
    checkpoints_dir.mkdir(parents=True)
    get_data_dir.mkdir()
    (model_dir / "training_config.json").write_text(
        json.dumps(
            {
                "model_type": "bst-ranker",
                "post_embedding_dim": 2,
                "max_history_len": 3,
                "use_author_embedding_table": True,
            }
        )
    )
    (model_dir / "manifest.json").write_text(
        json.dumps({"inputs": {"01_get_data": str(get_data_dir)}})
    )
    pl.DataFrame({"author_did": ["did:a"], "author_idx": [2]}).write_parquet(
        get_data_dir / "author_idx_test.parquet"
    )

    with pytest.raises(FileNotFoundError, match="ranker.pt"):
        comparison_module.load_model_bundle(
            "A", model_dir, None, torch.device("cpu")
        )

    torch.jit.script(torch.nn.Linear(2, 1)).save(str(checkpoints_dir / "ranker.pt"))
    with pytest.raises(RuntimeError, match="score_candidate_matrix"):
        comparison_module.load_model_bundle(
            "A", model_dir, None, torch.device("cpu")
        )


def test_model_loading_falls_back_to_best_pth_and_preserves_matrix_scores(
    comparison_module, tmp_path
):
    model_dir = tmp_path / "model"
    checkpoints_dir = model_dir / "checkpoints"
    get_data_dir = tmp_path / "get-data"
    checkpoints_dir.mkdir(parents=True)
    get_data_dir.mkdir()
    config = {
        "model_type": "bst-ranker",
        "post_embedding_dim": 2,
        "max_history_len": 3,
        "use_author_embedding_table": True,
        "author_table_num_rows": 3,
        "author_embedding_dim": 2,
        "content_projection_dim": 2,
        "author_projection_dim": 2,
        "model_dim": 4,
        "time_embedding_dim": 2,
        "num_attention_heads": 2,
        "num_transformer_layers": 1,
        "transformer_ff_dim": 8,
        "dropout_rate": 0.0,
        "author_unknown_dropout_rate": 0.0,
        "norm_first": False,
        "time_delta_bucket_boundaries_hours": [1.0, 24.0],
        "prediction_hidden_dims": [4],
        "bst_use_popularity_feature": True,
        "bst_popularity_projection_dim": 2,
        "bst_popularity_log_mean": 1.0,
        "bst_popularity_log_std": 2.0,
    }
    (model_dir / "training_config.json").write_text(json.dumps(config))
    (model_dir / "manifest.json").write_text(
        json.dumps({"inputs": {"01_get_data": str(get_data_dir)}})
    )
    pl.DataFrame({"author_did": ["did:a"], "author_idx": [2]}).write_parquet(
        get_data_dir / "author_idx_test.parquet"
    )

    stage_train_bst_ranker = importlib.import_module(
        "utils.03_train.stage_train_bst_ranker"
    )
    source_model = stage_train_bst_ranker.BSTRanker(
        post_embedding_dim=config["post_embedding_dim"],
        author_table_num_rows=config["author_table_num_rows"],
        author_embedding_dim=config["author_embedding_dim"],
        content_projection_dim=config["content_projection_dim"],
        author_projection_dim=config["author_projection_dim"],
        model_dim=config["model_dim"],
        time_embedding_dim=config["time_embedding_dim"],
        num_attention_heads=config["num_attention_heads"],
        num_transformer_layers=config["num_transformer_layers"],
        transformer_ff_dim=config["transformer_ff_dim"],
        dropout_rate=config["dropout_rate"],
        author_unknown_dropout_rate=config["author_unknown_dropout_rate"],
        norm_first=config["norm_first"],
        time_delta_bucket_boundaries_hours=config[
            "time_delta_bucket_boundaries_hours"
        ],
        prediction_hidden_dims=config["prediction_hidden_dims"],
        use_popularity_feature=config["bst_use_popularity_feature"],
        popularity_projection_dim=config["bst_popularity_projection_dim"],
        popularity_log_mean=config["bst_popularity_log_mean"],
        popularity_log_std=config["bst_popularity_log_std"],
    ).eval()
    torch.save(
        {"model_state_dict": source_model.state_dict()},
        checkpoints_dir / "bst_ranker_best.pth",
    )

    bundle = comparison_module.load_model_bundle(
        "A", model_dir, None, torch.device("cpu")
    )
    history_embeddings = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.0, 0.0]]])
    history_mask = torch.tensor([[True, True, False]])
    history_times = torch.tensor([[0.5, 3.0, 0.0]])
    candidate_embeddings = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    history_authors = torch.tensor([[2, 1, 0]])
    candidate_authors = torch.tensor([2, 1])
    history_likes = torch.tensor([[4.0, 8.0, 0.0]])
    candidate_likes = torch.tensor([16.0, 32.0])
    inputs = (
        history_embeddings,
        history_mask,
        history_times,
        candidate_embeddings,
        history_authors,
        candidate_authors,
        history_likes,
        candidate_likes,
    )

    with torch.inference_mode():
        expected = source_model.score_candidate_matrix(*inputs)
        actual = bundle.ranker.score_candidate_matrix(*inputs)

    assert torch.equal(actual, expected)
    assert bundle.use_popularity_feature is True


def test_summary_omits_api_key_and_writes_empty_outputs(comparison_module, tmp_path):
    ranker = object()
    model_a = _model_bundle(comparison_module, "A", ranker)
    model_b = _model_bundle(comparison_module, "B", ranker)
    args = argparse.Namespace(
        gcs_bucket="bucket",
        inference_prefix="bsky_inferences",
        start_date="2026-08-07",
        end_date="2026-08-10",
        political_threshold=0.95,
        class_sample_size=10_000,
        random_seed=42,
        elasticsearch_index="posts_recent",
        elasticsearch_batch_size=1000,
        elasticsearch_timeout_seconds=60.0,
        elasticsearch_insecure=False,
        embedding_model="all_MiniLM_L12_v2",
        device="cpu",
        elasticsearch_api_key="top-secret",
    )
    window = comparison_module.resolve_date_window("2026-08-07", "2026-08-10")
    frame = comparison_module.build_scores_frame([])
    diagnostics = comparison_module.RetrievalDiagnostics()
    summary = comparison_module.build_summary(
        args,
        tmp_path,
        window,
        [],
        {
            "unique_inference_uris": 0,
            "class_sample_size_requested": 10_000,
            "class_sample_size_selected": 0,
            "political_uris_available": 0,
            "political_uris_selected": 0,
            "non_political_uris_available": 0,
            "non_political_uris_selected": 0,
            "evaluation_uris": 0,
        },
        frame,
        diagnostics,
        model_a,
        model_b,
        "https://user:password@example.com:9200?token=secret",
        datetime(2026, 8, 10, tzinfo=timezone.utc),
    )
    assert summary["run_config"]["political_threshold"] == 0.95
    assert summary["run_config"]["political_score_field"] == (
        "topic.News & Social Concern"
    )

    serialized = json.dumps(summary)
    assert "top-secret" not in serialized
    assert "password" not in serialized
    assert "token=secret" not in serialized
    assert summary["run_config"]["elasticsearch_url"] == "https://example.com:9200"
    assert summary["comparison"]["model_b_higher_percent"] is None
    assert summary["models"]["A"]["ranking"]["political_average_rank"] is None

    scores_path = tmp_path / "political_comparison_scores.parquet"
    frame.write_parquet(scores_path)
    comparison_module.write_plots(frame, tmp_path)
    assert pl.read_parquet(scores_path).schema == frame.schema
    assert (tmp_path / "score_distributions.png").exists()
    assert (tmp_path / "score_deltas.png").exists()
    assert (tmp_path / "political_rank_comparison.png").exists()
