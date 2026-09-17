from __future__ import annotations

import logging
import warnings

import polars as pl
import pytest
import requests
from urllib3.exceptions import InsecureRequestWarning

from engagement_prediction.evaluation.media_hydration import hydrate_candidate_media


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.closed = False

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}
        self.trust_env = True
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        assert self.responses, "Unexpected Elasticsearch request"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _search(*sources, total=None, scroll_id=None):
    payload = {
        "timed_out": False,
        "_shards": {"total": 2, "successful": 2, "failed": 0},
        "hits": {
            "total": {"value": len(sources) if total is None else total, "relation": "eq"},
            "hits": [{"_source": source} for source in sources],
        },
    }
    if scroll_id is not None:
        payload["_scroll_id"] = scroll_id
    return FakeResponse(payload)


def _candidates(uris):
    return pl.DataFrame({
        "at_uri": pl.Series(uris, dtype=pl.String),
        "in_val": pl.Series([True] * len(uris), dtype=pl.Boolean),
        "in_val_unseen_users": pl.Series([False] * len(uris), dtype=pl.Boolean),
    })


def _run(monkeypatch, responses, candidates, **overrides):
    session = FakeSession(responses)
    monkeypatch.setattr(requests, "Session", lambda: session)
    kwargs = {
        "es_url": "https://127.0.0.1:9202",
        "es_index": "posts",
        "verify_ssl": True,
        "request_timeout": 12.5,
        "batch_size": 2,
        "api_key": None,
        "logger": logging.getLogger(__name__),
    }
    kwargs.update(overrides)
    result = hydrate_candidate_media(candidates, **kwargs)
    assert not session.responses
    assert session.closed
    return result, session


def test_hydrates_once_across_splits_and_preserves_false_and_missing(monkeypatch):
    candidates = pl.DataFrame({
        "at_uri": ["a", "b", "a", "c", "d"],
        "in_val": [True, True, False, False, True],
        "in_val_unseen_users": [False, False, True, True, True],
    })
    responses = [
        _search(),
        _search(
            {"at_uri": "a", "contains_images": True, "contains_video": False},
            {"at_uri": "b", "contains_images": False, "contains_video": False},
            scroll_id="first",
        ),
        FakeResponse({"succeeded": True, "num_freed": 1}),
        _search({"at_uri": "c", "contains_images": False}, scroll_id="second"),
        FakeResponse({"succeeded": True, "num_freed": 1}),
    ]
    result, session = _run(monkeypatch, responses, candidates)
    assert result.to_dicts() == [
        {"at_uri": "a", "in_val": True, "in_val_unseen_users": True,
         "contains_images": True, "contains_video": False, "media_status": "known"},
        {"at_uri": "b", "in_val": True, "in_val_unseen_users": False,
         "contains_images": False, "contains_video": False, "media_status": "known"},
        {"at_uri": "c", "in_val": False, "in_val_unseen_users": True,
         "contains_images": False, "contains_video": None, "media_status": "missing_flags"},
        {"at_uri": "d", "in_val": True, "in_val_unseen_users": True,
         "contains_images": None, "contains_video": None, "media_status": "missing_document"},
    ]
    assert result.schema["contains_images"] == pl.Boolean
    assert result.schema["contains_video"] == pl.Boolean
    assert session.calls[1][2]["json"]["query"] == {"terms": {"at_uri": ["a", "b"]}}
    assert session.calls[3][2]["json"]["query"] == {"terms": {"at_uri": ["c", "d"]}}
    assert all(response.closed for response in responses)


def test_uses_exact_endpoint_explicit_tls_and_auth_without_redirects(monkeypatch):
    result, session = _run(
        monkeypatch,
        [_search()],
        _candidates([]),
        es_url="https://127.0.0.1:9202/prefix/",
        es_index="posts,replies",
        verify_ssl=False,
        api_key="private-key",
    )
    assert result.is_empty()
    assert result.schema == {
        "at_uri": pl.String, "in_val": pl.Boolean, "in_val_unseen_users": pl.Boolean,
        "contains_images": pl.Boolean, "contains_video": pl.Boolean, "media_status": pl.String,
    }
    assert session.headers == {"Authorization": "ApiKey private-key"}
    assert session.trust_env is False
    assert session.calls == [(
        "POST", "https://127.0.0.1:9202/prefix/posts,replies/_search", {
            "json": {"query": {"match_none": {}}, "size": 0, "track_total_hits": True},
            "params": {"allow_partial_search_results": "false"},
            "timeout": 12.5, "verify": False, "allow_redirects": False,
        },
    )]


@pytest.mark.parametrize("verify_ssl", [True, False])
def test_tls_warning_suppression_is_local_and_specific(monkeypatch, verify_ssl):
    original_request = FakeSession.request

    def warn_then_request(self, *args, **kwargs):
        warnings.warn("TLS warning", InsecureRequestWarning)
        warnings.warn("unrelated warning", UserWarning)
        return original_request(self, *args, **kwargs)

    monkeypatch.setattr(FakeSession, "request", warn_then_request)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(monkeypatch, [_search()], _candidates([]), verify_ssl=verify_ssl)
        warnings.warn("outside the request", InsecureRequestWarning)
    assert sum(item.category is InsecureRequestWarning for item in caught) == 1 + int(verify_ssl)
    assert sum(item.category is UserWarning for item in caught) == 1


def test_reads_all_scroll_pages_and_collapses_identical_duplicate_hits(monkeypatch):
    image = {"at_uri": "a", "contains_images": True, "contains_video": False}
    mixed = {"at_uri": "b", "contains_images": True, "contains_video": True}
    result, session = _run(monkeypatch, [
        _search(),
        _search(image, image, total=3, scroll_id="page1"),
        _search(mixed, total=3, scroll_id="page2"),
        FakeResponse({"succeeded": True}),
    ], _candidates(["a", "b"]))
    assert result["media_status"].to_list() == ["known", "known"]
    assert result["contains_video"].to_list() == [False, True]
    assert session.calls[1][2]["json"]["_source"] == ["at_uri", "contains_images", "contains_video"]
    assert session.calls[2][1] == "https://127.0.0.1:9202/_search/scroll"
    assert session.calls[2][2]["json"] == {"scroll_id": "page1", "scroll": "25s"}
    assert session.calls[3][0] == "DELETE"
    assert session.calls[3][2]["json"] == {"scroll_id": ["page1", "page2"]}


@pytest.mark.parametrize("error", [
    requests.ConnectionError("private-key transport detail"),
    requests.exceptions.SSLError("private-key TLS detail"),
    requests.Timeout("private-key timeout detail"),
])
def test_preflight_transport_failure_is_immediate_and_redacted(monkeypatch, error):
    session = FakeSession([error])
    monkeypatch.setattr(requests, "Session", lambda: session)
    with pytest.raises(RuntimeError, match="connection check.*127.0.0.1:9202") as caught:
        hydrate_candidate_media(
            _candidates([]), es_url="https://127.0.0.1:9202", es_index="posts",
            verify_ssl=True, request_timeout=1, batch_size=2, api_key="private-key",
            logger=logging.getLogger(__name__),
        )
    assert "private-key" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert len(session.calls) == 1
    assert session.closed


@pytest.mark.parametrize("status", [301, 401, 403, 404, 503])
def test_preflight_http_failures_do_not_fallback_or_leak_response(monkeypatch, status):
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as caught:
        _run(monkeypatch, [FakeResponse({"error": "private-key"}, status)], _candidates(["a"]))
    assert "private-key" not in str(caught.value)


@pytest.mark.parametrize("payload, message", [
    (ValueError("private-key"), "invalid JSON"),
    ([], "invalid response"),
    ({"error": "private-key"}, "invalid response"),
    ({}, "timed_out"),
    ({"timed_out": True}, "timed out"),
    ({"timed_out": False}, "shard status"),
    ({"timed_out": False, "_shards": {"failed": 1}}, "partial shards"),
    ({"timed_out": False, "_shards": {"failed": 0}, "hits": {}}, "invalid hits"),
])
def test_rejects_invalid_or_partial_preflight_responses(monkeypatch, payload, message):
    with pytest.raises(RuntimeError, match=message) as caught:
        _run(monkeypatch, [FakeResponse(payload)], _candidates(["a"]))
    assert "private-key" not in str(caught.value)


@pytest.mark.parametrize("source, message", [
    ({"at_uri": "a", "contains_images": 0, "contains_video": False}, "must be booleans"),
    ({"at_uri": "a", "contains_images": "false"}, "must be booleans"),
    ({"at_uri": "unexpected"}, "unexpected post URI"),
    ({}, "unexpected post URI"),
    (None, "missing its source"),
])
def test_invalid_hits_still_clear_scroll_context(monkeypatch, source, message):
    session = FakeSession([
        _search(), _search(source, scroll_id="cleanup"), FakeResponse({"succeeded": True}),
    ])
    monkeypatch.setattr(requests, "Session", lambda: session)
    with pytest.raises(RuntimeError, match=message):
        hydrate_candidate_media(
            _candidates(["a"]), es_url="http://127.0.0.1:9202", es_index="posts",
            verify_ssl=True, request_timeout=1, batch_size=2, api_key=None,
            logger=logging.getLogger(__name__),
        )
    assert not session.responses
    assert session.calls[-1][0] == "DELETE"
    assert session.closed


def test_conflicting_duplicates_across_pages_fail_and_clear_context(monkeypatch):
    with pytest.raises(RuntimeError, match="conflicting media flags"):
        _run(monkeypatch, [
            _search(),
            _search({"at_uri": "a", "contains_images": True, "contains_video": False}, total=2, scroll_id="one"),
            _search({"at_uri": "a", "contains_images": False, "contains_video": False}, total=2, scroll_id="two"),
            FakeResponse({"succeeded": True}),
        ], _candidates(["a"]))


def test_pagination_failure_cleans_existing_context(monkeypatch):
    session = FakeSession([
        _search(), _search({"at_uri": "a"}, total=2, scroll_id="one"),
        requests.Timeout("private-key"), FakeResponse({"succeeded": True}),
    ])
    monkeypatch.setattr(requests, "Session", lambda: session)
    with pytest.raises(RuntimeError, match="media pagination") as caught:
        hydrate_candidate_media(
            _candidates(["a"]), es_url="http://127.0.0.1:9202", es_index="posts",
            verify_ssl=True, request_timeout=1, batch_size=2, api_key=None,
            logger=logging.getLogger(__name__),
        )
    assert "private-key" not in str(caught.value)
    assert not session.responses
    assert session.calls[-1][0] == "DELETE"
    assert session.calls[-1][2]["json"] == {"scroll_id": ["one"]}


def test_cleanup_failure_preserves_original_validation_error(monkeypatch, caplog):
    with pytest.raises(RuntimeError, match="must be booleans"):
        _run(monkeypatch, [
            _search(), _search({"at_uri": "a", "contains_images": "bad"}, scroll_id="one"),
            FakeResponse({"error": "private-key"}, 503),
        ], _candidates(["a"]))
    assert "cleanup also failed" in caplog.text
    assert "private-key" not in caplog.text


def test_cleanup_failure_after_successful_lookup_is_an_error(monkeypatch):
    with pytest.raises(RuntimeError, match="scroll cleanup.*HTTP 503"):
        _run(monkeypatch, [
            _search(), _search({"at_uri": "a"}, scroll_id="one"), FakeResponse({}, 503),
        ], _candidates(["a"]))


@pytest.mark.parametrize("first, remaining, message", [
    (_search({"at_uri": "a"}, total=2), [], "before all hits"),
    (_search({"at_uri": "a"}, total=0), [], "more hits than"),
    (_search({"at_uri": "a"}, total=2, scroll_id="one"),
     [_search(total=2, scroll_id="two"), FakeResponse({"succeeded": True})], "before all hits"),
])
def test_truncated_or_inconsistent_searches_fail(monkeypatch, first, remaining, message):
    with pytest.raises(RuntimeError, match=message):
        _run(monkeypatch, [_search(), first, *remaining], _candidates(["a"]))


def test_null_flags_are_unknown_and_known_companion_is_retained(monkeypatch):
    result, _ = _run(monkeypatch, [
        _search(), _search({"at_uri": "a", "contains_images": None, "contains_video": True}),
    ], _candidates(["a"]))
    assert result["contains_images"].to_list() == [None]
    assert result["contains_video"].to_list() == [True]
    assert result["media_status"].to_list() == ["missing_flags"]


@pytest.mark.parametrize("url", [
    "127.0.0.1:9202", "ftp://127.0.0.1:9202", "http://user:private-key@localhost:9202",
    "http://localhost:9202?token=private-key", "http://localhost:9202#private-key",
    "http://localhost:99999", "http://localhost:0", "http://bad host:9202",
])
def test_rejects_ambiguous_or_credential_bearing_urls_without_echoing_them(monkeypatch, url):
    with pytest.raises(ValueError, match="explicit HTTP") as caught:
        _run(monkeypatch, [], _candidates([]), es_url=url)
    assert "private-key" not in str(caught.value)


@pytest.mark.parametrize("overrides", [
    {"batch_size": 0}, {"batch_size": True}, {"request_timeout": 0},
    {"request_timeout": float("inf")}, {"verify_ssl": "false"},
    {"es_index": "posts/_search"}, {"es_index": ""}, {"api_key": "bad\nkey"},
])
def test_rejects_invalid_configuration_before_connecting(monkeypatch, overrides):
    with pytest.raises(ValueError):
        _run(monkeypatch, [], _candidates([]), **overrides)


@pytest.mark.parametrize("candidates", [
    pl.DataFrame({"at_uri": ["a"]}),
    _candidates([None]),
    _candidates([""]),
    _candidates(["a"]).with_columns(pl.lit(None, dtype=pl.Boolean).alias("in_val")),
])
def test_rejects_invalid_candidate_inputs_before_connecting(monkeypatch, candidates):
    with pytest.raises(ValueError):
        _run(monkeypatch, [], candidates)
