"""Read post media flags from an explicitly configured Elasticsearch endpoint."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from urllib.parse import quote, urlsplit
import warnings

import polars as pl
import requests
from urllib3.exceptions import InsecureRequestWarning


@dataclass
class _MediaClient:
    session: requests.Session
    base_url: str
    verify_ssl: bool
    request_timeout: float

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict,
        params: dict,
        operation: str,
    ) -> dict:
        try:
            with warnings.catch_warnings():
                if not self.verify_ssl:
                    warnings.simplefilter("ignore", InsecureRequestWarning)
                response = self.session.request(
                    method,
                    f"{self.base_url}{path}",
                    json=body,
                    params=params,
                    timeout=self.request_timeout,
                    verify=self.verify_ssl,
                    allow_redirects=False,
                )
        except requests.RequestException as exc:
            # Request exceptions can include credentials or response bodies.
            raise RuntimeError(
                f"Elasticsearch {operation} failed at {self.base_url} "
                f"({type(exc).__name__}); check the URL, TLS settings, and authentication"
            ) from None
        try:
            if not 200 <= response.status_code < 300:
                raise RuntimeError(
                    f"Elasticsearch {operation} failed at {self.base_url}: "
                    f"HTTP {response.status_code}"
                )
            try:
                payload = response.json()
            except ValueError:
                raise RuntimeError(
                    f"Elasticsearch {operation} returned invalid JSON"
                ) from None
            if not isinstance(payload, dict) or "error" in payload:
                raise RuntimeError(f"Elasticsearch {operation} returned an invalid response")
            return payload
        finally:
            response.close()


def _validate_search(payload: dict) -> list[dict]:
    if type(payload.get("timed_out")) is not bool:
        raise RuntimeError("Elasticsearch search response is missing a valid timed_out flag")
    if payload["timed_out"]:
        raise RuntimeError("Elasticsearch search timed out")
    shards = payload.get("_shards")
    if not isinstance(shards, dict) or type(shards.get("failed")) is not int:
        raise RuntimeError("Elasticsearch search response is missing shard status")
    if shards["failed"] != 0 or shards.get("failures"):
        raise RuntimeError("Elasticsearch search returned failed or partial shards")
    hits = payload.get("hits")
    if not isinstance(hits, dict) or not isinstance(hits.get("hits"), list):
        raise RuntimeError("Elasticsearch search returned invalid hits")
    if not all(isinstance(hit, dict) for hit in hits["hits"]):
        raise RuntimeError("Elasticsearch search returned invalid hit records")
    return hits["hits"]


def _total_hits(payload: dict) -> int:
    total = payload["hits"].get("total")
    if isinstance(total, dict):
        if total.get("relation") != "eq":
            raise RuntimeError("Elasticsearch search did not return an exact hit count")
        total = total.get("value")
    if type(total) is not int or total < 0:
        raise RuntimeError("Elasticsearch search returned an invalid total hit count")
    return total


def _hydrate_batch(
    client: _MediaClient,
    *,
    search_path: str,
    uris: list[str],
    batch_size: int,
    logger: logging.Logger,
) -> dict[str, tuple[bool | None, bool | None]]:
    scroll_ids: set[str] = set()
    finished = False
    flags: dict[str, tuple[bool | None, bool | None]] = {}
    requested = set(uris)
    scroll_ttl = f"{max(1, math.ceil(client.request_timeout * 2))}s"
    try:
        payload = client.request(
            "POST",
            search_path,
            body={
                "query": {"terms": {"at_uri": uris}},
                "size": batch_size,
                "track_total_hits": True,
                "sort": ["_doc"],
                "_source": ["at_uri", "contains_images", "contains_video"],
            },
            params={"scroll": scroll_ttl, "allow_partial_search_results": "false"},
            operation="media lookup",
        )
        expected: int | None = None
        seen = 0
        while True:
            scroll_id = payload.get("_scroll_id")
            if isinstance(scroll_id, str) and scroll_id:
                scroll_ids.add(scroll_id)
            hits = _validate_search(payload)
            if expected is None:
                expected = _total_hits(payload)
            seen += len(hits)
            if seen > expected:
                raise RuntimeError("Elasticsearch search returned more hits than its total")
            for hit in hits:
                source = hit.get("_source")
                if not isinstance(source, dict):
                    raise RuntimeError("Elasticsearch media hit is missing its source")
                uri = source.get("at_uri")
                if not isinstance(uri, str) or uri not in requested:
                    raise RuntimeError("Elasticsearch media hit has an unexpected post URI")
                pair = (source.get("contains_images"), source.get("contains_video"))
                if any(value is not None and type(value) is not bool for value in pair):
                    raise RuntimeError("Elasticsearch media flags must be booleans or null")
                if uri in flags and flags[uri] != pair:
                    raise RuntimeError(f"Elasticsearch contains conflicting media flags for {uri}")
                flags[uri] = pair
            if seen == expected:
                break
            if not hits or not isinstance(scroll_id, str) or not scroll_id:
                raise RuntimeError("Elasticsearch media search ended before all hits were read")
            payload = client.request(
                "POST",
                "/_search/scroll",
                body={"scroll_id": scroll_id, "scroll": scroll_ttl},
                params={},
                operation="media pagination",
            )
        finished = True
        return flags
    finally:
        if scroll_ids:
            try:
                cleared = client.request(
                    "DELETE",
                    "/_search/scroll",
                    body={"scroll_id": sorted(scroll_ids)},
                    params={},
                    operation="scroll cleanup",
                )
                if cleared.get("succeeded") is not True:
                    raise RuntimeError("Elasticsearch did not confirm scroll cleanup")
            except RuntimeError:
                if finished:
                    raise
                logger.warning("Elasticsearch scroll cleanup also failed after a lookup error")


def hydrate_candidate_media(
    candidates: pl.DataFrame,
    *,
    es_url: str,
    es_index: str,
    verify_ssl: bool,
    request_timeout: float,
    batch_size: int,
    api_key: str | None,
    logger: logging.Logger,
) -> pl.DataFrame:
    """Hydrate each unique URI, combining split membership and preserving unknowns.

    The configured endpoint is checked even for an empty candidate table. Missing
    documents or flags remain nullable and are never classified as text-only.
    """
    try:
        parsed = urlsplit(es_url)
        valid_url = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and parsed.port != 0
            and not any(char.isspace() for char in es_url)
        )
    except ValueError:
        valid_url = False
    if not valid_url:
        raise ValueError("Elasticsearch URL must be an explicit HTTP(S) URL without credentials, query, or fragment")
    if not es_index or any(char.isspace() or char in "/\\?#" for char in es_index):
        raise ValueError("Elasticsearch index must be a nonempty index name or alias")
    if type(verify_ssl) is not bool:
        raise ValueError("Elasticsearch TLS verification must be a boolean")
    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("Elasticsearch request timeout must be positive and finite")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Elasticsearch batch size must be a positive integer")
    if api_key is not None and (not isinstance(api_key, str) or "\n" in api_key or "\r" in api_key):
        raise ValueError("Elasticsearch API key must be a single-line string")
    schema = {"at_uri": pl.String, "in_val": pl.Boolean, "in_val_unseen_users": pl.Boolean}
    if any(candidates.schema.get(name) != dtype for name, dtype in schema.items()):
        raise ValueError("Media candidates require at_uri, in_val, and in_val_unseen_users columns with string/boolean types")
    if any(candidates[name].null_count() for name in schema):
        raise ValueError("Media candidate URIs and split membership must not contain nulls")
    if candidates.filter(pl.col("at_uri").str.strip_chars() == "").height:
        raise ValueError("Media candidate URIs must not be empty")
    unique = candidates.group_by("at_uri", maintain_order=True).agg(
        pl.col("in_val").any(),
        pl.col("in_val_unseen_users").any(),
    )
    base_url = es_url.rstrip("/")
    search_path = f"/{quote(es_index, safe=',*-_')}/_search"
    flags: dict[str, tuple[bool | None, bool | None]] = {}
    with requests.Session() as session:
        # Use only explicit transport/auth settings, including for local tunnels.
        session.trust_env = False
        if api_key:
            session.headers["Authorization"] = f"ApiKey {api_key}"
        client = _MediaClient(session, base_url, verify_ssl, request_timeout)
        preflight = client.request(
            "POST",
            search_path,
            body={"query": {"match_none": {}}, "size": 0, "track_total_hits": True},
            params={"allow_partial_search_results": "false"},
            operation="connection check",
        )
        if _validate_search(preflight) or _total_hits(preflight) != 0:
            raise RuntimeError("Elasticsearch connection check returned unexpected hits")
        uris = unique["at_uri"].to_list()
        logger.info("Hydrating media for %d unique candidate URIs from %s", len(uris), base_url)
        for start in range(0, len(uris), batch_size):
            flags.update(_hydrate_batch(
                client,
                search_path=search_path,
                uris=uris[start:start + batch_size],
                batch_size=batch_size,
                logger=logger,
            ))
            logger.info("Hydrated media lookup for %d/%d candidate URIs", min(start + batch_size, len(uris)), len(uris))
    images: list[bool | None] = []
    videos: list[bool | None] = []
    statuses: list[str] = []
    for uri in unique["at_uri"]:
        pair = flags.get(uri, (None, None))
        images.append(pair[0])
        videos.append(pair[1])
        statuses.append(
            "missing_document" if uri not in flags
            else "missing_flags" if None in pair
            else "known"
        )
    return unique.with_columns(
        pl.Series("contains_images", images, dtype=pl.Boolean),
        pl.Series("contains_video", videos, dtype=pl.Boolean),
        pl.Series("media_status", statuses, dtype=pl.String),
    )
