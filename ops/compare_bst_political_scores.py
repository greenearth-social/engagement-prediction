#!/usr/bin/env python3
"""Compare two TorchScript BST rankers on political and non-political posts.

Inference parquet files define balanced random samples of political and
non-political posts. Post features are hydrated from Elasticsearch in bounded
batches so post embeddings never need to be retained for the complete
evaluation population.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx
import numpy as np
import polars as pl
import torch
from google.cloud import storage
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.input_data_helpers import AUTHOR_PAD_IDX, AUTHOR_UNK_IDX


LOGGER = logging.getLogger("compare_bst_political_scores")

DEFAULT_GCS_BUCKET = "greenearth-471522-ingex-extract-prod"
DEFAULT_INFERENCE_PREFIX = "bsky_inferences"
DEFAULT_EMBEDDING_MODEL = "all_MiniLM_L12_v2"
DEFAULT_ELASTICSEARCH_INDEX = "posts_recent"
DEFAULT_OUTPUT_ROOT = Path("/mnt/data/dave/outputs/compare")
DEFAULT_POLITICAL_THRESHOLD = 0.95
POLITICAL_SCORE_FIELD = "topic.News & Social Concern"
RANK_CUTOFFS = (10, 100, 1000)

POLITICAL_INFERENCE_DTYPE = pl.Struct({
    "text": pl.Struct({
        "message.commit.record.text": pl.Struct({
            "topic": pl.Struct({
                "News & Social Concern": pl.Float64,
            }),
        }),
    }),
})

OUTPUT_SCHEMA = {
    "at_uri": pl.String,
    "author_did": pl.String,
    "created_at": pl.String,
    "elasticsearch_indexed_at": pl.String,
    "content": pl.String,
    "current_like_count": pl.Int64,
    "news_social_concern_score": pl.Float64,
    "inference_indexed_at": pl.String,
    "is_political": pl.Boolean,
    "model_a_logit": pl.Float64,
    "model_b_logit": pl.Float64,
    "model_b_minus_model_a": pl.Float64,
    "model_a_normalized_score": pl.Float64,
    "model_b_normalized_score": pl.Float64,
    "model_a_rank": pl.Int64,
    "model_b_rank": pl.Int64,
    "model_a_rank_percentile": pl.Float64,
    "model_b_rank_percentile": pl.Float64,
    "model_b_minus_model_a_rank": pl.Int64,
    "higher_scoring_model": pl.String,
}


@dataclass(frozen=True)
class DateWindow:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class HydratedPost:
    at_uri: str
    author_did: Optional[str]
    created_at: Optional[str]
    indexed_at: Optional[str]
    content: str
    like_count: int
    embedding: list[float]


@dataclass
class RetrievalDiagnostics:
    elasticsearch_batches: int = 0
    requested_uris: int = 0
    documents_found: int = 0
    missing_documents: int = 0
    duplicate_hits: int = 0
    missing_or_blank_content: int = 0
    missing_embeddings: int = 0
    malformed_embeddings: int = 0
    missing_authors: int = 0
    missing_like_counts: int = 0
    invalid_like_counts: int = 0
    hydrated_posts: int = 0
    scored_posts: int = 0


@dataclass
class ModelBundle:
    label: str
    model_dir: Path
    training_config: dict[str, Any]
    ranker: Any
    author_idx_by_did: dict[str, int]
    author_idx_path: Optional[Path]
    post_embedding_dim: int
    max_history_len: int
    use_popularity_feature: bool
    history_embeddings: torch.Tensor
    history_mask: torch.Tensor
    history_time_deltas_hours: torch.Tensor
    history_author_indices: torch.Tensor
    history_prior_cumulative_likes: torch.Tensor


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def normalize_elasticsearch_url(raw_url: str) -> str:
    url = raw_url.strip().rstrip("/")
    if not url:
        raise ValueError("Elasticsearch URL must not be empty")
    if "://" not in url:
        url = f"https://{url}"
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid Elasticsearch URL: {raw_url!r}")
    return url


def sanitize_elasticsearch_url(raw_url: str) -> str:
    parsed = urlsplit(normalize_elasticsearch_url(raw_url))
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    sanitized = SplitResult(parsed.scheme, netloc, parsed.path, "", "")
    return urlunsplit(sanitized).rstrip("/")


def is_loopback_elasticsearch_url(raw_url: str) -> bool:
    hostname = (urlsplit(normalize_elasticsearch_url(raw_url)).hostname or "").lower()
    return hostname in {"localhost", "127.0.0.1", "::1"}


def resolve_elasticsearch_settings(
    cli_url: Optional[str],
    cli_api_key: Optional[str],
    environ: Mapping[str, str],
) -> tuple[str, str]:
    raw_url = cli_url if cli_url is not None else environ.get("GE_ELASTICSEARCH_URL")
    raw_api_key = cli_api_key if cli_api_key is not None else environ.get("GE_ELASTICSEARCH_API_KEY")
    if raw_url is None or not raw_url.strip():
        raise ValueError(
            "Elasticsearch URL is required via --elasticsearch-url or GE_ELASTICSEARCH_URL"
        )
    if raw_api_key is None or not raw_api_key.strip():
        raise ValueError(
            "Elasticsearch API key is required via --elasticsearch-api-key or GE_ELASTICSEARCH_API_KEY"
        )
    return normalize_elasticsearch_url(raw_url), raw_api_key.strip()


def resolve_date_window(start_date: str, end_date: str) -> DateWindow:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", start_date) is None:
        raise ValueError("--start-date must use YYYY-MM-DD format")
    try:
        start_day = date.fromisoformat(start_date)
    except ValueError as exc:
        raise ValueError("--start-date must use YYYY-MM-DD format") from exc
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", end_date) is None:
        raise ValueError("--end-date must use YYYY-MM-DD format")
    try:
        end_day = date.fromisoformat(end_date)
    except ValueError as exc:
        raise ValueError("--end-date must use YYYY-MM-DD format") from exc
    if start_day >= end_day:
        raise ValueError("--start-date must be earlier than --end-date")
    return DateWindow(
        start=datetime.combine(start_day, time.min, tzinfo=timezone.utc),
        end=datetime.combine(end_day, time.min, tzinfo=timezone.utc),
    )


def _inference_blob_timestamp(blob_name: str, inference_prefix: str) -> Optional[datetime]:
    pattern = re.compile(
        rf"^{re.escape(inference_prefix)}_(\d{{8}})_(\d{{6}})\.parquet$"
    )
    match = pattern.match(blob_name)
    if match is None:
        return None
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )


def list_inference_parquet_paths(
    gcs_client: Any,
    gcs_bucket: str,
    inference_prefix: str,
    window: DateWindow,
) -> list[str]:
    paths_with_timestamps: list[tuple[datetime, str]] = []
    for blob in gcs_client.list_blobs(gcs_bucket, prefix=inference_prefix):
        timestamp = _inference_blob_timestamp(blob.name, inference_prefix)
        if timestamp is None or timestamp < window.start or timestamp >= window.end:
            continue
        paths_with_timestamps.append(
            (timestamp, f"gs://{gcs_bucket}/{blob.name}")
        )
    paths_with_timestamps.sort(key=lambda item: item[0])
    paths = [path for _, path in paths_with_timestamps]
    if not paths:
        raise FileNotFoundError(
            "No inference parquet files found for "
            f"gs://{gcs_bucket}/{inference_prefix}_*.parquet in "
            f"[{window.start.isoformat()}, {window.end.isoformat()})"
        )
    return paths


def build_evaluation_posts_df(
    inference_paths: Sequence[str],
    political_threshold: float,
    class_sample_size: int,
    random_seed: int,
) -> tuple[pl.DataFrame, dict[str, int]]:
    if not inference_paths:
        raise ValueError("At least one inference parquet path is required")
    if not 0.0 <= political_threshold <= 1.0:
        raise ValueError("--political-threshold must be between 0 and 1")
    if class_sample_size < 0:
        raise ValueError("--class-sample-size must be non-negative")

    parsed_record_expr = (
        pl.col("_parsed_inferences")
        .struct.field("text")
        .struct.field("message.commit.record.text")
    )
    latest = (
        pl.scan_parquet(list(inference_paths))
        .select(["at_uri", "indexed_at", "inferences"])
        .filter(pl.col("at_uri").is_not_null())
        .with_columns(
            pl.col("inferences")
            .str.json_decode(dtype=POLITICAL_INFERENCE_DTYPE)
            .alias("_parsed_inferences"),
            pl.col("indexed_at")
            .cast(pl.String)
            .str.to_datetime(strict=False, time_zone="UTC")
            .alias("_indexed_at_dt"),
        )
        .with_columns(
            parsed_record_expr
            .struct.field("topic")
            .struct.field("News & Social Concern")
            .fill_null(0.0)
            .alias("_news_social_concern_score"),
        )
        .group_by("at_uri")
        .agg(
            pl.col("indexed_at")
            .sort_by("_indexed_at_dt", "indexed_at")
            .last()
            .cast(pl.String)
            .alias("inference_indexed_at"),
            pl.col("_news_social_concern_score")
            .sort_by("_indexed_at_dt", "indexed_at")
            .last()
            .alias("news_social_concern_score"),
        )
        .with_columns(
            (pl.col("news_social_concern_score") >= political_threshold)
            .alias("is_political")
        )
        .select(
            "at_uri",
            "news_social_concern_score",
            "inference_indexed_at",
            "is_political",
        )
        .sort("at_uri")
        .collect(engine="streaming")
    )
    political = latest.filter(pl.col("is_political"))
    non_political = latest.filter(~pl.col("is_political"))
    selected_per_class = min(
        class_sample_size,
        political.height,
        non_political.height,
    )
    if selected_per_class > 0:
        selected_political = political.sample(
            n=selected_per_class,
            with_replacement=False,
            shuffle=True,
            seed=random_seed,
        )
        selected_non_political = non_political.sample(
            n=selected_per_class,
            with_replacement=False,
            shuffle=True,
            seed=random_seed,
        )
    else:
        selected_political = political.head(0)
        selected_non_political = non_political.head(0)
    evaluation = pl.concat([selected_political, selected_non_political]).sort("at_uri")
    stats = {
        "unique_inference_uris": latest.height,
        "class_sample_size_requested": class_sample_size,
        "class_sample_size_selected": selected_per_class,
        "political_uris_available": political.height,
        "political_uris_selected": selected_political.height,
        "non_political_uris_available": non_political.height,
        "non_political_uris_selected": selected_non_political.height,
        "evaluation_uris": evaluation.height,
    }
    return evaluation, stats


def _load_manifest(model_dir: Path) -> dict[str, Any]:
    for filename in ("manifest.json", "manifest.partial.json"):
        path = model_dir / filename
        if path.exists():
            return _read_json(path)
    return {}


def resolve_author_idx_path(get_data_dir: Path) -> Path:
    matches = sorted(
        get_data_dir.glob("author_idx_*.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No author_idx_*.parquet found under {get_data_dir}")
    return matches[0]


def _resolve_get_data_dir(
    model_dir: Path,
    manifest: Mapping[str, Any],
    get_data_override: Optional[Path],
) -> Path:
    if get_data_override is not None:
        get_data_dir = get_data_override.expanduser()
    else:
        inputs = manifest.get("inputs")
        raw_path = inputs.get("01_get_data") if isinstance(inputs, dict) else None
        if raw_path is None or not str(raw_path).strip():
            raise FileNotFoundError(
                f"{model_dir} does not record an inputs.01_get_data artifact path; "
                "provide the corresponding --model-*-get-data-dir override"
            )
        get_data_dir = Path(str(raw_path)).expanduser()
    if not get_data_dir.is_dir():
        raise FileNotFoundError(f"Stage 1 artifact directory was not found: {get_data_dir}")
    return get_data_dir


def load_author_idx_map(author_idx_path: Path) -> dict[str, int]:
    frame = pl.read_parquet(author_idx_path, columns=["author_did", "author_idx"])
    return {
        str(author_did): int(author_idx)
        for author_did, author_idx in frame.iter_rows()
        if author_did is not None and author_idx is not None
    }


def _required_positive_int(config: Mapping[str, Any], key: str, model_dir: Path) -> int:
    if key not in config:
        raise ValueError(f"{model_dir}/training_config.json is missing {key!r}")
    value = int(config[key])
    if value <= 0:
        raise ValueError(f"{model_dir}/training_config.json has invalid {key}={value}")
    return value


def _load_ranker(
    model_dir: Path,
    device: torch.device,
) -> tuple[Any, Path]:
    torchscript_path = model_dir / "checkpoints/ranker.pt"
    checkpoint_path = model_dir / "checkpoints/bst_ranker_best.pth"
    if torchscript_path.exists():
        ranker = torch.jit.load(str(torchscript_path), map_location=device).eval()
        ranker_path = torchscript_path
    elif checkpoint_path.exists():
        from utils.ranking_adapters import BstPthAdapter

        adapter = BstPthAdapter(checkpoint_path, candidate_chunk_size=1)
        adapter.prepare_for_eval(str(device))
        if adapter.model is None:
            raise RuntimeError(f"Failed to reconstruct BST ranker from {checkpoint_path}")
        ranker = adapter.model
        ranker_path = checkpoint_path
    else:
        raise FileNotFoundError(
            "No loadable BST ranker was found; expected either "
            f"{torchscript_path} or {checkpoint_path}"
        )

    if not callable(getattr(ranker, "score_candidate_matrix", None)):
        raise RuntimeError(f"BST ranker does not expose score_candidate_matrix: {ranker_path}")
    LOGGER.info("Loaded BST model from %s", ranker_path)
    return ranker, ranker_path


def load_model_bundle(
    label: str,
    model_dir: Path,
    get_data_override: Optional[Path],
    device: torch.device,
) -> ModelBundle:
    model_dir = model_dir.expanduser().resolve()
    config_path = model_dir / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Required BST training config was not found: {config_path}")

    config = _read_json(config_path)
    if config.get("model_type") != "bst-ranker":
        raise ValueError(
            f"Expected model_type='bst-ranker' in {config_path}, got {config.get('model_type')!r}"
        )
    post_embedding_dim = _required_positive_int(config, "post_embedding_dim", model_dir)
    max_history_len = _required_positive_int(config, "max_history_len", model_dir)
    if not bool(config.get("use_author_embedding_table")):
        raise ValueError(f"BST model {model_dir} does not enable the required author embedding table")

    manifest = _load_manifest(model_dir)
    get_data_dir = _resolve_get_data_dir(model_dir, manifest, get_data_override)
    author_idx_path = resolve_author_idx_path(get_data_dir)
    author_idx_by_did = load_author_idx_map(author_idx_path)

    ranker, _ = _load_ranker(model_dir, device)

    history_embeddings = torch.zeros(
        (1, max_history_len, post_embedding_dim), device=device, dtype=torch.float32
    )
    history_mask = torch.zeros((1, max_history_len), device=device, dtype=torch.bool)
    history_time_deltas_hours = torch.zeros(
        (1, max_history_len), device=device, dtype=torch.float32
    )
    history_author_indices = torch.full(
        (1, max_history_len), AUTHOR_PAD_IDX, device=device, dtype=torch.long
    )
    history_prior_cumulative_likes = torch.zeros(
        (1, max_history_len), device=device, dtype=torch.float32
    )

    return ModelBundle(
        label=label,
        model_dir=model_dir,
        training_config=config,
        ranker=ranker,
        author_idx_by_did=author_idx_by_did,
        author_idx_path=author_idx_path,
        post_embedding_dim=post_embedding_dim,
        max_history_len=max_history_len,
        use_popularity_feature=bool(config.get("bst_use_popularity_feature", False)),
        history_embeddings=history_embeddings,
        history_mask=history_mask,
        history_time_deltas_hours=history_time_deltas_hours,
        history_author_indices=history_author_indices,
        history_prior_cumulative_likes=history_prior_cumulative_likes,
    )


def validate_model_compatibility(model_a: ModelBundle, model_b: ModelBundle) -> int:
    if model_a.post_embedding_dim != model_b.post_embedding_dim:
        raise ValueError(
            "The BST models require different post embedding dimensions: "
            f"A={model_a.post_embedding_dim}, B={model_b.post_embedding_dim}"
        )
    return model_a.post_embedding_dim


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=timezone.utc)


def _hit_sort_key(hit: Mapping[str, Any]) -> tuple[datetime, str, str, str]:
    source = hit.get("_source")
    source = source if isinstance(source, dict) else {}
    indexed_at = source.get("indexed_at")
    created_at = source.get("created_at")
    timestamp_value = indexed_at if indexed_at is not None else created_at
    return (
        _parse_timestamp(timestamp_value),
        str(timestamp_value or ""),
        str(hit.get("_index") or ""),
        str(hit.get("_id") or ""),
    )


def _extract_embedding(
    hit: Mapping[str, Any],
    embedding_field: str,
    expected_dim: int,
) -> tuple[Optional[list[float]], str]:
    fields = hit.get("fields")
    values = fields.get(embedding_field) if isinstance(fields, dict) else None
    if values is None or values == []:
        return None, "missing"
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], list):
        return None, "malformed"
    vector = values[0]
    if len(vector) != expected_dim:
        return None, "malformed"
    try:
        array = np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError):
        return None, "malformed"
    if array.shape != (expected_dim,) or not np.isfinite(array).all():
        return None, "malformed"
    return array.tolist(), "valid"


def _coerce_like_count(value: Any) -> tuple[int, str]:
    if value is None:
        return 0, "missing"
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0, "invalid"
    return int(value), "valid"


async def search_elasticsearch(
    client: httpx.AsyncClient,
    elasticsearch_url: str,
    elasticsearch_index: str,
    api_key: str,
    body: Mapping[str, Any],
) -> dict[str, Any]:
    response = await client.post(
        f"{elasticsearch_url}/{elasticsearch_index}/_search",
        json=body,
        headers={
            "Authorization": f"ApiKey {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Elasticsearch query failed with HTTP {response.status_code} "
            f"for index {elasticsearch_index!r}"
        ) from exc
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Elasticsearch returned a non-object response")
    return payload


async def fetch_elasticsearch_batch(
    client: httpx.AsyncClient,
    elasticsearch_url: str,
    elasticsearch_index: str,
    api_key: str,
    at_uris: Sequence[str],
    embedding_field: str,
    expected_dim: int,
    diagnostics: RetrievalDiagnostics,
) -> list[HydratedPost]:
    if not at_uris:
        return []
    diagnostics.elasticsearch_batches += 1
    diagnostics.requested_uris += len(at_uris)
    payload = await search_elasticsearch(
        client,
        elasticsearch_url,
        elasticsearch_index,
        api_key,
        {
            "_source": [
                "at_uri",
                "author_did",
                "created_at",
                "indexed_at",
                "content",
                "like_count",
            ],
            "docvalue_fields": [embedding_field],
            "query": {"terms": {"at_uri": list(at_uris)}},
            "size": len(at_uris),
        },
    )
    hits_container = payload.get("hits")
    hits = hits_container.get("hits") if isinstance(hits_container, dict) else None
    if not isinstance(hits, list):
        raise RuntimeError("Elasticsearch response is missing hits.hits")

    hits_by_uri: dict[str, list[dict[str, Any]]] = {}
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        source = hit.get("_source")
        at_uri = source.get("at_uri") if isinstance(source, dict) else None
        if isinstance(at_uri, str) and at_uri:
            hits_by_uri.setdefault(at_uri, []).append(hit)

    requested_set = set(at_uris)
    found_uris = requested_set.intersection(hits_by_uri)
    diagnostics.documents_found += len(found_uris)
    diagnostics.missing_documents += len(requested_set.difference(found_uris))
    diagnostics.duplicate_hits += sum(
        max(0, len(uri_hits) - 1) for uri_hits in hits_by_uri.values()
    )

    hydrated: list[HydratedPost] = []
    for at_uri in at_uris:
        uri_hits = hits_by_uri.get(at_uri)
        if not uri_hits:
            continue
        hit = max(uri_hits, key=_hit_sort_key)
        source = hit.get("_source")
        source = source if isinstance(source, dict) else {}

        content = source.get("content")
        if not isinstance(content, str) or not content.strip():
            diagnostics.missing_or_blank_content += 1
            continue

        embedding, embedding_status = _extract_embedding(
            hit, embedding_field, expected_dim
        )
        if embedding_status == "missing":
            diagnostics.missing_embeddings += 1
            continue
        if embedding_status == "malformed" or embedding is None:
            diagnostics.malformed_embeddings += 1
            continue

        author_did = source.get("author_did")
        if not isinstance(author_did, str) or not author_did:
            diagnostics.missing_authors += 1
            author_did = None

        like_count, like_count_status = _coerce_like_count(source.get("like_count"))
        if like_count_status == "missing":
            diagnostics.missing_like_counts += 1
        elif like_count_status == "invalid":
            diagnostics.invalid_like_counts += 1

        hydrated.append(
            HydratedPost(
                at_uri=at_uri,
                author_did=author_did,
                created_at=(
                    str(source["created_at"])
                    if source.get("created_at") is not None
                    else None
                ),
                indexed_at=(
                    str(source["indexed_at"])
                    if source.get("indexed_at") is not None
                    else None
                ),
                content=content,
                like_count=like_count,
                embedding=embedding,
            )
        )
    diagnostics.hydrated_posts += len(hydrated)
    return hydrated


def score_model_batch(
    model: ModelBundle,
    candidate_embeddings: torch.Tensor,
    candidates: Sequence[HydratedPost],
    device: torch.device,
) -> np.ndarray:
    author_indices = torch.tensor(
        [
            model.author_idx_by_did.get(post.author_did, AUTHOR_UNK_IDX)
            if post.author_did is not None
            else AUTHOR_UNK_IDX
            for post in candidates
        ],
        device=device,
        dtype=torch.long,
    )
    candidate_like_counts = torch.tensor(
        [post.like_count for post in candidates],
        device=device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        scores = model.ranker.score_candidate_matrix(
            model.history_embeddings,
            model.history_mask,
            model.history_time_deltas_hours,
            candidate_embeddings,
            model.history_author_indices,
            author_indices,
            model.history_prior_cumulative_likes,
            candidate_like_counts,
        )
    if scores.dim() != 2 or scores.shape != (1, len(candidates)):
        raise RuntimeError(
            f"Model {model.label} score_candidate_matrix returned shape {tuple(scores.shape)}; "
            f"expected (1, {len(candidates)})"
        )
    return scores[0].detach().to(device="cpu", dtype=torch.float64).numpy()


def normalize_logits(values: Sequence[float]) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    normalized = np.full_like(logits, 0.5)
    if logits.size == 0:
        return normalized
    finite_mask = np.isfinite(logits)
    if finite_mask.any():
        finite = logits[finite_mask]
        low = float(finite.min())
        high = float(finite.max())
        if abs(high - low) >= 1e-6:
            normalized[finite_mask] = np.clip(
                (finite - low) / (high - low), 0.0, 1.0
            )
    has_finite = bool(finite_mask.any())
    positive_inf = np.isposinf(logits)
    negative_inf = np.isneginf(logits)
    if has_finite or (positive_inf.any() and negative_inf.any()):
        normalized[positive_inf] = 1.0
        normalized[negative_inf] = 0.0
    return np.clip(normalized, 0.0, 1.0)


def ordinal_ranks_descending(
    values: Sequence[float],
    at_uris: Sequence[str],
) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    uris = np.asarray(at_uris, dtype=str)
    if scores.shape != uris.shape:
        raise ValueError("values and at_uris must have the same shape")
    if scores.size == 0:
        return np.asarray([], dtype=np.int64)
    sortable_scores = np.where(np.isnan(scores), -np.inf, scores)
    order = np.lexsort((uris, -sortable_scores))
    ranks = np.empty(scores.size, dtype=np.int64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.int64)
    return ranks


def build_scores_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=OUTPUT_SCHEMA)
    model_a_logits = np.asarray([row["model_a_logit"] for row in rows], dtype=np.float64)
    model_b_logits = np.asarray([row["model_b_logit"] for row in rows], dtype=np.float64)
    model_a_normalized = normalize_logits(model_a_logits)
    model_b_normalized = normalize_logits(model_b_logits)
    at_uris = [str(row["at_uri"]) for row in rows]
    model_a_ranks = ordinal_ranks_descending(model_a_logits, at_uris)
    model_b_ranks = ordinal_ranks_descending(model_b_logits, at_uris)
    rank_denominator = max(1, len(rows) - 1)
    deltas = model_b_logits - model_a_logits
    for idx, row in enumerate(rows):
        delta = float(deltas[idx])
        row["model_b_minus_model_a"] = delta
        row["model_a_normalized_score"] = float(model_a_normalized[idx])
        row["model_b_normalized_score"] = float(model_b_normalized[idx])
        row["model_a_rank"] = int(model_a_ranks[idx])
        row["model_b_rank"] = int(model_b_ranks[idx])
        row["model_a_rank_percentile"] = float(
            (model_a_ranks[idx] - 1) / rank_denominator
        )
        row["model_b_rank_percentile"] = float(
            (model_b_ranks[idx] - 1) / rank_denominator
        )
        row["model_b_minus_model_a_rank"] = int(
            model_b_ranks[idx] - model_a_ranks[idx]
        )
        row["higher_scoring_model"] = "B" if delta > 0 else "A" if delta < 0 else "tie"
    return pl.DataFrame(rows, schema=OUTPUT_SCHEMA)


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "count": int(array.size),
            "finite_count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p95": None,
            "max": None,
            "std": None,
        }
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "min": float(finite.min()),
        "p05": float(np.quantile(finite, 0.05)),
        "p25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "p75": float(np.quantile(finite, 0.75)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(finite.max()),
        "std": float(finite.std()),
    }


def model_ranking_summary(
    scores_df: pl.DataFrame,
    score_column: str,
    rank_column: str,
    rank_percentile_column: str,
) -> dict[str, Any]:
    total = scores_df.height
    political = scores_df.filter(pl.col("is_political"))
    non_political = scores_df.filter(~pl.col("is_political"))
    political_count = political.height
    non_political_count = non_political.height
    baseline_share = political_count / total if total else None

    top_cutoffs: dict[str, Any] = {}
    for requested_cutoff in RANK_CUTOFFS:
        actual_cutoff = min(requested_cutoff, total)
        top = scores_df.filter(pl.col(rank_column) <= actual_cutoff)
        top_political_count = (
            int(top["is_political"].sum()) if actual_cutoff > 0 else 0
        )
        political_share = (
            top_political_count / actual_cutoff if actual_cutoff > 0 else None
        )
        top_cutoffs[str(requested_cutoff)] = {
            "requested_posts": requested_cutoff,
            "actual_posts": actual_cutoff,
            "political_posts": top_political_count,
            "non_political_posts": actual_cutoff - top_political_count,
            "political_share": political_share,
            "political_recall": (
                top_political_count / political_count if political_count else None
            ),
            "political_share_lift_vs_evaluation_population": (
                political_share / baseline_share
                if political_share is not None and baseline_share
                else None
            ),
        }

    roc_auc = None
    if political_count > 0 and non_political_count > 0:
        from sklearn.metrics import roc_auc_score

        labels = np.asarray(scores_df["is_political"].to_list(), dtype=np.int64)
        scores = np.asarray(scores_df[score_column].to_list(), dtype=np.float64)
        finite = np.isfinite(scores)
        if finite.any() and np.unique(labels[finite]).size == 2:
            roc_auc = float(roc_auc_score(labels[finite], scores[finite]))

    political_ranks = political[rank_column].to_list()
    non_political_ranks = non_political[rank_column].to_list()
    political_rank_percentiles = political[rank_percentile_column].to_list()
    non_political_rank_percentiles = non_political[rank_percentile_column].to_list()
    return {
        "scored_posts": total,
        "political_posts": political_count,
        "non_political_posts": non_political_count,
        "political_share_in_evaluation_population": baseline_share,
        "political_average_rank": (
            float(np.mean(political_ranks)) if political_ranks else None
        ),
        "political_median_rank": (
            float(np.median(political_ranks)) if political_ranks else None
        ),
        "non_political_average_rank": (
            float(np.mean(non_political_ranks)) if non_political_ranks else None
        ),
        "non_political_median_rank": (
            float(np.median(non_political_ranks)) if non_political_ranks else None
        ),
        "political_average_rank_percentile": (
            float(np.mean(political_rank_percentiles))
            if political_rank_percentiles
            else None
        ),
        "non_political_average_rank_percentile": (
            float(np.mean(non_political_rank_percentiles))
            if non_political_rank_percentiles
            else None
        ),
        "political_raw_logit": numeric_summary(political[score_column].to_list()),
        "non_political_raw_logit": numeric_summary(
            non_political[score_column].to_list()
        ),
        "political_vs_non_political_roc_auc": roc_auc,
        "top_cutoffs": top_cutoffs,
    }


def political_rank_shift_summary(scores_df: pl.DataFrame) -> dict[str, Any]:
    political = scores_df.filter(pl.col("is_political"))
    shifts = political["model_b_minus_model_a_rank"].to_list()
    moved_higher = sum(shift < 0 for shift in shifts)
    moved_lower = sum(shift > 0 for shift in shifts)
    unchanged = sum(shift == 0 for shift in shifts)
    total = len(shifts)
    return {
        "model_b_minus_model_a_rank": numeric_summary(shifts),
        "political_posts_ranked_higher_by_model_b": moved_higher,
        "political_posts_ranked_lower_by_model_b": moved_lower,
        "political_posts_with_unchanged_rank": unchanged,
        "political_posts_ranked_higher_by_model_b_percent": (
            100.0 * moved_higher / total if total else None
        ),
        "political_posts_ranked_lower_by_model_b_percent": (
            100.0 * moved_lower / total if total else None
        ),
    }


def build_summary(
    args: argparse.Namespace,
    output_dir: Path,
    window: DateWindow,
    inference_paths: Sequence[str],
    inference_selection_stats: Mapping[str, int],
    scores_df: pl.DataFrame,
    diagnostics: RetrievalDiagnostics,
    model_a: ModelBundle,
    model_b: ModelBundle,
    elasticsearch_url: str,
    evaluated_at: datetime,
) -> dict[str, Any]:
    model_a_logits = scores_df["model_a_logit"].to_list()
    model_b_logits = scores_df["model_b_logit"].to_list()
    deltas = scores_df["model_b_minus_model_a"].to_list()
    higher = scores_df["higher_scoring_model"].to_list()
    total = len(higher)
    a_higher = higher.count("A")
    b_higher = higher.count("B")
    ties = higher.count("tie")

    return {
        "evaluated_at": evaluated_at.isoformat(),
        "run_config": {
            "model_a_dir": str(model_a.model_dir),
            "model_b_dir": str(model_b.model_dir),
            "model_a_author_idx_path": str(model_a.author_idx_path) if model_a.author_idx_path else None,
            "model_b_author_idx_path": str(model_b.author_idx_path) if model_b.author_idx_path else None,
            "gcs_bucket": args.gcs_bucket,
            "inference_prefix": args.inference_prefix,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "start_date_inclusive": window.start.isoformat(),
            "end_date_exclusive": window.end.isoformat(),
            "political_threshold": args.political_threshold,
            "political_score_field": POLITICAL_SCORE_FIELD,
            "class_sample_size": args.class_sample_size,
            "random_seed": args.random_seed,
            "elasticsearch_url": sanitize_elasticsearch_url(elasticsearch_url),
            "elasticsearch_index": args.elasticsearch_index,
            "elasticsearch_batch_size": args.elasticsearch_batch_size,
            "elasticsearch_timeout_seconds": args.elasticsearch_timeout_seconds,
            "elasticsearch_insecure": args.elasticsearch_insecure,
            "embedding_field": f"embeddings.{args.embedding_model}",
            "device": args.device,
            "output_dir": str(output_dir),
        },
        "inference": {
            "files_scanned": len(inference_paths),
            **inference_selection_stats,
        },
        "retrieval_diagnostics": asdict(diagnostics),
        "models": {
            "A": {
                "model_dir": str(model_a.model_dir),
                "use_popularity_feature": model_a.use_popularity_feature,
                "raw_logit": numeric_summary(model_a_logits),
                "ranking": model_ranking_summary(
                    scores_df,
                    "model_a_logit",
                    "model_a_rank",
                    "model_a_rank_percentile",
                ),
            },
            "B": {
                "model_dir": str(model_b.model_dir),
                "use_popularity_feature": model_b.use_popularity_feature,
                "raw_logit": numeric_summary(model_b_logits),
                "ranking": model_ranking_summary(
                    scores_df,
                    "model_b_logit",
                    "model_b_rank",
                    "model_b_rank_percentile",
                ),
            },
        },
        "comparison": {
            "model_b_minus_model_a": numeric_summary(deltas),
            "model_a_higher_count": a_higher,
            "model_b_higher_count": b_higher,
            "tie_count": ties,
            "model_a_higher_percent": 100.0 * a_higher / total if total else None,
            "model_b_higher_percent": 100.0 * b_higher / total if total else None,
            "tie_percent": 100.0 * ties / total if total else None,
            "political_rank_shift": political_rank_shift_summary(scores_df),
        },
        "like_count_semantics": (
            "current Elasticsearch value at evaluated_at, not a historical count at the inference-window boundary"
        ),
        "ranking_semantics": (
            "rank 1 is the highest raw BST logit; equal logits are ordered by at_uri; "
            "rank percentile 0 is highest and 1 is lowest"
        ),
    }


def write_plots(scores_df: pl.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_a = np.asarray(scores_df["model_a_logit"].to_list(), dtype=np.float64)
    model_b = np.asarray(scores_df["model_b_logit"].to_list(), dtype=np.float64)
    political_mask = np.asarray(scores_df["is_political"].to_list(), dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, values, title in zip(axes, (model_a, model_b), ("Model A", "Model B")):
        finite = np.isfinite(values)
        political_values = values[finite & political_mask]
        non_political_values = values[finite & ~political_mask]
        if political_values.size or non_political_values.size:
            bins = min(
                60,
                max(10, int(np.sqrt(max(political_values.size, non_political_values.size)))),
            )
            if non_political_values.size:
                axis.hist(
                    non_political_values,
                    bins=bins,
                    density=True,
                    alpha=0.55,
                    label="Non-political",
                )
            if political_values.size:
                axis.hist(
                    political_values,
                    bins=bins,
                    density=True,
                    alpha=0.55,
                    label="Political",
                )
            axis.legend()
        else:
            axis.text(0.5, 0.5, "No scored posts", ha="center", va="center")
        axis.set_title(f"{title} raw logits")
        axis.set_xlabel("Logit")
        axis.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(output_dir / "score_distributions.png", dpi=160)
    plt.close(fig)

    delta = model_b - model_a
    finite_pairs = np.isfinite(model_a) & np.isfinite(model_b)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    finite_delta = delta[np.isfinite(delta)]
    if finite_delta.size:
        bins = min(60, max(10, int(np.sqrt(finite_delta.size))))
        axes[0].hist(finite_delta, bins=bins, color="slateblue", alpha=0.75)
        axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1)
        pair_indices = np.flatnonzero(finite_pairs)
        if pair_indices.size > 20_000:
            pair_indices = pair_indices[
                np.linspace(0, pair_indices.size - 1, 20_000, dtype=int)
            ]
        axes[1].scatter(
            model_a[pair_indices],
            model_b[pair_indices],
            s=8,
            alpha=0.25,
        )
        combined = np.concatenate([model_a[pair_indices], model_b[pair_indices]])
        low, high = float(combined.min()), float(combined.max())
        axes[1].plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    else:
        axes[0].text(0.5, 0.5, "No scored posts", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No scored posts", ha="center", va="center")
    axes[0].set_title("Paired logit difference")
    axes[0].set_xlabel("Model B - Model A")
    axes[0].set_ylabel("Posts")
    axes[1].set_title("Paired raw logits")
    axes[1].set_xlabel("Model A")
    axes[1].set_ylabel("Model B")
    fig.tight_layout()
    fig.savefig(output_dir / "score_deltas.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    total = scores_df.height
    political_count = int(political_mask.sum())
    if total > 0:
        baseline_share = political_count / total
        max_curve_rank = min(total, 1000)
        curve_ranks = np.arange(1, max_curve_rank + 1)
        for rank_column, label in (
            ("model_a_rank", "Model A"),
            ("model_b_rank", "Model B"),
        ):
            ranks = np.asarray(scores_df[rank_column].to_list(), dtype=np.int64)
            political_by_rank = np.zeros(total, dtype=np.int64)
            political_by_rank[ranks - 1] = political_mask.astype(np.int64)
            cumulative_political = np.cumsum(political_by_rank)
            axes[0].plot(
                curve_ranks,
                cumulative_political[:max_curve_rank] / curve_ranks,
                label=label,
            )
        axes[0].axhline(
            baseline_share,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Evaluation population",
        )
        axes[0].legend()
        if max_curve_rank > 10:
            axes[0].set_xscale("log")

        model_a_political_percentiles = np.asarray(
            scores_df.filter(pl.col("is_political"))["model_a_rank_percentile"].to_list(),
            dtype=np.float64,
        )
        model_b_political_percentiles = np.asarray(
            scores_df.filter(pl.col("is_political"))["model_b_rank_percentile"].to_list(),
            dtype=np.float64,
        )
        if model_a_political_percentiles.size:
            axes[1].hist(
                model_a_political_percentiles,
                bins=30,
                density=True,
                alpha=0.55,
                label="Model A",
            )
            axes[1].hist(
                model_b_political_percentiles,
                bins=30,
                density=True,
                alpha=0.55,
                label="Model B",
            )
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, "No political posts", ha="center", va="center")
    else:
        axes[0].text(0.5, 0.5, "No scored posts", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No scored posts", ha="center", va="center")
    axes[0].set_title("Political share among top-ranked posts")
    axes[0].set_xlabel("Top K")
    axes[0].set_ylabel("Political share")
    axes[1].set_title("Political-post rank percentiles")
    axes[1].set_xlabel("Rank percentile (0 = highest)")
    axes[1].set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(output_dir / "political_rank_comparison.png", dpi=160)
    plt.close(fig)


def print_console_summary(summary: Mapping[str, Any], output_dir: Path) -> None:
    inference = summary["inference"]
    diagnostics = summary["retrieval_diagnostics"]
    comparison = summary["comparison"]
    print("\nBST political-score comparison")
    print(
        f"  Evaluation URIs: {inference['evaluation_uris']:,} "
        f"({inference['political_uris_selected']:,} sampled political, "
        f"{inference['non_political_uris_selected']:,} sampled non-political) from "
        f"{inference['files_scanned']:,} inference files"
    )
    print(
        f"  Elasticsearch: {diagnostics['documents_found']:,} documents found; "
        f"{diagnostics['scored_posts']:,} posts scored; "
        f"{diagnostics['missing_documents']:,} documents missing"
    )
    print(
        f"  Model A higher: {comparison['model_a_higher_count']:,} | "
        f"Model B higher: {comparison['model_b_higher_count']:,} | "
        f"Ties: {comparison['tie_count']:,}"
    )
    delta = comparison["model_b_minus_model_a"]
    print(f"  B - A mean: {delta['mean']} | median: {delta['median']}")
    for model_label in ("A", "B"):
        ranking = summary["models"][model_label]["ranking"]
        top_100 = ranking["top_cutoffs"]["100"]
        top_100_share = top_100["political_share"]
        formatted_share = (
            f"{100.0 * top_100_share:.1f}%" if top_100_share is not None else "n/a"
        )
        print(
            f"  Model {model_label}: political average rank="
            f"{ranking['political_average_rank']}, median rank="
            f"{ranking['political_median_rank']}, top-100 political share="
            f"{formatted_share} ({top_100['political_posts']}/{top_100['actual_posts']}), "
            f"ROC AUC={ranking['political_vs_non_political_roc_auc']}"
        )
    print(f"  Output: {output_dir}")


def _create_output_dir(requested: Optional[Path], evaluated_at: datetime) -> Path:
    if requested is None:
        timestamp = evaluated_at.strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_ROOT / f"bst_political_{timestamp}"
    else:
        output_dir = requested.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


async def run(args: argparse.Namespace, environ: Mapping[str, str]) -> Path:
    if not 0.0 <= args.political_threshold <= 1.0:
        raise ValueError("--political-threshold must be between 0 and 1")
    if args.elasticsearch_batch_size <= 0:
        raise ValueError("--elasticsearch-batch-size must be positive")
    if args.elasticsearch_timeout_seconds <= 0:
        raise ValueError("--elasticsearch-timeout-seconds must be positive")
    if args.class_sample_size < 0:
        raise ValueError("--class-sample-size must be non-negative")

    evaluated_at = datetime.now(timezone.utc)
    window = resolve_date_window(args.start_date, args.end_date)
    elasticsearch_url, api_key = resolve_elasticsearch_settings(
        args.elasticsearch_url, args.elasticsearch_api_key, environ
    )
    if args.elasticsearch_insecure and not is_loopback_elasticsearch_url(elasticsearch_url):
        raise ValueError("--elasticsearch-insecure is only allowed for a loopback Elasticsearch URL")
    output_dir = _create_output_dir(args.output_dir, evaluated_at)
    device = torch.device(args.device)

    LOGGER.info("Loading BST models")
    model_a = load_model_bundle(
        "A", args.model_a_dir, args.model_a_get_data_dir, device
    )
    model_b = load_model_bundle(
        "B", args.model_b_dir, args.model_b_get_data_dir, device
    )
    embedding_dim = validate_model_compatibility(model_a, model_b)

    LOGGER.info(
        "Listing inference parquets for [%s, %s)",
        window.start.isoformat(),
        window.end.isoformat(),
    )
    inference_paths = list_inference_parquet_paths(
        storage.Client(),
        args.gcs_bucket,
        args.inference_prefix,
        window,
    )
    LOGGER.info("Extracting political labels from %d inference files", len(inference_paths))
    evaluation_df, inference_selection_stats = build_evaluation_posts_df(
        inference_paths,
        args.political_threshold,
        args.class_sample_size,
        args.random_seed,
    )
    inference_rows = {
        str(row["at_uri"]): row for row in evaluation_df.iter_rows(named=True)
    }
    evaluation_uris = list(inference_rows)
    LOGGER.info(
        "Selected %d evaluation URIs: %d political and %d non-political "
        "(%d political and %d non-political available)",
        inference_selection_stats["evaluation_uris"],
        inference_selection_stats["political_uris_selected"],
        inference_selection_stats["non_political_uris_selected"],
        inference_selection_stats["political_uris_available"],
        inference_selection_stats["non_political_uris_available"],
    )

    diagnostics = RetrievalDiagnostics()
    result_rows: list[dict[str, Any]] = []
    embedding_field = f"embeddings.{args.embedding_model}"
    timeout = httpx.Timeout(args.elasticsearch_timeout_seconds)
    LOGGER.info(
        "Hydrating posts from Elasticsearch %s index=%s in batches of %d",
        sanitize_elasticsearch_url(elasticsearch_url),
        args.elasticsearch_index,
        args.elasticsearch_batch_size,
    )
    async with httpx.AsyncClient(
        timeout=timeout,
        verify=not args.elasticsearch_insecure,
    ) as client:
        batch_starts = range(0, len(evaluation_uris), args.elasticsearch_batch_size)
        for start in tqdm(batch_starts, desc="Hydrating and scoring", unit="batch"):
            batch_uris = evaluation_uris[start : start + args.elasticsearch_batch_size]
            candidates = await fetch_elasticsearch_batch(
                client,
                elasticsearch_url,
                args.elasticsearch_index,
                api_key,
                batch_uris,
                embedding_field,
                embedding_dim,
                diagnostics,
            )
            if not candidates:
                continue
            candidate_embeddings = torch.tensor(
                [post.embedding for post in candidates],
                device=device,
                dtype=torch.float32,
            )
            model_a_scores = score_model_batch(
                model_a, candidate_embeddings, candidates, device
            )
            model_b_scores = score_model_batch(
                model_b, candidate_embeddings, candidates, device
            )
            for index, post in enumerate(candidates):
                inference_row = inference_rows[post.at_uri]
                result_rows.append(
                    {
                        "at_uri": post.at_uri,
                        "author_did": post.author_did,
                        "created_at": post.created_at,
                        "elasticsearch_indexed_at": post.indexed_at,
                        "content": post.content,
                        "current_like_count": post.like_count,
                        "news_social_concern_score": float(
                            inference_row["news_social_concern_score"]
                        ),
                        "inference_indexed_at": inference_row["inference_indexed_at"],
                        "is_political": bool(inference_row["is_political"]),
                        "model_a_logit": float(model_a_scores[index]),
                        "model_b_logit": float(model_b_scores[index]),
                    }
                )
            del candidate_embeddings, candidates, model_a_scores, model_b_scores

    diagnostics.scored_posts = len(result_rows)
    scores_df = build_scores_frame(result_rows)
    scores_path = output_dir / "political_comparison_scores.parquet"
    scores_df.write_parquet(scores_path, compression="zstd")
    summary = build_summary(
        args,
        output_dir,
        window,
        inference_paths,
        inference_selection_stats,
        scores_df,
        diagnostics,
        model_a,
        model_b,
        elasticsearch_url,
        evaluated_at,
    )
    _write_json(output_dir / "summary.json", summary)
    write_plots(scores_df, output_dir)
    print_console_summary(summary, output_dir)
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two TorchScript BST rankers on all recent political posts and a "
            "sample of non-political posts"
        )
    )
    parser.add_argument("model_a_dir", type=Path)
    parser.add_argument("model_b_dir", type=Path)
    parser.add_argument("--model-a-get-data-dir", type=Path)
    parser.add_argument("--model-b-get-data-dir", type=Path)
    parser.add_argument("--gcs-bucket", default=DEFAULT_GCS_BUCKET)
    parser.add_argument("--inference-prefix", default=DEFAULT_INFERENCE_PREFIX)
    parser.add_argument(
        "--start-date",
        required=True,
        help="Inclusive UTC start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Exclusive UTC end date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--political-threshold",
        type=float,
        default=DEFAULT_POLITICAL_THRESHOLD,
        help=(
            "Minimum topic.News & Social Concern score used to classify political "
            f"posts (default: {DEFAULT_POLITICAL_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--class-sample-size",
        type=int,
        default=10_000,
        help="Number of political and non-political posts to sample (default: 10000)",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--elasticsearch-url")
    parser.add_argument("--elasticsearch-api-key")
    parser.add_argument("--elasticsearch-index", default=DEFAULT_ELASTICSEARCH_INDEX)
    parser.add_argument("--elasticsearch-batch-size", type=int, default=1000)
    parser.add_argument("--elasticsearch-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--elasticsearch-insecure",
        dest="elasticsearch_insecure",
        action="store_true",
        help="Disable Elasticsearch TLS certificate verification (default)",
    )
    parser.add_argument(
        "--elasticsearch-secure",
        dest="elasticsearch_insecure",
        action="store_false",
        help="Verify Elasticsearch TLS certificates",
    )
    parser.set_defaults(elasticsearch_insecure=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        asyncio.run(run(args, os.environ))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
