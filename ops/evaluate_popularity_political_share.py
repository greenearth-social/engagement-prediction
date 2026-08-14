#!/usr/bin/env python3
"""Compare two heavy rankers on popularity candidates and politicalness."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx
import numpy as np
import polars as pl
import requests
import torch
from google.cloud import storage
from tqdm import tqdm


OPS_DIR = Path(__file__).resolve().parent
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

import compare_bst_political_scores as bst_comparison


LOGGER = logging.getLogger("evaluate_popularity_political_share")

DEFAULT_API_URL = "https://api.greenearth.social/"
DEFAULT_GCS_BUCKET = "greenearth-471522-ingex-extract-prod"
DEFAULT_INFERENCE_PREFIX = "bsky_inferences"
DEFAULT_OUTPUT_ROOT = Path("/mnt/data/dave/outputs/compare")
DEFAULT_USER_DID = "did:plc:aaaaaaaaaaaaaaaaaaaaaaaa"
DEFAULT_POLITICAL_THRESHOLD = 0.95
INFERENCE_BUFFER_DAYS = 6
SUPPORTED_MAX_AGE_HOURS = (6, 12, 24, 48, 72, 168)
SCORE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
POLITICAL_SCORE_FIELD = "topic.News & Social Concern"
POOL_NAMES = ("small", "large")
SPLIT_KEYS = ("full_pool", "candidate_score", "model_a", "model_b")
AT_URI_POST_RE = re.compile(
    r"^at://([^/]+)/app\.bsky\.feed\.post/([^/?#]+)/?(?:[?#].*)?$"
)

POLITICAL_INFERENCE_DTYPE = pl.Struct({
    "text": pl.Struct({
        "message.commit.record.text": pl.Struct({
            "topic": pl.Struct({
                "News & Social Concern": pl.Float64,
            }),
        }),
    }),
})

CANDIDATE_SCHEMA = {
    "candidate_pool": pl.String,
    "requested_pool_size": pl.Int64,
    "candidate_rank": pl.Int64,
    "at_uri": pl.String,
    "url": pl.String,
    "popularity_score": pl.Float64,
    "author_did": pl.String,
    "content": pl.String,
    "like_count": pl.Int64,
    "generator_name": pl.String,
}

INFERENCE_SCORE_SCHEMA = {
    "at_uri": pl.String,
    "news_social_concern_score": pl.Float64,
    "inference_indexed_at": pl.String,
    "is_political": pl.Boolean,
}

MODEL_SCORE_SCHEMA = {
    "at_uri": pl.String,
    "scoring_author_did": pl.String,
    "created_at": pl.String,
    "elasticsearch_indexed_at": pl.String,
    "scoring_content": pl.String,
    "current_like_count": pl.Int64,
    "model_a_logit": pl.Float64,
    "model_b_logit": pl.Float64,
}

RANKING_SCHEMA = {
    "candidate_score_rank": pl.Int64,
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

OUTPUT_SCHEMA = {
    **CANDIDATE_SCHEMA,
    **{key: value for key, value in INFERENCE_SCORE_SCHEMA.items() if key != "at_uri"},
    "created_at": pl.String,
    "elasticsearch_indexed_at": pl.String,
    "current_like_count": pl.Int64,
    "model_a_logit": pl.Float64,
    "model_b_logit": pl.Float64,
    "model_scored": pl.Boolean,
    "scoring_status": pl.String,
    **RANKING_SCHEMA,
}


@dataclass(frozen=True)
class InferenceWindow:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class CandidateDiagnostics:
    requested_candidates: int
    api_candidates_returned: int
    unique_candidate_uris: int
    candidates_missing_at_uri: int
    duplicate_candidate_uris: int


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def at_uri_to_url(at_uri: str) -> str:
    match = AT_URI_POST_RE.match(at_uri)
    if not match:
        raise ValueError(f"Could not parse AT URI: {at_uri}")
    author_did, post_id = match.groups()
    return f"https://bsky.app/profile/{author_did}/post/{post_id}"


def normalize_api_url(raw_url: str) -> str:
    url = raw_url.strip().rstrip("/")
    if not url:
        raise ValueError("GE_API_URL must not be empty")
    if "://" not in url:
        url = f"https://{url}"
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid GE_API_URL: {raw_url!r}")
    return url


def sanitize_api_url(raw_url: str) -> str:
    parsed = urlsplit(normalize_api_url(raw_url))
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    sanitized = SplitResult(parsed.scheme, netloc, parsed.path, "", "")
    return urlunsplit(sanitized).rstrip("/")


def resolve_api_settings(environ: Mapping[str, str]) -> tuple[str, str]:
    api_url = normalize_api_url(environ.get("GE_API_URL", DEFAULT_API_URL))
    api_key = environ.get("GE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GE_API_KEY is required")
    return api_url, api_key


def validate_parameters(
    small_num_candidates: int,
    large_num_candidates: int,
    top_k: int,
    max_age_hours: int,
    api_timeout_seconds: float,
    political_threshold: float,
    elasticsearch_batch_size: int,
    elasticsearch_timeout_seconds: float,
) -> None:
    if not 1 <= small_num_candidates <= 1000:
        raise ValueError("--small-num-candidates must be between 1 and 1000")
    if not 1 <= large_num_candidates <= 1000:
        raise ValueError("--large-num-candidates must be between 1 and 1000")
    if small_num_candidates > large_num_candidates:
        raise ValueError(
            "--small-num-candidates must be less than or equal to "
            "--large-num-candidates"
        )
    if top_k <= 0:
        raise ValueError("--top-k must be positive")
    if max_age_hours not in SUPPORTED_MAX_AGE_HOURS:
        supported = ", ".join(str(value) for value in SUPPORTED_MAX_AGE_HOURS)
        raise ValueError(f"--max-age-hours must be one of: {supported}")
    if not math.isfinite(api_timeout_seconds) or api_timeout_seconds <= 0.0:
        raise ValueError("--api-timeout-seconds must be positive")
    if not 0.0 <= political_threshold <= 1.0:
        raise ValueError("--political-threshold must be between 0 and 1")
    if elasticsearch_batch_size <= 0:
        raise ValueError("--elasticsearch-batch-size must be positive")
    if (
        not math.isfinite(elasticsearch_timeout_seconds)
        or elasticsearch_timeout_seconds <= 0.0
    ):
        raise ValueError("--elasticsearch-timeout-seconds must be positive")


def derive_inference_window(
    request_time: datetime,
    max_age_hours: int,
) -> InferenceWindow:
    if request_time.tzinfo is None:
        raise ValueError("request_time must be timezone-aware")
    request_time_utc = request_time.astimezone(timezone.utc)
    padding = timedelta(days=INFERENCE_BUFFER_DAYS)
    return InferenceWindow(
        start=request_time_utc - timedelta(hours=max_age_hours) - padding,
        end=request_time_utc + padding,
    )


def _inference_blob_timestamp(
    blob_name: str,
    inference_prefix: str,
) -> Optional[datetime]:
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
    window: InferenceWindow,
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
    return [path for _, path in paths_with_timestamps]


def fetch_popularity_candidates(
    session: Any,
    api_url: str,
    api_key: str,
    num_candidates: int,
    max_age_hours: int,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    try:
        response = session.post(
            f"{api_url}/candidates/generate",
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "generators": [{"name": "popularity", "weight": 1.0}],
                "user_did": DEFAULT_USER_DID,
                "num_candidates": num_candidates,
                "video_only": False,
                "max_age_hours": max_age_hours,
                "exclude_uris": [],
            },
            timeout=timeout_seconds,
        )
    except requests.Timeout as exc:
        raise RuntimeError(
            f"Green Earth API request timed out after {timeout_seconds:g} seconds"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError("Green Earth API request failed") from exc

    if not 200 <= response.status_code < 300:
        raise RuntimeError(
            f"Green Earth API request failed with HTTP {response.status_code}"
        )
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Green Earth API returned invalid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise RuntimeError("Green Earth API response is missing a candidates list")
    candidates = payload["candidates"]
    if not all(isinstance(candidate, dict) for candidate in candidates):
        raise RuntimeError("Green Earth API candidates must be JSON objects")
    return candidates


def _optional_string(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _optional_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _optional_nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def build_candidate_frame(
    candidates: Sequence[Mapping[str, Any]],
    requested_candidates: int,
    candidate_pool: str,
) -> tuple[pl.DataFrame, CandidateDiagnostics]:
    rows: list[dict[str, Any]] = []
    seen_uris: set[str] = set()
    missing_at_uri = 0
    duplicate_uris = 0
    for candidate_rank, candidate in enumerate(candidates, start=1):
        at_uri = candidate.get("at_uri")
        if not isinstance(at_uri, str) or not at_uri.strip():
            missing_at_uri += 1
            continue
        if at_uri in seen_uris:
            duplicate_uris += 1
            continue
        seen_uris.add(at_uri)
        rows.append({
            "candidate_pool": candidate_pool,
            "requested_pool_size": requested_candidates,
            "candidate_rank": candidate_rank,
            "at_uri": at_uri,
            "url": at_uri_to_url(at_uri),
            "popularity_score": _optional_float(candidate.get("score")),
            "author_did": _optional_string(candidate.get("author_did")),
            "content": _optional_string(candidate.get("content")),
            "like_count": _optional_nonnegative_int(candidate.get("like_count")),
            "generator_name": _optional_string(candidate.get("generator_name")),
        })
    frame = (
        pl.DataFrame(rows, schema=CANDIDATE_SCHEMA)
        if rows
        else pl.DataFrame(schema=CANDIDATE_SCHEMA)
    )
    diagnostics = CandidateDiagnostics(
        requested_candidates=requested_candidates,
        api_candidates_returned=len(candidates),
        unique_candidate_uris=frame.height,
        candidates_missing_at_uri=missing_at_uri,
        duplicate_candidate_uris=duplicate_uris,
    )
    return frame, diagnostics


def build_candidate_inference_scores(
    inference_paths: Sequence[str],
    candidate_uris: Sequence[str],
    political_threshold: float,
) -> pl.DataFrame:
    if not inference_paths or not candidate_uris:
        return pl.DataFrame(schema=INFERENCE_SCORE_SCHEMA)

    candidate_uris_lf = pl.DataFrame({
        "at_uri": pl.Series(candidate_uris, dtype=pl.String),
    }).lazy()
    parsed_record_expr = (
        pl.col("_parsed_inferences")
        .struct.field("text")
        .struct.field("message.commit.record.text")
    )
    matched = (
        pl.scan_parquet(list(inference_paths))
        .select(["at_uri", "indexed_at", "inferences"])
        .join(candidate_uris_lf, on="at_uri", how="semi")
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
            .alias("news_social_concern_score"),
        )
        .select(
            "at_uri",
            pl.col("indexed_at").cast(pl.String).alias("inference_indexed_at"),
            "_indexed_at_dt",
            "news_social_concern_score",
        )
        .collect(engine="streaming")
    )
    if matched.is_empty():
        return pl.DataFrame(schema=INFERENCE_SCORE_SCHEMA)

    latest = (
        matched
        .sort(
            ["at_uri", "_indexed_at_dt", "inference_indexed_at"],
            nulls_last=False,
        )
        .unique(subset=["at_uri"], keep="last", maintain_order=True)
        .with_columns(
            pl.when(
                pl.col("news_social_concern_score").is_not_null()
                & pl.col("news_social_concern_score").is_finite()
            )
            .then(pl.col("news_social_concern_score") >= political_threshold)
            .otherwise(None)
            .alias("is_political")
        )
        .select(*INFERENCE_SCORE_SCHEMA)
    )
    return latest


def build_model_scores_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return (
        pl.DataFrame(rows, schema=MODEL_SCORE_SCHEMA)
        if rows
        else pl.DataFrame(schema=MODEL_SCORE_SCHEMA)
    )


def rank_results_within_pools(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=OUTPUT_SCHEMA)

    rows = frame.to_dicts()
    pool_names = list(dict.fromkeys(str(row["candidate_pool"]) for row in rows))
    for pool_name in pool_names:
        pool_indices = [
            index
            for index, row in enumerate(rows)
            if row["candidate_pool"] == pool_name
        ]
        pool_uris = [str(rows[index]["at_uri"]) for index in pool_indices]
        popularity_scores = [
            (
                float(rows[index]["popularity_score"])
                if rows[index]["popularity_score"] is not None
                else float("nan")
            )
            for index in pool_indices
        ]
        candidate_score_ranks = bst_comparison.ordinal_ranks_descending(
            popularity_scores,
            pool_uris,
        )
        for position, row_index in enumerate(pool_indices):
            rows[row_index]["candidate_score_rank"] = int(
                candidate_score_ranks[position]
            )

        scored_indices = [
            index for index in pool_indices if bool(rows[index]["model_scored"])
        ]
        if not scored_indices:
            continue
        scored_uris = [str(rows[index]["at_uri"]) for index in scored_indices]
        model_a_logits = np.asarray(
            [rows[index]["model_a_logit"] for index in scored_indices],
            dtype=np.float64,
        )
        model_b_logits = np.asarray(
            [rows[index]["model_b_logit"] for index in scored_indices],
            dtype=np.float64,
        )
        model_a_ranks = bst_comparison.ordinal_ranks_descending(
            model_a_logits,
            scored_uris,
        )
        model_b_ranks = bst_comparison.ordinal_ranks_descending(
            model_b_logits,
            scored_uris,
        )
        model_a_normalized = bst_comparison.normalize_logits(model_a_logits)
        model_b_normalized = bst_comparison.normalize_logits(model_b_logits)
        rank_denominator = max(1, len(scored_indices) - 1)
        for position, row_index in enumerate(scored_indices):
            delta = float(model_b_logits[position] - model_a_logits[position])
            rows[row_index]["model_b_minus_model_a"] = delta
            rows[row_index]["model_a_normalized_score"] = float(
                model_a_normalized[position]
            )
            rows[row_index]["model_b_normalized_score"] = float(
                model_b_normalized[position]
            )
            rows[row_index]["model_a_rank"] = int(model_a_ranks[position])
            rows[row_index]["model_b_rank"] = int(model_b_ranks[position])
            rows[row_index]["model_a_rank_percentile"] = float(
                (model_a_ranks[position] - 1) / rank_denominator
            )
            rows[row_index]["model_b_rank_percentile"] = float(
                (model_b_ranks[position] - 1) / rank_denominator
            )
            rows[row_index]["model_b_minus_model_a_rank"] = int(
                model_b_ranks[position] - model_a_ranks[position]
            )
            rows[row_index]["higher_scoring_model"] = (
                "B" if delta > 0 else "A" if delta < 0 else "tie"
            )
    return pl.DataFrame(rows, schema=OUTPUT_SCHEMA)


def attach_results(
    candidates_df: pl.DataFrame,
    inference_scores_df: pl.DataFrame,
    model_scores_df: pl.DataFrame,
) -> pl.DataFrame:
    if candidates_df.is_empty():
        return pl.DataFrame(schema=OUTPUT_SCHEMA)
    attached = (
        candidates_df
        .join(inference_scores_df, on="at_uri", how="left")
        .join(model_scores_df, on="at_uri", how="left")
        .with_columns(
            pl.coalesce("scoring_author_did", "author_did").alias("author_did"),
            pl.coalesce("scoring_content", "content").alias("content"),
            (
                pl.col("model_a_logit").is_not_null()
                & pl.col("model_a_logit").is_finite()
                & pl.col("model_b_logit").is_not_null()
                & pl.col("model_b_logit").is_finite()
            ).alias("model_scored"),
            pl.when(
                pl.col("model_a_logit").is_not_null()
                & pl.col("model_a_logit").is_finite()
                & pl.col("model_b_logit").is_not_null()
                & pl.col("model_b_logit").is_finite()
            )
            .then(pl.lit("scored"))
            .otherwise(pl.lit("unscored"))
            .alias("scoring_status"),
            *[
                pl.lit(None, dtype=dtype).alias(column)
                for column, dtype in RANKING_SCHEMA.items()
            ],
        )
        .select(*OUTPUT_SCHEMA)
    )
    return rank_results_within_pools(attached)


def _linear_quantile(sorted_values: Sequence[float], quantile: float) -> float:
    position = quantile * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * (position - lower_index)


def score_distribution(values: Sequence[Optional[float]]) -> dict[str, Any]:
    finite_values: list[float] = []
    missing_count = 0
    non_finite_count = 0
    for value in values:
        if value is None:
            missing_count += 1
        elif not math.isfinite(value):
            non_finite_count += 1
        else:
            finite_values.append(float(value))
    finite_values.sort()

    result: dict[str, Any] = {
        "total_candidates": len(values),
        "finite_score_count": len(finite_values),
        "missing_score_count": missing_count,
        "non_finite_score_count": non_finite_count,
        "mean": None,
        "population_stddev": None,
        "min": None,
        "p10": None,
        "p25": None,
        "median": None,
        "p75": None,
        "p90": None,
        "p95": None,
        "p99": None,
        "max": None,
    }
    if not finite_values:
        return result

    mean = sum(finite_values) / len(finite_values)
    result.update({
        "mean": mean,
        "population_stddev": math.sqrt(
            sum((value - mean) ** 2 for value in finite_values)
            / len(finite_values)
        ),
        "min": finite_values[0],
        "max": finite_values[-1],
    })
    for quantile in SCORE_QUANTILES:
        key = "median" if quantile == 0.50 else f"p{round(quantile * 100)}"
        result[key] = _linear_quantile(finite_values, quantile)
    return result


def inference_score_summary(frame: pl.DataFrame) -> dict[str, Any]:
    total = frame.height
    matched = (
        frame["inference_indexed_at"].is_not_null().sum()
        if total
        else 0
    )
    news_values = frame["news_social_concern_score"].to_list() if total else []
    known_labels = sum(
        value is not None and math.isfinite(value) for value in news_values
    )
    political_count = (
        int(frame["is_political"].fill_null(False).sum()) if total else 0
    )
    return {
        "total_candidates": total,
        "matched_inference_candidates": matched,
        "candidates_without_inference": total - matched,
        "inference_coverage": matched / total if total else None,
        "known_political_labels": known_labels,
        "candidates_without_political_score": total - known_labels,
        "political_candidates": political_count,
        "non_political_candidates": known_labels - political_count,
        "political_share_known_labels": (
            political_count / known_labels if known_labels else None
        ),
        "political_share_all_candidates_lower_bound": (
            political_count / total if total else None
        ),
        "news_social_concern_score": score_distribution(news_values),
    }


def graded_ndcg(relevances_in_rank_order: Sequence[float], cutoff: int) -> Optional[float]:
    if cutoff <= 0 or not relevances_in_rank_order:
        return None
    relevances = [float(value) for value in relevances_in_rank_order]
    actual_cutoff = min(cutoff, len(relevances))

    def dcg(values: Sequence[float]) -> float:
        return sum(
            value / math.log2(rank + 2)
            for rank, value in enumerate(values[:actual_cutoff])
        )

    ideal = dcg(sorted(relevances, reverse=True))
    if ideal <= 0.0:
        return None
    return dcg(relevances) / ideal


def graded_ndcg_for_ranking(
    population: pl.DataFrame,
    rank_column: str,
    top_k: int,
) -> tuple[Optional[float], int, int]:
    known = (
        population
        .filter(
            pl.col("news_social_concern_score").is_not_null()
            & pl.col("news_social_concern_score").is_finite()
            & pl.col(rank_column).is_not_null()
        )
        .sort(rank_column)
    )
    relevances = known["news_social_concern_score"].to_list()
    actual_cutoff = min(top_k, len(relevances))
    return graded_ndcg(relevances, top_k), len(relevances), actual_cutoff


def _split_metric_row(
    pool_name: str,
    requested_pool_size: int,
    returned_pool_size: int,
    split_key: str,
    split_label: str,
    selected: pl.DataFrame,
    ranking_population_count: int,
    requested_top_k: Optional[int],
    ndcg: Optional[float],
    ndcg_known_label_count: int,
    ndcg_actual_cutoff: int,
) -> dict[str, Any]:
    summary = inference_score_summary(selected)
    distribution = summary["news_social_concern_score"]
    return {
        "candidate_pool": pool_name,
        "requested_pool_size": requested_pool_size,
        "returned_pool_size": returned_pool_size,
        "split_key": split_key,
        "split_label": split_label,
        "requested_top_k": requested_top_k,
        "ranking_population_count": ranking_population_count,
        "selected_count": selected.height,
        "known_label_count": summary["known_political_labels"],
        "political_count": summary["political_candidates"],
        "political_share": summary["political_share_known_labels"],
        "ndcg": ndcg,
        "ndcg_known_label_count": ndcg_known_label_count,
        "ndcg_actual_cutoff": ndcg_actual_cutoff,
        "news_social_concern_mean": distribution["mean"],
        "news_social_concern_median": distribution["median"],
    }


def _top_by_rank(
    population: pl.DataFrame,
    rank_column: str,
    top_k: int,
) -> pl.DataFrame:
    actual_cutoff = min(top_k, population.height)
    return population.filter(pl.col(rank_column) <= actual_cutoff).sort(rank_column)


def build_split_metrics(
    output_frame: pl.DataFrame,
    top_k: int,
    requested_pool_sizes: Optional[Mapping[str, int]] = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pool_name in POOL_NAMES:
        pool = output_frame.filter(pl.col("candidate_pool") == pool_name)
        requested_pool_size = (
            int(pool["requested_pool_size"][0])
            if not pool.is_empty()
            else int((requested_pool_sizes or {}).get(pool_name, 0))
        )
        returned_pool_size = pool.height
        full_summary = inference_score_summary(pool)
        rows.append(
            _split_metric_row(
                pool_name,
                requested_pool_size,
                returned_pool_size,
                "full_pool",
                "Full candidate pool",
                pool,
                pool.height,
                None,
                None,
                full_summary["known_political_labels"],
                0,
            )
        )

        ranking_specs = (
            ("candidate_score", "Candidate score", pool, "candidate_score_rank"),
            (
                "model_a",
                "Model A",
                pool.filter(pl.col("model_scored")),
                "model_a_rank",
            ),
            (
                "model_b",
                "Model B",
                pool.filter(pl.col("model_scored")),
                "model_b_rank",
            ),
        )
        for split_key, split_label, population, rank_column in ranking_specs:
            selected = _top_by_rank(population, rank_column, top_k)
            ndcg, ndcg_known_count, ndcg_actual_cutoff = graded_ndcg_for_ranking(
                population,
                rank_column,
                top_k,
            )
            rows.append(
                _split_metric_row(
                    pool_name,
                    requested_pool_size,
                    returned_pool_size,
                    split_key,
                    f"{split_label} top {top_k}",
                    selected,
                    population.height,
                    top_k,
                    ndcg,
                    ndcg_known_count,
                    ndcg_actual_cutoff,
                )
            )
    return rows


def create_output_dir(
    requested: Optional[Path],
    request_time: datetime,
) -> Path:
    if requested is None:
        timestamp = request_time.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_ROOT / f"popularity_political_{timestamp}"
    else:
        output_dir = requested.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def build_summary(
    args: argparse.Namespace,
    request_time: datetime,
    inference_window: InferenceWindow,
    inference_paths: Sequence[str],
    candidate_diagnostics: Mapping[str, CandidateDiagnostics],
    retrieval_diagnostics: bst_comparison.RetrievalDiagnostics,
    output_frame: pl.DataFrame,
    split_metrics: Sequence[Mapping[str, Any]],
    model_a: bst_comparison.ModelBundle,
    model_b: bst_comparison.ModelBundle,
    api_url: str,
    elasticsearch_url: str,
    output_dir: Path,
) -> dict[str, Any]:
    pool_summaries = {}
    for pool_name in POOL_NAMES:
        pool = output_frame.filter(pl.col("candidate_pool") == pool_name)
        pool_summaries[pool_name] = {
            **inference_score_summary(pool),
            "model_scored_candidates": int(
                pool["model_scored"].fill_null(False).sum()
            ) if not pool.is_empty() else 0,
        }
    return {
        "evaluated_at": request_time.astimezone(timezone.utc).isoformat(),
        "run_config": {
            "api_url": sanitize_api_url(api_url),
            "model_a_dir": str(model_a.model_dir),
            "model_b_dir": str(model_b.model_dir),
            "model_a_author_idx_path": (
                str(model_a.author_idx_path) if model_a.author_idx_path else None
            ),
            "model_b_author_idx_path": (
                str(model_b.author_idx_path) if model_b.author_idx_path else None
            ),
            "small_num_candidates": args.small_num_candidates,
            "large_num_candidates": args.large_num_candidates,
            "top_k": args.top_k,
            "max_age_hours": args.max_age_hours,
            "api_timeout_seconds": args.api_timeout_seconds,
            "gcs_bucket": args.gcs_bucket,
            "inference_prefix": args.inference_prefix,
            "inference_buffer_days": INFERENCE_BUFFER_DAYS,
            "political_score_field": POLITICAL_SCORE_FIELD,
            "political_threshold": args.political_threshold,
            "elasticsearch_url": bst_comparison.sanitize_elasticsearch_url(
                elasticsearch_url
            ),
            "elasticsearch_index": args.elasticsearch_index,
            "elasticsearch_batch_size": args.elasticsearch_batch_size,
            "elasticsearch_timeout_seconds": args.elasticsearch_timeout_seconds,
            "elasticsearch_insecure": args.elasticsearch_insecure,
            "embedding_field": f"embeddings.{args.embedding_model}",
            "device": args.device,
            "output_dir": str(output_dir),
        },
        "candidate_window": {
            "start_inclusive": (
                request_time.astimezone(timezone.utc)
                - timedelta(hours=args.max_age_hours)
            ).isoformat(),
            "end_inclusive": request_time.astimezone(timezone.utc).isoformat(),
        },
        "inference_export_window": {
            "start_inclusive": inference_window.start.isoformat(),
            "end_exclusive": inference_window.end.isoformat(),
            "files_scanned": len(inference_paths),
        },
        "api_candidates": {
            pool_name: asdict(candidate_diagnostics[pool_name])
            for pool_name in POOL_NAMES
        },
        "retrieval_diagnostics": asdict(retrieval_diagnostics),
        "pool_summaries": pool_summaries,
        "split_metrics": [dict(row) for row in split_metrics],
        "models": {
            "A": {
                "model_dir": str(model_a.model_dir),
                "use_popularity_feature": model_a.use_popularity_feature,
            },
            "B": {
                "model_dir": str(model_b.model_dir),
                "use_popularity_feature": model_b.use_popularity_feature,
            },
        },
        "score_semantics": (
            "The topic.News & Social Concern score is retained from the latest "
            "inference row per candidate URI. Missing inference rows and missing "
            "score fields remain null. Political classification uses an inclusive "
            "greater-than-or-equal threshold."
        ),
        "ranking_semantics": (
            "Candidate-score and model ranks are computed independently in each "
            "candidate pool. Rank 1 is the highest score and equal scores are "
            "ordered by at_uri. Only candidates successfully hydrated and scored by "
            "both models receive model ranks. A requested top-k larger than its "
            "ranking population is capped to the available population."
        ),
        "political_share_semantics": (
            "Political share divides by selected candidates with a finite News & "
            "Social Concern score. Unknown scores are excluded and are exposed by "
            "selected_count and known_label_count."
        ),
        "ndcg_methodology": (
            "NDCG uses the continuous News & Social Concern score as linear gain and "
            "log2 rank discount. Unknown labels are removed while preserving relative "
            "rank order, then NDCG is calculated at the effective top-k. Candidate "
            "score uses the returned candidate population; model rows use the "
            "successfully model-scored population. Full-pool NDCG is unavailable. "
            "Higher NDCG means more News & Social Concern content is concentrated "
            "near the top."
        ),
        "presentation_semantics": (
            "Console, split_comparison.csv, and split_comparison.png values are "
            "rounded to two decimal places. summary.json and the candidate Parquet "
            "retain raw precision."
        ),
        "distribution_methodology": {
            "quantiles": "linear interpolation between ordered finite scores",
            "dispersion": "population standard deviation",
        },
    }


def _format_percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _format_stat(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _presentation_row(metric: Mapping[str, Any]) -> dict[str, str]:
    return {
        "Pool": str(metric["candidate_pool"]),
        "Requested": str(metric["requested_pool_size"]),
        "Returned": str(metric["returned_pool_size"]),
        "Selection": str(metric["split_label"]),
        "Ranking population": str(metric["ranking_population_count"]),
        "Selected": str(metric["selected_count"]),
        "Known labels": str(metric["known_label_count"]),
        "Political share": _format_percent(metric["political_share"]),
        "NDCG": _format_stat(metric["ndcg"]),
        "Mean": _format_stat(metric["news_social_concern_mean"]),
        "Median": _format_stat(metric["news_social_concern_median"]),
    }


def _format_aligned_table(rows: Sequence[Mapping[str, str]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0])
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    separator = "  ".join("-" * widths[column] for column in columns)
    body = [
        "  ".join(str(row[column]).ljust(widths[column]) for column in columns)
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def write_split_comparison_csv(
    split_metrics: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Path:
    path = output_dir / "split_comparison.csv"
    rows = [_presentation_row(metric) for metric in split_metrics]
    if not rows:
        raise ValueError("Split comparison requires at least one row")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_split_comparison_plot(
    split_metrics: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = output_dir / "split_comparison.png"
    by_pool_and_split = {
        (str(row["candidate_pool"]), str(row["split_key"])): row
        for row in split_metrics
    }
    split_labels = ("Full pool", "Candidate score", "Model A", "Model B")
    panels = (
        ("political_share", "Political share (%)", 100.0, (0.0, 105.0)),
        ("ndcg", "NDCG", 1.0, (0.0, 1.08)),
        ("news_social_concern_mean", "Mean score", 1.0, (0.0, 1.08)),
        ("news_social_concern_median", "Median score", 1.0, (0.0, 1.08)),
    )
    colors = {"small": "#4C78A8", "large": "#F58518"}
    x_positions = np.arange(len(SPLIT_KEYS), dtype=np.float64)
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, (metric_name, title, scale, limits) in zip(axes.flat, panels):
        for pool_index, pool_name in enumerate(POOL_NAMES):
            offsets = x_positions + (pool_index - 0.5) * width
            values = []
            missing = []
            for split_key in SPLIT_KEYS:
                metric = by_pool_and_split[(pool_name, split_key)]
                value = metric[metric_name]
                missing.append(value is None)
                values.append(0.0 if value is None else float(value) * scale)
            bars = axis.bar(
                offsets,
                values,
                width,
                label=pool_name.capitalize(),
                color=colors[pool_name],
            )
            for bar, value, is_missing in zip(bars, values, missing):
                label = "n/a" if is_missing else f"{value:.2f}"
                label_height = 0.02 * limits[1] if is_missing else value
                axis.annotate(
                    label,
                    (bar.get_x() + bar.get_width() / 2, label_height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        axis.set_title(title)
        axis.set_ylim(*limits)
        axis.set_xticks(x_positions, split_labels)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="upper right")
    figure.suptitle("Popularity candidate ranking comparison", fontsize=15)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def print_console_summary(summary: Mapping[str, Any], output_dir: Path) -> None:
    inference_window = summary["inference_export_window"]
    print("\nPopularity candidate politicalness and model comparison")
    print(f"  Inference files scanned: {inference_window['files_scanned']:,}")
    print(_format_aligned_table([
        _presentation_row(metric) for metric in summary["split_metrics"]
    ]))
    print(f"  Output: {output_dir}")


async def run(
    args: argparse.Namespace,
    environ: Mapping[str, str],
    request_time: Optional[datetime] = None,
) -> Path:
    validate_parameters(
        args.small_num_candidates,
        args.large_num_candidates,
        args.top_k,
        args.max_age_hours,
        args.api_timeout_seconds,
        args.political_threshold,
        args.elasticsearch_batch_size,
        args.elasticsearch_timeout_seconds,
    )
    api_url, api_key = resolve_api_settings(environ)
    elasticsearch_url, elasticsearch_api_key = (
        bst_comparison.resolve_elasticsearch_settings(
            args.elasticsearch_url,
            args.elasticsearch_api_key,
            environ,
        )
    )
    if (
        args.elasticsearch_insecure
        and not bst_comparison.is_loopback_elasticsearch_url(elasticsearch_url)
    ):
        raise ValueError(
            "--elasticsearch-insecure is only allowed for a loopback Elasticsearch URL"
        )
    request_time = request_time or datetime.now(timezone.utc)
    inference_window = derive_inference_window(request_time, args.max_age_hours)
    output_dir = create_output_dir(args.output_dir, request_time)
    device = torch.device(args.device)

    LOGGER.info("Loading BST models")
    model_a = bst_comparison.load_model_bundle(
        "A", args.model_a_dir, args.model_a_get_data_dir, device
    )
    model_b = bst_comparison.load_model_bundle(
        "B", args.model_b_dir, args.model_b_get_data_dir, device
    )
    embedding_dim = bst_comparison.validate_model_compatibility(model_a, model_b)

    pool_specs = (
        ("small", args.small_num_candidates),
        ("large", args.large_num_candidates),
    )
    candidate_frames: list[pl.DataFrame] = []
    diagnostics_by_pool: dict[str, CandidateDiagnostics] = {}
    with requests.Session() as session:
        for pool_name, requested_candidates in pool_specs:
            LOGGER.info(
                "Requesting %d popularity candidates for the %s pool from %s",
                requested_candidates,
                pool_name,
                sanitize_api_url(api_url),
            )
            raw_candidates = fetch_popularity_candidates(
                session,
                api_url,
                api_key,
                requested_candidates,
                args.max_age_hours,
                args.api_timeout_seconds,
            )
            candidate_frame, diagnostics = build_candidate_frame(
                raw_candidates,
                requested_candidates,
                pool_name,
            )
            candidate_frames.append(candidate_frame)
            diagnostics_by_pool[pool_name] = diagnostics
            LOGGER.info(
                "%s pool returned %d candidates containing %d unique valid URIs",
                pool_name.capitalize(),
                diagnostics.api_candidates_returned,
                diagnostics.unique_candidate_uris,
            )
    candidates_df = pl.concat(candidate_frames, how="vertical")
    candidate_uris = candidates_df["at_uri"].unique(maintain_order=True).to_list()

    inference_paths: list[str] = []
    inference_scores_df = pl.DataFrame(schema=INFERENCE_SCORE_SCHEMA)
    if candidate_uris:
        LOGGER.info(
            "Listing inference exports for [%s, %s)",
            inference_window.start.isoformat(),
            inference_window.end.isoformat(),
        )
        inference_paths = list_inference_parquet_paths(
            storage.Client(),
            args.gcs_bucket,
            args.inference_prefix,
            inference_window,
        )
        if inference_paths:
            LOGGER.info(
                "Matching %d candidate URIs against %d inference files",
                len(candidate_uris),
                len(inference_paths),
            )
            inference_scores_df = build_candidate_inference_scores(
                inference_paths,
                candidate_uris,
                args.political_threshold,
            )
            LOGGER.info(
                "Matched inference rows for %d of %d candidate URIs",
                inference_scores_df.height,
                len(candidate_uris),
            )
        else:
            LOGGER.warning("No inference parquet files found in the buffered window")

    retrieval_diagnostics = bst_comparison.RetrievalDiagnostics()
    model_score_rows: list[dict[str, Any]] = []
    embedding_field = f"embeddings.{args.embedding_model}"
    timeout = httpx.Timeout(args.elasticsearch_timeout_seconds)
    LOGGER.info(
        "Hydrating %d popularity candidates from Elasticsearch %s index=%s",
        len(candidate_uris),
        bst_comparison.sanitize_elasticsearch_url(elasticsearch_url),
        args.elasticsearch_index,
    )
    async with httpx.AsyncClient(
        timeout=timeout,
        verify=not args.elasticsearch_insecure,
    ) as client:
        batch_starts = range(0, len(candidate_uris), args.elasticsearch_batch_size)
        for start in tqdm(batch_starts, desc="Hydrating and scoring", unit="batch"):
            batch_uris = candidate_uris[start : start + args.elasticsearch_batch_size]
            candidates = await bst_comparison.fetch_elasticsearch_batch(
                client,
                elasticsearch_url,
                args.elasticsearch_index,
                elasticsearch_api_key,
                batch_uris,
                embedding_field,
                embedding_dim,
                retrieval_diagnostics,
            )
            if not candidates:
                continue
            candidate_embeddings = torch.tensor(
                [post.embedding for post in candidates],
                device=device,
                dtype=torch.float32,
            )
            model_a_scores = bst_comparison.score_model_batch(
                model_a, candidate_embeddings, candidates, device
            )
            model_b_scores = bst_comparison.score_model_batch(
                model_b, candidate_embeddings, candidates, device
            )
            for index, post in enumerate(candidates):
                model_score_rows.append({
                    "at_uri": post.at_uri,
                    "scoring_author_did": post.author_did,
                    "created_at": post.created_at,
                    "elasticsearch_indexed_at": post.indexed_at,
                    "scoring_content": post.content,
                    "current_like_count": post.like_count,
                    "model_a_logit": float(model_a_scores[index]),
                    "model_b_logit": float(model_b_scores[index]),
                })
            del candidate_embeddings, candidates, model_a_scores, model_b_scores

    retrieval_diagnostics.scored_posts = len(model_score_rows)
    model_scores_df = build_model_scores_frame(model_score_rows)
    output_frame = attach_results(
        candidates_df,
        inference_scores_df,
        model_scores_df,
    )
    split_metrics = build_split_metrics(
        output_frame,
        args.top_k,
        {
            "small": args.small_num_candidates,
            "large": args.large_num_candidates,
        },
    )
    LOGGER.info("Writing candidate model scores and politicalness summary")
    output_frame.write_parquet(
        output_dir / "popularity_candidates_with_inference_scores.parquet",
        compression="zstd",
    )
    summary = build_summary(
        args,
        request_time,
        inference_window,
        inference_paths,
        diagnostics_by_pool,
        retrieval_diagnostics,
        output_frame,
        split_metrics,
        model_a,
        model_b,
        api_url,
        elasticsearch_url,
        output_dir,
    )
    _write_json(output_dir / "summary.json", summary)
    write_split_comparison_csv(split_metrics, output_dir)
    write_split_comparison_plot(split_metrics, output_dir)
    print_console_summary(summary, output_dir)
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two heavy rankers and politicalness on popularity candidates"
        )
    )
    parser.add_argument("model_a_dir", type=Path)
    parser.add_argument("model_b_dir", type=Path)
    parser.add_argument("--model-a-get-data-dir", type=Path)
    parser.add_argument("--model-b-get-data-dir", type=Path)
    parser.add_argument("--small-num-candidates", type=int, default=100)
    parser.add_argument("--large-num-candidates", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--max-age-hours",
        type=int,
        choices=SUPPORTED_MAX_AGE_HOURS,
        default=168,
    )
    parser.add_argument("--api-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--gcs-bucket", default=DEFAULT_GCS_BUCKET)
    parser.add_argument("--inference-prefix", default=DEFAULT_INFERENCE_PREFIX)
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
        "--embedding-model",
        default=bst_comparison.DEFAULT_EMBEDDING_MODEL,
    )
    parser.add_argument("--elasticsearch-url")
    parser.add_argument("--elasticsearch-api-key")
    parser.add_argument(
        "--elasticsearch-index",
        default=bst_comparison.DEFAULT_ELASTICSEARCH_INDEX,
    )
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
