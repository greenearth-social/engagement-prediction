#!/usr/bin/env python3
"""Measure political-inference score distributions in popularity candidates."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import SplitResult, urlsplit, urlunsplit

import polars as pl
import requests
from google.cloud import storage


LOGGER = logging.getLogger("evaluate_popularity_political_share")

DEFAULT_API_URL = "https://api.greenearth.social/"
DEFAULT_GCS_BUCKET = "greenearth-471522-ingex-extract-prod"
DEFAULT_INFERENCE_PREFIX = "bsky_inferences"
DEFAULT_OUTPUT_ROOT = Path("/mnt/data/dave/outputs/compare")
DEFAULT_USER_DID = "did:plc:aaaaaaaaaaaaaaaaaaaaaaaa"
INFERENCE_BUFFER_DAYS = 6
SUPPORTED_MAX_AGE_HOURS = (6, 12, 24, 48, 72, 168)
TOP_CUTOFFS = (10, 25, 50, 100)
SCORE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
POLITICAL_SCORE_FIELD = "topic.News & Social Concern"

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
    "candidate_rank": pl.Int64,
    "at_uri": pl.String,
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
}

OUTPUT_SCHEMA = {
    **CANDIDATE_SCHEMA,
    **{key: value for key, value in INFERENCE_SCORE_SCHEMA.items() if key != "at_uri"},
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
    num_candidates: int,
    max_age_hours: int,
    api_timeout_seconds: float,
) -> None:
    if not 1 <= num_candidates <= 1000:
        raise ValueError("--num-candidates must be between 1 and 1000")
    if max_age_hours not in SUPPORTED_MAX_AGE_HOURS:
        supported = ", ".join(str(value) for value in SUPPORTED_MAX_AGE_HOURS)
        raise ValueError(f"--max-age-hours must be one of: {supported}")
    if not math.isfinite(api_timeout_seconds) or api_timeout_seconds <= 0.0:
        raise ValueError("--api-timeout-seconds must be positive")


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
            "candidate_rank": candidate_rank,
            "at_uri": at_uri,
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
        .select(*INFERENCE_SCORE_SCHEMA)
    )
    return latest


def attach_inference_scores(
    candidates_df: pl.DataFrame,
    scores_df: pl.DataFrame,
) -> pl.DataFrame:
    if candidates_df.is_empty():
        return pl.DataFrame(schema=OUTPUT_SCHEMA)
    return (
        candidates_df
        .join(scores_df, on="at_uri", how="left")
        .select(*OUTPUT_SCHEMA)
        .sort("candidate_rank")
    )


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
    return {
        "total_candidates": total,
        "matched_inference_candidates": matched,
        "candidates_without_inference": total - matched,
        "inference_coverage": matched / total if total else None,
        "news_social_concern_score": score_distribution(news_values),
    }


def top_cutoff_summaries(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for requested_cutoff in TOP_CUTOFFS:
        actual_cutoff = min(requested_cutoff, frame.height)
        summary = inference_score_summary(frame.head(actual_cutoff))
        summaries[str(requested_cutoff)] = {
            "requested_cutoff": requested_cutoff,
            "actual_cutoff": actual_cutoff,
            **summary,
        }
    return summaries


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
    diagnostics: CandidateDiagnostics,
    output_frame: pl.DataFrame,
    api_url: str,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "evaluated_at": request_time.astimezone(timezone.utc).isoformat(),
        "run_config": {
            "api_url": sanitize_api_url(api_url),
            "num_candidates": args.num_candidates,
            "max_age_hours": args.max_age_hours,
            "api_timeout_seconds": args.api_timeout_seconds,
            "gcs_bucket": args.gcs_bucket,
            "inference_prefix": args.inference_prefix,
            "inference_buffer_days": INFERENCE_BUFFER_DAYS,
            "political_score_field": POLITICAL_SCORE_FIELD,
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
        "api_candidates": asdict(diagnostics),
        "overall": inference_score_summary(output_frame),
        "top_cutoffs": top_cutoff_summaries(output_frame),
        "score_semantics": (
            "The topic.News & Social Concern score is retained from the latest "
            "inference row per candidate URI. Missing inference rows and missing "
            "score fields remain null."
        ),
        "distribution_methodology": {
            "quantiles": "linear interpolation between ordered finite scores",
            "dispersion": "population standard deviation",
        },
    }


def _format_percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _format_stat(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def print_console_summary(summary: Mapping[str, Any], output_dir: Path) -> None:
    api = summary["api_candidates"]
    overall = summary["overall"]
    inference_window = summary["inference_export_window"]
    print("\nPopularity candidate inference-score evaluation")
    print(
        f"  API candidates: requested={api['requested_candidates']:,}, "
        f"returned={api['api_candidates_returned']:,}, "
        f"unique URIs={api['unique_candidate_uris']:,}"
    )
    print(
        f"  Candidate diagnostics: missing URI={api['candidates_missing_at_uri']:,}, "
        f"duplicate URI={api['duplicate_candidate_uris']:,}"
    )
    print(
        f"  Inferences: files={inference_window['files_scanned']:,}, "
        f"matched={overall['matched_inference_candidates']:,}/"
        f"{overall['total_candidates']:,}, "
        f"coverage={_format_percent(overall['inference_coverage'])}"
    )
    distribution = overall["news_social_concern_score"]
    print(
        f"  News & Social Concern: n={distribution['finite_score_count']:,}, "
        f"missing={distribution['missing_score_count']:,}, "
        f"mean={_format_stat(distribution['mean'])}, "
        f"median={_format_stat(distribution['median'])}, "
        f"p90={_format_stat(distribution['p90'])}, "
        f"p95={_format_stat(distribution['p95'])}, "
        f"p99={_format_stat(distribution['p99'])}, "
        f"max={_format_stat(distribution['max'])}"
    )
    for cutoff, cutoff_summary in summary["top_cutoffs"].items():
        news = cutoff_summary["news_social_concern_score"]
        print(
            f"  Top {cutoff}: matched={cutoff_summary['matched_inference_candidates']:,}/"
            f"{cutoff_summary['actual_cutoff']:,}, "
            f"News & Social Concern mean/median={_format_stat(news['mean'])}/"
            f"{_format_stat(news['median'])}"
        )
    print(f"  Output: {output_dir}")


def run(
    args: argparse.Namespace,
    environ: Mapping[str, str],
    request_time: Optional[datetime] = None,
) -> Path:
    validate_parameters(
        args.num_candidates,
        args.max_age_hours,
        args.api_timeout_seconds,
    )
    api_url, api_key = resolve_api_settings(environ)
    request_time = request_time or datetime.now(timezone.utc)
    inference_window = derive_inference_window(request_time, args.max_age_hours)
    output_dir = create_output_dir(args.output_dir, request_time)

    LOGGER.info(
        "Requesting %d popularity candidates from %s",
        args.num_candidates,
        sanitize_api_url(api_url),
    )
    with requests.Session() as session:
        raw_candidates = fetch_popularity_candidates(
            session,
            api_url,
            api_key,
            args.num_candidates,
            args.max_age_hours,
            args.api_timeout_seconds,
        )
    candidates_df, diagnostics = build_candidate_frame(
        raw_candidates,
        args.num_candidates,
    )
    LOGGER.info(
        "API returned %d candidates containing %d unique valid URIs",
        diagnostics.api_candidates_returned,
        diagnostics.unique_candidate_uris,
    )

    inference_paths: list[str] = []
    scores_df = pl.DataFrame(schema=INFERENCE_SCORE_SCHEMA)
    if not candidates_df.is_empty():
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
                candidates_df.height,
                len(inference_paths),
            )
            scores_df = build_candidate_inference_scores(
                inference_paths,
                candidates_df["at_uri"].to_list(),
            )
            LOGGER.info(
                "Matched inference rows for %d of %d candidate URIs",
                scores_df.height,
                candidates_df.height,
            )
        else:
            LOGGER.warning("No inference parquet files found in the buffered window")

    output_frame = attach_inference_scores(candidates_df, scores_df)
    LOGGER.info("Writing candidate inference scores and distribution summary")
    output_frame.write_parquet(
        output_dir / "popularity_candidates_with_inference_scores.parquet",
        compression="zstd",
    )
    summary = build_summary(
        args,
        request_time,
        inference_window,
        inference_paths,
        diagnostics,
        output_frame,
        api_url,
        output_dir,
    )
    _write_json(output_dir / "summary.json", summary)
    print_console_summary(summary, output_dir)
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure political-inference score distributions in popularity candidates"
    )
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument(
        "--max-age-hours",
        type=int,
        choices=SUPPORTED_MAX_AGE_HOURS,
        default=168,
    )
    parser.add_argument("--api-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--gcs-bucket", default=DEFAULT_GCS_BUCKET)
    parser.add_argument("--inference-prefix", default=DEFAULT_INFERENCE_PREFIX)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        run(args, os.environ)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
