"""Training-only normalization for BST as-of popularity features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import polars as pl


_MIN_STANDARD_DEVIATION = 1.0e-6


@dataclass(frozen=True)
class PopularityNormalizationStats:
    """Population moments and observation counts used by the BST encoder."""

    enabled: bool
    log_mean: float
    log_std: float
    history_observation_count: int
    candidate_observation_count: int
    total_observation_count: int

    def to_dict(self) -> dict[str, bool | float | int]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class _Moments:
    count: int
    value_sum: float
    squared_value_sum: float


def _log1p_moments(counts_lf: pl.LazyFrame) -> _Moments:
    """Collect only scalar moments from a lazy stream of nonnegative counts."""

    logged_lf = counts_lf.select(
        pl.col("prior_like_count")
        .cast(pl.Float64)
        .clip(lower_bound=0.0)
        .log1p()
        .alias("_logged_count")
    )
    row = (
        logged_lf.select(
            pl.len().alias("count"),
            pl.col("_logged_count").sum().alias("value_sum"),
            (pl.col("_logged_count") * pl.col("_logged_count"))
            .sum()
            .alias("squared_value_sum"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    return _Moments(
        count=int(row["count"]),
        value_sum=float(row["value_sum"] or 0.0),
        squared_value_sum=float(row["squared_value_sum"] or 0.0),
    )


def _validate_non_null_counts(counts_lf: pl.LazyFrame, *, description: str) -> None:
    invalid = (
        counts_lf.filter(pl.col("prior_like_count").is_null())
        .limit(1)
        .collect(engine="streaming")
    )
    if invalid.height:
        raise ValueError(f"{description} contain a null prior_like_count")


def fit_popularity_normalization(
    *,
    queries_lf: pl.LazyFrame,
    query_positives_lf: pl.LazyFrame,
    query_histories_lf: pl.LazyFrame,
    hourly_negative_candidates_lf: pl.LazyFrame,
    enabled: bool,
) -> PopularityNormalizationStats:
    """Fit one shared log-count normalization from model-facing train values.

    History list entries are actual transformer tokens and therefore retain
    their event multiplicity. Candidate features are shared within an hour, so
    positives and negatives are collapsed to one value per post-hour before
    moments are calculated. This makes the fitted transform independent of
    user count, batch size, and a particular epoch's negative subsample.
    """

    if not enabled:
        return PopularityNormalizationStats(
            enabled=False,
            log_mean=0.0,
            log_std=1.0,
            history_observation_count=0,
            candidate_observation_count=0,
            total_observation_count=0,
        )

    train_query_keys_lf = (
        queries_lf.filter(pl.col("split") == "train")
        .select("did", "query_hour")
        .unique()
    )
    train_query_hours_lf = train_query_keys_lf.select("query_hour").unique()

    history_counts_lf = (
        query_histories_lf.join(
            train_query_keys_lf,
            on=["did", "query_hour"],
            how="semi",
        )
        .select(
            pl.col("history_prior_like_counts")
            .explode(empty_as_null=False)
            .cast(pl.UInt64)
            .alias("prior_like_count")
        )
        .filter(pl.col("prior_like_count").is_not_null())
    )

    positive_candidate_counts_lf = (
        query_positives_lf.join(
            train_query_keys_lf,
            on=["did", "query_hour"],
            how="semi",
        )
        .select(
            "query_hour",
            "subject_uri",
            pl.col("prior_like_count").cast(pl.UInt64),
        )
    )
    negative_candidate_counts_lf = (
        hourly_negative_candidates_lf.join(
            train_query_hours_lf,
            on="query_hour",
            how="semi",
        )
        .select(
            "query_hour",
            "subject_uri",
            pl.col("prior_like_count").cast(pl.UInt64),
        )
    )
    candidate_counts_lf = pl.concat(
        [positive_candidate_counts_lf, negative_candidate_counts_lf],
        how="vertical",
    )
    _validate_non_null_counts(
        candidate_counts_lf,
        description="Training candidate rows",
    )
    grouped_candidate_counts_lf = candidate_counts_lf.group_by(
        "query_hour",
        "subject_uri",
    ).agg(
        pl.col("prior_like_count").n_unique().alias("_count_variants"),
        pl.col("prior_like_count").first().alias("prior_like_count"),
    )
    conflicts = (
        grouped_candidate_counts_lf.filter(pl.col("_count_variants") > 1)
        .select("query_hour", "subject_uri")
        .limit(5)
        .collect(engine="streaming")
    )
    if conflicts.height:
        examples = ", ".join(
            f"({row['query_hour']}, {row['subject_uri']})"
            for row in conflicts.iter_rows(named=True)
        )
        raise ValueError(
            "Training candidates contain conflicting prior_like_count values "
            f"for the same post-hour: {examples}"
        )
    unique_candidate_counts_lf = grouped_candidate_counts_lf.select(
        "prior_like_count"
    )

    history_moments = _log1p_moments(history_counts_lf)
    candidate_moments = _log1p_moments(unique_candidate_counts_lf)
    total_count = history_moments.count + candidate_moments.count
    if total_count == 0:
        log_mean = 0.0
        log_std = 1.0
    else:
        value_sum = history_moments.value_sum + candidate_moments.value_sum
        squared_value_sum = (
            history_moments.squared_value_sum
            + candidate_moments.squared_value_sum
        )
        log_mean = value_sum / total_count
        variance = max(squared_value_sum / total_count - log_mean * log_mean, 0.0)
        calculated_std = math.sqrt(variance)
        log_std = (
            calculated_std
            if math.isfinite(calculated_std)
            and calculated_std >= _MIN_STANDARD_DEVIATION
            else 1.0
        )

    return PopularityNormalizationStats(
        enabled=True,
        log_mean=float(log_mean),
        log_std=float(log_std),
        history_observation_count=history_moments.count,
        candidate_observation_count=candidate_moments.count,
        total_observation_count=total_count,
    )
