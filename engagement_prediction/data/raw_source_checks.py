"""Evaluate existing raw-source counters without reading the source data again."""

from __future__ import annotations

from statistics import median
from typing import Any


def evaluate_raw_sources(
    diagnostics_by_source: dict[str, dict[str, Any]],
    *,
    min_complete_days: int,
    fail_min_ratio: float,
    warn_min_ratio: float,
    warn_max_ratio: float,
    warn_invalid_timestamp_fraction: float,
) -> dict[str, Any]:
    """Check each source independently using its complete UTC creation days.

    The CLI validates the thresholds. Relative checks use the current window's
    median, so they detect individual bad days, not a uniformly low window.
    """

    checks = []
    source_summaries = {}
    for source_name, diagnostics in diagnostics_by_source.items():
        partial_dates = set(diagnostics["partial_dates"])
        complete_days = [
            (date, count)
            for date, count in zip(diagnostics["dates"], diagnostics["counts"])
            if date not in partial_dates
        ]
        median_count = median([count for _, count in complete_days]) if complete_days else None
        relative_checks_applied = (
            len(complete_days) >= min_complete_days
            and median_count is not None
            and median_count > 0
        )
        source_summaries[source_name] = {
            "complete_day_count": len(complete_days),
            "median_daily_count": median_count,
            "relative_checks_applied": relative_checks_applied,
        }

        in_window_count = diagnostics["in_window_row_count"]
        checks.append({
            "check": "nonempty_source",
            "source": source_name,
            "status": "pass" if in_window_count > 0 else "fail",
            "message": f"{source_name}: {in_window_count:,} raw rows in the source window",
            "in_window_row_count": in_window_count,
        })

        for date, count in complete_days:
            ratio = count / median_count if relative_checks_applied else None
            status = "pass"
            if count == 0 or (ratio is not None and ratio < fail_min_ratio):
                status = "fail"
            elif ratio is not None and (ratio < warn_min_ratio or ratio > warn_max_ratio):
                status = "warn"
            message = f"{source_name} on {date}: {count:,} raw rows"
            if ratio is not None:
                message += f" ({ratio:.1%} of the window median {median_count:,.0f})"
            checks.append({
                "check": "daily_volume",
                "source": source_name,
                "date": date,
                "status": status,
                "message": message,
                "row_count": count,
                "median_ratio": ratio,
            })

        # Export timestamps and record creation timestamps differ legitimately;
        # before/after-window counters remain informational in the raw report.
        total_count = diagnostics["total_row_count"]
        invalid_count = diagnostics["invalid_timestamp_count"]
        invalid_fraction = invalid_count / total_count if total_count else 0.0
        checks.append({
            "check": "invalid_timestamps",
            "source": source_name,
            "status": "warn" if invalid_fraction > warn_invalid_timestamp_fraction else "pass",
            "message": (
                f"{source_name}: {invalid_count:,}/{total_count:,} raw rows "
                f"({invalid_fraction:.2%}) have invalid timestamps"
            ),
            "invalid_timestamp_fraction": invalid_fraction,
        })

    statuses = {check["status"] for check in checks}
    return {
        "status": "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass",
        "checks": checks,
        "sources": source_summaries,
        "raw_source_diagnostics": diagnostics_by_source,
        "thresholds": {
            "min_complete_days": min_complete_days,
            "fail_min_ratio": fail_min_ratio,
            "warn_min_ratio": warn_min_ratio,
            "warn_max_ratio": warn_max_ratio,
            "warn_invalid_timestamp_fraction": warn_invalid_timestamp_fraction,
        },
    }
