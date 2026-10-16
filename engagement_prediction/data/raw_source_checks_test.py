from datetime import datetime, timedelta, timezone

import pytest

from cli import DEFAULTS
from engagement_prediction.data.raw_source_checks import evaluate_raw_sources


def _diagnostics(counts, *, partial_dates=(), invalid_count=0, outside_count=0):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return {
        "posts_start": start.isoformat(),
        "posts_end": (start + timedelta(days=len(counts))).isoformat(),
        "dates": [(start + timedelta(days=index)).date().isoformat() for index in range(len(counts))],
        "counts": counts,
        "partial_dates": list(partial_dates),
        "total_row_count": sum(counts) + invalid_count + outside_count,
        "in_window_row_count": sum(counts),
        "invalid_timestamp_count": invalid_count,
        "before_window_row_count": outside_count,
        "at_or_after_end_row_count": 0,
    }


def _evaluate(sources, **overrides):
    thresholds = {
        name: DEFAULTS[f"raw_data_{name}"]
        for name in (
            "min_complete_days", "fail_min_ratio", "warn_min_ratio",
            "warn_max_ratio", "warn_invalid_timestamp_fraction",
        )
    }
    return evaluate_raw_sources(sources, **(thresholds | overrides))


def _daily_checks(report):
    return [check for check in report["checks"] if check["check"] == "daily_volume"]


@pytest.mark.parametrize(
    ("count", "expected_status"),
    [(0, "fail"), (24, "fail"), (25, "warn"), (49, "warn"), (50, "pass"),
     (300, "pass"), (301, "warn")],
)
def test_daily_volume_thresholds_and_exact_boundaries(count, expected_status):
    report = _evaluate({"likes": _diagnostics([100] * 6 + [count])})

    assert report["status"] == expected_status
    assert _daily_checks(report)[-1]["status"] == expected_status
    assert _daily_checks(report)[-1]["median_ratio"] == count / 100
    assert report["sources"]["likes"]["median_daily_count"] == 100


def test_each_source_uses_its_own_median_and_failures_take_precedence():
    report = _evaluate({
        "likes": _diagnostics([10_000] * 6 + [4000]),
        "root_posts": _diagnostics([1000] * 7),
        "replies": _diagnostics([100] * 6 + [1]),
    })

    assert report["status"] == "fail"
    assert report["sources"]["likes"]["median_daily_count"] == 10_000
    assert report["sources"]["root_posts"]["median_daily_count"] == 1000
    assert report["sources"]["replies"]["median_daily_count"] == 100
    assert all(check["status"] == "pass" for check in _daily_checks(report) if check["source"] == "root_posts")
    assert {check["source"] for check in report["checks"] if check["status"] == "warn"} == {"likes"}
    assert {check["source"] for check in report["checks"] if check["status"] == "fail"} == {"replies"}


@pytest.mark.parametrize("counts", [[], [0], [0] * 7])
def test_empty_source_fails_without_dividing_by_zero(counts):
    report = _evaluate({"likes": _diagnostics(counts)})

    assert report["status"] == "fail"
    assert report["checks"][0]["check"] == "nonempty_source"
    assert report["checks"][0]["status"] == "fail"
    assert not report["sources"]["likes"]["relative_checks_applied"]


def test_partial_dates_are_excluded_from_checks_and_baseline():
    report = _evaluate({
        "likes": _diagnostics([0] + [100] * 7 + [10_000], partial_dates=("2026-01-01", "2026-01-09")),
    })

    assert report["status"] == "pass"
    assert report["sources"]["likes"] == {
        "complete_day_count": 7,
        "median_daily_count": 100,
        "relative_checks_applied": True,
    }
    assert len(_daily_checks(report)) == 7


def test_short_windows_skip_relative_checks_but_still_require_nonzero_days():
    sources = {"likes": _diagnostics([100, 1])}
    report = _evaluate(sources)
    assert report["status"] == "pass"
    assert not report["sources"]["likes"]["relative_checks_applied"]
    assert all(check["median_ratio"] is None for check in _daily_checks(report))

    sources["likes"] = _diagnostics([100, 0])
    assert _evaluate(sources)["status"] == "fail"


def test_min_complete_days_override_changes_baseline_eligibility():
    report = _evaluate({"likes": _diagnostics([100, 100, 1])}, min_complete_days=3)

    assert report["status"] == "fail"
    assert report["sources"]["likes"]["relative_checks_applied"]


@pytest.mark.parametrize(("invalid_count", "expected_status"), [(1, "pass"), (2, "warn")])
def test_invalid_timestamps_warn_only_above_fraction_threshold(invalid_count, expected_status):
    report = _evaluate({"likes": _diagnostics([99], invalid_count=invalid_count)})

    assert report["status"] == expected_status
    check = next(check for check in report["checks"] if check["check"] == "invalid_timestamps")
    assert check["status"] == expected_status
    assert check["invalid_timestamp_fraction"] == invalid_count / (99 + invalid_count)


def test_outside_window_rows_are_informational_and_raw_counters_are_preserved():
    sources = {"likes": _diagnostics([100] * 7, outside_count=10_000)}
    report = _evaluate(sources)

    assert report["status"] == "pass"
    assert report["raw_source_diagnostics"] == sources
    assert report["thresholds"]["fail_min_ratio"] == DEFAULTS["raw_data_fail_min_ratio"]
