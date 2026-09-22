from datetime import datetime, timedelta, timezone
import json
import logging

import polars as pl
import pytest

from engagement_prediction.data import raw_source_diagnostics as diagnostics


UTC = timezone.utc


def test_counts_raw_rows_across_files_and_reconciles_exclusions(tmp_path):
    paths = [tmp_path / "one.parquet", tmp_path / "two.parquet"]
    pl.DataFrame({
        "record_created_at": [
            "2026-01-02T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-01T23:00:00-02:00",
            "2026-01-04T12:00:00",
            None,
        ],
        "at_uri": ["duplicate", "duplicate", None, "", "ignored"],
        "payload": ["unused"] * 5,
    }).write_parquet(paths[0])
    pl.DataFrame({
        "record_created_at": [
            "2025-12-31T23:59:59Z",
            "2026-01-06T00:00:00Z",
            "2099-01-01T00:00:00Z",
            "bad timestamp",
        ],
        "at_uri": ["before", "end", "after", "invalid"],
        "payload": ["unused"] * 4,
    }).write_parquet(paths[1])

    result = diagnostics.count_daily_rows(
        pl.scan_parquet(paths),
        timestamp_column="record_created_at",
        posts_start=datetime(2026, 1, 1, tzinfo=UTC),
        posts_end=datetime(2026, 1, 6, tzinfo=UTC),
    )

    assert result == {
        "posts_start": "2026-01-01T00:00:00+00:00",
        "posts_end": "2026-01-06T00:00:00+00:00",
        "dates": [f"2026-01-{day:02d}" for day in range(1, 6)],
        "counts": [0, 3, 0, 1, 0],
        "partial_dates": [],
        "total_row_count": 9,
        "in_window_row_count": 4,
        "invalid_timestamp_count": 2,
        "before_window_row_count": 1,
        "at_or_after_end_row_count": 2,
    }
    assert json.loads(json.dumps(result)) == result


def test_exact_boundaries_and_partial_days():
    start = datetime(2026, 1, 1, 6, tzinfo=UTC)
    end = datetime(2026, 1, 3, 18, tzinfo=UTC)
    result = diagnostics.count_daily_rows(
        pl.DataFrame({
            "created_at": [
                start - timedelta(microseconds=1),
                start,
                datetime(2026, 1, 2, tzinfo=UTC),
                end - timedelta(microseconds=1),
                end,
            ],
        }).lazy(),
        timestamp_column="created_at",
        posts_start=start,
        posts_end=end,
    )

    assert result["counts"] == [1, 1, 1]
    assert result["before_window_row_count"] == 1
    assert result["at_or_after_end_row_count"] == 1
    assert result["partial_dates"] == ["2026-01-01", "2026-01-03"]
    assert diagnostics.daily_count_labels(result) == [
        "2026-01-01 (partial)", "2026-01-02", "2026-01-03 (partial)",
    ]


@pytest.mark.parametrize("source_timezone", [None, "America/Los_Angeles"])
def test_normalizes_typed_timestamps_before_choosing_day(source_timezone):
    values = pl.Series(
        "created_at",
        [datetime(2026, 1, 2, 1)],
        dtype=pl.Datetime("ns"),
    )
    if source_timezone is not None:
        values = values.dt.replace_time_zone("UTC").dt.convert_time_zone(source_timezone)
    result = diagnostics.count_daily_rows(
        values.to_frame().lazy(),
        timestamp_column="created_at",
        posts_start=datetime(2026, 1, 1, tzinfo=UTC),
        posts_end=datetime(2026, 1, 3, tzinfo=UTC),
    )
    assert result["dates"] == ["2026-01-01", "2026-01-02"]
    assert result["counts"] == [0, 1]
    assert result["partial_dates"] == []


@pytest.mark.parametrize("values", [[], [None, "invalid"]])
def test_empty_or_invalid_sources_keep_every_calendar_day(values):
    result = diagnostics.count_daily_rows(
        pl.DataFrame({"created_at": pl.Series(values, dtype=pl.String)}).lazy(),
        timestamp_column="created_at",
        posts_start=datetime(2026, 1, 1, tzinfo=UTC),
        posts_end=datetime(2026, 1, 4, tzinfo=UTC),
    )
    assert result["dates"] == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert result["counts"] == [0, 0, 0]
    assert result["total_row_count"] == len(values)
    assert result["invalid_timestamp_count"] == len(values)
    assert result["in_window_row_count"] == 0


def test_scan_staged_timestamps_includes_null_partitions_and_empty_routes(tmp_path):
    routes = tmp_path / "routes"
    routes.mkdir()
    start = datetime(2026, 1, 1, 6, tzinfo=UTC)
    end = datetime(2026, 1, 1, 18, tzinfo=UTC)
    empty = diagnostics.count_daily_rows(
        diagnostics.scan_staged_timestamps(routes, timestamp_column="post_created_at"),
        timestamp_column="post_created_at",
        posts_start=start,
        posts_end=end,
    )
    assert empty["counts"] == [0]
    assert diagnostics.daily_count_labels(empty) == ["2026-01-01 (partial)"]
    for partition, uri in (("0", "valid"), ("__HIVE_DEFAULT_PARTITION__", None)):
        directory = routes / f"_post_partition={partition}"
        directory.mkdir()
        pl.DataFrame({
            "post_created_at": [start],
            "subject_uri": pl.Series([uri], dtype=pl.String),
            "_post_row_valid": [uri is not None],
        }).write_parquet(directory / "part.parquet")

    scan = diagnostics.scan_staged_timestamps(routes, timestamp_column="post_created_at")
    assert scan.collect_schema().names() == ["post_created_at"]
    populated = diagnostics.count_daily_rows(
        scan, timestamp_column="post_created_at", posts_start=start, posts_end=end,
    )
    assert populated["counts"] == [2]
    assert populated["total_row_count"] == 2


def test_missing_staged_directory_is_not_reported_as_zero(tmp_path):
    with pytest.raises(FileNotFoundError, match="Staged source directory"):
        diagnostics.scan_staged_timestamps(
            tmp_path / "missing", timestamp_column="post_created_at",
        )


def test_logs_all_exclusion_counters(caplog):
    result = diagnostics.count_daily_rows(
        pl.DataFrame({
            "created_at": [
                "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z",
                "2026-01-03T00:00:00Z", None,
            ],
        }).lazy(),
        timestamp_column="created_at",
        posts_start=datetime(2026, 1, 2, tzinfo=UTC),
        posts_end=datetime(2026, 1, 3, tzinfo=UTC),
    )
    with caplog.at_level(logging.INFO):
        diagnostics.log_daily_counts(
            logging.getLogger("coverage_test"), source_name="likes", diagnostics=result,
        )
    for expected in (
        "Raw likes daily coverage", "total_rows=4", "in_window_rows=1",
        "invalid_timestamp_rows=1", "before_window_rows=1", "at_or_after_end_rows=1",
    ):
        assert expected in caplog.text
