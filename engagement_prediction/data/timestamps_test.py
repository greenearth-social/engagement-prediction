from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from engagement_prediction.data import timestamps


def test_parse_utc_datetime_normalizes_naive_and_offset_values():
    assert timestamps.parse_utc_datetime(
        "2026-01-01T12:00:00",
        field_name="timestamp",
    ) == datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    assert timestamps.parse_utc_datetime(
        "2026-01-01T12:00:00-05:00",
        field_name="timestamp",
    ) == datetime(2026, 1, 1, 17, tzinfo=timezone.utc)
    assert timestamps.parse_utc_datetime(None, field_name="timestamp") is None


@pytest.mark.parametrize("value", ["", "not-a-timestamp"])
def test_parse_utc_datetime_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="timestamp"):
        timestamps.parse_utc_datetime(value, field_name="timestamp")


def test_utc_timestamp_expr_normalizes_strings_and_datetime_units():
    strings_lf = pl.DataFrame({
        "created_at": ["2026-01-01T12:00:00", "2026-01-01T12:00:00-05:00"],
    }).lazy()
    strings = strings_lf.select(
        timestamps.utc_timestamp_expr(strings_lf, "created_at").alias("created_at")
    ).collect()
    assert strings.schema["created_at"] == timestamps.UTC_DATETIME
    assert strings["created_at"].to_list() == [
        datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 17, tzinfo=timezone.utc),
    ]

    datetimes_lf = pl.DataFrame({
        "created_at": pl.Series(
            [datetime(2026, 1, 1, 12)],
            dtype=pl.Datetime("ns"),
        ),
    }).lazy()
    datetimes = datetimes_lf.select(
        timestamps.utc_timestamp_expr(datetimes_lf, "created_at").alias("created_at")
    ).collect()
    assert datetimes.schema["created_at"] == timestamps.UTC_DATETIME
    assert datetimes.item() == datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def test_half_open_window_expr_includes_start_and_excludes_end():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
    result = pl.DataFrame({
        "created_at": [start, start + timedelta(hours=1), end],
    }).filter(
        timestamps.half_open_window_expr(
            "created_at",
            start=start,
            end=end,
        )
    )
    assert result["created_at"].to_list() == [start, start + timedelta(hours=1)]


def test_validate_half_open_utc_window_rejects_invalid_boundaries():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, tzinfo=timezone.utc)
    timestamps.validate_half_open_utc_window(
        start=start,
        end=end,
        start_field_name="start",
        end_field_name="end",
    )

    with pytest.raises(ValueError, match="timezone-aware UTC"):
        timestamps.validate_half_open_utc_window(
            start=start.replace(tzinfo=None),
            end=end,
            start_field_name="start",
            end_field_name="end",
        )
    with pytest.raises(ValueError, match="must use UTC"):
        timestamps.validate_half_open_utc_window(
            start=start.astimezone(timezone(timedelta(hours=1))),
            end=end,
            start_field_name="start",
            end_field_name="end",
        )
    with pytest.raises(ValueError, match="aligned to the start of an hour"):
        timestamps.validate_half_open_utc_window(
            start=start.replace(minute=1),
            end=end,
            start_field_name="start",
            end_field_name="end",
        )
    with pytest.raises(ValueError, match="end must be after start"):
        timestamps.validate_half_open_utc_window(
            start=end,
            end=start,
            start_field_name="start",
            end_field_name="end",
        )
