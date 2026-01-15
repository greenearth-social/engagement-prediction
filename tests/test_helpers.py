from datetime import datetime, timezone

import pytest

from utils.helpers import parse_one_ts


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024-02-10T13:45:00+0000", datetime(2024, 2, 10, 13, 45, tzinfo=timezone.utc)),
        ("2024-02-10T13:45:00", datetime(2024, 2, 10, 13, 45, tzinfo=timezone.utc)),
        ("2024-02-10", datetime(2024, 2, 10, 0, 0, tzinfo=timezone.utc)),
    ],
)
def test_parse_one_ts_accepts_known_formats(raw: str, expected: datetime):
    assert parse_one_ts(raw) == expected


def test_parse_one_ts_returns_none_for_missing():
    assert parse_one_ts(None) is None


def test_parse_one_ts_rejects_unrecognized_format():
    with pytest.raises(ValueError):
        parse_one_ts("10-02-2024 13:45")
