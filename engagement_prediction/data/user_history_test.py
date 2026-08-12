from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from engagement_prediction.data import user_history


UTC = timezone.utc


def _queries(hours, *, did="u1", split="train"):
    return pl.DataFrame({
        "did": [did] * len(hours),
        "query_hour": hours,
        "user_cohort": ["trainval"] * len(hours),
        "split": [split] * len(hours),
        "positive_count": pl.Series([1] * len(hours), dtype=pl.UInt32),
    })


def _likes(rows):
    if not rows:
        return user_history.empty_likes()
    return pl.DataFrame({
        "did": [row[0] for row in rows],
        "subject_uri": [row[1] for row in rows],
        "like_created_at": [row[2] for row in rows],
    })


def test_histories_are_query_conditioned_and_preserve_empty_queries():
    ten = datetime(2026, 1, 1, 10, tzinfo=UTC)
    twelve = datetime(2026, 1, 1, 12, tzinfo=UTC)
    histories, stats = user_history.build_query_histories_for_partition(
        _queries([ten, twelve]),
        _likes([
            ("u1", "p1", ten + timedelta(minutes=5)),
            ("u1", "p2", ten + timedelta(hours=1, minutes=5)),
        ]),
        max_history_posts_per_query=64,
    )

    assert histories["history_subject_uris"].to_list() == [[], ["p2", "p1"]]
    assert histories["history_like_created_ats"].to_list() == [
        [],
        [ten + timedelta(hours=1, minutes=5), ten + timedelta(minutes=5)],
    ]
    assert stats["train"]["empty_history_count"] == 1
    assert stats["train"]["retained_history_item_count"] == 2


def test_history_cap_keeps_64_most_recent_and_reports_truncation():
    query_hour = datetime(2026, 1, 4, tzinfo=UTC)
    rows = [
        ("u1", f"p{offset:02d}", query_hour - timedelta(hours=65 - offset))
        for offset in range(65)
    ]
    histories, stats = user_history.build_query_histories_for_partition(
        _queries([query_hour]),
        _likes(rows),
        max_history_posts_per_query=64,
    )

    assert len(histories["history_subject_uris"][0]) == 64
    assert histories["history_subject_uris"][0][0] == "p64"
    assert "p00" not in histories["history_subject_uris"][0]
    assert stats["train"]["truncated_history_count"] == 1


def test_history_keeps_duplicates_and_uses_subject_uri_tie_breaker():
    query_hour = datetime(2026, 1, 2, tzinfo=UTC)
    tied = query_hour - timedelta(hours=1)
    histories, _ = user_history.build_query_histories_for_partition(
        _queries([query_hour]),
        _likes([
            ("u1", "p-b", tied),
            ("u1", "p-a", tied),
            ("u1", "p-a", tied),
        ]),
        max_history_posts_per_query=64,
    )

    assert histories["history_subject_uris"][0].to_list() == ["p-a", "p-a", "p-b"]


def test_partition_expression_is_stable_and_logical_output_is_partition_independent():
    dids = [f"u{idx}" for idx in range(20)]
    frame = pl.DataFrame({"did": dids})
    first = frame.with_columns(user_history.user_partition_expr(4))
    second = frame.with_columns(user_history.user_partition_expr(4))
    assert first.equals(second)

    query_hour = datetime(2026, 1, 2, tzinfo=UTC)
    queries = pl.concat([_queries([query_hour], did=did) for did in dids])
    likes = _likes([
        (did, f"{did}-post", query_hour - timedelta(hours=1))
        for did in dids
    ])
    expected, _ = user_history.build_query_histories_for_partition(
        queries,
        likes,
        max_history_posts_per_query=64,
    )
    parts = []
    for partition_id in range(4):
        partition_dids = set(
            first.filter(pl.col("_user_partition") == partition_id)["did"].to_list()
        )
        if not partition_dids:
            continue
        part, _ = user_history.build_query_histories_for_partition(
            queries.filter(pl.col("did").is_in(partition_dids)),
            likes.filter(pl.col("did").is_in(partition_dids)),
            max_history_posts_per_query=64,
        )
        parts.append(part)
    actual = pl.concat(parts).sort(["query_hour", "did"])
    assert actual.equals(expected.sort(["query_hour", "did"]))


@pytest.mark.parametrize("field", ["max_history_posts_per_query", "user_history_partition_count"])
def test_positive_limits_are_required(field):
    if field == "max_history_posts_per_query":
        with pytest.raises(ValueError, match=field):
            user_history.build_query_histories_for_partition(
                _queries([datetime(2026, 1, 2, tzinfo=UTC)]),
                user_history.empty_likes(),
                max_history_posts_per_query=0,
            )
    else:
        with pytest.raises(ValueError, match=field):
            user_history.user_partition_expr(0)
