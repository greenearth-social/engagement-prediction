from datetime import datetime, timezone

import polars as pl

from engagement_prediction.data import author_statistics
from engagement_prediction.data import post_selection


UTC = timezone.utc


def _resolved_posts() -> pl.DataFrame:
    return pl.DataFrame({
        "subject_uri": ["p1", "p2", "r1", "p3"],
        "post_created_at": [
            datetime(2026, 1, 1, hour=1, tzinfo=UTC),
            datetime(2026, 1, 1, hour=2, tzinfo=UTC),
            datetime(2026, 1, 1, hour=3, tzinfo=UTC),
            datetime(2026, 1, 1, hour=4, tzinfo=UTC),
        ],
        "author_did": ["author-a", "author-a", "author-a", "author-b"],
        "is_reply": [False, False, True, False],
    }, schema=post_selection.POST_SCHEMA)


def test_author_aggregation_includes_zero_like_records_and_root_reply_breakdown():
    per_post = author_statistics.build_per_post_statistics(
        _resolved_posts(),
        pl.DataFrame({
            "subject_uri": ["p2", "r1", "p3"],
            "received_like_count": pl.Series([2, 4, 20], dtype=pl.UInt64),
        }),
    )
    stats = author_statistics.aggregate_author_statistics(per_post)

    author_a = stats.filter(pl.col("author_did") == "author-a").row(0, named=True)
    assert author_a["post_count"] == 3
    assert author_a["root_post_count"] == 2
    assert author_a["reply_post_count"] == 1
    assert author_a["received_like_count"] == 6
    assert author_a["root_received_like_count"] == 2
    assert author_a["reply_received_like_count"] == 4
    assert author_a["liked_post_count"] == 2
    assert author_a["mean_likes_per_post"] == 2.0
    assert author_a["median_likes_per_post"] == 2.0
    assert author_a["max_likes_per_post"] == 4

    author_b = stats.filter(pl.col("author_did") == "author-b").row(0, named=True)
    assert author_b["post_count"] == 1
    assert stats["author_did"].to_list() == [
        "author-a",
        "author-b",
    ]


def test_author_statistics_are_unfiltered_and_schema_valid():
    stats = author_statistics.aggregate_author_statistics(
        author_statistics.build_per_post_statistics(
            _resolved_posts(),
            author_statistics.empty_frame({
                "subject_uri": pl.String,
                "received_like_count": pl.UInt64,
            }),
        )
    )

    assert stats.schema == pl.Schema(author_statistics.AUTHOR_STAT_SCHEMA)
    assert stats["author_did"].to_list() == ["author-a", "author-b"]
    assert "author_idx" not in stats.columns
    assert author_statistics.validate_author_statistics_dataset(stats.lazy()) == {
        "author_count": 2,
    }


def test_empty_author_statistics_preserve_public_schema():
    empty = author_statistics.empty_frame(author_statistics.AUTHOR_STAT_SCHEMA)

    assert empty.schema == pl.Schema(author_statistics.AUTHOR_STAT_SCHEMA)
    assert empty.is_empty()
    assert author_statistics.validate_author_statistics_dataset(empty.lazy())[
        "author_count"
    ] == 0


def test_author_partition_assignment_is_stable():
    authors = pl.DataFrame({"author_did": ["a", "b", "a"]}).with_columns(
        author_statistics.author_partition_expr(7)
    )

    assert authors["_author_partition"][0] == authors["_author_partition"][2]
    assert authors["_author_partition"].min() >= 0
    assert authors["_author_partition"].max() < 7
