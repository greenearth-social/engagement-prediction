from datetime import datetime, timezone

import polars as pl
import pytest

from engagement_prediction.data import post_liker_history
from engagement_prediction.data import post_selection


UTC = timezone.utc


def test_build_selected_posts_combines_roles_and_excludes_unresolved_history():
    required = pl.DataFrame({
        "subject_uri": ["positive", "history", "missing", "overlap"],
        "is_positive": [True, False, False, True],
        "is_history": [False, True, True, True],
    }, schema=post_selection.REQUIRED_POST_SCHEMA)
    missing = pl.DataFrame({
        "subject_uri": ["missing"],
        "is_positive": [False],
        "is_history": [True],
    }, schema=post_selection.REQUIRED_POST_SCHEMA)
    negatives = pl.DataFrame({
        "subject_uri": ["negative", "overlap"],
    })

    selected = post_liker_history.build_selected_posts(required, missing, negatives)

    assert selected.to_dicts() == [
        {
            "subject_uri": "history",
            "is_positive": False,
            "is_history": True,
            "is_negative": False,
        },
        {
            "subject_uri": "negative",
            "is_positive": False,
            "is_history": False,
            "is_negative": True,
        },
        {
            "subject_uri": "overlap",
            "is_positive": True,
            "is_history": True,
            "is_negative": True,
        },
        {
            "subject_uri": "positive",
            "is_positive": True,
            "is_history": False,
            "is_negative": False,
        },
    ]


def test_post_summaries_count_duplicate_events_and_keep_zero_like_posts():
    selected = pl.DataFrame({
        "subject_uri": ["liked", "zero"],
        "is_positive": [True, False],
        "is_history": [False, False],
        "is_negative": [False, True],
    }, schema=post_liker_history.SELECTED_POST_SCHEMA)
    events = pl.DataFrame({
        "subject_uri": ["liked", "liked"],
        "liker_did": ["same-user", "same-user"],
        "like_created_at": [
            datetime(2026, 1, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 1, 1, tzinfo=UTC),
        ],
    }, schema=post_liker_history.POST_LIKER_EVENT_SCHEMA)

    event_audit = post_liker_history.audit_event_partition(events.lazy())
    assert event_audit.row(0, named=True) == {
        "subject_uri": "liked",
        "like_event_count": 2,
        "first_like_created_at": datetime(2026, 1, 1, 1, tzinfo=UTC),
        "last_like_created_at": datetime(2026, 1, 1, 1, tzinfo=UTC),
        "null_liker_did_count": 0,
        "empty_liker_did_count": 0,
        "null_timestamp_count": 0,
    }
    summaries = post_liker_history.build_post_liker_posts(
        selected,
        post_liker_history.event_stats_from_audit(event_audit),
    )

    assert summaries.schema == pl.Schema(post_liker_history.POST_LIKER_POST_SCHEMA)
    assert summaries.filter(pl.col("subject_uri") == "liked")[
        "like_event_count"
    ].item() == 2
    zero = summaries.filter(pl.col("subject_uri") == "zero").row(0, named=True)
    assert zero["like_event_count"] == 0
    assert zero["first_like_created_at"] is None
    assert zero["last_like_created_at"] is None


def test_partition_validation_rejects_events_outside_selected_posts():
    selected = pl.DataFrame({
        "subject_uri": ["selected"],
        "is_positive": [True],
        "is_history": [False],
        "is_negative": [False],
    }, schema=post_liker_history.SELECTED_POST_SCHEMA)
    events = pl.DataFrame({
        "subject_uri": ["other"],
        "liker_did": ["u"],
        "like_created_at": [datetime(2026, 1, 1, 1, tzinfo=UTC)],
    }, schema=post_liker_history.POST_LIKER_EVENT_SCHEMA)
    summaries = post_liker_history.build_post_liker_posts(
        selected,
        post_liker_history.empty_frame({
            "subject_uri": pl.String,
            "like_event_count": pl.UInt64,
            "first_like_created_at": post_liker_history.UTC_DATETIME,
            "last_like_created_at": post_liker_history.UTC_DATETIME,
        }),
    )

    with pytest.raises(ValueError, match="outside post_liker_posts"):
        post_liker_history.validate_public_partition(
            event_audit_df=post_liker_history.audit_event_partition(events.lazy()),
            post_liker_posts_df=summaries,
            selected_posts_df=selected,
            source_start=datetime(2026, 1, 1, tzinfo=UTC),
            source_end=datetime(2026, 1, 2, tzinfo=UTC),
            partition_id=0,
            partition_count=1,
        )
