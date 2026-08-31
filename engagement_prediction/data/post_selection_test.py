from datetime import datetime, timezone

import polars as pl
import pytest

from engagement_prediction.data import post_selection


UTC = timezone.utc


def test_normalize_and_select_latest_rows_with_reply_flag_and_author_tie_break():
    raw = pl.DataFrame({
        "at_uri": ["p1", "p1", "p1", None],
        "record_created_at": [
            "2026-01-01T01:00:00Z",
            "2026-01-01T02:00:00Z",
            "2026-01-01T02:00:00Z",
            "bad",
        ],
        "did": ["old", "z-author", "a-author", "invalid"],
    })
    normalized = post_selection.normalize_posts(
        raw.lazy(),
        posts_start=datetime(2026, 1, 1, tzinfo=UTC),
        posts_end=datetime(2026, 1, 2, tzinfo=UTC),
        is_reply=True,
    ).collect()

    selected, stats = post_selection.select_latest_post_rows(normalized)

    assert selected.to_dicts() == [{
        "subject_uri": "p1",
        "post_created_at": datetime(2026, 1, 1, 2, tzinfo=UTC),
        "author_did": "a-author",
        "is_reply": True,
    }]
    assert stats == {
        "source_row_count": 4,
        "invalid_row_count": 1,
        "duplicate_row_count": 2,
        "duplicate_uri_count": 1,
        "unique_valid_count": 1,
    }


def test_normalize_posts_applies_half_open_creation_window():
    raw = pl.DataFrame({
        "at_uri": ["start", "end"],
        "record_created_at": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
        "did": ["a", "b"],
    })
    normalized = post_selection.normalize_posts(
        raw.lazy(),
        posts_start=datetime(2026, 1, 1, tzinfo=UTC),
        posts_end=datetime(2026, 1, 2, tzinfo=UTC),
        is_reply=False,
    ).collect()

    assert normalized["_post_row_valid"].to_list() == [True, False]
    assert normalized["is_reply"].to_list() == [False, False]


def test_root_precedence_when_uri_occurs_in_both_sources():
    root = pl.DataFrame({
        "subject_uri": ["overlap", "root"],
        "post_created_at": [datetime(2026, 1, 1, tzinfo=UTC)] * 2,
        "author_did": ["root-author", "root-author"],
        "is_reply": [False, False],
    }, schema=post_selection.POST_SCHEMA)
    replies = pl.DataFrame({
        "subject_uri": ["overlap", "reply"],
        "post_created_at": [datetime(2026, 1, 1, tzinfo=UTC)] * 2,
        "author_did": ["reply-author", "reply-author"],
        "is_reply": [True, True],
    }, schema=post_selection.POST_SCHEMA)

    resolved, overlap_count = post_selection.resolve_root_and_reply_posts(root, replies)

    assert overlap_count == 1
    assert resolved.to_dicts() == [
        {
            "subject_uri": "overlap",
            "post_created_at": datetime(2026, 1, 1, tzinfo=UTC),
            "author_did": "root-author",
            "is_reply": False,
        },
        {
            "subject_uri": "reply",
            "post_created_at": datetime(2026, 1, 1, tzinfo=UTC),
            "author_did": "reply-author",
            "is_reply": True,
        },
        {
            "subject_uri": "root",
            "post_created_at": datetime(2026, 1, 1, tzinfo=UTC),
            "author_did": "root-author",
            "is_reply": False,
        },
    ]


def test_random_sampling_is_stable_and_handles_zero_and_one():
    posts = pl.DataFrame({"subject_uri": [f"p{idx}" for idx in range(1000)]})
    first = posts.filter(post_selection.random_candidate_expr(0.1, 42))
    second = posts.filter(post_selection.random_candidate_expr(0.1, 42))
    changed_seed = posts.filter(post_selection.random_candidate_expr(0.1, 43))

    assert first.equals(second)
    assert not first.equals(changed_seed)
    assert posts.filter(post_selection.random_candidate_expr(0.0, 42)).is_empty()
    assert posts.filter(post_selection.random_candidate_expr(1.0, 42)).height == 1000


def test_validate_public_partition_rejects_reply_candidates():
    posts = pl.DataFrame({
        "subject_uri": ["reply"],
        "post_created_at": [datetime(2026, 1, 1, tzinfo=UTC)],
        "author_did": ["author"],
        "is_reply": [True],
    }, schema=post_selection.POST_SCHEMA)
    partition_id = posts.with_columns(
        post_selection.post_partition_expr(4)
    )["_post_partition"].item()
    candidates = pl.DataFrame({
        "subject_uri": ["reply"],
        "candidate_source": ["random"],
    }, schema=post_selection.CANDIDATE_SOURCE_SCHEMA)

    with pytest.raises(ValueError, match="contains a reply"):
        post_selection.validate_public_partition(
            posts_df=posts,
            required_posts_df=post_selection.empty_frame(post_selection.REQUIRED_POST_SCHEMA),
            candidate_sources_df=candidates,
            missing_required_posts_df=post_selection.empty_frame(
                post_selection.REQUIRED_POST_SCHEMA
            ),
            partition_id=partition_id,
            partition_count=4,
        )
