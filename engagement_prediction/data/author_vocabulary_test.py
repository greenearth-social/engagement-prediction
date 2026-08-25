import polars as pl
import pytest

from engagement_prediction.data import author_vocabulary


def test_support_aggregation_counts_roles_and_repeated_history_events():
    rows = pl.DataFrame({
        "author_did": ["a", "a", "a", "b"],
        "training_positive_count": pl.Series([1, 0, 0, 1], dtype=pl.UInt64),
        "training_history_count": pl.Series([0, 1, 1, 0], dtype=pl.UInt64),
        "training_negative_count": pl.Series([0, 0, 0, 0], dtype=pl.UInt64),
    })

    result = author_vocabulary.aggregate_support_rows(rows)

    assert result.to_dicts() == [
        {
            "author_did": "a",
            "training_feature_count": 3,
            "training_positive_count": 1,
            "training_history_count": 2,
            "training_negative_count": 0,
        },
        {
            "author_did": "b",
            "training_feature_count": 1,
            "training_positive_count": 1,
            "training_history_count": 0,
            "training_negative_count": 0,
        },
    ]


def test_vocabulary_keeps_exact_threshold_and_assigns_dense_sorted_indices():
    support = pl.DataFrame({
        "author_did": ["z", "a"],
        "training_feature_count": pl.Series([49, 50], dtype=pl.UInt64),
        "training_positive_count": pl.Series([49, 50], dtype=pl.UInt64),
        "training_history_count": pl.Series([0, 0], dtype=pl.UInt64),
        "training_negative_count": pl.Series([0, 0], dtype=pl.UInt64),
    }, schema=author_vocabulary.AUTHOR_SUPPORT_SCHEMA)

    authors = author_vocabulary.add_author_indices(
        support.lazy().filter(pl.col("training_feature_count") >= 50)
    ).collect()

    assert authors.select("author_did", "author_idx").to_dicts() == [
        {"author_did": "a", "author_idx": 2},
    ]
    assert author_vocabulary.validate_author_vocabulary(
        authors.lazy(),
        min_training_feature_count=50,
    )["author_table_num_rows"] == 3


def test_empty_vocabulary_preserves_schema_and_pad_unk_table_size():
    authors = author_vocabulary.empty_frame(
        author_vocabulary.AUTHOR_VOCABULARY_SCHEMA
    )

    assert author_vocabulary.validate_author_vocabulary(
        authors.lazy(),
        min_training_feature_count=50,
    ) == {
        "author_count": 0,
        "author_table_num_rows": 2,
        "training_feature_count": 0,
        "training_positive_count": 0,
        "training_history_count": 0,
        "training_negative_count": 0,
    }


def test_vocabulary_validation_rejects_below_threshold_rows():
    authors = pl.DataFrame({
        "author_did": ["a"],
        "author_idx": pl.Series([2], dtype=pl.UInt32),
        "training_feature_count": pl.Series([49], dtype=pl.UInt64),
        "training_positive_count": pl.Series([49], dtype=pl.UInt64),
        "training_history_count": pl.Series([0], dtype=pl.UInt64),
        "training_negative_count": pl.Series([0], dtype=pl.UInt64),
    }, schema=author_vocabulary.AUTHOR_VOCABULARY_SCHEMA)

    with pytest.raises(ValueError, match="below its support threshold"):
        author_vocabulary.validate_author_vocabulary(
            authors.lazy(),
            min_training_feature_count=50,
        )
