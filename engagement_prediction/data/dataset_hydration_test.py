from datetime import datetime, timezone
import base64
import struct
import zlib

import numpy as np
import polars as pl

from engagement_prediction.data import dataset_hydration


UTC = timezone.utc


def _compressed(values: list[float]) -> str:
    payload = struct.pack(f"<{len(values)}f", *values)
    return base64.b85encode(zlib.compress(payload)).decode()


def _embedding(value: str) -> list[dict[str, str]]:
    return [{"key": "all_MiniLM_L12_v2", "value": value}]


def test_select_latest_valid_embeddings_uses_latest_usable_row_and_stable_ties():
    rows = pl.DataFrame({
        "subject_uri": ["a", "a", "b", "b", "c"],
        "post_created_at": [
            datetime(2026, 1, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 1, 2, tzinfo=UTC),
            datetime(2026, 1, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 1, 1, tzinfo=UTC),
        ],
        "author_did": ["z", "z", "z", "a", "a"],
        "embeddings": [
            _embedding(_compressed([1.0, 2.0])),
            _embedding("corrupt"),
            _embedding(_compressed([3.0, 4.0])),
            _embedding(_compressed([5.0, 6.0])),
            _embedding(_compressed([1.0])),
        ],
    })

    selected, stats = dataset_hydration.select_latest_valid_embeddings(
        rows,
        embedding_model="all_MiniLM_L12_v2",
        embedding_dim=2,
    )

    assert selected.get_column("subject_uri").to_list() == ["a", "b"]
    assert selected.get_column("_emb_vec").to_list() == [[1.0, 2.0], [5.0, 6.0]]
    assert stats["invalid_embedding_count"] == 1
    assert stats["wrong_dimension_count"] == 1


def test_select_latest_valid_embeddings_reports_missing_and_non_finite_vectors():
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    rows = pl.DataFrame({
        "subject_uri": ["missing", "null", "nan"],
        "post_created_at": [created_at, created_at, created_at],
        "author_did": ["a", "a", "a"],
        "embeddings": [
            [{"key": "different-model", "value": _compressed([1.0, 2.0])}],
            None,
            _embedding(_compressed([float("nan"), 2.0])),
        ],
    })

    selected, stats = dataset_hydration.select_latest_valid_embeddings(
        rows,
        embedding_model="all_MiniLM_L12_v2",
        embedding_dim=2,
    )

    assert selected.is_empty()
    assert stats["null_embedding_count"] == 2
    assert stats["non_finite_embedding_count"] == 1


def test_hydrated_post_metadata_defers_then_applies_author_vocabulary():
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    metadata = pl.DataFrame({
        "subject_uri": ["a", "b"],
        "post_created_at": [created_at, created_at],
        "author_did": ["known", "unknown"],
        "is_reply": [False, True],
        "is_positive": [True, False],
        "is_history": [False, True],
        "is_negative": [False, False],
    })
    indices = pl.DataFrame({"subject_uri": ["a", "b"], "emb_idx": [0, 1]})
    authors = pl.DataFrame({"author_did": ["known"], "author_idx": [2]})

    hydrated_metadata = dataset_hydration.build_hydrated_post_metadata(
        metadata,
        indices,
    )
    result = dataset_hydration.attach_post_author_indices(
        hydrated_metadata,
        authors,
    )

    assert "author_idx" not in hydrated_metadata.columns
    assert result.get_column("author_idx").to_list() == [2, 1]
    assert result.schema == pl.Schema(dataset_hydration.POST_SCHEMA)
