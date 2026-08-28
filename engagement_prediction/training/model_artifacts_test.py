"""Tests for model-independent training artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from engagement_prediction.data.author_vocabulary import AUTHOR_VOCABULARY_SCHEMA
from engagement_prediction.training.model_artifacts import (
    AUTHOR_MAP_SCHEMA,
    file_sha256,
    write_author_map,
    write_json_atomically,
)


def _write_authors(path: Path) -> None:
    path.mkdir()
    pl.DataFrame({
        "author_did": ["did:a", "did:b"],
        "author_idx": [2, 3],
        "training_feature_count": [50, 75],
        "training_positive_count": [10, 25],
        "training_history_count": [30, 25],
        "training_negative_count": [10, 25],
    }, schema=AUTHOR_VOCABULARY_SCHEMA).write_parquet(path / "part-00000.parquet")


def test_write_author_map_publishes_minimal_dense_mapping(tmp_path):
    authors_path = tmp_path / "authors"
    _write_authors(authors_path)
    output_path = tmp_path / "author_idx.parquet"

    stats = write_author_map(
        authors_path=authors_path,
        output_path=output_path,
        author_table_num_rows=4,
    )

    assert pl.read_parquet_schema(output_path) == pl.Schema(AUTHOR_MAP_SCHEMA)
    assert pl.read_parquet(output_path).to_dicts() == [
        {"author_did": "did:a", "author_idx": 2},
        {"author_did": "did:b", "author_idx": 3},
    ]
    assert stats["author_count"] == 2
    assert not (tmp_path / "author_idx.parquet.partial").exists()


def test_write_author_map_rejects_mismatched_table_size(tmp_path):
    authors_path = tmp_path / "authors"
    _write_authors(authors_path)

    with pytest.raises(ValueError, match="does not match"):
        write_author_map(
            authors_path=authors_path,
            output_path=tmp_path / "author_idx.parquet",
            author_table_num_rows=5,
        )


def test_write_json_atomically_publishes_deterministic_portable_json(tmp_path):
    output_path = tmp_path / "metadata.json"

    write_json_atomically(output_path, {"z": 1, "a": [2, 3]})

    assert output_path.read_text() == '{\n  "a": [\n    2,\n    3\n  ],\n  "z": 1\n}\n'
    assert json.loads(output_path.read_text()) == {"a": [2, 3], "z": 1}
    assert not (tmp_path / "metadata.json.partial").exists()


def test_write_json_atomically_rejects_nan_without_replacing_existing_file(tmp_path):
    output_path = tmp_path / "metadata.json"
    output_path.write_text("existing\n")

    with pytest.raises(ValueError, match="Out of range float values"):
        write_json_atomically(output_path, {"value": float("nan")})

    assert output_path.read_text() == "existing\n"
    assert not (tmp_path / "metadata.json.partial").exists()


def test_file_sha256_returns_known_digest(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")

    assert file_sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
