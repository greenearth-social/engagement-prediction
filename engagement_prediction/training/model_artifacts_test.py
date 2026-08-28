"""Tests for model-independent training artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import torch

from engagement_prediction.data.author_vocabulary import AUTHOR_VOCABULARY_SCHEMA
from engagement_prediction.training.model_artifacts import (
    AUTHOR_MAP_SCHEMA,
    file_sha256,
    write_author_map,
    write_json_atomically,
    write_torch_checkpoint_atomically,
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


def test_write_torch_checkpoint_atomically_replaces_existing_file(tmp_path):
    checkpoint_path = tmp_path / "model.pth"
    partial_path = tmp_path / "model.pth.partial"
    checkpoint_path.write_bytes(b"old checkpoint")

    write_torch_checkpoint_atomically(
        checkpoint_path,
        {"epoch": 2, "tensor": torch.tensor([1.0, 2.0])},
    )

    assert checkpoint_path.is_file()
    assert not partial_path.exists()
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["epoch"] == 2
    torch.testing.assert_close(checkpoint["tensor"], torch.tensor([1.0, 2.0]))


def test_write_torch_checkpoint_atomically_cleans_partial_on_failure(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"old checkpoint")

    def fail_save(checkpoint, path):
        Path(path).write_bytes(b"partial checkpoint")
        raise RuntimeError("injected save failure")

    monkeypatch.setattr(torch, "save", fail_save)

    with pytest.raises(RuntimeError, match="injected save failure"):
        write_torch_checkpoint_atomically(checkpoint_path, {"epoch": 2})

    assert checkpoint_path.read_bytes() == b"old checkpoint"
    assert not (tmp_path / "model.pth.partial").exists()
