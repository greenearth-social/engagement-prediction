from __future__ import annotations

from datetime import datetime, timezone
import json
import pickle

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from engagement_prediction.data import dataset_hydration
from engagement_prediction.data.training_index import (
    FORMAT_VERSION,
    SPLITS,
    MemoryMappedUtf8Table,
    build_loader_index,
    load_index_array,
    load_loader_index_metadata,
    validate_loader_index,
)


UTC = timezone.utc


def _write_dataset(root, name, frame):
    path = root / name
    path.mkdir()
    frame.write_parquet(path / "part-00000.parquet")
    return path


def _hydrated_inputs(tmp_path):
    bundle = tmp_path / "hydrated_training_data_test"
    bundle.mkdir()
    embeddings_path = bundle / "embeddings.npy"
    embeddings = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(4, 2),
    )
    embeddings[:] = np.array([[1, 0], [0, 1], [2, 0], [0, 2]], dtype=np.float32)
    embeddings.flush()
    del embeddings
    hour = datetime(2026, 1, 1, 12, tzinfo=UTC)
    created = datetime(2026, 1, 1, 11, 30, tzinfo=UTC)
    posts = pl.DataFrame(
        {
            "subject_uri": ["p2", "p1", "h1", "n1"],
            "emb_idx": [1, 0, 2, 3],
            "post_created_at": [created] * 4,
            "author_did": ["a3", "a2", "a4", "a5"],
            "author_idx": [3, 2, 4, 5],
            "is_reply": [False] * 4,
            "is_positive": [True, True, False, False],
            "is_history": [False, False, True, False],
            "is_negative": [False, False, False, True],
        },
        schema=dataset_hydration.POST_SCHEMA,
    )
    queries = pl.DataFrame(
        {
            "did": ["u2", "u1"],
            "query_hour": [hour, hour],
            "user_cohort": ["seen", "seen"],
            "split": ["train", "train"],
            "positive_count": [1, 1],
        },
        schema=dataset_hydration.QUERY_SCHEMA,
    )
    positives = pl.DataFrame(
        {
            "did": ["u2", "u1"],
            "query_hour": [hour, hour],
            "subject_uri": ["p2", "p1"],
            "like_created_at": [hour, hour],
            "emb_idx": [1, 0],
            "post_created_at": [created, created],
            "author_idx": [3, 2],
            "prior_like_count": [5, 4],
        },
        schema=dataset_hydration.QUERY_POSITIVE_SCHEMA,
    )
    histories = pl.DataFrame(
        {
            "did": ["u2", "u1"],
            "query_hour": [hour, hour],
            "history_subject_uris": [[], ["h1", "h1"]],
            "history_like_created_ats": [
                [],
                [
                    datetime(2026, 1, 1, 10, 30, tzinfo=UTC),
                    datetime(2026, 1, 1, 9, 30, tzinfo=UTC),
                ],
            ],
            "history_emb_indices": [[], [2, 2]],
            "history_author_indices": [[], [4, 4]],
            "history_prior_like_counts": [[], [3, 2]],
        },
        schema=dataset_hydration.QUERY_HISTORY_SCHEMA,
    )
    negatives = pl.DataFrame(
        {
            "query_hour": [hour, hour],
            "subject_uri": ["p1", "n1"],
            "selection_source": ["random", "popular"],
            "emb_idx": [0, 3],
            "post_created_at": [created, created],
            "author_idx": [2, 5],
            "prior_like_count": [4, 10],
        },
        schema=dataset_hydration.HOURLY_NEGATIVE_SCHEMA,
    )
    paths = {
        "posts_path": _write_dataset(bundle, "posts", posts),
        "queries_path": _write_dataset(bundle, "queries", queries),
        "query_positives_path": _write_dataset(bundle, "query_positives", positives),
        "query_histories_path": _write_dataset(bundle, "query_histories", histories),
        "hourly_negative_candidates_path": _write_dataset(
            bundle, "hourly_negative_candidates", negatives
        ),
        "embeddings_path": embeddings_path,
        "authors_path": _write_dataset(
            bundle,
            "authors",
            pl.DataFrame(
                {
                    "author_did": ["a2", "a3", "a4", "a5"],
                    "author_idx": pl.Series([2, 3, 4, 5], dtype=pl.UInt32),
                }
            ),
        ),
        "output_path": bundle / "loader_index",
        "logger": None,
    }
    return bundle, paths


def test_builds_exact_index_with_all_splits_and_relative_embedding_path(tmp_path):
    bundle, paths = _hydrated_inputs(tmp_path)

    stats = build_loader_index(**paths)
    metadata = load_loader_index_metadata(paths["output_path"])
    validation = validate_loader_index(paths["output_path"])

    assert stats["format_version"] == FORMAT_VERSION
    assert stats["splits"]["train"] == {
        "query_count": 2,
        "history_count": 2,
        "positive_count": 2,
        "hour_count": 1,
        "negative_count": 2,
    }
    assert set(metadata["splits"]) == set(SPLITS)
    assert metadata["embedding"]["path"] == "../embeddings.npy"
    assert validation["embedding_count"] == 4
    assert not (bundle / "loader_index.partial").exists()
    for split in SPLITS[1:]:
        assert stats["splits"][split] == {
            "query_count": 0,
            "history_count": 0,
            "positive_count": 0,
            "hour_count": 0,
            "negative_count": 0,
        }

    assert load_index_array(paths["output_path"], "post_author_idx").tolist() == [2, 3, 4, 5]
    assert load_index_array(
        paths["output_path"], "history_offsets", split="train"
    ).tolist() == [0, 2, 2]
    assert load_index_array(
        paths["output_path"], "history_emb_indices", split="train"
    ).tolist() == [2, 2]
    assert load_index_array(
        paths["output_path"], "positive_emb_indices", split="train"
    ).tolist() == [0, 1]
    assert load_index_array(
        paths["output_path"], "negative_emb_indices", split="train"
    ).tolist() == [3, 0]

    post_table_meta = metadata["arrow_tables"]["post_uris"]
    with MemoryMappedUtf8Table(
        paths["output_path"] / post_table_meta["path"],
        post_table_meta["batch_offsets"],
    ) as table:
        assert table.take([3, 0, 3]) == ["n1", "p1", "n1"]
    did_meta = metadata["splits"]["train"]["arrow_tables"]["query_dids"]
    with MemoryMappedUtf8Table(
        paths["output_path"] / did_meta["path"], did_meta["batch_offsets"]
    ) as table:
        assert table.take(np.array([1, 0])) == ["u2", "u1"]


def test_utf8_table_supports_multiple_batches_and_pickle_safe_lazy_reopen(tmp_path):
    path = tmp_path / "strings.arrow"
    schema = pa.schema([pa.field("value", pa.string(), nullable=False)])
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            writer.write_batch(pa.record_batch([pa.array(["a", "b"])], schema=schema))
            writer.write_batch(pa.record_batch([pa.array(["c"])], schema=schema))

    table = MemoryMappedUtf8Table(path, [0, 2, 3])
    assert not table.is_open
    assert table.take([2, 0, 1, 2]) == ["c", "a", "b", "c"]
    restored = pickle.loads(pickle.dumps(table))
    assert not restored.is_open
    assert restored.take([1]) == ["b"]
    assert restored.row_count == 3
    with pytest.raises(IndexError):
        restored.take([3])


def test_partition_local_fill_preserves_global_query_order(tmp_path):
    _, paths = _hydrated_inputs(tmp_path)
    for artifact_name in ("queries_path", "query_positives_path", "query_histories_path"):
        artifact_path = paths[artifact_name]
        source_path = artifact_path / "part-00000.parquet"
        frame = pl.read_parquet(source_path)
        source_path.unlink()
        frame.filter(pl.col("did") == "u2").write_parquet(
            artifact_path / "part-00000.parquet"
        )
        frame.filter(pl.col("did") == "u1").write_parquet(
            artifact_path / "part-00001.parquet"
        )

    build_loader_index(**paths)

    assert load_index_array(
        paths["output_path"], "history_offsets", split="train"
    ).tolist() == [0, 2, 2]
    assert load_index_array(
        paths["output_path"], "history_prior_like_counts", split="train"
    ).tolist() == [3, 2]
    assert load_index_array(
        paths["output_path"], "positive_emb_indices", split="train"
    ).tolist() == [0, 1]


def test_validator_rejects_corrupt_offsets(tmp_path):
    _, paths = _hydrated_inputs(tmp_path)
    build_loader_index(**paths)
    offset_path = paths["output_path"] / "splits/train/history_offsets.npy"
    offsets = np.load(offset_path, mmap_mode="r+")
    offsets[-1] = 1
    offsets.flush()
    del offsets

    with pytest.raises(ValueError, match="history_offsets does not span"):
        validate_loader_index(paths["output_path"])


def test_validator_rejects_invalid_embedding_indices(tmp_path):
    _, paths = _hydrated_inputs(tmp_path)
    build_loader_index(**paths)
    index_path = paths["output_path"] / "splits/train/history_emb_indices.npy"
    indices = np.load(index_path, mmap_mode="r+")
    indices[0] = 4
    indices.flush()
    del indices

    with pytest.raises(ValueError, match="invalid embedding index"):
        validate_loader_index(paths["output_path"])


def test_validator_rejects_malformed_format_and_arrow_alignment(tmp_path):
    _, paths = _hydrated_inputs(tmp_path)
    build_loader_index(**paths)
    format_path = paths["output_path"] / "format.json"
    metadata = json.loads(format_path.read_text())
    metadata["arrow_tables"]["post_uris"]["batch_offsets"] = [0, 3]
    format_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="Invalid Arrow batch offsets"):
        validate_loader_index(paths["output_path"])

    metadata["arrays"].pop("post_author_idx")
    format_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="unexpected global arrays"):
        validate_loader_index(paths["output_path"])


def test_builder_leaves_partial_output_when_source_integrity_fails(tmp_path):
    bundle, paths = _hydrated_inputs(tmp_path)
    posts_path = paths["posts_path"] / "part-00000.parquet"
    posts = pl.read_parquet(posts_path).with_columns(
        pl.when(pl.col("emb_idx") == 3)
        .then(pl.lit(2, dtype=pl.UInt32))
        .otherwise(pl.col("emb_idx"))
        .alias("emb_idx")
    )
    posts.write_parquet(posts_path)

    with pytest.raises(ValueError, match="unique and dense"):
        build_loader_index(**paths)

    assert not paths["output_path"].exists()
    assert (bundle / "loader_index.partial").exists()


def test_metadata_rejects_missing_and_unsupported_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="regenerate Stage 7"):
        load_loader_index_metadata(tmp_path)
    (tmp_path / "format.json").write_text(json.dumps({"format_version": 999}))
    with pytest.raises(ValueError, match="Unsupported.*Regenerate Stage 7"):
        load_loader_index_metadata(tmp_path)
