"""Build and read the compact, memory-mapped Stage 7 training index.

The hydrated Parquet datasets remain the inspectable source of truth.  This
module creates an equivalent, versioned representation designed for PyTorch
workers: numeric values live in read-only NumPy ``.npy`` mappings, variable
length relations use flat arrays plus offsets, and the two identifier columns
live in memory-mapped Arrow IPC files.

The builder deliberately uses temporary, externally sorted Parquet relations.
That keeps canonical ordering and exact allocation sizes without constructing
millions of Python dictionaries or lists.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterator

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from engagement_prediction.data.parquet import (
    scan_parquet_artifact,
    sink_partitioned_parquet,
)


FORMAT_VERSION = 1
SPLITS = (
    "train",
    "val",
    "val_unseen_users",
    "holdout_seen_users",
    "holdout_unseen_users",
)

INDEX_DTYPE = np.dtype("<u4")
OFFSET_DTYPE = np.dtype("<u8")
TIMESTAMP_DTYPE = np.dtype("<i8")
COUNT_DTYPE = np.dtype("<u8")
EMBEDDING_DTYPE = np.dtype("<f4")
ARROW_BATCH_SIZE = 65_536

GLOBAL_ARRAY_DTYPES = {
    "post_created_at_us": TIMESTAMP_DTYPE,
    "post_author_idx": INDEX_DTYPE,
}
SPLIT_ARRAY_DTYPES = {
    "query_hours_us": TIMESTAMP_DTYPE,
    "history_offsets": OFFSET_DTYPE,
    "history_emb_indices": INDEX_DTYPE,
    "history_like_created_at_us": TIMESTAMP_DTYPE,
    "history_prior_like_counts": COUNT_DTYPE,
    "positive_offsets": OFFSET_DTYPE,
    "positive_emb_indices": INDEX_DTYPE,
    "positive_prior_like_counts": COUNT_DTYPE,
    "hour_values_us": TIMESTAMP_DTYPE,
    "hour_query_offsets": OFFSET_DTYPE,
    "negative_offsets": OFFSET_DTYPE,
    "negative_emb_indices": INDEX_DTYPE,
    "negative_prior_like_counts": COUNT_DTYPE,
}


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _collect_scalar(lf: pl.LazyFrame, expression: pl.Expr, name: str) -> int:
    value = lf.select(expression.alias(name)).collect(engine="streaming").item()
    return int(value or 0)


def _row_count(lf: pl.LazyFrame) -> int:
    return _collect_scalar(lf, pl.len(), "row_count")


def _sink_sorted(lf: pl.LazyFrame, path: Path, columns: list[str]) -> None:
    """Stream an externally sortable narrow relation to one temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lf.select(columns).sink_parquet(
        path,
        compression="zstd",
        maintain_order=True,
        engine="streaming",
    )


def _iter_parquet_batches(
    path: Path,
    *,
    columns: list[str],
    batch_size: int = ARROW_BATCH_SIZE,
) -> Iterator[pa.RecordBatch]:
    parquet_file = pq.ParquetFile(path)
    yield from parquet_file.iter_batches(columns=columns, batch_size=batch_size)


def _timestamp_values(array: pa.Array) -> np.ndarray:
    if array.null_count:
        raise ValueError("Training-index timestamp arrays may not contain nulls")
    if not pa.types.is_timestamp(array.type):
        raise ValueError(f"Expected an Arrow timestamp array, found {array.type}")
    if array.type.tz != "UTC":
        raise ValueError(f"Training-index timestamps must use UTC, found {array.type}")
    normalized = array.cast(pa.timestamp("us", tz="UTC"))
    return np.asarray(
        normalized.cast(pa.int64()).to_numpy(zero_copy_only=False),
        dtype=TIMESTAMP_DTYPE,
    )


def _integer_values(array: pa.Array, dtype: np.dtype[Any]) -> np.ndarray:
    if array.null_count:
        raise ValueError("Training-index integer arrays may not contain nulls")
    values = array.to_numpy(zero_copy_only=False)
    if np.issubdtype(values.dtype, np.signedinteger) and np.any(values < 0):
        raise ValueError("Training-index unsigned values may not be negative")
    info = np.iinfo(dtype)
    if values.size and int(np.max(values)) > int(info.max):
        raise ValueError(f"Training-index value exceeds {dtype.str}")
    return np.asarray(values, dtype=dtype)


def _list_values(
    array: pa.Array,
    *,
    value_kind: str,
    dtype: np.dtype[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized local offsets and flat values without Python lists."""

    if array.null_count:
        raise ValueError("Training-index list arrays may not contain null lists")
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        raise ValueError(f"Expected an Arrow list array, found {array.type}")
    raw_offsets = np.asarray(array.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
    offsets = raw_offsets - raw_offsets[0]
    start = int(raw_offsets[0])
    end = int(raw_offsets[-1])
    values = array.values.slice(start, end - start)
    if value_kind == "timestamp":
        flat = _timestamp_values(values)
    else:
        flat = _integer_values(values, dtype)
    return np.asarray(offsets, dtype=OFFSET_DTYPE), flat


def _allocate_array(path: Path, dtype: np.dtype[Any], length: int) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(int(length),))


def _array_metadata(root: Path, path: Path, dtype: np.dtype[Any], length: int) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "dtype": dtype.str,
        "shape": [int(length)],
        "file_size_bytes": path.stat().st_size,
    }


class _Utf8IpcWriter:
    """Incrementally write one UTF-8 column while recording batch boundaries."""

    def __init__(self, path: Path, column: str):
        self.path = path
        self.column = column
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sink = pa.OSFile(str(path), "wb")
        self._schema = pa.schema([pa.field(column, pa.string(), nullable=False)])
        self._writer = ipc.new_file(self._sink, self._schema)
        self.batch_offsets = [0]

    def write(self, values: pa.Array) -> None:
        if values.null_count:
            raise ValueError(f"Arrow identifier column {self.column!r} contains nulls")
        if not pa.types.is_string(values.type):
            values = pc.cast(values, pa.string())
        batch = pa.RecordBatch.from_arrays([values], schema=self._schema)
        self._writer.write_batch(batch)
        self.batch_offsets.append(self.batch_offsets[-1] + len(values))

    def close(self) -> None:
        self._writer.close()
        self._sink.close()

    def __enter__(self) -> _Utf8IpcWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _arrow_metadata(
    root: Path,
    path: Path,
    *,
    column: str,
    row_count: int,
    batch_offsets: list[int],
) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "column": column,
        "row_count": int(row_count),
        "batch_offsets": [int(value) for value in batch_offsets],
        "file_size_bytes": path.stat().st_size,
    }


class MemoryMappedUtf8Table:
    """Process-safe, lazy reader for a one-column Arrow IPC UTF-8 table.

    Open Arrow mappings are intentionally excluded from pickle state.  A forked
    worker also notices the changed PID, closes its inherited handles, and opens
    its own read-only view before the first indexed access.
    """

    def __init__(self, path: Path, batch_offsets: Sequence[int] | None = None):
        self.path = Path(path)
        self._expected_batch_offsets = (
            tuple(int(value) for value in batch_offsets) if batch_offsets is not None else None
        )
        self._source: pa.MemoryMappedFile | None = None
        self._reader: ipc.RecordBatchFileReader | None = None
        self._batch_offsets: tuple[int, ...] | None = None
        self._owner_pid: int | None = None

    @property
    def is_open(self) -> bool:
        return self._reader is not None and self._owner_pid == os.getpid()

    @property
    def owner_pid(self) -> int | None:
        return self._owner_pid

    @property
    def batch_offsets(self) -> tuple[int, ...]:
        self._ensure_open()
        assert self._batch_offsets is not None
        return self._batch_offsets

    @property
    def row_count(self) -> int:
        return self.batch_offsets[-1]

    @property
    def column_name(self) -> str:
        self._ensure_open()
        assert self._reader is not None
        return self._reader.schema.field(0).name

    def _close_handles(self) -> None:
        self._reader = None
        if self._source is not None:
            self._source.close()
        self._source = None
        self._batch_offsets = None
        self._owner_pid = None

    def _ensure_open(self) -> None:
        pid = os.getpid()
        if self._reader is not None and self._owner_pid == pid:
            return
        self._close_handles()
        if not self.path.is_file():
            raise FileNotFoundError(f"Arrow training-index table does not exist: {self.path}")
        self._source = pa.memory_map(str(self.path), "r")
        self._reader = ipc.open_file(self._source)
        if len(self._reader.schema) != 1 or not pa.types.is_string(self._reader.schema.field(0).type):
            self._close_handles()
            raise ValueError(f"Expected one non-null UTF-8 column in {self.path}")
        if self._reader.schema.field(0).nullable:
            self._close_handles()
            raise ValueError(f"Expected a non-null UTF-8 field in {self.path}")
        offsets = [0]
        for batch_index in range(self._reader.num_record_batches):
            batch = self._reader.get_batch(batch_index)
            if batch.column(0).null_count:
                self._close_handles()
                raise ValueError(f"UTF-8 table contains null identifiers: {self.path}")
            offsets.append(offsets[-1] + batch.num_rows)
        actual_offsets = tuple(offsets)
        if (
            self._expected_batch_offsets is not None
            and actual_offsets != self._expected_batch_offsets
        ):
            self._close_handles()
            raise ValueError(f"Arrow batch offsets do not match metadata for {self.path}")
        self._batch_offsets = actual_offsets
        self._owner_pid = pid

    def take(self, indices: Sequence[int] | np.ndarray) -> list[str]:
        """Decode only the requested rows, preserving order and duplicates."""

        self._ensure_open()
        assert self._reader is not None
        assert self._batch_offsets is not None
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("UTF-8 table indices must be one-dimensional")
        if requested.size == 0:
            return []
        if np.any(requested < 0) or np.any(requested >= self._batch_offsets[-1]):
            raise IndexError("UTF-8 table index is out of range")
        boundaries = np.asarray(self._batch_offsets, dtype=np.int64)
        batch_indices = np.searchsorted(boundaries[1:], requested, side="right")
        output: list[str | None] = [None] * requested.size
        for batch_index in np.unique(batch_indices):
            positions = np.flatnonzero(batch_indices == batch_index)
            local_indices = requested[positions] - boundaries[batch_index]
            values = pc.take(
                self._reader.get_batch(int(batch_index)).column(0),
                pa.array(local_indices, type=pa.int64()),
            ).to_pylist()
            for position, value in zip(positions.tolist(), values, strict=True):
                output[position] = value
        if any(value is None for value in output):
            raise ValueError(f"UTF-8 table unexpectedly returned nulls: {self.path}")
        return [str(value) for value in output]

    def close(self) -> None:
        self._close_handles()

    def __getstate__(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "_expected_batch_offsets": self._expected_batch_offsets,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.path = Path(state["path"])
        self._expected_batch_offsets = state["_expected_batch_offsets"]
        self._source = None
        self._reader = None
        self._batch_offsets = None
        self._owner_pid = None

    def __enter__(self) -> MemoryMappedUtf8Table:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def load_loader_index_metadata(index_path: Path) -> dict[str, Any]:
    """Load the format descriptor and reject unsupported index versions."""

    index_path = Path(index_path)
    metadata_path = index_path / "format.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Stage 7 loader index is missing {metadata_path}; regenerate Stage 7"
        )
    try:
        metadata = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Could not read Stage 7 loader-index metadata: {metadata_path}") from exc
    version = metadata.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported Stage 7 loader-index format {version!r}; expected "
            f"{FORMAT_VERSION}. Regenerate Stage 7."
        )
    return metadata


def _metadata_entry(
    metadata: dict[str, Any],
    name: str,
    split: str | None,
) -> dict[str, Any]:
    if split is None:
        section = metadata.get("arrays", {})
    else:
        if split not in SPLITS:
            raise ValueError(f"Unknown loader-index split: {split!r}")
        section = metadata.get("splits", {}).get(split, {}).get("arrays", {})
    if name not in section:
        location = "global index" if split is None else f"split {split!r}"
        raise KeyError(f"No array {name!r} in {location}")
    return section[name]


def load_index_array(index_path: Path, name: str, split: str | None = None) -> np.memmap:
    """Open one declared numeric index array as a read-only NumPy mapping."""

    index_path = Path(index_path)
    metadata = load_loader_index_metadata(index_path)
    entry = _metadata_entry(metadata, name, split)
    array = np.load(index_path / entry["path"], mmap_mode="r", allow_pickle=False)
    if array.dtype.str != entry["dtype"] or list(array.shape) != entry["shape"]:
        raise ValueError(f"Loader-index array does not match metadata: {entry['path']}")
    return array


def _write_global_post_index(
    *,
    root: Path,
    sorted_posts_path: Path,
    post_count: int,
    embedding_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if post_count != embedding_count:
        raise ValueError(
            "Hydrated posts must contain exactly one row per embedding: "
            f"posts={post_count:,} embeddings={embedding_count:,}"
        )
    created_path = root / "post_created_at_us.npy"
    author_path = root / "post_author_idx.npy"
    uri_path = root / "post_uris.arrow"
    created = _allocate_array(created_path, TIMESTAMP_DTYPE, post_count)
    authors = _allocate_array(author_path, INDEX_DTYPE, post_count)
    position = 0
    last_emb_idx = -1
    with _Utf8IpcWriter(uri_path, "subject_uri") as uri_writer:
        for batch in _iter_parquet_batches(
            sorted_posts_path,
            columns=["subject_uri", "emb_idx", "post_created_at", "author_idx"],
        ):
            size = batch.num_rows
            emb_indices = _integer_values(batch.column("emb_idx"), INDEX_DTYPE)
            expected = np.arange(position, position + size, dtype=INDEX_DTYPE)
            if not np.array_equal(emb_indices, expected):
                raise ValueError("Hydrated post emb_idx values must be unique and dense from zero")
            if size:
                last_emb_idx = int(emb_indices[-1])
            created[position:position + size] = _timestamp_values(batch.column("post_created_at"))
            authors[position:position + size] = _integer_values(
                batch.column("author_idx"), INDEX_DTYPE
            )
            uri_writer.write(batch.column("subject_uri"))
            position += size
        uri_offsets = uri_writer.batch_offsets
    created.flush()
    authors.flush()
    del created, authors
    if position != post_count or (post_count and last_emb_idx != post_count - 1):
        raise ValueError("Hydrated post index did not fill the expected dense embedding rows")
    arrays = {
        "post_created_at_us": _array_metadata(
            root, created_path, TIMESTAMP_DTYPE, post_count
        ),
        "post_author_idx": _array_metadata(root, author_path, INDEX_DTYPE, post_count),
    }
    arrow_tables = {
        "post_uris": _arrow_metadata(
            root,
            uri_path,
            column="subject_uri",
            row_count=post_count,
            batch_offsets=uri_offsets,
        )
    }
    return arrays, arrow_tables


def _write_query_core(
    *,
    root: Path,
    split_dir: Path,
    query_path: Path,
    query_count: int,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    hours_path = split_dir / "query_hours_us.npy"
    dids_path = split_dir / "query_dids.arrow"
    hours = _allocate_array(hours_path, TIMESTAMP_DTYPE, query_count)
    position = 0
    with _Utf8IpcWriter(dids_path, "did") as did_writer:
        for batch in _iter_parquet_batches(query_path, columns=["did", "query_hour"]):
            size = batch.num_rows
            hours[position:position + size] = _timestamp_values(batch.column("query_hour"))
            did_writer.write(batch.column("did"))
            position += size
        did_offsets = did_writer.batch_offsets
    hours.flush()
    if position != query_count:
        raise ValueError(f"Query index for {split_dir.name!r} has an unexpected row count")
    hour_values, hour_counts = np.unique(np.asarray(hours), return_counts=True)
    query_offsets = np.empty(hour_values.size + 1, dtype=OFFSET_DTYPE)
    query_offsets[0] = 0
    np.cumsum(hour_counts, dtype=OFFSET_DTYPE, out=query_offsets[1:])
    del hours
    arrays = {
        "query_hours_us": _array_metadata(root, hours_path, TIMESTAMP_DTYPE, query_count),
    }
    arrow_tables = {
        "query_dids": _arrow_metadata(
            root,
            dids_path,
            column="did",
            row_count=query_count,
            batch_offsets=did_offsets,
        )
    }
    return arrays, arrow_tables, hour_values.astype(TIMESTAMP_DTYPE), query_offsets


def _route_paths(path: Path, partition_id: int) -> list[Path]:
    partition_path = path / f"_source_partition={partition_id}"
    return sorted(partition_path.rglob("*.parquet")) if partition_path.exists() else []


def _read_query_route(path: Path, partition_id: int) -> pl.DataFrame:
    paths = _route_paths(path, partition_id)
    if not paths:
        return pl.DataFrame(
            schema={
                "_query_idx": pl.UInt32,
                "did": pl.String,
                "query_hour": pl.Datetime("us", "UTC"),
                "positive_count": pl.UInt32,
            }
        )
    return pl.read_parquet(paths).sort("_query_idx")


def _joined_partition(
    *,
    source_path: Path | None,
    query_route: pl.DataFrame,
    kind: str,
) -> pl.DataFrame:
    if source_path is None:
        raise ValueError(f"Hydrated query partition is missing its aligned {kind} part")
    if kind == "history":
        columns = [
            "did",
            "query_hour",
            "history_emb_indices",
            "history_like_created_ats",
            "history_prior_like_counts",
        ]
    elif kind == "positive":
        columns = [
            "did",
            "query_hour",
            "subject_uri",
            "emb_idx",
            "prior_like_count",
        ]
    else:
        raise ValueError(f"Unknown partitioned query relation: {kind!r}")
    source = pl.read_parquet(source_path, columns=columns)
    return source.join(
        query_route.select("_query_idx", "did", "query_hour"),
        on=["did", "query_hour"],
        how="inner",
    )


def _scatter_history_slice(
    *,
    frame: pl.DataFrame,
    global_offsets: np.ndarray,
    emb_indices: np.memmap,
    liked_ats: np.memmap,
    prior_counts: np.memmap,
) -> int:
    table = frame.rechunk().to_arrow()
    query_indices = np.asarray(
        table.column("_query_idx").chunk(0).to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    emb_offsets, local_emb_indices = _list_values(
        table.column("history_emb_indices").chunk(0),
        value_kind="integer",
        dtype=INDEX_DTYPE,
    )
    liked_offsets, local_liked_ats = _list_values(
        table.column("history_like_created_ats").chunk(0),
        value_kind="timestamp",
        dtype=TIMESTAMP_DTYPE,
    )
    count_offsets, local_counts = _list_values(
        table.column("history_prior_like_counts").chunk(0),
        value_kind="integer",
        dtype=COUNT_DTYPE,
    )
    if not np.array_equal(emb_offsets, liked_offsets) or not np.array_equal(
        emb_offsets, count_offsets
    ):
        raise ValueError("Hydrated history feature lists are not aligned")
    lengths = np.diff(emb_offsets.astype(np.int64, copy=False))
    value_count = int(lengths.sum())
    if not value_count:
        return 0
    local_starts = np.repeat(emb_offsets[:-1].astype(np.int64, copy=False), lengths)
    global_starts = np.repeat(
        np.asarray(global_offsets[query_indices], dtype=np.int64), lengths
    )
    destinations = global_starts + np.arange(value_count, dtype=np.int64) - local_starts
    emb_indices[destinations] = local_emb_indices
    liked_ats[destinations] = local_liked_ats
    prior_counts[destinations] = local_counts
    return value_count


def _write_partitioned_query_relations(
    *,
    root: Path,
    split_dir: Path,
    query_routes_path: Path,
    source_part_keys: dict[int, str],
    positive_parts: dict[str, Path],
    history_parts: dict[str, Path],
    query_count: int,
) -> tuple[dict[str, Any], int, int]:
    """Count, allocate, and fill histories/positives one user partition at a time."""

    history_lengths = np.zeros(query_count, dtype=OFFSET_DTYPE)
    positive_lengths = np.zeros(query_count, dtype=OFFSET_DTYPE)
    seen_histories = np.zeros(query_count, dtype=bool)
    expected_positive_lengths = np.zeros(query_count, dtype=OFFSET_DTYPE)

    # The first bounded pass establishes exact ragged dimensions without
    # copying list-bearing histories into a global sorted temporary relation.
    for partition_id, source_key in source_part_keys.items():
        query_route = _read_query_route(query_routes_path, partition_id)
        if query_route.is_empty():
            continue
        query_indices = query_route.get_column("_query_idx").to_numpy().astype(
            np.intp, copy=False
        )
        expected_positive_lengths[query_indices] = query_route.get_column(
            "positive_count"
        ).to_numpy().astype(OFFSET_DTYPE, copy=False)
        histories = _joined_partition(
            source_path=history_parts.get(source_key),
            query_route=query_route,
            kind="history",
        ).select("_query_idx", "history_emb_indices")
        if histories.height != query_route.height or histories.get_column(
            "_query_idx"
        ).n_unique() != query_route.height:
            raise ValueError("Every query must have exactly one aligned history row")
        history_query_indices = histories.get_column("_query_idx").to_numpy().astype(
            np.intp, copy=False
        )
        history_lengths[history_query_indices] = histories.get_column(
            "history_emb_indices"
        ).list.len().to_numpy().astype(OFFSET_DTYPE, copy=False)
        seen_histories[history_query_indices] = True

        positives = _joined_partition(
            source_path=positive_parts.get(source_key),
            query_route=query_route,
            kind="positive",
        )
        positive_counts = positives.group_by("_query_idx").len()
        positive_query_indices = positive_counts.get_column("_query_idx").to_numpy().astype(
            np.intp, copy=False
        )
        positive_lengths[positive_query_indices] = positive_counts.get_column(
            "len"
        ).to_numpy().astype(OFFSET_DTYPE, copy=False)

    if query_count and not seen_histories.all():
        raise ValueError("Hydrated queries are missing aligned history rows")
    if not np.array_equal(positive_lengths, expected_positive_lengths):
        raise ValueError("Query positive_count does not match hydrated positive rows")
    if query_count and np.any(positive_lengths == 0):
        raise ValueError("Every model-ready query must retain at least one positive")
    history_count = int(history_lengths.sum(dtype=OFFSET_DTYPE))
    positive_count = int(positive_lengths.sum(dtype=OFFSET_DTYPE))

    paths = {
        "history_offsets": split_dir / "history_offsets.npy",
        "history_emb_indices": split_dir / "history_emb_indices.npy",
        "history_like_created_at_us": split_dir / "history_like_created_at_us.npy",
        "history_prior_like_counts": split_dir / "history_prior_like_counts.npy",
        "positive_offsets": split_dir / "positive_offsets.npy",
        "positive_emb_indices": split_dir / "positive_emb_indices.npy",
        "positive_prior_like_counts": split_dir / "positive_prior_like_counts.npy",
    }
    history_offsets = _allocate_array(
        paths["history_offsets"], OFFSET_DTYPE, query_count + 1
    )
    history_offsets[0] = 0
    np.cumsum(history_lengths, dtype=OFFSET_DTYPE, out=history_offsets[1:])
    history_emb_indices = _allocate_array(
        paths["history_emb_indices"], INDEX_DTYPE, history_count
    )
    history_liked_ats = _allocate_array(
        paths["history_like_created_at_us"], TIMESTAMP_DTYPE, history_count
    )
    history_prior_counts = _allocate_array(
        paths["history_prior_like_counts"], COUNT_DTYPE, history_count
    )
    positive_offsets = _allocate_array(
        paths["positive_offsets"], OFFSET_DTYPE, query_count + 1
    )
    positive_offsets[0] = 0
    np.cumsum(positive_lengths, dtype=OFFSET_DTYPE, out=positive_offsets[1:])
    positive_emb_indices = _allocate_array(
        paths["positive_emb_indices"], INDEX_DTYPE, positive_count
    )
    positive_prior_counts = _allocate_array(
        paths["positive_prior_like_counts"], COUNT_DTYPE, positive_count
    )

    filled_histories = 0
    filled_positives = 0
    # The second bounded pass scatters each partition into the exact global
    # positions assigned by canonical (query_hour, did) order.
    for partition_id, source_key in source_part_keys.items():
        query_route = _read_query_route(query_routes_path, partition_id)
        if query_route.is_empty():
            continue
        histories = _joined_partition(
            source_path=history_parts.get(source_key),
            query_route=query_route,
            kind="history",
        ).sort("_query_idx")
        for history_slice in histories.iter_slices(16_384):
            filled_histories += _scatter_history_slice(
                frame=history_slice,
                global_offsets=history_offsets,
                emb_indices=history_emb_indices,
                liked_ats=history_liked_ats,
                prior_counts=history_prior_counts,
            )

        positives = _joined_partition(
            source_path=positive_parts.get(source_key),
            query_route=query_route,
            kind="positive",
        ).sort("_query_idx", "subject_uri")
        if positives.height:
            query_indices = positives.get_column("_query_idx").to_numpy().astype(
                np.int64, copy=False
            )
            group_starts = np.maximum.accumulate(
                np.where(
                    np.concatenate(([True], query_indices[1:] != query_indices[:-1])),
                    np.arange(query_indices.size, dtype=np.int64),
                    0,
                )
            )
            positions = np.arange(query_indices.size, dtype=np.int64) - group_starts
            destinations = (
                np.asarray(positive_offsets[query_indices], dtype=np.int64) + positions
            )
            positive_emb_indices[destinations] = positives.get_column(
                "emb_idx"
            ).to_numpy().astype(INDEX_DTYPE, copy=False)
            positive_prior_counts[destinations] = positives.get_column(
                "prior_like_count"
            ).to_numpy().astype(COUNT_DTYPE, copy=False)
            filled_positives += positives.height

    for array in (
        history_offsets,
        history_emb_indices,
        history_liked_ats,
        history_prior_counts,
        positive_offsets,
        positive_emb_indices,
        positive_prior_counts,
    ):
        array.flush()
    del (
        history_offsets,
        history_emb_indices,
        history_liked_ats,
        history_prior_counts,
        positive_offsets,
        positive_emb_indices,
        positive_prior_counts,
    )
    if filled_histories != history_count or filled_positives != positive_count:
        raise ValueError("Partitioned query relations did not fill their exact allocations")
    lengths = {
        "history_offsets": query_count + 1,
        "history_emb_indices": history_count,
        "history_like_created_at_us": history_count,
        "history_prior_like_counts": history_count,
        "positive_offsets": query_count + 1,
        "positive_emb_indices": positive_count,
        "positive_prior_like_counts": positive_count,
    }
    return (
        {
            name: _array_metadata(root, path, SPLIT_ARRAY_DTYPES[name], lengths[name])
            for name, path in paths.items()
        },
        history_count,
        positive_count,
    )


def _write_hour_and_negatives(
    *,
    root: Path,
    split_dir: Path,
    negatives_path: Path,
    hour_values: np.ndarray,
    query_offsets: np.ndarray,
    negative_count: int,
) -> dict[str, Any]:
    paths = {
        "hour_values_us": split_dir / "hour_values_us.npy",
        "hour_query_offsets": split_dir / "hour_query_offsets.npy",
        "negative_offsets": split_dir / "negative_offsets.npy",
        "negative_emb_indices": split_dir / "negative_emb_indices.npy",
        "negative_prior_like_counts": split_dir / "negative_prior_like_counts.npy",
    }
    hour_array = _allocate_array(paths["hour_values_us"], TIMESTAMP_DTYPE, hour_values.size)
    hour_array[:] = hour_values
    query_offset_array = _allocate_array(
        paths["hour_query_offsets"], OFFSET_DTYPE, query_offsets.size
    )
    query_offset_array[:] = query_offsets
    negative_offsets = _allocate_array(
        paths["negative_offsets"], OFFSET_DTYPE, hour_values.size + 1
    )
    emb_indices = _allocate_array(
        paths["negative_emb_indices"], INDEX_DTYPE, negative_count
    )
    prior_counts = _allocate_array(
        paths["negative_prior_like_counts"], COUNT_DTYPE, negative_count
    )
    counts_by_hour = np.zeros(hour_values.size, dtype=OFFSET_DTYPE)
    position = 0
    for batch in _iter_parquet_batches(
        negatives_path,
        columns=["query_hour", "emb_idx", "prior_like_count"],
    ):
        hours = _timestamp_values(batch.column("query_hour"))
        hour_indices = np.searchsorted(hour_values, hours)
        if hours.size and (
            np.any(hour_indices >= hour_values.size)
            or not np.array_equal(hour_values[hour_indices], hours)
        ):
            raise ValueError("Negative row refers to an hour without retained queries")
        if hour_indices.size and np.any(hour_indices[1:] < hour_indices[:-1]):
            raise ValueError("Negative rows are not in canonical hour order")
        np.add.at(counts_by_hour, hour_indices.astype(np.intp), 1)
        size = batch.num_rows
        emb_indices[position:position + size] = _integer_values(
            batch.column("emb_idx"), INDEX_DTYPE
        )
        prior_counts[position:position + size] = _integer_values(
            batch.column("prior_like_count"), COUNT_DTYPE
        )
        position += size
    negative_offsets[0] = 0
    np.cumsum(counts_by_hour, dtype=OFFSET_DTYPE, out=negative_offsets[1:])
    for array in (
        hour_array,
        query_offset_array,
        negative_offsets,
        emb_indices,
        prior_counts,
    ):
        array.flush()
    del hour_array, query_offset_array, negative_offsets, emb_indices, prior_counts
    if position != negative_count:
        raise ValueError("Negative index did not fill its exact allocated dimensions")
    lengths = {
        "hour_values_us": hour_values.size,
        "hour_query_offsets": query_offsets.size,
        "negative_offsets": hour_values.size + 1,
        "negative_emb_indices": negative_count,
        "negative_prior_like_counts": negative_count,
    }
    return {
        name: _array_metadata(root, path, SPLIT_ARRAY_DTYPES[name], int(lengths[name]))
        for name, path in paths.items()
    }


def _prepare_split_relations(
    *,
    split: str,
    working_path: Path,
    queries_with_source_lf: pl.LazyFrame,
    negatives_lf: pl.LazyFrame,
) -> tuple[Path, Path, Path]:
    query_path = working_path / f"{split}_queries.parquet"
    query_routes_path = working_path / f"{split}_query_routes"
    negative_path = working_path / f"{split}_negatives.parquet"
    query_map = (
        queries_with_source_lf.filter(pl.col("split") == split)
        .select("did", "query_hour", "positive_count", "_source_partition")
        .sort("query_hour", "did")
        .with_row_index("_query_idx")
    )
    _sink_sorted(
        query_map,
        query_path,
        [
            "_query_idx",
            "did",
            "query_hour",
            "positive_count",
            "_source_partition",
        ],
    )
    persisted_queries = pl.scan_parquet(query_path)
    sink_partitioned_parquet(
        persisted_queries,
        output_path=query_routes_path,
        key="_source_partition",
    )
    query_hours = persisted_queries.select("query_hour").unique()
    _sink_sorted(
        negatives_lf.join(query_hours, on="query_hour", how="semi").sort(
            "query_hour", "subject_uri"
        ),
        negative_path,
        ["query_hour", "emb_idx", "prior_like_count"],
    )
    return query_path, query_routes_path, negative_path


def _build_split(
    *,
    root: Path,
    split: str,
    working_path: Path,
    queries_with_source_lf: pl.LazyFrame,
    negatives_lf: pl.LazyFrame,
    source_part_keys: dict[int, str],
    positive_parts: dict[str, Path],
    history_parts: dict[str, Path],
) -> dict[str, Any]:
    split_dir = root / "splits" / split
    split_dir.mkdir(parents=True, exist_ok=False)
    query_path, query_routes_path, negative_path = _prepare_split_relations(
        split=split,
        working_path=working_path,
        queries_with_source_lf=queries_with_source_lf,
        negatives_lf=negatives_lf,
    )
    query_lf = pl.scan_parquet(query_path)
    negative_lf = pl.scan_parquet(negative_path)
    query_count = _row_count(query_lf)
    negative_count = _row_count(negative_lf)
    arrays, arrow_tables, hour_values, query_offsets = _write_query_core(
        root=root,
        split_dir=split_dir,
        query_path=query_path,
        query_count=query_count,
    )
    query_relation_arrays, history_count, positive_count = (
        _write_partitioned_query_relations(
            root=root,
            split_dir=split_dir,
            query_routes_path=query_routes_path,
            source_part_keys=source_part_keys,
            positive_parts=positive_parts,
            history_parts=history_parts,
            query_count=query_count,
        )
    )
    arrays.update(query_relation_arrays)
    arrays.update(
        _write_hour_and_negatives(
            root=root,
            split_dir=split_dir,
            negatives_path=negative_path,
            hour_values=hour_values,
            query_offsets=query_offsets,
            negative_count=negative_count,
        )
    )
    return {
        "counts": {
            "query_count": query_count,
            "history_count": history_count,
            "positive_count": positive_count,
            "hour_count": int(hour_values.size),
            "negative_count": negative_count,
        },
        "arrays": arrays,
        "arrow_tables": arrow_tables,
    }


def _total_declared_bytes(metadata: dict[str, Any]) -> int:
    total = sum(entry["file_size_bytes"] for entry in metadata["arrays"].values())
    total += sum(entry["file_size_bytes"] for entry in metadata["arrow_tables"].values())
    for split_metadata in metadata["splits"].values():
        total += sum(
            entry["file_size_bytes"] for entry in split_metadata["arrays"].values()
        )
        total += sum(
            entry["file_size_bytes"]
            for entry in split_metadata["arrow_tables"].values()
        )
    return int(total)


def _artifact_part_map(path: Path) -> dict[str, Path]:
    """Return stable relative part keys shared by aligned Stage 7 artifacts."""

    path = Path(path)
    if path.is_file():
        return {"__single__": path}
    if not path.is_dir():
        raise FileNotFoundError(f"Parquet artifact does not exist: {path}")
    parts = sorted(path.rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No Parquet parts found under {path}")
    return {part.relative_to(path).as_posix(): part for part in parts}


def _queries_with_source_partitions(
    query_parts: dict[str, Path],
) -> tuple[pl.LazyFrame, dict[int, str]]:
    source_part_keys = {
        partition_id: source_key
        for partition_id, source_key in enumerate(sorted(query_parts))
    }
    frames = [
        pl.scan_parquet(query_parts[source_key]).with_columns(
            pl.lit(partition_id, dtype=pl.UInt32).alias("_source_partition")
        )
        for partition_id, source_key in source_part_keys.items()
    ]
    return pl.concat(frames, how="vertical"), source_part_keys


def build_loader_index(
    posts_path: Path,
    queries_path: Path,
    query_positives_path: Path,
    query_histories_path: Path,
    hourly_negative_candidates_path: Path,
    embeddings_path: Path,
    authors_path: Path,
    output_path: Path,
    logger: logging.Logger | None,
) -> dict[str, Any]:
    """Build, validate, and atomically publish a compact Stage 7 loader index."""

    started_at = time.time()
    output_path = Path(output_path)
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    if output_path.exists():
        raise FileExistsError(f"Loader index already exists: {output_path}")
    if partial_path.exists():
        raise FileExistsError(f"Loader-index partial output already exists: {partial_path}")
    partial_path.mkdir(parents=True)
    working_path = partial_path / "_working"
    working_path.mkdir()
    embeddings = np.load(Path(embeddings_path), mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.dtype.str != EMBEDDING_DTYPE.str:
        raise ValueError(
            f"Stage 7 embeddings must be a {EMBEDDING_DTYPE.str} matrix, found "
            f"{embeddings.dtype.str} {embeddings.shape}"
        )
    embedding_count, embedding_dim = (int(value) for value in embeddings.shape)
    del embeddings
    posts_lf = scan_parquet_artifact(Path(posts_path))
    query_parts = _artifact_part_map(Path(queries_path))
    positive_parts = _artifact_part_map(Path(query_positives_path))
    history_parts = _artifact_part_map(Path(query_histories_path))
    if set(positive_parts) != set(query_parts):
        raise ValueError(
            "Hydrated query-positive partitions do not align with query partitions"
        )
    if set(history_parts) != set(query_parts):
        raise ValueError(
            "Hydrated query-history partitions do not align with query partitions"
        )
    queries_with_source_lf, source_part_keys = _queries_with_source_partitions(
        query_parts
    )
    negatives_lf = scan_parquet_artifact(Path(hourly_negative_candidates_path))
    post_count = _row_count(posts_lf)
    sorted_posts_path = working_path / "posts_by_emb_idx.parquet"
    if logger:
        logger.info(
            "Building loader index v%s: posts=%s embeddings=%s dim=%s",
            FORMAT_VERSION,
            f"{post_count:,}",
            f"{embedding_count:,}",
            f"{embedding_dim:,}",
        )
    _sink_sorted(
        posts_lf.sort("emb_idx"),
        sorted_posts_path,
        ["subject_uri", "emb_idx", "post_created_at", "author_idx"],
    )
    arrays, arrow_tables = _write_global_post_index(
        root=partial_path,
        sorted_posts_path=sorted_posts_path,
        post_count=post_count,
        embedding_count=embedding_count,
    )
    author_max = (
        scan_parquet_artifact(Path(authors_path))
        .select(pl.col("author_idx").max().alias("author_idx"))
        .collect(engine="streaming")
        .item()
    )
    author_table_num_rows = max(int(author_max or 1) + 1, 2)
    metadata: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "embedding": {
            "path": os.path.relpath(Path(embeddings_path).resolve(), partial_path.resolve()),
            "dtype": EMBEDDING_DTYPE.str,
            "shape": [embedding_count, embedding_dim],
        },
        "author_table_num_rows": author_table_num_rows,
        "arrays": arrays,
        "arrow_tables": arrow_tables,
        "splits": {},
    }
    for split in SPLITS:
        if logger:
            logger.info("Building loader-index split %s", split)
        metadata["splits"][split] = _build_split(
            root=partial_path,
            split=split,
            working_path=working_path,
            queries_with_source_lf=queries_with_source_lf,
            negatives_lf=negatives_lf,
            source_part_keys=source_part_keys,
            positive_parts=positive_parts,
            history_parts=history_parts,
        )
    shutil.rmtree(working_path)
    metadata["total_data_bytes"] = _total_declared_bytes(metadata)
    _json_dump(partial_path / "format.json", metadata)
    validation = validate_loader_index(partial_path)
    partial_path.replace(output_path)
    total_bytes = sum(
        path.stat().st_size for path in output_path.rglob("*") if path.is_file()
    )
    stats = {
        "format_version": FORMAT_VERSION,
        "output_path": str(output_path),
        "total_bytes": int(total_bytes),
        "post_count": post_count,
        "embedding_count": embedding_count,
        "embedding_dim": embedding_dim,
        "author_table_num_rows": author_table_num_rows,
        "splits": {
            split: dict(metadata["splits"][split]["counts"]) for split in SPLITS
        },
        "validated_data_bytes": validation["total_data_bytes"],
        "build_time_seconds": time.time() - started_at,
    }
    if logger:
        logger.info(
            "Published loader index: path=%s bytes=%s queries=%s build_time=%.2fs",
            output_path,
            f"{total_bytes:,}",
            f"{sum(value['query_count'] for value in stats['splits'].values()):,}",
            stats["build_time_seconds"],
        )
    return stats


def _validate_array(
    *,
    root: Path,
    entry: dict[str, Any],
    expected_dtype: np.dtype[Any],
    expected_length: int,
) -> np.memmap:
    if entry.get("dtype") != expected_dtype.str or entry.get("shape") != [expected_length]:
        raise ValueError(f"Invalid loader-index array declaration: {entry}")
    path = root / entry["path"]
    if not path.is_file() or path.stat().st_size != entry.get("file_size_bytes"):
        raise ValueError(f"Loader-index array size does not match metadata: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype.str != expected_dtype.str or array.shape != (expected_length,):
        raise ValueError(f"Loader-index array shape or dtype is invalid: {path}")
    if array.flags.writeable:
        raise ValueError(f"Loader-index array did not open read-only: {path}")
    return array


def _validate_offsets(offsets: np.ndarray, value_count: int, name: str) -> None:
    if offsets.size == 0 or int(offsets[0]) != 0 or int(offsets[-1]) != value_count:
        raise ValueError(f"{name} does not span its flat value array")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"{name} must be monotonic")


def _validate_arrow_table(root: Path, entry: dict[str, Any], expected_rows: int) -> None:
    if entry.get("row_count") != expected_rows:
        raise ValueError("Arrow table row count does not match the declared index count")
    path = root / entry["path"]
    if not path.is_file() or path.stat().st_size != entry.get("file_size_bytes"):
        raise ValueError(f"Arrow table size does not match metadata: {path}")
    offsets = entry.get("batch_offsets")
    if not isinstance(offsets, list) or not offsets or offsets[0] != 0 or offsets[-1] != expected_rows:
        raise ValueError(f"Invalid Arrow batch offsets for {path}")
    if any(right < left for left, right in zip(offsets, offsets[1:])):
        raise ValueError(f"Arrow batch offsets must be monotonic for {path}")
    with MemoryMappedUtf8Table(path, offsets) as table:
        if table.row_count != expected_rows:
            raise ValueError(f"Arrow table row count is invalid: {path}")
        if table.column_name != entry.get("column"):
            raise ValueError(f"Arrow table column name does not match metadata: {path}")


def validate_loader_index(index_path: Path) -> dict[str, Any]:
    """Validate all files and structural invariants in a loader index."""

    index_path = Path(index_path)
    metadata = load_loader_index_metadata(index_path)
    if set(metadata.get("arrays", {})) != set(GLOBAL_ARRAY_DTYPES):
        raise ValueError("Loader index has unexpected global arrays")
    if set(metadata.get("arrow_tables", {})) != {"post_uris"}:
        raise ValueError("Loader index has unexpected global Arrow tables")
    if set(metadata.get("splits", {})) != set(SPLITS):
        raise ValueError("Loader index must contain all canonical splits in order")
    embedding_meta = metadata.get("embedding", {})
    embedding_shape = embedding_meta.get("shape")
    if (
        embedding_meta.get("dtype") != EMBEDDING_DTYPE.str
        or not isinstance(embedding_shape, list)
        or len(embedding_shape) != 2
    ):
        raise ValueError("Loader-index embedding metadata is invalid")
    embedding_count, embedding_dim = (int(value) for value in embedding_shape)
    embedding_path = (index_path / embedding_meta["path"]).resolve()
    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    if embeddings.dtype.str != EMBEDDING_DTYPE.str or list(embeddings.shape) != embedding_shape:
        raise ValueError("Loader-index embedding metadata does not match embeddings.npy")
    post_created = _validate_array(
        root=index_path,
        entry=metadata["arrays"]["post_created_at_us"],
        expected_dtype=TIMESTAMP_DTYPE,
        expected_length=embedding_count,
    )
    post_authors = _validate_array(
        root=index_path,
        entry=metadata["arrays"]["post_author_idx"],
        expected_dtype=INDEX_DTYPE,
        expected_length=embedding_count,
    )
    del post_created
    author_table_num_rows = int(metadata.get("author_table_num_rows", 0))
    if author_table_num_rows < 2 or (
        post_authors.size
        and (
            int(np.min(post_authors)) < 1
            or int(np.max(post_authors)) >= author_table_num_rows
        )
    ):
        raise ValueError("Post author index is outside the declared author table")
    del post_authors, embeddings
    _validate_arrow_table(
        index_path,
        metadata["arrow_tables"]["post_uris"],
        embedding_count,
    )
    split_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_meta = metadata["splits"][split]
        if set(split_meta.get("arrays", {})) != set(SPLIT_ARRAY_DTYPES):
            raise ValueError(f"Split {split!r} has unexpected numeric arrays")
        if set(split_meta.get("arrow_tables", {})) != {"query_dids"}:
            raise ValueError(f"Split {split!r} has unexpected Arrow tables")
        counts = {name: int(value) for name, value in split_meta.get("counts", {}).items()}
        expected_count_names = {
            "query_count",
            "history_count",
            "positive_count",
            "hour_count",
            "negative_count",
        }
        if set(counts) != expected_count_names or any(value < 0 for value in counts.values()):
            raise ValueError(f"Split {split!r} has invalid count metadata")
        lengths = {
            "query_hours_us": counts["query_count"],
            "history_offsets": counts["query_count"] + 1,
            "history_emb_indices": counts["history_count"],
            "history_like_created_at_us": counts["history_count"],
            "history_prior_like_counts": counts["history_count"],
            "positive_offsets": counts["query_count"] + 1,
            "positive_emb_indices": counts["positive_count"],
            "positive_prior_like_counts": counts["positive_count"],
            "hour_values_us": counts["hour_count"],
            "hour_query_offsets": counts["hour_count"] + 1,
            "negative_offsets": counts["hour_count"] + 1,
            "negative_emb_indices": counts["negative_count"],
            "negative_prior_like_counts": counts["negative_count"],
        }
        arrays = {
            name: _validate_array(
                root=index_path,
                entry=split_meta["arrays"][name],
                expected_dtype=SPLIT_ARRAY_DTYPES[name],
                expected_length=length,
            )
            for name, length in lengths.items()
        }
        _validate_offsets(
            arrays["history_offsets"], counts["history_count"], "history_offsets"
        )
        _validate_offsets(
            arrays["positive_offsets"], counts["positive_count"], "positive_offsets"
        )
        _validate_offsets(
            arrays["hour_query_offsets"], counts["query_count"], "hour_query_offsets"
        )
        _validate_offsets(
            arrays["negative_offsets"], counts["negative_count"], "negative_offsets"
        )
        if counts["query_count"] and np.any(
            arrays["positive_offsets"][1:] == arrays["positive_offsets"][:-1]
        ):
            raise ValueError("Every indexed query must have at least one positive")
        query_hours = arrays["query_hours_us"]
        hour_values = arrays["hour_values_us"]
        if query_hours.size and np.any(query_hours[1:] < query_hours[:-1]):
            raise ValueError(f"Split {split!r} query hours are not sorted")
        if query_hours.size and np.any(query_hours % 3_600_000_000 != 0):
            raise ValueError(f"Split {split!r} query hours are not hour-aligned")
        if hour_values.size and np.any(hour_values[1:] <= hour_values[:-1]):
            raise ValueError(f"Split {split!r} hour values are not unique and sorted")
        for hour_index in range(counts["hour_count"]):
            start = int(arrays["hour_query_offsets"][hour_index])
            end = int(arrays["hour_query_offsets"][hour_index + 1])
            if start == end or not np.all(query_hours[start:end] == hour_values[hour_index]):
                raise ValueError(f"Split {split!r} hour offsets do not align to queries")
        for name in ("history_emb_indices", "positive_emb_indices", "negative_emb_indices"):
            if arrays[name].size and int(np.max(arrays[name])) >= embedding_count:
                raise ValueError(f"Split {split!r} {name} contains an invalid embedding index")
        signed_max = np.iinfo(np.int64).max
        for name in (
            "history_prior_like_counts",
            "positive_prior_like_counts",
            "negative_prior_like_counts",
        ):
            if arrays[name].size and int(np.max(arrays[name])) > signed_max:
                raise ValueError(f"Split {split!r} {name} cannot be converted to torch.int64")
        _validate_arrow_table(
            index_path,
            split_meta["arrow_tables"]["query_dids"],
            counts["query_count"],
        )
        split_counts[split] = counts
        arrays.clear()
    total_data_bytes = _total_declared_bytes(metadata)
    if int(metadata.get("total_data_bytes", -1)) != total_data_bytes:
        raise ValueError("Loader-index byte count does not match declared files")
    return {
        "metadata": metadata,
        "format_version": FORMAT_VERSION,
        "total_data_bytes": total_data_bytes,
        "post_count": embedding_count,
        "embedding_count": embedding_count,
        "embedding_dim": embedding_dim,
        "author_table_num_rows": author_table_num_rows,
        "splits": split_counts,
    }
