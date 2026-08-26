"""Memory-mapped PyTorch datasets for the permanent Stage 7 artifact contract."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from engagement_prediction.data.parquet import find_artifact_path
from engagement_prediction.data.training_index import (
    FORMAT_VERSION,
    MemoryMappedUtf8Table,
    load_index_array,
    load_loader_index_metadata,
)


MICROSECONDS_PER_HOUR = 3_600_000_000
UTC_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

_GLOBAL_ARRAY_NAMES = (
    "post_created_at_us",
    "post_author_idx",
)
_SPLIT_ARRAY_NAMES = (
    "query_hours_us",
    "history_offsets",
    "history_emb_indices",
    "history_like_created_at_us",
    "history_prior_like_counts",
    "positive_offsets",
    "positive_emb_indices",
    "positive_prior_like_counts",
    "hour_values_us",
    "hour_query_offsets",
    "negative_offsets",
    "negative_emb_indices",
    "negative_prior_like_counts",
)


def _datetime_from_epoch_us(value: int) -> datetime:
    """Convert an integer UTC-microsecond timestamp without a float round trip."""

    return UTC_EPOCH + timedelta(microseconds=int(value))


def _close_memmap(value: Optional[np.ndarray]) -> None:
    """Close a NumPy mapping if it owns an mmap handle."""

    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


class HydratedBucketedEngagementDataset(Dataset):
    """One compact row reference per Stage 7 user-hour query.

    Large feature columns stay in read-only NumPy and Arrow mappings. The
    dataset object itself contains only paths and small metadata, so DataLoader
    workers do not copy millions of Python dictionaries through fork
    copy-on-write or spawn pickling.
    """

    def __init__(
        self,
        stage7_path: Path,
        *,
        split: str,
        max_history_len: int,
        bst_additional_batch_negatives: Optional[int],
        seed: int,
        logger: Optional[logging.Logger],
    ):
        """Read the loader-index header without loading feature rows."""

        if max_history_len <= 0:
            raise ValueError("max_history_len must be positive")
        if bst_additional_batch_negatives is not None and bst_additional_batch_negatives <= 0:
            raise ValueError("bst_additional_batch_negatives must be positive when provided")

        stage7_path = Path(stage7_path)
        bundle_path = (
            stage7_path
            if stage7_path.name.startswith("hydrated_training_data_")
            else find_artifact_path(stage7_path, "hydrated_training_data_")
        )
        loader_index_path = bundle_path / "loader_index"
        if not loader_index_path.is_dir():
            raise ValueError(
                "Stage 7 loader_index is missing; regenerate Stage 7 with the "
                "current dataset-hydration implementation"
            )

        metadata = load_loader_index_metadata(loader_index_path)
        format_version = int(metadata.get("format_version", -1))
        if format_version != FORMAT_VERSION:
            raise ValueError(
                "Unsupported Stage 7 loader_index format version "
                f"{format_version}; regenerate Stage 7"
            )

        self.split = str(split)
        try:
            split_metadata = metadata["splits"][self.split]
            counts = split_metadata["counts"]
            embedding_metadata = metadata["embedding"]
            embedding_shape = tuple(int(value) for value in embedding_metadata["shape"])
            post_uri_metadata = metadata["arrow_tables"]["post_uris"]
            query_did_metadata = split_metadata["arrow_tables"]["query_dids"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Stage 7 loader_index does not contain a valid '{self.split}' split"
            ) from exc
        if len(embedding_shape) != 2 or embedding_shape[1] <= 0:
            raise ValueError("Stage 7 loader_index records an invalid embedding shape")

        self.bundle_path = bundle_path
        self.loader_index_path = loader_index_path
        self.max_history_len = int(max_history_len)
        self.bst_additional_batch_negatives = (
            int(bst_additional_batch_negatives)
            if bst_additional_batch_negatives is not None
            else None
        )
        self.seed = int(seed)
        self.embed_dim = int(embedding_shape[1])
        self.embedding_count = int(embedding_shape[0])
        self.author_table_num_rows = int(metadata["author_table_num_rows"])
        self.query_count = int(counts["query_count"])
        self.history_count = int(counts["history_count"])
        self.positive_count = int(counts["positive_count"])
        self.hour_count = int(counts["hour_count"])
        self.negative_count = int(counts["negative_count"])

        self._embeddings_path = loader_index_path / str(embedding_metadata["path"])
        self._post_uri_path = loader_index_path / str(post_uri_metadata["path"])
        self._post_uri_batch_offsets = tuple(
            int(value) for value in post_uri_metadata["batch_offsets"]
        )
        self._query_did_path = loader_index_path / str(query_did_metadata["path"])
        self._query_did_batch_offsets = tuple(
            int(value) for value in query_did_metadata["batch_offsets"]
        )

        # These handles are process-local. They are deliberately absent from
        # serialized state and are reopened if a forked object changes PID.
        self._owner_pid: Optional[int] = None
        self._embeddings: Optional[np.ndarray] = None
        self._arrays: Optional[dict[str, np.memmap]] = None
        self._post_uris: Optional[MemoryMappedUtf8Table] = None
        self._query_dids: Optional[MemoryMappedUtf8Table] = None

        if logger:
            logger.info(
                "HydratedBucketedEngagementDataset('%s'): queries=%s hours=%s negatives=%s",
                self.split,
                f"{self.query_count:,}",
                f"{self.hour_count:,}",
                f"{self.negative_count:,}",
            )

    def __getstate__(self) -> dict[str, Any]:
        """Serialize only paths and compact metadata for spawned workers."""

        state = self.__dict__.copy()
        state["_owner_pid"] = None
        state["_embeddings"] = None
        state["_arrays"] = None
        state["_post_uris"] = None
        state["_query_dids"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a dataset with all file mappings still unopened."""

        self.__dict__.update(state)
        self._owner_pid = None
        self._embeddings = None
        self._arrays = None
        self._post_uris = None
        self._query_dids = None

    def _close_mappings(self) -> None:
        """Release all mappings owned by the current process."""

        _close_memmap(self._embeddings)
        for array in (self._arrays or {}).values():
            _close_memmap(array)
        if self._post_uris is not None:
            self._post_uris.close()
        if self._query_dids is not None:
            self._query_dids.close()
        self._owner_pid = None
        self._embeddings = None
        self._arrays = None
        self._post_uris = None
        self._query_dids = None

    def close(self) -> None:
        """Explicitly release process-local memory mappings."""

        self._close_mappings()

    def _ensure_open(self) -> None:
        """Lazily open read-only numeric mappings, reopening after a PID change.

        Arrow identifier tables are opened separately because model training
        consumes only tensors. Keeping that path independent avoids mapping the
        URI and DID files in workers that never decode an identifier.
        """

        current_pid = os.getpid()
        if self._owner_pid == current_pid:
            return
        if self._owner_pid is not None:
            self._close_mappings()

        embeddings = np.load(self._embeddings_path, mmap_mode="r")
        if (
            not isinstance(embeddings, np.memmap)
            or embeddings.dtype != np.dtype("<f4")
            or embeddings.shape != (self.embedding_count, self.embed_dim)
        ):
            _close_memmap(embeddings)
            raise ValueError("Stage 7 embeddings do not match the loader_index metadata")

        arrays: dict[str, np.memmap] = {}
        try:
            for name in _GLOBAL_ARRAY_NAMES:
                arrays[name] = load_index_array(self.loader_index_path, name)
            for name in _SPLIT_ARRAY_NAMES:
                arrays[name] = load_index_array(
                    self.loader_index_path,
                    name,
                    split=self.split,
                )
        except Exception:
            _close_memmap(embeddings)
            for array in arrays.values():
                _close_memmap(array)
            raise

        self._embeddings = embeddings
        self._arrays = arrays
        self._owner_pid = current_pid

    def _ensure_identifier_tables_open(self) -> None:
        """Open the Arrow URI and DID tables only for the full batch contract."""

        self._ensure_open()
        if self._post_uris is not None and self._query_dids is not None:
            return

        post_uris: Optional[MemoryMappedUtf8Table] = None
        query_dids: Optional[MemoryMappedUtf8Table] = None
        try:
            post_uris = MemoryMappedUtf8Table(
                self._post_uri_path,
                batch_offsets=self._post_uri_batch_offsets,
            )
            query_dids = MemoryMappedUtf8Table(
                self._query_did_path,
                batch_offsets=self._query_did_batch_offsets,
            )
        except Exception:
            if post_uris is not None:
                post_uris.close()
            if query_dids is not None:
                query_dids.close()
            raise

        self._post_uris = post_uris
        self._query_dids = query_dids

    @property
    def embeddings(self) -> np.ndarray:
        """Return the process-local read-only embedding memmap."""

        self._ensure_open()
        assert self._embeddings is not None
        return self._embeddings

    def _array(self, name: str) -> np.memmap:
        """Return one process-local loader-index array."""

        self._ensure_open()
        assert self._arrays is not None
        return self._arrays[name]

    def __len__(self) -> int:
        """Return the number of user-hour queries in this split."""

        return self.query_count

    def __getitem__(self, idx: Any) -> tuple[int, int]:
        """Return only the row and candidate-resampling epoch references."""

        if isinstance(idx, tuple):
            row_idx, epoch = idx
        else:
            row_idx, epoch = idx, 0
        row_idx = int(row_idx)
        if row_idx < 0 or row_idx >= self.query_count:
            raise IndexError(row_idx)
        return row_idx, int(epoch)

    def _history_tensors(
        self,
        row_indices: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather and pad all ragged histories with one vectorized operation."""

        batch_size = int(row_indices.size)
        history_offsets = self._array("history_offsets")
        starts = np.asarray(history_offsets[row_indices], dtype=np.int64)
        ends = np.asarray(history_offsets[row_indices + 1], dtype=np.int64)
        lengths = np.minimum(ends - starts, self.max_history_len)

        padded_embeddings = np.zeros(
            (batch_size, self.max_history_len, self.embed_dim),
            dtype=np.float32,
        )
        mask = np.zeros((batch_size, self.max_history_len), dtype=bool)
        author_indices = np.zeros((batch_size, self.max_history_len), dtype=np.int64)
        time_deltas = np.zeros((batch_size, self.max_history_len), dtype=np.float32)
        prior_counts = np.zeros((batch_size, self.max_history_len), dtype=np.int64)

        total_items = int(lengths.sum())
        if total_items:
            batch_rows = np.repeat(np.arange(batch_size, dtype=np.int64), lengths)
            repeated_starts = np.repeat(starts, lengths)
            segment_starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
            history_positions = np.arange(total_items, dtype=np.int64) - segment_starts
            source_positions = repeated_starts + history_positions

            history_emb_indices = np.asarray(
                self._array("history_emb_indices")[source_positions], dtype=np.int64
            )
            padded_embeddings[batch_rows, history_positions] = self.embeddings[
                history_emb_indices
            ]
            mask[batch_rows, history_positions] = True
            author_indices[batch_rows, history_positions] = np.asarray(
                self._array("post_author_idx")[history_emb_indices], dtype=np.int64
            )
            liked_at_us = np.asarray(
                self._array("history_like_created_at_us")[source_positions],
                dtype=np.int64,
            )
            query_hours_us = np.asarray(
                self._array("query_hours_us")[row_indices], dtype=np.int64
            )
            elapsed_us = np.maximum(query_hours_us[batch_rows] - liked_at_us, 0)
            time_deltas[batch_rows, history_positions] = (
                elapsed_us.astype(np.float64) / MICROSECONDS_PER_HOUR
            ).astype(np.float32)
            prior_counts[batch_rows, history_positions] = np.asarray(
                self._array("history_prior_like_counts")[source_positions],
                dtype=np.int64,
            )

        return (
            torch.from_numpy(padded_embeddings),
            torch.from_numpy(mask),
            torch.from_numpy(author_indices),
            torch.from_numpy(time_deltas),
            torch.from_numpy(prior_counts),
        )

    def _negative_positions(
        self,
        *,
        query_hour_us: int,
        row_indices: np.ndarray,
        epoch: int,
    ) -> np.ndarray:
        """Select flattened negative positions for one shared query hour."""

        hour_values = self._array("hour_values_us")
        hour_idx = int(np.searchsorted(hour_values, np.int64(query_hour_us)))
        if hour_idx >= self.hour_count or int(hour_values[hour_idx]) != query_hour_us:
            return np.empty(0, dtype=np.int64)
        negative_offsets = self._array("negative_offsets")
        start = int(negative_offsets[hour_idx])
        end = int(negative_offsets[hour_idx + 1])
        count = end - start
        cap = self.bst_additional_batch_negatives
        if cap is None or count <= cap:
            return np.arange(start, end, dtype=np.int64)

        sorted_rows = np.sort(row_indices)
        row_seed = sum(
            (position + 1) * (int(row_idx) + 1)
            for position, row_idx in enumerate(sorted_rows)
        )
        rng = np.random.default_rng(
            self.seed + int(epoch) * max(self.query_count, 1) + row_seed
        )
        selected = np.sort(rng.choice(count, size=cap, replace=False))
        return np.asarray(selected + start, dtype=np.int64)

    def _collate_batch(
        self,
        items: Sequence[tuple[int, int]],
        *,
        include_metadata: bool,
    ) -> dict[str, Any]:
        """Build one hour-homogeneous listwise batch and its label matrix."""

        if not items:
            raise ValueError("HydratedBucketedEngagementDataset received an empty batch")
        row_indices = np.asarray([int(item[0]) for item in items], dtype=np.int64)
        epochs = {int(item[1]) for item in items}
        if len(epochs) != 1:
            raise ValueError("Bucketed batches must contain one sampling epoch")
        epoch = next(iter(epochs))
        query_hours = np.asarray(
            self._array("query_hours_us")[row_indices], dtype=np.int64
        )
        query_hour_us = int(query_hours[0])
        if np.any(query_hours != query_hour_us):
            raise ValueError("Bucketed batches must contain one query hour")

        history_tensors = self._history_tensors(row_indices)
        positive_offsets = self._array("positive_offsets")
        positive_emb_indices = self._array("positive_emb_indices")
        positive_prior_counts = self._array("positive_prior_like_counts")

        candidate_emb_indices: list[int] = []
        candidate_prior_counts: list[int] = []
        candidate_index: dict[int, int] = {}
        positive_indices_by_row: list[list[int]] = []

        def add_candidate(emb_idx: int, prior_like_count: int) -> int:
            candidate_idx = candidate_index.get(emb_idx)
            if candidate_idx is None:
                candidate_idx = len(candidate_emb_indices)
                candidate_index[emb_idx] = candidate_idx
                candidate_emb_indices.append(emb_idx)
                candidate_prior_counts.append(prior_like_count)
            return candidate_idx

        # Positives enter first, preserving query order and each query's URI
        # order. Candidate identity is the dense emb_idx rather than a string.
        for row_idx in row_indices:
            start = int(positive_offsets[row_idx])
            end = int(positive_offsets[row_idx + 1])
            row_positive_indices: list[int] = []
            for source_idx in range(start, end):
                emb_idx = int(positive_emb_indices[source_idx])
                candidate_idx = add_candidate(
                    emb_idx, int(positive_prior_counts[source_idx])
                )
                row_positive_indices.append(candidate_idx)
            positive_indices_by_row.append(row_positive_indices)

        negative_positions = self._negative_positions(
            query_hour_us=query_hour_us,
            row_indices=row_indices,
            epoch=epoch,
        )
        negative_emb_indices = self._array("negative_emb_indices")
        negative_prior_counts = self._array("negative_prior_like_counts")
        for source_idx in negative_positions:
            add_candidate(
                int(negative_emb_indices[source_idx]),
                int(negative_prior_counts[source_idx]),
            )

        candidate_emb_array = np.asarray(candidate_emb_indices, dtype=np.int64)
        candidate_embeddings = np.array(
            self.embeddings[candidate_emb_array], dtype=np.float32, copy=True
        )
        labels = torch.zeros(
            (len(row_indices), len(candidate_emb_indices)), dtype=torch.float32
        )
        for batch_row, row_positive_indices in enumerate(positive_indices_by_row):
            if row_positive_indices:
                labels[batch_row, row_positive_indices] = 1.0

        batch = {
            "history_embeddings": history_tensors[0],
            "history_mask": history_tensors[1],
            "history_author_indices": history_tensors[2],
            "history_time_deltas_hours": history_tensors[3],
            "history_prior_cumulative_likes": history_tensors[4],
            "candidate_post_embeddings": torch.from_numpy(candidate_embeddings),
            "candidate_post_author_idx": torch.from_numpy(
                np.asarray(
                    self._array("post_author_idx")[candidate_emb_array],
                    dtype=np.int64,
                ).copy()
            ),
            "candidate_prior_cumulative_likes": torch.tensor(
                candidate_prior_counts, dtype=torch.float32
            ),
            "label_matrix": labels,
        }
        if not include_metadata:
            return batch

        post_created_at_us = np.asarray(
            self._array("post_created_at_us")[candidate_emb_array], dtype=np.int64
        )
        creation_hour_us = (
            post_created_at_us // MICROSECONDS_PER_HOUR
        ) * MICROSECONDS_PER_HOUR
        candidate_ages = np.maximum(query_hour_us - creation_hour_us, 0).astype(
            np.float64
        ) / MICROSECONDS_PER_HOUR

        self._ensure_identifier_tables_open()
        assert self._query_dids is not None
        assert self._post_uris is not None
        query_hour = _datetime_from_epoch_us(query_hour_us)
        batch.update({
            "candidate_post_age_hours": torch.from_numpy(
                candidate_ages.astype(np.float32)
            ),
            "user_id": self._query_dids.take(row_indices),
            "candidate_post_id": self._post_uris.take(candidate_emb_array),
            "query_hour": query_hour,
            "bucket": query_hour,
        })
        return batch

    def collate_batch(self, items: Sequence[tuple[int, int]]) -> dict[str, Any]:
        """Build the full native batch, including identifiers and auxiliary age."""

        return self._collate_batch(items, include_metadata=True)

    def collate_tensor_batch(
        self,
        items: Sequence[tuple[int, int]],
    ) -> dict[str, torch.Tensor]:
        """Build the tensor-only batch consumed by canonical model training."""

        return self._collate_batch(items, include_metadata=False)


class HydratedBucketedBatchSampler(Sampler[list[Any]]):
    """Yield compact query-index batches containing one query hour."""

    def __init__(
        self,
        dataset: HydratedBucketedEngagementDataset,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
        resample_candidates_each_epoch: bool,
    ):
        """Configure deterministic hour-aware batching and epoch shuffling."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.resample_candidates_each_epoch = bool(resample_candidates_each_epoch)
        self._epoch = 0
        self._evaluation_mode = False

    def set_evaluation_mode(self, enabled: bool = True) -> None:
        """Use stable row order and epoch-zero negatives for final evaluation."""

        self._evaluation_mode = bool(enabled)

    def __iter__(self) -> Iterator[list[Any]]:
        """Build one compact row-order vector and yield homogeneous batches."""

        evaluation_mode = self._evaluation_mode
        epoch = 0 if evaluation_mode else self._epoch
        if not evaluation_mode:
            self._epoch += 1
        shuffle = self.shuffle and not evaluation_mode
        rng = np.random.default_rng(self.seed + epoch)
        hour_offsets = self.dataset._array("hour_query_offsets")
        index_dtype = (
            np.uint32
            if self.dataset.query_count <= np.iinfo(np.uint32).max
            else np.uint64
        )
        query_order = np.empty(self.dataset.query_count, dtype=index_dtype)
        descriptors = np.empty((len(self), 2), dtype=np.uint64)
        descriptor_idx = 0
        hour_order = np.arange(self.dataset.hour_count, dtype=np.uint32)
        if shuffle:
            # Preserve the legacy RNG sequence: shuffle hours, then the rows
            # within each hour, and finally the completed batch descriptors.
            rng.shuffle(hour_order)

        for hour_idx_value in hour_order:
            hour_idx = int(hour_idx_value)
            start = int(hour_offsets[hour_idx])
            end = int(hour_offsets[hour_idx + 1])
            indices = np.arange(start, end, dtype=index_dtype)
            if shuffle:
                rng.shuffle(indices)
            query_order[start:end] = indices
            for batch_start in range(start, end, self.batch_size):
                batch_end = min(batch_start + self.batch_size, end)
                if batch_end - batch_start < self.batch_size and self.drop_last:
                    continue
                descriptors[descriptor_idx] = (batch_start, batch_end)
                descriptor_idx += 1

        if descriptor_idx != descriptors.shape[0]:
            raise RuntimeError("Loader-index hour offsets produced an incorrect batch count")
        if shuffle:
            rng.shuffle(descriptors)

        candidate_epoch = (
            epoch
            if self.resample_candidates_each_epoch and not evaluation_mode
            else 0
        )
        for start, end in descriptors:
            row_indices = query_order[int(start):int(end)]
            yield [(int(row_idx), int(candidate_epoch)) for row_idx in row_indices]

    def __len__(self) -> int:
        """Return the exact batch count implied by compact hour offsets."""

        hour_offsets = self.dataset._array("hour_query_offsets")
        lengths = np.diff(hour_offsets.astype(np.uint64, copy=False))
        if self.drop_last:
            counts = lengths // self.batch_size
        else:
            counts = (lengths + self.batch_size - 1) // self.batch_size
        return int(counts.sum(dtype=np.uint64))


def create_hydrated_data_loader(
    dataset: HydratedBucketedEngagementDataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    seed: int,
    resample_candidates_each_epoch: bool,
    tensor_only: bool = False,
) -> DataLoader:
    """Wrap the native Stage 7 dataset in its hour-aware PyTorch DataLoader.

    The default retains the complete native batch contract. Canonical model
    training can request a tensor-only batch to skip auxiliary candidate ages
    and Arrow identifier decoding that its hot loop does not consume.
    """

    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    worker_options: dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        worker_options.update(
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor),
        )
    return DataLoader(
        dataset,
        batch_sampler=HydratedBucketedBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            resample_candidates_each_epoch=resample_candidates_each_epoch,
        ),
        collate_fn=(
            dataset.collate_tensor_batch if tensor_only else dataset.collate_batch
        ),
        **worker_options,
    )
