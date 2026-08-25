"""Native PyTorch datasets for the permanent Stage 7 artifact contract."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from engagement_prediction.data import dataset_hydration
from engagement_prediction.data.parquet import find_artifact_path, scan_parquet_artifact
from shared.input_data_helpers import (
    get_padded_author_indices,
    get_padded_embedding_history_and_mask,
    get_padded_history_time_deltas,
    get_padded_prior_cumulative_likes,
)


def _as_epoch_us(value: datetime) -> int:
    """Convert a timezone-aware datetime to integer UTC microseconds."""

    normalized = value.astimezone(timezone.utc).replace(tzinfo=None)
    return int(np.datetime64(normalized, "us").astype(np.int64))


def _hours_between(later: datetime, earlier: datetime) -> float:
    """Return a nonnegative elapsed-hour feature at microsecond precision."""

    return max((_as_epoch_us(later) - _as_epoch_us(earlier)) / 3_600_000_000.0, 0.0)


def _creation_age_hours(query_hour: datetime, created_at: datetime) -> float:
    """Measure candidate age from its UTC creation-hour bucket."""

    creation_hour = created_at.replace(minute=0, second=0, microsecond=0)
    return _hours_between(query_hour, creation_hour)


class HydratedBucketedEngagementDataset(Dataset):
    """One row per Stage 7 user-hour query with a shared same-hour candidate pool."""

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
        """Load one split's relational artifacts and index them by query/hour."""

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
        self.bundle_path = bundle_path
        self.split = str(split)
        self.max_history_len = int(max_history_len)
        self.bst_additional_batch_negatives = (
            int(bst_additional_batch_negatives)
            if bst_additional_batch_negatives is not None
            else None
        )
        self.seed = int(seed)

        embeddings_path = bundle_path / "embeddings.npy"
        self.embeddings: np.ndarray = np.load(embeddings_path, mmap_mode="r")
        if self.embeddings.dtype != np.float32 or self.embeddings.ndim != 2:
            raise ValueError(
                f"Stage 7 embeddings must be a Float32 matrix, found "
                f"{self.embeddings.dtype} {self.embeddings.shape}"
            )
        self.embed_dim = int(self.embeddings.shape[1])

        queries_lf = scan_parquet_artifact(bundle_path / "queries").filter(
            pl.col("split") == self.split
        )
        queries_df = queries_lf.collect(engine="streaming").sort(["query_hour", "did"])
        query_keys = queries_df.select("did", "query_hour")
        positives_df = (
            scan_parquet_artifact(bundle_path / "query_positives")
            .join(query_keys.lazy(), on=["did", "query_hour"], how="semi")
            .collect(engine="streaming")
            .sort(["query_hour", "did", "subject_uri"])
        )
        histories_df = (
            scan_parquet_artifact(bundle_path / "query_histories")
            .join(query_keys.lazy(), on=["did", "query_hour"], how="semi")
            .collect(engine="streaming")
        )
        query_hours = queries_df.select("query_hour").unique()
        negatives_df = (
            scan_parquet_artifact(bundle_path / "hourly_negative_candidates")
            .join(query_hours.lazy(), on="query_hour", how="semi")
            .collect(engine="streaming")
            .sort(["query_hour", "subject_uri"])
        )
        authors_df = scan_parquet_artifact(bundle_path / "authors").select(
            "author_idx"
        ).collect(engine="streaming")
        self.author_table_num_rows = max(
            int(authors_df.get_column("author_idx").max() or dataset_hydration.AUTHOR_UNK_IDX) + 1,
            dataset_hydration.AUTHOR_UNK_IDX + 1,
        )

        # Convert columnar tables into query-keyed structures once so batch
        # collation does not perform joins during training.
        history_by_key = {
            (row["did"], row["query_hour"]): row
            for row in histories_df.iter_rows(named=True)
        }
        positives_by_key: dict[tuple[str, datetime], list[dict[str, Any]]] = {}
        for row in positives_df.iter_rows(named=True):
            positives_by_key.setdefault((row["did"], row["query_hour"]), []).append(row)
        self.rows: list[dict[str, Any]] = []
        for query in queries_df.iter_rows(named=True):
            key = (query["did"], query["query_hour"])
            history = history_by_key.get(key)
            positives = positives_by_key.get(key, [])
            if history is None:
                raise ValueError(f"Stage 7 query is missing its history row: {key}")
            if len(positives) != int(query["positive_count"]):
                raise ValueError(f"Stage 7 query has an incorrect positive_count: {key}")
            self.rows.append({**query, **history, "positives": positives})

        self.negatives_by_hour: dict[datetime, list[dict[str, Any]]] = {}
        for row in negatives_df.iter_rows(named=True):
            self.negatives_by_hour.setdefault(row["query_hour"], []).append(row)
        self.row_indices_by_bucket: dict[datetime, list[int]] = {}
        for row_idx, row in enumerate(self.rows):
            self.row_indices_by_bucket.setdefault(row["query_hour"], []).append(row_idx)
        if logger:
            logger.info(
                "HydratedBucketedEngagementDataset('%s'): queries=%s hours=%s negatives=%s",
                self.split,
                f"{len(self.rows):,}",
                f"{len(self.row_indices_by_bucket):,}",
                f"{negatives_df.height:,}",
            )

    def __len__(self) -> int:
        """Return the number of user-hour queries in this split."""

        return len(self.rows)

    def __getitem__(self, idx: Any) -> dict[str, Any]:
        """Return a lightweight query reference; collation fetches features."""

        if isinstance(idx, tuple):
            row_idx, epoch = idx
        else:
            row_idx, epoch = idx, 0
        row_idx = int(row_idx)
        return {
            "row_idx": row_idx,
            "bucket": self.rows[row_idx]["query_hour"],
            "user_id": self.rows[row_idx]["did"],
            "epoch": int(epoch),
        }

    def _history_tensors(
        self,
        row: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch, derive, and pad the aligned features for one user history."""

        emb_indices = np.asarray(row["history_emb_indices"], dtype=np.int64)
        embeddings = self.embeddings[emb_indices]
        padded_embeddings, mask = get_padded_embedding_history_and_mask(
            embeddings,
            self.max_history_len,
            self.embed_dim,
        )
        author_indices = get_padded_author_indices(
            row["history_author_indices"],
            self.max_history_len,
        )
        ages = [
            _hours_between(row["query_hour"], liked_at)
            for liked_at in row["history_like_created_ats"]
        ]
        time_deltas = get_padded_history_time_deltas(ages, self.max_history_len)
        prior_counts = get_padded_prior_cumulative_likes(
            row["history_prior_like_counts"],
            self.max_history_len,
        )
        return (
            torch.from_numpy(padded_embeddings),
            torch.from_numpy(mask),
            torch.from_numpy(author_indices),
            torch.from_numpy(time_deltas),
            torch.from_numpy(prior_counts),
        )

    def _negative_candidates(
        self,
        *,
        query_hour: datetime,
        row_indices: list[int],
        epoch: int,
    ) -> list[dict[str, Any]]:
        """Return the shared hour pool, optionally resampled for this epoch."""

        negatives = list(self.negatives_by_hour.get(query_hour, []))
        if self.bst_additional_batch_negatives is None or len(negatives) <= self.bst_additional_batch_negatives:
            return negatives
        sorted_rows = sorted(row_indices)
        row_seed = sum((position + 1) * (row_idx + 1) for position, row_idx in enumerate(sorted_rows))
        rng = np.random.default_rng(
            self.seed + epoch * max(len(self.rows), 1) + row_seed
        )
        selected = sorted(
            rng.choice(
                len(negatives),
                size=self.bst_additional_batch_negatives,
                replace=False,
            ).tolist()
        )
        return [negatives[index] for index in selected]

    def collate_batch(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Build one hour-homogeneous listwise batch and its label matrix."""

        if not items:
            raise ValueError("HydratedBucketedEngagementDataset received an empty batch")
        row_indices = [int(item["row_idx"]) for item in items]
        epochs = {int(item.get("epoch", 0)) for item in items}
        if len(epochs) != 1:
            raise ValueError("Bucketed batches must contain one sampling epoch")
        epoch = next(iter(epochs))
        query_hour = self.rows[row_indices[0]]["query_hour"]
        if any(self.rows[row_idx]["query_hour"] != query_hour for row_idx in row_indices):
            raise ValueError("Bucketed batches must contain one query hour")

        history_tensors = [self._history_tensors(self.rows[row_idx]) for row_idx in row_indices]
        candidate_rows: list[dict[str, Any]] = []
        candidate_index: dict[str, int] = {}

        def add_candidate(row: dict[str, Any]) -> None:
            uri = row["subject_uri"]
            if uri in candidate_index:
                return
            candidate_index[uri] = len(candidate_rows)
            candidate_rows.append(row)

        # Positives enter first. If the same URI is also in the shared negative
        # pool, it remains one candidate and is labeled per user below.
        for row_idx in row_indices:
            for positive in self.rows[row_idx]["positives"]:
                add_candidate(positive)
        for negative in self._negative_candidates(
            query_hour=query_hour,
            row_indices=row_indices,
            epoch=epoch,
        ):
            add_candidate(negative)

        candidate_emb_indices = np.asarray(
            [row["emb_idx"] for row in candidate_rows],
            dtype=np.int64,
        )
        candidate_embeddings = np.asarray(
            self.embeddings[candidate_emb_indices],
            dtype=np.float32,
        )
        labels = torch.zeros((len(row_indices), len(candidate_rows)), dtype=torch.float32)
        for batch_row, row_idx in enumerate(row_indices):
            for positive in self.rows[row_idx]["positives"]:
                labels[batch_row, candidate_index[positive["subject_uri"]]] = 1.0

        return {
            "history_embeddings": torch.stack([values[0] for values in history_tensors]),
            "history_mask": torch.stack([values[1] for values in history_tensors]),
            "history_author_indices": torch.stack([values[2] for values in history_tensors]),
            "history_time_deltas_hours": torch.stack([values[3] for values in history_tensors]),
            "history_prior_cumulative_likes": torch.stack([values[4] for values in history_tensors]),
            "candidate_post_embeddings": torch.from_numpy(candidate_embeddings),
            "candidate_post_author_idx": torch.tensor(
                [row["author_idx"] for row in candidate_rows],
                dtype=torch.long,
            ),
            "candidate_prior_cumulative_likes": torch.tensor(
                [row["prior_like_count"] for row in candidate_rows],
                dtype=torch.float32,
            ),
            "candidate_post_age_hours": torch.tensor(
                [
                    _creation_age_hours(query_hour, row["post_created_at"])
                    for row in candidate_rows
                ],
                dtype=torch.float32,
            ),
            "label_matrix": labels,
            "user_id": [self.rows[row_idx]["did"] for row_idx in row_indices],
            "candidate_post_id": [row["subject_uri"] for row in candidate_rows],
            "query_hour": query_hour,
            "bucket": query_hour,
        }


class HydratedBucketedBatchSampler(Sampler[list[int]]):
    """Yield query-index batches containing exactly one query hour."""

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
        """Configure deterministic query-hour batching and epoch shuffling."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.resample_candidates_each_epoch = bool(resample_candidates_each_epoch)
        self._epoch = 0

    def __iter__(self) -> Iterator[list[Any]]:
        """Yield batches that never mix scoring hours."""

        epoch = self._epoch
        self._epoch += 1
        rng = np.random.default_rng(self.seed + epoch)
        hours = list(self.dataset.row_indices_by_bucket)
        if self.shuffle:
            rng.shuffle(hours)
        batches: list[list[int]] = []
        for hour in hours:
            indices = list(self.dataset.row_indices_by_bucket[hour])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        for batch in batches:
            if self.resample_candidates_each_epoch:
                yield [(int(row_idx), int(epoch)) for row_idx in batch]
            else:
                yield batch

    def __len__(self) -> int:
        """Return the exact batch count implied by all hour buckets."""

        total = 0
        for indices in self.dataset.row_indices_by_bucket.values():
            total += len(indices) // self.batch_size
            if len(indices) % self.batch_size and not self.drop_last:
                total += 1
        return total


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
) -> DataLoader:
    """Wrap the native Stage 7 dataset in its hour-aware PyTorch DataLoader."""
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
        collate_fn=dataset.collate_batch,
        **worker_options,
    )
