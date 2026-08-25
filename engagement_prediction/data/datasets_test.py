from datetime import datetime, timezone

import numpy as np
import polars as pl
import torch

from engagement_prediction.data import dataset_hydration
from engagement_prediction.data.datasets import (
    HydratedBucketedBatchSampler,
    HydratedBucketedEngagementDataset,
    create_hydrated_data_loader,
)


UTC = timezone.utc


def _write_dataset(path, name, df):
    dataset_path = path / name
    dataset_path.mkdir()
    df.write_parquet(dataset_path / "part-00000.parquet")


def _bundle(tmp_path):
    bundle = tmp_path / "hydrated_training_data_test"
    bundle.mkdir()
    mmap = np.lib.format.open_memmap(
        bundle / "embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(4, 2),
    )
    mmap[:] = np.array([[1, 0], [0, 1], [2, 0], [0, 2]], dtype=np.float32)
    mmap.flush()
    del mmap
    hour = datetime(2026, 1, 1, 12, tzinfo=UTC)
    created = datetime(2026, 1, 1, 11, 30, tzinfo=UTC)
    queries = pl.DataFrame({
        "did": ["u1", "u2"],
        "query_hour": [hour, hour],
        "user_cohort": ["seen", "seen"],
        "split": ["train", "train"],
        "positive_count": [1, 1],
    }, schema=dataset_hydration.QUERY_SCHEMA)
    positives = pl.DataFrame({
        "did": ["u1", "u2"],
        "query_hour": [hour, hour],
        "subject_uri": ["p1", "p2"],
        "like_created_at": [hour, hour],
        "emb_idx": [0, 1],
        "post_created_at": [created, created],
        "author_idx": [2, 3],
        "prior_like_count": [4, 5],
    }, schema=dataset_hydration.QUERY_POSITIVE_SCHEMA)
    histories = pl.DataFrame({
        "did": ["u1", "u2"],
        "query_hour": [hour, hour],
        "history_subject_uris": [["h1"], []],
        "history_like_created_ats": [[datetime(2026, 1, 1, 10, 30, tzinfo=UTC)], []],
        "history_emb_indices": [[2], []],
        "history_author_indices": [[4], []],
        "history_prior_like_counts": [[3], []],
    }, schema=dataset_hydration.QUERY_HISTORY_SCHEMA)
    negatives = pl.DataFrame({
        "query_hour": [hour, hour],
        "subject_uri": ["p1", "n1"],
        "selection_source": ["random", "popular"],
        "emb_idx": [0, 3],
        "post_created_at": [created, created],
        "author_idx": [2, 5],
        "prior_like_count": [4, 10],
    }, schema=dataset_hydration.HOURLY_NEGATIVE_SCHEMA)
    _write_dataset(bundle, "queries", queries)
    _write_dataset(bundle, "query_positives", positives)
    _write_dataset(bundle, "query_histories", histories)
    _write_dataset(bundle, "hourly_negative_candidates", negatives)
    _write_dataset(bundle, "authors", pl.DataFrame({"author_idx": [2, 3, 4, 5]}))
    return bundle


def test_native_dataset_builds_multi_user_hour_batch_without_duplicate_candidates(tmp_path):
    dataset = HydratedBucketedEngagementDataset(
        _bundle(tmp_path),
        split="train",
        max_history_len=2,
        bst_additional_batch_negatives=None,
        seed=7,
        logger=None,
    )

    assert dataset.embeddings.flags.writeable is False
    batch = dataset.collate_batch([dataset[0], dataset[1]])

    assert batch["candidate_post_id"] == ["p1", "p2", "n1"]
    assert batch["label_matrix"].tolist() == [[1, 0, 0], [0, 1, 0]]
    assert batch["history_mask"].tolist() == [[True, False], [False, False]]
    assert batch["history_time_deltas_hours"].tolist() == [[1.5, 0.0], [0.0, 0.0]]
    assert batch["history_prior_cumulative_likes"].tolist() == [[3, 0], [0, 0]]
    assert batch["candidate_post_age_hours"].tolist() == [1.0, 1.0, 1.0]
    assert torch.equal(
        batch["candidate_post_embeddings"],
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]]),
    )


def test_native_sampler_keeps_each_batch_in_one_hour_and_is_deterministic(tmp_path):
    dataset = HydratedBucketedEngagementDataset(
        _bundle(tmp_path),
        split="train",
        max_history_len=2,
        bst_additional_batch_negatives=None,
        seed=7,
        logger=None,
    )
    first = list(HydratedBucketedBatchSampler(
        dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        seed=11,
        resample_candidates_each_epoch=False,
    ))
    second = list(HydratedBucketedBatchSampler(
        dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        seed=11,
        resample_candidates_each_epoch=False,
    ))

    assert first == second
    assert sorted(index for batch in first for index in batch) == [0, 1]


def test_native_pytorch_dataloader_emits_the_permanent_feature_contract(tmp_path):
    dataset = HydratedBucketedEngagementDataset(
        _bundle(tmp_path),
        split="train",
        max_history_len=2,
        bst_additional_batch_negatives=None,
        seed=7,
        logger=None,
    )
    loader = create_hydrated_data_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=1,
        seed=7,
        resample_candidates_each_epoch=False,
    )

    batch = next(iter(loader))

    assert batch["history_embeddings"].shape == (2, 2, 2)
    assert batch["candidate_post_embeddings"].shape == (3, 2)
    assert batch["history_author_indices"].shape == (2, 2)
    assert batch["history_prior_cumulative_likes"].shape == (2, 2)
    assert batch["candidate_post_author_idx"].shape == (3,)
    assert batch["candidate_prior_cumulative_likes"].shape == (3,)


def test_native_sampler_resamples_a_bounded_negative_pool_deterministically_by_epoch(tmp_path):
    bundle = _bundle(tmp_path)
    hour = datetime(2026, 1, 1, 12, tzinfo=UTC)
    created = datetime(2026, 1, 1, 11, 30, tzinfo=UTC)
    negative_uris = [f"n{index}" for index in range(8)]
    pl.DataFrame({
        "query_hour": [hour] * len(negative_uris),
        "subject_uri": negative_uris,
        "selection_source": ["random"] * len(negative_uris),
        "emb_idx": [3] * len(negative_uris),
        "post_created_at": [created] * len(negative_uris),
        "author_idx": [5] * len(negative_uris),
        "prior_like_count": list(range(len(negative_uris))),
    }, schema=dataset_hydration.HOURLY_NEGATIVE_SCHEMA).write_parquet(
        bundle / "hourly_negative_candidates" / "part-00000.parquet"
    )

    def epoch_candidates():
        dataset = HydratedBucketedEngagementDataset(
            bundle,
            split="train",
            max_history_len=2,
            bst_additional_batch_negatives=1,
            seed=7,
            logger=None,
        )
        sampler = HydratedBucketedBatchSampler(
            dataset,
            batch_size=2,
            shuffle=False,
            drop_last=False,
            seed=11,
            resample_candidates_each_epoch=True,
        )
        first_batch = dataset.collate_batch([
            dataset[index] for index in next(iter(sampler))
        ])
        second_batch = dataset.collate_batch([
            dataset[index] for index in next(iter(sampler))
        ])
        return first_batch["candidate_post_id"], second_batch["candidate_post_id"]

    first_sequence = epoch_candidates()
    second_sequence = epoch_candidates()

    assert first_sequence == second_sequence
    assert all(len(candidate_ids) == 3 for candidate_ids in first_sequence)
    assert first_sequence[0] != first_sequence[1]
