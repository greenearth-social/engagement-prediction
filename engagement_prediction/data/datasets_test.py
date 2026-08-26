from datetime import datetime, timezone
import pickle

import numpy as np
import polars as pl
import torch

from engagement_prediction.data import dataset_hydration, datasets as datasets_module
from engagement_prediction.data.datasets import (
    HydratedBucketedBatchSampler,
    HydratedBucketedEngagementDataset,
    create_hydrated_data_loader,
)
from engagement_prediction.data.training_index import build_loader_index


UTC = timezone.utc


def _write_dataset(path, name, df):
    dataset_path = path / name
    dataset_path.mkdir()
    df.write_parquet(dataset_path / "part-00000.parquet")


def _bundle(tmp_path, *, many_negatives=False):
    bundle = tmp_path / "hydrated_training_data_test"
    bundle.mkdir()
    hour = datetime(2026, 1, 1, 12, tzinfo=UTC)
    created = datetime(2026, 1, 1, 11, 30, tzinfo=UTC)
    if many_negatives:
        negative_uris = [f"n{index}" for index in range(8)]
        post_uris = ["p1", "p2", "h1", "h2", *negative_uris]
    else:
        negative_uris = ["p1", "n1"]
        post_uris = ["p1", "p2", "h1", "h2", "n1"]
    embeddings = np.lib.format.open_memmap(
        bundle / "embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(post_uris), 2),
    )
    embeddings[:] = np.asarray(
        [[float(index + 1), float(-(index + 1))] for index in range(len(post_uris))],
        dtype=np.float32,
    )
    embeddings.flush()
    del embeddings

    emb_idx_by_uri = {uri: index for index, uri in enumerate(post_uris)}
    posts = pl.DataFrame({
        "subject_uri": post_uris,
        "emb_idx": list(range(len(post_uris))),
        "post_created_at": [created] * len(post_uris),
        "author_did": [f"author{index}" for index in range(len(post_uris))],
        "author_idx": [index + 2 for index in range(len(post_uris))],
        "is_reply": [False] * len(post_uris),
        "is_positive": [uri in {"p1", "p2"} for uri in post_uris],
        "is_history": [uri in {"h1", "h2"} for uri in post_uris],
        "is_negative": [uri in negative_uris for uri in post_uris],
    }, schema=dataset_hydration.POST_SCHEMA)
    queries = pl.DataFrame({
        "did": ["u1", "u2"],
        "query_hour": [hour, hour],
        "user_cohort": ["seen", "seen"],
        "split": ["train", "train"],
        "positive_count": [2, 1],
    }, schema=dataset_hydration.QUERY_SCHEMA)
    positives = pl.DataFrame({
        "did": ["u1", "u1", "u2"],
        "query_hour": [hour, hour, hour],
        "subject_uri": ["p1", "p2", "p2"],
        "like_created_at": [hour, hour, hour],
        "emb_idx": [emb_idx_by_uri["p1"], emb_idx_by_uri["p2"], emb_idx_by_uri["p2"]],
        "post_created_at": [created, created, created],
        "author_idx": [2, 3, 3],
        "prior_like_count": [4, 5, 5],
    }, schema=dataset_hydration.QUERY_POSITIVE_SCHEMA)
    histories = pl.DataFrame({
        "did": ["u1", "u2"],
        "query_hour": [hour, hour],
        "history_subject_uris": [["h1", "h1", "h2"], []],
        "history_like_created_ats": [[
            datetime(2026, 1, 1, 10, 30, tzinfo=UTC),
            datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
        ], []],
        "history_emb_indices": [[emb_idx_by_uri["h1"], emb_idx_by_uri["h1"], emb_idx_by_uri["h2"]], []],
        "history_author_indices": [[4, 4, 5], []],
        "history_prior_like_counts": [[3, 4, 5], []],
    }, schema=dataset_hydration.QUERY_HISTORY_SCHEMA)
    negatives = pl.DataFrame({
        "query_hour": [hour] * len(negative_uris),
        "subject_uri": negative_uris,
        "selection_source": ["random"] * len(negative_uris),
        "emb_idx": [emb_idx_by_uri[uri] for uri in negative_uris],
        "post_created_at": [created] * len(negative_uris),
        "author_idx": [emb_idx_by_uri[uri] + 2 for uri in negative_uris],
        "prior_like_count": list(range(len(negative_uris))) if many_negatives else [4, 10],
    }, schema=dataset_hydration.HOURLY_NEGATIVE_SCHEMA)
    _write_dataset(bundle, "posts", posts)
    _write_dataset(bundle, "queries", queries)
    _write_dataset(bundle, "query_positives", positives)
    _write_dataset(bundle, "query_histories", histories)
    _write_dataset(bundle, "hourly_negative_candidates", negatives)
    _write_dataset(
        bundle,
        "authors",
        pl.DataFrame({"author_idx": list(range(2, len(post_uris) + 2))}),
    )
    build_loader_index(
        posts_path=bundle / "posts",
        queries_path=bundle / "queries",
        query_positives_path=bundle / "query_positives",
        query_histories_path=bundle / "query_histories",
        hourly_negative_candidates_path=bundle / "hourly_negative_candidates",
        embeddings_path=bundle / "embeddings.npy",
        authors_path=bundle / "authors",
        output_path=bundle / "loader_index",
        logger=None,
    )
    return bundle


def _dataset(bundle, *, negative_cap=None):
    return HydratedBucketedEngagementDataset(
        bundle,
        split="train",
        max_history_len=2,
        bst_additional_batch_negatives=negative_cap,
        seed=7,
        logger=None,
    )


def test_native_dataset_builds_parity_batch_from_memory_mapped_index(tmp_path):
    dataset = _dataset(_bundle(tmp_path))

    assert dataset._owner_pid is None
    assert not hasattr(dataset, "rows")
    assert not hasattr(dataset, "history_by_key")
    assert not hasattr(dataset, "negatives_by_hour")
    assert dataset[0] == (0, 0)
    batch = dataset.collate_batch([dataset[0], dataset[1]])

    assert dataset.embeddings.flags.writeable is False
    assert all(not array.flags.writeable for array in dataset._arrays.values())
    assert batch["candidate_post_id"] == ["p1", "p2", "n1"]
    assert batch["user_id"] == ["u1", "u2"]
    assert batch["label_matrix"].tolist() == [[1, 1, 0], [0, 1, 0]]
    assert batch["history_mask"].tolist() == [[True, True], [False, False]]
    assert batch["history_time_deltas_hours"].tolist() == [[1.5, 2.0], [0.0, 0.0]]
    assert batch["history_prior_cumulative_likes"].tolist() == [[3, 4], [0, 0]]
    assert batch["history_author_indices"].tolist() == [[4, 4], [0, 0]]
    assert batch["candidate_post_age_hours"].tolist() == [1.0, 1.0, 1.0]
    torch.testing.assert_close(
        batch["history_embeddings"][0],
        torch.tensor([[3.0, -3.0], [3.0, -3.0]]),
    )
    torch.testing.assert_close(
        batch["candidate_post_embeddings"],
        torch.tensor([[1.0, -1.0], [2.0, -2.0], [5.0, -5.0]]),
    )


def test_dataset_pickle_excludes_open_mappings_and_large_python_state(tmp_path):
    dataset = _dataset(_bundle(tmp_path))
    expected = dataset.collate_batch([dataset[0], dataset[1]])
    assert dataset._owner_pid is not None

    payload = pickle.dumps(dataset)
    restored = pickle.loads(payload)

    assert len(payload) < 20_000
    assert restored._owner_pid is None
    assert restored._arrays is None
    assert restored._embeddings is None
    actual = restored.collate_batch([restored[0], restored[1]])
    assert actual["user_id"] == expected["user_id"]
    assert actual["candidate_post_id"] == expected["candidate_post_id"]
    torch.testing.assert_close(actual["history_embeddings"], expected["history_embeddings"])
    torch.testing.assert_close(actual["label_matrix"], expected["label_matrix"])


def test_dataset_reopens_inherited_mappings_after_a_pid_change(tmp_path, monkeypatch):
    dataset = _dataset(_bundle(tmp_path))
    dataset.collate_batch([dataset[0], dataset[1]])
    original_owner_pid = dataset._owner_pid
    original_embeddings = dataset.embeddings

    monkeypatch.setattr(datasets_module.os, "getpid", lambda: original_owner_pid + 1)
    reopened_embeddings = dataset.embeddings

    assert dataset._owner_pid == original_owner_pid + 1
    assert reopened_embeddings is not original_embeddings
    assert original_embeddings._mmap.closed
    assert reopened_embeddings.flags.writeable is False


def test_native_sampler_is_hour_homogeneous_deterministic_and_bounded(tmp_path):
    dataset = _dataset(_bundle(tmp_path))
    first_sampler = HydratedBucketedBatchSampler(
        dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        seed=11,
        resample_candidates_each_epoch=False,
    )
    second_sampler = HydratedBucketedBatchSampler(
        dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        seed=11,
        resample_candidates_each_epoch=False,
    )

    first = list(first_sampler)
    second = list(second_sampler)

    assert first == second
    assert sorted(index for batch in first for index, _ in batch) == [0, 1]
    assert not hasattr(dataset, "row_indices_by_bucket")
    assert not any(isinstance(value, (list, dict, np.ndarray)) for value in first_sampler.__dict__.values())


def test_sampler_preserves_the_legacy_seeded_shuffle_sequence():
    class CompactDataset:
        query_count = 9
        hour_count = 3

        def _array(self, name):
            assert name == "hour_query_offsets"
            return np.asarray([0, 2, 7, 9], dtype=np.uint64)

    seed = 19
    batch_size = 2
    sampler = HydratedBucketedBatchSampler(
        CompactDataset(),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        seed=seed,
        resample_candidates_each_epoch=True,
    )

    # This is the previous list-backed algorithm. The compact sampler should
    # consume RNG in the same hour/row/batch sequence despite using arrays.
    rng = np.random.default_rng(seed)
    rows_by_hour = {0: [0, 1], 1: [2, 3, 4, 5, 6], 2: [7, 8]}
    hours = list(rows_by_hour)
    rng.shuffle(hours)
    expected = []
    for hour in hours:
        indices = list(rows_by_hour[hour])
        rng.shuffle(indices)
        expected.extend(
            indices[start:start + batch_size]
            for start in range(0, len(indices), batch_size)
        )
    rng.shuffle(expected)
    expected = [[(row_idx, 0) for row_idx in batch] for batch in expected]

    assert list(sampler) == expected


def test_native_pytorch_dataloader_matches_with_zero_and_multiple_workers(tmp_path):
    dataset = _dataset(_bundle(tmp_path))

    def load(num_workers):
        loader = create_hydrated_data_loader(
            dataset,
            batch_size=2,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=1,
            seed=7,
            resample_candidates_each_epoch=False,
        )
        return next(iter(loader))

    direct = load(0)
    worker = load(2)

    assert worker["user_id"] == direct["user_id"]
    assert worker["candidate_post_id"] == direct["candidate_post_id"]
    for key in (
        "history_embeddings",
        "history_mask",
        "history_author_indices",
        "history_time_deltas_hours",
        "history_prior_cumulative_likes",
        "candidate_post_embeddings",
        "candidate_post_author_idx",
        "candidate_prior_cumulative_likes",
        "candidate_post_age_hours",
        "label_matrix",
    ):
        torch.testing.assert_close(worker[key], direct[key])


def test_persistent_workers_receive_new_epoch_refs_without_copying_dataset_state(tmp_path):
    dataset = _dataset(_bundle(tmp_path, many_negatives=True), negative_cap=1)
    loader = create_hydrated_data_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=1,
        seed=11,
        resample_candidates_each_epoch=True,
    )
    try:
        epoch_zero = next(iter(loader))["candidate_post_id"]
        epoch_one = next(iter(loader))["candidate_post_id"]
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()

    assert epoch_zero != epoch_one
    assert all(len(candidate_ids) == 3 for candidate_ids in (epoch_zero, epoch_one))


def test_sampler_resamples_negatives_by_epoch_and_has_stable_evaluation_mode(tmp_path):
    dataset = _dataset(_bundle(tmp_path, many_negatives=True), negative_cap=1)
    sampler = HydratedBucketedBatchSampler(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        seed=11,
        resample_candidates_each_epoch=True,
    )

    def next_candidates():
        refs = next(iter(sampler))
        return dataset.collate_batch([dataset[ref] for ref in refs])["candidate_post_id"]

    epoch_zero = next_candidates()
    epoch_one = next_candidates()
    sampler.set_evaluation_mode()
    evaluation_a = next_candidates()
    evaluation_b = next_candidates()

    assert epoch_zero != epoch_one
    assert evaluation_a == evaluation_b == epoch_zero
    assert all(len(candidate_ids) == 3 for candidate_ids in (epoch_zero, epoch_one))


def test_empty_split_has_schema_correct_compact_index(tmp_path):
    bundle = _bundle(tmp_path)
    dataset = HydratedBucketedEngagementDataset(
        bundle,
        split="val",
        max_history_len=2,
        bst_additional_batch_negatives=None,
        seed=7,
        logger=None,
    )
    sampler = HydratedBucketedBatchSampler(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        seed=7,
        resample_candidates_each_epoch=False,
    )

    assert len(dataset) == 0
    assert len(sampler) == 0
    assert list(sampler) == []
