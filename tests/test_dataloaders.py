"""Tests for bucketed two-tower dataloaders."""
import json
import logging
from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest
from torch.utils.data import DataLoader

from shared.input_data_helpers import AUTHOR_PAD_IDX, AUTHOR_UNK_IDX
from utils.dataloaders import (
    BucketedBatchSampler,
    BucketedEngagementDataset,
    create_bucketed_data_loaders,
    get_author_table_num_rows,
    validate_history_popularity_semantics,
)


def _dt(hour: int) -> datetime:
    return datetime(2024, 1, 1, hour, tzinfo=timezone.utc)


@pytest.fixture
def mock_embeddings_mmap():
    values = np.arange(40 * 4, dtype=np.float32).reshape(40, 4)
    return values / 10.0


@pytest.fixture
def mock_likes_core_df():
    return pl.DataFrame({
        "did": ["u1", "u2", "u1", "u1", "u3", "u4", "u5"],
        "subject_uri": ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        "split": ["train", "train", "train", "train", "train", "val", "val_unseen_users"],
        "record_created_at": [
            datetime(2024, 1, 1, 10, 15, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 10, 45, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 11, 10, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 13, 20, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 13, 40, tzinfo=timezone.utc),
        ],
        "like_hour_bucket": [_dt(10), _dt(10), _dt(10), _dt(11), _dt(12), _dt(13), _dt(13)],
        "emb_idx": [0, 1, 2, 3, 4, 5, 6],
        "prior_cumulative_likes": [101, 102, 103, 104, 105, 106, 107],
        "author_idx": pl.Series([2, 3, 4, None, 2, 2, 4], dtype=pl.UInt32),
    })


@pytest.fixture
def mock_posts_core_df():
    return pl.DataFrame({
        "at_uri": ["n1", "p2", "p3", "p4", "n2", "n_val", "n_holdout"],
        "in_random_sample": [True, True, True, True, True, True, True],
        "negative_hour_bucket": [_dt(10), _dt(10), _dt(10), _dt(10), _dt(11), _dt(13), _dt(14)],
        "split_window": ["train", "train", "train", "train", "train", "val", "holdout"],
        "emb_idx": [20, 1, 2, 3, 21, 22, 23],
        "prior_cumulative_likes": [None, 202, 203, 204, 205, 206, 207],
        "author_idx": pl.Series([2, 3, 4, None, 2, 4, 2], dtype=pl.UInt32),
    })


@pytest.fixture
def mock_history_df():
    return pl.DataFrame({
        "did": ["u1", "u2", "u1", "u3", "u4", "u5"],
        "like_hour_bucket": [_dt(10), _dt(10), _dt(11), _dt(12), _dt(13), _dt(13)],
        "prior_emb_indices": [[5, 6, 7], [], [8], [], [9], [10]],
        "prior_like_age_hours_at_bucket_start": [[1.0, 2.0, 3.0], [], [0.25], [], [4.0], [5.0]],
        "prior_cumulative_likes": [[15, None, 17], [], [18], [], [19], [20]],
        "prior_author_indices": [[2, None, 4], [], [3], [], [2], [4]],
    })


def _political_pool_posts_df(post_ids, political_labels):
    return pl.DataFrame({
        "at_uri": post_ids,
        "in_random_sample": [True] * len(post_ids),
        "negative_hour_bucket": [_dt(10)] * len(post_ids),
        "split_window": ["train"] * len(post_ids),
        "emb_idx": list(range(20, 20 + len(post_ids))),
        "is_political": pl.Series(political_labels, dtype=pl.Boolean),
    })


def _political_sampling_dataset(
    mock_embeddings_mmap,
    mock_likes_core_df,
    mock_history_df,
    posts_core_df,
    *,
    total_negatives,
    political_negatives,
    seed=0,
    logger=None,
):
    return BucketedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        likes_core_df=mock_likes_core_df,
        posts_core_df=posts_core_df,
        history_df=mock_history_df,
        split="train",
        max_history_len=3,
        embed_dim=4,
        bst_political_batch_negatives=political_negatives,
        bst_additional_batch_negatives=total_negatives,
        seed=seed,
        logger=logger,
    )


@pytest.fixture
def bucketed_dataset(mock_embeddings_mmap, mock_likes_core_df, mock_posts_core_df, mock_history_df):
    return BucketedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        likes_core_df=mock_likes_core_df,
        posts_core_df=mock_posts_core_df,
        history_df=mock_history_df,
        split="train",
        max_history_len=3,
        embed_dim=4,
        bst_political_batch_negatives=0,
    )


def test_get_author_table_num_rows_uses_stage_one_author_indices_directly():
    author_idx_mapping_df = pl.DataFrame({
        "author_idx": pl.Series([2, 3, 4], dtype=pl.UInt32),
        "author_train_count": [10, 3, 5],
    })

    assert get_author_table_num_rows(author_idx_mapping_df) == 5


def test_get_author_table_num_rows_empty_mapping_still_reserves_pad_and_unk_rows():
    author_idx_mapping_df = pl.DataFrame({
        "author_idx": pl.Series([], dtype=pl.UInt32),
        "author_train_count": pl.Series([], dtype=pl.UInt32),
    })

    assert get_author_table_num_rows(author_idx_mapping_df) == 2


def test_validate_history_popularity_semantics_accepts_target_hour_summary(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({
        "history_prior_cumulative_likes_semantics": "target_hour",
    }))

    validate_history_popularity_semantics(tmp_path)


def test_validate_history_popularity_semantics_rejects_stale_summary(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({
        "history_prior_cumulative_likes_semantics": "liked_hour",
    }))

    with pytest.raises(RuntimeError, match="target-hour history prior_cumulative_likes"):
        validate_history_popularity_semantics(tmp_path)


def test_validate_history_popularity_semantics_rejects_missing_summary(tmp_path):
    with pytest.raises(RuntimeError, match="summary.json"):
        validate_history_popularity_semantics(tmp_path)


def test_bucketed_dataset_groups_user_hours_and_joins_history(bucketed_dataset):
    assert len(bucketed_dataset) == 4
    assert bucketed_dataset.row_indices_by_bucket == {
        _dt(10): [0, 1],
        _dt(11): [2],
        _dt(12): [3],
    }
    assert bucketed_dataset.user_ids == ["u1", "u2", "u1", "u3"]
    assert bucketed_dataset.liked_post_ids == [["p1", "p3"], ["p2"], ["p4"], ["p5"]]
    assert bucketed_dataset.prior_emb_indices[0].tolist() == [5, 6, 7]
    assert bucketed_dataset.prior_emb_indices[1].tolist() == []


def test_bucketed_batch_sampler_keeps_each_batch_in_one_bucket(bucketed_dataset):
    sampler = BucketedBatchSampler(
        dataset=bucketed_dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        seed=0,
    )

    batches = list(sampler)

    assert batches == [[0, 1], [2], [3]]
    for batch in batches:
        buckets = {bucketed_dataset.like_hour_buckets[row_idx] for row_idx in batch}
        assert len(buckets) == 1
        assert len(batch) <= 2


def test_bucketed_collate_builds_candidates_and_same_hour_labels(bucketed_dataset):
    batch = bucketed_dataset.collate_batch([bucketed_dataset[0], bucketed_dataset[1]])

    assert batch["bucket"] == _dt(10)
    assert batch["user_id"] == ["u1", "u2"]
    assert batch["candidate_post_id"] == ["p1", "p3", "p2", "n1", "p4"]
    assert batch["history_embeddings"].shape == (2, 3, 4)
    assert batch["history_mask"].tolist() == [[True, True, True], [False, False, False]]
    np.testing.assert_allclose(
        batch["history_time_deltas_hours"].numpy(),
        np.array([
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    assert batch["candidate_post_embeddings"].shape == (5, 4)
    assert "history_prior_cumulative_likes" not in batch
    assert "candidate_prior_cumulative_likes" not in batch

    labels_by_user = {
        user_id: batch["label_matrix"][idx].tolist()
        for idx, user_id in enumerate(batch["user_id"])
    }
    assert labels_by_user["u1"] == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert labels_by_user["u2"] == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_bucketed_collate_dedupes_candidates(bucketed_dataset):
    batch = bucketed_dataset.collate_batch([bucketed_dataset[0]])

    assert batch["user_id"] == ["u1"]
    assert batch["candidate_post_id"] == ["p1", "p3", "n1", "p2", "p4"]
    assert batch["label_matrix"].shape == (1, 5)
    assert batch["label_matrix"][0].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_bucketed_collate_additional_negatives_are_added_after_positives(
    mock_embeddings_mmap,
    mock_likes_core_df,
    mock_posts_core_df,
    mock_history_df,
):
    dataset = BucketedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        likes_core_df=mock_likes_core_df,
        posts_core_df=mock_posts_core_df,
        history_df=mock_history_df,
        split="train",
        max_history_len=3,
        embed_dim=4,
        bst_political_batch_negatives=0,
        bst_additional_batch_negatives=1,
        seed=0,
    )

    batch = dataset.collate_batch([dataset[0], dataset[1]])

    assert len(batch["candidate_post_id"]) == 4
    assert {"p1", "p2", "p3"} <= set(batch["candidate_post_id"])
    assert len(set(batch["candidate_post_id"]).intersection({"n1", "p4"})) == 1
    positive_indices = {
        post_id: idx
        for idx, post_id in enumerate(batch["candidate_post_id"])
        if post_id in {"p1", "p2", "p3"}
    }
    assert batch["label_matrix"][0, positive_indices["p1"]].item() == 1.0
    assert batch["label_matrix"][0, positive_indices["p3"]].item() == 1.0
    assert batch["label_matrix"][1, positive_indices["p2"]].item() == 1.0


def test_bucketed_collate_additional_negatives_do_not_cap_positives(
    mock_embeddings_mmap,
    mock_likes_core_df,
    mock_posts_core_df,
    mock_history_df,
):
    dataset = BucketedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        likes_core_df=mock_likes_core_df,
        posts_core_df=mock_posts_core_df,
        history_df=mock_history_df,
        split="train",
        max_history_len=3,
        embed_dim=4,
        bst_political_batch_negatives=0,
        bst_additional_batch_negatives=2,
        seed=0,
    )

    batch = dataset.collate_batch([dataset[0], dataset[1]])

    assert batch["candidate_post_id"] == ["p1", "p3", "p2", "n1", "p4"]
    assert batch["label_matrix"].shape == (2, 5)


def test_bucketed_collate_returns_popularity_tensors_when_enabled(
    mock_embeddings_mmap,
    mock_likes_core_df,
    mock_posts_core_df,
    mock_history_df,
):
    dataset = BucketedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        likes_core_df=mock_likes_core_df,
        posts_core_df=mock_posts_core_df,
        history_df=mock_history_df,
        split="train",
        max_history_len=4,
        embed_dim=4,
        bst_political_batch_negatives=0,
        use_popularity_feature=True,
    )

    batch = dataset.collate_batch([dataset[0], dataset[1]])

    assert batch["history_prior_cumulative_likes"].shape == (2, 4)
    np.testing.assert_allclose(
        batch["history_prior_cumulative_likes"].numpy(),
        np.array([
            [15.0, 0.0, 17.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    assert batch["candidate_post_id"] == ["p1", "p3", "p2", "n1", "p4"]
    np.testing.assert_allclose(
        batch["candidate_prior_cumulative_likes"].numpy(),
        np.array([101.0, 103.0, 102.0, 0.0, 204.0], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )


def test_bucketed_candidate_sampling_changes_by_epoch(
    mock_embeddings_mmap,
    mock_likes_core_df,
    mock_posts_core_df,
    mock_history_df,
):
    dataset = BucketedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        likes_core_df=mock_likes_core_df,
        posts_core_df=mock_posts_core_df,
        history_df=mock_history_df,
        split="train",
        max_history_len=3,
        embed_dim=4,
        bst_political_batch_negatives=0,
        bst_additional_batch_negatives=1,
        seed=0,
    )

    sampled_candidates = [
        tuple(dataset.collate_batch([dataset[(0, epoch)], dataset[(1, epoch)]])["candidate_post_id"])
        for epoch in range(8)
    ]

    assert len(set(sampled_candidates)) > 1
    assert sampled_candidates == [
        tuple(dataset.collate_batch([dataset[(0, epoch)], dataset[(1, epoch)]])["candidate_post_id"])
        for epoch in range(8)
    ]


def test_bucketed_validation_candidate_sampling_is_deterministic(
    mock_embeddings_mmap,
    mock_likes_core_df,
    mock_posts_core_df,
    mock_history_df,
):
    dataset = BucketedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        likes_core_df=mock_likes_core_df,
        posts_core_df=mock_posts_core_df,
        history_df=mock_history_df,
        split="train",
        max_history_len=3,
        embed_dim=4,
        bst_political_batch_negatives=0,
        bst_additional_batch_negatives=1,
        seed=0,
    )

    first = dataset.collate_batch([dataset[0], dataset[1]])
    second = dataset.collate_batch([dataset[0], dataset[1]])

    assert first["candidate_post_id"] == second["candidate_post_id"]


def test_bucketed_political_sampling_replaces_only_the_ordinary_shortfall_and_excludes_positives(
    mock_embeddings_mmap,
    mock_likes_core_df,
    mock_history_df,
):
    pool_ids = ["n1", "n2", "n3", "n4", "n5", "n6", "p2"]
    ordinary_posts_df = _political_pool_posts_df(pool_ids, [False] * len(pool_ids))
    ordinary_dataset = _political_sampling_dataset(
        mock_embeddings_mmap,
        mock_likes_core_df,
        mock_history_df,
        ordinary_posts_df,
        total_negatives=3,
        political_negatives=0,
        seed=17,
    )
    positive_ids = {"p1", "p2", "p3"}
    ordinary_batch = ordinary_dataset.collate_batch([ordinary_dataset[0], ordinary_dataset[1]])
    ordinary_negative_ids = set(ordinary_batch["candidate_post_id"]) - positive_ids
    political_in_ordinary = next(iter(ordinary_negative_ids))
    political_outside_ordinary = next(
        post_id
        for post_id in pool_ids
        if post_id not in ordinary_negative_ids and post_id not in positive_ids
    )
    political_ids = {political_in_ordinary, political_outside_ordinary, "p2"}
    labeled_posts_df = _political_pool_posts_df(
        pool_ids,
        [post_id in political_ids for post_id in pool_ids],
    )
    dataset = _political_sampling_dataset(
        mock_embeddings_mmap,
        mock_likes_core_df,
        mock_history_df,
        labeled_posts_df,
        total_negatives=3,
        political_negatives=2,
        seed=17,
    )

    batch = dataset.collate_batch([dataset[0], dataset[1]])
    sampled_negative_ids = set(batch["candidate_post_id"]) - positive_ids

    assert len(sampled_negative_ids) == 3
    assert political_in_ordinary in sampled_negative_ids
    assert political_outside_ordinary in sampled_negative_ids
    assert len(sampled_negative_ids & ordinary_negative_ids) == 2
    assert batch["candidate_post_id"].count("p2") == 1
    assert batch["sampled_negative_count"] == 3
    assert batch["sampled_political_negative_count"] == 2
    assert batch["political_negative_quota"] == 2
    assert batch["political_negative_quota_met"] is True


def test_bucketed_political_sampling_uses_available_posts_and_logs_shortage(
    mock_embeddings_mmap,
    mock_likes_core_df,
    mock_history_df,
    caplog,
):
    posts_df = _political_pool_posts_df(
        ["political", "nonpolitical", "unknown"],
        [True, False, None],
    )
    logger = logging.getLogger("test_dataloaders.political_shortage")

    with caplog.at_level(logging.INFO):
        dataset = _political_sampling_dataset(
            mock_embeddings_mmap,
            mock_likes_core_df,
            mock_history_df,
            posts_df,
            total_negatives=5,
            political_negatives=2,
            logger=logger,
        )
        batch = dataset.collate_batch([dataset[0], dataset[1]])

    assert batch["sampled_negative_count"] == 3
    assert batch["sampled_political_negative_count"] == 1
    assert batch["political_negative_quota_met"] is False
    assert "total=3, political=1, non-political=1, unknown=1" in caplog.text
    assert "1 hourly buckets below quota 2" in caplog.text


def test_bucketed_political_sampling_keeps_ordinary_sample_when_quota_is_already_met(
    mock_embeddings_mmap,
    mock_likes_core_df,
    mock_history_df,
):
    posts_df = _political_pool_posts_df(
        [f"n{idx}" for idx in range(6)],
        [True] * 6,
    )
    ordinary_dataset = _political_sampling_dataset(
        mock_embeddings_mmap,
        mock_likes_core_df,
        mock_history_df,
        posts_df,
        total_negatives=3,
        political_negatives=0,
        seed=31,
    )
    political_dataset = _political_sampling_dataset(
        mock_embeddings_mmap,
        mock_likes_core_df,
        mock_history_df,
        posts_df,
        total_negatives=3,
        political_negatives=2,
        seed=31,
    )

    ordinary_batch = ordinary_dataset.collate_batch([ordinary_dataset[0], ordinary_dataset[1]])
    political_batch = political_dataset.collate_batch([political_dataset[0], political_dataset[1]])

    assert political_batch["candidate_post_id"] == ordinary_batch["candidate_post_id"]
    assert political_batch["sampled_political_negative_count"] == 3
    assert political_batch["political_negative_quota_met"] is True


def test_bucketed_political_sampling_is_deterministic_and_resamples_by_epoch(
    mock_embeddings_mmap,
    mock_likes_core_df,
    mock_history_df,
):
    posts_df = _political_pool_posts_df(
        [f"n{idx}" for idx in range(8)],
        [True, True, True, False, False, False, None, None],
    )
    dataset = _political_sampling_dataset(
        mock_embeddings_mmap,
        mock_likes_core_df,
        mock_history_df,
        posts_df,
        total_negatives=4,
        political_negatives=2,
        seed=23,
    )

    sampled_candidates = [
        tuple(dataset.collate_batch([dataset[(0, epoch)], dataset[(1, epoch)]])["candidate_post_id"])
        for epoch in range(8)
    ]

    assert len(set(sampled_candidates)) > 1
    assert sampled_candidates == [
        tuple(dataset.collate_batch([dataset[(0, epoch)], dataset[(1, epoch)]])["candidate_post_id"])
        for epoch in range(8)
    ]
    for epoch in range(8):
        batch = dataset.collate_batch([dataset[(0, epoch)], dataset[(1, epoch)]])
        assert batch["sampled_negative_count"] == 4
        assert batch["sampled_political_negative_count"] >= 2
        assert batch["political_negative_quota_met"] is True


def test_bucketed_political_sampling_requires_boolean_label_only_when_enabled(
    mock_embeddings_mmap,
    mock_likes_core_df,
    mock_posts_core_df,
    mock_history_df,
):
    disabled_dataset = _political_sampling_dataset(
        mock_embeddings_mmap,
        mock_likes_core_df,
        mock_history_df,
        mock_posts_core_df,
        total_negatives=2,
        political_negatives=0,
    )
    assert disabled_dataset.collate_batch(
        [disabled_dataset[0], disabled_dataset[1]]
    )["sampled_negative_count"] == 2

    with pytest.raises(ValueError, match="posts_core.is_political"):
        _political_sampling_dataset(
            mock_embeddings_mmap,
            mock_likes_core_df,
            mock_history_df,
            mock_posts_core_df,
            total_negatives=2,
            political_negatives=1,
        )

    invalid_posts_df = mock_posts_core_df.with_columns(
        pl.lit("false").alias("is_political")
    )
    with pytest.raises(ValueError, match="nullable Boolean"):
        _political_sampling_dataset(
            mock_embeddings_mmap,
            mock_likes_core_df,
            mock_history_df,
            invalid_posts_df,
            total_negatives=2,
            political_negatives=1,
        )


def test_bucketed_collate_handles_empty_sampled_negative_bucket(bucketed_dataset):
    batch = bucketed_dataset.collate_batch([bucketed_dataset[3]])

    assert batch["bucket"] == _dt(12)
    assert batch["user_id"] == ["u3"]
    assert batch["candidate_post_id"] == ["p5"]
    assert batch["label_matrix"].tolist() == [[1.0]]


def test_bucketed_collate_returns_author_tensors_when_enabled(
    mock_embeddings_mmap,
    mock_likes_core_df,
    mock_posts_core_df,
    mock_history_df,
):
    dataset = BucketedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        likes_core_df=mock_likes_core_df,
        posts_core_df=mock_posts_core_df,
        history_df=mock_history_df,
        split="train",
        max_history_len=4,
        embed_dim=4,
        bst_political_batch_negatives=0,
        use_author_embedding_table=True,
    )

    batch = dataset.collate_batch([dataset[0], dataset[1]])

    assert batch["history_author_indices"].shape == (2, 4)
    assert batch["history_author_indices"][0].tolist() == [2, AUTHOR_UNK_IDX, 4, AUTHOR_PAD_IDX]
    assert batch["history_author_indices"][1].tolist() == [AUTHOR_PAD_IDX] * 4
    assert batch["candidate_post_author_idx"].tolist() == [
        2,
        4,
        3,
        2,
        AUTHOR_UNK_IDX,
    ]


def test_create_bucketed_data_loaders_returns_iterable_loaders(
    bucketed_dataset,
    mock_embeddings_mmap,
    mock_likes_core_df,
    mock_posts_core_df,
    mock_history_df,
):
    val_dataset = BucketedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        likes_core_df=mock_likes_core_df,
        posts_core_df=mock_posts_core_df,
        history_df=mock_history_df,
        split="val",
        max_history_len=3,
        embed_dim=4,
        bst_political_batch_negatives=0,
    )
    val_unseen_dataset = BucketedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        likes_core_df=mock_likes_core_df,
        posts_core_df=mock_posts_core_df,
        history_df=mock_history_df,
        split="val_unseen_users",
        max_history_len=3,
        embed_dim=4,
        bst_political_batch_negatives=0,
    )

    train_loader, val_loader, val_unseen_loader, holdout_loader = create_bucketed_data_loaders(
        train_dataset=bucketed_dataset,
        val_dataset=val_dataset,
        val_unseen_dataset=val_unseen_dataset,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
        seed=0,
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(val_unseen_loader, DataLoader)
    assert holdout_loader is None
    batch = next(iter(train_loader))
    assert {"history_embeddings", "history_mask", "history_time_deltas_hours", "candidate_post_embeddings", "label_matrix"} <= set(batch)
