"""Tests for the minimal collaborative-filter Stage 4 implementation."""

import importlib
import logging

import polars as pl
import torch

stage_train_cf = importlib.import_module("utils.04_train.stage_train_collaborative_filter")
CollaborativeFilteringModel = stage_train_cf.CollaborativeFilteringModel
_build_candidate_dataset = stage_train_cf._build_candidate_dataset
_build_item_index = stage_train_cf._build_item_index
_build_train_positive_lookup = stage_train_cf._build_train_positive_lookup
_build_user_index = stage_train_cf._build_user_index


def test_collaborative_filter_model_outputs_probabilities():
    model = CollaborativeFilteringModel(num_users=5, num_items=7, latent_dim=3)

    preds = model(
        user_index=torch.tensor([0, 1, 2], dtype=torch.long),
        item_index=torch.tensor([0, 3, 6], dtype=torch.long),
    )

    assert preds.shape == (3,)
    assert torch.all(preds >= 0)
    assert torch.all(preds <= 1)


def test_build_user_index_uses_train_split_only():
    target_posts_df = pl.DataFrame({
        "target_did": ["u_train", "u_val", "u_holdout"],
        "split": ["train", "val", "holdout_unseen_users"],
        "neg_emb_idx": [10, 11, 12],
    })

    mapping, unknown_index = _build_user_index(target_posts_df, logging.getLogger("cf-test"))

    assert mapping == {"u_train": 0}
    assert unknown_index == 1


def test_build_train_positive_lookup_uses_only_observed_likes():
    target_posts_df = pl.DataFrame({
        "target_did": ["u_train", "u_train", "u_other"],
        "like_uri": ["like_a", "like_b", "like_c"],
        "neg_uri": ["neg_a", "neg_b", "neg_c"],
        "like_emb_idx": [11, 12, 13],
        "neg_emb_idx": [99, 98, 97],
        "split": ["train", "train", "val"],
    })

    user_mapping, _unknown_user_index = _build_user_index(target_posts_df, logging.getLogger("cf-test"))
    item_mapping, _unknown_item_index = _build_item_index(target_posts_df, logging.getLogger("cf-test"))
    lookup, train_user_indices, num_positive_pairs = _build_train_positive_lookup(
        target_posts_df,
        user_mapping,
        item_mapping,
        logging.getLogger("cf-test"),
    )

    assert train_user_indices.tolist() == [0]
    assert num_positive_pairs == 2
    assert lookup[0].tolist() == [item_mapping[11], item_mapping[12]]


def test_build_candidate_dataset_maps_unseen_users_to_unknown_bucket():
    target_posts_df = pl.DataFrame({
        "target_did": ["u_train", "u_holdout"],
        "like_uri": ["like_train", "like_holdout"],
        "neg_uri": ["neg_train", "neg_holdout"],
        "like_emb_idx": [1, 2],
        "neg_emb_idx": [3, 4],
        "split": ["train", "holdout_unseen_users"],
    })

    mapping, unknown_index = _build_user_index(target_posts_df, logging.getLogger("cf-test"))
    item_mapping, unknown_item_index = _build_item_index(target_posts_df, logging.getLogger("cf-test"))
    dataset, stats = _build_candidate_dataset(
        target_posts_df=target_posts_df,
        split="holdout_unseen_users",
        user_to_index=mapping,
        unknown_user_index=unknown_index,
        item_to_index=item_mapping,
        unknown_item_index=unknown_item_index,
        logger=logging.getLogger("cf-test"),
    )

    first_row = dataset[0]
    second_row = dataset[1]

    assert len(dataset) == 2
    assert stats["unknown_users"] == 1
    assert first_row["user_index"].item() == unknown_index
    assert second_row["user_index"].item() == unknown_index
    assert first_row["label"].item() == 1.0
    assert second_row["label"].item() == 0.0
    assert first_row["post_id"] == "like_holdout"
    assert second_row["post_id"] == "neg_holdout"
