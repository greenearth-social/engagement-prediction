import base64
from datetime import datetime, timezone
import logging
from pathlib import Path
import struct
import threading
import zlib

import numpy as np
import polars as pl

from engagement_prediction.data import dataset_hydration
from engagement_prediction.data import dataset_hydration_artifacts
from engagement_prediction.data import author_vocabulary
from engagement_prediction.data import post_liker_users
from engagement_prediction.data.parquet import scan_parquet_artifact


UTC = timezone.utc


def _write_part(path, frame):
    path.mkdir(parents=True)
    frame.write_parquet(path / "part-00000.parquet")


def _compressed(values: list[float]) -> str:
    payload = struct.pack(f"<{len(values)}f", *values)
    return base64.b85encode(zlib.compress(payload)).decode()


def _embedding(values: list[float]) -> list[dict[str, str]]:
    return [{"key": "all_MiniLM_L12_v2", "value": _compressed(values)}]


def test_embedding_source_batching_scans_each_file_once_and_writes_only_selected_payloads(
    tmp_path,
    monkeypatch,
):
    selected_source_path = tmp_path / "selected-posts.parquet"
    unselected_source_path = tmp_path / "unselected-posts.parquet"
    other_unselected_source_path = tmp_path / "other-unselected-posts.parquet"
    source_rows = pl.DataFrame({
        "at_uri": ["selected", "unselected", "selected", "other-unselected"],
        "record_created_at": [
            datetime(2026, 1, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 1, 2, tzinfo=UTC),
            datetime(2026, 1, 1, 3, tzinfo=UTC),
            datetime(2026, 1, 1, 4, tzinfo=UTC),
        ],
        "did": ["author-a", "author-b", "author-a", "author-c"],
        "embeddings": [
            [{"key": "model", "value": "payload-1"}],
            [{"key": "model", "value": "payload-2"}],
            [{"key": "model", "value": "payload-3"}],
            [{"key": "model", "value": "payload-4"}],
        ],
    })
    source_rows.filter(pl.col("at_uri") == "selected").write_parquet(
        selected_source_path
    )
    source_rows.filter(pl.col("at_uri") == "unselected").write_parquet(
        unselected_source_path
    )
    source_rows.filter(pl.col("at_uri") == "other-unselected").write_parquet(
        other_unselected_source_path
    )

    selected_metadata_path = tmp_path / "selected-metadata"
    selected_metadata_path.mkdir()
    pl.DataFrame({
        "subject_uri": ["selected"],
        "post_created_at": [datetime(2026, 1, 1, 3, tzinfo=UTC)],
        "author_did": ["author-a"],
        "is_reply": [False],
        "is_positive": [True],
        "is_history": [False],
        "is_negative": [False],
    }).write_parquet(selected_metadata_path / "part-00000.parquet")

    selected_metadata_scans = []
    original_artifact_scan = dataset_hydration_artifacts.scan_parquet_artifact

    def recording_artifact_scan(path):
        selected_metadata_scans.append(path)
        return original_artifact_scan(path)

    monkeypatch.setattr(
        dataset_hydration_artifacts,
        "scan_parquet_artifact",
        recording_artifact_scan,
    )

    source_scans = []
    source_scan_barrier = threading.Barrier(2)
    original_source_scan = dataset_hydration_artifacts.ingex.scan_parquet_files

    def recording_source_scan(paths, **kwargs):
        source_scans.append((list(paths), kwargs.get("include_file_paths")))
        source_scan_barrier.wait(timeout=5)
        return original_source_scan(paths, **kwargs)

    monkeypatch.setattr(
        dataset_hydration_artifacts.ingex,
        "scan_parquet_files",
        recording_source_scan,
    )

    routed_inputs = []
    original_sink = dataset_hydration_artifacts.sink_partitioned_parquet

    def recording_sink(lf, *, output_path, key):
        frame = lf.collect(engine="streaming")
        routed_inputs.append(frame)
        original_sink(frame.lazy(), output_path=output_path, key=key)

    monkeypatch.setattr(
        dataset_hydration_artifacts,
        "sink_partitioned_parquet",
        recording_sink,
    )

    output_path = tmp_path / "selected-embedding-rows"
    temporary_routes_root = tmp_path / "temporary-routes"
    stats = dataset_hydration_artifacts.materialize_selected_embedding_rows(
        post_paths=[
            str(selected_source_path),
            str(unselected_source_path),
            str(other_unselected_source_path),
        ],
        reply_paths=[],
        posts_start=datetime(2026, 1, 1, tzinfo=UTC),
        posts_end=datetime(2026, 1, 2, tzinfo=UTC),
        selected_metadata_path=selected_metadata_path,
        output_path=output_path,
        temporary_routes_root=temporary_routes_root,
        partition_count=1,
        source_batch_size=2,
        worker_count=2,
        logger=logging.getLogger("dataset-hydration-test"),
    )

    assert selected_metadata_scans == [selected_metadata_path]
    assert sorted(
        (tuple(paths), file_column) for paths, file_column in source_scans
    ) == sorted([
        ((str(selected_source_path), str(unselected_source_path)), None),
        ((str(other_unselected_source_path),), None),
    ])
    assert len(routed_inputs) == 2
    assert all("embeddings" in frame.columns for frame in routed_inputs)
    selected_routed_input = next(frame for frame in routed_inputs if not frame.is_empty())
    assert selected_routed_input.get_column("subject_uri").to_list() == [
        "selected",
        "selected",
    ]
    assert sum(frame.is_empty() for frame in routed_inputs) == 1

    selected_rows = scan_parquet_artifact(output_path).collect()
    assert selected_rows.get_column("subject_uri").to_list() == [
        "selected",
        "selected",
    ]
    assert stats == {
        "embedding_source_file_count": 3,
        "embedding_source_batch_count": 2,
        "embedding_source_worker_count": 2,
        "payload_embedding_source_file_count": 3,
        "selected_embedding_source_row_count": 2,
    }
    assert not temporary_routes_root.exists()


def test_embedding_shard_writer_reuses_the_vectors_decoded_during_validation(
    tmp_path,
    monkeypatch,
):
    source_path = tmp_path / "selected-embedding-rows" / "partition-00000"
    source_path.mkdir(parents=True)
    pl.DataFrame({
        "subject_uri": ["post", "post"],
        "post_created_at": [
            datetime(2026, 1, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 1, 2, tzinfo=UTC),
        ],
        "author_did": ["author", "author"],
        "embeddings": [_embedding([1.0, 2.0]), _embedding([3.0, 4.0])],
    }).write_parquet(source_path / "rows.parquet")

    decode_count = 0
    original_decode = dataset_hydration.get_expanded_embedding_vector

    def count_decode(payload, model):
        nonlocal decode_count
        decode_count += 1
        return original_decode(payload, model)

    monkeypatch.setattr(
        dataset_hydration,
        "get_expanded_embedding_vector",
        count_decode,
    )
    stats = dataset_hydration_artifacts.write_embedding_shards(
        selected_embedding_rows_path=tmp_path / "selected-embedding-rows",
        valid_embedding_rows_path=tmp_path / "valid-embedding-rows",
        embedding_shards_path=tmp_path / "embedding-shards",
        embedding_model="all_MiniLM_L12_v2",
        embedding_dim=2,
        partition_count=1,
        worker_count=1,
        logger=logging.getLogger("embedding-shard-test"),
    )

    # Both source rows must be validated, but the selected second row is not
    # decoded again when its vector is written to the NumPy shard.
    assert decode_count == 2
    shard = np.load(tmp_path / "embedding-shards" / "part-00000.npy")
    assert shard.tolist() == [[3.0, 4.0]]
    assert stats["embedding_partition_worker_count"] == 1
    assert stats["embedding_partition_counts"] == [1]


def test_embedding_shards_are_deterministic_with_parallel_partition_workers(tmp_path):
    selected_rows_path = tmp_path / "selected-embedding-rows"
    for partition_id, uri, vector in (
        (0, "a", [1.0, 2.0]),
        (1, "b", [3.0, 4.0]),
    ):
        partition_path = selected_rows_path / f"partition-{partition_id:05d}"
        partition_path.mkdir(parents=True)
        pl.DataFrame({
            "subject_uri": [uri],
            "post_created_at": [datetime(2026, 1, 1, tzinfo=UTC)],
            "author_did": [f"author-{uri}"],
            "embeddings": [_embedding(vector)],
        }).write_parquet(partition_path / "rows.parquet")

    stats = dataset_hydration_artifacts.write_embedding_shards(
        selected_embedding_rows_path=selected_rows_path,
        valid_embedding_rows_path=tmp_path / "valid-embedding-rows",
        embedding_shards_path=tmp_path / "embedding-shards",
        embedding_model="all_MiniLM_L12_v2",
        embedding_dim=2,
        partition_count=2,
        worker_count=2,
        logger=logging.getLogger("parallel-embedding-shard-test"),
    )

    assert stats["embedding_partition_worker_count"] == 2
    assert stats["embedding_partition_counts"] == [1, 1]
    assert [row["partition_id"] for row in stats["embedding_partition_stats"]] == [
        0,
        1,
    ]
    assert np.load(tmp_path / "embedding-shards" / "part-00000.npy").tolist() == [
        [1.0, 2.0]
    ]
    assert np.load(tmp_path / "embedding-shards" / "part-00001.npy").tolist() == [
        [3.0, 4.0]
    ]
    assert pl.read_parquet(
        tmp_path / "valid-embedding-rows" / "part-00000.parquet"
    ).get_column("subject_uri").to_list() == ["a"]
    assert pl.read_parquet(
        tmp_path / "valid-embedding-rows" / "part-00001.parquet"
    ).get_column("subject_uri").to_list() == ["b"]


def test_embedding_shard_copy_validates_values_during_the_copy():
    destination = np.zeros((2, 2), dtype=np.float32)
    dataset_hydration_artifacts._copy_finite_embedding_shard(
        shard=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        destination=destination,
        destination_offset=0,
    )
    assert destination.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    with np.testing.assert_raises_regex(ValueError, "non-finite"):
        dataset_hydration_artifacts._copy_finite_embedding_shard(
            shard=np.array([[1.0, np.nan]], dtype=np.float32),
            destination=destination,
            destination_offset=0,
        )


def test_prior_counts_read_each_usage_partition_once(tmp_path, monkeypatch):
    query_hour = datetime(2026, 1, 1, 2, tzinfo=UTC)
    route_paths = {
        "positive": tmp_path / "positive-routes",
        "history": tmp_path / "history-routes",
        "negative": tmp_path / "negative-routes",
    }
    route_frames = {
        "positive": pl.DataFrame({
            "did": ["user"],
            "query_hour": [query_hour],
            "subject_uri": ["shared"],
            "like_created_at": [query_hour],
        }),
        "history": pl.DataFrame({
            "did": ["user"],
            "query_hour": [query_hour],
            "_history_position": [0],
            "subject_uri": ["shared"],
            "like_created_at": [datetime(2026, 1, 1, 1, tzinfo=UTC)],
        }),
        "negative": pl.DataFrame({
            "query_hour": [query_hour],
            "subject_uri": ["negative"],
            "selection_source": ["random"],
            "_stage4_prior_like_count": [2],
        }),
    }
    route_parts = []
    for role, path in route_paths.items():
        partition_path = path / "_post_partition=0"
        partition_path.mkdir(parents=True)
        part = partition_path / "part.parquet"
        route_frames[role].write_parquet(part)
        route_parts.append(part)

    events_path = tmp_path / "post-liker-events"
    events_path.mkdir()
    event_part = events_path / "part-00000.parquet"
    pl.DataFrame({
        "subject_uri": ["shared", "shared", "shared", "negative", "negative"],
        "liker_did": ["a", "a", "b", "c", "d"],
        "like_created_at": [
            datetime(2026, 1, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 1, 1, tzinfo=UTC),
            query_hour,
            datetime(2026, 1, 1, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 1, tzinfo=UTC),
        ],
    }).write_parquet(event_part)
    unused_event_part = events_path / "part-00001.parquet"
    pl.DataFrame({
        "subject_uri": ["unused"],
        "liker_did": ["unused-liker"],
        "like_created_at": [datetime(2026, 1, 1, 1, tzinfo=UTC)],
    }).write_parquet(unused_event_part)
    use_windows_path = tmp_path / "post-liker-use-windows"
    use_windows_partition = use_windows_path / "_post_partition=0"
    use_windows_partition.mkdir(parents=True)
    pl.DataFrame({
        "subject_uri": ["shared", "negative"],
        "emb_idx": [0, 1],
        "final_use_query_hour": [query_hour, query_hour],
        "final_training_use_query_hour": [query_hour, query_hour],
    }, schema=post_liker_users.POST_LIKER_USE_WINDOW_SCHEMA).write_parquet(
        use_windows_partition / "part.parquet"
    )

    read_paths = []
    original_read_parquet = dataset_hydration_artifacts.pl.read_parquet

    def recording_read_parquet(paths, *args, **kwargs):
        input_paths = paths if isinstance(paths, list) else [paths]
        read_paths.extend(Path(path) for path in input_paths)
        return original_read_parquet(paths, *args, **kwargs)

    monkeypatch.setattr(
        dataset_hydration_artifacts.pl,
        "read_parquet",
        recording_read_parquet,
    )
    cumulative_build_count = 0
    original_build_cumulative = (
        dataset_hydration_artifacts.like_counts.build_cumulative_like_counts
    )

    def recording_build_cumulative(events):
        nonlocal cumulative_build_count
        cumulative_build_count += 1
        return original_build_cumulative(events)

    monkeypatch.setattr(
        dataset_hydration_artifacts.like_counts,
        "build_cumulative_like_counts",
        recording_build_cumulative,
    )

    stats = dataset_hydration_artifacts.attach_prior_counts(
        positive_routes_path=route_paths["positive"],
        history_routes_path=route_paths["history"],
        negative_routes_path=route_paths["negative"],
        post_liker_events_path=events_path,
        post_liker_use_windows_path=use_windows_path,
        post_liker_feature_events_path=tmp_path / "post-liker-feature-events",
        counted_positives_path=tmp_path / "counted-positives",
        counted_histories_path=tmp_path / "counted-histories",
        counted_negatives_path=tmp_path / "counted-negatives",
        partition_count=2,
        logger=logging.getLogger("prior-count-read-test"),
    )

    assert stats == {
        "post_query_hour_pair_count": 2,
        "raw_post_liker_event_count": 5,
        "deduplicated_post_liker_event_count": 4,
        "retained_post_liker_feature_event_count": 3,
        "training_visible_post_liker_event_count": 3,
    }
    assert all(read_paths.count(path) == 1 for path in route_parts)
    assert read_paths.count(event_part) == 1
    assert unused_event_part not in read_paths
    assert cumulative_build_count == 1
    assert original_read_parquet(
        tmp_path / "counted-positives" / "part-00000.parquet"
    ).get_column("prior_like_count").to_list() == [2]
    assert original_read_parquet(
        tmp_path / "counted-histories" / "part-00000.parquet"
    ).get_column("prior_like_count").to_list() == [2]
    assert original_read_parquet(
        tmp_path / "counted-negatives" / "part-00000.parquet"
    ).get_column("prior_like_count").to_list() == [2]
    feature_events = original_read_parquet(
        tmp_path / "post-liker-feature-events" / "part-00000.parquet"
    )
    assert feature_events.select(
        "emb_idx", "liker_did", "like_created_at", "is_training_visible"
    ).to_dicts() == [
        {
            "emb_idx": 0,
            "liker_did": "a",
            "like_created_at": datetime(2026, 1, 1, 1, tzinfo=UTC),
            "is_training_visible": True,
        },
        {
            "emb_idx": 1,
            "liker_did": "c",
            "like_created_at": datetime(2026, 1, 1, 0, tzinfo=UTC),
            "is_training_visible": True,
        },
        {
            "emb_idx": 1,
            "liker_did": "d",
            "like_created_at": datetime(2026, 1, 1, 1, tzinfo=UTC),
            "is_training_visible": True,
        },
    ]


def test_post_liker_use_windows_exclude_dropped_queries_and_track_train_horizon(
    tmp_path,
):
    train_hour = datetime(2026, 1, 1, 2, tzinfo=UTC)
    val_hour = datetime(2026, 1, 1, 3, tzinfo=UTC)
    dropped_hour = datetime(2026, 1, 1, 4, tzinfo=UTC)
    queries_lf = pl.DataFrame({
        "did": ["train-user", "val-user", "dropped-user"],
        "query_hour": [train_hour, val_hour, dropped_hour],
        "split": ["train", "val", "train"],
    }).lazy()
    positives_path = tmp_path / "hydrated-positives"
    histories_path = tmp_path / "hydrated-histories"
    negatives_path = tmp_path / "hydrated-negatives"
    _write_part(positives_path / "_post_partition=0", pl.DataFrame({
        "did": ["train-user", "val-user"],
        "query_hour": [train_hour, val_hour],
        "subject_uri": ["positive", "positive"],
        "emb_idx": pl.Series([0, 0], dtype=pl.UInt32),
    }))
    _write_part(histories_path / "_post_partition=0", pl.DataFrame({
        "did": ["train-user", "val-user", "dropped-user"],
        "query_hour": [train_hour, val_hour, dropped_hour],
        "subject_uri": ["shared-history", "shared-history", "dropped-history"],
        "emb_idx": pl.Series([1, 1, 2], dtype=pl.UInt32),
    }))
    _write_part(negatives_path / "_post_partition=0", pl.DataFrame({
        "query_hour": [train_hour, val_hour, dropped_hour],
        "subject_uri": ["negative", "negative", "dropped-negative"],
        "emb_idx": pl.Series([3, 3, 4], dtype=pl.UInt32),
    }))

    output_path = tmp_path / "use-windows"
    stats = dataset_hydration_artifacts.materialize_post_liker_use_windows(
        queries_lf=queries_lf,
        positive_routes_path=positives_path,
        history_routes_path=histories_path,
        negative_routes_path=negatives_path,
        output_path=output_path,
        partition_count=1,
    )

    windows = scan_parquet_artifact(output_path).collect().sort("emb_idx")
    assert stats == {"post_count": 3, "training_post_count": 3}
    assert windows.select(
        "subject_uri",
        "emb_idx",
        "final_use_query_hour",
        "final_training_use_query_hour",
    ).to_dicts() == [
        {
            "subject_uri": "positive",
            "emb_idx": 0,
            "final_use_query_hour": val_hour,
            "final_training_use_query_hour": train_hour,
        },
        {
            "subject_uri": "shared-history",
            "emb_idx": 1,
            "final_use_query_hour": val_hour,
            "final_training_use_query_hour": train_hour,
        },
        {
            "subject_uri": "negative",
            "emb_idx": 3,
            "final_use_query_hour": val_hour,
            "final_training_use_query_hour": train_hour,
        },
    ]


def test_post_liker_vocabulary_caps_support_and_maps_remaining_events_to_unk(
    tmp_path,
):
    feature_events_path = tmp_path / "feature-events"
    event_time = datetime(2026, 1, 1, 1, tzinfo=UTC)
    rows = []
    for emb_idx, liker_did, count, is_training_visible in (
        (0, "a", 3, True),
        (1, "b", 2, True),
        (2, "c", 2, True),
        (3, "validation-only", 1, False),
    ):
        rows.extend(
            {
                "emb_idx": emb_idx,
                "liker_did": liker_did,
                "like_created_at": datetime(
                    2026, 1, 1, 1, event_index, tzinfo=UTC
                ),
                "is_training_visible": is_training_visible,
            }
            for event_index in range(count)
        )
    _write_part(
        feature_events_path,
        pl.DataFrame(rows, schema=post_liker_users.POST_LIKER_FEATURE_EVENT_SCHEMA),
    )
    vocabulary_path = tmp_path / "post-liker-users"
    stats = dataset_hydration_artifacts.build_post_liker_user_vocabulary(
        feature_events_path=feature_events_path,
        support_routes_path=tmp_path / "support-routes",
        support_shards_path=tmp_path / "support-shards",
        vocabulary_path=vocabulary_path,
        min_training_event_count=2,
        max_vocabulary_size=2,
        partition_count=2,
        logger=logging.getLogger("post-liker-vocabulary-test"),
    )

    vocabulary = scan_parquet_artifact(vocabulary_path).collect()
    assert vocabulary.to_dicts() == [
        {"liker_did": "a", "liker_idx": 2, "training_event_count": 3},
        {"liker_did": "b", "liker_idx": 3, "training_event_count": 2},
    ]
    assert stats["threshold_eligible_user_count"] == 3
    assert stats["excluded_by_cap_user_count"] == 1
    assert stats["all_training_event_count"] == 7
    assert stats["known_training_event_count"] == 5
    assert stats["unk_training_event_count"] == 2

    other_vocabulary_path = tmp_path / "post-liker-users-one-partition"
    dataset_hydration_artifacts.build_post_liker_user_vocabulary(
        feature_events_path=feature_events_path,
        support_routes_path=tmp_path / "support-routes-one-partition",
        support_shards_path=tmp_path / "support-shards-one-partition",
        vocabulary_path=other_vocabulary_path,
        min_training_event_count=2,
        max_vocabulary_size=2,
        partition_count=1,
        logger=logging.getLogger("post-liker-vocabulary-partition-test"),
    )
    assert scan_parquet_artifact(other_vocabulary_path).collect().equals(vocabulary)

    indexed_path = tmp_path / "indexed-events"
    indexed_stats = dataset_hydration_artifacts.attach_post_liker_user_indices(
        feature_events_path=feature_events_path,
        vocabulary_path=vocabulary_path,
        indexed_events_path=indexed_path,
        partition_count=1,
    )
    indexed = scan_parquet_artifact(indexed_path).collect()
    assert indexed_stats == {
        "event_count": 8,
        "unknown_event_count": 3,
        "known_event_count": 5,
        "post_count": 4,
    }
    assert indexed.filter(pl.col("emb_idx") == 2).get_column("liker_idx").to_list() == [
        post_liker_users.POST_LIKER_USER_UNK_IDX,
        post_liker_users.POST_LIKER_USER_UNK_IDX,
    ]
    assert indexed.filter(pl.col("emb_idx") == 3).get_column("liker_idx").to_list() == [
        post_liker_users.POST_LIKER_USER_UNK_IDX
    ]


def test_post_liker_vocabulary_and_index_are_schema_correct_when_events_are_empty(
    tmp_path,
):
    feature_events_path = tmp_path / "feature-events"
    _write_part(
        feature_events_path,
        post_liker_users.empty_frame(
            post_liker_users.POST_LIKER_FEATURE_EVENT_SCHEMA
        ),
    )
    vocabulary_path = tmp_path / "post-liker-users"
    stats = dataset_hydration_artifacts.build_post_liker_user_vocabulary(
        feature_events_path=feature_events_path,
        support_routes_path=tmp_path / "support-routes",
        support_shards_path=tmp_path / "support-shards",
        vocabulary_path=vocabulary_path,
        min_training_event_count=2,
        max_vocabulary_size=10,
        partition_count=2,
        logger=logging.getLogger("empty-post-liker-vocabulary-test"),
    )
    assert stats["user_count"] == 0
    assert stats["user_table_num_rows"] == 2
    assert scan_parquet_artifact(vocabulary_path).collect_schema() == pl.Schema(
        post_liker_users.POST_LIKER_USER_VOCABULARY_SCHEMA
    )

    indexed_path = tmp_path / "indexed-events"
    indexed_stats = dataset_hydration_artifacts.attach_post_liker_user_indices(
        feature_events_path=feature_events_path,
        vocabulary_path=vocabulary_path,
        indexed_events_path=indexed_path,
        partition_count=1,
    )
    assert indexed_stats["event_count"] == 0
    assert scan_parquet_artifact(indexed_path).collect_schema() == pl.Schema(
        post_liker_users.INDEXED_POST_LIKER_EVENT_SCHEMA
    )


def test_author_vocabulary_uses_only_surviving_training_feature_occurrences(tmp_path):
    hour1 = datetime(2026, 1, 1, 1, tzinfo=UTC)
    hour2 = datetime(2026, 1, 1, 2, tzinfo=UTC)
    val_hour = datetime(2026, 1, 1, 3, tzinfo=UTC)
    dropped_hour = datetime(2026, 1, 1, 4, tzinfo=UTC)
    queries_lf = pl.DataFrame({
        "did": ["u1", "u2", "uv", "dropped"],
        "query_hour": [hour1, hour2, val_hour, dropped_hour],
        "split": ["train", "train", "val", "train"],
    }).lazy()
    positives_path = tmp_path / "positives"
    histories_path = tmp_path / "histories"
    negatives_path = tmp_path / "negatives"
    _write_part(positives_path, pl.DataFrame({
        "did": ["u1", "u2", "uv"],
        "query_hour": [hour1, hour2, val_hour],
        "author_did": ["a", "a", "validation-only"],
    }))
    _write_part(histories_path, pl.DataFrame({
        "did": ["u1", "u1", "u1", "uv", "dropped"],
        "query_hour": [hour1, hour1, hour1, val_hour, dropped_hour],
        "author_did": ["a", "b", "b", "validation-only", "dropped-only"],
    }))
    _write_part(negatives_path, pl.DataFrame({
        "query_hour": [hour1, hour2, val_hour, dropped_hour],
        "author_did": ["c", "a", "validation-only", "dropped-only"],
    }))

    stats = dataset_hydration_artifacts.build_author_vocabulary(
        queries_lf=queries_lf,
        counted_positives_path=positives_path,
        counted_histories_path=histories_path,
        counted_negatives_path=negatives_path,
        exposure_routes_path=tmp_path / "exposure-routes",
        eligible_shards_path=tmp_path / "eligible-shards",
        authors_path=tmp_path / "authors",
        min_training_feature_count=2,
        partition_count=3,
        logger=logging.getLogger("author-vocabulary-test"),
    )

    authors = scan_parquet_artifact(tmp_path / "authors").collect()
    assert authors.schema == pl.Schema(author_vocabulary.AUTHOR_VOCABULARY_SCHEMA)
    assert authors.to_dicts() == [
        {
            "author_did": "a",
            "author_idx": 2,
            "training_feature_count": 4,
            "training_positive_count": 2,
            "training_history_count": 1,
            "training_negative_count": 1,
        },
        {
            "author_did": "b",
            "author_idx": 3,
            "training_feature_count": 2,
            "training_positive_count": 0,
            "training_history_count": 2,
            "training_negative_count": 0,
        },
    ]
    assert stats["pre_threshold_author_count"] == 3
    assert stats["eligible_author_count"] == 2
    assert stats["training_feature_count"] == 7
    assert "validation-only" not in authors.get_column("author_did").to_list()
    assert "dropped-only" not in authors.get_column("author_did").to_list()


def test_publish_query_artifacts_drops_only_queries_without_surviving_positives(tmp_path):
    first_hour = datetime(2026, 1, 1, 12, tzinfo=UTC)
    second_hour = datetime(2026, 1, 1, 13, tzinfo=UTC)
    queries = pl.DataFrame({
        "did": ["u1", "u2"],
        "query_hour": [first_hour, second_hour],
        "user_cohort": ["seen", "seen"],
        "split": ["train", "train"],
        "positive_count": [1, 1],
    }, schema=dataset_hydration.QUERY_SCHEMA)
    counted_positives_path = tmp_path / "counted_positives"
    counted_positives_path.mkdir()
    pl.DataFrame({
        "did": ["u1"],
        "query_hour": [first_hour],
        "subject_uri": ["p1"],
        "like_created_at": [datetime(2026, 1, 1, 12, 5, tzinfo=UTC)],
        "emb_idx": [0],
        "post_created_at": [datetime(2026, 1, 1, 11, tzinfo=UTC)],
        "author_idx": [2],
        "prior_like_count": [3],
    }, schema=dataset_hydration.QUERY_POSITIVE_SCHEMA).write_parquet(
        counted_positives_path / "part-00000.parquet"
    )
    counted_histories_path = tmp_path / "counted_histories"
    counted_histories_path.mkdir()
    pl.DataFrame({
        "did": ["u1", "u1"],
        "query_hour": [first_hour, first_hour],
        "_history_position": [0, 1],
        "subject_uri": ["h1", "h1"],
        "like_created_at": [
            datetime(2026, 1, 1, 11, tzinfo=UTC),
            datetime(2026, 1, 1, 10, tzinfo=UTC),
        ],
        "emb_idx": [1, 1],
        "post_created_at": [
            datetime(2026, 1, 1, 9, tzinfo=UTC),
            datetime(2026, 1, 1, 9, tzinfo=UTC),
        ],
        "author_idx": [3, 3],
        "prior_like_count": [5, 5],
    }, schema={
        "did": pl.String,
        "query_hour": dataset_hydration.UTC_DATETIME,
        "_history_position": pl.UInt32,
        "subject_uri": pl.String,
        "like_created_at": dataset_hydration.UTC_DATETIME,
        "emb_idx": pl.UInt32,
        "post_created_at": dataset_hydration.UTC_DATETIME,
        "author_idx": pl.UInt32,
        "prior_like_count": pl.UInt64,
    }).write_parquet(counted_histories_path / "part-00000.parquet")

    stats = dataset_hydration_artifacts.publish_query_artifacts(
        queries_lf=queries.lazy(),
        counted_positives_path=counted_positives_path,
        counted_histories_path=counted_histories_path,
        queries_path=tmp_path / "queries",
        query_positives_path=tmp_path / "query_positives",
        query_histories_path=tmp_path / "query_histories",
        staging_path=tmp_path / "staging",
        partition_count=2,
    )

    hydrated_queries = scan_parquet_artifact(tmp_path / "queries").collect()
    hydrated_histories = scan_parquet_artifact(tmp_path / "query_histories").collect()
    assert hydrated_queries.select("did", "query_hour", "positive_count").to_dicts() == [{
        "did": "u1",
        "query_hour": first_hour,
        "positive_count": 1,
    }]
    assert hydrated_histories.get_column("history_subject_uris").to_list() == [["h1", "h1"]]
    assert hydrated_histories.get_column("history_emb_indices").to_list() == [[1, 1]]
    assert stats["input_query_count"] == 2
    assert stats["retained_query_count"] == 1
    assert stats["dropped_zero_positive_query_count"] == 1
    assert stats["retained_query_hours"] == [first_hour]
