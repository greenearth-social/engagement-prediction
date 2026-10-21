from datetime import datetime, timezone
import base64
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import zlib

import numpy as np
import polars as pl
import pytest

from engagement_prediction.data import author_statistics
from engagement_prediction.data import dataset_hydration
from engagement_prediction.data import dataset_hydration_artifacts
from engagement_prediction.data import ingex
from engagement_prediction.data import negative_selection
from engagement_prediction.data import post_liker_history
from engagement_prediction.data import post_selection
from engagement_prediction.data import training_index
from engagement_prediction.data import user_history
from engagement_prediction.data.parquet import scan_parquet_artifact
from engagement_prediction.pipeline import registry
from engagement_prediction.pipeline.artifacts import PartialArtifactBundle
from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline import logging as pipeline_logging


UTC = timezone.utc


def _encoded_vector(value: float) -> str:
    payload = struct.pack("<384f", *([value] * 384))
    return base64.b85encode(zlib.compress(payload)).decode()


def _manifest(prefix: str, path: Path) -> dict:
    return ingex.build_source_manifest(
        gcs_bucket="unused",
        blob_prefix=prefix,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        paths=[str(path)],
        timestamps=[datetime(2026, 1, 1, tzinfo=UTC)],
    )


def _write_manifest(path: Path, stage_folder: str, inputs: dict[str, str]) -> None:
    (path / "manifest.json").write_text(json.dumps({
        "stage_folder": stage_folder,
        "inputs": inputs,
    }) + "\n")


def _write_dataset(path: Path, df: pl.DataFrame) -> None:
    path.mkdir(parents=True)
    df.write_parquet(path / "part-00000.parquet")


def _upstream(tmp_path: Path) -> Path:
    hour = datetime(2026, 1, 1, 12, tzinfo=UTC)
    posts_file = tmp_path / "raw-posts.parquet"
    replies_file = tmp_path / "raw-replies.parquet"
    likes_file = tmp_path / "raw-likes.parquet"
    embeddings = lambda value: [{"key": "all_MiniLM_L12_v2", "value": _encoded_vector(value)}]
    pl.DataFrame({
        "at_uri": ["positive", "negative", "missing-embedding"],
        "record_created_at": [
            "2026-01-01T11:10:00Z",
            "2026-01-01T11:20:00Z",
            "2026-01-01T10:00:00Z",
        ],
        "did": ["a", "b", "a"],
        "embeddings": [embeddings(1.0), embeddings(2.0), None],
    }).write_parquet(posts_file)
    pl.DataFrame({
        "at_uri": ["reply"],
        "record_created_at": ["2026-01-01T09:00:00Z"],
        "did": ["c"],
        "embeddings": [embeddings(3.0)],
    }).write_parquet(replies_file)
    pl.DataFrame({
        "did": ["liker", "liker", "liker"],
        "subject_uri": ["positive", "negative", "negative"],
        "record_created_at": [
            "2026-01-01T10:00:00Z",
            "2026-01-01T11:00:00Z",
            "2026-01-01T12:00:00Z",
        ],
    }).write_parquet(likes_file)
    post_manifest = _manifest("bsky_posts", posts_file)
    reply_manifest = _manifest("bsky_replies", replies_file)
    like_manifest = _manifest("bsky_likes", likes_file)

    root = tmp_path / "artifacts"
    source_metadata = root / "00_source_metadata" / "s0"
    source_bundle = source_metadata / "source_metadata_s0"
    _write_dataset(source_bundle / "post_metadata", pl.DataFrame({
        "subject_uri": ["missing-embedding", "negative", "positive", "reply"],
        "post_created_at": [
            datetime(2026, 1, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 1, 11, 20, tzinfo=UTC),
            datetime(2026, 1, 1, 11, 10, tzinfo=UTC),
            datetime(2026, 1, 1, 9, tzinfo=UTC),
        ],
        "author_did": ["a", "b", "a", "c"],
        "is_reply": [False, False, False, True],
    }, schema=post_selection.POST_SCHEMA))
    ingex.write_source_manifest(source_bundle / "post_sources_s0.json", post_manifest)
    ingex.write_source_manifest(source_bundle / "reply_sources_s0.json", reply_manifest)
    (source_metadata / "summary.json").write_text(json.dumps({
        "parameters": {"source_metadata_partition_count": 1},
        "index": {
            "root_source_stats": {},
            "reply_source_stats": {},
            "root_reply_overlap_count": 0,
        },
    }))
    _write_manifest(source_metadata, "00_source_metadata", {})
    stage1 = root / "01_query_selection" / "s1"
    stage1.mkdir(parents=True)
    pl.DataFrame({
        "did": ["user"],
        "query_hour": [hour],
        "user_cohort": ["seen"],
        "split": ["train"],
        "positive_count": [1],
    }, schema=dataset_hydration.QUERY_SCHEMA).write_parquet(stage1 / "queries_s1.parquet")
    pl.DataFrame({
        "did": ["user"],
        "query_hour": [hour],
        "subject_uri": ["positive"],
        "like_created_at": [datetime(2026, 1, 1, 12, 30, tzinfo=UTC)],
    }).write_parquet(stage1 / "query_positives_s1.parquet")
    lineage = {"00_source_metadata": str(source_metadata.resolve())}
    _write_manifest(stage1, "01_query_selection", lineage)

    stage2 = root / "02_user_history" / "s2"
    stage2.mkdir(parents=True)
    _write_dataset(stage2 / "query_histories_s2", pl.DataFrame({
        "did": ["user"],
        "query_hour": [hour],
        "history_subject_uris": [["reply", "missing-embedding"]],
        "history_like_created_ats": [[
            datetime(2026, 1, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 1, 9, tzinfo=UTC),
        ]],
    }, schema=user_history.HISTORY_SCHEMA))
    (stage2 / "summary.json").write_text(json.dumps({
        "parameters": {"user_history_partition_count": 1},
    }))
    _write_manifest(stage2, "02_user_history", {
        **lineage,
        "01_query_selection": str(stage1.resolve()),
    })

    stage3 = root / "03_post_selection" / "s3"
    bundle3 = stage3 / "post_universe_s3"
    bundle3.mkdir(parents=True)
    _write_dataset(bundle3 / "posts", pl.DataFrame({
        "subject_uri": ["missing-embedding", "negative", "positive", "reply"],
        "post_created_at": [
            datetime(2026, 1, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 1, 11, 20, tzinfo=UTC),
            datetime(2026, 1, 1, 11, 10, tzinfo=UTC),
            datetime(2026, 1, 1, 9, tzinfo=UTC),
        ],
        "author_did": ["a", "b", "a", "c"],
        "is_reply": [False, False, False, True],
    }, schema=post_selection.POST_SCHEMA))
    ingex.write_source_manifest(bundle3 / "post_sources_s3.json", post_manifest)
    ingex.write_source_manifest(bundle3 / "reply_sources_s3.json", reply_manifest)
    (stage3 / "summary.json").write_text(json.dumps({
        "parameters": {"source_metadata_partition_count": 1},
    }))
    _write_manifest(stage3, "03_post_selection", {
        **lineage,
        "01_query_selection": str(stage1.resolve()),
        "02_user_history": str(stage2.resolve()),
    })

    stage4 = root / "04_negative_selection" / "s4"
    bundle4 = stage4 / "negative_candidates_s4"
    bundle4.mkdir(parents=True)
    _write_dataset(bundle4 / "hourly_candidates", pl.DataFrame({
        "query_hour": [hour],
        "subject_uri": ["negative"],
        "selection_source": ["popular"],
        "prior_like_count": [1],
    }, schema=negative_selection.HOURLY_CANDIDATE_SCHEMA))
    _write_manifest(stage4, "04_negative_selection", {
        **lineage,
        "01_query_selection": str(stage1.resolve()),
        "02_user_history": str(stage2.resolve()),
        "03_post_selection": str(stage3.resolve()),
    })

    stage5 = root / "05_post_liker_history" / "s5"
    bundle5 = stage5 / "post_liker_histories_s5"
    bundle5.mkdir(parents=True)
    _write_dataset(bundle5 / "post_liker_events", pl.DataFrame({
        "subject_uri": ["negative", "negative", "positive"],
        "liker_did": ["liker", "liker", "liker"],
        "like_created_at": [
            datetime(2026, 1, 1, 11, tzinfo=UTC),
            hour,
            datetime(2026, 1, 1, 10, tzinfo=UTC),
        ],
    }, schema=post_liker_history.POST_LIKER_EVENT_SCHEMA))
    _write_dataset(bundle5 / "post_liker_posts", pl.DataFrame({
        "subject_uri": ["missing-embedding", "negative", "positive", "reply"],
        "is_positive": [False, False, True, False],
        "is_history": [True, False, False, True],
        "is_negative": [False, True, False, False],
        "like_event_count": [0, 2, 1, 0],
        "first_like_created_at": [None, datetime(2026, 1, 1, 11, tzinfo=UTC), datetime(2026, 1, 1, 10, tzinfo=UTC), None],
        "last_like_created_at": [None, hour, datetime(2026, 1, 1, 10, tzinfo=UTC), None],
    }, schema=post_liker_history.POST_LIKER_POST_SCHEMA))
    ingex.write_source_manifest(bundle5 / "like_sources_s5.json", like_manifest)
    (stage5 / "summary.json").write_text(json.dumps({
        "parameters": {"post_liker_history_partition_count": 1},
    }))
    _write_manifest(stage5, "05_post_liker_history", {
        **lineage,
        "01_query_selection": str(stage1.resolve()),
        "02_user_history": str(stage2.resolve()),
        "03_post_selection": str(stage3.resolve()),
        "04_negative_selection": str(stage4.resolve()),
    })

    stage6 = root / "06_author_statistics" / "s6"
    bundle6 = stage6 / "author_statistics_s6"
    bundle6.mkdir(parents=True)
    _write_dataset(
        bundle6 / "author_statistics",
        author_statistics.empty_frame(author_statistics.AUTHOR_STAT_SCHEMA),
    )
    ingex.write_source_manifest(bundle6 / "post_sources_s6.json", post_manifest)
    ingex.write_source_manifest(bundle6 / "reply_sources_s6.json", reply_manifest)
    ingex.write_source_manifest(bundle6 / "like_sources_s6.json", like_manifest)
    (stage6 / "summary.json").write_text(json.dumps({
        "parameters": {"author_statistics_partition_count": 1},
    }))
    _write_manifest(stage6, "06_author_statistics", {
        **lineage,
        "01_query_selection": str(stage1.resolve()),
        "02_user_history": str(stage2.resolve()),
        "03_post_selection": str(stage3.resolve()),
        "04_negative_selection": str(stage4.resolve()),
        "05_post_liker_history": str(stage5.resolve()),
    })
    return stage6


@pytest.mark.parametrize("max_liker_vocabulary_size", [100, 0])
def test_stage_hydrates_memmap_filters_missing_embeddings_and_counts_as_of(
    tmp_path,
    monkeypatch,
    max_liker_vocabulary_size,
):
    stage6 = _upstream(tmp_path)
    monkeypatch.setattr(
        ingex,
        "list_ingex_parquet_files",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Stage 7 must not relist Ingex")
        ),
    )
    publish_embeddings_and_post_metadata = (
        dataset_hydration_artifacts.publish_embeddings_and_post_metadata
    )

    def publish_after_releasing_intermediates(**kwargs):
        staging_root = kwargs["embedding_shards_path"].parent
        assert not (staging_root / "selected_embedding_rows").exists()
        result = publish_embeddings_and_post_metadata(**kwargs)
        assert not kwargs["embedding_shards_path"].exists()
        return result

    monkeypatch.setattr(
        dataset_hydration_artifacts,
        "publish_embeddings_and_post_metadata",
        publish_after_releasing_intermediates,
    )
    staging_roots = []
    observed_phases = []

    def observe_cleanup_before(function_name, retired_names):
        original = getattr(dataset_hydration_artifacts, function_name)

        def run_after_cleanup(**kwargs):
            if not staging_roots:
                staging_roots.append(kwargs["selected_metadata_path"].parent)
            for name in retired_names:
                assert not (staging_roots[0] / name).exists(), (function_name, name)
            observed_phases.append(function_name)
            return original(**kwargs)

        monkeypatch.setattr(dataset_hydration_artifacts, function_name, run_after_cleanup)

    # Inspect actual phase boundaries while running every data transformation.
    # End-of-stage cleanup alone must not satisfy these peak-disk assertions.
    phase_retirements = {
        "materialize_selected_embedding_rows": ("selected_post_routes",),
        "materialize_usage_routes": (
            "selected_metadata", "valid_embedding_rows", "embedding_shards",
        ),
        "route_hydrated_usage_for_counts": (
            "positive_routes", "history_routes", "negative_routes",
        ),
        "materialize_post_liker_use_windows": (
            "hydrated_positives", "hydrated_histories", "hydrated_negatives",
        ),
        "build_post_liker_user_vocabulary": (
            "liker_positive_routes", "liker_history_routes", "liker_negative_routes",
            "post_liker_use_windows",
        ),
        "attach_post_liker_user_indices": (
            "post_liker_support_routes", "post_liker_support_shards",
            "selected_liker_support.parquet",
        ),
        "build_author_vocabulary": ("post_liker_feature_events",),
        "publish_posts_with_author_indices": (
            "author_exposure_routes", "eligible_author_shards",
        ),
        "attach_author_indices_to_usage": ("hydrated_post_metadata",),
        "publish_query_artifacts": (
            "counted_positives", "counted_histories", "counted_negatives",
        ),
        "publish_negative_candidates": (
            "indexed_positives", "indexed_histories", "query_publication",
        ),
        "validate_public_bundle": ("indexed_negatives",),
    }
    for function_name, retired_names in phase_retirements.items():
        observe_cleanup_before(function_name, retired_names)

    build_loader_index = training_index.build_loader_index

    def build_with_only_required_staging(**kwargs):
        indexed_events_path = kwargs["indexed_post_liker_events_path"]
        assert set(indexed_events_path.parent.iterdir()) == {indexed_events_path}
        assert list(indexed_events_path.glob("*.parquet"))
        return build_loader_index(**kwargs)

    monkeypatch.setattr(training_index, "build_loader_index", build_with_only_required_staging)
    publish = PartialArtifactBundle.publish

    def publish_after_loader_inputs_released(self, **kwargs):
        assert not list(self.staging_path.iterdir())
        assert (self.partial_path / "loader_index" / "format.json").is_file()
        return publish(self, **kwargs)

    monkeypatch.setattr(PartialArtifactBundle, "publish", publish_after_loader_inputs_released)
    logger = pipeline_logging._stage_loggers.pop("07_DATASET_HYDRATION", None)
    if logger:
        for handler in logger.handlers:
            handler.close()
    context = Context(
        run_dir=tmp_path / "runs" / "run",
        artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "runs",
        pipeline_run_id="run",
    )
    context.prior_outputs["06_author_statistics"] = stage6

    result = registry.run_stage(
        "dataset_hydration",
        context,
        SimpleNamespace(
            embedding_model="all_MiniLM_L12_v2",
            embedding_source_batch_size=64,
            dataset_hydration_worker_count=2,
            min_author_training_feature_count=1,
            min_post_liker_user_training_event_count=1,
            max_post_liker_user_vocabulary_size=max_liker_vocabulary_size,
            _argv=["--stop-after", "dataset_hydration"],
        ),
    )

    mmap = np.load(result["artifacts"]["embeddings_path"], mmap_mode="r")
    assert mmap.shape == (3, 384)
    posts = scan_parquet_artifact(Path(result["artifacts"]["posts_path"])).collect()
    assert set(posts["subject_uri"]) == {"positive", "negative", "reply"}
    assert posts["emb_idx"].sort().to_list() == [0, 1, 2]
    expected_values = {"positive": 1.0, "negative": 2.0, "reply": 3.0}
    for row in posts.iter_rows(named=True):
        assert mmap[row["emb_idx"], 0] == expected_values[row["subject_uri"]]
    positives = scan_parquet_artifact(Path(result["artifacts"]["query_positives_path"])).collect()
    assert positives["prior_like_count"].to_list() == [1]
    histories = scan_parquet_artifact(Path(result["artifacts"]["query_histories_path"])).collect()
    assert histories.columns == dataset_hydration.QUERY_HISTORY_COLUMNS
    assert "history_subject_uris" not in histories.columns
    assert "history_author_indices" not in histories.columns
    reply_emb_idx = posts.filter(pl.col("subject_uri") == "reply")["emb_idx"].item()
    assert histories["history_emb_indices"].to_list() == [[reply_emb_idx]]
    negatives = scan_parquet_artifact(Path(result["artifacts"]["hourly_negative_candidates_path"])).collect()
    assert negatives["prior_like_count"].to_list() == [1]
    authors = scan_parquet_artifact(Path(result["artifacts"]["authors_path"])).collect()
    assert authors.select("author_did", "author_idx").to_dicts() == [
        {"author_did": "a", "author_idx": 2},
        {"author_did": "b", "author_idx": 3},
        {"author_did": "c", "author_idx": 4},
    ]
    summary = json.loads(Path(result["output_dir"], "summary.json").read_text())
    assert summary["parameters"]["dataset_hydration_worker_count"] == 2
    assert summary["embedding_sources"]["embedding_source_worker_count"] == 2
    assert summary["embeddings"]["embedding_partition_worker_count"] == 1
    assert summary["parameters"]["min_author_training_feature_count"] == 1
    assert summary["parameters"]["min_post_liker_user_training_event_count"] == 1
    assert summary["parameters"]["max_post_liker_user_vocabulary_size"] == max_liker_vocabulary_size
    assert summary["author_vocabulary"]["training_positive_count"] == 1
    assert summary["author_vocabulary"]["training_history_count"] == 1
    assert summary["author_vocabulary"]["training_negative_count"] == 1
    assert summary["author_index_usage_by_split"]["train"] == {
        "positive_feature_count": 1,
        "positive_known_count": 1,
        "positive_unk_count": 0,
        "history_feature_count": 1,
        "history_known_count": 1,
        "history_unk_count": 0,
        "negative_feature_count": 1,
        "negative_known_count": 1,
        "negative_unk_count": 0,
    }
    loader_index_path = Path(result["artifacts"]["loader_index_path"])
    loader_format = json.loads((loader_index_path / "format.json").read_text())
    assert loader_format["format_version"] == training_index.FORMAT_VERSION
    assert summary["loader_index"]["format_version"] == training_index.FORMAT_VERSION
    assert loader_format["post_liker_event_count"] == 2
    assert loader_format["post_liker_user_table_num_rows"] == (3 if max_liker_vocabulary_size else 2)
    post_liker_vocabulary = scan_parquet_artifact(
        Path(result["artifacts"]["post_liker_users_path"])
    ).collect()
    expected_liker_vocabulary = [
        {"liker_did": "liker", "liker_idx": 2, "training_event_count": 2}
    ] if max_liker_vocabulary_size else []
    assert post_liker_vocabulary.to_dicts() == expected_liker_vocabulary
    assert summary["loader_index"]["splits"]["train"] == {
        "query_count": 1,
        "history_count": 1,
        "positive_count": 1,
        "hour_count": 1,
        "negative_count": 1,
    }
    assert "loader_index_path: loader_index" in Path(
        result["output_dir"], "stage_info.txt"
    ).read_text()
    assert ingex.load_source_manifest(
        Path(result["artifacts"]["post_sources_path"])
    )["blob_prefix"] == "bsky_posts"
    assert ingex.load_source_manifest(
        Path(result["artifacts"]["reply_sources_path"])
    )["blob_prefix"] == "bsky_replies"
    assert ingex.load_source_manifest(
        Path(result["artifacts"]["like_sources_path"])
    )["blob_prefix"] == "bsky_likes"
    manifest = json.loads(Path(result["output_dir"], "manifest.json").read_text())
    assert set(manifest["inputs"]) == {
        "00_source_metadata",
        "01_query_selection",
        "02_user_history",
        "03_post_selection",
        "04_negative_selection",
        "05_post_liker_history",
        "06_author_statistics",
    }
    assert not list(Path(result["output_dir"]).glob("*.partial"))
    assert observed_phases == list(phase_retirements)


@pytest.mark.parametrize(("function_name", "required_arguments"), [
    ("write_embedding_shards", ("selected_embedding_rows_path",)),
    ("attach_prior_counts", (
        "positive_routes_path", "history_routes_path", "negative_routes_path",
        "post_liker_use_windows_path",
    )),
    ("publish_query_artifacts", ("counted_positives_path", "counted_histories_path")),
])
def test_stage_failure_retains_partial_diagnostics_without_publishing(
    tmp_path, monkeypatch, function_name, required_arguments,
):
    stage6 = _upstream(tmp_path)
    logger = pipeline_logging._stage_loggers.pop("07_DATASET_HYDRATION", None)
    if logger:
        for handler in logger.handlers:
            handler.close()
    context = Context(
        run_dir=tmp_path / "runs" / "run",
        artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "runs",
        pipeline_run_id="run",
    )
    context.prior_outputs["06_author_statistics"] = stage6

    required_paths = []

    def fail_consumer(**kwargs):
        required_paths.extend(kwargs[name] for name in required_arguments)
        assert all(list(path.rglob("*.parquet")) for path in required_paths)
        raise RuntimeError("injected consumer failure")

    monkeypatch.setattr(
        dataset_hydration_artifacts,
        function_name,
        fail_consumer,
    )

    with pytest.raises(RuntimeError, match="injected consumer failure"):
        registry.run_stage(
            "dataset_hydration",
            context,
            SimpleNamespace(
                embedding_model="all_MiniLM_L12_v2",
                embedding_source_batch_size=64,
                dataset_hydration_worker_count=2,
                min_author_training_feature_count=1,
                min_post_liker_user_training_event_count=1,
                max_post_liker_user_vocabulary_size=100,
                _argv=["--stop-after", "dataset_hydration"],
            ),
        )

    stage7_dirs = list((tmp_path / "artifacts" / "07_dataset_hydration").iterdir())
    assert len(stage7_dirs) == 1
    stage7_dir = stage7_dirs[0]
    assert not (stage7_dir / "manifest.json").exists()
    assert not [
        path
        for path in stage7_dir.glob("hydrated_training_data_*")
        if not path.name.endswith(".partial")
    ]
    assert list(stage7_dir.glob("hydrated_training_data_*.partial"))
    assert required_paths
    assert all(list(path.rglob("*.parquet")) for path in required_paths)


def test_loader_index_failure_keeps_outer_bundle_partial(tmp_path, monkeypatch):
    stage6 = _upstream(tmp_path)
    logger = pipeline_logging._stage_loggers.pop("07_DATASET_HYDRATION", None)
    if logger:
        for handler in logger.handlers:
            handler.close()
    context = Context(
        run_dir=tmp_path / "runs" / "run",
        artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "runs",
        pipeline_run_id="run",
    )
    context.prior_outputs["06_author_statistics"] = stage6

    loader_input_paths = []

    def fail_loader_index(**kwargs):
        loader_input_paths.extend([
            kwargs["posts_path"],
            kwargs["query_histories_path"],
            kwargs["indexed_post_liker_events_path"],
        ])
        assert all(list(path.rglob("*.parquet")) for path in loader_input_paths)
        raise RuntimeError("injected loader-index failure")

    monkeypatch.setattr(training_index, "build_loader_index", fail_loader_index)

    with pytest.raises(RuntimeError, match="injected loader-index failure"):
        registry.run_stage(
            "dataset_hydration",
            context,
            SimpleNamespace(
                embedding_model="all_MiniLM_L12_v2",
                embedding_source_batch_size=64,
                dataset_hydration_worker_count=2,
                min_author_training_feature_count=1,
                min_post_liker_user_training_event_count=1,
                max_post_liker_user_vocabulary_size=100,
                _argv=["--stop-after", "dataset_hydration"],
            ),
        )

    stage7_dirs = list((tmp_path / "artifacts" / "07_dataset_hydration").iterdir())
    assert len(stage7_dirs) == 1
    stage7_dir = stage7_dirs[0]
    assert not (stage7_dir / "manifest.json").exists()
    assert not [
        path
        for path in stage7_dir.glob("hydrated_training_data_*")
        if not path.name.endswith(".partial")
    ]
    assert list(stage7_dir.glob("hydrated_training_data_*.partial"))
    assert loader_input_paths
    assert all(list(path.rglob("*.parquet")) for path in loader_input_paths)


def test_stage_rejects_obsolete_indexed_stage6_schema(tmp_path):
    stage6 = _upstream(tmp_path)
    author_statistics_path = stage6 / "author_statistics_s6" / "author_statistics"
    for part in author_statistics_path.glob("*.parquet"):
        part.unlink()
    pl.DataFrame({
        "author_did": ["old-author"],
        "author_idx": pl.Series([2], dtype=pl.UInt32),
    }).write_parquet(author_statistics_path / "part-00000.parquet")
    context = Context(
        run_dir=tmp_path / "runs" / "old-stage6",
        artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "runs",
        pipeline_run_id="old-stage6",
    )
    context.prior_outputs["06_author_statistics"] = stage6

    with pytest.raises(ValueError, match="obsolete indexed/filtered author schema"):
        registry.run_stage(
            "dataset_hydration",
            context,
            SimpleNamespace(
                embedding_model="all_MiniLM_L12_v2",
                embedding_source_batch_size=64,
                dataset_hydration_worker_count=2,
                min_author_training_feature_count=1,
                min_post_liker_user_training_event_count=1,
                max_post_liker_user_vocabulary_size=100,
                _argv=["--stop-after", "dataset_hydration"],
            ),
        )
