from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from engagement_prediction.data import ingex
from engagement_prediction.data.parquet import scan_parquet_artifact
from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline import registry
from engagement_prediction.stages import user_history as stage


UTC = timezone.utc


def _write_query_selection_artifact(tmp_path: Path) -> tuple[Path, Path]:
    source_metadata_dir = tmp_path / "artifacts" / "00_source_metadata" / "stage0"
    source_metadata_dir.mkdir(parents=True)
    (source_metadata_dir / "manifest.json").write_text(json.dumps({
        "stage_key": "source_metadata",
        "stage_folder": "00_source_metadata",
        "inputs": {},
    }) + "\n")
    source_path = tmp_path / "likes.parquet"
    pl.DataFrame({
        "did": ["u1", "u1", "u1", "u1", "u1", "u2", "u2", "u3"],
        "subject_uri": [
            "outside-source-window",
            "history-unselected",
            "history-unselected",
            "target-one",
            "target-two",
            "u2-pre-validation",
            "u2-target",
            "u3-target",
        ],
        "record_created_at": [
            "2026-01-01T07:30:00Z",
            "2026-01-01T09:00:00Z",
            "2026-01-01T09:00:00Z",
            "2026-01-01T10:05:00Z",
            "2026-01-01T12:05:00Z",
            "2026-01-01T08:00:00Z",
            "2026-01-01T10:05:00Z",
            "2026-01-01T10:05:00Z",
        ],
    }).write_parquet(source_path)
    pl.DataFrame({
        "did": ["u1"],
        "subject_uri": ["not-in-recorded-snapshot"],
        "record_created_at": ["2026-01-01T09:30:00Z"],
    }).write_parquet(tmp_path / "unrecorded-likes.parquet")

    stage_dir = tmp_path / "artifacts" / "01_query_selection" / "stage1"
    stage_dir.mkdir(parents=True)
    queries_path = stage_dir / "queries_stage1.parquet"
    pl.DataFrame({
        "did": ["u1", "u2", "u3", "u1"],
        "query_hour": [
            datetime(2026, 1, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 1, 12, tzinfo=UTC),
        ],
        "user_cohort": ["trainval", "unseen_eval", "unseen_eval", "trainval"],
        "split": ["train", "val_unseen_users", "val_unseen_users", "train"],
        "positive_count": pl.Series([1, 1, 1, 1], dtype=pl.UInt32),
    }).write_parquet(queries_path)
    source_manifest = ingex.build_source_manifest(
        gcs_bucket="unused",
        blob_prefix="bsky_likes",
        start=datetime(2026, 1, 1, 8, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        paths=[str(source_path)],
        timestamps=[datetime(2026, 1, 1, 8, tzinfo=UTC)],
    )
    ingex.write_source_manifest(stage_dir / "like_sources_stage1.json", source_manifest)
    (stage_dir / "manifest.json").write_text(json.dumps({
        "stage_key": "query_selection",
        "stage_folder": "01_query_selection",
        "inputs": {"00_source_metadata": str(source_metadata_dir.resolve())},
    }) + "\n")
    return stage_dir, source_path


def _context(tmp_path: Path, stage1_dir: Path) -> Context:
    context = Context(
        run_dir=tmp_path / "runs" / "run",
        artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "runs",
        pipeline_run_id="run",
    )
    context.prior_outputs["01_query_selection"] = stage1_dir
    return context


def _args(partition_count=4):
    return SimpleNamespace(
        max_history_posts_per_query=64,
        user_history_partition_count=partition_count,
        _argv=["--start-from", "user_history", "--stop-after", "user_history"],
    )


def test_registry_run_writes_query_conditioned_histories_and_lineage(tmp_path):
    stage1_dir, _ = _write_query_selection_artifact(tmp_path)
    result = registry.run_stage("user_history", _context(tmp_path, stage1_dir), _args())

    output_dir = Path(result["output_dir"])
    history_path = Path(result["artifacts"]["query_histories_path"])
    history_post_uris_path = Path(result["artifacts"]["history_post_uris_path"])
    histories = scan_parquet_artifact(history_path).collect().sort(["query_hour", "did"])
    history_post_uris = (
        scan_parquet_artifact(history_post_uris_path).collect().sort("subject_uri")
    )
    assert histories.columns == [
        "did",
        "query_hour",
        "history_subject_uris",
        "history_like_created_ats",
    ]
    assert histories.height == 4
    assert histories["history_subject_uris"].to_list() == [
        ["history-unselected"],
        ["u2-pre-validation"],
        [],
        ["target-one", "history-unselected"],
    ]
    assert history_post_uris.schema == pl.Schema(stage.history_data.HISTORY_POST_URI_SCHEMA)
    assert history_post_uris["subject_uri"].to_list() == [
        "history-unselected",
        "target-one",
        "u2-pre-validation",
    ]
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["inputs"] == {
        "00_source_metadata": str(
            (tmp_path / "artifacts" / "00_source_metadata" / "stage0").resolve()
        ),
        "01_query_selection": str(stage1_dir.resolve()),
    }
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["outputs"]["query_count"] == 4
    assert summary["outputs"]["history_post_uris_path"] == history_post_uris_path.name
    assert summary["outputs"]["unique_history_post_count"] == 3
    assert summary["selection_stats_by_split"]["val_unseen_users"]["empty_history_count"] == 1
    stage_info = (output_dir / "stage_info.txt").read_text()
    assert f"history_post_uris_path: {history_post_uris_path.name}" in stage_info
    assert "unique_history_post_count: 3" in stage_info
    assert list(output_dir.glob("query_histories_*.partial")) == []
    assert list(output_dir.glob("history_post_uris_*.partial")) == []
    assert list(output_dir.glob("_history_post_uri_*")) == []
    assert list(output_dir.glob("_query_partitions_*")) == []
    assert list(output_dir.glob("_queried_user_likes_*")) == []


def test_logical_output_is_independent_of_stage_partition_count(tmp_path):
    stage1_dir, _ = _write_query_selection_artifact(tmp_path)
    first = registry.run_stage("user_history", _context(tmp_path, stage1_dir), _args(1))
    second = registry.run_stage("user_history", _context(tmp_path, stage1_dir), _args(7))

    first_df = scan_parquet_artifact(Path(first["artifacts"]["query_histories_path"])).collect().sort(
        ["query_hour", "did"]
    )
    second_df = scan_parquet_artifact(Path(second["artifacts"]["query_histories_path"])).collect().sort(
        ["query_hour", "did"]
    )
    first_history_post_uris = (
        scan_parquet_artifact(Path(first["artifacts"]["history_post_uris_path"]))
        .collect()
        .sort("subject_uri")
    )
    second_history_post_uris = (
        scan_parquet_artifact(Path(second["artifacts"]["history_post_uris_path"]))
        .collect()
        .sort("subject_uri")
    )
    assert first_df.equals(second_df)
    assert first_history_post_uris.equals(second_history_post_uris)


def test_history_post_uri_publication_deduplicates_across_user_partition_shards(tmp_path):
    shards_path = tmp_path / "shards.partial"
    shards_path.mkdir()
    pl.DataFrame({"subject_uri": ["only-a", "shared"]}).write_parquet(
        shards_path / "part-00000.parquet"
    )
    pl.DataFrame({"subject_uri": ["only-b", "shared"]}).write_parquet(
        shards_path / "part-00001.parquet"
    )
    routed_path = tmp_path / "routed.partial"
    output_path = tmp_path / "history_post_uris.partial"

    unique_count = stage._write_history_post_uris(
        history_post_uri_shards_path=shards_path,
        routed_history_post_uris_path=routed_path,
        partial_output_path=output_path,
        partition_count=4,
    )

    actual = scan_parquet_artifact(output_path).collect().sort("subject_uri")
    assert unique_count == 3
    assert actual["subject_uri"].to_list() == ["only-a", "only-b", "shared"]
    assert actual["subject_uri"].n_unique() == actual.height


def test_history_post_uri_publication_writes_schema_correct_empty_dataset(tmp_path):
    shards_path = tmp_path / "shards.partial"
    shards_path.mkdir()
    output_path = tmp_path / "history_post_uris.partial"

    unique_count = stage._write_history_post_uris(
        history_post_uri_shards_path=shards_path,
        routed_history_post_uris_path=tmp_path / "routed.partial",
        partial_output_path=output_path,
        partition_count=4,
    )

    actual = scan_parquet_artifact(output_path).collect()
    assert unique_count == 0
    assert actual.is_empty()
    assert actual.schema == pl.Schema(stage.history_data.HISTORY_POST_URI_SCHEMA)


def test_failed_partition_does_not_publish_primary_history_dataset(tmp_path, monkeypatch):
    stage1_dir, _ = _write_query_selection_artifact(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("partition failed")

    monkeypatch.setattr(stage.history_data, "build_query_histories_for_partition", fail)
    with pytest.raises(RuntimeError, match="partition failed"):
        registry.run_stage("user_history", _context(tmp_path, stage1_dir), _args())

    stage2_dirs = list((tmp_path / "artifacts" / "02_user_history").iterdir())
    assert len(stage2_dirs) == 1
    stage2_dir = stage2_dirs[0]
    assert list(stage2_dir.glob("query_histories_*"))
    assert all(path.name.endswith(".partial") for path in stage2_dir.glob("query_histories_*"))
    assert not (stage2_dir / "manifest.json").exists()
    assert not any(
        not path.name.endswith(".partial")
        for path in stage2_dir.glob("history_post_uris_*")
    )
    assert list(stage2_dir.glob("_query_partitions_*"))
    assert list(stage2_dir.glob("_queried_user_likes_*"))


def test_config_requires_positive_limits():
    with pytest.raises(ValueError, match="max_history_posts_per_query"):
        stage.build_config(SimpleNamespace(
            max_history_posts_per_query=0,
            user_history_partition_count=4,
        ))
    with pytest.raises(ValueError, match="user_history_partition_count"):
        stage.build_config(SimpleNamespace(
            max_history_posts_per_query=64,
            user_history_partition_count=0,
        ))


def test_history_source_manifest_is_required(tmp_path):
    stage1_dir, _ = _write_query_selection_artifact(tmp_path)
    next(stage1_dir.glob("like_sources_*.json")).unlink()
    with pytest.raises(FileNotFoundError, match="like_sources"):
        registry.run_stage("user_history", _context(tmp_path, stage1_dir), _args())
