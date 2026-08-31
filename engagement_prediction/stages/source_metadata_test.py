from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from engagement_prediction.data import source_metadata
from engagement_prediction.data.parquet import scan_parquet_artifact
from engagement_prediction.pipeline import registry
from engagement_prediction.pipeline.core import Context
from engagement_prediction.stages import source_metadata as stage
from engagement_prediction.pipeline import logging as pipeline_logging


UTC = timezone.utc


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    posts = tmp_path / "bsky_posts_20260101_000000.parquet"
    replies = tmp_path / "bsky_replies_20260101_000000.parquet"
    pl.DataFrame({
        "at_uri": ["root", "both", "duplicate", "duplicate", "tie", "tie", "", "late"],
        "record_created_at": [
            "2026-01-01T01:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T02:00:00Z",
            "2026-01-01T03:00:00Z",
            "2026-01-01T03:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-02T00:00:00Z",
        ],
        "did": ["a", "root-author", "old", "new", "z", "a", "a", "a"],
    }).write_parquet(posts)
    pl.DataFrame({
        "at_uri": ["reply", "both", None],
        "record_created_at": [
            "2026-01-01T01:00:00Z",
            "2026-01-01T04:00:00Z",
            "2026-01-01T01:00:00Z",
        ],
        "did": ["reply-author", "reply-author", "a"],
    }).write_parquet(replies)
    return posts, replies


def _args(partition_count: int, worker_count: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        gcs_bucket="bucket",
        posts_start="2026-01-01T00:00:00Z",
        posts_end="2026-01-02T00:00:00Z",
        source_metadata_partition_count=partition_count,
        data_partition_worker_count=worker_count,
        _argv=["--stop-after", "source_metadata"],
    )


def _context(tmp_path: Path, suffix: str) -> Context:
    return Context(
        run_dir=tmp_path / "runs" / suffix,
        artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "runs",
        pipeline_run_id=suffix,
    )


def _reset_logger() -> None:
    logger = pipeline_logging._stage_loggers.pop("00_SOURCE_METADATA", None)
    if logger is not None:
        for handler in logger.handlers:
            handler.close()


def _run(tmp_path: Path, monkeypatch, partition_count: int, worker_count: int = 1):
    tmp_path.mkdir(parents=True, exist_ok=True)
    posts, replies = _sources(tmp_path)

    def list_files(**kwargs):
        path = posts if kwargs["blob_prefix"] == "bsky_posts" else replies
        return [str(path)], [datetime(2026, 1, 1, tzinfo=UTC)]

    monkeypatch.setattr(stage.ingex, "list_ingex_parquet_files", list_files)
    _reset_logger()
    return registry.run_stage(
        "source_metadata",
        _context(tmp_path, f"run-{partition_count}"),
        _args(partition_count, worker_count),
    )


def test_publishes_canonical_metadata_and_exact_snapshots(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, 3)
    metadata = scan_parquet_artifact(
        Path(result["artifacts"]["post_metadata_path"])
    ).collect().sort("subject_uri")

    assert metadata.schema == pl.Schema(source_metadata.POST_METADATA_SCHEMA)
    assert metadata["subject_uri"].to_list() == ["both", "duplicate", "reply", "root", "tie"]
    assert metadata.filter(pl.col("subject_uri") == "both")["author_did"].item() == "root-author"
    assert metadata.filter(pl.col("subject_uri") == "duplicate")["author_did"].item() == "new"
    assert metadata.filter(pl.col("subject_uri") == "tie")["author_did"].item() == "a"
    assert metadata.filter(pl.col("subject_uri") == "reply")["is_reply"].item()

    output_dir = Path(result["output_dir"])
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["index"]["root_reply_overlap_count"] == 1
    assert summary["index"]["canonical_record_count"] == 5
    assert summary["parameters"]["source_metadata_partition_count"] == 3
    assert summary["parameters"]["data_partition_worker_count"] == 1
    assert summary["index"]["partition_worker_count"] == 1
    assert not list(output_dir.glob("*.partial"))
    assert not list(output_dir.glob("_source_metadata_staging_*"))
    assert json.loads((output_dir / "manifest.json").read_text())["inputs"] == {}


def test_logical_output_is_partition_count_independent(tmp_path, monkeypatch):
    first = _run(tmp_path / "one", monkeypatch, 1)
    first_df = scan_parquet_artifact(Path(first["artifacts"]["post_metadata_path"])).collect()
    second = _run(tmp_path / "four", monkeypatch, 4)
    second_df = scan_parquet_artifact(Path(second["artifacts"]["post_metadata_path"])).collect()
    assert first_df.sort("subject_uri").equals(second_df.sort("subject_uri"))


def test_parallel_partition_workers_preserve_logical_output(tmp_path, monkeypatch):
    serial = _run(tmp_path / "serial", monkeypatch, 4, 1)
    parallel = _run(tmp_path / "parallel", monkeypatch, 4, 2)
    serial_df = scan_parquet_artifact(Path(serial["artifacts"]["post_metadata_path"])).collect()
    parallel_df = scan_parquet_artifact(
        Path(parallel["artifacts"]["post_metadata_path"])
    ).collect()
    assert serial_df.sort("subject_uri").equals(parallel_df.sort("subject_uri"))
    parallel_summary = json.loads((Path(parallel["output_dir"]) / "summary.json").read_text())
    assert parallel_summary["index"]["partition_worker_count"] == 2
    assert [
        row["partition_id"] for row in parallel_summary["index"]["partition_stats"]
    ] == [0, 1, 2, 3]


def test_failed_partition_does_not_publish_bundle_or_manifest(tmp_path, monkeypatch):
    posts, replies = _sources(tmp_path)
    monkeypatch.setattr(
        stage.ingex,
        "list_ingex_parquet_files",
        lambda **kwargs: (
            [str(posts if kwargs["blob_prefix"] == "bsky_posts" else replies)],
            [datetime(2026, 1, 1, tzinfo=UTC)],
        ),
    )
    monkeypatch.setattr(
        stage.source_metadata_artifacts,
        "process_uri_partitions",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("partition failed")),
    )
    _reset_logger()

    with pytest.raises(RuntimeError, match="partition failed"):
        registry.run_stage("source_metadata", _context(tmp_path, "failed"), _args(2))

    output_dir = next((tmp_path / "artifacts" / "00_source_metadata").iterdir())
    assert list(output_dir.glob("source_metadata_*.partial"))
    assert not [
        path
        for path in output_dir.glob("source_metadata_*")
        if not path.name.endswith(".partial")
    ]
    assert not (output_dir / "manifest.json").exists()


def test_config_validation():
    with pytest.raises(ValueError, match="source_metadata_partition_count"):
        stage.build_config(_args(0))
    with pytest.raises(ValueError, match="data_partition_worker_count"):
        stage.build_config(_args(1, 0))
