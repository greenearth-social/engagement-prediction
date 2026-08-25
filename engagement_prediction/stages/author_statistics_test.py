from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from engagement_prediction.data import author_statistics
from engagement_prediction.data import ingex
from engagement_prediction.data import source_metadata
from engagement_prediction.data.parquet import scan_parquet_artifact
from engagement_prediction.pipeline import registry
from engagement_prediction.pipeline.core import Context
from engagement_prediction.stages import author_statistics as stage
from utils import helpers


UTC = timezone.utc


def _manifest(
    *,
    prefix: str,
    path: Path,
) -> dict:
    return ingex.build_source_manifest(
        gcs_bucket="unused",
        blob_prefix=prefix,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        paths=[str(path)],
        timestamps=[datetime(2026, 1, 1, tzinfo=UTC)],
    )


def _write_upstream_artifacts(tmp_path: Path) -> tuple[Path, list[Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    posts_path = tmp_path / "posts.parquet"
    pl.DataFrame({
        "at_uri": [
            "p-post-1",
            "p-post-2",
            "p-like",
            "p-excluded",
            "collision",
            "duplicate",
            "duplicate",
            "p-at-val",
            None,
        ],
        "record_created_at": [
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T02:00:00Z",
            "2026-01-01T03:00:00Z",
            "2026-01-01T04:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T05:00:00Z",
            "2026-01-01T12:00:00Z",
            "2026-01-01T06:00:00Z",
        ],
        "did": [
            "author-post",
            "author-post",
            "author-like",
            "author-excluded",
            "author-collision",
            "author-old",
            "author-latest",
            "author-late",
            "invalid",
        ],
    }).write_parquet(posts_path)
    replies_path = tmp_path / "replies.parquet"
    pl.DataFrame({
        "at_uri": ["r-1", "r-2", "collision", "r-at-val"],
        "record_created_at": [
            "2026-01-01T00:30:00Z",
            "2026-01-01T01:30:00Z",
            "2026-01-01T06:00:00Z",
            "2026-01-01T12:00:00Z",
        ],
        "did": ["author-reply", "author-reply", "reply-collision", "reply-late"],
    }).write_parquet(replies_path)
    likes_path = tmp_path / "likes.parquet"
    pl.DataFrame({
        "did": [
            "u1",
            "u2",
            "u2",
            "u2",
            "u3",
            "u4",
            "u5",
            "u6",
            None,
        ],
        "subject_uri": [
            "p-post-1",
            "p-like",
            "p-like",
            "p-like",
            "r-1",
            "collision",
            "missing-post",
            "p-post-1",
            "p-like",
        ],
        "record_created_at": [
            "2026-01-01T00:00:00Z",
            "2026-01-01T03:00:00Z",
            "2026-01-01T03:00:00Z",
            "2026-01-01T03:00:00Z",
            "2026-01-01T04:00:00Z",
            "2026-01-01T05:00:00Z",
            "2026-01-01T06:00:00Z",
            "2026-01-01T12:00:00Z",
            "2026-01-01T07:00:00Z",
        ],
    }).write_parquet(likes_path)
    post_manifest = _manifest(prefix="bsky_posts", path=posts_path)
    reply_manifest = _manifest(prefix="bsky_replies", path=replies_path)
    like_manifest = _manifest(prefix="bsky_likes", path=likes_path)

    root = tmp_path / "artifacts"
    source_metadata_dir = root / "00_source_metadata" / "stage0"
    source_bundle = source_metadata_dir / "source_metadata_stage0"
    metadata_path = source_bundle / "post_metadata"
    metadata_path.mkdir(parents=True)
    roots, root_stats = source_metadata.select_latest_metadata_rows(
        source_metadata.normalize_source_records(
            pl.scan_parquet(posts_path),
            posts_start=datetime(2026, 1, 1, tzinfo=UTC),
            posts_end=datetime(2026, 1, 2, tzinfo=UTC),
            is_reply=False,
        ).collect()
    )
    replies, reply_stats = source_metadata.select_latest_metadata_rows(
        source_metadata.normalize_source_records(
            pl.scan_parquet(replies_path),
            posts_start=datetime(2026, 1, 1, tzinfo=UTC),
            posts_end=datetime(2026, 1, 2, tzinfo=UTC),
            is_reply=True,
        ).collect()
    )
    metadata, overlap_count = source_metadata.apply_root_precedence(roots, replies)
    source_partition_count = 3
    routed = metadata.with_columns(
        source_metadata.uri_partition_expr(source_partition_count)
    )
    for partition_id in range(source_partition_count):
        part = routed.filter(pl.col("_post_partition") == partition_id).drop(
            "_post_partition"
        )
        if not part.is_empty():
            part.write_parquet(metadata_path / f"part-{partition_id:05d}.parquet")
    ingex.write_source_manifest(source_bundle / "post_sources_stage0.json", post_manifest)
    ingex.write_source_manifest(source_bundle / "reply_sources_stage0.json", reply_manifest)
    (source_metadata_dir / "summary.json").write_text(json.dumps({
        "parameters": {"source_metadata_partition_count": source_partition_count},
        "index": {
            "root_source_stats": root_stats,
            "reply_source_stats": reply_stats,
            "root_reply_overlap_count": overlap_count,
        },
    }))
    (source_metadata_dir / "manifest.json").write_text(json.dumps({
        "stage_key": "source_metadata",
        "stage_folder": "00_source_metadata",
        "inputs": {},
    }))
    stage1_dir = root / "01_query_selection" / "stage1"
    stage1_dir.mkdir(parents=True)
    ingex.write_source_manifest(stage1_dir / "like_sources_stage1.json", like_manifest)
    (stage1_dir / "summary.json").write_text(json.dumps({
        "posts_start": "2026-01-01T00:00:00+00:00",
        "posts_end": "2026-01-02T00:00:00+00:00",
        "parameters": {
            "train_start": "2026-01-01T08:00:00+00:00",
            "val_start": "2026-01-01T12:00:00+00:00",
        },
    }) + "\n")
    (stage1_dir / "manifest.json").write_text(json.dumps({
        "stage_key": "query_selection",
        "stage_folder": "01_query_selection",
        "inputs": {"00_source_metadata": str(source_metadata_dir.resolve())},
    }) + "\n")

    stage2_dir = root / "02_user_history" / "stage2"
    stage2_dir.mkdir(parents=True)
    (stage2_dir / "manifest.json").write_text(json.dumps({
        "stage_key": "user_history",
        "stage_folder": "02_user_history",
        "inputs": {
            "00_source_metadata": str(source_metadata_dir.resolve()),
            "01_query_selection": str(stage1_dir.resolve()),
        },
    }) + "\n")

    stage3_dir = root / "03_post_selection" / "stage3"
    stage3_bundle = stage3_dir / "post_universe_stage3"
    stage3_bundle.mkdir(parents=True)
    ingex.write_source_manifest(stage3_bundle / "post_sources_stage3.json", post_manifest)
    ingex.write_source_manifest(stage3_bundle / "reply_sources_stage3.json", reply_manifest)
    (stage3_dir / "manifest.json").write_text(json.dumps({
        "stage_key": "post_selection",
        "stage_folder": "03_post_selection",
        "inputs": {
            "00_source_metadata": str(source_metadata_dir.resolve()),
            "01_query_selection": str(stage1_dir.resolve()),
            "02_user_history": str(stage2_dir.resolve()),
        },
    }) + "\n")

    stage4_dir = root / "04_negative_selection" / "stage4"
    stage4_dir.mkdir(parents=True)
    (stage4_dir / "manifest.json").write_text(json.dumps({
        "stage_key": "negative_selection",
        "stage_folder": "04_negative_selection",
        "inputs": {
            "00_source_metadata": str(source_metadata_dir.resolve()),
            "01_query_selection": str(stage1_dir.resolve()),
            "02_user_history": str(stage2_dir.resolve()),
            "03_post_selection": str(stage3_dir.resolve()),
        },
    }) + "\n")

    stage5_dir = root / "05_post_liker_history" / "stage5"
    stage5_bundle = stage5_dir / "post_liker_histories_stage5"
    stage5_bundle.mkdir(parents=True)
    ingex.write_source_manifest(stage5_bundle / "like_sources_stage5.json", like_manifest)
    (stage5_dir / "manifest.json").write_text(json.dumps({
        "stage_key": "post_liker_history",
        "stage_folder": "05_post_liker_history",
        "inputs": {
            "00_source_metadata": str(source_metadata_dir.resolve()),
            "01_query_selection": str(stage1_dir.resolve()),
            "02_user_history": str(stage2_dir.resolve()),
            "03_post_selection": str(stage3_dir.resolve()),
            "04_negative_selection": str(stage4_dir.resolve()),
        },
    }) + "\n")
    return stage5_dir, [
        source_metadata_dir,
        stage1_dir,
        stage2_dir,
        stage3_dir,
        stage4_dir,
    ]


def _args(partition_count: int) -> SimpleNamespace:
    return SimpleNamespace(
        author_statistics_partition_count=partition_count,
        _argv=[
            "--start-from",
            "author_statistics",
            "--stop-after",
            "author_statistics",
        ],
    )


def _context(tmp_path: Path, stage5_dir: Path, suffix: str) -> Context:
    context = Context(
        run_dir=tmp_path / "runs" / suffix,
        artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "runs",
        pipeline_run_id=suffix,
    )
    context.prior_outputs["05_post_liker_history"] = stage5_dir
    return context


def _reset_logger() -> None:
    logger = helpers._stage_loggers.pop("06_AUTHOR_STATISTICS", None)
    if logger is not None:
        for handler in logger.handlers:
            handler.close()


def _run(tmp_path: Path, partition_count: int):
    _reset_logger()
    stage5_dir, ancestors = _write_upstream_artifacts(tmp_path)
    result = registry.run_stage(
        "author_statistics",
        _context(tmp_path, stage5_dir, f"run-{partition_count}"),
        _args(partition_count),
    )
    return result, stage5_dir, ancestors


def test_stage_builds_unfiltered_training_only_author_statistics(tmp_path):
    result, stage5_dir, ancestors = _run(tmp_path, 3)
    authors = scan_parquet_artifact(
        Path(result["artifacts"]["author_statistics_dataset_path"])
    ).collect().sort("author_did")

    assert authors.schema == pl.Schema(author_statistics.AUTHOR_STAT_SCHEMA)
    assert authors["author_did"].to_list() == [
        "author-collision",
        "author-excluded",
        "author-latest",
        "author-like",
        "author-post",
        "author-reply",
    ]
    assert "author_idx" not in authors.columns
    like_author = authors.filter(pl.col("author_did") == "author-like").row(
        0,
        named=True,
    )
    assert like_author["post_count"] == 1
    assert like_author["received_like_count"] == 3
    post_author = authors.filter(pl.col("author_did") == "author-post").row(
        0,
        named=True,
    )
    assert post_author["root_post_count"] == 2
    assert post_author["reply_post_count"] == 0
    assert post_author["received_like_count"] == 1
    assert post_author["mean_likes_per_post"] == 0.5
    assert post_author["median_likes_per_post"] == 0.5
    reply_author = authors.filter(pl.col("author_did") == "author-reply").row(
        0,
        named=True,
    )
    assert reply_author["root_post_count"] == 0
    assert reply_author["reply_post_count"] == 2
    assert reply_author["reply_received_like_count"] == 1
    assert "reply-collision" not in authors["author_did"].to_list()
    assert "author-late" not in authors["author_did"].to_list()

    source_prefixes = {
        "post_sources_path": "bsky_posts",
        "reply_sources_path": "bsky_replies",
        "like_sources_path": "bsky_likes",
    }
    for artifact_name, expected_prefix in source_prefixes.items():
        manifest = ingex.load_source_manifest(
            Path(result["artifacts"][artifact_name])
        )
        assert manifest["blob_prefix"] == expected_prefix
        assert manifest["start"] == "2026-01-01T00:00:00+00:00"
        assert manifest["end"] == "2026-01-02T00:00:00+00:00"

    output_dir = Path(result["output_dir"])
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["input"]["support_start"] == "2026-01-01T00:00:00+00:00"
    assert summary["input"]["support_end"] == "2026-01-01T12:00:00+00:00"
    assert summary["post_aggregation"]["root_reply_overlap_count"] == 1
    assert summary["post_aggregation"]["matched_like_event_count"] == 6
    assert summary["post_aggregation"]["unmatched_like_event_count"] == 1
    assert summary["author_aggregation"]["author_count"] == 6
    assert summary["outputs"]["author_statistics_dataset_path"].endswith(
        "/author_statistics"
    )
    assert not list(output_dir.glob("author_statistics_*.partial"))
    assert not list(output_dir.glob("_author_statistics_staging_*"))
    assert json.loads((output_dir / "manifest.json").read_text())["inputs"] == {
        "00_source_metadata": str(ancestors[0].resolve()),
        "01_query_selection": str(ancestors[1].resolve()),
        "02_user_history": str(ancestors[2].resolve()),
        "03_post_selection": str(ancestors[3].resolve()),
        "04_negative_selection": str(ancestors[4].resolve()),
        "05_post_liker_history": str(stage5_dir.resolve()),
    }


def test_logical_output_is_independent_of_partition_count(tmp_path):
    first, _, _ = _run(tmp_path / "one", 1)
    second, _, _ = _run(tmp_path / "five", 5)

    first_df = scan_parquet_artifact(
        Path(first["artifacts"]["author_statistics_dataset_path"])
    ).collect().sort("author_did")
    second_df = scan_parquet_artifact(
        Path(second["artifacts"]["author_statistics_dataset_path"])
    ).collect().sort("author_did")
    assert first_df.equals(second_df)


def test_stage_reuses_exact_snapshots_without_relisting(tmp_path, monkeypatch):
    stage5_dir, _ = _write_upstream_artifacts(tmp_path)
    monkeypatch.setattr(
        stage.ingex,
        "list_ingex_parquet_files",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not relist")),
    )
    _reset_logger()
    registry.run_stage(
        "author_statistics",
        _context(tmp_path, stage5_dir, "no-relist"),
        _args(2),
    )


def test_failed_partition_does_not_publish_bundle_or_manifest(tmp_path, monkeypatch):
    stage5_dir, _ = _write_upstream_artifacts(tmp_path)
    monkeypatch.setattr(
        stage.author_statistics_artifacts,
        "process_uri_partitions",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("partition failed")),
    )
    _reset_logger()

    with pytest.raises(RuntimeError, match="partition failed"):
        registry.run_stage(
            "author_statistics",
            _context(tmp_path, stage5_dir, "failed"),
            _args(2),
        )

    stage6_dir = next((tmp_path / "artifacts" / "06_author_statistics").iterdir())
    assert list(stage6_dir.glob("author_statistics_*.partial"))
    assert not [
        path
        for path in stage6_dir.glob("author_statistics_*")
        if not path.name.endswith(".partial")
    ]
    assert not (stage6_dir / "manifest.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("author_statistics_partition_count", 0),
    ],
)
def test_config_rejects_invalid_counts(field, value):
    args = _args(2)
    setattr(args, field, value)
    with pytest.raises(ValueError, match=field):
        stage.build_config(
            args,
            support_start=datetime(2026, 1, 1, tzinfo=UTC),
            support_end=datetime(2026, 1, 2, tzinfo=UTC),
            source_metadata_partition_count=2,
        )
