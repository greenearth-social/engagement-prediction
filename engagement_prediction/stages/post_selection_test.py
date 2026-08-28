from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from engagement_prediction.data import ingex, source_metadata
from engagement_prediction.data import post_selection as post_data
from engagement_prediction.data.parquet import scan_parquet_artifact
from engagement_prediction.pipeline import registry
from engagement_prediction.pipeline.core import Context
from engagement_prediction.stages import post_selection as stage


UTC = timezone.utc


def _write_source_metadata_artifact(
    tmp_path: Path,
    posts_path: Path,
    replies_path: Path,
    *,
    partition_count: int,
) -> Path:
    stage_dir = tmp_path / "artifacts" / "00_source_metadata" / "stage0"
    bundle = stage_dir / "source_metadata_stage0"
    metadata_path = bundle / "post_metadata"
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
    routed = metadata.with_columns(source_metadata.uri_partition_expr(partition_count))
    for partition_id in range(partition_count):
        part = routed.filter(pl.col("_post_partition") == partition_id).drop(
            "_post_partition"
        )
        if not part.is_empty():
            part.write_parquet(metadata_path / f"part-{partition_id:05d}.parquet")
    for prefix, blob_prefix, path in (
        ("post", "bsky_posts", posts_path),
        ("reply", "bsky_replies", replies_path),
    ):
        ingex.write_source_manifest(
            bundle / f"{prefix}_sources_stage0.json",
            ingex.build_source_manifest(
                gcs_bucket="unused",
                blob_prefix=blob_prefix,
                start=datetime(2026, 1, 1, tzinfo=UTC),
                end=datetime(2026, 1, 2, tzinfo=UTC),
                paths=[str(path)],
                timestamps=[datetime(2026, 1, 1, tzinfo=UTC)],
            ),
        )
    (stage_dir / "summary.json").write_text(json.dumps({
        "parameters": {"source_metadata_partition_count": partition_count},
        "index": {
            "root_source_stats": root_stats,
            "reply_source_stats": reply_stats,
            "root_reply_overlap_count": overlap_count,
        },
    }))
    (stage_dir / "manifest.json").write_text(json.dumps({
        "stage_key": "source_metadata",
        "stage_folder": "00_source_metadata",
        "inputs": {},
    }))
    return stage_dir


def _write_upstream_artifacts(
    tmp_path: Path,
    source_metadata_dir: Path,
) -> tuple[Path, Path]:
    stage1_dir = tmp_path / "artifacts" / "01_query_selection" / "stage1"
    stage1_dir.mkdir(parents=True)
    pl.DataFrame({
        "did": ["u1"],
        "query_hour": [datetime(2026, 1, 1, 10, tzinfo=UTC)],
        "user_cohort": ["trainval"],
        "split": ["train"],
        "positive_count": pl.Series([2], dtype=pl.UInt32),
    }).write_parquet(stage1_dir / "queries_stage1.parquet")
    pl.DataFrame({
        "did": ["u1", "u1"],
        "query_hour": [datetime(2026, 1, 1, 10, tzinfo=UTC)] * 2,
        "subject_uri": ["positive", "overlap"],
        "like_created_at": [
            datetime(2026, 1, 1, 10, 5, tzinfo=UTC),
            datetime(2026, 1, 1, 10, 6, tzinfo=UTC),
        ],
    }).write_parquet(stage1_dir / "query_positives_stage1.parquet")
    (stage1_dir / "manifest.json").write_text(json.dumps({
        "stage_key": "query_selection",
        "stage_folder": "01_query_selection",
        "inputs": {"00_source_metadata": str(source_metadata_dir.resolve())},
    }) + "\n")

    stage2_dir = tmp_path / "artifacts" / "02_user_history" / "stage2"
    stage2_dir.mkdir(parents=True)
    history_uris = stage2_dir / "history_post_uris_stage2"
    history_uris.mkdir()
    pl.DataFrame({
        "subject_uri": ["history-root", "reply-history", "missing", "overlap"],
    }).write_parquet(history_uris / "part-00000.parquet")
    query_histories = stage2_dir / "query_histories_stage2"
    query_histories.mkdir()
    pl.DataFrame({"unused": ["Stage 3 must not read this"]}).write_parquet(
        query_histories / "part-00000.parquet"
    )
    (stage2_dir / "manifest.json").write_text(json.dumps({
        "stage_key": "user_history",
        "stage_folder": "02_user_history",
        "inputs": {
            "00_source_metadata": str(source_metadata_dir.resolve()),
            "01_query_selection": str(stage1_dir.resolve()),
        },
    }) + "\n")
    return stage1_dir, stage2_dir


def _write_sources(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    posts_path = tmp_path / "bsky_posts_20260101_000000.parquet"
    pl.DataFrame({
        "at_uri": [
            "positive",
            "overlap",
            "overlap",
            "overlap",
            "history-root",
            "random",
            None,
        ],
        "record_created_at": [
            "2026-01-01T01:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T02:00:00Z",
            "2026-01-01T02:00:00Z",
            "2026-01-01T03:00:00Z",
            "2026-01-01T04:00:00Z",
            "bad",
        ],
        "did": [
            "a-positive",
            "old-author",
            "z-author",
            "a-author",
            "a-history",
            "a-random",
            "invalid",
        ],
    }).write_parquet(posts_path)
    replies_path = tmp_path / "bsky_replies_20260101_000000.parquet"
    pl.DataFrame({
        "at_uri": ["reply-history", "overlap", "unused-reply"],
        "record_created_at": [
            "2026-01-01T05:00:00Z",
            "2026-01-01T06:00:00Z",
            "2026-01-01T07:00:00Z",
        ],
        "did": ["reply-author", "reply-overlap", "unused-author"],
    }).write_parquet(replies_path)
    return posts_path, replies_path


def _context(tmp_path: Path, stage2_dir: Path) -> Context:
    context = Context(
        run_dir=tmp_path / "runs" / "run",
        artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "runs",
        pipeline_run_id="run",
    )
    context.prior_outputs["02_user_history"] = stage2_dir
    return context


def _args(random_fraction=1.0, worker_count=1):
    return SimpleNamespace(
        gcs_bucket="unused",
        posts_start="2026-01-01T00:00:00Z",
        posts_end="2026-01-02T00:00:00Z",
        random_candidate_sampling_fraction=random_fraction,
        random_seed=42,
        data_partition_worker_count=worker_count,
        _argv=["--start-from", "post_selection", "--stop-after", "post_selection"],
    )


def _run_stage(tmp_path, monkeypatch, partition_count=4, worker_count=1):
    posts_source, replies_source = _write_sources(tmp_path)
    source_metadata_dir = _write_source_metadata_artifact(
        tmp_path,
        posts_source,
        replies_source,
        partition_count=partition_count,
    )
    stage1_dir, stage2_dir = _write_upstream_artifacts(tmp_path, source_metadata_dir)
    monkeypatch.setattr(
        stage.ingex,
        "list_ingex_parquet_files",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Stage 3 must not list raw sources")
        ),
    )
    result = registry.run_stage(
        "post_selection",
        _context(tmp_path, stage2_dir),
        _args(worker_count=worker_count),
    )
    return (
        result,
        source_metadata_dir,
        stage1_dir,
        stage2_dir,
        posts_source,
        replies_source,
    )


def test_registry_run_publishes_root_and_reply_post_universe(tmp_path, monkeypatch):
    (
        result,
        source_metadata_dir,
        stage1_dir,
        stage2_dir,
        posts_source,
        replies_source,
    ) = _run_stage(tmp_path, monkeypatch)
    output_dir = Path(result["output_dir"])
    bundle_path = Path(result["artifacts"]["post_universe_path"])
    posts = scan_parquet_artifact(Path(result["artifacts"]["posts_path"])).collect().sort(
        "subject_uri"
    )
    required = scan_parquet_artifact(
        Path(result["artifacts"]["required_posts_path"])
    ).collect().sort("subject_uri")
    candidates = scan_parquet_artifact(
        Path(result["artifacts"]["candidate_sources_path"])
    ).collect().sort(["subject_uri", "candidate_source"])
    missing = scan_parquet_artifact(
        Path(result["artifacts"]["missing_required_posts_path"])
    ).collect()

    assert posts.schema == pl.Schema(post_data.POST_SCHEMA)
    assert posts["subject_uri"].to_list() == [
        "history-root",
        "overlap",
        "positive",
        "random",
        "reply-history",
    ]
    assert posts.filter(pl.col("subject_uri") == "reply-history")["is_reply"].item()
    assert not posts.filter(pl.col("subject_uri") == "overlap")["is_reply"].item()
    assert posts.filter(pl.col("subject_uri") == "overlap")["author_did"].item() == "a-author"
    assert "unused-reply" not in posts["subject_uri"].to_list()
    assert missing.to_dicts() == [
        {"subject_uri": "missing", "is_positive": False, "is_history": True}
    ]
    assert required.height == 5
    assert set(candidates["candidate_source"]) == {"random"}
    assert set(candidates["subject_uri"]) == {
        "history-root", "overlap", "positive", "random"
    }
    assert bundle_path.is_dir()
    assert not list(bundle_path.glob("inference_sources_*.json"))
    post_sources = json.loads(Path(result["artifacts"]["post_sources_path"]).read_text())
    reply_sources = json.loads(Path(result["artifacts"]["reply_sources_path"]).read_text())
    assert [row["uri"] for row in post_sources["files"]] == [str(posts_source)]
    assert [row["uri"] for row in reply_sources["files"]] == [str(replies_source)]
    assert post_sources["start"] == reply_sources["start"]
    assert post_sources["end"] == reply_sources["end"]
    assert not list(output_dir.glob("post_universe_*.partial"))
    assert not list(output_dir.glob("_post_selection_staging_*"))
    assert json.loads((output_dir / "manifest.json").read_text())["inputs"] == {
        "00_source_metadata": str(source_metadata_dir.resolve()),
        "01_query_selection": str(stage1_dir.resolve()),
        "02_user_history": str(stage2_dir.resolve()),
    }
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["outputs"]["post_count"] == 5
    assert summary["outputs"]["reply_post_count"] == 1
    assert summary["required_post_stats"]["history_resolved_as_root_count"] == 2
    assert summary["required_post_stats"]["history_resolved_as_reply_count"] == 1
    assert summary["required_post_stats"]["root_reply_overlap_count"] == 1
    assert summary["parameters"]["data_partition_worker_count"] == 1
    stage_info = (output_dir / "stage_info.txt").read_text()
    assert "source_metadata_partition_count: 4" in stage_info
    stage_log = (output_dir / "stage.log").read_text()
    assert "no raw post/reply rescan is needed" in stage_log


def test_logical_output_is_partition_count_independent(tmp_path, monkeypatch):
    first = _run_stage(tmp_path / "one", monkeypatch, partition_count=1)[0]
    second = _run_stage(tmp_path / "seven", monkeypatch, partition_count=7)[0]

    for artifact in (
        "posts_path",
        "required_posts_path",
        "candidate_sources_path",
        "missing_required_posts_path",
    ):
        first_df = scan_parquet_artifact(Path(first["artifacts"][artifact])).collect()
        second_df = scan_parquet_artifact(Path(second["artifacts"][artifact])).collect()
        assert first_df.sort(first_df.columns).equals(second_df.sort(second_df.columns))


def test_parallel_partition_workers_preserve_logical_output(tmp_path, monkeypatch):
    serial = _run_stage(
        tmp_path / "serial", monkeypatch, partition_count=4, worker_count=1
    )[0]
    parallel = _run_stage(
        tmp_path / "parallel", monkeypatch, partition_count=4, worker_count=2
    )[0]

    for artifact in (
        "posts_path",
        "required_posts_path",
        "candidate_sources_path",
        "missing_required_posts_path",
    ):
        serial_df = scan_parquet_artifact(Path(serial["artifacts"][artifact])).collect()
        parallel_df = scan_parquet_artifact(Path(parallel["artifacts"][artifact])).collect()
        assert serial_df.sort(serial_df.columns).equals(
            parallel_df.sort(parallel_df.columns)
        )
    summary = json.loads((Path(parallel["output_dir"]) / "summary.json").read_text())
    assert summary["parameters"]["data_partition_worker_count"] == 2
    assert summary["partition_processing"]["partition_worker_count"] == 2
    assert [
        row["partition_id"]
        for row in summary["partition_processing"]["partition_stats"]
    ] == [0, 1, 2, 3]


def test_reply_only_positive_fails(tmp_path, monkeypatch):
    posts_source, replies_source = _write_sources(tmp_path)
    posts = pl.read_parquet(posts_source).filter(pl.col("at_uri") != "positive")
    posts.write_parquet(posts_source)
    replies = pl.read_parquet(replies_source).vstack(pl.DataFrame({
        "at_uri": ["positive"],
        "record_created_at": ["2026-01-01T08:00:00Z"],
        "did": ["reply-positive-author"],
    }))
    replies.write_parquet(replies_source)
    source_metadata_dir = _write_source_metadata_artifact(
        tmp_path, posts_source, replies_source, partition_count=4
    )
    _, stage2_dir = _write_upstream_artifacts(tmp_path, source_metadata_dir)

    with pytest.raises(ValueError, match="resolved only as replies"):
        registry.run_stage(
            "post_selection", _context(tmp_path, stage2_dir), _args()
        )


def test_missing_positive_fails(tmp_path, monkeypatch):
    posts_source, replies_source = _write_sources(tmp_path)
    pl.read_parquet(posts_source).filter(
        pl.col("at_uri") != "positive"
    ).write_parquet(posts_source)
    source_metadata_dir = _write_source_metadata_artifact(
        tmp_path, posts_source, replies_source, partition_count=4
    )
    _, stage2_dir = _write_upstream_artifacts(tmp_path, source_metadata_dir)

    with pytest.raises(ValueError, match="absent from the exact root snapshot"):
        registry.run_stage(
            "post_selection", _context(tmp_path, stage2_dir), _args(worker_count=2)
        )

    stage3_dir = next((tmp_path / "artifacts" / "03_post_selection").iterdir())
    assert list(stage3_dir.glob("post_universe_*.partial"))
    assert not (stage3_dir / "manifest.json").exists()


def test_failed_partition_leaves_partial_bundle_and_no_manifest(tmp_path, monkeypatch):
    posts_source, replies_source = _write_sources(tmp_path)
    source_metadata_dir = _write_source_metadata_artifact(
        tmp_path, posts_source, replies_source, partition_count=4
    )
    _, stage2_dir = _write_upstream_artifacts(tmp_path, source_metadata_dir)
    monkeypatch.setattr(
        stage.post_selection_artifacts,
        "process_uri_partitions",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("partition failed")),
    )

    with pytest.raises(RuntimeError, match="partition failed"):
        registry.run_stage(
            "post_selection", _context(tmp_path, stage2_dir), _args()
        )

    stage3_dir = next((tmp_path / "artifacts" / "03_post_selection").iterdir())
    assert list(stage3_dir.glob("post_universe_*.partial"))
    assert not [
        path for path in stage3_dir.glob("post_universe_*")
        if not path.name.endswith(".partial")
    ]
    assert not (stage3_dir / "manifest.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("random_candidate_sampling_fraction", 1.1),
    ],
)
def test_config_validation(field, value):
    args = _args()
    setattr(args, field, value)
    with pytest.raises(ValueError, match=field):
        stage.build_config(args)


def test_worker_count_must_be_positive():
    args = _args(worker_count=0)
    with pytest.raises(ValueError, match="data_partition_worker_count"):
        stage.build_config(args)
