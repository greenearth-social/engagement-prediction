from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from engagement_prediction.data import ingex
from engagement_prediction.data import post_liker_history
from engagement_prediction.data import post_selection
from engagement_prediction.data.parquet import scan_parquet_artifact
from engagement_prediction.pipeline import registry
from engagement_prediction.pipeline.core import Context
from engagement_prediction.stages import post_liker_history as stage
from utils import helpers


UTC = timezone.utc


def _write_uri_partitioned(
    path: Path,
    frame: pl.DataFrame,
    partition_count: int,
) -> None:
    path.mkdir(parents=True)
    routed = frame.with_columns(post_selection.post_partition_expr(partition_count))
    for partition_id in range(partition_count):
        part = routed.filter(pl.col("_post_partition") == partition_id).drop(
            "_post_partition"
        )
        if not part.is_empty():
            part.write_parquet(path / f"part-{partition_id:05d}.parquet")


def _write_upstream_artifacts(
    tmp_path: Path,
    *,
    stage3_partition_count: int,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    likes_path = tmp_path / f"likes-{stage3_partition_count}.parquet"
    pl.DataFrame({
        "did": [
            "target-user",
            "external-user",
            "duplicate-user",
            "duplicate-user",
            "reply-liker",
            "negative-liker",
            "unused-liker",
            None,
            "too-late",
            "",
        ],
        "subject_uri": [
            "positive",
            "positive",
            "history-root",
            "history-root",
            "history-reply",
            "negative",
            "reservoir-only",
            "positive",
            "negative",
            "negative",
        ],
        "record_created_at": [
            "2026-01-01T10:05:00Z",
            "2026-01-01T08:00:00Z",
            "2026-01-01T07:00:00Z",
            "2026-01-01T07:00:00Z",
            "2026-01-01T06:00:00Z",
            "2026-01-01T09:00:00Z",
            "2026-01-01T09:00:00Z",
            "2026-01-01T09:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-01T09:00:00Z",
        ],
    }).write_parquet(likes_path)

    root = tmp_path / "artifacts"
    source_metadata_dir = root / "00_source_metadata" / "stage0"
    stage1_dir = root / "01_query_selection" / "stage1"
    stage1_dir.mkdir(parents=True)
    like_manifest = ingex.build_source_manifest(
        gcs_bucket="unused",
        blob_prefix="bsky_likes",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        paths=[str(likes_path)],
        timestamps=[datetime(2026, 1, 1, tzinfo=UTC)],
    )
    ingex.write_source_manifest(stage1_dir / "like_sources_stage1.json", like_manifest)
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
    stage3_dir.mkdir(parents=True)
    posts = pl.DataFrame({
        "subject_uri": [
            "positive",
            "history-root",
            "history-reply",
            "negative",
            "zero-like",
            "reservoir-only",
        ],
        "post_created_at": [datetime(2026, 1, 1, tzinfo=UTC)] * 6,
        "author_did": ["author"] * 6,
        "is_reply": [False, False, True, False, False, False],
    }, schema=post_selection.POST_SCHEMA)
    source_bundle = source_metadata_dir / "source_metadata_stage0"
    _write_uri_partitioned(
        source_bundle / "post_metadata", posts, stage3_partition_count
    )
    for prefix, blob_prefix in (("post", "bsky_posts"), ("reply", "bsky_replies")):
        ingex.write_source_manifest(
            source_bundle / f"{prefix}_sources_stage0.json",
            ingex.build_source_manifest(
                gcs_bucket="unused",
                blob_prefix=blob_prefix,
                start=datetime(2026, 1, 1, tzinfo=UTC),
                end=datetime(2026, 1, 2, tzinfo=UTC),
                paths=[str(tmp_path / f"{blob_prefix}.parquet")],
                timestamps=[datetime(2026, 1, 1, tzinfo=UTC)],
            ),
        )
    (source_metadata_dir / "summary.json").write_text(json.dumps({
        "parameters": {"source_metadata_partition_count": stage3_partition_count},
        "index": {
            "root_source_stats": {},
            "reply_source_stats": {},
            "root_reply_overlap_count": 0,
        },
    }))
    (source_metadata_dir / "manifest.json").write_text(json.dumps({
        "stage_key": "source_metadata",
        "stage_folder": "00_source_metadata",
        "inputs": {},
    }))
    required = pl.DataFrame({
        "subject_uri": [
            "positive",
            "history-root",
            "history-reply",
            "missing-history",
        ],
        "is_positive": [True, False, False, False],
        "is_history": [False, True, True, True],
    }, schema=post_selection.REQUIRED_POST_SCHEMA)
    _write_uri_partitioned(stage3_bundle / "posts", posts, stage3_partition_count)
    _write_uri_partitioned(
        stage3_bundle / "required_posts", required, stage3_partition_count
    )
    _write_uri_partitioned(
        stage3_bundle / "missing_required_posts",
        required.filter(pl.col("subject_uri") == "missing-history"),
        stage3_partition_count,
    )
    (stage3_dir / "summary.json").write_text(json.dumps({
        "parameters": {"source_metadata_partition_count": stage3_partition_count},
    }) + "\n")
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
    stage4_bundle = stage4_dir / "negative_candidates_stage4"
    stage4_dir.mkdir(parents=True)
    negatives = pl.DataFrame({
        "subject_uri": ["negative", "zero-like", "positive"],
    })
    _write_uri_partitioned(
        stage4_bundle / "negative_post_uris",
        negatives,
        stage3_partition_count,
    )
    ingex.write_source_manifest(
        stage4_bundle / "like_sources_stage4.json",
        like_manifest,
    )
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
    return source_metadata_dir, stage1_dir, stage2_dir, stage3_dir, stage4_dir, likes_path


def _args(partition_count: int) -> SimpleNamespace:
    return SimpleNamespace(
        post_liker_history_partition_count=partition_count,
        _argv=[
            "--start-from",
            "post_liker_history",
            "--stop-after",
            "post_liker_history",
        ],
    )


def _context(tmp_path: Path, stage4_dir: Path, suffix: str) -> Context:
    context = Context(
        run_dir=tmp_path / "runs" / suffix,
        artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "runs",
        pipeline_run_id=suffix,
    )
    context.prior_outputs["04_negative_selection"] = stage4_dir
    return context


def _reset_logger() -> None:
    logger = helpers._stage_loggers.pop("05_POST_LIKER_HISTORY", None)
    if logger is not None:
        for handler in logger.handlers:
            handler.close()


def _run(tmp_path: Path, *, stage3_partitions: int, stage5_partitions: int):
    _reset_logger()
    upstream = _write_upstream_artifacts(
        tmp_path,
        stage3_partition_count=stage3_partitions,
    )
    result = registry.run_stage(
        "post_liker_history",
        _context(tmp_path, upstream[4], f"run-{stage5_partitions}"),
        _args(stage5_partitions),
    )
    return result, upstream


def test_stage_publishes_all_selected_post_events_and_summaries(tmp_path):
    result, (
        source_metadata_dir,
        stage1_dir,
        stage2_dir,
        stage3_dir,
        stage4_dir,
        likes_path,
    ) = _run(
        tmp_path,
        stage3_partitions=2,
        stage5_partitions=3,
    )
    events = scan_parquet_artifact(
        Path(result["artifacts"]["post_liker_events_path"])
    ).collect().sort(["subject_uri", "like_created_at", "liker_did"])
    posts = scan_parquet_artifact(
        Path(result["artifacts"]["post_liker_posts_path"])
    ).collect().sort("subject_uri")

    assert events.schema == pl.Schema(post_liker_history.POST_LIKER_EVENT_SCHEMA)
    assert posts.schema == pl.Schema(post_liker_history.POST_LIKER_POST_SCHEMA)
    assert posts["subject_uri"].to_list() == [
        "history-reply",
        "history-root",
        "negative",
        "positive",
        "zero-like",
    ]
    assert "missing-history" not in posts["subject_uri"].to_list()
    assert "reservoir-only" not in posts["subject_uri"].to_list()
    assert events.filter(pl.col("subject_uri") == "history-root").height == 2
    assert set(events.filter(pl.col("subject_uri") == "positive")["liker_did"]) == {
        "target-user",
        "external-user",
    }
    assert posts.filter(pl.col("subject_uri") == "history-reply")[
        "is_history"
    ].item()
    positive = posts.filter(pl.col("subject_uri") == "positive").row(0, named=True)
    assert positive["is_positive"]
    assert positive["is_negative"]
    assert positive["like_event_count"] == 2
    zero = posts.filter(pl.col("subject_uri") == "zero-like").row(0, named=True)
    assert zero["like_event_count"] == 0
    assert zero["first_like_created_at"] is None
    assert zero["last_like_created_at"] is None
    for part_path in sorted(
        Path(result["artifacts"]["post_liker_events_path"]).glob("*.parquet")
    ):
        part = pl.read_parquet(part_path)
        assert part.equals(part.sort(["subject_uri", "like_created_at", "liker_did"]))

    source_manifest = json.loads(Path(result["artifacts"]["like_sources_path"]).read_text())
    assert [entry["uri"] for entry in source_manifest["files"]] == [str(likes_path)]
    output_dir = Path(result["output_dir"])
    assert not list(output_dir.glob("post_liker_histories_*.partial"))
    assert not list(output_dir.glob("_post_liker_history_staging_*"))
    assert json.loads((output_dir / "manifest.json").read_text())["inputs"] == {
        "00_source_metadata": str(source_metadata_dir.resolve()),
        "01_query_selection": str(stage1_dir.resolve()),
        "02_user_history": str(stage2_dir.resolve()),
        "03_post_selection": str(stage3_dir.resolve()),
        "04_negative_selection": str(stage4_dir.resolve()),
    }
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["extraction"]["selected_post_count"] == 5
    assert summary["extraction"]["valid_source_like_count"] == 7
    assert summary["extraction"]["matched_like_event_count"] == 6
    assert summary["extraction"]["posts_without_likes_count"] == 1
    assert summary["extraction"]["event_count_distribution"]["0"] == 1
    assert "Phase 3/5: stream-sinking all valid events" in (
        output_dir / "stage.log"
    ).read_text()


def test_logical_output_is_independent_of_stage5_partition_count(tmp_path):
    first, _ = _run(
        tmp_path / "one",
        stage3_partitions=2,
        stage5_partitions=1,
    )
    second, _ = _run(
        tmp_path / "five",
        stage3_partitions=2,
        stage5_partitions=5,
    )

    for artifact in ("post_liker_events_path", "post_liker_posts_path"):
        first_df = scan_parquet_artifact(Path(first["artifacts"][artifact])).collect()
        second_df = scan_parquet_artifact(Path(second["artifacts"][artifact])).collect()
        assert first_df.sort(first_df.columns).equals(second_df.sort(second_df.columns))


def test_stage_does_not_relist_ingex_sources(tmp_path, monkeypatch):
    _, _, _, _, stage4_dir, _ = _write_upstream_artifacts(
        tmp_path,
        stage3_partition_count=1,
    )
    monkeypatch.setattr(
        stage.ingex,
        "list_ingex_parquet_files",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not relist")),
    )
    _reset_logger()
    registry.run_stage(
        "post_liker_history",
        _context(tmp_path, stage4_dir, "no-relist"),
        _args(2),
    )


def test_failed_partition_does_not_publish_bundle_or_manifest(tmp_path, monkeypatch):
    _, _, _, _, stage4_dir, _ = _write_upstream_artifacts(
        tmp_path,
        stage3_partition_count=1,
    )
    monkeypatch.setattr(
        stage.post_liker_history_artifacts,
        "process_uri_partitions",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("partition failed")),
    )
    _reset_logger()

    with pytest.raises(RuntimeError, match="partition failed"):
        registry.run_stage(
            "post_liker_history",
            _context(tmp_path, stage4_dir, "failed"),
            _args(2),
        )

    stage5_dir = next((tmp_path / "artifacts" / "05_post_liker_history").iterdir())
    assert list(stage5_dir.glob("post_liker_histories_*.partial"))
    assert not [
        path
        for path in stage5_dir.glob("post_liker_histories_*")
        if not path.name.endswith(".partial")
    ]
    assert not (stage5_dir / "manifest.json").exists()


def test_misaligned_explicit_ancestor_is_rejected(tmp_path):
    _, stage1_dir, _, _, stage4_dir, _ = _write_upstream_artifacts(
        tmp_path,
        stage3_partition_count=1,
    )
    other_stage1 = tmp_path / "artifacts" / "01_query_selection" / "other"
    other_stage1.mkdir(parents=True)
    context = _context(tmp_path, stage4_dir, "misaligned")
    context.prior_outputs["01_query_selection"] = other_stage1
    _reset_logger()

    with pytest.raises(ValueError, match="does not match Stage 4 lineage"):
        registry.run_stage("post_liker_history", context, _args(2))

    assert stage1_dir != other_stage1


def test_config_requires_positive_partition_count():
    with pytest.raises(ValueError, match="post_liker_history_partition_count"):
        stage.build_config(_args(0))
