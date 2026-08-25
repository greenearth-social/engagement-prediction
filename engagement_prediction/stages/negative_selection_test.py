from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from engagement_prediction.data import ingex, negative_selection, post_selection
from engagement_prediction.data.parquet import scan_parquet_artifact
from engagement_prediction.pipeline import registry
from engagement_prediction.pipeline.core import Context
from engagement_prediction.stages import negative_selection as stage
from utils import helpers


UTC = timezone.utc


def _write_partitioned_stage3_dataset(
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
    partition_count: int,
) -> tuple[Path, Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    likes_path = tmp_path / f"likes-{partition_count}.parquet"
    pl.DataFrame({
        "did": ["u"] * 8 + [None],
        "subject_uri": ["p1", "p1", "p2", "p3", "p3", "p3", "positive", "outside", "p1"],
        "record_created_at": [
            "2026-01-01T10:05:00Z",
            "2026-01-01T10:05:00Z",
            "2026-01-01T09:00:00Z",
            "2026-01-01T08:00:00Z",
            "2026-01-01T08:01:00Z",
            "2026-01-01T08:02:00Z",
            "2026-01-01T09:30:00Z",
            "2026-01-01T09:00:00Z",
            "2026-01-01T09:00:00Z",
        ],
    }).write_parquet(likes_path)

    root = tmp_path / "artifacts"
    source_metadata_dir = root / "00_source_metadata" / f"stage0-{partition_count}"
    source_bundle = source_metadata_dir / f"source_metadata_stage0-{partition_count}"
    source_metadata_frame = pl.DataFrame({
        "subject_uri": ["p1", "p2", "p3", "p4", "positive", "required-only", "reply"],
        "post_created_at": [datetime(2026, 1, 1, 10, tzinfo=UTC)] * 7,
        "author_did": ["a"] * 7,
        "is_reply": [False] * 6 + [True],
    }, schema=post_selection.POST_SCHEMA)
    _write_partitioned_stage3_dataset(
        source_bundle / "post_metadata", source_metadata_frame, partition_count
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
        "parameters": {"source_metadata_partition_count": partition_count},
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
    stage1_dir = root / "01_query_selection" / f"stage1-{partition_count}"
    stage1_dir.mkdir(parents=True)
    pl.DataFrame({
        "did": ["u1", "u2", "u1"],
        "query_hour": [
            datetime(2026, 1, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 1, 11, tzinfo=UTC),
        ],
        "user_cohort": ["trainval"] * 3,
        "split": ["train"] * 3,
        "positive_count": pl.Series([1, 1, 1], dtype=pl.UInt32),
    }).write_parquet(stage1_dir / "queries_stage1.parquet")
    ingex.write_source_manifest(
        stage1_dir / "like_sources_stage1.json",
        ingex.build_source_manifest(
            gcs_bucket="unused",
            blob_prefix="bsky_likes",
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 2, tzinfo=UTC),
            paths=[str(likes_path)],
            timestamps=[datetime(2026, 1, 1, tzinfo=UTC)],
        ),
    )
    (stage1_dir / "manifest.json").write_text(json.dumps({
        "stage_key": "query_selection",
        "stage_folder": "01_query_selection",
        "inputs": {"00_source_metadata": str(source_metadata_dir.resolve())},
    }) + "\n")

    stage2_dir = root / "02_user_history" / f"stage2-{partition_count}"
    stage2_dir.mkdir(parents=True)
    (stage2_dir / "manifest.json").write_text(json.dumps({
        "stage_key": "user_history",
        "stage_folder": "02_user_history",
        "inputs": {
            "00_source_metadata": str(source_metadata_dir.resolve()),
            "01_query_selection": str(stage1_dir.resolve()),
        },
    }) + "\n")

    stage3_dir = root / "03_post_selection" / f"stage3-{partition_count}"
    bundle_path = stage3_dir / f"post_universe_stage3-{partition_count}"
    stage3_dir.mkdir(parents=True)
    posts = pl.DataFrame({
        "subject_uri": ["p1", "p2", "p3", "p4", "positive", "required-only", "reply"],
        "post_created_at": [
            datetime(2026, 1, 1, 10, 30, tzinfo=UTC),
            datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
        ],
        "author_did": ["a"] * 7,
        "is_reply": [False] * 6 + [True],
    }, schema=post_selection.POST_SCHEMA)
    sources = pl.DataFrame({
        "subject_uri": ["p1", "p2", "p3", "p4", "positive"],
        "candidate_source": ["random"] * 5,
    }, schema=post_selection.CANDIDATE_SOURCE_SCHEMA)
    _write_partitioned_stage3_dataset(bundle_path / "posts", posts, partition_count)
    _write_partitioned_stage3_dataset(
        bundle_path / "candidate_sources", sources, partition_count
    )
    (stage3_dir / "summary.json").write_text(json.dumps({
        "parameters": {"source_metadata_partition_count": partition_count},
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
    return source_metadata_dir, stage1_dir, stage2_dir, stage3_dir, likes_path


def _args(**overrides):
    values = {
        "negative_candidates_per_hour": 3,
        "min_likes_for_popular_candidate": 2,
        "popular_candidate_fraction": 0.5,
        "max_candidate_age_hours": 2,
        "random_seed": 42,
        "_argv": ["--start-from", "negative_selection", "--stop-after", "negative_selection"],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _context(tmp_path: Path, stage3_dir: Path, suffix: str) -> Context:
    context = Context(
        run_dir=tmp_path / "runs" / suffix,
        artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "runs",
        pipeline_run_id=suffix,
    )
    context.prior_outputs["03_post_selection"] = stage3_dir
    return context


def _reset_stage_logger() -> None:
    logger = helpers._stage_loggers.pop("04_NEGATIVE_SELECTION", None)
    if logger is not None:
        for handler in logger.handlers:
            handler.close()


def _run(tmp_path: Path, *, partition_count: int):
    _reset_stage_logger()
    source_metadata_dir, stage1_dir, stage2_dir, stage3_dir, likes_path = _write_upstream_artifacts(
        tmp_path,
        partition_count=partition_count,
    )
    result = registry.run_stage(
        "negative_selection",
        _context(tmp_path, stage3_dir, f"run-{partition_count}"),
        _args(),
    )
    return result, source_metadata_dir, stage1_dir, stage2_dir, stage3_dir, likes_path


def test_registry_run_publishes_hourly_candidates_and_unique_uris(tmp_path):
    result, source_metadata_dir, stage1_dir, stage2_dir, stage3_dir, likes_path = _run(
        tmp_path,
        partition_count=3,
    )
    output_dir = Path(result["output_dir"])
    hourly = scan_parquet_artifact(
        Path(result["artifacts"]["hourly_candidates_path"])
    ).collect().sort(["query_hour", "subject_uri"])
    uris = scan_parquet_artifact(
        Path(result["artifacts"]["negative_post_uris_path"])
    ).collect().sort("subject_uri")

    assert hourly.schema == pl.Schema(negative_selection.HOURLY_CANDIDATE_SCHEMA)
    assert uris.schema == pl.Schema(negative_selection.NEGATIVE_POST_URI_SCHEMA)
    assert hourly.group_by("query_hour").len()["len"].to_list() == [3, 3]
    assert hourly.filter(
        (pl.col("query_hour") == datetime(2026, 1, 1, 10, tzinfo=UTC))
        & (pl.col("selection_source") == "popular")
    ).height == 1
    assert hourly.filter(
        (pl.col("query_hour") == datetime(2026, 1, 1, 11, tzinfo=UTC))
        & (pl.col("subject_uri") == "p1")
    )["prior_like_count"].item() == 2
    assert "required-only" not in uris["subject_uri"].to_list()
    assert "reply" not in uris["subject_uri"].to_list()
    assert "positive" in set(hourly["subject_uri"])
    assert set(uris["subject_uri"]) == set(hourly["subject_uri"])

    source_manifest = json.loads(
        Path(result["artifacts"]["like_sources_path"]).read_text()
    )
    assert [entry["uri"] for entry in source_manifest["files"]] == [str(likes_path)]
    assert not list(output_dir.glob("negative_candidates_*.partial"))
    assert not list(output_dir.glob("_negative_selection_staging_*"))
    assert json.loads((output_dir / "manifest.json").read_text())["inputs"] == {
        "00_source_metadata": str(source_metadata_dir.resolve()),
        "01_query_selection": str(stage1_dir.resolve()),
        "02_user_history": str(stage2_dir.resolve()),
        "03_post_selection": str(stage3_dir.resolve()),
    }
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["parameters"]["popular_candidate_quota"] == 2
    assert summary["selection"]["selected_candidate_row_count"] == 6
    assert summary["selection"]["short_query_hour_count"] == 0
    assert "Phase 3/7: calculating strictly prior candidate popularity" in (
        output_dir / "stage.log"
    ).read_text()


def test_logical_output_is_partition_count_independent(tmp_path):
    first, *_ = _run(tmp_path / "one", partition_count=1)
    second, *_ = _run(tmp_path / "three", partition_count=3)

    for artifact in ("hourly_candidates_path", "negative_post_uris_path"):
        first_df = scan_parquet_artifact(Path(first["artifacts"][artifact])).collect()
        second_df = scan_parquet_artifact(Path(second["artifacts"][artifact])).collect()
        assert first_df.sort(first_df.columns).equals(second_df.sort(second_df.columns))


def test_zero_k_publishes_schema_correct_empty_outputs(tmp_path):
    _reset_stage_logger()
    _, _, _, stage3_dir, _ = _write_upstream_artifacts(tmp_path, partition_count=2)
    result = registry.run_stage(
        "negative_selection",
        _context(tmp_path, stage3_dir, "zero"),
        _args(negative_candidates_per_hour=0),
    )

    hourly = scan_parquet_artifact(
        Path(result["artifacts"]["hourly_candidates_path"])
    ).collect()
    uris = scan_parquet_artifact(
        Path(result["artifacts"]["negative_post_uris_path"])
    ).collect()
    assert hourly.is_empty()
    assert hourly.schema == pl.Schema(negative_selection.HOURLY_CANDIDATE_SCHEMA)
    assert uris.is_empty()
    assert uris.schema == pl.Schema(negative_selection.NEGATIVE_POST_URI_SCHEMA)


def test_failed_partition_does_not_publish_bundle_or_manifest(tmp_path, monkeypatch):
    _reset_stage_logger()
    _, _, _, stage3_dir, _ = _write_upstream_artifacts(tmp_path, partition_count=2)
    monkeypatch.setattr(
        stage.negative_selection_artifacts,
        "process_uri_partitions",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("partition failed")),
    )

    with pytest.raises(RuntimeError, match="partition failed"):
        registry.run_stage(
            "negative_selection",
            _context(tmp_path, stage3_dir, "failed"),
            _args(),
        )

    stage4_dir = next((tmp_path / "artifacts" / "04_negative_selection").iterdir())
    assert list(stage4_dir.glob("negative_candidates_*.partial"))
    assert not [
        path
        for path in stage4_dir.glob("negative_candidates_*")
        if not path.name.endswith(".partial")
    ]
    assert not (stage4_dir / "manifest.json").exists()


def test_misaligned_explicit_ancestor_is_rejected(tmp_path):
    _reset_stage_logger()
    _, stage1_dir, _, stage3_dir, _ = _write_upstream_artifacts(
        tmp_path,
        partition_count=1,
    )
    other_stage1 = tmp_path / "artifacts" / "01_query_selection" / "other"
    other_stage1.mkdir(parents=True)
    context = _context(tmp_path, stage3_dir, "misaligned")
    context.prior_outputs["01_query_selection"] = other_stage1

    with pytest.raises(ValueError, match="does not match Stage 3 lineage"):
        registry.run_stage("negative_selection", context, _args())

    assert stage1_dir != other_stage1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("negative_candidates_per_hour", -1),
        ("min_likes_for_popular_candidate", -1),
        ("popular_candidate_fraction", 1.1),
        ("max_candidate_age_hours", 0),
    ],
)
def test_config_validation(field, value):
    args = _args()
    setattr(args, field, value)
    with pytest.raises(ValueError, match=field):
        stage.build_config(args, partition_count=2)
