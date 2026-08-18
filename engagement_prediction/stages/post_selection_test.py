from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from engagement_prediction.data import post_selection as post_data
from engagement_prediction.data.parquet import scan_parquet_artifact
from engagement_prediction.pipeline import registry
from engagement_prediction.pipeline.core import Context
from engagement_prediction.stages import post_selection as stage


UTC = timezone.utc


def _inference_json(news_score, politics_score=0.0):
    return json.dumps({
        "text": {
            "message.commit.record.text": {
                "topic": {"News & Social Concern": news_score},
                "text_arbitrary": {"Politics": politics_score},
            }
        }
    })


def _write_upstream_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    stage1_dir = tmp_path / "artifacts" / "01_query_selection" / "stage1"
    stage1_dir.mkdir(parents=True)
    pl.DataFrame({
        "did": ["u1", "u1"],
        "query_hour": [
            datetime(2026, 1, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 1, 12, tzinfo=UTC),
        ],
        "user_cohort": ["trainval", "trainval"],
        "split": ["train", "train"],
        "positive_count": pl.Series([2, 1], dtype=pl.UInt32),
    }).write_parquet(stage1_dir / "queries_stage1.parquet")
    pl.DataFrame({
        "did": ["u1", "u1", "u1"],
        "query_hour": [
            datetime(2026, 1, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 1, 12, tzinfo=UTC),
        ],
        "subject_uri": ["positive", "overlap", "positive"],
        "like_created_at": [
            datetime(2026, 1, 1, 10, 5, tzinfo=UTC),
            datetime(2026, 1, 1, 10, 6, tzinfo=UTC),
            datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
        ],
    }).write_parquet(stage1_dir / "query_positives_stage1.parquet")
    (stage1_dir / "manifest.json").write_text(json.dumps({
        "stage_key": "query_selection",
        "stage_folder": "01_query_selection",
        "inputs": {},
    }) + "\n")

    stage2_dir = tmp_path / "artifacts" / "02_user_history" / "stage2"
    stage2_dir.mkdir(parents=True)
    history_uris = stage2_dir / "history_post_uris_stage2"
    history_uris.mkdir()
    pl.DataFrame({
        "subject_uri": ["history", "missing", "overlap"],
    }).write_parquet(history_uris / "part-00000.parquet")
    query_histories = stage2_dir / "query_histories_stage2"
    query_histories.mkdir()
    pl.DataFrame({"unused": ["Stage 3 must not read this"]}).write_parquet(
        query_histories / "part-00000.parquet"
    )
    (stage2_dir / "manifest.json").write_text(json.dumps({
        "stage_key": "user_history",
        "stage_folder": "02_user_history",
        "inputs": {"01_query_selection": str(stage1_dir.resolve())},
    }) + "\n")
    return stage1_dir, stage2_dir


def _write_sources(tmp_path: Path) -> tuple[Path, Path]:
    posts_path = tmp_path / "bsky_posts_20260101_000000.parquet"
    pl.DataFrame({
        "at_uri": [
            "positive",
            "overlap",
            "overlap",
            "overlap",
            "history",
            "random",
            "political-a",
            "political-b",
            None,
        ],
        "record_created_at": [
            "2026-01-01T01:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T02:00:00Z",
            "2026-01-01T02:00:00Z",
            "2026-01-01T03:00:00Z",
            "2026-01-01T04:00:00Z",
            "2026-01-01T05:00:00Z",
            "2026-01-01T05:30:00Z",
            "bad",
        ],
        "did": [
            "a-positive",
            "old-author",
            "z-author",
            "a-author",
            "a-history",
            "a-random",
            "a-political",
            "b-political",
            "invalid",
        ],
    }).write_parquet(posts_path)
    inference_path = tmp_path / "bsky_inferences_20260101_120000.parquet"
    pl.DataFrame({
        "at_uri": ["positive", "political-a", "political-b"],
        "indexed_at": [
            "2026-01-01T12:00:00Z",
            "2026-01-01T12:00:00Z",
            "2026-01-01T12:00:00Z",
        ],
        "inferences": [
            _inference_json(0.5, 1.0),
            _inference_json(0.99, 0.0),
            _inference_json(0.99, 0.0),
        ],
    }).write_parquet(inference_path)
    return posts_path, inference_path


def _context(tmp_path: Path, stage1_dir: Path, stage2_dir: Path) -> Context:
    context = Context(
        run_dir=tmp_path / "runs" / "run",
        artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "runs",
        pipeline_run_id="run",
    )
    context.prior_outputs["01_query_selection"] = stage1_dir
    context.prior_outputs["02_user_history"] = stage2_dir
    return context


def _args(partition_count=4, political_cap=1):
    return SimpleNamespace(
        gcs_bucket="unused",
        posts_start="2026-01-01T00:00:00Z",
        posts_end="2026-01-02T00:00:00Z",
        random_candidate_sampling_fraction=1.0,
        max_political_candidates_per_creation_hour=political_cap,
        political_score_threshold=0.95,
        political_inference_window_padding_days=5,
        post_selection_partition_count=partition_count,
        random_seed=42,
        _argv=["--start-from", "post_selection", "--stop-after", "post_selection"],
    )


def _install_source_listing(monkeypatch, posts_path: Path, inference_path: Path, calls):
    def list_sources(*, gcs_bucket, blob_prefix, start, end):
        calls.append((blob_prefix, start, end))
        if blob_prefix == "bsky_posts":
            return [str(posts_path)], [datetime(2026, 1, 1, tzinfo=UTC)]
        return [str(inference_path)], [datetime(2026, 1, 1, 12, tzinfo=UTC)]

    monkeypatch.setattr(stage.ingex, "list_ingex_parquet_files", list_sources)


def test_registry_run_publishes_complete_post_universe_bundle(tmp_path, monkeypatch):
    stage1_dir, stage2_dir = _write_upstream_artifacts(tmp_path)
    posts_source, inference_source = _write_sources(tmp_path)
    calls = []
    _install_source_listing(monkeypatch, posts_source, inference_source, calls)

    result = registry.run_stage(
        "post_selection",
        _context(tmp_path, stage1_dir, stage2_dir),
        _args(),
    )

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

    assert bundle_path.is_dir()
    assert posts.schema == pl.Schema(post_data.POST_SCHEMA)
    assert posts["subject_uri"].to_list() == [
        "history",
        "overlap",
        "political-a",
        "political-b",
        "positive",
        "random",
    ]
    assert posts.filter(pl.col("subject_uri") == "overlap")["author_did"].item() == "a-author"
    assert required.to_dicts() == [
        {"subject_uri": "history", "is_positive": False, "is_history": True},
        {"subject_uri": "missing", "is_positive": False, "is_history": True},
        {"subject_uri": "overlap", "is_positive": True, "is_history": True},
        {"subject_uri": "positive", "is_positive": True, "is_history": False},
    ]
    assert missing["subject_uri"].to_list() == ["missing"]
    assert candidates.filter(pl.col("candidate_source") == "random").height == 6
    assert candidates.filter(pl.col("candidate_source") == "political").height == 1
    assert set(candidates.filter(pl.col("candidate_source") == "political")["subject_uri"]) <= {
        "political-a",
        "political-b",
    }
    assert calls == [
        (
            "bsky_posts",
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
        ),
        (
            "bsky_inferences",
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 7, tzinfo=UTC),
        ),
    ]
    assert list(bundle_path.glob("post_sources_*.json"))
    assert list(bundle_path.glob("inference_sources_*.json"))
    post_sources = json.loads(next(bundle_path.glob("post_sources_*.json")).read_text())
    inference_sources = json.loads(
        next(bundle_path.glob("inference_sources_*.json")).read_text()
    )
    assert [row["uri"] for row in post_sources["files"]] == [str(posts_source)]
    assert post_sources["start"] == "2026-01-01T00:00:00+00:00"
    assert post_sources["end"] == "2026-01-02T00:00:00+00:00"
    assert [row["uri"] for row in inference_sources["files"]] == [
        str(inference_source)
    ]
    assert inference_sources["start"] == "2026-01-01T00:00:00+00:00"
    assert inference_sources["end"] == "2026-01-07T00:00:00+00:00"
    assert not list(output_dir.glob("post_universe_*.partial"))
    assert not list(output_dir.glob("_post_selection_staging_*"))
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["inputs"] == {
        "01_query_selection": str(stage1_dir.resolve()),
        "02_user_history": str(stage2_dir.resolve()),
    }
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["outputs"]["post_count"] == 6
    assert summary["required_post_stats"]["missing_required_post_count"] == 1
    assert summary["political_candidate_stats"]["eligible_count"] == 2
    assert summary["political_candidate_stats"]["selected_count"] == 1
    stage_info = (output_dir / "stage_info.txt").read_text()
    assert "post_file_count: 1" in stage_info
    assert "inference_file_count: 1" in stage_info
    assert "duplicate_post_uri_count: 1" in stage_info
    assert "political_discarded_count: 1" in stage_info
    stage_log = (output_dir / "stage.log").read_text()
    assert "Phase 4/7: normalizing and partitioning raw post source rows" in stage_log
    assert "Scanning and stream-sinking 1 post source files" in stage_log
    assert "Processing URI partition 1/4" in stage_log
    assert "Published post-universe bundle" in stage_log


def test_logical_output_is_partition_count_independent(tmp_path, monkeypatch):
    stage1_dir, stage2_dir = _write_upstream_artifacts(tmp_path)
    posts_source, inference_source = _write_sources(tmp_path)
    _install_source_listing(monkeypatch, posts_source, inference_source, [])

    first = registry.run_stage(
        "post_selection", _context(tmp_path, stage1_dir, stage2_dir), _args(1)
    )
    second = registry.run_stage(
        "post_selection", _context(tmp_path, stage1_dir, stage2_dir), _args(7)
    )

    for artifact in (
        "posts_path",
        "required_posts_path",
        "candidate_sources_path",
        "missing_required_posts_path",
    ):
        first_df = scan_parquet_artifact(Path(first["artifacts"][artifact])).collect()
        second_df = scan_parquet_artifact(Path(second["artifacts"][artifact])).collect()
        first_df = first_df.sort(first_df.columns)
        second_df = second_df.sort(second_df.columns)
        assert first_df.equals(second_df)


def test_political_cap_zero_skips_inference_listing(tmp_path, monkeypatch):
    stage1_dir, stage2_dir = _write_upstream_artifacts(tmp_path)
    posts_source, _ = _write_sources(tmp_path)
    calls = []

    def list_sources(*, gcs_bucket, blob_prefix, start, end):
        calls.append(blob_prefix)
        return [str(posts_source)], [datetime(2026, 1, 1, tzinfo=UTC)]

    monkeypatch.setattr(stage.ingex, "list_ingex_parquet_files", list_sources)
    result = registry.run_stage(
        "post_selection",
        _context(tmp_path, stage1_dir, stage2_dir),
        _args(political_cap=0),
    )

    assert calls == ["bsky_posts"]
    assert "inference_sources_path" not in result["artifacts"]
    candidates = scan_parquet_artifact(
        Path(result["artifacts"]["candidate_sources_path"])
    ).collect()
    assert candidates.filter(pl.col("candidate_source") == "political").is_empty()


def test_invalid_post_rows_publish_schema_correct_empty_universe(tmp_path, monkeypatch):
    stage1_dir, stage2_dir = _write_upstream_artifacts(tmp_path)
    posts_source = tmp_path / "invalid-posts.parquet"
    pl.DataFrame({
        "at_uri": [None],
        "record_created_at": ["bad"],
        "did": [None],
    }).write_parquet(posts_source)

    monkeypatch.setattr(
        stage.ingex,
        "list_ingex_parquet_files",
        lambda **kwargs: (
            [str(posts_source)],
            [datetime(2026, 1, 1, tzinfo=UTC)],
        ),
    )
    result = registry.run_stage(
        "post_selection",
        _context(tmp_path, stage1_dir, stage2_dir),
        _args(political_cap=0),
    )

    posts = scan_parquet_artifact(Path(result["artifacts"]["posts_path"])).collect()
    candidates = scan_parquet_artifact(
        Path(result["artifacts"]["candidate_sources_path"])
    ).collect()
    missing = scan_parquet_artifact(
        Path(result["artifacts"]["missing_required_posts_path"])
    ).collect()
    assert posts.is_empty()
    assert posts.schema == pl.Schema(post_data.POST_SCHEMA)
    assert candidates.is_empty()
    assert candidates.schema == pl.Schema(post_data.CANDIDATE_SOURCE_SCHEMA)
    assert missing.height == 4
    assert missing.schema == pl.Schema(post_data.REQUIRED_POST_SCHEMA)


def test_failed_partition_leaves_partial_bundle_and_no_final_bundle(tmp_path, monkeypatch):
    stage1_dir, stage2_dir = _write_upstream_artifacts(tmp_path)
    posts_source, inference_source = _write_sources(tmp_path)
    _install_source_listing(monkeypatch, posts_source, inference_source, [])

    def fail(*args, **kwargs):
        raise RuntimeError("partition failed")

    monkeypatch.setattr(
        stage.post_selection_artifacts.post_data,
        "select_latest_post_rows",
        fail,
    )
    with pytest.raises(RuntimeError, match="partition failed"):
        registry.run_stage(
            "post_selection",
            _context(tmp_path, stage1_dir, stage2_dir),
            _args(),
        )

    stage3_dir = next((tmp_path / "artifacts" / "03_post_selection").iterdir())
    assert list(stage3_dir.glob("post_universe_*.partial"))
    assert not [
        path for path in stage3_dir.glob("post_universe_*")
        if not path.name.endswith(".partial")
    ]
    assert not (stage3_dir / "manifest.json").exists()


def test_stage_requires_new_history_uri_artifact(tmp_path, monkeypatch):
    stage1_dir, stage2_dir = _write_upstream_artifacts(tmp_path)
    for path in stage2_dir.glob("history_post_uris_*"):
        for part in path.glob("*.parquet"):
            part.unlink()
        path.rmdir()
    posts_source, inference_source = _write_sources(tmp_path)
    _install_source_listing(monkeypatch, posts_source, inference_source, [])

    with pytest.raises(FileNotFoundError, match="history_post_uris"):
        registry.run_stage(
            "post_selection",
            _context(tmp_path, stage1_dir, stage2_dir),
            _args(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("random_candidate_sampling_fraction", 1.1),
        ("max_political_candidates_per_creation_hour", -1),
        ("political_score_threshold", -0.1),
        ("political_inference_window_padding_days", -1),
        ("post_selection_partition_count", 0),
    ],
)
def test_config_validation(field, value):
    args = _args()
    setattr(args, field, value)
    with pytest.raises(ValueError, match=field):
        stage.build_config(args)
