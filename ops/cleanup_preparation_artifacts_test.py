import json
from pathlib import Path
import shutil

import pytest

from engagement_prediction.data.datasets_test import _bundle
from engagement_prediction.evaluation.artifacts import resolve_stage7_artifact
from ops import cleanup_preparation_artifacts as cleanup


STAGE_FOLDERS = (
    "00_source_metadata",
    "01_query_selection",
    "02_user_history",
    "03_post_selection",
    "04_negative_selection",
    "05_post_liker_history",
    "06_author_statistics",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def _snapshot(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


@pytest.fixture
def artifacts(tmp_path, request):
    lineage: dict[str, Path] = {}
    payloads: list[Path] = []
    for index, stage_folder in enumerate(STAGE_FOLDERS):
        stage_dir = tmp_path / "artifacts" / stage_folder / f"20260101_00000{index}_test"
        stage_dir.mkdir(parents=True)
        _write_json(stage_dir / "manifest.json", {
            "stage_key": stage_folder[3:],
            "stage_folder": stage_folder,
            "stage_run_id": stage_dir.name,
            "status": "complete",
            "inputs": {name: str(path) for name, path in lineage.items()},
        })
        lineage[stage_folder] = stage_dir
        _write_json(stage_dir / "summary.json", {"stage": stage_folder})
        _write_json(stage_dir / "resolved_config.json", {"max_history_len": 32})
        (stage_dir / "stage_info.txt").write_text(stage_folder)
        (stage_dir / "stage.log").write_text("Completed successfully")
        suffix = stage_dir.name
        if index == 0:
            bundle = stage_dir / f"source_metadata_{suffix}"
            data_paths = [bundle / "post_metadata"]
        elif index == 1:
            bundle = stage_dir
            data_paths = [
                stage_dir / f"queries_{suffix}.parquet",
                stage_dir / f"query_positives_{suffix}.parquet",
                stage_dir / f"_candidate_query_counts_{suffix}.parquet",
            ]
        elif index == 2:
            bundle = stage_dir
            data_paths = [
                stage_dir / f"query_histories_{suffix}",
                stage_dir / f"history_post_uris_{suffix}",
            ]
        elif index == 3:
            bundle = stage_dir / f"post_universe_{suffix}"
            data_paths = [bundle / name for name in (
                "posts", "required_posts", "candidate_sources", "missing_required_posts"
            )]
        elif index == 4:
            bundle = stage_dir / f"negative_candidates_{suffix}"
            data_paths = [bundle / "hourly_candidates", bundle / "negative_post_uris"]
        elif index == 5:
            bundle = stage_dir / f"post_liker_histories_{suffix}"
            data_paths = [bundle / "post_liker_events", bundle / "post_liker_posts"]
        else:
            bundle = stage_dir / f"author_statistics_{suffix}"
            data_paths = [bundle / "author_statistics"]
        bundle.mkdir(exist_ok=True)
        _write_json(bundle / f"like_sources_{suffix}.json", {"files": []})
        for path in data_paths:
            if path.suffix == ".parquet":
                path.write_bytes(b"preparation payload")
            else:
                path.mkdir()
                (path / "part-00000.parquet").write_bytes(b"preparation payload")
            if index in {0, 1, 2, 3, 5}:
                payloads.append(path)

    stage7_dir = tmp_path / "artifacts" / "07_dataset_hydration" / "20260101_000007_test"
    stage7_dir.mkdir(parents=True)
    bundle = _bundle(stage7_dir, **getattr(request, "param", {}))
    _write_json(stage7_dir / "manifest.json", {
        "stage_key": "dataset_hydration",
        "stage_folder": "07_dataset_hydration",
        "stage_run_id": stage7_dir.name,
        "status": "complete",
        "inputs": {name: str(path) for name, path in lineage.items()},
    })
    _write_json(stage7_dir / "summary.json", {
        "parameters": {"embedding_model": "all_MiniLM_L12_v2", "embedding_dim": 2},
        "outputs": {"hydrated_training_data_path": bundle.name},
    })
    return stage7_dir, lineage, tuple(payloads)


def test_dry_run_reports_exact_lineage_without_writing_or_removing(artifacts, tmp_path, capsys):
    stage7_dir, lineage, payloads = artifacts
    unrelated = tmp_path / "artifacts" / "05_post_liker_history" / "unreferenced"
    unrelated.mkdir()
    (unrelated / "keep.parquet").write_bytes(b"unrelated")
    before = _snapshot(tmp_path)

    plan = cleanup.build_cleanup_plan(stage7_dir)
    assert plan.stage7_dir == stage7_dir
    assert plan.lineage == {**lineage, "07_dataset_hydration": stage7_dir}
    assert {candidate.path for candidate in plan.candidates} == set(payloads)
    assert {candidate.stage_folder for candidate in plan.candidates} == {
        STAGE_FOLDERS[index] for index in (0, 1, 2, 3, 5)
    }
    assert all(candidate.file_count == 1 for candidate in plan.candidates)
    assert plan.total_bytes == len(payloads) * len(b"preparation payload")
    assert plan.total_allocated_bytes == sum(
        candidate.allocated_bytes for candidate in plan.candidates
    )
    assert cleanup.main(["--stage7", str(stage7_dir)]) == 0

    assert str(payloads[0]) in capsys.readouterr().out
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("artifacts", [
    {"legacy_history_columns": True, "prepare_post_liker_features": False},
    {"legacy_history_columns": True, "prepare_post_liker_features": True},
], indirect=True)
def test_legacy_wide_history_bundles_are_supported_with_either_index_format(artifacts):
    stage7_dir, _, payloads = artifacts

    plan = cleanup.build_cleanup_plan(stage7_dir)

    assert {candidate.path for candidate in plan.candidates} == set(payloads)


def test_apply_removes_only_allowlisted_payloads_and_is_repeatable(artifacts, tmp_path):
    stage7_dir, lineage, payloads = artifacts
    before = _snapshot(tmp_path)
    removed_files = {
        path.relative_to(tmp_path)
        for payload in payloads
        for path in ([payload] if payload.is_file() else payload.rglob("*.parquet"))
    }
    plan = cleanup.build_cleanup_plan(stage7_dir)

    report_path = cleanup.apply_cleanup(plan)

    assert report_path.parent == stage7_dir
    report = json.loads(report_path.read_text())
    assert report["status"] == "complete"
    assert report["stage7_dir"] == str(stage7_dir)
    assert report["lineage"] == {
        name: str(path)
        for name, path in {**lineage, "07_dataset_hydration": stage7_dir}.items()
    }
    assert {entry["path"] for entry in report["candidates"]} == {str(path) for path in payloads}
    assert all(entry["status"] == "deleted" for entry in report["candidates"])
    assert all(not payload.exists() for payload in payloads)
    for relative, contents in before.items():
        if relative not in removed_files:
            assert (tmp_path / relative).read_bytes() == contents
    # Training and evaluation still have a real, validated Stage 7 loader bundle.
    assert resolve_stage7_artifact(stage7_dir).embedding_count == 5
    assert not cleanup.build_cleanup_plan(stage7_dir).candidates

    second_report = json.loads(cleanup.apply_cleanup(plan).read_text())
    assert second_report["status"] == "complete"
    assert all(entry["status"] == "already_absent" for entry in second_report["candidates"])


def test_cli_apply_removes_payloads_only_when_requested(artifacts):
    stage7_dir, _, payloads = artifacts

    assert cleanup.main(["--stage7", str(stage7_dir), "--apply"]) == 0

    assert all(not payload.exists() for payload in payloads)
    assert resolve_stage7_artifact(stage7_dir).embedding_count == 5


def test_cli_failure_keeps_audit_of_completed_and_pending_deletions(artifacts, monkeypatch):
    stage7_dir, _, payloads = artifacts
    failed_payload = payloads[-2]
    original_rmtree = cleanup.shutil.rmtree

    def failing_rmtree(path):
        if path == failed_payload:
            raise OSError("injected cleanup failure")
        original_rmtree(path)

    monkeypatch.setattr(cleanup.shutil, "rmtree", failing_rmtree)

    assert cleanup.main(["--stage7", str(stage7_dir), "--apply"]) == 1

    reports = list(stage7_dir.glob("cleanup_preparation_report_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["status"] == "failed"
    assert report["finished_at"]
    assert report["error"] == "injected cleanup failure"
    statuses = {entry["path"]: entry["status"] for entry in report["candidates"]}
    assert all(statuses[str(path)] == "deleted" for path in payloads[:-2])
    assert all(statuses[str(path)] == "pending" for path in payloads[-2:])
    assert all(not path.exists() for path in payloads[:-2])
    assert all(path.exists() for path in payloads[-2:])


@pytest.mark.parametrize("failure", ["incomplete", "missing_ancestor", "misaligned_ancestor"])
def test_invalid_completion_or_lineage_aborts_without_deletion(artifacts, failure):
    stage7_dir, lineage, payloads = artifacts
    if failure == "incomplete":
        manifest_path = stage7_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "partial"
    elif failure == "missing_ancestor":
        manifest_path = stage7_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        del manifest["inputs"]["02_user_history"]
    else:
        manifest_path = lineage["05_post_liker_history"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["inputs"]["02_user_history"] = str(stage7_dir)
    _write_json(manifest_path, manifest)

    with pytest.raises((ValueError, FileNotFoundError)):
        cleanup.build_cleanup_plan(stage7_dir)

    assert all(payload.exists() for payload in payloads)


@pytest.mark.parametrize("missing_path", [
    "query_histories/part-00000.parquet",
    "loader_index/post_author_idx.npy",
])
def test_incomplete_stage7_payload_aborts_without_deletion(artifacts, missing_path):
    stage7_dir, _, payloads = artifacts
    (stage7_dir / "hydrated_training_data_test" / missing_path).unlink()

    with pytest.raises((ValueError, FileNotFoundError)):
        cleanup.build_cleanup_plan(stage7_dir)

    assert all(payload.exists() for payload in payloads)


@pytest.mark.parametrize("link_kind", ["payload", "broken_payload", "nested_file", "bundle_parent"])
def test_payload_symlinks_are_rejected_without_touching_targets(artifacts, tmp_path, link_kind):
    stage7_dir, _, payloads = artifacts
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "part-00000.parquet"
    sentinel.write_bytes(b"must survive")
    payload = payloads[0]
    if link_kind in {"payload", "broken_payload"}:
        shutil.rmtree(payload)
        target = outside if link_kind == "payload" else tmp_path / "missing"
        payload.symlink_to(target, target_is_directory=True)
    elif link_kind == "nested_file":
        (payload / "part-00000.parquet").unlink()
        (payload / "part-00000.parquet").symlink_to(sentinel)
    else:
        bundle = payload.parent
        original = outside / "original_bundle"
        bundle.rename(original)
        bundle.symlink_to(original, target_is_directory=True)

    with pytest.raises(ValueError, match="[Ss]ymlink"):
        cleanup.build_cleanup_plan(stage7_dir)

    assert sentinel.read_bytes() == b"must survive"
    assert all(path.exists() for path in payloads[1:])


def test_run_view_link_is_resolved_but_bundle_argument_is_rejected(artifacts, tmp_path):
    stage7_dir, _, payloads = artifacts
    run_view = tmp_path / "run-stage7"
    run_view.symlink_to(stage7_dir, target_is_directory=True)

    assert cleanup.build_cleanup_plan(run_view).stage7_dir == stage7_dir
    with pytest.raises(ValueError):
        cleanup.build_cleanup_plan(stage7_dir / "hydrated_training_data_test")

    assert all(path.exists() for path in payloads)


def test_stage7_bundle_link_outside_stage_directory_is_rejected(artifacts, tmp_path):
    stage7_dir, _, payloads = artifacts
    bundle = stage7_dir / "hydrated_training_data_test"
    outside_parent = tmp_path / "outside"
    outside_parent.mkdir()
    outside_bundle = outside_parent / bundle.name
    bundle.rename(outside_bundle)
    bundle.symlink_to(outside_bundle, target_is_directory=True)

    with pytest.raises(ValueError):
        cleanup.build_cleanup_plan(stage7_dir)

    assert (outside_bundle / "embeddings.npy").is_file()
    assert all(path.exists() for path in payloads)


def test_apply_rechecks_payload_links_after_dry_run(artifacts, tmp_path):
    stage7_dir, _, payloads = artifacts
    plan = cleanup.build_cleanup_plan(stage7_dir)
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"must survive")
    replaced = payloads[-1] / "part-00000.parquet"
    replaced.unlink()
    replaced.symlink_to(outside)

    with pytest.raises(ValueError, match="[Ss]ymlink"):
        cleanup.apply_cleanup(plan)

    assert outside.read_bytes() == b"must survive"
    assert all(path.exists() for path in payloads)


def test_apply_revalidates_retained_stage7_after_dry_run(artifacts):
    stage7_dir, _, payloads = artifacts
    plan = cleanup.build_cleanup_plan(stage7_dir)
    shutil.rmtree(stage7_dir / "hydrated_training_data_test")

    with pytest.raises((ValueError, FileNotFoundError)):
        cleanup.apply_cleanup(plan)

    assert all(path.exists() for path in payloads)
    assert not list(stage7_dir.glob("cleanup_preparation_report_*.json"))


def test_apply_rejects_ancestor_replaced_by_symlink_after_dry_run(artifacts, tmp_path):
    stage7_dir, lineage, payloads = artifacts
    plan = cleanup.build_cleanup_plan(stage7_dir)
    ancestor = lineage["05_post_liker_history"]
    outside = tmp_path / "relocated_ancestor"
    ancestor.rename(outside)
    ancestor.symlink_to(outside, target_is_directory=True)
    before = _snapshot(outside)

    with pytest.raises(ValueError, match="[Ss]ymlink"):
        cleanup.apply_cleanup(plan)

    assert _snapshot(outside) == before
    assert all(path.exists() for path in payloads)
    assert not list(stage7_dir.glob("cleanup_preparation_report_*.json"))


def test_unexpected_metadata_inside_payload_aborts_without_deletion(artifacts):
    stage7_dir, _, payloads = artifacts
    (payloads[0] / "keep.json").write_text("{}")

    with pytest.raises(ValueError):
        cleanup.build_cleanup_plan(stage7_dir)

    assert all(path.exists() for path in payloads)
