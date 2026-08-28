import json
from pathlib import Path

import polars as pl
import pytest

from engagement_prediction.pipeline.artifacts import (
    PartialArtifactBundle,
    complete_stage_artifacts,
)


def _publication(tmp_path: Path) -> PartialArtifactBundle:
    return PartialArtifactBundle.create(
        output_dir=tmp_path,
        bundle_name="bundle_run",
        staging_name="_staging_run.partial",
        dataset_schemas={
            "rows": {"subject_uri": pl.String, "count": pl.UInt64},
            "empty_rows": {"subject_uri": pl.String},
        },
    )


def test_partial_bundle_publishes_validated_datasets_and_metadata(tmp_path):
    publication = _publication(Path(tmp_path))
    rows_path = publication.public_path("rows")
    rows_path.mkdir()
    pl.DataFrame(
        {"subject_uri": ["at://post"], "count": [2]},
        schema={"subject_uri": pl.String, "count": pl.UInt64},
    ).write_parquet(rows_path / "part-00003.parquet")
    (publication.staging_path / "diagnostic.txt").write_text("temporary")

    part_counts = publication.publish(
        summary={"row_count": 1},
        stage_info="stage: test",
    )

    assert part_counts == {"rows": 1, "empty_rows": 1}
    assert publication.final_path.is_dir()
    assert not publication.partial_path.exists()
    assert not publication.staging_path.exists()
    empty_part = publication.final_public_path("empty_rows/part-00000.parquet")
    assert pl.read_parquet(empty_part).schema == pl.Schema({"subject_uri": pl.String})
    assert json.loads((Path(tmp_path) / "summary.json").read_text()) == {"row_count": 1}
    assert (Path(tmp_path) / "stage_info.txt").read_text() == "stage: test\n"


def test_partial_bundle_schema_failure_retains_diagnostics(tmp_path):
    publication = _publication(Path(tmp_path))
    rows_path = publication.public_path("rows")
    rows_path.mkdir()
    pl.DataFrame({"subject_uri": ["at://post"], "count": ["wrong"]}).write_parquet(
        rows_path / "part-00000.parquet"
    )

    with pytest.raises(ValueError, match="Unexpected Parquet schema"):
        publication.publish(summary={}, stage_info="stage: test")

    assert publication.partial_path.is_dir()
    assert publication.staging_path.is_dir()
    assert not publication.final_path.exists()
    assert not (Path(tmp_path) / "summary.json").exists()
    assert not (Path(tmp_path) / "stage_info.txt").exists()


def test_partial_bundle_rolls_back_post_rename_failure(tmp_path, monkeypatch):
    publication = _publication(Path(tmp_path))
    original_replace = Path.replace

    def fail_stage_info_replace(path, target):
        if path.name == "stage_info.txt.partial":
            raise OSError("stage-info promotion failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_stage_info_replace)
    with pytest.raises(OSError, match="stage-info promotion failed"):
        publication.publish(summary={"status": "partial"}, stage_info="stage: test")

    assert publication.partial_path.is_dir()
    assert not publication.final_path.exists()
    assert not (Path(tmp_path) / "summary.json").exists()
    assert (Path(tmp_path) / "summary.json.partial").is_file()
    assert not (Path(tmp_path) / "stage_info.txt").exists()
    assert (Path(tmp_path) / "stage_info.txt.partial").is_file()


@pytest.mark.parametrize(
    "existing_name",
    ["bundle_run.partial", "_staging_run.partial"],
)
def test_partial_bundle_creation_rejects_existing_paths_without_new_paths(
    tmp_path,
    existing_name,
):
    existing_path = Path(tmp_path) / existing_name
    existing_path.mkdir()

    with pytest.raises(FileExistsError, match="already exist"):
        _publication(Path(tmp_path))

    assert existing_path.is_dir()
    other_name = (
        "_staging_run.partial"
        if existing_name == "bundle_run.partial"
        else "bundle_run.partial"
    )
    assert not (Path(tmp_path) / other_name).exists()


def test_complete_stage_artifacts_records_then_finalizes(tmp_path):
    calls = []

    class RecordingContext:
        def record_artifact(self, stage, output_dir, extras):
            calls.append(("record", stage, output_dir, extras))

        def finalize_stage(self, **kwargs):
            calls.append(("finalize", kwargs))

    output_dir = Path(tmp_path) / "output"
    output_dir.mkdir()
    result = {
        "output_dir": output_dir,
        "artifacts": {"bundle_path": str(output_dir / "bundle")},
    }
    args = type("Args", (), {"_argv": ["run-all"]})()

    assert complete_stage_artifacts(
        context=RecordingContext(),
        stage_key="test_stage",
        stage_folder="00_test",
        result=result,
        args=args,
    ) is result
    assert calls[0] == (
        "record",
        "test_stage",
        output_dir,
        result["artifacts"],
    )
    assert calls[1][0] == "finalize"
    assert calls[1][1]["argv"] == ["run-all"]
