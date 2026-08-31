from pathlib import Path

import polars as pl
import pytest

from engagement_prediction.data.parquet import (
    ensure_typed_parquet_dataset,
    find_artifact_path,
    load_parquet_from_prior,
    read_parquet_parts,
    scan_parquet_artifact,
    sink_partitioned_parquet,
    validate_parquet_part_schemas,
    write_parquet_part_if_not_empty,
)


def test_load_parquet_from_prior_supports_single_file(tmp_path):
    artifact = Path(tmp_path) / "queries_run.parquet"
    pl.DataFrame({"value": [1, 2]}).write_parquet(artifact)

    assert load_parquet_from_prior(Path(tmp_path), "queries_").collect()["value"].to_list() == [1, 2]


def test_load_parquet_from_prior_supports_partitioned_dataset(tmp_path):
    artifact = Path(tmp_path) / "query_histories_run"
    artifact.mkdir()
    pl.DataFrame({"value": [1]}).write_parquet(artifact / "part-00000.parquet")
    nested = artifact / "nested"
    nested.mkdir()
    pl.DataFrame({"value": [2]}).write_parquet(nested / "part-00001.parquet")

    assert load_parquet_from_prior(Path(tmp_path), "query_histories_").collect()["value"].sort().to_list() == [1, 2]


def test_parquet_helpers_ignore_partial_and_reject_empty_dataset(tmp_path):
    partial = Path(tmp_path) / "queries_run.partial.parquet"
    pl.DataFrame({"value": [1]}).write_parquet(partial)
    with pytest.raises(FileNotFoundError):
        find_artifact_path(Path(tmp_path), "queries_")

    empty = Path(tmp_path) / "query_histories_run"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No Parquet parts"):
        scan_parquet_artifact(empty)


def test_sink_and_read_partitioned_parquet_parts(tmp_path):
    output_path = Path(tmp_path) / "partitioned"
    sink_partitioned_parquet(
        pl.DataFrame({"value": [1, 2], "partition": [0, 1]}).lazy(),
        output_path=output_path,
        key="partition",
    )

    first_partition_paths = sorted((output_path / "partition=0").glob("*.parquet"))
    assert read_parquet_parts(first_partition_paths)["value"].to_list() == [1]


def test_read_parquet_parts_supports_schema_correct_empty_frame():
    empty = pl.DataFrame(schema={"value": pl.Int64})

    assert read_parquet_parts([], empty=empty).schema == empty.schema
    with pytest.raises(ValueError, match="Expected at least one Parquet part"):
        read_parquet_parts([])


def test_sparse_part_writer_and_typed_empty_dataset(tmp_path):
    dataset_path = Path(tmp_path) / "dataset"
    schema = {"subject_uri": pl.String, "count": pl.UInt64}

    assert not write_parquet_part_if_not_empty(
        pl.DataFrame(schema=schema),
        dataset_path / "part-00001.parquet",
    )
    ensure_typed_parquet_dataset(dataset_path, schema)

    parts = sorted(dataset_path.glob("*.parquet"))
    assert [part.name for part in parts] == ["part-00000.parquet"]
    assert pl.read_parquet(parts).schema == pl.Schema(schema)


def test_sparse_part_writer_keeps_nonempty_rows(tmp_path):
    dataset_path = Path(tmp_path) / "dataset"
    part_path = dataset_path / "part-00003.parquet"

    assert write_parquet_part_if_not_empty(
        pl.DataFrame({"value": [3]}),
        part_path,
    )
    assert pl.read_parquet(part_path).get_column("value").to_list() == [3]


def test_validate_parquet_part_schemas_checks_each_physical_part(tmp_path):
    dataset_path = Path(tmp_path) / "dataset"
    dataset_path.mkdir()
    schema = {"value": pl.Int64}
    pl.DataFrame({"value": [1]}, schema=schema).write_parquet(
        dataset_path / "part-00000.parquet"
    )
    pl.DataFrame({"value": [2]}, schema=schema).write_parquet(
        dataset_path / "part-00001.parquet"
    )

    assert validate_parquet_part_schemas(dataset_path, schema) == 2
    pl.DataFrame({"value": ["wrong"]}).write_parquet(
        dataset_path / "part-00001.parquet"
    )
    with pytest.raises(ValueError, match="Unexpected Parquet schema"):
        validate_parquet_part_schemas(dataset_path, schema)
