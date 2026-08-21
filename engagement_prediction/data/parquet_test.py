from pathlib import Path

import polars as pl
import pytest

from engagement_prediction.data.parquet import (
    find_artifact_path,
    load_parquet_from_prior,
    read_parquet_parts,
    scan_parquet_artifact,
    sink_partitioned_parquet,
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
