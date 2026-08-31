from pathlib import Path

import polars as pl

from engagement_prediction.data import post_selection_artifacts


def _materialize(tmp_path: Path, partition_count: int) -> pl.DataFrame:
    output_path = tmp_path / f"required-{partition_count}"
    post_selection_artifacts.materialize_required_rows(
        query_positives_lf=pl.DataFrame({
            "subject_uri": ["positive", "overlap", "positive"],
        }).lazy(),
        history_post_uris_lf=pl.DataFrame({
            "subject_uri": ["history", "overlap"],
        }).lazy(),
        output_path=output_path,
        partition_count=partition_count,
    )
    return pl.read_parquet(sorted(output_path.rglob("*.parquet"))).sort(
        ["subject_uri", "is_positive", "is_history"]
    )


def test_required_row_routing_is_logically_partition_count_independent(tmp_path):
    one_partition = _materialize(tmp_path, 1)
    seven_partitions = _materialize(tmp_path, 7)

    assert one_partition.equals(seven_partitions)
    assert one_partition.to_dicts() == [
        {"subject_uri": "history", "is_positive": False, "is_history": True},
        {"subject_uri": "overlap", "is_positive": False, "is_history": True},
        {"subject_uri": "overlap", "is_positive": True, "is_history": False},
        {"subject_uri": "positive", "is_positive": True, "is_history": False},
        {"subject_uri": "positive", "is_positive": True, "is_history": False},
    ]
