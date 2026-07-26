import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from utils.stage1_emissions import (
    COUNT_COLUMN,
    SURROGATE_COLUMN,
    write_concentration,
    write_stage1_ledger,
    write_surrogate_counts_from_parquet,
    write_surrogate_counts_from_rows,
)


def test_surrogate_count_export_excludes_did(tmp_path):
    raw = tmp_path / "raw.parquet"
    out = tmp_path / "counts.parquet"
    pq.write_table(pa.table({"did": ["did:a", "did:b"], "like_count": [2, 3]}), raw)
    write_surrogate_counts_from_parquet(raw, out, b"s" * 32)
    table = pq.read_table(out)
    assert table.column_names == [SURROGATE_COLUMN, COUNT_COLUMN]
    assert table[COUNT_COLUMN].to_pylist() == [2, 3]
    assert all(len(value) == 64 for value in table[SURROGATE_COLUMN].to_pylist())


def test_concentration_has_lorenz_and_top_table(tmp_path):
    population = tmp_path / "population.parquet"
    sampled = tmp_path / "sampled.parquet"
    out = tmp_path / "concentration.json"
    write_surrogate_counts_from_rows(["a", "b", "c", "d"], [1, 2, 3, 4], population, b"x" * 32)
    write_surrogate_counts_from_rows(["a", "b"], [1, 4], sampled, b"x" * 32)
    write_concentration(out, population, sampled)
    payload = json.loads(out.read_text())
    section = payload["population_before_min_like_filter_or_sampling"]
    assert section["gini"] == pytest.approx(0.25)
    assert len(section["lorenz_curve"]) == 1001
    assert section["lorenz_curve"][0]["cumulative_like_share"] == 0
    assert section["lorenz_curve"][-1]["cumulative_like_share"] == 1
    assert section["top_user_share_to_like_share"][0]["like_share"] == pytest.approx(0.4)


def test_stage1_ledger_writes_p1_through_p9(tmp_path):
    stats = {
        "likes": {
            "n_users_initial": 10, "n_likes_initial": 100,
            "n_users_eligible_for_sampling": 8, "n_likes_eligible": 98,
            "n_users_sampled": 5, "n_likes_after_user_sample": 50,
            "n_users_post_cap": 5, "n_likes_after_per_user_cap": 20,
            "n_users_final": 4,
        },
        "inferences": {"coverage_pct": 95.0},
    }
    out_json, out_md = tmp_path / "ledger.json", tmp_path / "ledger.md"
    write_stage1_ledger(out_json, out_md, stats, {"likes_core_rows": 18, "posts_core_rows": 17, "inferences_core_rows": 16})
    payload = json.loads(out_json.read_text())
    assert [row["point"] for row in payload["rows"]] == [f"P{i}" for i in range(1, 10)]
    assert "P9" in out_md.read_text()
