from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline import registry
from engagement_prediction.stages import query_selection as stage


UTC = timezone.utc


def _config(**overrides):
    values = {
        "unseen_user_fraction": 0.0,
        "max_hours_per_user_per_split": 64,
        "max_train_query_hours": None,
        "max_eval_query_hours_per_split": None,
        "max_positives_per_user_hour": 32,
        "random_seed": 42,
        "likes_start": datetime(2026, 1, 1, tzinfo=UTC),
        "likes_end": datetime(2026, 1, 10, tzinfo=UTC),
        "train_start": datetime(2026, 1, 1, tzinfo=UTC),
        "val_start": datetime(2026, 1, 4, tzinfo=UTC),
        "holdout_start": datetime(2026, 1, 7, tzinfo=UTC),
        "holdout_end": datetime(2026, 1, 10, tzinfo=UTC),
    }
    values.update(overrides)
    return stage.QuerySelectionConfig(**values)


def _collect(rows, config):
    positive_rows_lf = stage._prepare_likes(pl.DataFrame(rows).lazy(), config)
    lazyframes = stage._build_query_lazyframes_from_counts(
        positive_rows_lf=positive_rows_lf,
        candidate_query_counts_lf=stage._candidate_query_counts_lf(positive_rows_lf),
        config=config,
    )
    return stage.collect_query_artifacts(lazyframes, config)


def _cohorts(dids, fraction=0.1, seed=42):
    return stage._with_user_cohort(
        pl.DataFrame({"did": dids}).lazy(),
        unseen_user_fraction=fraction,
        random_seed=seed,
    ).collect()


def _seen_and_unseen_dids():
    cohorts = _cohorts([f"did:{idx}" for idx in range(100)], fraction=0.5)
    seen = cohorts.filter(pl.col("user_cohort") == "trainval")["did"].to_list()
    unseen = cohorts.filter(pl.col("user_cohort") == "unseen_eval")["did"].to_list()
    assert seen and unseen
    return seen, unseen


def test_stable_user_hash_assigns_disjoint_approximately_ten_percent_unseen():
    dids = [f"did:{idx}" for idx in range(10_000)]
    first = _cohorts(dids)
    second = _cohorts(dids)

    assert first.equals(second)
    assert first["did"].n_unique() == len(dids)
    unseen_count = first.filter(pl.col("user_cohort") == "unseen_eval").height
    assert 900 <= unseen_count <= 1_100
    assert set(first["user_cohort"].unique()) == {"trainval", "unseen_eval"}


def test_split_boundaries_include_all_five_splits_and_exclude_holdout_end():
    seen, unseen = _seen_and_unseen_dids()
    rows = []
    for did in (seen[0], unseen[0]):
        for label, timestamp in (
            ("train", "2026-01-01T00:00:00Z"),
            ("val", "2026-01-04T00:00:00Z"),
            ("holdout", "2026-01-07T00:00:00Z"),
            ("end", "2026-01-10T00:00:00Z"),
        ):
            rows.append({
                "did": did,
                "subject_uri": f"at://{did}/{label}",
                "record_created_at": timestamp,
            })

    queries, _, _ = _collect(rows, _config(unseen_user_fraction=0.5))
    by_user = {
        did: set(queries.filter(pl.col("did") == did)["split"].to_list())
        for did in (seen[0], unseen[0])
    }

    assert by_user[seen[0]] == {"train", "val", "holdout_seen_users"}
    assert by_user[unseen[0]] == {"val_unseen_users", "holdout_unseen_users"}
    assert set(queries["split"].to_list()) == set(stage.SPLITS)


def test_user_with_one_query_is_eligible():
    queries, positives, _ = _collect(
        [{
            "did": "did:one",
            "subject_uri": "at://post/one",
            "record_created_at": "2026-01-02T03:15:00Z",
        }],
        _config(),
    )

    assert queries.height == 1
    assert positives.height == 1
    assert queries.row(0, named=True)["positive_count"] == 1


def test_duplicate_likes_are_deduplicated_only_within_a_user_hour():
    queries, positives, stats = _collect(
        [
            {
                "did": "did:one",
                "subject_uri": "at://post/one",
                "record_created_at": "2026-01-02T03:45:00Z",
            },
            {
                "did": "did:one",
                "subject_uri": "at://post/one",
                "record_created_at": "2026-01-02T03:30:00Z",
            },
            {
                "did": "did:one",
                "subject_uri": "at://post/two",
                "record_created_at": "2026-01-02T03:50:00Z",
            },
        ],
        _config(),
    )

    assert queries.height == 1
    assert positives.height == 2
    assert queries["positive_count"].to_list() == [2]
    assert queries["query_hour"][0] == datetime(2026, 1, 2, 3, tzinfo=UTC)
    assert positives["like_created_at"][0] == datetime(2026, 1, 2, 3, 30, tzinfo=UTC)
    assert stats["positive_count_distribution"]["candidate"]["train"] == {"3": 1}
    assert stats["positive_count_distribution"]["final"]["train"] == {"2": 1}


def test_same_user_post_pair_is_retained_in_different_query_hours():
    queries, positives, _ = _collect(
        [
            {
                "did": "did:one",
                "subject_uri": "at://post/one",
                "record_created_at": "2026-01-02T03:30:00Z",
            },
            {
                "did": "did:one",
                "subject_uri": "at://post/one",
                "record_created_at": "2026-01-02T04:30:00Z",
            },
        ],
        _config(),
    )

    assert queries.height == 2
    assert positives.height == 2
    assert queries["positive_count"].to_list() == [1, 1]


def test_per_user_cap_is_applied_independently_in_each_split():
    rows = []
    for offset in range(65):
        rows.append({
            "did": "did:active",
            "subject_uri": f"at://train/{offset}",
            "record_created_at": (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=offset)).isoformat(),
        })
        rows.append({
            "did": "did:active",
            "subject_uri": f"at://val/{offset}",
            "record_created_at": (datetime(2026, 1, 5, tzinfo=UTC) + timedelta(hours=offset)).isoformat(),
        })
    config = _config(
        val_start=datetime(2026, 1, 5, tzinfo=UTC),
        holdout_start=None,
        holdout_end=None,
    )

    queries, _, stats = _collect(rows, config)

    assert queries.filter(pl.col("split") == "train").height == 64
    assert queries.filter(pl.col("split") == "val").height == 64
    assert stats["queries_by_phase_and_split"]["candidate"]["train"]["query_count"] == 65
    assert stats["queries_by_phase_and_split"]["candidate"]["val"]["query_count"] == 65


def test_split_query_budgets_are_independent():
    seen, unseen = _seen_and_unseen_dids()
    rows = []
    for did in seen[:10] + unseen[:10]:
        for label, timestamp in (
            ("train", "2026-01-02T00:00:00Z"),
            ("val", "2026-01-05T00:00:00Z"),
            ("holdout", "2026-01-08T00:00:00Z"),
        ):
            rows.append({
                "did": did,
                "subject_uri": f"at://{did}/{label}",
                "record_created_at": timestamp,
            })
    config = _config(
        unseen_user_fraction=0.5,
        max_train_query_hours=2,
        max_eval_query_hours_per_split=1,
    )

    queries, _, _ = _collect(rows, config)
    counts = dict(queries.group_by("split").len().iter_rows())

    assert counts == {
        "train": 2,
        "val": 1,
        "val_unseen_users": 1,
        "holdout_seen_users": 1,
        "holdout_unseen_users": 1,
    }


def test_query_priority_does_not_depend_on_positive_count():
    base_rows = [
        {"did": "did:one", "subject_uri": "at://a/0", "record_created_at": "2026-01-02T00:10:00Z"},
        {"did": "did:one", "subject_uri": "at://b/0", "record_created_at": "2026-01-02T01:10:00Z"},
    ]
    first_rows = base_rows + [
        {"did": "did:one", "subject_uri": f"at://a/{idx}", "record_created_at": "2026-01-02T00:20:00Z"}
        for idx in range(1, 6)
    ]
    second_rows = base_rows + [
        {"did": "did:one", "subject_uri": f"at://b/{idx}", "record_created_at": "2026-01-02T01:20:00Z"}
        for idx in range(1, 6)
    ]
    config = _config(max_train_query_hours=1)

    first_queries, _, _ = _collect(first_rows, config)
    second_queries, _, _ = _collect(second_rows, config)

    assert first_queries.select(stage.QUERY_KEY).equals(second_queries.select(stage.QUERY_KEY))


def test_oversized_query_is_dropped_after_sampling_without_backfill():
    candidate_keys = pl.DataFrame({
        "did": ["did:one"] * 8,
        "split": ["train"] * 8,
        "query_hour": [datetime(2026, 1, 2, hour, tzinfo=UTC) for hour in range(8)],
    }).with_columns(stage._query_priority_expr(42).alias("priority"))
    selected_hours = candidate_keys.sort(["priority", "did", "query_hour"])["query_hour"].to_list()[:2]
    oversized_hour = selected_hours[0]

    rows = []
    for hour in range(8):
        timestamp = datetime(2026, 1, 2, hour, 5, tzinfo=UTC)
        positive_count = 33 if timestamp.replace(minute=0) == oversized_hour else 1
        for idx in range(positive_count):
            rows.append({
                "did": "did:one",
                "subject_uri": f"at://{hour}/{idx}",
                "record_created_at": timestamp.isoformat(),
            })
    config = _config(max_train_query_hours=2)

    queries, _, stats = _collect(rows, config)

    assert queries.height == 1
    assert stats["queries_by_phase_and_split"]["after_split_cap"]["train"]["query_count"] == 2
    assert stats["queries_by_phase_and_split"]["final"]["train"]["query_count"] == 1
    assert stats["oversized_query_count_by_split"]["train"] == 1


def test_thirty_two_positives_are_retained_and_thirty_three_are_discarded():
    rows = []
    for hour, count in ((0, 32), (1, 33)):
        for idx in range(count):
            rows.append({
                "did": "did:one",
                "subject_uri": f"at://{hour}/{idx}",
                "record_created_at": f"2026-01-02T0{hour}:05:00Z",
            })

    queries, positives, _ = _collect(rows, _config())

    assert queries.height == 1
    assert queries["positive_count"].to_list() == [32]
    assert positives.height == 32


def test_raw_row_count_discards_query_that_deduplication_would_make_valid():
    rows = [
        {
            "did": "did:one",
            "subject_uri": "at://post/duplicate",
            "record_created_at": f"2026-01-02T00:{minute:02d}:00Z",
        }
        for minute in range(33)
    ]
    rows.append({
        "did": "did:one",
        "subject_uri": "at://post/retained",
        "record_created_at": "2026-01-02T01:05:00Z",
    })

    queries, positives, stats = _collect(rows, _config())

    assert queries["query_hour"].to_list() == [datetime(2026, 1, 2, 1, tzinfo=UTC)]
    assert positives["subject_uri"].to_list() == ["at://post/retained"]
    assert stats["oversized_query_count_by_split"]["train"] == 1


def test_build_config_validates_hour_alignment():
    args = SimpleNamespace(
        likes_start="2026-01-01T00:00:00Z",
        likes_end="2026-01-10T00:00:00Z",
        train_start="2026-01-01T00:30:00Z",
        val_start="2026-01-04T00:00:00Z",
        holdout_start="2026-01-07T00:00:00Z",
        holdout_end="2026-01-10T00:00:00Z",
        unseen_user_fraction=0.1,
        max_hours_per_user_per_split=64,
        max_train_query_hours=None,
        max_eval_query_hours_per_split=None,
        max_positives_per_user_hour=32,
        random_seed=42,
    )

    with pytest.raises(ValueError, match="train_start must be aligned"):
        stage.build_config(args)


@pytest.mark.parametrize(
    ("field_name", "value", "error_match"),
    [
        ("unseen_user_fraction", 1.0, "unseen_user_fraction"),
        ("max_hours_per_user_per_split", 0, "max_hours_per_user_per_split"),
        ("max_train_query_hours", -1, "max_train_query_hours"),
        ("max_eval_query_hours_per_split", -1, "max_eval_query_hours_per_split"),
        ("max_positives_per_user_hour", 0, "max_positives_per_user_hour"),
    ],
)
def test_build_config_validates_sampling_parameters(field_name, value, error_match):
    args = SimpleNamespace(
        likes_start="2026-01-01T00:00:00Z",
        likes_end="2026-01-10T00:00:00Z",
        train_start="2026-01-01T00:00:00Z",
        val_start="2026-01-04T00:00:00Z",
        holdout_start="2026-01-07T00:00:00Z",
        holdout_end="2026-01-10T00:00:00Z",
        unseen_user_fraction=0.1,
        max_hours_per_user_per_split=64,
        max_train_query_hours=None,
        max_eval_query_hours_per_split=None,
        max_positives_per_user_hour=32,
        random_seed=42,
    )
    setattr(args, field_name, value)

    with pytest.raises(ValueError, match=error_match):
        stage.build_config(args)


def test_registry_run_writes_query_artifacts_and_manifest(tmp_path, monkeypatch):
    likes_path = tmp_path / "likes.parquet"
    pl.DataFrame({
        "did": ["did:one", "did:one"],
        "subject_uri": ["at://post/one", "at://post/two"],
        "record_created_at": ["2026-01-02T01:05:00Z", "2026-01-02T01:10:00Z"],
    }).write_parquet(likes_path)
    monkeypatch.setattr(
        stage.ingex,
        "list_ingex_parquet_files",
        lambda **kwargs: ([str(likes_path)], [datetime(2026, 1, 2, 1, tzinfo=UTC)]),
    )
    args = SimpleNamespace(
        gcs_bucket="unused",
        likes_start="2026-01-01T00:00:00Z",
        likes_end="2026-01-10T00:00:00Z",
        train_start="2026-01-01T00:00:00Z",
        val_start="2026-01-04T00:00:00Z",
        holdout_start="2026-01-07T00:00:00Z",
        holdout_end="2026-01-10T00:00:00Z",
        unseen_user_fraction=0.0,
        max_hours_per_user_per_split=64,
        max_train_query_hours=None,
        max_eval_query_hours_per_split=None,
        max_positives_per_user_hour=32,
        random_seed=42,
        _argv=["--stop-after", "query_selection"],
    )
    context = Context(
        run_dir=tmp_path / "runs" / "run",
        artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "runs",
        pipeline_run_id="run",
    )

    result = registry.run_stage("query_selection", context, args)
    output_dir = Path(result["output_dir"])
    queries = pl.read_parquet(result["artifacts"]["queries_path"])
    positives = pl.read_parquet(result["artifacts"]["query_positives_path"])
    candidate_query_counts_paths = list(output_dir.glob("_candidate_query_counts_*.parquet"))

    assert output_dir.parent.name == "01_query_selection"
    assert len(candidate_query_counts_paths) == 1
    candidate_query_counts = pl.read_parquet(candidate_query_counts_paths[0])
    assert candidate_query_counts.columns == [
        "did",
        "query_hour",
        "user_cohort",
        "split",
        "raw_positive_count",
    ]
    assert candidate_query_counts["raw_positive_count"].to_list() == [2]
    assert queries.columns == ["did", "query_hour", "user_cohort", "split", "positive_count"]
    assert positives.columns == ["did", "query_hour", "subject_uri", "like_created_at"]
    assert queries.height == 1
    assert positives.height == 2
    assert json.loads((output_dir / "manifest.json").read_text())["stage_key"] == "query_selection"
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "stage.log").exists()
