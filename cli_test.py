from pathlib import Path
import base64
import json
import struct
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
import zlib

import numpy as np
import polars as pl
import pytest

import cli


def _encoded_test_embedding(value: float) -> list[dict[str, str]]:
    payload = struct.pack("<384f", *([value] * 384))
    encoded = base64.b85encode(zlib.compress(payload)).decode()
    return [{"key": "all_MiniLM_L12_v2", "value": encoded}]


@pytest.mark.parametrize(
    "argv",
    [
        # New behavior: `run-all` is optional.
        ["--config", "{config}", "--epochs", "7", "--batch-size", "512"],
        # Backwards compatible: still accepts `run-all`.
        ["--config", "{config}", "run-all", "--epochs", "7", "--batch-size", "512"],
    ],
)
def test_merge_args_with_config_prioritizes_cli_over_config(tmp_path, argv):
    config_path = Path(tmp_path) / "config.yml"
    config_path.write_text(
        textwrap.dedent(
            """
            epochs: 5
            embedding_model: all_MiniLM_L12_v2
            """
        ).strip()
    )

    parser = cli.build_parser()
    args = parser.parse_args([a.format(config=str(config_path)) for a in argv])

    merged = cli._merge_args_with_config(args)

    assert merged.epochs == 7  # CLI overrides config
    assert merged.embedding_model == "all_MiniLM_L12_v2"  # Config overrides defaults
    assert merged.batch_size == 512  # CLI overrides default
    assert merged.learning_rate == cli.DEFAULTS["learning_rate"]


def test_history_length_bucket_defaults_are_copied():
    args = cli._merge_args_with_config(cli.build_parser().parse_args([]))

    assert args.history_length_bucket_boundaries == [0, 1, 2, 4, 8, 16, 32]
    assert args.history_length_bucket_boundaries is not cli.DEFAULTS["history_length_bucket_boundaries"]


def test_health_checks_config_and_cli_overrides(tmp_path):
    config_path = tmp_path / "checks.yml"
    config_path.write_text("data_checks: true\ndisk_checks: true\nraw_data_fail_min_ratio: 0.2\nmin_free_output_gib: 200\n")
    parser = cli.build_parser()
    args = cli._merge_args_with_config(parser.parse_args([
        "--config", str(config_path), "--no-disk-checks", "--raw-data-fail-min-ratio", "0.1",
    ]))

    assert args.data_checks is True
    assert args.disk_checks is False
    assert args.raw_data_fail_min_ratio == 0.1
    assert args.min_free_output_gib == 200
    assert args.raw_data_warn_min_ratio == cli.DEFAULTS["raw_data_warn_min_ratio"]


def test_automation_config_enables_checks_and_keeps_full_foreground_run():
    args = cli._merge_args_with_config(cli.build_parser().parse_args([
        "--config", str(Path(cli.__file__).parent / "config.automation.yml"),
    ]))

    assert args.data_checks and args.disk_checks
    assert args.compare_to_production
    assert args.production_cloud_run_service == "engagement-prediction-inference-prod"
    assert args.background is False
    assert args.start_from is None
    assert args.stop_after is None
    assert args.output_dir == "/mnt/data/dave/outputs/automated-ranker"


@pytest.mark.parametrize("overrides", [
    {"data_checks": "false"},
    {"raw_data_min_complete_days": 0},
    {"raw_data_min_complete_days": 1.5},
    {"raw_data_fail_min_ratio": 0},
    {"raw_data_fail_min_ratio": 0.8},
    {"raw_data_warn_min_ratio": 1.1},
    {"raw_data_warn_max_ratio": 0.9},
    {"raw_data_warn_invalid_timestamp_fraction": -0.1},
    {"raw_data_warn_invalid_timestamp_fraction": 1.1},
    {"min_free_output_gib": float("nan")},
    {"min_free_output_gib": float("inf")},
    {"min_free_output_gib": -1},
])
def test_invalid_health_check_config_is_rejected(tmp_path, overrides):
    config_path = tmp_path / "checks.json"
    config_path.write_text(json.dumps(overrides))
    with pytest.raises(ValueError):
        cli._merge_args_with_config(cli.build_parser().parse_args(["--config", str(config_path)]))


@pytest.mark.parametrize("command", [[], ["run-all"]])
def test_history_length_bucket_cli_overrides_config(tmp_path, command):
    config_path = tmp_path / "history_lengths.json"
    config_path.write_text(json.dumps({"history_length_bucket_boundaries": [0, 3, 7]}))
    parser = cli.build_parser()

    configured = cli._merge_args_with_config(parser.parse_args([
        "--config", str(config_path), *command,
    ]))
    overridden = cli._merge_args_with_config(parser.parse_args([
        "--config", str(config_path), *command,
        "--history-length-bucket-boundaries", "0", "5", "10",
    ]))

    assert configured.history_length_bucket_boundaries == [0, 3, 7]
    assert overridden.history_length_bucket_boundaries == [0, 5, 10]


@pytest.mark.parametrize(
    "boundaries",
    [[], None, [1, 2], [-1, 0], [0, -1], [0, 2, 1], [0, 1, 1], [False, 1], [0, True], [0, 1.0], "0,1"],
)
def test_history_length_bucket_invalid_config_is_rejected(tmp_path, boundaries):
    config_path = tmp_path / "invalid_history_lengths.json"
    config_path.write_text(json.dumps({"history_length_bucket_boundaries": boundaries}))
    raw = cli.build_parser().parse_args(["--config", str(config_path)])

    with pytest.raises(ValueError, match="history_length_bucket_boundaries"):
        cli._merge_args_with_config(raw)


@pytest.mark.parametrize("boundaries", [["0", "2", "1"], ["0", "0"], ["1"]])
def test_history_length_bucket_invalid_cli_is_rejected(boundaries):
    raw = cli.build_parser().parse_args(["--history-length-bucket-boundaries", *boundaries])

    with pytest.raises(ValueError, match="history_length_bucket_boundaries"):
        cli._merge_args_with_config(raw)


@pytest.mark.parametrize("loss_type", ["listwise", "bce"])
def test_cli_argument_merge_does_not_import_torch(loss_type):
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import cli; cli._merge_args_with_config(cli.build_parser().parse_args(sys.argv[1:])); assert 'torch' not in sys.modules", "--loss-type", loss_type],
        cwd=Path(cli.__file__).parent,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["--config", "{config}"],
        ["--config", "{config}", "run-all"],
    ],
)
def test_merge_args_with_config_rejects_unknown_keys(tmp_path, argv):
    config_path = Path(tmp_path) / "invalid.yml"
    config_path.write_text("unknown_flag: true\n")

    parser = cli.build_parser()
    args = parser.parse_args([a.format(config=str(config_path)) for a in argv])

    with pytest.raises(ValueError):
        cli._merge_args_with_config(args)


@pytest.mark.parametrize(
    "flag",
    [
        "--negative-samples-per-hour",
        "--political-negative-samples-per-hour",
        "--max-political-candidates-per-creation-hour",
        "--political-score-threshold",
        "--political-inference-window-padding-days",
        "--negative-sampling-alpha",
        "--min-likes-per-negative-post",
        "--initial-negative-sampling-pct",
        "--cap-random-seed",
        "--likes-start",
        "--likes-end",
        "--bst-political-batch-negatives",
        "--post-selection-partition-count",
    ],
)
def test_obsolete_negative_sampling_flags_are_rejected(flag):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([flag, "1"])
    assert flag.removeprefix("--").replace("-", "_") not in cli.DEFAULTS


def test_source_metadata_args_merge_from_cli_and_config(tmp_path):
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--random-candidate-sampling-fraction", "0.2",
        "--source-metadata-partition-count", "64",
    ])
    merged = cli._merge_args_with_config(raw)

    assert merged.random_candidate_sampling_fraction == 0.2
    assert merged.source_metadata_partition_count == 64

    config_path = Path(tmp_path) / "post_selection.yml"
    config_path.write_text(
        "random_candidate_sampling_fraction: 0.3\n"
        "source_metadata_partition_count: 32\n"
    )
    raw = parser.parse_args(["--config", str(config_path)])
    merged = cli._merge_args_with_config(raw)

    assert merged.random_candidate_sampling_fraction == 0.3
    assert merged.source_metadata_partition_count == 32


def test_negative_selection_args_merge_from_cli_and_defaults():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--negative-candidates-per-hour", "250",
        "--min-likes-for-popular-candidate", "20",
        "--popular-candidate-fraction", "0.25",
        "--max-candidate-age-hours", "12",
    ])
    merged = cli._merge_args_with_config(raw)

    assert merged.negative_candidates_per_hour == 250
    assert merged.min_likes_for_popular_candidate == 20
    assert merged.popular_candidate_fraction == 0.25
    assert merged.max_candidate_age_hours == 12


def test_post_liker_history_args_merge_from_cli():
    merged = cli._merge_args_with_config(
        cli.build_parser().parse_args([
            "--post-liker-history-partition-count",
            "64",
        ])
    )

    assert merged.post_liker_history_partition_count == 64


def test_author_statistics_args_merge_from_cli():
    merged = cli._merge_args_with_config(
        cli.build_parser().parse_args([
            "--author-statistics-partition-count",
            "64",
        ])
    )

    assert merged.author_statistics_partition_count == 64


def test_dataset_hydration_args_merge_from_cli():
    merged = cli._merge_args_with_config(
        cli.build_parser().parse_args([
            "--embedding-source-batch-size",
            "32",
            "--dataset-hydration-worker-count",
            "3",
            "--min-author-training-feature-count",
            "75",
        ])
    )

    assert merged.embedding_source_batch_size == 32
    assert merged.dataset_hydration_worker_count == 3
    assert merged.min_author_training_feature_count == 75

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([
            "--embedding-partition-worker-count",
            "3",
        ])


def test_data_partition_worker_count_merges_from_cli():
    merged = cli._merge_args_with_config(
        cli.build_parser().parse_args(["--data-partition-worker-count", "8"])
    )

    assert merged.data_partition_worker_count == 8


def test_query_sampling_args_replace_user_sampling_args(tmp_path):
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--unseen-user-fraction", "0.2",
        "--max-hours-per-user-per-split", "48",
        "--max-train-query-hours", "123",
        "--max-eval-query-hours-per-split", "45",
        "--max-positives-per-user-hour", "24",
    ])
    merged = cli._merge_args_with_config(raw)

    assert merged.unseen_user_fraction == 0.2
    assert merged.max_hours_per_user_per_split == 48
    assert merged.max_train_query_hours == 123
    assert merged.max_eval_query_hours_per_split == 45
    assert merged.max_positives_per_user_hour == 24
    assert cli.DEFAULTS["max_hours_per_user_per_split"] == 16
    assert cli.DEFAULTS["max_positives_per_user_hour"] == 32

    with pytest.raises(SystemExit):
        parser.parse_args(["--max-trainval-users", "123"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--max-unseen-eval-users", "45"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--max-likes-per-user", "16"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--min-likes-per-user", "2"])

    config_path = Path(tmp_path) / "old_sampling.yml"
    config_path.write_text("max_trainval_users: 123\nmax_unseen_eval_users: 45\n")
    raw = parser.parse_args(["--config", str(config_path)])
    with pytest.raises(ValueError):
        cli._merge_args_with_config(raw)


def test_query_sampling_defaults():
    merged = cli._merge_args_with_config(cli.build_parser().parse_args([]))

    assert merged.unseen_user_fraction == 0.10
    assert merged.max_hours_per_user_per_split == 16
    assert merged.max_train_query_hours == 10_000_000
    assert merged.max_eval_query_hours_per_split == 500_000
    assert merged.max_positives_per_user_hour == 32
    assert merged.max_history_posts_per_query == 32
    assert merged.user_history_partition_count == 32
    assert merged.random_candidate_sampling_fraction == 0.10
    assert merged.source_metadata_partition_count == 16
    assert merged.data_partition_worker_count == 4
    assert merged.negative_candidates_per_hour == 1000
    assert merged.min_likes_for_popular_candidate == 10
    assert merged.popular_candidate_fraction == 0.50
    assert merged.max_candidate_age_hours == 24
    assert merged.post_liker_history_partition_count == 16
    assert merged.author_statistics_partition_count == 16
    assert merged.embedding_source_batch_size == 64
    assert merged.dataset_hydration_worker_count == 8
    assert merged.min_author_training_feature_count == 50
    assert merged.min_post_liker_user_training_event_count == 10
    assert merged.max_post_liker_user_vocabulary_size == 10_000_000

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--max-prior-likes", "64"])
    for removed_flag in (
        "--min-author-post-count",
        "--min-author-received-like-count",
    ):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([removed_flag, "1"])


def test_bst_ranker_is_the_default_model():
    args = cli._merge_args_with_config(cli.build_parser().parse_args([]))

    assert args.model_type == "bst-ranker"
    assert cli._get_train_key(args.model_type) == "train_bst_ranker"
    assert cli._get_stage_order_for_model_type("train_bst_ranker")[-1] == "train_bst_ranker"


@pytest.mark.parametrize("model_type", ["bst-ranker", "two-tower"])
def test_loss_type_defaults_to_listwise(model_type):
    args = cli._merge_args_with_config(cli.build_parser().parse_args([
        "--model-type", model_type,
    ]))

    assert args.loss_type == "listwise"


@pytest.mark.parametrize("model_type", ["bst-ranker", "two-tower"])
@pytest.mark.parametrize("loss_type", ["listwise", "bce"])
def test_loss_type_cli_for_both_models(model_type, loss_type):
    args = cli._merge_args_with_config(cli.build_parser().parse_args([
        "--model-type", model_type, "--loss-type", loss_type,
    ]))

    assert args.loss_type == loss_type


@pytest.mark.parametrize("command", [[], ["run-all"]])
@pytest.mark.parametrize("extension", ["yml", "json"])
def test_loss_type_cli_overrides_config(tmp_path, command, extension):
    config_path = tmp_path / f"loss.{extension}"
    config_path.write_text(
        "loss_type: bce\n" if extension == "yml" else json.dumps({"loss_type": "bce"})
    )
    parser = cli.build_parser()
    arguments = ["--config", str(config_path), *command]

    configured = cli._merge_args_with_config(parser.parse_args(arguments))
    overridden = cli._merge_args_with_config(parser.parse_args([
        *arguments, "--loss-type", "listwise",
    ]))

    assert configured.loss_type == "bce"
    assert overridden.loss_type == "listwise"


def test_invalid_loss_type_cli_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--loss-type", "unknown"])


@pytest.mark.parametrize("loss_type", ["unknown", None, True, [], {}])
def test_invalid_loss_type_config_is_rejected(tmp_path, loss_type):
    config_path = tmp_path / "invalid_loss.json"
    config_path.write_text(json.dumps({"loss_type": loss_type}))
    raw = cli.build_parser().parse_args(["--config", str(config_path)])

    with pytest.raises(ValueError, match="loss_type"):
        cli._merge_args_with_config(raw)


@pytest.mark.parametrize(
    "stop_after",
    [
        "user_history",
        "post_selection",
        "negative_selection",
        "post_liker_history",
        "author_statistics",
        "dataset_hydration",
    ],
)
def test_new_pipeline_can_stop_after_data_stage(stop_after):
    args = cli._merge_args_with_config(
        cli.build_parser().parse_args(["--stop-after", stop_after])
    )
    train_key = cli._get_train_key(args.model_type)
    stage_order = cli._get_stage_order_for_model_type(train_key)
    _start_idx, stop_idx, includes_train = cli._get_stage_folder_and_start_stop_indices(
        stage_order,
        args.start_from,
        args.stop_after,
        train_key,
    )

    assert stage_order[stop_idx] == stop_after
    assert includes_train is False


def test_new_pipeline_runs_query_selection_through_user_history(tmp_path, monkeypatch):
    likes_path = Path(tmp_path) / "likes.parquet"
    pl.DataFrame({
        "did": ["u1", "u1", "u1"],
        "subject_uri": ["history", "target-one", "target-two"],
        "record_created_at": [
            "2026-01-01T09:00:00Z",
            "2026-01-01T10:05:00Z",
            "2026-01-01T12:05:00Z",
        ],
    }).write_parquet(likes_path)
    posts_path = Path(tmp_path) / "posts.parquet"
    pl.DataFrame({
        "at_uri": ["target-one", "target-two"],
        "record_created_at": [
            "2026-01-01T09:00:00Z",
            "2026-01-01T11:00:00Z",
        ],
        "did": ["author-one", "author-two"],
    }).write_parquet(posts_path)
    replies_path = Path(tmp_path) / "replies.parquet"
    pl.DataFrame({
        "at_uri": ["unused-reply"],
        "record_created_at": ["2026-01-01T08:30:00Z"],
        "did": ["reply-author"],
    }).write_parquet(replies_path)

    def list_sources(*, blob_prefix, **kwargs):
        if blob_prefix == "bsky_likes":
            return [str(likes_path)], [datetime(2026, 1, 1, 8, tzinfo=timezone.utc)]
        if blob_prefix == "bsky_posts":
            return [str(posts_path)], [datetime(2026, 1, 1, 8, tzinfo=timezone.utc)]
        assert blob_prefix == "bsky_replies"
        return [str(replies_path)], [datetime(2026, 1, 1, 8, tzinfo=timezone.utc)]

    monkeypatch.setattr(
        "engagement_prediction.data.ingex.list_ingex_parquet_files",
        list_sources,
    )
    args = cli._merge_args_with_config(cli.build_parser().parse_args([
        "--stop-after", "user_history",
        "--posts-end", "2026-01-02T00:00:00Z",
        "--unseen-user-fraction", "0",
        "--user-history-partition-count", "2",
        "--source-metadata-partition-count", "2",
    ]))
    # Exercise downstream hourly windows directly, without the CLI date resolver.
    args.posts_start = "2026-01-01T00:00:00Z"
    args.train_start = "2026-01-01T10:00:00Z"
    args.val_start = "2026-01-01T20:00:00Z"
    args.holdout_start = None
    args.holdout_end = None
    output_root = Path(tmp_path) / "output"
    output_root.mkdir()
    args.output_dir = str(output_root)
    args._argv = ["--stop-after", "user_history"]
    context = cli.Context(
        run_dir=output_root / "runs" / "run",
        artifacts_dir=output_root / "artifacts",
        runs_dir=output_root / "runs",
        pipeline_run_id="run",
    )

    assert cli.cmd__run_all_exec(args, context) == 0
    assert context.get_artifact_dir("query_selection") is not None
    history_dir = context.get_artifact_dir("user_history")
    assert history_dir is not None
    assert list(history_dir.glob("query_histories_*"))


def test_new_pipeline_runs_query_selection_through_dataset_hydration(tmp_path, monkeypatch):
    likes_path = Path(tmp_path) / "likes.parquet"
    pl.DataFrame({
        "did": ["u1", "u1", "u1"],
        "subject_uri": ["history", "target-one", "target-two"],
        "record_created_at": [
            "2026-01-01T09:00:00Z",
            "2026-01-01T10:05:00Z",
            "2026-01-01T12:05:00Z",
        ],
    }).write_parquet(likes_path)
    posts_path = Path(tmp_path) / "posts.parquet"
    pl.DataFrame({
        "at_uri": ["history", "target-one", "target-two"],
        "record_created_at": [
            "2026-01-01T08:00:00Z",
            "2026-01-01T09:00:00Z",
            "2026-01-01T11:00:00Z",
        ],
        "did": ["author-one", "author-two", "author-three"],
        "embeddings": [
            _encoded_test_embedding(1.0),
            _encoded_test_embedding(2.0),
            _encoded_test_embedding(3.0),
        ],
    }).write_parquet(posts_path)
    replies_path = Path(tmp_path) / "replies.parquet"
    pl.DataFrame({
        "at_uri": ["unused-reply"],
        "record_created_at": ["2026-01-01T08:30:00Z"],
        "did": ["reply-author"],
        "embeddings": [_encoded_test_embedding(4.0)],
    }).write_parquet(replies_path)

    def list_sources(*, blob_prefix, **kwargs):
        if blob_prefix == "bsky_likes":
            return [str(likes_path)], [datetime(2026, 1, 1, 8, tzinfo=timezone.utc)]
        if blob_prefix == "bsky_posts":
            return [str(posts_path)], [datetime(2026, 1, 1, 8, tzinfo=timezone.utc)]
        assert blob_prefix == "bsky_replies"
        return [str(replies_path)], [datetime(2026, 1, 1, 8, tzinfo=timezone.utc)]

    monkeypatch.setattr(
        "engagement_prediction.data.ingex.list_ingex_parquet_files",
        list_sources,
    )
    args = cli._merge_args_with_config(cli.build_parser().parse_args([
        "--stop-after", "dataset_hydration",
        "--posts-end", "2026-01-02T00:00:00Z",
        "--unseen-user-fraction", "0",
        "--user-history-partition-count", "2",
        "--source-metadata-partition-count", "2",
        "--random-candidate-sampling-fraction", "1",
        "--negative-candidates-per-hour", "2",
        "--min-likes-for-popular-candidate", "0",
        "--popular-candidate-fraction", "0",
        "--post-liker-history-partition-count", "2",
        "--author-statistics-partition-count", "2",
        "--min-author-training-feature-count", "1",
    ]))
    # Exercise downstream hourly windows directly, without the CLI date resolver.
    args.posts_start = "2026-01-01T00:00:00Z"
    args.train_start = "2026-01-01T10:00:00Z"
    args.val_start = "2026-01-01T20:00:00Z"
    args.holdout_start = None
    args.holdout_end = None
    output_root = Path(tmp_path) / "output"
    output_root.mkdir()
    args.output_dir = str(output_root)
    args._argv = ["--stop-after", "dataset_hydration"]
    context = cli.Context(
        run_dir=output_root / "runs" / "run",
        artifacts_dir=output_root / "artifacts",
        runs_dir=output_root / "runs",
        pipeline_run_id="run",
    )

    assert cli.cmd__run_all_exec(args, context) == 0
    post_selection_dir = context.get_artifact_dir("post_selection")
    assert post_selection_dir is not None
    posts = pl.scan_parquet(
        post_selection_dir / "post_universe_*" / "posts" / "*.parquet"
    ).collect()
    assert set(posts["subject_uri"]) == {"history", "target-one", "target-two"}
    negative_selection_dir = context.get_artifact_dir("negative_selection")
    assert negative_selection_dir is not None
    hourly_candidates = pl.scan_parquet(
        negative_selection_dir
        / "negative_candidates_*"
        / "hourly_candidates"
        / "*.parquet"
    ).collect()
    assert hourly_candidates.group_by("query_hour").len()["len"].to_list() == [2, 2]
    post_liker_history_dir = context.get_artifact_dir("post_liker_history")
    assert post_liker_history_dir is not None
    post_liker_posts = pl.scan_parquet(
        post_liker_history_dir
        / "post_liker_histories_*"
        / "post_liker_posts"
        / "*.parquet"
    ).collect()
    assert set(post_liker_posts["subject_uri"]) == {
        "history",
        "target-one",
        "target-two",
    }
    author_statistics_dir = context.get_artifact_dir("author_statistics")
    assert author_statistics_dir is not None
    author_stats = pl.scan_parquet(
        author_statistics_dir
        / "author_statistics_*"
        / "author_statistics"
        / "*.parquet"
    ).collect()
    assert set(author_stats["author_did"]) == {
        "author-one",
        "author-two",
        "author-three",
        "reply-author",
    }
    assert "author_idx" not in author_stats.columns
    dataset_hydration_dir = context.get_artifact_dir("dataset_hydration")
    assert dataset_hydration_dir is not None
    hydrated_bundle = next(dataset_hydration_dir.glob("hydrated_training_data_*"))
    embeddings = np.load(hydrated_bundle / "embeddings.npy", mmap_mode="r")
    assert embeddings.shape == (3, 384)
    authors = pl.scan_parquet(hydrated_bundle / "authors" / "*.parquet").collect()
    assert authors.select("author_did", "author_idx").to_dicts() == [
        {"author_did": "author-one", "author_idx": 2},
        {"author_did": "author-three", "author_idx": 3},
        {"author_did": "author-two", "author_idx": 4},
    ]


@pytest.mark.parametrize(
    ("model_type", "train_stage"),
    [
        ("bst-ranker", "train_bst_ranker"),
        ("two-tower", "train_two_tower"),
    ],
)
def test_canonical_model_can_stop_after_its_training_stage(model_type, train_stage):
    args = cli._merge_args_with_config(cli.build_parser().parse_args([
        "--model-type", model_type,
        "--stop-after", train_stage,
    ]))
    train_key = cli._get_train_key(args.model_type)
    stage_order = cli._get_stage_order_for_model_type(train_key)
    _start_idx, stop_idx, includes_train = cli._get_stage_folder_and_start_stop_indices(
        stage_order,
        args.start_from,
        args.stop_after,
        train_key,
    )

    assert train_key == train_stage
    assert stage_order[stop_idx] == train_stage
    assert includes_train is True


def test_native_bst_sequential_run_executes_stages_zero_through_eight(
    tmp_path,
    monkeypatch,
):
    args = cli._merge_args_with_config(cli.build_parser().parse_args([
        "--model-type", "bst-ranker",
        "--stop-after", "train_bst_ranker",
    ]))
    args.output_dir = str(tmp_path)
    context = cli.Context(
        run_dir=Path(tmp_path) / "runs" / "run",
        artifacts_dir=Path(tmp_path) / "artifacts",
        runs_dir=Path(tmp_path) / "runs",
        pipeline_run_id="run",
    )
    executed = []
    monkeypatch.setattr(cli, "pin_lineage_aligned_inputs", lambda *args: None)
    monkeypatch.setattr(
        cli.reg,
        "run_stage",
        lambda stage_key, context, args: executed.append(stage_key),
    )

    assert cli.cmd__run_all_exec(args, context) == 0
    assert executed == [
        "source_metadata",
        "query_selection",
        "user_history",
        "post_selection",
        "negative_selection",
        "post_liker_history",
        "author_statistics",
        "dataset_hydration",
        "train_bst_ranker",
    ]


def test_native_two_tower_sequential_run_executes_stages_zero_through_eight(
    tmp_path,
    monkeypatch,
):
    args = cli._merge_args_with_config(cli.build_parser().parse_args([
        "--model-type", "two-tower",
        "--stop-after", "train_two_tower",
    ]))
    args.output_dir = str(tmp_path)
    context = cli.Context(
        run_dir=Path(tmp_path) / "runs" / "run",
        artifacts_dir=Path(tmp_path) / "artifacts",
        runs_dir=Path(tmp_path) / "runs",
        pipeline_run_id="run",
    )
    executed = []
    monkeypatch.setattr(cli, "pin_lineage_aligned_inputs", lambda *args: None)
    monkeypatch.setattr(
        cli.reg,
        "run_stage",
        lambda stage_key, context, args: executed.append(stage_key),
    )

    assert cli.cmd__run_all_exec(args, context) == 0
    assert executed == [
        "source_metadata",
        "query_selection",
        "user_history",
        "post_selection",
        "negative_selection",
        "post_liker_history",
        "author_statistics",
        "dataset_hydration",
        "train_two_tower",
    ]


@pytest.mark.parametrize("fail_before_stage", [None, "source_metadata", "dataset_hydration"])
def test_disk_checks_stop_pipeline_at_boundaries_and_preserve_report(tmp_path, monkeypatch, fail_before_stage):
    from engagement_prediction.experiment_tracking import NoOpExperimentTracker

    class Tracker(NoOpExperimentTracker):
        closed = False
        failed = False

        def mark_failed(self, reason):
            assert not self.closed
            self.failed = True

        def close(self):
            self.closed = True

    args = cli._merge_args_with_config(cli.build_parser().parse_args(["--disk-checks"]))
    args.output_dir = str(tmp_path)
    tracker = Tracker()
    context = cli.Context(
        run_dir=tmp_path / "runs" / "run",
        artifacts_dir=tmp_path / "artifacts",
        tracker=tracker,
    )
    stage_order = cli._get_stage_order_for_model_type("train_bst_ranker")
    executed, checked_stages = [], []

    def check_space(**kwargs):
        stage = stage_order[len(executed)]
        checked_stages.append(stage)
        assert kwargs == {"output_dir": tmp_path, "min_output_free_gib": 150}
        status = "fail" if stage == fail_before_stage else "pass"
        return {"status": status, "checks": [{"status": status, "message": f"disk before {stage}"}]}

    monkeypatch.setattr(cli.disk_checks, "check_disk_space", check_space)
    monkeypatch.setattr(cli, "pin_lineage_aligned_inputs", lambda *args: None)
    monkeypatch.setattr(cli.reg, "run_stage", lambda stage, *args: executed.append(stage))

    if fail_before_stage is None:
        assert cli.cmd__run_all_exec(args, context) == 0
        assert executed == stage_order
        assert checked_stages == ["source_metadata", "dataset_hydration"]
    else:
        with pytest.raises(ValueError, match=f"disk before {fail_before_stage}"):
            cli.cmd__run_all_exec(args, context)
        assert executed == stage_order[:stage_order.index(fail_before_stage)]
        report = json.loads((context.run_dir / f"disk_checks_{fail_before_stage}.json").read_text())
        assert report["status"] == "fail"
        assert report["stage"] == fail_before_stage
    assert tracker.closed
    assert tracker.failed == (fail_before_stage is not None)


@pytest.mark.parametrize(
    ("model_type", "start_from", "train_stage"),
    [
        ("bst-ranker", "train", "train_bst_ranker"),
        ("bst-ranker", "train_bst_ranker", "train_bst_ranker"),
        ("two-tower", "train", "train_two_tower"),
        ("two-tower", "train_two_tower", "train_two_tower"),
    ],
)
def test_disk_checks_skip_runs_starting_at_train(
    tmp_path, monkeypatch, model_type, start_from, train_stage
):
    args = cli._merge_args_with_config(cli.build_parser().parse_args([
        "--disk-checks", "--model-type", model_type, "--start-from", start_from,
    ]))
    args.output_dir = str(tmp_path)
    context = cli.Context(
        run_dir=tmp_path / "runs" / "run",
        artifacts_dir=tmp_path / "artifacts",
    )
    executed = []
    monkeypatch.setattr(
        cli.disk_checks,
        "check_disk_space",
        lambda **kwargs: pytest.fail("Training-only runs must not check disk space"),
    )
    monkeypatch.setattr(cli, "pin_lineage_aligned_inputs", lambda *args: None)
    monkeypatch.setattr(cli.reg, "run_stage", lambda stage, *args: executed.append(stage))

    assert cli.cmd__run_all_exec(args, context) == 0
    assert executed == [train_stage]


def test_runtime_revalidates_check_thresholds_after_remote_overrides(tmp_path, monkeypatch):
    args = cli._merge_args_with_config(cli.build_parser().parse_args([]))
    args.output_dir = str(tmp_path)
    args.raw_data_fail_min_ratio = float("nan")
    context = cli.Context(run_dir=tmp_path / "run", artifacts_dir=tmp_path / "artifacts")
    monkeypatch.setattr(cli.reg, "run_stage", lambda *args: pytest.fail("Stage must not run"))

    with pytest.raises(ValueError, match="raw_data_fail_min_ratio"):
        cli.cmd__run_all_exec(args, context)


def test_two_tower_uses_fixed_encoder_and_output_dimension_default():
    parser = cli.build_parser()
    raw = parser.parse_args(["--model-type", "two-tower"])
    merged = cli._merge_args_with_config(raw)

    assert merged.two_tower_output_embedding_dim == 128
    cli._validate_two_tower_config(merged)


@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("--two-tower-output-embedding-dim", "two-tower-output-embedding-dim"),
        ("--two-tower-user-hidden-dim", "two-tower-user-hidden-dim"),
        ("--two-tower-post-hidden-dim", "two-tower-post-hidden-dim"),
        ("--max-history-len", "max-history-len"),
        ("--batch-size", "batch-size"),
        ("--eval-batch-size", "eval-batch-size"),
    ],
)
def test_two_tower_rejects_non_positive_dimensions(flag, message):
    parser = cli.build_parser()
    merged = cli._merge_args_with_config(parser.parse_args([
        "--model-type", "two-tower",
        flag, "0",
    ]))

    with pytest.raises(ValueError, match=message):
        cli._validate_two_tower_config(merged)


def test_canonical_cli_removes_shared_dim():
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--shared-dim", "64"])
    assert "shared_dim" not in cli.DEFAULTS


@pytest.mark.parametrize(
    "argv",
    [
        ["--model-type", "mlp"],
        ["--start-from", "train_mlp"],
        ["--stop-after", "evaluate"],
        ["--user-summarization", "mean"],
        ["--ema-alpha", "0.1"],
        ["--user-encoder", "cross_attention"],
        ["--user-output-dim", "128"],
        ["--post-encoder"],
        ["--num-attention-heads", "4"],
        ["--num-attention-layers", "2"],
        ["--attention-dropout", "0.1"],
        ["--l2-normalize-embeddings"],
        ["--use-author-embedding-table"],
        ["--min-author-support", "20"],
        ["--weight-decay-mlp", "0.1"],
        ["--hidden-dims", "64"],
        ["--dropout-rate-mlp", "0.5"],
        ["--prediction-posts-per-user", "1"],
        ["--no-save-model"],
        ["--eval-holdout-type", "unseen_users"],
        ["--skip-modules", "cold_start_curves"],
        ["--prior-03-train", "legacy"],
        ["--max-memory-gb", "10"],
        ["--max-memory-pct", "0.5"],
        ["--memory-check", "full"],
        ["--global-topic-k", "10"],
        ["--debug"],
    ],
)
def test_removed_legacy_cli_surfaces_are_rejected(argv):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(argv)


def test_removed_mlp_config_is_rejected(tmp_path):
    config_path = Path(tmp_path) / "mlp.yml"
    config_path.write_text("model_type: mlp\n")
    raw = cli.build_parser().parse_args(["--config", str(config_path)])

    with pytest.raises(ValueError, match="Unknown model_type.*mlp"):
        cli._merge_args_with_config(raw)


def test_bst_ranker_model_type_maps_train_alias():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--start-from", "train",
        "--stop-after", "train",
        "--bst-prediction-hidden-dims", "144", "72",
    ])
    merged = cli._merge_args_with_config(raw)

    train_key = cli._get_train_key(merged.model_type)
    stage_order = cli._get_stage_order_for_model_type(train_key)
    start_idx, stop_idx, includes_train = cli._get_stage_folder_and_start_stop_indices(
        stage_order,
        merged.start_from,
        merged.stop_after,
        train_key,
    )

    assert train_key == "train_bst_ranker"
    assert stage_order[start_idx] == "train_bst_ranker"
    assert stage_order[stop_idx] == "train_bst_ranker"
    assert includes_train is True
    assert merged.bst_num_transformer_layers == 1


def test_bst_ranker_explicit_train_stage_names_parse():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--start-from", "train_bst_ranker",
        "--stop-after", "train_bst_ranker",
        "--bst-prediction-hidden-dims", "144", "72",
    ])
    merged = cli._merge_args_with_config(raw)

    assert merged.start_from == "train_bst_ranker"
    assert merged.stop_after == "train_bst_ranker"


def test_merge_args_with_config_accepts_bst_ranker_keys(tmp_path):
    config_path = Path(tmp_path) / "bst.yml"
    config_path.write_text(
        textwrap.dedent(
            """
            model_type: bst-ranker
            bst_model_dim: 96
            content_projection_dim: 80
            author_projection_dim: 24
            bst_time_embedding_dim: 32
            bst_num_attention_heads: 8
            bst_num_transformer_layers: 1
            bst_transformer_ff_dim: 384
            bst_dropout_rate: 0.2
            bst_norm_first: true
            bst_time_delta_bucket_boundaries_hours: [1, 2, 4]
            bst_prediction_hidden_dims: [128, 64]
            bst_weight_decay: 0.02
            bst_additional_batch_negatives: 32
            batch_size: 16
            bst_max_train_batches_per_epoch: 5
            bst_use_popularity_feature: false
            bst_popularity_projection_dim: 12
            """
        ).strip()
        + "\n"
    )

    parser = cli.build_parser()
    raw = parser.parse_args(["--config", str(config_path)])
    merged = cli._merge_args_with_config(raw)

    assert merged.model_type == "bst-ranker"
    assert merged.bst_model_dim == 96
    assert merged.content_projection_dim == 80
    assert merged.author_projection_dim == 24
    assert merged.bst_time_embedding_dim == 32
    assert merged.bst_num_attention_heads == 8
    assert merged.bst_prediction_hidden_dims == [128, 64]
    assert merged.bst_additional_batch_negatives == 32
    assert merged.batch_size == 16
    assert merged.bst_max_train_batches_per_epoch == 5
    assert merged.bst_use_popularity_feature is False
    assert merged.bst_popularity_projection_dim == 12
    cli._validate_bst_config(merged)


def test_bst_ranker_training_defaults():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
    ])
    merged = cli._merge_args_with_config(raw)

    assert merged.bst_additional_batch_negatives == 64
    assert merged.batch_size == cli.DEFAULTS["batch_size"]
    assert merged.eval_batch_size == 128
    assert merged.bst_max_train_batches_per_epoch is None
    assert merged.bst_use_popularity_feature is True
    assert merged.bst_popularity_projection_dim == 8
    assert merged.bst_use_post_liker_feature is False
    assert merged.bst_post_liker_user_embedding_dim == 16
    assert merged.bst_post_liker_projection_dim == 32
    assert merged.bst_post_liker_pooling_tau_hours == 168.0
    assert merged.bst_max_post_liker_replay_events_per_post == 128
    assert merged.bst_post_liker_user_unknown_dropout_rate == 0.4
    cli._validate_bst_config(merged)


def test_bst_ranker_requires_one_transformer_layer():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--bst-num-transformer-layers", "2",
    ])
    merged = cli._merge_args_with_config(raw)

    with pytest.raises(ValueError, match="requires --bst-num-transformer-layers=1"):
        cli._validate_bst_config(merged)


@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("--bst-additional-batch-negatives", "bst-additional-batch-negatives"),
        ("--batch-size", "batch-size"),
        ("--eval-batch-size", "eval-batch-size"),
        ("--bst-max-train-batches-per-epoch", "bst-max-train-batches-per-epoch"),
        ("--bst-popularity-projection-dim", "bst-popularity-projection-dim"),
        ("--bst-post-liker-user-embedding-dim", "bst-post-liker-user-embedding-dim"),
        ("--bst-post-liker-projection-dim", "bst-post-liker-projection-dim"),
        ("--bst-post-liker-pooling-tau-hours", "bst-post-liker-pooling-tau-hours"),
        ("--bst-max-post-liker-replay-events-per-post", "bst-max-post-liker-replay-events-per-post"),
    ],
)
def test_bst_ranker_rejects_non_positive_listwise_training_controls(flag, message):
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        flag, "0",
    ])
    merged = cli._merge_args_with_config(raw)

    with pytest.raises(ValueError, match=message):
        cli._validate_bst_config(merged)


def test_bst_ranker_rejects_negative_eval_batch_size():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--eval-batch-size", "-1",
    ])
    merged = cli._merge_args_with_config(raw)

    with pytest.raises(ValueError, match="eval-batch-size"):
        cli._validate_bst_config(merged)


def test_bst_ranker_uses_stage7_author_indices_without_a_feature_flag():
    parser = cli.build_parser()
    raw = parser.parse_args(["--model-type", "bst-ranker", "--bst-prediction-hidden-dims", "144", "72"])
    merged = cli._merge_args_with_config(raw)

    assert not hasattr(merged, "use_author_embedding_table")
    cli._validate_bst_config(merged)


def test_bst_ranker_requires_prediction_hidden_dims():
    parser = cli.build_parser()
    raw = parser.parse_args(["--model-type", "bst-ranker"])
    merged = cli._merge_args_with_config(raw)
    merged.bst_prediction_hidden_dims = None

    with pytest.raises(ValueError, match="bst-prediction-hidden-dims"):
        cli._validate_bst_config(merged)


def test_bst_ranker_accepts_explicit_empty_prediction_hidden_dims():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--bst-prediction-hidden-dims",
    ])
    merged = cli._merge_args_with_config(raw)

    assert merged.bst_prediction_hidden_dims == []
    cli._validate_bst_config(merged)


@pytest.mark.parametrize(
    ("arg_name", "error_match"),
    [
        ("content_projection_dim", "content-projection-dim"),
        ("author_projection_dim", "author-projection-dim"),
    ],
)
def test_bst_ranker_validates_projection_dims(arg_name, error_match):
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--bst-prediction-hidden-dims", "144", "72",
    ])
    merged = cli._merge_args_with_config(raw)
    setattr(merged, arg_name, 0)

    with pytest.raises(ValueError, match=error_match):
        cli._validate_bst_config(merged)


def test_bst_ranker_validates_transformer_head_divisibility():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--bst-time-embedding-dim", "15",
        "--bst-prediction-hidden-dims", "144", "72",
    ])
    merged = cli._merge_args_with_config(raw)

    with pytest.raises(ValueError, match="divisible"):
        cli._validate_bst_config(merged)


@pytest.mark.parametrize(
    ("config_key", "disable_flag"),
    [
        ("dataloader_pin_memory", "--no-dataloader-pin-memory"),
        ("dataloader_persistent_workers", "--no-dataloader-persistent-workers"),
        ("background", "--no-background"),
    ],
)
def test_merge_args_with_config_allows_cli_to_disable_true_config_bool(tmp_path, config_key, disable_flag):
    config_path = Path(tmp_path) / "config.yml"
    config_path.write_text(f"{config_key}: true\n")

    parser = cli.build_parser()
    raw = parser.parse_args(["--config", str(config_path), disable_flag])
    merged = cli._merge_args_with_config(raw)

    assert getattr(merged, config_key) is False


def test_merge_args_with_config_accepts_prior_pins(tmp_path):
    config_path = Path(tmp_path) / "config.yml"
    config_path.write_text(
        textwrap.dedent(
            """
            prior_00_source_metadata: 20260100_000000_source00
            prior_01_query_selection: 20260100_000000_feedface
            prior_02_user_history: 20260102_000000_cafebabe
            prior_03_post_selection: 20260103_000000_postface
            prior_04_negative_selection: 20260104_000000_baadf00d
            prior_05_post_liker_history: 20260105_000000_liker00
            prior_06_author_statistics: 20260106_000000_decafbad
            prior_07_dataset_hydration: 20260107_000000_abcd1234
            """
        ).strip()
        + "\n"
    )

    parser = cli.build_parser()
    raw = parser.parse_args(["--config", str(config_path)])
    merged = cli._merge_args_with_config(raw)

    assert merged.prior_00_source_metadata == "20260100_000000_source00"
    assert merged.prior_01_query_selection == "20260100_000000_feedface"
    assert merged.prior_02_user_history == "20260102_000000_cafebabe"
    assert merged.prior_03_post_selection == "20260103_000000_postface"
    assert merged.prior_04_negative_selection == "20260104_000000_baadf00d"
    assert merged.prior_05_post_liker_history == "20260105_000000_liker00"
    assert merged.prior_06_author_statistics == "20260106_000000_decafbad"
    assert merged.prior_07_dataset_hydration == "20260107_000000_abcd1234"


@pytest.mark.parametrize(
    "stage_folder",
    [
        "00_source_metadata",
        "01_query_selection",
        "02_user_history",
        "03_post_selection",
        "04_negative_selection",
        "05_post_liker_history",
        "06_author_statistics",
        "07_dataset_hydration",
    ],
)
def test_resolve_prior_spec_resolves_canonical_stage_run_id(tmp_path, stage_folder):
    output_root = Path(tmp_path) / "out"
    artifacts_dir = output_root / "artifacts"
    target = artifacts_dir / stage_folder / "20260101_000000_abcd1234"
    target.mkdir(parents=True)

    resolved = cli._resolve_prior_spec(
        target.name,
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder=stage_folder,
    )

    assert resolved == target.resolve()


def test_resolve_prior_spec_resolves_relative_path_against_output_root(tmp_path):
    output_root = Path(tmp_path) / "out"
    artifacts_dir = output_root / "artifacts"
    target = output_root / "some" / "custom_prior"
    target.mkdir(parents=True)

    resolved = cli._resolve_prior_spec(
        "some/custom_prior",
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="02_user_history",
    )

    assert resolved == target.resolve()


def test_resolve_prior_spec_raises_if_missing(tmp_path):
    output_root = Path(tmp_path) / "out"
    with pytest.raises(FileNotFoundError, match="01_query_selection"):
        cli._resolve_prior_spec(
            "does_not_exist",
            output_root=output_root,
            artifacts_dir=output_root / "artifacts",
            stage_folder="01_query_selection",
        )


def test_removed_legacy_training_pin_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--prior-01-get-data", "legacy"])


def test_removed_skip_embeddings_setting_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--skip-embeddings"])


_DATE_FIELDS = (
    "posts_start", "posts_end", "train_start", "val_start", "holdout_start", "holdout_end",
)


def _date_args(argv, values=None):
    raw = cli.build_parser().parse_args(argv)
    if values is not None:
        raw = cli.argparse.Namespace(**{**values, **vars(raw)})
    return cli._merge_args_with_config(raw)


def _assert_resolved_days(args, expected):
    for name, day in zip(_DATE_FIELDS, expected, strict=True):
        actual = getattr(args, name)
        if day is None:
            assert actual is None
        else:
            assert isinstance(actual, str)
            assert datetime.fromisoformat(actual) == datetime.fromisoformat(day + "T00:00:00+00:00")


@pytest.mark.parametrize("anchor, expected", [
    ("2026-10-16", ("2026-09-09", "2026-10-16", "2026-09-16", "2026-10-08", "2026-10-13", "2026-10-16")),
    ("2024-03-01", ("2024-01-24", "2024-03-01", "2024-01-31", "2024-02-22", "2024-02-27", "2024-03-01")),
    ("2026-01-05", ("2025-11-29", "2026-01-05", "2025-12-06", "2025-12-28", "2026-01-02", "2026-01-05")),
])
def test_relative_dates_default_windows_cross_calendar_boundaries(anchor, expected):
    raw = _date_args(["--posts-end", anchor])

    resolved = cli._resolve_date_args(raw)

    _assert_resolved_days(resolved, expected)
    assert not hasattr(raw, "posts_start")
    assert raw.posts_end == anchor


def test_relative_dates_config_and_cli_offsets_follow_normal_precedence(tmp_path):
    config_path = tmp_path / "relative.yml"
    config_path.write_text(
        "posts_end: 2026-10-14\nposts_start_days_before_end: 42\n"
        "train_start_days_after_posts_start: 10\nval_start_days_before_end: 9\n"
        "holdout_start_days_before_end: 5\nepochs: 2\n"
    )
    raw = _date_args(["--config", str(config_path), "--holdout-start-days-before-end", "4"])

    resolved = cli._resolve_date_args(raw)

    _assert_resolved_days(resolved, (
        "2026-09-02", "2026-10-14", "2026-09-12", "2026-10-05", "2026-10-10", "2026-10-14",
    ))
    assert resolved.epochs == 2
    assert resolved.learning_rate == cli.DEFAULTS["learning_rate"]
    # YAML dates must already be serializable when a background parent saves its inputs.
    assert isinstance(raw.posts_end, str)
    json.dumps({key: getattr(raw, key) for key in cli.DEFAULTS})


@pytest.mark.parametrize("name", [name for name in _DATE_FIELDS if name != "posts_end"])
@pytest.mark.parametrize("value", ["2026-10-01", "7"])
def test_explicit_derived_date_cli_flags_are_rejected(name, value):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--" + name.replace("_", "-"), value])

    assert name not in cli.DEFAULTS


@pytest.mark.parametrize("name", [name for name in _DATE_FIELDS if name != "posts_end"])
def test_explicit_derived_date_config_keys_are_rejected(tmp_path, name):
    config_path = tmp_path / "absolute.yml"
    config_path.write_text(f"{name}: 2026-10-01\n")

    with pytest.raises(ValueError, match=name):
        _date_args(["--config", str(config_path)])


def test_relative_dates_nullable_holdout_offset_disables_holdout():
    resolved = cli._resolve_date_args(_date_args(["--posts-end", "2026-10-16"], {
        "holdout_start_days_before_end": None,
        "train_start_days_after_posts_start": 0,
    }))

    assert resolved.holdout_start is None and resolved.holdout_end is None
    assert resolved.train_start == resolved.posts_start


@pytest.mark.parametrize("config", [
    {"posts_start_days_before_end": -1},
    {"posts_start_days_before_end": 37.5},
    {"train_start_days_after_posts_start": True},
    {"val_start_days_before_end": "8"},
    {"holdout_start_days_before_end": -1},
    {"posts_start_days_before_end": 0},
    {"train_start_days_after_posts_start": 30},
    {"val_start_days_before_end": 3},
    {"val_start_days_before_end": 40},
    {"holdout_start_days_before_end": 0},
    {"posts_end": "2026-10-16T00:30:00Z"},
])
def test_relative_dates_reject_invalid_offsets_and_windows(config):
    raw = _date_args([], {"posts_end": "2026-10-16", **config})

    with pytest.raises(ValueError):
        cli._resolve_date_args(raw)


def test_relative_dates_source_only_run_does_not_require_split_window():
    raw = _date_args([
        "--stop-after", "source_metadata", "--posts-start-days-before-end", "1",
        "--posts-end", "2026-01-02", "--val-start-days-before-end", "-1",
    ])

    resolved = cli._resolve_date_args(raw)

    assert set(cli._date_fields_for_run(raw)) == {"posts_start", "posts_end"}
    assert datetime.fromisoformat(resolved.posts_start).day == 1
    assert resolved.train_start is None and resolved.holdout_start is None


@pytest.mark.parametrize("stage", ["train", "train_bst_ranker", "dataset_hydration"])
def test_relative_dates_prepared_artifact_runs_do_not_resolve_unused_dates(stage):
    raw = _date_args(["--start-from", stage], {
        "val_start_days_before_end": -1,
    })
    raw.posts_start = "stale remote value"

    resolved = cli._resolve_date_args(cli._freeze_posts_end(raw))

    assert not cli._date_fields_for_run(raw)
    assert resolved.posts_end is None and resolved.train_start is None
    assert resolved.posts_start is None


def test_relative_dates_freeze_current_utc_day_once(monkeypatch):
    calls = []

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            calls.append(tz)
            return cls(2026, 10, 16, 23, 59, 59, tzinfo=timezone.utc)

    monkeypatch.setattr(cli, "datetime", FrozenDatetime)
    raw = _date_args([])
    frozen = cli._freeze_posts_end(raw)
    resolved = cli._resolve_date_args(cli._freeze_posts_end(frozen))

    assert calls == [timezone.utc]
    assert raw.posts_end is None
    assert datetime.fromisoformat(resolved.posts_end) == datetime(2026, 10, 16, tzinfo=timezone.utc)


@pytest.mark.parametrize("background", [False, True])
def test_relative_dates_dry_run_has_no_tracker_output_or_background_process(tmp_path, monkeypatch, capsys, background):
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(cli, "build_experiment_tracker", lambda *a, **k: pytest.fail("dry-run created tracker"))
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: pytest.fail("dry-run started child"))
    monkeypatch.setattr(cli, "cmd__run_all_exec", lambda *a, **k: pytest.fail("dry-run ran pipeline"))
    argv = ["--dry-run", "--posts-end", "2026-10-16", "--output-dir", str(output_dir)]
    if background:
        argv.append("--background")

    assert cli.main(argv) == 0

    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == set(cli.DEFAULTS) | set(_DATE_FIELDS)
    assert datetime.fromisoformat(printed["posts_start"]).date().isoformat() == "2026-09-09"
    assert printed["background"] is background
    assert not output_dir.exists()


def test_relative_dates_background_config_pins_anchor_but_keeps_starts_unresolved(tmp_path, monkeypatch):
    monkeypatch.delenv("ENGAGEMENT_PIPELINE_RUN_ID", raising=False)
    monkeypatch.setenv("ENGAGEMENT_RUN_TIMESTAMP", "20261016_000001")
    children = []

    def launch_child(command, **kwargs):
        children.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="12345\n", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", launch_child)
    monkeypatch.setattr(cli, "build_experiment_tracker", lambda *a, **k: pytest.fail("parent created tracker"))
    args = _date_args(["--background", "--posts-end", "2026-10-16", "--output-dir", str(tmp_path)])

    assert cli.cmd_run_all(args) == 0

    assert len(children) == 1
    paths = list((tmp_path / "runs").glob("*/run-all.resolved-config.json"))
    paths = [path for path in paths if not path.parent.is_symlink()]
    assert len(paths) == 1
    child_config = json.loads(paths[0].read_text())
    assert child_config["background"] is False
    assert datetime.fromisoformat(child_config["posts_end"]).date().isoformat() == "2026-10-16"
    assert not (set(child_config) & (set(_DATE_FIELDS) - {"posts_end"}))
    child_args = _date_args(["--config", str(paths[0])])
    resolved = cli._resolve_date_args(cli._freeze_posts_end(child_args))
    assert datetime.fromisoformat(resolved.posts_start).date().isoformat() == "2026-09-09"


def test_relative_dates_remote_overrides_resolve_before_stages_and_save_reloadable_config(tmp_path, monkeypatch):
    from engagement_prediction.experiment_tracking.base import NoOpExperimentTracker

    monkeypatch.delenv("ENGAGEMENT_PIPELINE_RUN_ID", raising=False)
    received = {}
    logged = {}

    class RemoteTracker(NoOpExperimentTracker):
        def connect_args(self, args, name=None):
            assert not hasattr(args, "posts_start") and not hasattr(args, "train_start")
            args.posts_end = "2026-10-14"
            args.posts_start_days_before_end = 42
            # Older remote tasks can contain absolute values; offsets must win.
            for key in _DATE_FIELDS:
                if key != "posts_end":
                    setattr(args, key, "2020-01-01")
            return args

        def log_params(self, params, name=None):
            logged[name] = params

    monkeypatch.setattr(cli, "build_experiment_tracker", lambda *a, **k: RemoteTracker())

    def execute(args, ctx):
        received["args"] = args
        received["run_dir"] = ctx.run_dir
        return 7

    monkeypatch.setattr(cli, "cmd__run_all_exec", execute)

    assert cli.main(["--posts-end", "2026-10-16", "--output-dir", str(tmp_path)]) == 7

    expected = ("2026-09-02", "2026-10-14", "2026-09-09", "2026-10-06", "2026-10-11", "2026-10-14")
    _assert_resolved_days(received["args"], expected)
    assert logged["ResolvedDates"] == {name: getattr(received["args"], name) for name in _DATE_FIELDS}
    saved = received["run_dir"] / "run-all.resolved-config.json"
    assert set(json.loads(saved.read_text())) == set(cli.DEFAULTS)
    _assert_resolved_days(cli._resolve_date_args(_date_args(["--config", str(saved)])), expected)


def test_relative_dates_invalid_remote_override_marks_task_failed_and_closes(tmp_path, monkeypatch):
    from engagement_prediction.experiment_tracking.base import NoOpExperimentTracker

    events = []

    class InvalidRemoteTracker(NoOpExperimentTracker):
        def connect_args(self, args, name=None):
            args.val_start_days_before_end = 40
            return args

        def mark_failed(self, reason):
            events.append("failed")

        def close(self):
            events.append("closed")

    monkeypatch.setattr(cli, "build_experiment_tracker", lambda *a, **k: InvalidRemoteTracker())
    monkeypatch.setattr(cli, "cmd__run_all_exec", lambda *a, **k: pytest.fail("invalid dates ran pipeline"))

    with pytest.raises(ValueError):
        cli.main(["--posts-end", "2026-10-16", "--output-dir", str(tmp_path / "outputs")])

    assert events == ["failed", "closed"]
    assert not (tmp_path / "outputs").exists()
