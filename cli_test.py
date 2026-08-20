from pathlib import Path
import textwrap
from datetime import datetime, timezone

import polars as pl
import pytest

import cli


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
    ],
)
def test_obsolete_negative_sampling_flags_are_rejected(flag):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([flag, "1"])
    assert flag.removeprefix("--").replace("-", "_") not in cli.DEFAULTS


def test_post_selection_args_merge_from_cli_and_config(tmp_path):
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--random-candidate-sampling-fraction", "0.2",
        "--post-selection-partition-count", "64",
    ])
    merged = cli._merge_args_with_config(raw)

    assert merged.random_candidate_sampling_fraction == 0.2
    assert merged.post_selection_partition_count == 64

    config_path = Path(tmp_path) / "post_selection.yml"
    config_path.write_text(
        "random_candidate_sampling_fraction: 0.3\n"
        "post_selection_partition_count: 32\n"
    )
    raw = parser.parse_args(["--config", str(config_path)])
    merged = cli._merge_args_with_config(raw)

    assert merged.random_candidate_sampling_fraction == 0.3
    assert merged.post_selection_partition_count == 32


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
    assert cli.DEFAULTS["max_hours_per_user_per_split"] == 64
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
    assert merged.max_hours_per_user_per_split == 64
    assert merged.max_train_query_hours is None
    assert merged.max_eval_query_hours_per_split is None
    assert merged.max_positives_per_user_hour == 32
    assert merged.max_history_posts_per_query == 64
    assert merged.user_history_partition_count == 32
    assert merged.random_candidate_sampling_fraction == 0.10
    assert merged.post_selection_partition_count == 32

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--max-prior-likes", "64"])


def test_post_selection_cannot_continue_into_legacy_training(tmp_path):
    parser = cli.build_parser()
    merged = cli._merge_args_with_config(parser.parse_args([]))
    merged.output_dir = str(tmp_path)
    ctx = cli.Context(
        run_dir=Path(tmp_path) / "runs" / "run",
        artifacts_dir=Path(tmp_path) / "artifacts",
        runs_dir=Path(tmp_path) / "runs",
        pipeline_run_id="run",
    )

    with pytest.raises(ValueError, match="--stop-after post_selection"):
        cli.cmd__run_all_exec(merged, ctx)


def test_new_pipeline_can_stop_after_user_history():
    args = cli._merge_args_with_config(
        cli.build_parser().parse_args(["--stop-after", "user_history"])
    )
    stage_order = cli._get_stage_order_for_model_type(cli._get_train_key(args.model_type))
    start_idx, stop_idx, _ = cli._get_stage_folder_and_start_stop_indices(
        stage_order,
        args.start_from,
        args.stop_after,
        cli._get_train_key(args.model_type),
    )

    cli._validate_data_pipeline_boundary(args, stage_order, start_idx, stop_idx)


def test_new_pipeline_can_stop_after_post_selection():
    args = cli._merge_args_with_config(
        cli.build_parser().parse_args(["--stop-after", "post_selection"])
    )
    stage_order = cli._get_stage_order_for_model_type(cli._get_train_key(args.model_type))
    start_idx, stop_idx, _ = cli._get_stage_folder_and_start_stop_indices(
        stage_order,
        args.start_from,
        args.stop_after,
        cli._get_train_key(args.model_type),
    )

    cli._validate_data_pipeline_boundary(args, stage_order, start_idx, stop_idx)


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

    def list_sources(*, blob_prefix, **kwargs):
        if blob_prefix == "bsky_likes":
            return [str(likes_path)], [datetime(2026, 1, 1, 8, tzinfo=timezone.utc)]
        assert blob_prefix == "bsky_posts"
        return [str(posts_path)], [datetime(2026, 1, 1, 8, tzinfo=timezone.utc)]

    monkeypatch.setattr(
        "engagement_prediction.data.ingex.list_ingex_parquet_files",
        list_sources,
    )
    args = cli._merge_args_with_config(cli.build_parser().parse_args([
        "--stop-after", "user_history",
        "--posts-start", "2026-01-01T00:00:00Z",
        "--posts-end", "2026-01-02T00:00:00Z",
        "--train-start", "2026-01-01T10:00:00Z",
        "--val-start", "2026-01-01T20:00:00Z",
        "--unseen-user-fraction", "0",
        "--user-history-partition-count", "2",
        "--post-selection-partition-count", "2",
    ]))
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


def test_new_pipeline_runs_query_selection_through_post_selection(tmp_path, monkeypatch):
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
        "--stop-after", "post_selection",
        "--posts-start", "2026-01-01T00:00:00Z",
        "--posts-end", "2026-01-02T00:00:00Z",
        "--train-start", "2026-01-01T10:00:00Z",
        "--val-start", "2026-01-01T20:00:00Z",
        "--unseen-user-fraction", "0",
        "--user-history-partition-count", "2",
        "--post-selection-partition-count", "2",
        "--random-candidate-sampling-fraction", "0",
    ]))
    output_root = Path(tmp_path) / "output"
    output_root.mkdir()
    args.output_dir = str(output_root)
    args._argv = ["--stop-after", "post_selection"]
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


def test_direct_legacy_training_requires_both_explicit_pins():
    args = cli._merge_args_with_config(
        cli.build_parser().parse_args(["--start-from", "train", "--stop-after", "train"])
    )
    stage_order = cli._get_stage_order_for_model_type(cli._get_train_key(args.model_type))
    start_idx, stop_idx, _ = cli._get_stage_folder_and_start_stop_indices(
        stage_order,
        args.start_from,
        args.stop_after,
        cli._get_train_key(args.model_type),
    )
    with pytest.raises(ValueError, match="explicit --prior-01-get-data"):
        cli._validate_data_pipeline_boundary(args, stage_order, start_idx, stop_idx)

    args.prior_01_get_data = "get"
    args.prior_02_user_history = "history"
    cli._validate_data_pipeline_boundary(args, stage_order, start_idx, stop_idx)


def test_background_effective_config_preserves_no_post_encoder(tmp_path):
    parser = cli.build_parser()
    raw = parser.parse_args(["--no-post-encoder"])
    merged = cli._merge_args_with_config(raw)

    output_root = Path(tmp_path) / "out"
    run_dir = output_root / "runs" / "run"
    initial_log = run_dir / "run-all.log"
    cfg = cli._build_effective_config_for_background_run(
        merged, output_root=output_root, initial_log=initial_log
    )

    assert cfg["use_post_encoder"] is False
    assert cfg["background"] is False
    assert cfg["output_dir"] == str(output_root.resolve())
    assert cfg["_initial_log"] == str(initial_log)


def test_merge_args_with_config_defaults_l2_normalize_embeddings_to_false():
    parser = cli.build_parser()
    raw = parser.parse_args([])
    merged = cli._merge_args_with_config(raw)

    assert merged.l2_normalize_embeddings is False


def test_mlp_allows_cross_attention_user_encoder():
    parser = cli.build_parser()
    raw = parser.parse_args(["--model-type", "mlp", "--user-encoder", "cross_attention"])
    merged = cli._merge_args_with_config(raw)

    assert merged.user_encoder in cli.VALID_USER_ENCODERS_BY_MODEL_TYPE[merged.model_type]


def test_two_tower_rejects_summarized_user_encoder_before_running_stages(tmp_path):
    parser = cli.build_parser()
    raw = parser.parse_args(["--model-type", "two-tower", "--user-encoder", "summarized"])
    merged = cli._merge_args_with_config(raw)
    merged.output_dir = str(tmp_path)
    ctx = cli.Context(
        run_dir=Path(tmp_path) / "runs" / "run",
        artifacts_dir=Path(tmp_path) / "artifacts",
        runs_dir=Path(tmp_path) / "runs",
        pipeline_run_id="run",
    )

    with pytest.raises(ValueError, match="user-encoder 'summarized'"):
        cli.cmd__run_all_exec(merged, ctx)


def test_mlp_allows_author_embedding_table():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "mlp",
        "--user-encoder", "summarized",
        "--use-author-embedding-table",
        "--author-embedding-dim", "8",
    ])
    merged = cli._merge_args_with_config(raw)

    assert merged.use_author_embedding_table is True
    assert merged.model_type == "mlp"


def test_bst_ranker_model_type_maps_train_alias():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--start-from", "train",
        "--stop-after", "train",
        "--use-author-embedding-table",
        "--prediction-hidden-dims", "144", "72",
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
        "--use-author-embedding-table",
        "--prediction-hidden-dims", "144", "72",
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
            use_author_embedding_table: true
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
            prediction_hidden_dims: [128, 64]
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
    assert merged.prediction_hidden_dims == [128, 64]
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
        "--use-author-embedding-table",
    ])
    merged = cli._merge_args_with_config(raw)

    assert merged.bst_additional_batch_negatives == 64
    assert merged.batch_size == cli.DEFAULTS["batch_size"]
    assert merged.bst_max_train_batches_per_epoch is None
    assert merged.bst_use_popularity_feature is True
    assert merged.bst_popularity_projection_dim == 8
    cli._validate_bst_config(merged)


def test_bst_ranker_requires_one_transformer_layer():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--use-author-embedding-table",
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
        ("--bst-max-train-batches-per-epoch", "bst-max-train-batches-per-epoch"),
        ("--bst-popularity-projection-dim", "bst-popularity-projection-dim"),
    ],
)
def test_bst_ranker_rejects_non_positive_listwise_training_controls(flag, message):
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--use-author-embedding-table",
        flag, "0",
    ])
    merged = cli._merge_args_with_config(raw)

    with pytest.raises(ValueError, match=message):
        cli._validate_bst_config(merged)


def test_bst_ranker_requires_author_embedding_table():
    parser = cli.build_parser()
    raw = parser.parse_args(["--model-type", "bst-ranker", "--prediction-hidden-dims", "144", "72"])
    merged = cli._merge_args_with_config(raw)

    with pytest.raises(ValueError, match="use-author-embedding-table"):
        cli._validate_bst_config(merged)


def test_bst_ranker_requires_prediction_hidden_dims():
    parser = cli.build_parser()
    raw = parser.parse_args(["--model-type", "bst-ranker", "--use-author-embedding-table"])
    merged = cli._merge_args_with_config(raw)
    merged.prediction_hidden_dims = None

    with pytest.raises(ValueError, match="prediction-hidden-dims"):
        cli._validate_bst_config(merged)


def test_bst_ranker_accepts_explicit_empty_prediction_hidden_dims():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--use-author-embedding-table",
        "--prediction-hidden-dims",
    ])
    merged = cli._merge_args_with_config(raw)

    assert merged.prediction_hidden_dims == []
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
        "--use-author-embedding-table",
        "--prediction-hidden-dims", "144", "72",
    ])
    merged = cli._merge_args_with_config(raw)
    setattr(merged, arg_name, 0)

    with pytest.raises(ValueError, match=error_match):
        cli._validate_bst_config(merged)


def test_bst_ranker_validates_transformer_head_divisibility():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--use-author-embedding-table",
        "--bst-time-embedding-dim", "15",
        "--prediction-hidden-dims", "144", "72",
    ])
    merged = cli._merge_args_with_config(raw)

    with pytest.raises(ValueError, match="divisible"):
        cli._validate_bst_config(merged)


def test_min_author_support_must_be_positive_even_without_author_table(tmp_path):
    parser = cli.build_parser()
    raw = parser.parse_args(["--min-author-support", "0", "--stop-after", "query_selection"])
    merged = cli._merge_args_with_config(raw)
    merged.output_dir = str(tmp_path)
    ctx = cli.Context(
        run_dir=Path(tmp_path) / "runs" / "run",
        artifacts_dir=Path(tmp_path) / "artifacts",
        runs_dir=Path(tmp_path) / "runs",
        pipeline_run_id="run",
    )

    with pytest.raises(ValueError, match="min-author-support"):
        cli.cmd__run_all_exec(merged, ctx)


def test_background_effective_config_preserves_no_l2_normalize_embeddings(tmp_path):
    parser = cli.build_parser()
    raw = parser.parse_args(["--no-l2-normalize-embeddings"])
    merged = cli._merge_args_with_config(raw)

    output_root = Path(tmp_path) / "out"
    run_dir = output_root / "runs" / "run"
    initial_log = run_dir / "run-all.log"
    cfg = cli._build_effective_config_for_background_run(
        merged, output_root=output_root, initial_log=initial_log
    )

    assert cfg["l2_normalize_embeddings"] is False


def test_background_effective_config_allows_cli_to_override_config_to_default(tmp_path):
    # Config disables post encoder, CLI re-enables it (even though True is the DEFAULTS value).
    config_path = Path(tmp_path) / "config.yml"
    config_path.write_text("use_post_encoder: false\n")

    parser = cli.build_parser()
    raw = parser.parse_args(["--config", str(config_path), "--post-encoder"])
    merged = cli._merge_args_with_config(raw)

    output_root = Path(tmp_path) / "out"
    run_dir = output_root / "runs" / "run"
    initial_log = run_dir / "run-all.log"
    cfg = cli._build_effective_config_for_background_run(
        merged, output_root=output_root, initial_log=initial_log
    )

    assert cfg["use_post_encoder"] is True


def test_background_effective_config_allows_cli_to_override_config_to_default_l2_normalization(tmp_path):
    config_path = Path(tmp_path) / "config.yml"
    config_path.write_text("l2_normalize_embeddings: false\n")

    parser = cli.build_parser()
    raw = parser.parse_args(["--config", str(config_path), "--l2-normalize-embeddings"])
    merged = cli._merge_args_with_config(raw)

    output_root = Path(tmp_path) / "out"
    run_dir = output_root / "runs" / "run"
    initial_log = run_dir / "run-all.log"
    cfg = cli._build_effective_config_for_background_run(
        merged, output_root=output_root, initial_log=initial_log
    )

    assert cfg["l2_normalize_embeddings"] is True


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
            prior_01_query_selection: 20260100_000000_feedface
            prior_01_get_data: 20260101_000000_deadbeef
            prior_02_user_history: 20260102_000000_cafebabe
            """
        ).strip()
        + "\n"
    )

    parser = cli.build_parser()
    raw = parser.parse_args(["--config", str(config_path)])
    merged = cli._merge_args_with_config(raw)

    assert merged.prior_01_query_selection == "20260100_000000_feedface"
    assert merged.prior_01_get_data == "20260101_000000_deadbeef"
    assert merged.prior_02_user_history == "20260102_000000_cafebabe"
