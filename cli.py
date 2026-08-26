#!/usr/bin/env python3

"""
Unified CLI for Engagement Prediction Pipeline
=============================================

Runs the engagement prediction artifact pipeline.

Note: The historical `run-all` subcommand is now optional (kept for backwards compatibility).

Usage examples:
    python cli.py --config config.yml --stop-after dataset_hydration
    python cli.py run-all --config config.yml --stop-after dataset_hydration
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
import copy

import compare as compare_rankers
from engagement_prediction.pipeline import registry as reg
from engagement_prediction.pipeline.dependencies import (
    pin_lineage_aligned_inputs,
    validate_explicit_prior_pin_consistency,
)
from engagement_prediction.experiment_tracking import build_experiment_tracker
from engagement_prediction.pipeline.core import (
    Context,
    generate_run_timestamp,
    LINEAGE_FILENAME,
    new_pipeline_run_dir,
    ensure_pipeline_run_dir,
    update_latest_symlink,
)


# Avoid heavy imports at module import time; import lazily inside handlers

CLI_FILE_DIR = Path(__file__).parent

TRAIN_PLACEHOLDER = 'train_placeholder'
STAGE_ORDER = ['source_metadata', 'query_selection', 'user_history', 'post_selection', 'negative_selection', 'post_liker_history', 'author_statistics', 'dataset_hydration', TRAIN_PLACEHOLDER]

# Central default map for all run-all parameters
DEFAULTS: Dict[str, Any] = {
    "output_dir": None,
    "random_seed": 42,
    "embedding_model": "all_MiniLM_L12_v2",
    # Stage 00: Canonical source metadata
    "gcs_bucket": 'greenearth-471522-ingex-extract-stage',
    "posts_start": None,
    "posts_end": None,
    "source_metadata_partition_count": 16,
    # Stage 1: Query selection
    "unseen_user_fraction": 0.1,
    "max_hours_per_user_per_split": 16,
    "max_train_query_hours": None,
    "max_eval_query_hours_per_split": None,
    "max_positives_per_user_hour": 32,
    "train_start": None,
    "val_start": None,
    "holdout_start": None,
    "holdout_end": None,
    # Stage 2: User history
    "max_history_posts_per_query": 64,
    "user_history_partition_count": 16,
    # Stage 3: Post selection
    "random_candidate_sampling_fraction": 0.1,
    # Stage 4: Popularity-aware negative selection
    "negative_candidates_per_hour": 1000,
    "min_likes_for_popular_candidate": 10,
    "popular_candidate_fraction": 0.50,
    "max_candidate_age_hours": 24,
    # Stage 5: Post-liker history extraction
    "post_liker_history_partition_count": 16,
    # Stage 6: Training-only author statistics
    "author_statistics_partition_count": 16,
    # Stage 7: Dataset hydration
    "embedding_source_batch_size": 64,
    "embedding_partition_worker_count": 4,
    "min_author_training_feature_count": 50,
    # Stage 8: Model architecture
    "model_type": "bst-ranker",
    "output_embedding_dim": 128,
    "user_hidden_dim": 256,
    "post_hidden_dim": 256,
    "max_history_len": 20,
    "similarity_temperature": 1.0,
    "author_embedding_dim": 16,
    "author_unknown_dropout_rate": 0.3,
    "epochs": 300,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay_two_tower": 0.01,
    "content_projection_dim": 128,
    "author_projection_dim": 32,
    "prediction_hidden_dims": [64, 32, 16],
    "bst_additional_batch_negatives": 64,
    "bst_model_dim": 128,
    "bst_time_embedding_dim": 16,
    "bst_num_attention_heads": 4,
    "bst_num_transformer_layers": 1,
    "bst_transformer_ff_dim": 256,
    "bst_dropout_rate": 0.1,
    "bst_norm_first": False,
    "bst_time_delta_bucket_boundaries_hours": [1.0, 3.0, 6.0, 12.0, 24.0, 72.0, 168.0, 720.0, 2160.0],
    "bst_weight_decay": 0.01,
    "bst_max_train_batches_per_epoch": None,
    "bst_use_popularity_feature": True,
    "bst_popularity_projection_dim": 8,
    "dropout_rate_two_tower": 0.1,
    "device": None,
    "patience": 50,
    "early_stopping_min_delta": 0.002,
    "run_tag": None,  # Optional tag appended to training output directory name
    "no_plots": False,
    "disable_progress": False,  # Disable progress bars during training
    "metrics_top_ks": [30],
    # Stage 8 - DataLoader settings
    "num_dataloader_workers": 4,
    "dataloader_pin_memory": True,
    "dataloader_persistent_workers": True,
    "dataloader_prefetch_factor": 2,
    # Stage 8 - Learning rate scheduler
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 5,
    # Stage 8 - Training optimization
    "gradient_clip_max_norm": 1.0,
    # Validation
    "eval_batch_size": 128,
    # Selection/prior behavior
    "use_latest": False,
    "start_from": None,
    "stop_after": None,
    "pick_prior": False,
    # Prior pins (optional): may be a stage_run_id (dir name under artifacts/<stage>/)
    # or a path (absolute, or relative to --output-dir).
    "prior_00_source_metadata": None,
    "prior_01_query_selection": None,
    "prior_02_user_history": None,
    "prior_03_post_selection": None,
    "prior_04_negative_selection": None,
    "prior_05_post_liker_history": None,
    "prior_06_author_statistics": None,
    "prior_07_dataset_hydration": None,
    # Execution behavior
    # Default is foreground execution (recommended for ClearML remote execution).
    "background": False,
    "_initial_log": None,
    # Experiment tracking
    "experiment_tracker": "clearml",
    "experiment_project": "Engagement Prediction",
    "experiment_task": None,
    "experiment_tags": None,
    # ClearML / model registry
    # If set, used as ClearML Task output URI (e.g. gs://...); if None, ClearML uses its default output.
    "model_output_uri": 'gs://greenearth-471522-engagement-prediction-test',
}


def _help_with_default(text: Optional[str], key: str) -> Optional[str]:
    """Append default value text without duplicating default assignments."""
    default_val = DEFAULTS.get(key, None)
    if text is None:
        text = ""
    if default_val is None:
        return text
    return f"{text} (default: {default_val})"


def _arg_key_from_flag(flag: str) -> str:
    """Convert a CLI flag (e.g., --posts-start) to the DEFAULTS key."""
    return flag.lstrip("-").replace("-", "_")


def _add_arg_with_default(parser: argparse.ArgumentParser, flag: str, *, key: Optional[str] = None,
                          help_text: Optional[str] = None, **kwargs: Any) -> None:
    """Add an argument with standardized default-aware help text."""
    if help_text is not None:
        effective_key = key or _arg_key_from_flag(flag)
        kwargs["help"] = _help_with_default(help_text or "", effective_key)
    parser.add_argument(flag, **kwargs)


def _extract_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key, default in DEFAULTS.items():
        if hasattr(args, key):
            value = getattr(args, key)
            if value != default:
                overrides[key] = value
    return overrides


def _load_config_file(path_str: str) -> Dict[str, Any]:
    """Load a YAML (or JSON) config file mapping CLI args to values."""
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None  # type: ignore
    if yaml is not None:
        data = yaml.safe_load(path.read_text())
    else:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError("PyYAML is not installed and the config file is not valid JSON") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping of argument names to values")
    # Normalize kebab-case to snake_case to match argparse dest names
    return {k.replace("-", "_"): v for k, v in data.items()}


def _merge_args_with_config(raw_args: argparse.Namespace) -> argparse.Namespace:
    """Apply defaults, then config file values, then CLI overrides."""
    args_dict = vars(raw_args).copy()
    command = args_dict.get("command")
    func = args_dict.get("func")
    config_path = args_dict.pop("config", None)
    config_data: Dict[str, Any] = {}
    if config_path:
        config_data = _load_config_file(config_path)

    merged: Dict[str, Any] = copy.deepcopy(DEFAULTS)
    if config_data:
        unknown_keys = set(config_data.keys()) - set(DEFAULTS.keys())
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown_keys))}")
        merged.update(config_data)
    merged.update({k: v for k, v in args_dict.items() if k not in ("command", "func")})
    if merged["model_type"] not in {"bst-ranker", "two-tower"}:
        raise ValueError(
            f"Unknown model_type: {merged['model_type']!r}. "
            "Choose 'bst-ranker' or 'two-tower'."
        )
    final_ns = argparse.Namespace(**merged)
    # Preserve argparse-injected metadata
    setattr(final_ns, "command", command)
    setattr(final_ns, "func", func)
    return final_ns


def _build_effective_config_for_background_run(
    args: argparse.Namespace, *, output_root: Path, initial_log: Path
) -> Dict[str, Any]:
    """Materialize an effective config to re-invoke run-all in the background.

    We prefer passing a config file rather than reconstructing CLI flags from
    argparse destination names.
    """
    cfg: Dict[str, Any] = {k: getattr(args, k) for k in DEFAULTS.keys()}
    cfg["output_dir"] = str(Path(output_root).resolve())
    cfg["_initial_log"] = str(initial_log)
    # Prevent recursive backgrounding: the child process should run in the foreground.
    cfg["background"] = False
    return cfg


def _generate_run_name(args: argparse.Namespace) -> str:
    stages_str = "all"
    if args.start_from is not None or args.stop_after is not None:
        if args.start_from == args.stop_after:
            stages_str = args.start_from
        else:
            if args.start_from is None:
                stages_str = "start_to_"
            else:
                stages_str = f"{args.start_from}_to_"
            if args.stop_after is None:
                stages_str += "end"
            else:
                stages_str += args.stop_after

    # Add the model type if the training stage is included
    model_type = args.model_type
    train_key = _get_train_key(model_type)
    stage_order = _get_stage_order_for_model_type(train_key)
    _, _, includes_train = _get_stage_folder_and_start_stop_indices(
        stage_order,
        args.start_from,
        args.stop_after,
        train_key
    )
    if includes_train:
        stages_str += f"_{model_type}"

    return stages_str


def _resolve_run_dir(args: argparse.Namespace, run_timestamp: str) -> Path:
    """Resolve the output root directory as an absolute path.

    ClearML remote execution may run with a different working directory than local runs.
    If `--output-dir` is provided as a relative path, interpret it relative to the repo root
    (this file's directory) to keep behavior stable across environments.
    """
    output_dir = args.output_dir
    if output_dir:
        p = Path(str(output_dir)).expanduser()
        if not p.is_absolute():
            p = (CLI_FILE_DIR / p)
        return p.resolve()
    return (CLI_FILE_DIR / "outputs").resolve()


def _resolve_pipeline_run_dir(args: argparse.Namespace, *, output_root: Path, run_timestamp: str) -> Path:
    runs_dir = (Path(output_root) / "runs").resolve()
    pinned = (os.environ.get("ENGAGEMENT_PIPELINE_RUN_ID") or "").strip()
    if pinned:
        return ensure_pipeline_run_dir(runs_dir, pipeline_run_id=pinned).resolve()
    run_name = _generate_run_name(args)
    base_name = f"{run_timestamp}_{run_name}"
    return new_pipeline_run_dir(runs_dir, base_name=base_name).resolve()


def _resolve_prior_spec(
    spec: Optional[str],
    *,
    output_root: Path,
    artifacts_dir: Path,
    stage_folder: str,
) -> Optional[Path]:
    """Resolve a prior pin to a concrete artifact directory path.

    `spec` may be:
      - an absolute path
      - a path relative to output_root
      - a stage_run_id (directory name under artifacts/<stage_folder>/)
    """
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        return None

    p = Path(s).expanduser()
    candidate = p if p.is_absolute() else (Path(output_root) / p)
    if candidate.exists():
        return candidate.resolve()

    by_id = (Path(artifacts_dir) / stage_folder / s)
    if by_id.exists():
        return by_id.resolve()

    raise FileNotFoundError(
        f"Could not resolve prior spec for '{stage_folder}': {spec!r}. "
        f"Expected an existing path (absolute or relative to {Path(output_root).resolve()}) "
        f"or a stage_run_id under {Path(artifacts_dir).resolve() / stage_folder}."
    )


def cmd_compare_rankers(args: argparse.Namespace) -> int:
    return compare_rankers.cmd_compare_rankers(
        args,
        resolve_run_dir=_resolve_run_dir,
        resolve_prior_spec=_resolve_prior_spec,
    )


def cmd_run_all(args: argparse.Namespace) -> int:
    """Run the selected pipeline stages.

    Creates a run directory up front and backgrounds itself with nohup if --background.
    """
    train_key = _get_train_key(args.model_type)
    stage_order = _get_stage_order_for_model_type(train_key)
    start_idx, stop_idx, _ = _get_stage_folder_and_start_stop_indices(
        stage_order,
        args.start_from,
        args.stop_after,
        train_key,
    )

    # Store the single timestamp in Context; for background runs we pass it via env.
    run_timestamp = (os.environ.get("ENGAGEMENT_RUN_TIMESTAMP") or "").strip() or generate_run_timestamp()

    if bool(args.background):
        output_root = _resolve_run_dir(args, run_timestamp=run_timestamp)
        output_root.mkdir(parents=True, exist_ok=True)
        run_dir = _resolve_pipeline_run_dir(args, output_root=output_root, run_timestamp=run_timestamp)
        update_latest_symlink(output_root / "runs", run_dir)

        # Choose log path inside run_dir
        if args._initial_log:
            initial_log = Path(args._initial_log)
        else:
            initial_log = (run_dir / "run-all.log")
        try:
            initial_log.parent.mkdir(parents=True, exist_ok=True)
            with open(initial_log, 'a') as f:
                f.write(f"run-all started at {run_timestamp}\n")
        except Exception:
            pass

        # Background via nohup by re-invoking run-all in the foreground (background disabled)
        # with a pinned --output-dir.
        import shlex
        resolved_config = _build_effective_config_for_background_run(
            args, output_root=output_root, initial_log=initial_log
        )
        resolved_config_path = run_dir / "run-all.resolved-config.json"
        resolved_config_path.write_text(json.dumps(resolved_config, indent=2, sort_keys=True) + "\n")
        cli_args = ["--config", str(resolved_config_path)]

        py = shlex.quote(sys.executable)
        script = shlex.quote(str(Path(__file__).resolve()))
        args_str = ' '.join(shlex.quote(a) for a in cli_args)
        redir = shlex.quote(str(initial_log))
        env_prefix = (
            f"ENGAGEMENT_RUN_TIMESTAMP={shlex.quote(run_timestamp)} "
            f"ENGAGEMENT_PIPELINE_RUN_ID={shlex.quote(run_dir.name)}"
        )
        cmd = f"{env_prefix} nohup {py} {script} {args_str} > {redir} 2>&1 & echo $!"
        print(f"▶️  Backgrounding run-all with nohup. Log: {initial_log}")
        import subprocess as sp
        proc = sp.run(["bash", "-lc", cmd], stdout=sp.PIPE, stderr=sp.PIPE, text=True)
        if proc.returncode == 0:
            pid_str = (proc.stdout or "").strip().splitlines()[-1] if (proc.stdout or "").strip() else None
            pid_file = (run_dir / "run-all.pid")
            if pid_str and pid_str.isdigit():
                try:
                    with open(pid_file, "w") as f:
                        f.write(pid_str + "\n")
                except Exception:
                    pass
                print(f"✅ run-all started in background (PID {pid_str}). Kill with: kill {pid_str}\n📝 PID file: {pid_file}")
            else:
                print("✅ run-all started in background")
            return 0
        print("❌ Failed to start run-all in background")
        return proc.returncode or 1

    # Foreground execution: initialize experiment tracker and run
    # Only initialize ClearML here (not before backgrounding) to avoid creating
    # a task in the parent process that gets "aborted" when the parent exits.
    #
    # Pre-import torch so it's fully cached in sys.modules before ClearML patches
    # builtins.__import__. ClearML's patched importer breaks torch's internal
    # circular import chain (torch.jit._async -> torch.utils.set_module).
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    tracker = build_experiment_tracker(
        args.experiment_tracker,
        project_name=args.experiment_project,
        task_name=args.experiment_task or _generate_run_name(args),
        tags=args.experiment_tags,
        model_output_uri=args.model_output_uri,
    )
    # ClearML remote execution can override parameters on the server/UI.
    # Connect args and rehydrate a Namespace so downstream code sees the updated values.
    args = tracker.connect_args(args, "Args")

    output_root = _resolve_run_dir(args, run_timestamp=run_timestamp)
    output_root.mkdir(parents=True, exist_ok=True)

    # Resolve pipeline run dir after ClearML connects args, since output_dir might have been overridden.
    run_dir = _resolve_pipeline_run_dir(args, output_root=output_root, run_timestamp=run_timestamp)
    update_latest_symlink(output_root / "runs", run_dir)
    tracker.log_params(params={
            "run_dir": str(run_dir.resolve()),
            "run_name": run_dir.name
        },
        name="Directories"
    )

    # Ensure args.output_dir is set (this is the output root).
    setattr(args, 'output_dir', str(output_root))
    setattr(args, "_argv", sys.argv[:])

    # Pipeline run scaffolding
    artifacts_dir = (output_root / "artifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name in ("tmp", "metrics", "plots", "logs"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    run_resolved_config_path = run_dir / "run-all.resolved-config.json"
    args_dict = {k: v for k, v in vars(args).items() if k != "func" and not callable(v)}
    run_resolved_config_path.write_text(json.dumps(args_dict, indent=2, sort_keys=True) + "\n")

    lineage_path = run_dir / LINEAGE_FILENAME
    if not lineage_path.exists():
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(CLI_FILE_DIR),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            git_sha = (proc.stdout or "").strip() if proc.returncode == 0 else None
        except Exception:
            git_sha = None
        lineage_path.write_text(json.dumps({
            "pipeline_run_id": run_dir.name,
            "created_at": datetime.now().isoformat(),
            "git_sha": git_sha,
            "argv": sys.argv[:],
            "stages": {},
        }, indent=2, sort_keys=True) + "\n")

    # In sequential execution, always allow stages to resolve latest artifacts from prior stages
    ctx = Context(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        runs_dir=(output_root / "runs").resolve(),
        pipeline_run_id=run_dir.name,
        run_timestamp=run_timestamp,
        use_latest=True,
        tracker=tracker,
    )
    return cmd__run_all_exec(args, ctx)


def _get_train_key(model_type: str) -> str:
    if model_type == 'two-tower':
        return 'train_two_tower'
    elif model_type == 'bst-ranker':
        return 'train_bst_ranker'
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _validate_bst_config(args: argparse.Namespace) -> None:
    if args.prediction_hidden_dims is None:
        raise ValueError("--prediction-hidden-dims is required when --model-type is 'bst-ranker'.")

    if isinstance(args.prediction_hidden_dims, (str, bytes)):
        raise ValueError("--prediction-hidden-dims must be a list of integers.")
    try:
        prediction_hidden_dims = tuple(int(v) for v in args.prediction_hidden_dims)
    except (TypeError, ValueError) as exc:
        raise ValueError("--prediction-hidden-dims must be a list of integers.") from exc
    if any(dim <= 0 for dim in prediction_hidden_dims):
        raise ValueError("--prediction-hidden-dims values must be positive integers.")

    model_dim = int(args.bst_model_dim)
    content_projection_dim = int(args.content_projection_dim)
    author_projection_dim = int(args.author_projection_dim)
    time_embedding_dim = int(args.bst_time_embedding_dim)
    num_attention_heads = int(args.bst_num_attention_heads)
    num_transformer_layers = int(args.bst_num_transformer_layers)
    bst_additional_batch_negatives = int(args.bst_additional_batch_negatives)
    batch_size = int(args.batch_size)
    eval_batch_size = int(args.eval_batch_size)
    bst_max_train_batches_per_epoch = args.bst_max_train_batches_per_epoch
    bst_popularity_projection_dim = int(args.bst_popularity_projection_dim)
    author_embedding_dim = int(args.author_embedding_dim)
    author_unknown_dropout_rate = float(args.author_unknown_dropout_rate)
    if model_dim <= 0:
        raise ValueError("--bst-model-dim must be positive.")
    if content_projection_dim <= 0:
        raise ValueError("--content-projection-dim must be positive.")
    if author_projection_dim <= 0:
        raise ValueError("--author-projection-dim must be positive.")
    if time_embedding_dim <= 0:
        raise ValueError("--bst-time-embedding-dim must be positive.")
    if num_attention_heads <= 0:
        raise ValueError("--bst-num-attention-heads must be positive.")
    if (model_dim + time_embedding_dim) % num_attention_heads != 0:
        raise ValueError("--bst-model-dim + --bst-time-embedding-dim must be divisible by --bst-num-attention-heads.")
    if num_transformer_layers != 1:
        raise ValueError("BST ranker requires --bst-num-transformer-layers=1.")
    if bst_additional_batch_negatives <= 0:
        raise ValueError("--bst-additional-batch-negatives must be positive.")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive.")
    if bst_max_train_batches_per_epoch is not None and int(bst_max_train_batches_per_epoch) <= 0:
        raise ValueError("--bst-max-train-batches-per-epoch must be positive when provided.")
    if bst_popularity_projection_dim <= 0:
        raise ValueError("--bst-popularity-projection-dim must be positive.")
    if author_embedding_dim <= 0:
        raise ValueError("--author-embedding-dim must be positive for the BST ranker.")
    if not 0.0 <= author_unknown_dropout_rate < 1.0:
        raise ValueError("--author-unknown-dropout-rate must be in [0, 1) for the BST ranker.")


def _validate_two_tower_config(args: argparse.Namespace) -> None:
    """Validate the fixed canonical cross-attention two-tower contract."""

    positive_dimensions = {
        "--output-embedding-dim": args.output_embedding_dim,
        "--user-hidden-dim": args.user_hidden_dim,
        "--post-hidden-dim": args.post_hidden_dim,
        "--max-history-len": args.max_history_len,
        "--author-embedding-dim": args.author_embedding_dim,
        "--content-projection-dim": args.content_projection_dim,
        "--author-projection-dim": args.author_projection_dim,
        "--batch-size": args.batch_size,
        "--eval-batch-size": args.eval_batch_size,
    }
    for flag, value in positive_dimensions.items():
        if int(value) <= 0:
            raise ValueError(f"{flag} must be positive for the two-tower model.")
    if not 0.0 <= float(args.dropout_rate_two_tower) < 1.0:
        raise ValueError("--dropout-rate-two-tower must be in [0, 1).")
    if float(args.similarity_temperature) <= 0.0:
        raise ValueError("--similarity-temperature must be positive.")
    if not 0.0 <= float(args.author_unknown_dropout_rate) < 1.0:
        raise ValueError(
            "--author-unknown-dropout-rate must be in [0, 1) for the two-tower model."
        )


def _get_stage_order_for_model_type(train_key: str) -> List[str]:
    # replace the generic 'train_placeholder' with the actual train stage key based on train_key:
    stage_order = copy.deepcopy(STAGE_ORDER)
    return [train_key if s == TRAIN_PLACEHOLDER else s for s in stage_order]


def _get_stage_folder(stage_order: List[str]) -> Dict[str, str]:
    stage_folder = {}
    for key in stage_order:
        _mp, _folder = reg.get_stage_spec(key)
        stage_folder[key] = _folder
    return stage_folder


def _get_stage_folder_and_start_stop_indices(
    stage_order: List[str],
    start_from: Optional[str],
    stop_after: Optional[str],
    train_key: str,
) -> Tuple[int, int, bool]:
    # Respect selective reruns (map the generic "train" alias to the concrete train stage key)
    if start_from == 'train':
        start_from = train_key
    if stop_after == 'train':
        stop_after = train_key
    if start_from and start_from not in stage_order:
        raise ValueError(f"Unrecognized start_from: {start_from}. Please choose from: {stage_order}")
    if stop_after and stop_after not in stage_order:
        raise ValueError(f"Unrecognized stop_after: {stop_after}. Please choose from: {stage_order}")
    start_idx = stage_order.index(start_from) if start_from in stage_order else 0
    stop_idx = stage_order.index(stop_after) if stop_after in stage_order else (len(stage_order) - 1)

    # Does this run include the training stage? Used for naming the run
    train_idx = stage_order.index(train_key)
    includes_train = False
    if start_idx <= train_idx <= stop_idx:
        includes_train = True

    return start_idx, stop_idx, includes_train


def cmd__run_all_exec(args: argparse.Namespace, ctx: Context) -> int:
    """Execute the modular pipeline stages in the foreground sequentially."""
    run_dir = Path(ctx.run_dir).resolve()
    artifacts_dir = Path(ctx.artifacts_dir).resolve()
    output_root = Path(args.output_dir).resolve()

    # Apply non-interactive prior pins (paths or stage_run_ids).
    prior_00_source_metadata = _resolve_prior_spec(
        args.prior_00_source_metadata,
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="00_source_metadata",
    )
    prior_01_query_selection = _resolve_prior_spec(
        args.prior_01_query_selection,
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="01_query_selection",
    )
    prior_02_user_history = _resolve_prior_spec(
        args.prior_02_user_history,
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="02_user_history",
    )
    prior_03_post_selection = _resolve_prior_spec(
        args.prior_03_post_selection,
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="03_post_selection",
    )
    prior_04_negative_selection = _resolve_prior_spec(
        args.prior_04_negative_selection,
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="04_negative_selection",
    )
    prior_05_post_liker_history = _resolve_prior_spec(
        args.prior_05_post_liker_history,
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="05_post_liker_history",
    )
    prior_06_author_statistics = _resolve_prior_spec(
        args.prior_06_author_statistics,
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="06_author_statistics",
    )
    prior_07_dataset_hydration = _resolve_prior_spec(
        args.prior_07_dataset_hydration,
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="07_dataset_hydration",
    )
    if prior_00_source_metadata is not None:
        ctx.prior_outputs["00_source_metadata"] = prior_00_source_metadata
    if prior_01_query_selection is not None:
        ctx.prior_outputs["01_query_selection"] = prior_01_query_selection
    if prior_02_user_history is not None:
        ctx.prior_outputs["02_user_history"] = prior_02_user_history
    if prior_03_post_selection is not None:
        ctx.prior_outputs["03_post_selection"] = prior_03_post_selection
    if prior_04_negative_selection is not None:
        ctx.prior_outputs["04_negative_selection"] = prior_04_negative_selection
    if prior_05_post_liker_history is not None:
        ctx.prior_outputs["05_post_liker_history"] = prior_05_post_liker_history
    if prior_06_author_statistics is not None:
        ctx.prior_outputs["06_author_statistics"] = prior_06_author_statistics
    if prior_07_dataset_hydration is not None:
        ctx.prior_outputs["07_dataset_hydration"] = prior_07_dataset_hydration
    validate_explicit_prior_pin_consistency(ctx)
    
    model_type = args.model_type

    if model_type == "bst-ranker":
        _validate_bst_config(args)
    elif model_type == "two-tower":
        _validate_two_tower_config(args)

    # Override train stage key if --model-type is specified
    train_key = _get_train_key(model_type)
    stage_order = _get_stage_order_for_model_type(train_key)
    stage_folder = _get_stage_folder(stage_order)
    start_idx, stop_idx, _ = _get_stage_folder_and_start_stop_indices(
        stage_order,
        args.start_from,
        args.stop_after,
        train_key
    )

    # Optional interactive chooser (foreground only)
    def _maybe_choose_prior(stage_key: str):
        if not args.pick_prior:
            return
        folder = stage_folder[stage_key]
        base = (artifacts_dir / folder)
        if not base.exists():
            return
        subdirs = [p for p in base.iterdir() if p.is_dir()]
        if len(subdirs) <= 1:
            return
        # Prompt only in foreground mode
        if bool(args.background):
            return
        subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"\nPick prior output for stage '{stage_key}' under {base}:")
        for i, p in enumerate(subdirs):
            print(f"  [{i}] {p.name}")
        try:
            choice = input("Enter index (blank for latest): ").strip()
            if choice:
                idx = int(choice)
                if 0 <= idx < len(subdirs):
                    ctx.prior_outputs[folder] = subdirs[idx]
        except Exception:
            pass

    # Execute selected subset
    try:
        for idx, key in enumerate(stage_order):
            if idx < start_idx or idx > stop_idx:
                continue
            # Before running, offer prior selection for this stage's dependency (if any)
            if idx > 0:
                prev_key = stage_order[idx - 1]
                if stage_folder[prev_key] not in ctx.prior_outputs:
                    _maybe_choose_prior(prev_key)
            label_map = {
                'source_metadata': "Stage 00: Build reusable source metadata…",
                'query_selection': "Stage 1: Select user-hour queries…",
                'user_history': "Stage 2: Generate user history…",
                'post_selection': "Stage 3: Select post universe…",
                'negative_selection': "Stage 4: Select hourly negative candidates…",
                'post_liker_history': "Stage 5: Extract post-liker histories…",
                'author_statistics': "Stage 6: Build author statistics…",
                'dataset_hydration': "Stage 7: Hydrate the model-training dataset…",
                'train_two_tower': "Stage 8: Train two-tower model…",
                'train_bst_ranker': "Stage 8: Train BST ranker…",
            }
            label = label_map.get(key, f"Stage {idx+1}: {key}…")
            print(f"\n[{idx+1}/{len(stage_order)}] ▶️  {label}")
            pin_lineage_aligned_inputs(ctx, key, stage_folder)
            reg.run_stage(key, ctx, args)
    finally:
        ctx.tracker.close()

    print("\n✅ run-all completed successfully")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Engagement Prediction Pipeline CLI",
        argument_default=argparse.SUPPRESS,
    )
    # Backwards compatible vestige: `run-all` used to be a subcommand; now it's implicit.
    parser.add_argument(
        "command",
        nargs="?",
        default="run-all",
        choices=["run-all", "compare-rankers"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="YAML/JSON config file with run-all parameters (CLI flags override config)",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=argparse.SUPPRESS,
        help="compare-rankers model spec in name:type:path format; repeat for multiple models",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=argparse.SUPPRESS,
        help=f"compare-rankers splits to evaluate (default: {' '.join(compare_rankers.DEFAULT_COMPARE_SPLITS)})",
    )
    parser.add_argument(
        "--bst-candidate-chunk-size",
        type=int,
        default=argparse.SUPPRESS,
        help=f"compare-rankers BST candidate chunk size (default: {compare_rankers.DEFAULT_COMPARE_BST_CANDIDATE_CHUNK_SIZE})",
    )
    # run-all (modular pipeline)
    p_all = parser
    # Data-source options
    _add_arg_with_default(p_all, "--gcs-bucket", type=str, default=argparse.SUPPRESS,
                          help_text="GCS bucket name for ingex data")
    _add_arg_with_default(p_all, "--posts-start", type=str, default=argparse.SUPPRESS,
                          help_text="UTC start of the common Ingex posts, replies, and likes window (inclusive)")
    _add_arg_with_default(p_all, "--posts-end", type=str, default=argparse.SUPPRESS,
                          help_text="UTC end of the common Ingex posts, replies, and likes window (exclusive)")
    _add_arg_with_default(p_all, "--source-metadata-partition-count", type=int,
                          default=argparse.SUPPRESS,
                          help_text="Stable URI-hash partition count owned by the Stage 00 metadata index")
    _add_arg_with_default(p_all, "--unseen-user-fraction", type=float, default=argparse.SUPPRESS,
                          help_text="Stable fraction of users reserved for unseen-user evaluation")
    _add_arg_with_default(p_all, "--max-hours-per-user-per-split", type=int, default=argparse.SUPPRESS,
                          help_text="Maximum query-hours retained per user within each split")
    _add_arg_with_default(p_all, "--max-train-query-hours", type=int, default=argparse.SUPPRESS,
                          help_text="Optional cap on selected training query-hours")
    _add_arg_with_default(p_all, "--max-eval-query-hours-per-split", type=int, default=argparse.SUPPRESS,
                          help_text="Optional independent query-hour cap for each evaluation split")
    _add_arg_with_default(p_all, "--max-positives-per-user-hour", type=int, default=argparse.SUPPRESS,
                          help_text="Discard selected user-hours with more than this many retained root-post positives")
    _add_arg_with_default(p_all, "--output-dir", type=str, default=argparse.SUPPRESS,
                          help_text="Optional explicit run directory root")
    _add_arg_with_default(p_all, "--random-seed", type=int, default=argparse.SUPPRESS,
                          help_text="Random seed for splitting")
    _add_arg_with_default(p_all, "--embedding-model", type=str, choices=["all_MiniLM_L6_v2", "all_MiniLM_L12_v2"],
                          default=argparse.SUPPRESS, help_text="SentenceTransformers model for embeddings")
    # Stage 1 split / Stage 2 options
    _add_arg_with_default(p_all, "--max-history-posts-per-query", type=int, default=argparse.SUPPRESS,
                          help_text="Maximum recent like events retained in each Stage 2 query history")
    _add_arg_with_default(p_all, "--user-history-partition-count", type=int, default=argparse.SUPPRESS,
                          help_text="Stable DID-hash partition count used to bound Stage 2 memory")
    # Stage 3 post selection
    _add_arg_with_default(p_all, "--random-candidate-sampling-fraction", type=float,
                          default=argparse.SUPPRESS,
                          help_text="Stable fraction of unique posts retained in the random candidate reservoir")
    # Stage 4 popularity-aware negative selection
    _add_arg_with_default(p_all, "--negative-candidates-per-hour", type=int,
                          default=argparse.SUPPRESS,
                          help_text="Target shared negative-candidate count for each selected query hour")
    _add_arg_with_default(p_all, "--min-likes-for-popular-candidate", type=int,
                          default=argparse.SUPPRESS,
                          help_text="Minimum strictly prior like count for the popular candidate method")
    _add_arg_with_default(p_all, "--popular-candidate-fraction", type=float,
                          default=argparse.SUPPRESS,
                          help_text="Desired fraction of each hourly pool selected by the popular method")
    _add_arg_with_default(p_all, "--max-candidate-age-hours", type=int,
                          default=argparse.SUPPRESS,
                          help_text="Number of creation-hour buckets in which a post remains candidate-eligible")
    # Stage 5 post-liker history extraction
    _add_arg_with_default(p_all, "--post-liker-history-partition-count", type=int,
                          default=argparse.SUPPRESS,
                          help_text="Stable URI-hash partition count used to bound Stage 5 liker-event processing")
    # Stage 6 training-only author statistics
    _add_arg_with_default(p_all, "--author-statistics-partition-count", type=int,
                          default=argparse.SUPPRESS,
                          help_text="Stable hash partition count used to bound Stage 6 post and author aggregation")
    # Stage 7 dataset hydration
    _add_arg_with_default(p_all, "--embedding-source-batch-size", type=int,
                          default=argparse.SUPPRESS,
                          help_text="Raw post/reply files processed per Stage 7 embedding scan")
    _add_arg_with_default(p_all, "--embedding-partition-worker-count", type=int,
                          default=argparse.SUPPRESS,
                          help_text="Worker processes used for Stage 7 URI-partition embedding decoding")
    _add_arg_with_default(p_all, "--min-author-training-feature-count", type=int,
                          default=argparse.SUPPRESS,
                          help_text="Minimum final training-feature occurrences required for a dedicated author index")
    _add_arg_with_default(p_all, "--train-start", type=str, default=argparse.SUPPRESS,
                          help_text="UTC start of target eligibility and the training split")
    _add_arg_with_default(p_all, "--val-start", type=str, default=argparse.SUPPRESS,
                          help_text="ISO date string for start of validation dataset window. Must be >= train-start")
    _add_arg_with_default(p_all, "--holdout-start", type=str, default=argparse.SUPPRESS,
                          help_text="ISO date string for start of seen-users holdout window. Non-holdout users' rows at/after this date become holdout_seen_users. Must be after val-start.")
    _add_arg_with_default(p_all, "--holdout-end", type=str, default=argparse.SUPPRESS,
                          help_text="ISO date string for end of holdout window. Applies to both holdout_seen_users and holdout_unseen_users. Rows at/after this date get split=None. Default: no upper bound.")
    # Stage 8 model selection
    _add_arg_with_default(p_all, "--model-type", type=str, choices=["two-tower", "bst-ranker"],
                          default=argparse.SUPPRESS, help_text="Model architecture: two-tower or bst-ranker")
    # Two-tower specific options
    _add_arg_with_default(p_all, "--output-embedding-dim", type=int, default=argparse.SUPPRESS,
                          help_text="Canonical two-tower user/post output embedding dimension")
    _add_arg_with_default(p_all, "--user-hidden-dim", type=int, default=argparse.SUPPRESS,
                          help_text="User encoder hidden dimension")
    _add_arg_with_default(p_all, "--post-hidden-dim", type=int, default=argparse.SUPPRESS,
                          help_text="Two-tower post encoder hidden dimension")
    _add_arg_with_default(p_all, "--max-history-len", type=int, default=argparse.SUPPRESS,
                          help_text="Max user history length")
    _add_arg_with_default(p_all, "--similarity-temperature", type=float, default=argparse.SUPPRESS,
                          help_text="Temperature used to scale cosine-similarity logits in the two-tower model")
    _add_arg_with_default(p_all, "--author-embedding-dim", type=int, default=argparse.SUPPRESS,
                          help_text="Embedding dimension for the trainable author embedding table")
    _add_arg_with_default(p_all, "--author-unknown-dropout-rate", type=float, default=argparse.SUPPRESS,
                          help_text="Training-time probability of replacing a supported history author with the UNK row")
    # Ranker shared options
    _add_arg_with_default(p_all, "--content-projection-dim", type=int, default=argparse.SUPPRESS,
                          help_text="Ranker content branch projection dimension")
    _add_arg_with_default(p_all, "--author-projection-dim", type=int, default=argparse.SUPPRESS,
                          help_text="Ranker author branch projection dimension")
    _add_arg_with_default(p_all, "--prediction-hidden-dims", type=int, nargs="*", default=argparse.SUPPRESS,
                          help_text="Ranker prediction-head hidden dimensions. Use no values for a direct linear head")
    _add_arg_with_default(p_all, "--bst-additional-batch-negatives", type=int, default=argparse.SUPPRESS,
                          help_text="Additional same-hour negative-pool posts to sample per BST training batch")
    # BST ranker specific options
    _add_arg_with_default(p_all, "--bst-model-dim", type=int, default=argparse.SUPPRESS,
                          help_text="BST ranker fused post/author model dimension")
    _add_arg_with_default(p_all, "--bst-time-embedding-dim", type=int, default=argparse.SUPPRESS,
                          help_text="BST ranker time-delta embedding dimension")
    _add_arg_with_default(p_all, "--bst-num-attention-heads", type=int, default=argparse.SUPPRESS,
                          help_text="BST ranker transformer attention heads")
    _add_arg_with_default(p_all, "--bst-num-transformer-layers", type=int, default=argparse.SUPPRESS,
                          help_text="BST ranker transformer encoder layers")
    _add_arg_with_default(p_all, "--bst-transformer-ff-dim", type=int, default=argparse.SUPPRESS,
                          help_text="BST ranker transformer feed-forward dimension")
    _add_arg_with_default(p_all, "--bst-dropout-rate", type=float, default=argparse.SUPPRESS,
                          help_text="BST ranker dropout rate")
    _add_arg_with_default(p_all, "--bst-norm-first", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
                          help_text="Enable pre-norm transformer layers for the BST ranker")
    _add_arg_with_default(p_all, "--bst-time-delta-bucket-boundaries-hours", type=float, nargs="+",
                          default=argparse.SUPPRESS,
                          help_text="BST ranker time-delta bucket boundaries in hours")
    _add_arg_with_default(p_all, "--bst-weight-decay", type=float, default=argparse.SUPPRESS,
                          help_text="Weight decay for BST ranker model")
    _add_arg_with_default(p_all, "--bst-max-train-batches-per-epoch", type=int, default=argparse.SUPPRESS,
                          help_text="Optional cap on BST train batches per epoch for fast experiments")
    _add_arg_with_default(p_all, "--bst-use-popularity-feature", action=argparse.BooleanOptionalAction,
                          default=argparse.SUPPRESS,
                          help_text="Enable or disable BST prior-cumulative-like popularity features")
    _add_arg_with_default(p_all, "--bst-popularity-projection-dim", type=int, default=argparse.SUPPRESS,
                          help_text="BST popularity feature projection dimension")
    # Stage 8 options (shared)
    _add_arg_with_default(p_all, "--epochs", type=int, default=argparse.SUPPRESS,
                          help_text="Training epochs")
    _add_arg_with_default(p_all, "--batch-size", type=int, default=argparse.SUPPRESS,
                          help_text="Training batch size")
    _add_arg_with_default(p_all, "--learning-rate", type=float, default=argparse.SUPPRESS,
                          help_text="Learning rate")
    _add_arg_with_default(p_all, "--weight-decay-two-tower", type=float, default=argparse.SUPPRESS,
                          help_text="Weight decay for two tower model")
    _add_arg_with_default(p_all, "--dropout-rate-two-tower", type=float, default=argparse.SUPPRESS,
                          help_text="Dropout rate for two tower model")
    _add_arg_with_default(p_all, "--device", type=str, choices=["cpu", "cuda"], default=argparse.SUPPRESS,
                          help_text="Device for training")
    _add_arg_with_default(p_all, "--patience", type=int, default=argparse.SUPPRESS,
                          help_text="Early stopping patience")
    _add_arg_with_default(p_all, "--early-stopping-min-delta", type=float, default=argparse.SUPPRESS,
                          help_text="Minimum absolute validation primary-metric improvement required to reset early stopping patience")
    _add_arg_with_default(p_all, "--run-tag", type=str, default=argparse.SUPPRESS,
                          help_text="Tag appended to the training output directory name")
    _add_arg_with_default(p_all, "--no-plots", action="store_true", default=argparse.SUPPRESS,
                          help_text="Disable training plots")
    _add_arg_with_default(p_all, "--disable-progress", action="store_true", default=argparse.SUPPRESS,
                          help_text="Disable progress bars during training")
    _add_arg_with_default(p_all, "--metrics-top-ks", type=int, nargs="+", default=argparse.SUPPRESS,
                          help_text="Values of K to use for training NDCG@K metrics")
    # Stage 8 - DataLoader settings
    _add_arg_with_default(p_all, "--num-dataloader-workers", type=int, default=argparse.SUPPRESS,
                          help_text="Number of DataLoader worker processes")
    _add_arg_with_default(p_all, "--dataloader-pin-memory", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
                          help_text="Enable DataLoader pin_memory for faster GPU transfer")
    _add_arg_with_default(p_all, "--dataloader-persistent-workers", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
                          help_text="Keep DataLoader workers alive between epochs")
    _add_arg_with_default(p_all, "--dataloader-prefetch-factor", type=int, default=argparse.SUPPRESS,
                          help_text="Number of batches to prefetch per DataLoader worker")
    # Stage 8 - Learning rate scheduler
    _add_arg_with_default(p_all, "--lr-scheduler-factor", type=float, default=argparse.SUPPRESS,
                          help_text="Factor by which to reduce learning rate")
    _add_arg_with_default(p_all, "--lr-scheduler-patience", type=int, default=argparse.SUPPRESS,
                          help_text="Number of epochs with no improvement before reducing LR")
    # Stage 8 - Training optimization
    _add_arg_with_default(p_all, "--gradient-clip-max-norm", type=float, default=argparse.SUPPRESS,
                          help_text="Maximum gradient norm for clipping")
    # Validation
    _add_arg_with_default(p_all, "--eval-batch-size", type=int, default=argparse.SUPPRESS,
                          help_text="Batch size for evaluation")
    # Selection behavior
    _add_arg_with_default(p_all, "--use-latest", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
                          help_text="(Deprecated) Always enabled during sequential run-all")
    # Selective reruns and prior pinning
    _add_arg_with_default(p_all, "--start-from", type=str,
                          choices=["source_metadata", "query_selection", "user_history", "post_selection", "negative_selection", "post_liker_history", "author_statistics", "dataset_hydration", "train", "train_two_tower", "train_bst_ranker"],
                          default=argparse.SUPPRESS, help_text="Begin execution at this stage")
    _add_arg_with_default(p_all, "--stop-after", type=str,
                          choices=["source_metadata", "query_selection", "user_history", "post_selection", "negative_selection", "post_liker_history", "author_statistics", "dataset_hydration", "train", "train_two_tower", "train_bst_ranker"],
                          default=argparse.SUPPRESS, help_text="Stop after this stage completes")
    _add_arg_with_default(p_all, "--pick-prior", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
                          help_text="If multiple prior outputs exist, prompt to pick (foreground only)")
    _add_arg_with_default(p_all, "--prior-00-source-metadata", type=str, default=argparse.SUPPRESS,
                          help_text="Pin prior Stage 00 (00_source_metadata) artifact dir by stage_run_id or path")
    _add_arg_with_default(p_all, "--prior-01-query-selection", type=str, default=argparse.SUPPRESS,
                          help_text="Pin prior Stage 1 (01_query_selection) artifact dir by stage_run_id or path")
    _add_arg_with_default(p_all, "--prior-02-user-history", type=str, default=argparse.SUPPRESS,
                          help_text="Pin Stage 2 (02_user_history) for a direct post-selection rerun")
    _add_arg_with_default(p_all, "--prior-03-post-selection", type=str, default=argparse.SUPPRESS,
                          help_text="Pin Stage 3 (03_post_selection) for a direct negative-selection rerun")
    _add_arg_with_default(p_all, "--prior-04-negative-selection", type=str, default=argparse.SUPPRESS,
                          help_text="Pin Stage 4 (04_negative_selection) for a direct post-liker-history rerun")
    _add_arg_with_default(p_all, "--prior-05-post-liker-history", type=str, default=argparse.SUPPRESS,
                          help_text="Pin Stage 5 (05_post_liker_history) for a direct author-statistics rerun")
    _add_arg_with_default(p_all, "--prior-06-author-statistics", type=str, default=argparse.SUPPRESS,
                          help_text="Pin Stage 6 (06_author_statistics) for a direct dataset-hydration rerun")
    _add_arg_with_default(p_all, "--prior-07-dataset-hydration", type=str, default=argparse.SUPPRESS,
                          help_text="Pin Stage 7 (07_dataset_hydration) for a direct native-training rerun")
    # Execution behavior
    _add_arg_with_default(p_all, "--background", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
                          help_text="Run in background with nohup (default: foreground)")
    p_all.add_argument("--_initial-log", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    # Experiment tracking
    _add_arg_with_default(p_all, "--experiment-tracker", type=str, choices=["none", "clearml"], default=argparse.SUPPRESS,
                          help_text="Type of experiment tracker to use")
    _add_arg_with_default(p_all, "--experiment-project", type=str, default=argparse.SUPPRESS,
                          help_text="Experiment tracking project name")
    _add_arg_with_default(p_all, "--experiment-task", type=str, default=argparse.SUPPRESS,
                          help_text="Experiment tracking task name")
    _add_arg_with_default(p_all, "--experiment-tags", type=str, nargs="*", default=argparse.SUPPRESS,
                          help_text="Optional tags for the experiment tracker")
    _add_arg_with_default(p_all, "--model-output-uri", type=str, default=argparse.SUPPRESS,
                          help_text="Model/task output URI for ClearML (e.g. gs://bucket/path)")
    p_all.set_defaults(func=cmd_run_all)

    return parser


def main() -> int:
    parser = build_parser()
    raw_args = parser.parse_args()
    if raw_args.command == "compare-rankers":
        return cmd_compare_rankers(raw_args)
    merged_args = _merge_args_with_config(raw_args)
    return merged_args.func(merged_args)


if __name__ == "__main__":
    sys.exit(main()) 
