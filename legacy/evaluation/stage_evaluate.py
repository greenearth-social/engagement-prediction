#!/usr/bin/env python3

"""
Archived evaluator for legacy Stage 3 ranking-row artifacts.

This stage orchestrates the evaluation pipeline by:
1. Loading holdout ranking rows from Stage 3 (03_train)
2. Computing user metadata from those rows
3. Creating an EvalContext and running all discovered evaluation modules

Evaluation modules are auto-discovered from legacy/evaluation/evals/ and each
produces its own set of artifacts (plots, CSVs, JSON summaries).

Inputs (from prior pipeline stages):
- eval/holdout_<type>_ranking_rows.parquet from 03_train

Outputs under artifacts/04_evaluate/<stage_run_id>/
- eval_summary.json: Combined results from all modules
- stage_info.txt: Stage metadata
- <module_name>/: Subdirectory for each evaluation module's artifacts
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from engagement_prediction.pipeline.core import Context, generate_run_timestamp
from engagement_prediction.pipeline.logging import get_stage_logger, log_operation_start
from legacy.evaluation.evals import EvalContext, run_all_modules

STAGE_LOG_NAME = 'STAGE_04_EVALUATE'


# ---------------------------------------------------------------------------
# Asset resolution
# ---------------------------------------------------------------------------

def resolve_train_output(
    context: Context,
    *,
    prior_03_train: Optional[str | Path],
) -> Path:
    """Resolve the explicitly supplied legacy ``03_train`` artifact.

    This archived evaluator is deliberately detached from active pipeline
    discovery. It never searches for the latest training run and never consumes
    an artifact merely because it was produced earlier in the current process.
    """
    if prior_03_train is None or not str(prior_03_train).strip():
        raise ValueError(
            "Archived evaluation requires an explicit prior_03_train artifact path"
        )
    return context.resolve_prior_output(
        "03_train",
        prior_path=Path(prior_03_train).expanduser(),
    )


# ---------------------------------------------------------------------------
# Holdout ranking rows
# ---------------------------------------------------------------------------

def load_holdout_ranking_rows(
    eval_dir: Optional[Path],
    holdout_type: str,
    logger=None,
) -> Optional[pd.DataFrame]:
    if eval_dir is None:
        return None
    ranking_path = eval_dir / f'holdout_{holdout_type}_ranking_rows.parquet'
    if not ranking_path.exists():
        return None
    if logger:
        logger.info(f"Loading ranking rows from {ranking_path}")
    return pd.read_parquet(ranking_path)


def compute_user_metadata_from_ranking_rows(ranking_rows_df: pd.DataFrame) -> pd.DataFrame:
    metadata_df = (
        ranking_rows_df
        .groupby('did', as_index=False)
        .agg(
            num_embedding_likes=('num_embedding_likes', 'max'),
            num_total_likes=('num_total_likes', 'max'),
        )
    )
    metadata_df['did'] = metadata_df['did'].astype(str)
    metadata_df['num_embedding_likes'] = metadata_df['num_embedding_likes'].fillna(0).astype(int)
    metadata_df['num_total_likes'] = metadata_df['num_total_likes'].fillna(0).astype(int)
    return metadata_df[['did', 'num_embedding_likes', 'num_total_likes']]


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run(context: Context, args) -> Dict[str, Any]:
    """
    Main entry point for Stage 4: Evaluation.

    Loads holdout ranking rows from the training stage and runs all evaluation
    modules.
    """
    t0 = time.time()

    # --- hyperparams ---
    eval_batch_size = int(args.eval_batch_size)
    eval_holdout_type = str(args.eval_holdout_type)
    skip_modules = args.skip_modules
    if skip_modules and isinstance(skip_modules, str):
        skip_modules = [m.strip() for m in skip_modules.split(',')]

    # Resolve training output (inputs)
    train_dir = resolve_train_output(
        context,
        prior_03_train=getattr(args, "prior_03_train", None),
    )
    train_eval_dir = train_dir / 'eval'

    # Canonical stage output
    out_dir = context.new_stage_dir("04_evaluate", tag=eval_holdout_type)

    # Initialize logger
    logger = get_stage_logger(STAGE_LOG_NAME, log_file=out_dir / 'stage.log')
    log_operation_start('Stage 4: Evaluation', STAGE_LOG_NAME, logger)
    logger.info(f"Training output dir: {train_dir}")
    logger.info(f"Holdout type for evaluation: {eval_holdout_type}")

    log_operation_start('Load evaluation artifact', STAGE_LOG_NAME, logger)
    ranking_rows_df = load_holdout_ranking_rows(
        eval_dir=train_eval_dir if train_eval_dir.exists() else None,
        holdout_type=eval_holdout_type,
        logger=logger,
    )
    if ranking_rows_df is None:
        raise FileNotFoundError(
            f"No holdout ranking rows found. Expected {train_eval_dir / f'holdout_{eval_holdout_type}_ranking_rows.parquet'}. "
            "Please rerun Stage 3 training so it writes matrix ranking-row artifacts."
        )
    predictions_df = pd.DataFrame(columns=['did', 'post_id', 'y_true', 'y_pred_proba'])
    user_metadata_df = compute_user_metadata_from_ranking_rows(ranking_rows_df)
    embed_dim = None
    logger.info(f"Loaded {len(ranking_rows_df)} ranking rows for {ranking_rows_df['did'].nunique()} users")

    # Step 4: Create EvalContext
    timestamp = generate_run_timestamp()

    eval_config: Dict[str, Any] = {
        'batch_size': eval_batch_size,
        'embed_dim': embed_dim,
        'eval_mode': 'ranking_rows',
    }

    ctx = EvalContext(
        predictions_df=predictions_df,
        user_metadata_df=user_metadata_df,
        output_dir=out_dir,
        timestamp=timestamp,
        config=eval_config,
        ranking_rows_df=ranking_rows_df,
    )

    # Step 5: Discover and run evaluation modules
    log_operation_start('Discover and run evaluation modules', STAGE_LOG_NAME, logger)
    logger.info("Running evaluation modules...")
    module_results = run_all_modules(ctx, skip_modules=skip_modules)

    # Step 6: Save combined summary
    log_operation_start('Save evaluation summary', STAGE_LOG_NAME, logger)

    eval_summary = {
        'timestamp': timestamp,
        'runtime_seconds': time.time() - t0,
        'num_holdout_users': ctx.num_holdout_users,
        'num_predictions': ctx.num_predictions,
        'num_ranking_rows': ctx.num_ranking_rows,
        'train_dir': str(train_dir),
        'embed_dim': embed_dim,
        'eval_mode': eval_config['eval_mode'],
        'modules': module_results,
    }

    summary_path = out_dir / 'eval_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(eval_summary, f, indent=2, default=str)

    info_lines = [
        "stage: evaluate",
        f"runtime_seconds: {time.time() - t0:.2f}",
        f"timestamp: {timestamp}",
        f"num_holdout_users: {ctx.num_holdout_users}",
        f"num_predictions: {ctx.num_predictions}",
        f"num_ranking_rows: {ctx.num_ranking_rows}",
        f"modules_run: {', '.join(module_results.keys())}",
        "inputs: ranking rows",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    logger.info(f"Evaluation complete. Output: {out_dir}")

    return {
        'output_dir': out_dir,
        'artifacts': eval_summary,
    }
