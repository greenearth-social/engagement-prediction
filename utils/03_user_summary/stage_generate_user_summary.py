#!/usr/bin/env python3

"""
Stage 3: 

Inputs:


Outputs under <run_dir>/featurize/<timestamp>/:

"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import argparse
import polars as pl
import time
import json

from utils.pipeline.core import new_stage_timestamp_dir, select_prior_output, Context
from utils.helpers import get_stage_logger, log_operation_start, validate_dataframe_schema, load_parquet_from_prior, get_embed_cols_from_lf, calc_time_weighted_exp_mov_avg

# #region agent log
_DEBUG_LOG_PATH = "/mnt/data/wm.s.schulz/modules/engagement-prediction/.cursor/debug.log"
def _debug_log(hypothesis_id: str, location: str, message: str, data: dict):
    import time as _t
    entry = {"hypothesisId": hypothesis_id, "location": location, "message": message, "data": data, "timestamp": int(_t.time()*1000), "sessionId": "debug-session"}
    with open(_DEBUG_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
# #endregion


def _generate_user_summary_from_history_batched(
    posts_core_lf: pl.LazyFrame,
    user_history_lf: pl.LazyFrame, 
    tau_hours: int,
    logger,
    batch_size: int = 10000,
) -> pl.DataFrame:
    """
    Memory-efficient batched processing of user summaries.
    Processes users in batches to avoid OOM on large datasets.
    """
    embedding_cols = get_embed_cols_from_lf(posts_core_lf)
    _debug_log("H3", "stage_generate_user_summary.py:_generate", "embedding_cols count", {"num_embed_cols": len(embedding_cols), "first_few": embedding_cols[:5] if embedding_cols else []})
    
    # Get unique users
    log_operation_start('Getting unique users for batched processing', 'STAGE_03_USER_SUMMARY', logger)
    unique_users = user_history_lf.select("did").unique().collect().to_series().to_list()
    num_users = len(unique_users)
    _debug_log("BATCH1", "stage_generate_user_summary.py:_generate", "unique users", {"num_users": num_users, "batch_size": batch_size})
    
    # Pre-collect posts_core with only needed columns for efficiency
    log_operation_start('Collecting posts_core subset', 'STAGE_03_USER_SUMMARY', logger)
    posts_core_lf_cols = ["at_uri", "record_created_at"] + embedding_cols
    posts_core_df = posts_core_lf.select(posts_core_lf_cols).collect()
    _debug_log("BATCH2", "stage_generate_user_summary.py:_generate", "posts_core collected", {"rows": len(posts_core_df)})
    
    # Process in batches
    results = []
    num_batches = (num_users + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_users)
        batch_users = unique_users[start_idx:end_idx]
        
        log_operation_start(f'Processing user batch {batch_idx + 1}/{num_batches} ({len(batch_users)} users)', 'STAGE_03_USER_SUMMARY', logger)
        
        # Filter user_history for this batch of users
        batch_history_df = (
            user_history_lf
            .filter(pl.col("did").is_in(batch_users))
            .collect()
        )
        
        if len(batch_history_df) == 0:
            continue
            
        # Join with posts_core
        joined_df = batch_history_df.join(
            posts_core_df,
            left_on="subject_uri",
            right_on="at_uri",
            how="inner",
        )
        
        if len(joined_df) == 0:
            continue
        
        # Compute time-weighted EMA for this batch
        batch_result = calc_time_weighted_exp_mov_avg(
            joined_df.lazy(),
            value_cols=embedding_cols,
            group_cols=["did"],
            time_col="record_created_at",
            tau_hours=tau_hours,
        ).collect()
        
        results.append(batch_result)
        
        # Log progress
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            _debug_log("BATCH_PROGRESS", "stage_generate_user_summary.py:_generate", "batch progress", 
                      {"completed": batch_idx + 1, "total": num_batches, "users_processed": end_idx})
    
    # Concatenate all batch results
    log_operation_start('Concatenating batch results', 'STAGE_03_USER_SUMMARY', logger)
    if not results:
        raise ValueError("No user summaries were generated - check if user_history and posts_core have matching records")
    
    final_result = pl.concat(results)
    _debug_log("BATCH_DONE", "stage_generate_user_summary.py:_generate", "batched processing complete", 
              {"total_users_in_result": len(final_result)})
    
    return final_result


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '03_user_summary')

    # Initialize logger
    logger = get_stage_logger('STAGE_03_USER_SUMMARY', log_file=out_dir / 'stage.log')

    # get args
    tau_hours = args.tau_hours
    t0 = time.time()

    log_operation_start('Load raw data from prior stage', 'STAGE_03_USER_SUMMARY', logger)
    posts_core_path = select_prior_output(run_dir, '01_get_data', use_latest=context.use_latest, prior_path=context.prior_outputs.get('01_get_data'))
    if posts_core_path is None:
        raise FileNotFoundError(f"Could not find posts_core_*.parquet in 01_get_data")
    posts_core_lf: pl.LazyFrame = load_parquet_from_prior(posts_core_path, "posts_core_")
    # #region agent log
    _debug_log("H1", "stage_generate_user_summary.py:run", "posts_core columns", {"columns": posts_core_lf.collect_schema().names()[:20], "has_subject_uri": "subject_uri" in posts_core_lf.collect_schema().names(), "has_at_uri": "at_uri" in posts_core_lf.collect_schema().names()})
    # #endregion
    
    user_history_path = select_prior_output(run_dir, '02_featurize', use_latest=context.use_latest, prior_path=context.prior_outputs.get('02_featurize'))
    if user_history_path is None:
        raise FileNotFoundError(f"Could not find user_history_*.parquet in 02_featurize")
    user_history_lf: pl.LazyFrame = load_parquet_from_prior(user_history_path, "user_history_")
    # #region agent log
    _debug_log("H2", "stage_generate_user_summary.py:run", "user_history columns", {"columns": user_history_lf.collect_schema().names(), "has_subject_uri": "subject_uri" in user_history_lf.collect_schema().names()})
    # #endregion
    validate_dataframe_schema(user_history_lf, {"did": str, "subject_uri": str, "record_created_at_bucket": pl.Datetime})

    log_operation_start('Generate user summary from history (batched)', 'STAGE_03_USER_SUMMARY', logger)
    user_summary_df: pl.DataFrame = _generate_user_summary_from_history_batched(
        posts_core_lf, user_history_lf, tau_hours, logger, batch_size=10000
    )

    # Write out result
    user_summary_output_path = out_dir / f"user_summary_{out_dir.name}.parquet"
    log_operation_start('Writing user_summary to parquet', 'STAGE_03_USER_SUMMARY', logger)
    _debug_log("MEM5", "stage_generate_user_summary.py:run", "writing parquet", {"output_path": str(user_summary_output_path), "rows": len(user_summary_df)})
    user_summary_df.write_parquet(user_summary_output_path)
    _debug_log("MEM6", "stage_generate_user_summary.py:run", "write_parquet completed", {"status": "ok"})

    # Stage info
    info_lines = [
        f"stage: user_summary",
        f"runtime_seconds: {time.time()-t0:.2f}",
        f"settings: ",
        f"inputs: posts_core, user_history",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    return {
        'output_dir': out_dir,
        'artifacts': {
            'user_summary_path': str(user_summary_output_path),
        }
    }
