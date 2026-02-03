#!/usr/bin/env python3

"""
Stage 2: 

Inputs:


Outputs under <run_dir>/target_posts/<timestamp>/:

"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import argparse
import polars as pl
import time

from utils.pipeline.core import new_stage_timestamp_dir, select_prior_output, Context
from utils.helpers import get_stage_logger, log_operation_start, validate_dataframe_schema, load_parquet_from_prior

STAGE_NAME_FOR_LOGGING = '02_TARGET_POSTS'


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '02_target_posts')

    # Initialize logger
    logger = get_stage_logger(STAGE_NAME_FOR_LOGGING, log_file=out_dir / 'stage.log')

    # get args
    t0 = time.time()

    prior_stage_path = select_prior_output(run_dir, '01_get_data', use_latest=context.use_latest, prior_path=context.prior_outputs.get('01_get_data'))
    if prior_stage_path is None:
        raise FileNotFoundError(f"Could not find directory {prior_stage_path}")
    
    log_operation_start('Load raw posts data from prior stage', STAGE_NAME_FOR_LOGGING, logger)
    posts_core_lf: pl.LazyFrame = load_parquet_from_prior(prior_stage_path, "posts_core_")
    validate_dataframe_schema(posts_core_lf, {})
    
    log_operation_start('Load raw likes data from prior stage', STAGE_NAME_FOR_LOGGING, logger)
    likes_core_lf: pl.LazyFrame = load_parquet_from_prior(prior_stage_path, "likes_core_")
    validate_dataframe_schema(likes_core_lf, {})

    log_operation_start('', STAGE_NAME_FOR_LOGGING, logger)
    target_posts_lf: pl.LazyFrame = _get_target_posts(posts_core_lf, likes_core_lf, )
    validate_dataframe_schema(target_posts_lf, {})

    # Write out result
    target_posts_output_path = out_dir / f"target_posts_{out_dir.name}.parquet"
    target_posts_lf.sink_parquet(target_posts_output_path)

    # Stage info
    info_lines = [
        f"stage: target_posts",
        f"runtime_seconds: {time.time()-t0:.2f}",
        f"settings: ",
        f"inputs: posts_core, likes_core",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    return {
        'output_dir': out_dir,
        'artifacts': {
            'user_summary_path': str(target_posts_output_path),
        }
    }
