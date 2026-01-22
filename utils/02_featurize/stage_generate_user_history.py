#!/usr/bin/env python3

"""
Stage 2: 

Inputs:


Outputs under <run_dir>/featurize/<timestamp>/:

"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import polars as pl
import time

from utils.pipeline.core import new_stage_timestamp_dir, select_prior_output, Context
from utils.helpers import get_stage_logger, log_operation_start, validate_dataframe_schema


def _load_likes_core_lf_from_prior(prior_path: Path) -> pl.LazyFrame:
    # Load the most recent likes_core_*.parquet found in the given directory
    # candidates = sorted(prior_path.glob('likes_core_*.parquet'), key=lambda p: p.stat().st_mtime, reverse=True)
    # if not candidates:
    #     raise FileNotFoundError(f"No likes_core_*.parquet found under {prior_path}")
    # return pl.scan_parquet(candidates[0])
    return pl.scan_parquet(prior_path)


# TODO: Add an "end_window_lookback"?
# e.g. if bucket is at t, only include likes up to bucket t-k (instead of up to t or t-1)
def _generate_user_history_from_likes(
    likes_core_lf: pl.LazyFrame, 
    bucket_duration: str,
    num_buckets_lookback: int,
    max_likes_per_bucket: Optional[int],
) -> pl.LazyFrame:
    return likes_core_lf.with_columns(
        (pl.col("inserted_at").dt.truncate("1h") + pl.duration(hours=1)).alias("inserted_at_ceil")
    ).with_columns(
        pl.int_ranges(0, num_buckets_lookback).alias("bucket_offset")
    ).explode("bucket_offset").with_columns(
        (pl.col("inserted_at_ceil") + pl.duration(hours=pl.col("bucket_offset"))).alias("inserted_at_bucket")
    ).drop("bucket_offset", "inserted_at_ceil")


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '02_featurize')

    # Initialize logger
    logger = get_stage_logger('STAGE_02_FEATURIZE', log_file=out_dir / 'stage.log')

    # Try to use prior get_data output when available
    # prior_path = select_prior_output(run_dir, '01_get_data', use_latest=context.use_latest, prior_path=context.prior_outputs.get('01_get_data'))
    prior_path = Path("/mnt/data/dave/outputs/01_get_data/20260120_210159/dummy_likes_core.parquet")
    if prior_path is None:
        raise FileNotFoundError(f"Could not find raw data in 01_get_data")

    # get args
    bucket_duration = str(args.bucket_duration)
    num_buckets_lookback = int(args.num_buckets_lookback)
    max_likes_per_bucket = int(args.max_likes_per_bucket)

    log_operation_start('Load raw data from prior stage', 'STAGE_02_FEATURIZE', logger)
    t0 = time.time()
    likes_core_lf: pl.LazyFrame = _load_likes_core_lf_from_prior(prior_path)
    validate_dataframe_schema(likes_core_lf, {"did": str, "inserted_at": pl.Datetime, "subject_uri": str})

    log_operation_start('Aggregate likes into user history store', 'STAGE_02_FEATURIZE', logger)
    user_history_lf: pl.LazyFrame = _generate_user_history_from_likes(likes_core_lf, bucket_duration, num_buckets_lookback, max_likes_per_bucket)

    # Write out result
    user_history_output_path = out_dir / f"user_history_{out_dir.name}.parquet"
    user_history_lf.sink_parquet(user_history_output_path)

    # Stage info
    info_lines = [
        f"stage: featurize",
        f"runtime_seconds: {time.time()-t0:.2f}",
        f"settings: bucket_duration={bucket_duration}, num_buckets_lookback={num_buckets_lookback}, max_likes_per_bucket={max_likes_per_bucket}",
        f"inputs: likes_core",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    return {
        'output_dir': out_dir,
        'artifacts': {
            'user_history_path': str(user_history_output_path),
        }
    }
