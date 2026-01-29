#!/usr/bin/env python3
# ^ Shebang: tells Unix-like systems to run this file with the Python 3 interpreter.

"""
Stage 2: 
# ^ Top-level docstring describing what this pipeline stage is.

Inputs:
# ^ Placeholder for documenting required inputs for this stage.

Outputs under <run_dir>/featurize/<timestamp>/:
# ^ Placeholder for documenting where outputs are written.
"""

from __future__ import annotations
# ^ Enables postponed evaluation of type annotations (helps with forward references and avoids some import cycles).

from pathlib import Path
# ^ Imports Path for platform-independent filesystem path handling.

from typing import Dict, Any, Optional
# ^ Imports type hints: Dict/Any for return structures, Optional for nullable values.

import argparse
# ^ Imports argparse for command-line argument parsing.

import polars as pl
# ^ Imports Polars (pl) for fast DataFrame/LazyFrame operations.

import time
# ^ Imports time utilities for timing stage runtime.

from utils.pipeline.core import new_stage_timestamp_dir, select_prior_output, Context
# ^ Imports your pipeline helpers:
#   - new_stage_timestamp_dir: create a timestamped output dir for this stage
#   - select_prior_output: find the previous stage’s output directory
#   - Context: structured config/info passed through the pipeline

from utils.helpers import get_stage_logger, log_operation_start, validate_dataframe_schema, load_parquet_from_prior
# ^ Imports helper utilities:
#   - get_stage_logger: create a file/logger for this stage
#   - log_operation_start: log a "starting operation" message
#   - validate_dataframe_schema: enforce expected columns + types
#   - load_parquet_from_prior: load parquet artifacts from prior stage output


# TODO: Add an "end_window_lookback"?
# ^ Future idea: allow the bucket end time to lag behind the bucket timestamp by k buckets.
# e.g. if bucket is at t, only include likes up to bucket t-k (instead of up to t or t-1)
# ^ Concrete example of the TODO: you may want to exclude the most recent bucket(s).


def _generate_user_history_from_likes(
    likes_core_lf: pl.LazyFrame,
    # ^ Input: a Polars LazyFrame of likes (lazy = query plan, not executed yet).

    bucket_duration: str,
    # ^ "daily" or "hourly": defines the time granularity for bucketing user history.

    num_buckets_lookback: int,
    # ^ How many future buckets (relative to each like’s bucket ceiling) each like should contribute to.

    max_likes_per_bucket: Optional[int],
    # ^ If set, caps how many likes are retained per (user, bucket) after aggregation.

    random_seed: Optional[int],
    # ^ If set, makes per-bucket sampling deterministic/reproducible.
) -> pl.LazyFrame:
    # ^ Returns a LazyFrame representing user history events (did, subject_uri, record_created_at_bucket).

    # set random seed
    # ^ Comment describing the next block’s intent.
    if random_seed is not None:
        # ^ Only set the seed if the caller provided one.
        pl.set_random_seed(random_seed)
        # ^ Sets Polars’ random seed (affects sampling ops like list.sample).

    # repeat likes num_buckets_lookback times
    # ^ We will replicate each like across multiple future buckets so each bucket contains the prior lookback window.
    if bucket_duration == 'daily':
        # ^ Daily bucketing case.
        user_history_lf = likes_core_lf.with_columns(
            # ^ Add/replace columns on the LazyFrame (still lazy).

            (pl.col("record_created_at").dt.truncate("1d") + pl.duration(days=1)).alias("record_created_at_ceil")
            # ^ Compute the next-day boundary ("ceiling"):
            #   1) take record_created_at
            #   2) truncate to day start (00:00)
            #   3) add 1 day -> start of the *next* day
            #   4) store as record_created_at_ceil
        ).with_columns(
            pl.int_ranges(0, num_buckets_lookback).alias("bucket_offset")
            # ^ Create a list column bucket_offset = [0, 1, ..., num_buckets_lookback-1]
            #   (one list per row), to represent offsets into future buckets.
        ).explode("bucket_offset").with_columns(
            # ^ Explode turns each element of the bucket_offset list into its own row (replicating likes).
            (pl.col("record_created_at_ceil") + pl.duration(days=pl.col("bucket_offset"))).alias("record_created_at_bucket")
            # ^ For each replicated row, compute the bucket timestamp by adding offset days to the ceiling time.
        )

    elif bucket_duration == 'hourly':
        # ^ Hourly bucketing case (same pattern, but using hours).
        user_history_lf = likes_core_lf.with_columns(
            # ^ Add/replace columns on the LazyFrame.

            (pl.col("record_created_at").dt.truncate("1h") + pl.duration(hours=1)).alias("record_created_at_ceil")
            # ^ Compute the next-hour boundary:
            #   1) truncate record_created_at to the hour
            #   2) add 1 hour -> start of the next hour
            #   3) store as record_created_at_ceil
        ).with_columns(
            pl.int_ranges(0, num_buckets_lookback).alias("bucket_offset")
            # ^ Create a list column of offsets [0..num_buckets_lookback-1] (in hours).
        ).explode("bucket_offset").with_columns(
            # ^ Replicate each like across offsets by exploding the list.
            (pl.col("record_created_at_ceil") + pl.duration(hours=pl.col("bucket_offset"))).alias("record_created_at_bucket")
            # ^ Compute the bucket timestamp by adding offset hours to the ceiling time.
        )

    else:
        # ^ If caller passed an unsupported bucket_duration, fail loudly.
        raise ValueError(f"Unsupported bucket_duration: {bucket_duration}")
        # ^ Raise a descriptive error.

    # get unique likes per bucket, and count them
    # ^ Next we aggregate to (did, bucket) and deduplicate liked URIs within each bucket.
    user_history_lf = user_history_lf.drop(
        "bucket_offset", "record_created_at_ceil", "record_created_at"
        # ^ Drop intermediate helper columns we no longer need:
        #   - bucket_offset (only used to replicate rows)
        #   - record_created_at_ceil (only used to compute bucket)
        #   - record_created_at (original event time not needed after bucketing)
    ).group_by(["did", "record_created_at_bucket"]).agg(
        # ^ Group by user DID and computed bucket time, then aggregate.

        pl.col("subject_uri").unique().alias("subject_uri")
        # ^ Within each (did, bucket), keep the unique subject_uri values as a *list column* named subject_uri.
    ).with_columns(
        # ^ Add derived columns after aggregation.

        pl.col("subject_uri").list.len().alias("num_likes_in_bucket")
        # ^ Compute how many unique likes are in each bucket and store as num_likes_in_bucket.
    )

    # random sample likes per bucket if max_likes_per_bucket is set
    # ^ Optionally cap the number of likes per bucket by sampling within each list.
    if max_likes_per_bucket is not None and max_likes_per_bucket > 0:
        # ^ Only do this if the cap is provided and is positive.
        user_history_lf = user_history_lf.with_columns(
            # ^ Replace the subject_uri list column with a sampled version.

            pl.col("subject_uri")
            # ^ Start from the list column of URIs per (did, bucket).

            .list.sample(
                # ^ Sample elements from each list.

                pl.min_horizontal([
                    # ^ Compute the sample size per row as the minimum of:
                    pl.col("subject_uri").list.len(),
                    # ^   - the number of URIs available in that bucket
                    pl.lit(max_likes_per_bucket),
                    # ^   - the configured maximum cap
                ]),
                with_replacement=False,
                # ^ Sampling is without replacement (no duplicates introduced).
                shuffle=False,
                # ^ Do not shuffle before sampling (keeps deterministic order behavior under some conditions).
            )
            .alias("subject_uri")
            # ^ Name the sampled list column subject_uri (overwriting the previous one).
        )

    return user_history_lf.explode("subject_uri").drop("num_likes_in_bucket")
    # ^ Convert from “one row per (did, bucket) with a list of URIs”
    #   to “one row per (did, bucket, subject_uri)” by exploding subject_uri,
    #   then drop num_likes_in_bucket because it’s no longer needed downstream.


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    # ^ Main entry point for this pipeline stage. Returns a dict describing outputs.

    run_dir = Path(context.run_dir).resolve()
    # ^ Convert the run_dir string to an absolute Path.

    out_dir = new_stage_timestamp_dir(run_dir, '02_featurize')
    # ^ Create a timestamped output directory for this stage under run_dir.

    # Initialize logger
    # ^ Set up logging for this stage.
    logger = get_stage_logger('STAGE_02_FEATURIZE', log_file=out_dir / 'stage.log')
    # ^ Create a logger named STAGE_02_FEATURIZE that writes to <out_dir>/stage.log.

    # Try to use prior get_data output when available
    # ^ Find the output folder from stage 01_get_data.
    prior_path = select_prior_output(
        run_dir,
        '01_get_data',
        # ^ Stage name to look for.

        use_latest=context.use_latest,
        # ^ Whether to automatically use the latest prior output if multiple exist.

        prior_path=context.prior_outputs.get('01_get_data')
        # ^ Optionally use an explicitly provided prior path from context if available.
    )
    if prior_path is None:
        # ^ If we couldn't find prior output, we can't proceed.
        raise FileNotFoundError(f"Could not find raw data in 01_get_data")
        # ^ Crash with a clear error message.

    # get args
    # ^ Pull command-line arguments out of args, normalizing types.
    bucket_duration = str(args.bucket_duration)
    # ^ Bucket granularity ("daily" or "hourly") as a string.

    num_buckets_lookback = int(args.num_buckets_lookback)
    # ^ Lookback window length (in buckets), coerced to int.

    max_likes_per_bucket = int(args.max_likes_per_bucket)
    # ^ Maximum likes per bucket, coerced to int.
    #   (Note: if args.max_likes_per_bucket can be None, this int(...) would fail.)

    random_seed = args.random_seed
    # ^ Random seed (possibly None) copied directly.

    log_operation_start('Load raw data from prior stage', 'STAGE_02_FEATURIZE', logger)
    # ^ Log that we're starting the "load raw data" operation.

    t0 = time.time()
    # ^ Start a timer so we can report runtime.

    likes_core_lf: pl.LazyFrame = load_parquet_from_prior(prior_path, "likes_core_")
    # ^ Load parquet file(s) from the prior stage whose name starts with "likes_core_"
    #   as a Polars LazyFrame (lazy scan).

    validate_dataframe_schema(likes_core_lf, {"did": str, "record_created_at": pl.Datetime, "subject_uri": str})
    # ^ Assert the input has required columns with expected types:
    #   - did: string
    #   - record_created_at: datetime
    #   - subject_uri: string

    log_operation_start('Aggregate likes into user history store', 'STAGE_02_FEATURIZE', logger)
    # ^ Log that we're starting the aggregation operation.

    user_history_lf: pl.LazyFrame = _generate_user_history_from_likes(
        likes_core_lf,
        bucket_duration,
        num_buckets_lookback,
        max_likes_per_bucket,
        random_seed
    )
    # ^ Build the lazy transformation that turns likes into a per-bucket user history table.

    validate_dataframe_schema(user_history_lf, {"did": str, "subject_uri": str, "record_created_at_bucket": pl.Datetime})
    # ^ Assert the output has required columns with expected types:
    #   - did: string
    #   - subject_uri: string
    #   - record_created_at_bucket: datetime

    # Write out result
    # ^ Persist the computed LazyFrame to disk as parquet.
    user_history_output_path = out_dir / f"user_history_{out_dir.name}.parquet"
    # ^ Construct the output parquet path using the timestamped directory name.

    user_history_lf.sink_parquet(user_history_output_path)
    # ^ Execute the lazy query plan and write the result to the parquet file.

    # Stage info
    # ^ Write a small metadata file describing what happened in this stage.
    info_lines = [
        f"stage: featurize",
        # ^ Human-readable stage name.

        f"runtime_seconds: {time.time()-t0:.2f}",
        # ^ Runtime (seconds) since t0, formatted to 2 decimals.

        f"settings: bucket_duration={bucket_duration}, num_buckets_lookback={num_buckets_lookback}, max_likes_per_bucket={max_likes_per_bucket}",
        # ^ Record key settings used for this run.

        f"inputs: likes_core",
        # ^ Record which primary input artifact(s) were used.
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')
    # ^ Write stage_info.txt containing one line per entry, and end with a newline.

    return {
        # ^ Return a structured summary for the pipeline runner / orchestrator.

        'output_dir': out_dir,
        # ^ The folder where this stage wrote its outputs.

        'artifacts': {
            # ^ A mapping of artifact names to their file paths.
            'user_history_path': str(user_history_output_path),
            # ^ Path (as string) to the output parquet file.
        }
    }
    # ^ End of run().
