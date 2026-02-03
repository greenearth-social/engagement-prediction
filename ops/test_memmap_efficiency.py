#!/usr/bin/env python3
"""
Test script to evaluate the efficiency of the memmap embedding lookup approach.

This script:
1. Loads likes_core and posts_core parquet files
2. Samples 100 complete cases (target likes)
3. For each target like, finds the user's prior likes (up to 5 most recent before target)
4. Retrieves embeddings from memmap for each prior like
5. Averages the embeddings to create a user history representation
6. Reports timing statistics

Usage:
    python ops/test_memmap_efficiency.py --data-dir <path_to_get_data_output>
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class PriorLikesEntry:
    """Entry in the prior likes directory for a target like."""
    target_like_idx: int  # Index in the sampled target likes
    user_did: str
    target_subject_uri: str
    target_emb_idx: int
    target_created_at: str
    prior_emb_indices: List[int]  # emb_idx values for prior likes (up to 5)


def build_prior_likes_directory(
    likes_df: pl.DataFrame,
    sample_size: int = 100,
    max_prior_likes: int = 5,
    random_seed: int = 42,
) -> List[PriorLikesEntry]:
    """
    Build a directory mapping each sampled target like to its prior likes.
    
    Args:
        likes_df: DataFrame with columns [did, subject_uri, record_created_at, emb_idx]
        sample_size: Number of target likes to sample
        max_prior_likes: Maximum number of prior likes to include per target
        random_seed: Random seed for sampling
        
    Returns:
        List of PriorLikesEntry objects, one per sampled target like
    """
    print(f"Building prior likes directory for {sample_size} sampled target likes...")
    
    # Sort likes by user and timestamp to find prior likes
    likes_df = likes_df.sort(["did", "record_created_at"])
    
    # Sample target likes (we need users with at least 2 likes to have prior history)
    # Count likes per user
    user_like_counts = likes_df.group_by("did").agg(pl.len().alias("n_likes"))
    users_with_history = user_like_counts.filter(pl.col("n_likes") >= 2)["did"].to_list()
    
    print(f"  {len(users_with_history):,} users have >=2 likes (can have prior history)")
    
    # Filter to likes from users with history, excluding their first like
    # (which has no prior history)
    likes_with_history = likes_df.filter(pl.col("did").is_in(users_with_history))
    
    # Add row number within each user to identify which likes have priors
    likes_with_history = likes_with_history.with_columns(
        pl.col("record_created_at").rank("ordinal").over("did").alias("like_order")
    )
    
    # Filter to likes that have at least 1 prior (like_order > 1)
    eligible_targets = likes_with_history.filter(pl.col("like_order") > 1)
    print(f"  {len(eligible_targets):,} likes have at least 1 prior like")
    
    # Sample target likes
    np.random.seed(random_seed)
    n_eligible = len(eligible_targets)
    if n_eligible < sample_size:
        print(f"  WARNING: Only {n_eligible} eligible targets, using all")
        sample_indices = list(range(n_eligible))
    else:
        sample_indices = np.random.choice(n_eligible, size=sample_size, replace=False).tolist()
    
    sampled_targets = eligible_targets[sample_indices]
    print(f"  Sampled {len(sampled_targets)} target likes")
    
    # Build the directory
    directory: List[PriorLikesEntry] = []
    
    # Group original likes by user for efficient lookup
    user_likes_map: Dict[str, pl.DataFrame] = {}
    for user_did in sampled_targets["did"].unique().to_list():
        user_likes_map[user_did] = likes_df.filter(pl.col("did") == user_did).sort("record_created_at")
    
    # Build entries
    for idx, row in enumerate(sampled_targets.iter_rows(named=True)):
        user_did = row["did"]
        target_uri = row["subject_uri"]
        target_created_at = row["record_created_at"]
        target_emb_idx = row["emb_idx"]
        
        # Get all likes for this user, sorted by time
        user_likes = user_likes_map[user_did]
        
        # Find prior likes (before target_created_at)
        prior_likes = user_likes.filter(
            pl.col("record_created_at") < target_created_at
        ).sort("record_created_at", descending=True)  # Most recent first
        
        # Take up to max_prior_likes
        prior_emb_indices = prior_likes.head(max_prior_likes)["emb_idx"].to_list()
        
        directory.append(PriorLikesEntry(
            target_like_idx=idx,
            user_did=user_did,
            target_subject_uri=target_uri,
            target_emb_idx=target_emb_idx,
            target_created_at=str(target_created_at),
            prior_emb_indices=prior_emb_indices,
        ))
    
    # Stats on prior likes counts
    prior_counts = [len(e.prior_emb_indices) for e in directory]
    print(f"  Prior likes per target: min={min(prior_counts)}, max={max(prior_counts)}, "
          f"mean={np.mean(prior_counts):.1f}")
    
    return directory


def retrieve_and_average_embeddings(
    directory: List[PriorLikesEntry],
    mmap: np.memmap,
    embed_dim: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Retrieve embeddings for each entry's prior likes and compute averages.
    
    Args:
        directory: List of PriorLikesEntry objects
        mmap: Memory-mapped embedding file
        embed_dim: Embedding dimension
        
    Returns:
        Tuple of (averaged_embeddings array, timing_stats dict)
    """
    n_entries = len(directory)
    averaged_embeddings = np.zeros((n_entries, embed_dim), dtype=np.float32)
    
    # Timing
    lookup_times = []
    avg_times = []
    total_lookups = 0
    
    for i, entry in enumerate(directory):
        if not entry.prior_emb_indices:
            # No prior likes - leave as zeros (shouldn't happen given our filtering)
            continue
        
        # Time the lookup
        t0 = time.perf_counter()
        
        # Retrieve embeddings for prior likes
        prior_embeddings = mmap[entry.prior_emb_indices]  # Shape: (n_priors, embed_dim)
        
        lookup_time = time.perf_counter() - t0
        lookup_times.append(lookup_time)
        total_lookups += len(entry.prior_emb_indices)
        
        # Time the averaging
        t1 = time.perf_counter()
        averaged_embeddings[i] = np.mean(prior_embeddings, axis=0)
        avg_time = time.perf_counter() - t1
        avg_times.append(avg_time)
    
    stats = {
        "n_entries": n_entries,
        "total_lookups": total_lookups,
        "mean_lookup_time_ms": np.mean(lookup_times) * 1000,
        "median_lookup_time_ms": np.median(lookup_times) * 1000,
        "max_lookup_time_ms": np.max(lookup_times) * 1000,
        "total_lookup_time_ms": np.sum(lookup_times) * 1000,
        "mean_avg_time_ms": np.mean(avg_times) * 1000,
        "total_avg_time_ms": np.sum(avg_times) * 1000,
        "lookups_per_second": total_lookups / np.sum(lookup_times) if np.sum(lookup_times) > 0 else 0,
    }
    
    return averaged_embeddings, stats


def main():
    parser = argparse.ArgumentParser(description="Test memmap embedding lookup efficiency")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to get_data output directory containing parquets and memmap"
    )
    parser.add_argument("--sample-size", type=int, default=100, help="Number of target likes to sample")
    parser.add_argument("--max-prior-likes", type=int, default=5, help="Max prior likes per target")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    # Find the files
    likes_parquets = list(data_dir.glob("likes_core_*.parquet"))
    posts_parquets = list(data_dir.glob("posts_core_*.parquet"))
    embeddings_files = list(data_dir.glob("embeddings_*.npy"))
    
    if not likes_parquets:
        raise FileNotFoundError(f"No likes_core parquet found in {data_dir}")
    if not embeddings_files:
        raise FileNotFoundError(f"No embeddings npy found in {data_dir}")
    
    likes_path = likes_parquets[0]
    embeddings_path = embeddings_files[0]
    
    print(f"Loading data from {data_dir}")
    print(f"  Likes: {likes_path.name}")
    print(f"  Embeddings: {embeddings_path.name}")
    
    # Load likes
    t0 = time.perf_counter()
    likes_df = pl.read_parquet(likes_path)
    likes_load_time = time.perf_counter() - t0
    print(f"  Loaded {len(likes_df):,} likes in {likes_load_time:.2f}s")
    
    # Get embedding dimension from file size
    file_size = embeddings_path.stat().st_size
    embed_dim = 384  # Known from stage_info.txt
    n_embeddings = file_size // (embed_dim * 4)  # float32 = 4 bytes
    print(f"  Embeddings: {n_embeddings:,} x {embed_dim} ({file_size / 1e9:.2f} GB)")
    
    # Open memmap
    t0 = time.perf_counter()
    mmap = np.memmap(embeddings_path, dtype=np.float32, mode='r', shape=(n_embeddings, embed_dim))
    mmap_open_time = time.perf_counter() - t0
    print(f"  Opened memmap in {mmap_open_time*1000:.2f}ms")
    
    print()
    
    # Build prior likes directory
    t0 = time.perf_counter()
    directory = build_prior_likes_directory(
        likes_df=likes_df,
        sample_size=args.sample_size,
        max_prior_likes=args.max_prior_likes,
        random_seed=args.random_seed,
    )
    directory_build_time = time.perf_counter() - t0
    print(f"  Built directory in {directory_build_time:.2f}s")
    
    print()
    
    # Retrieve and average embeddings
    print("Retrieving and averaging embeddings...")
    t0 = time.perf_counter()
    averaged_embeddings, timing_stats = retrieve_and_average_embeddings(
        directory=directory,
        mmap=mmap,
        embed_dim=embed_dim,
    )
    total_retrieval_time = time.perf_counter() - t0
    print(f"  Total retrieval time: {total_retrieval_time*1000:.2f}ms")
    
    print()
    print("=" * 60)
    print("TIMING RESULTS")
    print("=" * 60)
    print(f"  Sample size: {timing_stats['n_entries']} target likes")
    print(f"  Total lookups: {timing_stats['total_lookups']} embeddings")
    print()
    print("  Lookup timing:")
    print(f"    Mean:   {timing_stats['mean_lookup_time_ms']:.4f} ms/entry")
    print(f"    Median: {timing_stats['median_lookup_time_ms']:.4f} ms/entry")
    print(f"    Max:    {timing_stats['max_lookup_time_ms']:.4f} ms/entry")
    print(f"    Total:  {timing_stats['total_lookup_time_ms']:.2f} ms")
    print()
    print("  Averaging timing:")
    print(f"    Mean:  {timing_stats['mean_avg_time_ms']:.4f} ms/entry")
    print(f"    Total: {timing_stats['total_avg_time_ms']:.2f} ms")
    print()
    print(f"  Throughput: {timing_stats['lookups_per_second']:,.0f} embeddings/second")
    print()
    
    # Sanity check the embeddings
    print("Sanity checks:")
    print(f"  Output shape: {averaged_embeddings.shape}")
    print(f"  Non-zero entries: {np.sum(np.any(averaged_embeddings != 0, axis=1))}/{len(averaged_embeddings)}")
    print(f"  Embedding norm range: [{np.linalg.norm(averaged_embeddings, axis=1).min():.3f}, "
          f"{np.linalg.norm(averaged_embeddings, axis=1).max():.3f}]")
    
    # Show a few example entries from the directory
    print()
    print("=" * 60)
    print("SAMPLE DIRECTORY ENTRIES")
    print("=" * 60)
    for i, entry in enumerate(directory[:3]):
        print(f"\nEntry {i}:")
        print(f"  User: {entry.user_did[:30]}...")
        print(f"  Target post: {entry.target_subject_uri[:50]}...")
        print(f"  Target emb_idx: {entry.target_emb_idx}")
        print(f"  Target created_at: {entry.target_created_at}")
        print(f"  Prior emb_indices: {entry.prior_emb_indices}")
    
    # Clean up
    del mmap


if __name__ == "__main__":
    main()
