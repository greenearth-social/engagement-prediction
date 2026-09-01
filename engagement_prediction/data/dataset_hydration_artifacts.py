"""Disk-bounded artifact construction for Stage 7 dataset hydration.

The important intermediate lifecycle is:

``selected_metadata``
    Stage 5's selected URI roles joined to Stage 3's canonical metadata.
``selected_embedding_rows``
    Encoded raw payload rows for those URIs. Duplicates are retained so an
    older source row can supply a valid embedding.
``embedding_shards``
    One decoded ``Float32`` NumPy array per URI partition, aligned with a
    narrow URI-key Parquet part.
``embeddings.npy`` and ``hydrated_post_metadata``
    The final concatenated memmap and its pre-vocabulary URI metadata mapping.

Later shuffles temporarily change from Stage 00 URI partitions to Stage 5 URI
partitions for liker counts, then to Stage 2 DID partitions for public query
lists. Large source and event relations are eager only as bounded partitions or
file batches; the narrow selected-key lookup and final author vocabulary are
the intentional resident exceptions.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import logging
import shutil
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from engagement_prediction.data import author_vocabulary
from engagement_prediction.data import dataset_hydration
from engagement_prediction.data import ingex
from engagement_prediction.data import like_counts
from engagement_prediction.data import post_liker_users
from engagement_prediction.data import post_selection
from engagement_prediction.data import user_history
from engagement_prediction.data.author_indices import AUTHOR_UNK_IDX
from engagement_prediction.data.parquet import (
    ensure_typed_parquet_dataset,
    read_parquet_parts,
    scan_parquet_artifact,
    sink_partitioned_parquet,
    write_parquet_part_if_not_empty,
)


EMBEDDING_COPY_BATCH_SIZE = 8_192


def _public_part(path: Path, partition_id: int) -> list[Path]:
    """Return one public URI-partition file, if it exists."""

    candidate = Path(path) / f"part-{partition_id:05d}.parquet"
    return [candidate] if candidate.exists() else []


def _partition_paths(path: Path, partition_id: int) -> list[Path]:
    """Return files from a hash-partitioned staging dataset."""

    return post_selection.partition_parquet_paths(path, partition_id)


def route_selected_posts(
    *,
    post_liker_posts_path: Path,
    output_path: Path,
    partition_count: int,
) -> None:
    """Re-route Stage 5 selected roles to Stage 3's URI partitions.

    Stage 5 may use a different physical partition count for liker events.
    Re-routing only the narrow role table lets the next join remain local to
    Stage 3's authoritative metadata partitions.
    """
    sink_partitioned_parquet(
        scan_parquet_artifact(post_liker_posts_path)
        .select("subject_uri", "is_positive", "is_history", "is_negative")
        .with_columns(post_selection.post_partition_expr(partition_count)),
        output_path=output_path,
        key="_post_partition",
    )


def build_selected_metadata(
    *,
    stage3_posts_path: Path,
    selected_post_routes_path: Path,
    output_path: Path,
    partition_count: int,
    logger: logging.Logger,
) -> dict[str, int]:
    """Join Stage 5 roles to authoritative Stage 3 metadata one partition at a time.

    The result is ``selected_metadata``: exactly one row per post that Stage 7
    may expose to a model, with canonical creation/author/source fields and
    potentially overlapping positive, history, and negative flags.
    """
    output_path.mkdir(parents=True, exist_ok=False)
    selected_count = 0
    role_counts = {
        "selected_positive_post_count": 0,
        "selected_history_post_count": 0,
        "selected_negative_post_count": 0,
    }
    for partition_id in range(partition_count):
        metadata_df = read_parquet_parts(
            _public_part(stage3_posts_path, partition_id),
            empty=post_selection.empty_frame(post_selection.POST_SCHEMA),
        )
        roles_df = read_parquet_parts(
            _partition_paths(selected_post_routes_path, partition_id),
            empty=dataset_hydration.empty_frame({
                "subject_uri": pl.String,
                "is_positive": pl.Boolean,
                "is_history": pl.Boolean,
                "is_negative": pl.Boolean,
            }),
        )
        selected_df = (
            metadata_df.join(roles_df, on="subject_uri", how="inner")
            .sort("subject_uri")
        )
        if selected_df.height != roles_df.height:
            raise ValueError("Stage 5 selected posts are not all present in Stage 3 posts")
        write_parquet_part_if_not_empty(
            selected_df,
            output_path / f"part-{partition_id:05d}.parquet",
        )
        selected_count += selected_df.height
        for role, output_name in (
            ("is_positive", "selected_positive_post_count"),
            ("is_history", "selected_history_post_count"),
            ("is_negative", "selected_negative_post_count"),
        ):
            role_counts[output_name] += selected_df.filter(pl.col(role)).height
        logger.info(
            "Prepared selected metadata partition %s/%s: posts=%s",
            partition_id + 1,
            partition_count,
            f"{selected_df.height:,}",
        )
    return {"selected_post_count": selected_count, **role_counts}


def materialize_selected_embedding_rows(
    *,
    post_paths: list[str],
    reply_paths: list[str],
    posts_start: Any,
    posts_end: Any,
    selected_metadata_path: Path,
    output_path: Path,
    temporary_routes_root: Path,
    partition_count: int,
    source_batch_size: int,
    worker_count: int,
    logger: logging.Logger,
) -> dict[str, int]:
    """Route selected embedding payloads in bounded parallel file batches.

    The narrow selected-URI lookup is loaded once. Each source batch is then
    scanned once with its encoded payloads, semi-joined to those selected keys,
    and streamed into its own temporary URI partitions. A thread pool lets
    independent scans overlap while sharing the immutable lookup instead of
    copying it into worker processes. This keeps memory bounded without the
    prior discovery pass, which was ineffective when selected URIs appeared in
    almost every physical source file. Batch payload routes are deleted after
    their files are moved into the shared URI-partitioned output.
    """
    if source_batch_size <= 0:
        raise ValueError("embedding_source_batch_size must be positive")
    if worker_count <= 0:
        raise ValueError("dataset_hydration_worker_count must be positive")
    output_path.mkdir(parents=True, exist_ok=False)
    selected_keys_df = (
        scan_parquet_artifact(selected_metadata_path)
        .select("subject_uri", "is_reply")
        .collect(engine="streaming")
    )
    if selected_keys_df.get_column("subject_uri").null_count():
        raise ValueError("Selected embedding metadata contains a null subject_uri")
    if selected_keys_df.height != selected_keys_df.get_column("subject_uri").n_unique():
        raise ValueError("Selected embedding metadata contains duplicate subject_uri rows")
    selected_keys_by_reply = {
        is_reply: (
            selected_keys_df
            .filter(pl.col("is_reply") == is_reply)
            .select("subject_uri")
        )
        for is_reply in (False, True)
    }
    logger.info(
        "Loaded selected embedding lookup once: posts=%s roots=%s replies=%s "
        "estimated_memory_mb=%.1f",
        f"{selected_keys_df.height:,}",
        f"{selected_keys_by_reply[False].height:,}",
        f"{selected_keys_by_reply[True].height:,}",
        selected_keys_df.estimated_size("mb"),
    )
    del selected_keys_df

    total_source_files = len(post_paths) + len(reply_paths)
    batch_tasks: list[dict[str, Any]] = []
    for label, paths, is_reply in (
        ("root", post_paths, False),
        ("reply", reply_paths, True),
    ):
        source_keys_df = selected_keys_by_reply[is_reply]
        if source_keys_df.is_empty():
            logger.info(
                "Skipping %s embedding sources because no selected %s posts exist",
                label,
                label,
            )
            continue
        batch_count = (len(paths) + source_batch_size - 1) // source_batch_size
        for batch_index, batch_start in enumerate(
            range(0, len(paths), source_batch_size)
        ):
            batch_paths = paths[batch_start:batch_start + source_batch_size]
            batch_tasks.append({
                "label": label,
                "batch_index": batch_index,
                "batch_count": batch_count,
                "batch_paths": batch_paths,
                "is_reply": is_reply,
                "source_keys_df": source_keys_df,
                "posts_start": posts_start,
                "posts_end": posts_end,
                "output_path": output_path,
                "temporary_routes_root": temporary_routes_root,
                "partition_count": partition_count,
                "logger": logger,
            })

    effective_worker_count = min(worker_count, len(batch_tasks))
    logger.info(
        "Filtering %s embedding source batches with %s worker threads",
        len(batch_tasks),
        effective_worker_count,
    )
    if effective_worker_count <= 1:
        results = [
            _filter_embedding_source_batch(**task)
            for task in batch_tasks
        ]
    else:
        with ThreadPoolExecutor(max_workers=effective_worker_count) as executor:
            futures = [
                executor.submit(_filter_embedding_source_batch, **task)
                for task in batch_tasks
            ]
            results = [future.result() for future in as_completed(futures)]

    payload_source_files = sum(result["source_file_count"] for result in results)
    selected_rows = sum(result["selected_row_count"] for result in results)
    if temporary_routes_root.exists():
        temporary_routes_root.rmdir()
    return {
        "embedding_source_file_count": total_source_files,
        "embedding_source_batch_count": len(batch_tasks),
        "embedding_source_worker_count": effective_worker_count,
        "payload_embedding_source_file_count": payload_source_files,
        "selected_embedding_source_row_count": selected_rows,
    }


def _filter_embedding_source_batch(
    *,
    label: str,
    batch_index: int,
    batch_count: int,
    batch_paths: list[str],
    is_reply: bool,
    source_keys_df: pl.DataFrame,
    posts_start: Any,
    posts_end: Any,
    output_path: Path,
    temporary_routes_root: Path,
    partition_count: int,
    logger: logging.Logger,
) -> dict[str, int]:
    """Filter and route one independent raw embedding source batch."""

    batch_name = f"{label}-batch-{batch_index:06d}"
    logger.info(
        "Filtering embedding %s batch %s/%s: files=%s",
        label,
        batch_index + 1,
        batch_count,
        len(batch_paths),
    )
    payload_routes_path = temporary_routes_root / f"{batch_name}-payloads"
    # The semi-join is part of the same streaming plan as the payload scan and
    # partition sink, so unselected payload rows are read but never retained
    # across batches or materialized outside this batch's temporary route.
    selected_payloads_lf = (
        dataset_hydration.normalize_embedding_source_rows(
            ingex.scan_parquet_files(batch_paths),
            posts_start=posts_start,
            posts_end=posts_end,
            is_reply=is_reply,
        )
        .join(source_keys_df.lazy(), on="subject_uri", how="semi")
        .select(
            "subject_uri",
            "post_created_at",
            "author_did",
            "embeddings",
        )
        .with_columns(post_selection.post_partition_expr(partition_count))
    )
    sink_partitioned_parquet(
        selected_payloads_lf,
        output_path=payload_routes_path,
        key="_post_partition",
    )
    selected_row_count = 0
    for partition_id in range(partition_count):
        payload_parts = _partition_paths(payload_routes_path, partition_id)
        if not payload_parts:
            continue
        partition_dir = output_path / f"partition-{partition_id:05d}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        for part_index, payload_part in enumerate(payload_parts):
            selected_row_count += pq.read_metadata(payload_part).num_rows
            payload_part.replace(
                partition_dir / f"{batch_name}-{part_index:05d}.parquet"
            )
    shutil.rmtree(payload_routes_path)
    logger.info(
        "Filtered embedding %s batch %s/%s in one pass: files=%s selected_rows=%s",
        label,
        batch_index + 1,
        batch_count,
        len(batch_paths),
        f"{selected_row_count:,}",
    )
    return {
        "source_file_count": len(batch_paths),
        "selected_row_count": selected_row_count,
    }


def _write_embedding_partition(
    *,
    selected_embedding_rows_path: Path,
    valid_embedding_rows_path: Path,
    embedding_shards_path: Path,
    embedding_model: str,
    embedding_dim: int,
    partition_id: int,
) -> dict[str, Any]:
    """Decode, select, and write one independent URI partition.

    This top-level worker is intentionally process-safe. It reads and writes
    only files owned by ``partition_id``, so multiple workers can run without
    coordinating mutable state. The returned object contains only compact
    statistics; decoded vectors never cross the process boundary.
    """
    source_paths = sorted(
        (selected_embedding_rows_path / f"partition-{partition_id:05d}").glob(
            "*.parquet"
        )
    )
    if source_paths:
        source_df = pl.read_parquet(source_paths)
    else:
        source_df = dataset_hydration.empty_frame({
            "subject_uri": pl.String,
            "post_created_at": dataset_hydration.UTC_DATETIME,
            "author_did": pl.String,
            "embeddings": pl.List(
                pl.Struct({"key": pl.String, "value": pl.String})
            ),
        })

    # Validation produces the actual Float32 winner. Keeping that array avoids
    # decompressing and unpacking the same selected payload again while writing
    # the shard.
    selected_vectors, stats = dataset_hydration.select_latest_valid_embedding_vectors(
        source_df,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )
    source_row_count = source_df.height
    del source_df

    selected_metadata_df = pl.DataFrame(
        {"subject_uri": [row[0] for row in selected_vectors]},
        schema=dataset_hydration.VALID_EMBEDDING_KEY_SCHEMA,
    )
    selected_metadata_df.write_parquet(
        valid_embedding_rows_path / f"part-{partition_id:05d}.parquet",
        compression="zstd",
    )
    shard_path = embedding_shards_path / f"part-{partition_id:05d}.npy"
    shard = np.lib.format.open_memmap(
        shard_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(selected_vectors), embedding_dim),
    )
    for row_index, (_uri, _created_at, _author_did, vector) in enumerate(
        selected_vectors
    ):
        shard[row_index] = vector
    shard.flush()
    del shard
    return {
        "partition_id": partition_id,
        "embedding_count": len(selected_vectors),
        "source_row_count": source_row_count,
        "stats": stats,
    }


def write_embedding_shards(
    *,
    selected_embedding_rows_path: Path,
    valid_embedding_rows_path: Path,
    embedding_shards_path: Path,
    embedding_model: str,
    embedding_dim: int,
    partition_count: int,
    worker_count: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Select and write embedding shards with bounded partition concurrency.

    ``selected_embedding_rows`` still contains encoded payloads and duplicate
    source rows. Every URI-hash partition is complete and independent, so a
    spawn-based process pool can decode several partitions concurrently. The
    aligned ``valid_embedding_rows`` keys preserve each shard's row order, and
    the caller still concatenates shards in partition-ID order for stable
    global embedding indices.

    Set ``worker_count=1`` for a serial low-memory/debugging path.
    """
    if partition_count <= 0:
        raise ValueError("partition_count must be positive")
    if worker_count <= 0:
        raise ValueError("dataset_hydration_worker_count must be positive")
    valid_embedding_rows_path.mkdir(parents=True, exist_ok=False)
    embedding_shards_path.mkdir(parents=True, exist_ok=False)
    effective_worker_count = min(worker_count, partition_count)
    logger.info(
        "Selecting embeddings across %s URI partitions with %s worker processes",
        partition_count,
        effective_worker_count,
    )

    results: list[dict[str, Any]] = []
    worker_kwargs = [
        {
            "selected_embedding_rows_path": selected_embedding_rows_path,
            "valid_embedding_rows_path": valid_embedding_rows_path,
            "embedding_shards_path": embedding_shards_path,
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim,
            "partition_id": partition_id,
        }
        for partition_id in range(partition_count)
    ]
    if effective_worker_count == 1:
        for kwargs in worker_kwargs:
            result = _write_embedding_partition(**kwargs)
            results.append(result)
            logger.info(
                "Selected embeddings for URI partition %s/%s: valid=%s source_rows=%s",
                result["partition_id"] + 1,
                partition_count,
                f"{result['embedding_count']:,}",
                f"{result['source_row_count']:,}",
            )
    else:
        # Polars is multithreaded, so use ``spawn`` rather than forking its
        # existing thread pool. Each child owns one bounded eager partition and
        # exits after the pool closes, releasing its retained decoded vectors.
        with ProcessPoolExecutor(
            max_workers=effective_worker_count,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = {
                executor.submit(_write_embedding_partition, **kwargs): kwargs[
                    "partition_id"
                ]
                for kwargs in worker_kwargs
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                logger.info(
                    "Selected embeddings for URI partition %s/%s: valid=%s "
                    "source_rows=%s",
                    result["partition_id"] + 1,
                    partition_count,
                    f"{result['embedding_count']:,}",
                    f"{result['source_row_count']:,}",
                )

    results.sort(key=lambda result: result["partition_id"])
    counts = [int(result["embedding_count"]) for result in results]
    totals: dict[str, int] = defaultdict(int)
    partition_stats: list[dict[str, int]] = []
    for result in results:
        stats = result["stats"]
        for name, value in stats.items():
            totals[name] += int(value)
        partition_stats.append({"partition_id": result["partition_id"], **stats})
    return {
        **dict(totals),
        "embedding_partition_worker_count": effective_worker_count,
        "embedding_partition_counts": counts,
        "embedding_partition_stats": partition_stats,
    }


def _copy_finite_embedding_shard(
    *,
    shard: np.ndarray,
    destination: np.ndarray,
    destination_offset: int,
) -> None:
    """Validate and copy one shard while its rows are already being read."""

    if (
        shard.dtype != np.float32
        or shard.ndim != 2
        or destination.ndim != 2
        or shard.shape[1] != destination.shape[1]
    ):
        raise ValueError(
            f"Unexpected embedding shard shape/dtype: {shard.shape} {shard.dtype}"
        )
    for start in range(0, shard.shape[0], EMBEDDING_COPY_BATCH_SIZE):
        end = min(start + EMBEDDING_COPY_BATCH_SIZE, shard.shape[0])
        # A bounded copy causes each source page to be read once, then reused
        # for both the finite-value check and the destination write.
        vectors = np.array(shard[start:end], dtype=np.float32, copy=True)
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding shard contains non-finite values")
        destination[
            destination_offset + start:destination_offset + end
        ] = vectors


def publish_embeddings_and_post_metadata(
    *,
    selected_metadata_path: Path,
    valid_embedding_rows_path: Path,
    embedding_shards_path: Path,
    embeddings_path: Path,
    hydrated_post_metadata_path: Path,
    embedding_dim: int,
    partition_count: int,
) -> dict[str, int]:
    """Concatenate vector shards and publish their aligned raw-author metadata.

    Each shard is validated and removed after its vectors and post-index rows
    have been published. This keeps final-memmap growth from overlapping with a
    complete second copy of the embeddings and avoids a later whole-memmap
    finite-value pass. The matching URI key rows define ``emb_idx`` offsets,
    while Stage 3 metadata—not potentially older embedding-source metadata—
    remains authoritative. Author indices are attached later, after final
    training-feature support is known.
    """
    # Reading only .npy headers gives the exact final shape without loading any
    # shard vectors. Partition order defines dense global emb_idx ranges.
    counts = [
        int(np.load(embedding_shards_path / f"part-{partition_id:05d}.npy", mmap_mode="r").shape[0])
        for partition_id in range(partition_count)
    ]
    total = sum(counts)
    if total == 0:
        raise ValueError("Dataset hydration found no valid content embeddings")
    memmap = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(total, embedding_dim),
    )
    hydrated_post_metadata_path.mkdir(parents=True, exist_ok=False)
    offset = 0
    for partition_id, count in enumerate(counts):
        shard_path = embedding_shards_path / f"part-{partition_id:05d}.npy"
        shard = np.load(
            shard_path,
            mmap_mode="r",
        )
        if shard.shape[0] != count:
            raise ValueError("Embedding shard row count changed during publication")
        _copy_finite_embedding_shard(
            shard=shard,
            destination=memmap,
            destination_offset=offset,
        )
        valid_df = pl.read_parquet(
            valid_embedding_rows_path / f"part-{partition_id:05d}.parquet"
        ).select("subject_uri").with_row_index("_local_idx")
        index_df = valid_df.with_columns(
            (pl.col("_local_idx") + offset).cast(pl.UInt32).alias("emb_idx")
        ).select("subject_uri", "emb_idx")
        metadata_df = read_parquet_parts(
            _public_part(selected_metadata_path, partition_id),
            empty=post_selection.empty_frame({
                **post_selection.POST_SCHEMA,
                "is_positive": pl.Boolean,
                "is_history": pl.Boolean,
                "is_negative": pl.Boolean,
            }),
        )
        posts_df = dataset_hydration.build_hydrated_post_metadata(
            metadata_df,
            index_df,
        )
        posts_df.write_parquet(
            hydrated_post_metadata_path / f"part-{partition_id:05d}.parquet",
            compression="zstd",
        )
        del shard
        shard_path.unlink()
        offset += count
    embedding_shards_path.rmdir()
    memmap.flush()
    del memmap
    return {
        "hydrated_post_count": total,
        "embedding_dim": embedding_dim,
    }


def publish_posts_with_author_indices(
    *,
    hydrated_post_metadata_path: Path,
    authors_df: pl.DataFrame,
    posts_path: Path,
    partition_count: int,
) -> dict[str, int]:
    """Attach the final Stage 7 vocabulary to URI-partitioned post metadata."""

    posts_path.mkdir(parents=True, exist_ok=False)
    post_count = 0
    unk_count = 0
    for partition_id in range(partition_count):
        metadata_df = read_parquet_parts(
            _public_part(hydrated_post_metadata_path, partition_id),
            empty=dataset_hydration.empty_frame(
                dataset_hydration.HYDRATED_POST_METADATA_SCHEMA
            ),
        )
        posts_df = dataset_hydration.attach_post_author_indices(
            metadata_df,
            authors_df,
        )
        posts_df.write_parquet(
            posts_path / f"part-{partition_id:05d}.parquet",
            compression="zstd",
        )
        post_count += posts_df.height
        unk_count += posts_df.filter(
            pl.col("author_idx") == AUTHOR_UNK_IDX
        ).height
    return {
        "hydrated_post_count": post_count,
        "author_unk_post_count": unk_count,
    }


def materialize_usage_routes(
    *,
    query_positives_lf: pl.LazyFrame,
    query_histories_lf: pl.LazyFrame,
    hourly_candidates_lf: pl.LazyFrame,
    positive_routes_path: Path,
    history_routes_path: Path,
    negative_routes_path: Path,
    partition_count: int,
) -> None:
    """Relationalize every model-facing post use and route it by post URI.

    Histories become one row per list position so missing embeddings can be
    filtered without losing the alignment or order needed to rebuild the
    public lists later.
    """
    sink_partitioned_parquet(
        query_positives_lf.select(
            "did", "query_hour", "subject_uri", "like_created_at"
        ).with_columns(post_selection.post_partition_expr(partition_count)),
        output_path=positive_routes_path,
        key="_post_partition",
    )
    flat_histories_lf = (
        query_histories_lf.with_columns(
            pl.int_ranges(
                pl.lit(0),
                pl.col("history_subject_uris").list.len(),
            ).alias("_history_position")
        )
        .explode(
            "history_subject_uris",
            "history_like_created_ats",
            "_history_position",
            empty_as_null=True,
        )
        .filter(pl.col("history_subject_uris").is_not_null())
        .select(
            "did",
            "query_hour",
            pl.col("_history_position").cast(pl.UInt32),
            pl.col("history_subject_uris").alias("subject_uri"),
            pl.col("history_like_created_ats").alias("like_created_at"),
        )
        .with_columns(post_selection.post_partition_expr(partition_count))
    )
    sink_partitioned_parquet(
        flat_histories_lf,
        output_path=history_routes_path,
        key="_post_partition",
    )
    sink_partitioned_parquet(
        hourly_candidates_lf.select(
            "query_hour",
            "subject_uri",
            "selection_source",
            pl.col("prior_like_count").alias("_stage4_prior_like_count"),
        ).with_columns(post_selection.post_partition_expr(partition_count)),
        output_path=negative_routes_path,
        key="_post_partition",
    )


def hydrate_usage_partitions(
    *,
    hydrated_post_metadata_path: Path,
    positive_routes_path: Path,
    history_routes_path: Path,
    negative_routes_path: Path,
    hydrated_positives_path: Path,
    hydrated_histories_path: Path,
    hydrated_negatives_path: Path,
    partition_count: int,
) -> dict[str, int]:
    """Join model-facing uses to hydrated posts one URI partition at a time.

    A URI missing from ``posts/`` has no valid content embedding. Its uses are
    removed without replacement, but the query itself is not dropped here;
    query publication later drops only queries with no surviving positive.
    """

    for path in (hydrated_positives_path, hydrated_histories_path, hydrated_negatives_path):
        path.mkdir(parents=True, exist_ok=False)
    counts = {
        "input_positive_count": 0,
        "input_history_item_count": 0,
        "input_negative_count": 0,
        "retained_positive_count": 0,
        "retained_history_item_count": 0,
        "retained_negative_count": 0,
    }
    for partition_id in range(partition_count):
        posts_df = read_parquet_parts(
            _public_part(hydrated_post_metadata_path, partition_id),
            empty=dataset_hydration.empty_frame(
                dataset_hydration.HYDRATED_POST_METADATA_SCHEMA
            ),
        ).select("subject_uri", "emb_idx", "post_created_at", "author_did")
        input_positives_df = read_parquet_parts(
            _partition_paths(positive_routes_path, partition_id),
            empty=dataset_hydration.empty_frame({
                "did": pl.String,
                "query_hour": dataset_hydration.UTC_DATETIME,
                "subject_uri": pl.String,
                "like_created_at": dataset_hydration.UTC_DATETIME,
            }),
        )
        positives_df = input_positives_df.join(posts_df, on="subject_uri", how="inner")
        input_histories_df = read_parquet_parts(
            _partition_paths(history_routes_path, partition_id),
            empty=dataset_hydration.empty_frame({
                "did": pl.String,
                "query_hour": dataset_hydration.UTC_DATETIME,
                "_history_position": pl.UInt32,
                "subject_uri": pl.String,
                "like_created_at": dataset_hydration.UTC_DATETIME,
            }),
        )
        histories_df = input_histories_df.join(posts_df, on="subject_uri", how="inner")
        input_negatives_df = read_parquet_parts(
            _partition_paths(negative_routes_path, partition_id),
            empty=dataset_hydration.empty_frame({
                "query_hour": dataset_hydration.UTC_DATETIME,
                "subject_uri": pl.String,
                "selection_source": pl.String,
                "_stage4_prior_like_count": pl.UInt64,
            }),
        )
        negatives_df = input_negatives_df.join(posts_df, on="subject_uri", how="inner")
        for df, output in (
            (positives_df, hydrated_positives_path),
            (histories_df, hydrated_histories_path),
            (negatives_df, hydrated_negatives_path),
        ):
            write_parquet_part_if_not_empty(
                df,
                output / f"part-{partition_id:05d}.parquet",
            )
        counts["input_positive_count"] += input_positives_df.height
        counts["input_history_item_count"] += input_histories_df.height
        counts["input_negative_count"] += input_negatives_df.height
        counts["retained_positive_count"] += positives_df.height
        counts["retained_history_item_count"] += histories_df.height
        counts["retained_negative_count"] += negatives_df.height
    return {
        **counts,
        "missing_embedding_positive_count": (
            counts["input_positive_count"] - counts["retained_positive_count"]
        ),
        "missing_embedding_history_item_count": (
            counts["input_history_item_count"] - counts["retained_history_item_count"]
        ),
        "missing_embedding_negative_count": (
            counts["input_negative_count"] - counts["retained_negative_count"]
        ),
    }


def route_hydrated_usage_for_counts(
    *,
    hydrated_paths: tuple[Path, Path, Path],
    output_paths: tuple[Path, Path, Path],
    partition_count: int,
) -> None:
    """Re-route hydrated uses from Stage 3 partitions to Stage 5 partitions.

    This second physical shuffle is necessary because strict as-of counts join
    against Stage 5's complete post-liker event partitions rather than Stage
    3's metadata layout.
    """

    for input_path, output_path in zip(hydrated_paths, output_paths):
        parts = sorted(input_path.glob("*.parquet"))
        if not parts:
            output_path.mkdir(parents=True, exist_ok=False)
            continue
        sink_partitioned_parquet(
            pl.scan_parquet(parts).with_columns(
                post_selection.post_partition_expr(partition_count)
            ),
            output_path=output_path,
            key="_post_partition",
        )


def materialize_post_liker_use_windows(
    *,
    queries_lf: pl.LazyFrame,
    positive_routes_path: Path,
    history_routes_path: Path,
    negative_routes_path: Path,
    output_path: Path,
    partition_count: int,
) -> dict[str, int]:
    """Record the final training and all-split use horizon for each post.

    Only queries retaining at least one hydrated positive are model-facing.
    Their positive and history occurrences are keyed by the complete query;
    shared negatives are keyed by query hour. A like event before a post's
    maximum use hour is visible to at least one use, so these narrow windows
    let the Stage 5 scan discard events that no model example can consume.
    """

    positive_parts = sorted(positive_routes_path.rglob("*.parquet"))
    if not positive_parts:
        raise ValueError("Cannot build post-liker use windows without surviving positives")
    positive_lf = pl.scan_parquet(positive_parts)
    surviving_keys_lf = positive_lf.select("did", "query_hour").unique()
    retained_queries_df = (
        queries_lf.join(
            surviving_keys_lf,
            on=["did", "query_hour"],
            how="semi",
        )
        .select("did", "query_hour", "split")
        .unique()
        .collect(engine="streaming")
    )
    retained_hours_df = retained_queries_df.group_by("query_hour").agg(
        (pl.col("split") == "train").any().alias("_is_training_use")
    )

    def query_usage(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.join(retained_queries_df, on=["did", "query_hour"], how="inner")
            .select(
                "subject_uri",
                "emb_idx",
                "query_hour",
                (pl.col("split") == "train").alias("_is_training_use"),
            )
        )

    output_path.mkdir(parents=True, exist_ok=False)
    post_count = 0
    training_post_count = 0
    for partition_id in range(partition_count):
        usage_frames = []
        positive_df = read_parquet_parts(
            _partition_paths(positive_routes_path, partition_id),
            empty=pl.DataFrame(),
        )
        if not positive_df.is_empty():
            usage_frames.append(query_usage(positive_df))
        history_df = read_parquet_parts(
            _partition_paths(history_routes_path, partition_id),
            empty=pl.DataFrame(),
        )
        if not history_df.is_empty():
            usage_frames.append(query_usage(history_df))
        negative_df = read_parquet_parts(
            _partition_paths(negative_routes_path, partition_id),
            empty=pl.DataFrame(),
        )
        if not negative_df.is_empty():
            usage_frames.append(
                negative_df.join(retained_hours_df, on="query_hour", how="inner")
                .select(
                    "subject_uri",
                    "emb_idx",
                    "query_hour",
                    "_is_training_use",
                )
            )
        nonempty_usage_frames = [frame for frame in usage_frames if not frame.is_empty()]
        if not nonempty_usage_frames:
            continue
        windows_df = (
            pl.concat(nonempty_usage_frames, how="vertical")
            .group_by("subject_uri")
            .agg(
                pl.col("emb_idx").first().cast(pl.UInt32),
                pl.col("emb_idx").n_unique().alias("_embedding_index_count"),
                pl.col("query_hour").max().alias("final_use_query_hour"),
                pl.when(pl.col("_is_training_use"))
                .then(pl.col("query_hour"))
                .otherwise(None)
                .max()
                .alias("final_training_use_query_hour"),
            )
        )
        conflicting_count = windows_df.filter(
            pl.col("_embedding_index_count") != 1
        ).height
        if conflicting_count:
            raise ValueError(
                f"Post-liker use windows contain {conflicting_count} URIs with conflicting emb_idx values"
            )
        public_windows_df = windows_df.select(
            post_liker_users.POST_LIKER_USE_WINDOW_COLUMNS
        ).sort("subject_uri")
        partition_path = output_path / f"_post_partition={partition_id}"
        write_parquet_part_if_not_empty(
            public_windows_df,
            partition_path / "part-00000.parquet",
        )
        post_count += public_windows_df.height
        training_post_count += int(
            public_windows_df.get_column("final_training_use_query_hour")
            .is_not_null()
            .sum()
        )
    ensure_typed_parquet_dataset(
        output_path,
        post_liker_users.POST_LIKER_USE_WINDOW_SCHEMA,
    )
    return {
        "post_count": post_count,
        "training_post_count": training_post_count,
    }


def attach_prior_counts(
    *,
    positive_routes_path: Path,
    history_routes_path: Path,
    negative_routes_path: Path,
    post_liker_events_path: Path,
    post_liker_use_windows_path: Path,
    post_liker_feature_events_path: Path,
    counted_positives_path: Path,
    counted_histories_path: Path,
    counted_negatives_path: Path,
    partition_count: int,
    logger: logging.Logger,
) -> dict[str, int]:
    """Attach strict as-of Stage 5 event counts to every hydrated post use.

    The unique ``(subject_uri, query_hour)`` relation avoids recalculating a
    count when the same post is used by several users in one hour. The narrow
    cumulative event table is built once per URI partition, while each full
    usage partition is read, counted, and written sequentially. Negative counts
    are cross-checked against Stage 4 before its private column is discarded.
    """

    for path in (
        counted_positives_path,
        counted_histories_path,
        counted_negatives_path,
        post_liker_feature_events_path,
    ):
        path.mkdir(parents=True, exist_ok=False)
    pair_count = 0
    raw_event_count = 0
    deduplicated_event_count = 0
    retained_feature_event_count = 0
    training_visible_event_count = 0
    for partition_id in range(partition_count):
        relation_partitions = []
        for source_path, output_path, is_negative in (
            (positive_routes_path, counted_positives_path, False),
            (history_routes_path, counted_histories_path, False),
            (negative_routes_path, counted_negatives_path, True),
        ):
            parts = _partition_paths(source_path, partition_id)
            if parts:
                relation_partitions.append((parts, output_path, is_negative))
        if not relation_partitions:
            logger.info(
                "Computed as-of popularity for Stage 5 URI partition %s/%s: pairs=0",
                partition_id + 1,
                partition_count,
            )
            continue

        event_parts = _public_part(post_liker_events_path, partition_id)
        events_df = (
            pl.read_parquet(event_parts)
            if event_parts
            else dataset_hydration.empty_frame({
                "subject_uri": pl.String,
                "liker_did": pl.String,
                "like_created_at": dataset_hydration.UTC_DATETIME,
            })
        )
        raw_event_count += events_df.height
        # One cumulative row per event timestamp supports strict as-of lookups
        # for all uses in this partition via the shared join-asof algorithm.
        cumulative_likes_df = like_counts.build_cumulative_like_counts(events_df)

        window_parts = _partition_paths(post_liker_use_windows_path, partition_id)
        windows_df = read_parquet_parts(
            window_parts,
            empty=post_liker_users.empty_frame(
                post_liker_users.POST_LIKER_USE_WINDOW_SCHEMA
            ),
        )
        deduplicated_events_df = events_df.unique(
            ["subject_uri", "liker_did", "like_created_at"],
            keep="first",
        )
        deduplicated_event_count += deduplicated_events_df.height
        feature_events_df = (
            deduplicated_events_df.join(windows_df, on="subject_uri", how="inner")
            .filter(pl.col("like_created_at") < pl.col("final_use_query_hour"))
            .with_columns(
                (
                    pl.col("final_training_use_query_hour").is_not_null()
                    & (
                        pl.col("like_created_at")
                        < pl.col("final_training_use_query_hour")
                    )
                ).alias("is_training_visible")
            )
            .select(post_liker_users.POST_LIKER_FEATURE_EVENT_COLUMNS)
            .sort("emb_idx", "like_created_at", "liker_did")
        )
        write_parquet_part_if_not_empty(
            feature_events_df,
            post_liker_feature_events_path / f"part-{partition_id:05d}.parquet",
        )
        retained_feature_event_count += feature_events_df.height
        training_visible_event_count += int(
            feature_events_df.get_column("is_training_visible").sum() or 0
        )
        del events_df, deduplicated_events_df, feature_events_df, windows_df

        pair_frames = []
        for parts, output_path, is_negative in relation_partitions:
            source_df = pl.read_parquet(parts)
            counts_df = like_counts.lookup_prior_like_counts(
                source_df.select("subject_uri", "query_hour"),
                cumulative_likes_df,
            )
            pair_frames.append(counts_df.select("subject_uri", "query_hour"))
            counted_df = source_df.join(
                counts_df,
                on=["subject_uri", "query_hour"],
                how="left",
            ).with_columns(pl.col("prior_like_count").fill_null(0).cast(pl.UInt64))
            if is_negative:
                mismatches = counted_df.filter(
                    pl.col("_stage4_prior_like_count") != pl.col("prior_like_count")
                ).height
                if mismatches:
                    raise ValueError(
                        f"Stage 4/Stage 5 popularity mismatch for {mismatches} negative rows"
                    )
            counted_df.write_parquet(
                output_path / f"part-{partition_id:05d}.parquet",
                compression="zstd",
            )
            del source_df, counts_df, counted_df

        partition_pair_count = (
            pl.concat(pair_frames).unique().height if pair_frames else 0
        )
        pair_count += partition_pair_count
        logger.info(
            "Computed as-of popularity for Stage 5 URI partition %s/%s: pairs=%s",
            partition_id + 1,
            partition_count,
            f"{partition_pair_count:,}",
        )
    ensure_typed_parquet_dataset(
        post_liker_feature_events_path,
        post_liker_users.POST_LIKER_FEATURE_EVENT_SCHEMA,
    )
    return {
        "post_query_hour_pair_count": pair_count,
        "raw_post_liker_event_count": raw_event_count,
        "deduplicated_post_liker_event_count": deduplicated_event_count,
        "retained_post_liker_feature_event_count": retained_feature_event_count,
        "training_visible_post_liker_event_count": training_visible_event_count,
    }


def _liker_support_partition_paths(path: Path, partition_id: int) -> list[Path]:
    """Return training-visible events routed to one liker-DID partition."""

    partition_dir = Path(path) / f"_liker_partition={partition_id}"
    return sorted(partition_dir.rglob("*.parquet")) if partition_dir.exists() else []


def build_post_liker_user_vocabulary(
    *,
    feature_events_path: Path,
    support_routes_path: Path,
    support_shards_path: Path,
    vocabulary_path: Path,
    min_training_event_count: int,
    max_vocabulary_size: int,
    partition_count: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Build a bounded vocabulary from exact events visible to train uses."""

    if min_training_event_count < 1:
        raise ValueError("min_post_liker_user_training_event_count must be at least 1")
    if max_vocabulary_size < 0:
        raise ValueError("max_post_liker_user_vocabulary_size may not be negative")
    event_parts = sorted(feature_events_path.glob("*.parquet"))
    events_lf = (
        pl.scan_parquet(event_parts)
        if event_parts
        else post_liker_users.empty_frame(
            post_liker_users.POST_LIKER_FEATURE_EVENT_SCHEMA
        ).lazy()
    )
    training_events_lf = events_lf.filter(pl.col("is_training_visible")).select(
        "liker_did"
    )
    sink_partitioned_parquet(
        training_events_lf.with_columns(
            post_liker_users.support_partition_expr(partition_count)
        ),
        output_path=support_routes_path,
        key="_liker_partition",
    )

    support_shards_path.mkdir(parents=True, exist_ok=False)
    pre_threshold_user_count = 0
    threshold_eligible_user_count = 0
    training_event_count = 0
    for partition_id in range(partition_count):
        rows_df = read_parquet_parts(
            _liker_support_partition_paths(support_routes_path, partition_id),
            empty=pl.DataFrame(schema={"liker_did": pl.String}),
        )
        support_df = (
            rows_df.group_by("liker_did")
            .len(name="training_event_count")
            .with_columns(pl.col("training_event_count").cast(pl.UInt64))
            .sort("liker_did")
        )
        eligible_df = support_df.filter(
            pl.col("training_event_count") >= min_training_event_count
        )
        write_parquet_part_if_not_empty(
            eligible_df,
            support_shards_path / f"part-{partition_id:05d}.parquet",
        )
        pre_threshold_user_count += support_df.height
        threshold_eligible_user_count += eligible_df.height
        training_event_count += int(support_df.get_column("training_event_count").sum() or 0)
        logger.info(
            "Aggregated post-liker user support partition %s/%s: users=%s eligible=%s",
            partition_id + 1,
            partition_count,
            f"{support_df.height:,}",
            f"{eligible_df.height:,}",
        )

    vocabulary_path.mkdir(parents=True, exist_ok=False)
    vocabulary_part_path = vocabulary_path / "part-00000.parquet"
    support_parts = sorted(support_shards_path.glob("*.parquet"))
    if support_parts and max_vocabulary_size:
        selected_support_path = support_shards_path.parent / "selected_liker_support.parquet"
        (
            pl.scan_parquet(support_parts)
            .sort(
                ["training_event_count", "liker_did"],
                descending=[True, False],
            )
            .head(max_vocabulary_size)
            .sink_parquet(
                selected_support_path,
                compression="zstd",
                maintain_order=True,
                engine="streaming",
            )
        )
        post_liker_users.add_liker_indices(
            pl.scan_parquet(selected_support_path)
        ).sink_parquet(
            vocabulary_part_path,
            compression="zstd",
            maintain_order=True,
            engine="streaming",
        )
    else:
        post_liker_users.empty_frame(
            post_liker_users.POST_LIKER_USER_VOCABULARY_SCHEMA
        ).write_parquet(vocabulary_part_path, compression="zstd")
    validation = post_liker_users.validate_post_liker_user_vocabulary(
        pl.scan_parquet(vocabulary_part_path),
        min_training_event_count=min_training_event_count,
        max_vocabulary_size=max_vocabulary_size,
    )
    return {
        **validation,
        "pre_threshold_user_count": pre_threshold_user_count,
        "threshold_eligible_user_count": threshold_eligible_user_count,
        "excluded_below_threshold_user_count": (
            pre_threshold_user_count - threshold_eligible_user_count
        ),
        "excluded_by_cap_user_count": (
            threshold_eligible_user_count - validation["user_count"]
        ),
        "all_training_event_count": training_event_count,
        "known_training_event_count": validation["training_event_count"],
        "unk_training_event_count": (
            training_event_count - validation["training_event_count"]
        ),
    }


def attach_post_liker_user_indices(
    *,
    feature_events_path: Path,
    vocabulary_path: Path,
    indexed_events_path: Path,
    partition_count: int,
) -> dict[str, int]:
    """Map retained raw DIDs to known indices or the shared UNK row."""

    vocabulary_df = scan_parquet_artifact(vocabulary_path).collect(engine="streaming")
    indexed_events_path.mkdir(parents=True, exist_ok=False)
    event_count = 0
    unknown_event_count = 0
    post_count = 0
    for partition_id in range(partition_count):
        parts = _public_part(feature_events_path, partition_id)
        if not parts:
            continue
        events_df = pl.read_parquet(parts)
        indexed_df = (
            events_df.join(
                vocabulary_df.select("liker_did", "liker_idx"),
                on="liker_did",
                how="left",
            )
            .with_columns(
                pl.col("liker_idx")
                .fill_null(post_liker_users.POST_LIKER_USER_UNK_IDX)
                .cast(pl.UInt32)
            )
            # Raw DID is the stable final tie-breaker before it is omitted from
            # the compact loader relation. Distinct OOV users at the same time
            # therefore remain distinct, deterministic UNK events.
            .sort("emb_idx", "like_created_at", "liker_did")
            .select(post_liker_users.INDEXED_POST_LIKER_EVENT_COLUMNS)
        )
        write_parquet_part_if_not_empty(
            indexed_df,
            indexed_events_path / f"part-{partition_id:05d}.parquet",
        )
        event_count += indexed_df.height
        unknown_event_count += int(
            (indexed_df.get_column("liker_idx") == post_liker_users.POST_LIKER_USER_UNK_IDX)
            .sum()
        )
        post_count += indexed_df.get_column("emb_idx").n_unique()
    ensure_typed_parquet_dataset(
        indexed_events_path,
        post_liker_users.INDEXED_POST_LIKER_EVENT_SCHEMA,
    )
    return {
        "event_count": event_count,
        "unknown_event_count": unknown_event_count,
        "known_event_count": event_count - unknown_event_count,
        "post_count": post_count,
    }


def _author_support_distribution(df: pl.DataFrame) -> dict[str, int]:
    """Return compact feature-count buckets for Stage 7 diagnostics."""

    count = pl.col("training_feature_count")
    return {
        "1_to_4": df.filter(count.is_between(1, 4)).height,
        "5_to_19": df.filter(count.is_between(5, 19)).height,
        "20_to_49": df.filter(count.is_between(20, 49)).height,
        "50_to_99": df.filter(count.is_between(50, 99)).height,
        "100_to_999": df.filter(count.is_between(100, 999)).height,
        "1000_plus": df.filter(count >= 1000).height,
    }


def build_author_vocabulary(
    *,
    queries_lf: pl.LazyFrame,
    counted_positives_path: Path,
    counted_histories_path: Path,
    counted_negatives_path: Path,
    exposure_routes_path: Path,
    eligible_shards_path: Path,
    authors_path: Path,
    min_training_feature_count: int,
    partition_count: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Build the vocabulary from final train-facing feature occurrences.

    A query is eligible for support only if at least one positive survived
    embedding hydration. Histories are counted as retained event positions,
    positives as user/post labels, and negatives as shared post/hour rows.
    This is deterministic and independent of dataloader batching or epochs.
    """

    if min_training_feature_count < 1:
        raise ValueError("min_author_training_feature_count must be at least 1")
    positive_parts = sorted(counted_positives_path.glob("*.parquet"))
    if not positive_parts:
        raise ValueError("Cannot build an author vocabulary without surviving positives")
    positive_lf = pl.scan_parquet(positive_parts)
    surviving_keys_lf = positive_lf.select("did", "query_hour").unique()
    retained_train_queries_lf = (
        queries_lf.filter(pl.col("split") == "train")
        .join(surviving_keys_lf, on=["did", "query_hour"], how="semi")
        .select("did", "query_hour")
        .unique()
    )
    train_hours_lf = retained_train_queries_lf.select("query_hour").unique()

    def role_rows(lf: pl.LazyFrame, role: str) -> pl.LazyFrame:
        """Represent each feature occurrence as one narrow support counter row."""

        return lf.select(
            "author_did",
            pl.lit(1 if role == "positive" else 0, dtype=pl.UInt64).alias(
                "training_positive_count"
            ),
            pl.lit(1 if role == "history" else 0, dtype=pl.UInt64).alias(
                "training_history_count"
            ),
            pl.lit(1 if role == "negative" else 0, dtype=pl.UInt64).alias(
                "training_negative_count"
            ),
        )

    exposure_frames = [
        role_rows(
            positive_lf.join(
                retained_train_queries_lf,
                on=["did", "query_hour"],
                how="semi",
            ),
            "positive",
        )
    ]
    history_parts = sorted(counted_histories_path.glob("*.parquet"))
    if history_parts:
        exposure_frames.append(
            role_rows(
                pl.scan_parquet(history_parts).join(
                    retained_train_queries_lf,
                    on=["did", "query_hour"],
                    how="semi",
                ),
                "history",
            )
        )
    negative_parts = sorted(counted_negatives_path.glob("*.parquet"))
    if negative_parts:
        exposure_frames.append(
            role_rows(
                pl.scan_parquet(negative_parts).join(
                    train_hours_lf,
                    on="query_hour",
                    how="semi",
                ),
                "negative",
            )
        )
    sink_partitioned_parquet(
        pl.concat(exposure_frames).with_columns(
            author_vocabulary.support_partition_expr(partition_count)
        ),
        output_path=exposure_routes_path,
        key="_author_partition",
    )

    eligible_shards_path.mkdir(parents=True, exist_ok=False)
    totals = {
        "pre_threshold_author_count": 0,
        "eligible_author_count": 0,
        "training_feature_count": 0,
        "training_positive_count": 0,
        "training_history_count": 0,
        "training_negative_count": 0,
    }
    distribution = {
        name: 0
        for name in (
            "1_to_4",
            "5_to_19",
            "20_to_49",
            "50_to_99",
            "100_to_999",
            "1000_plus",
        )
    }
    partition_stats = []
    exposure_schema = {
        "author_did": pl.String,
        "training_positive_count": pl.UInt64,
        "training_history_count": pl.UInt64,
        "training_negative_count": pl.UInt64,
    }
    for partition_id in range(partition_count):
        rows_df = read_parquet_parts(
            _author_support_partition_paths(exposure_routes_path, partition_id),
            empty=author_vocabulary.empty_frame(exposure_schema),
        ).select(list(exposure_schema))
        support_df = author_vocabulary.aggregate_support_rows(rows_df)
        eligible_df = support_df.filter(
            pl.col("training_feature_count") >= min_training_feature_count
        )
        write_parquet_part_if_not_empty(
            eligible_df,
            eligible_shards_path / f"part-{partition_id:05d}.parquet",
        )
        totals["pre_threshold_author_count"] += support_df.height
        totals["eligible_author_count"] += eligible_df.height
        for column in (
            "training_feature_count",
            "training_positive_count",
            "training_history_count",
            "training_negative_count",
        ):
            totals[column] += int(support_df.get_column(column).sum() or 0)
        for name, value in _author_support_distribution(support_df).items():
            distribution[name] += value
        partition_stats.append({
            "partition_id": partition_id,
            "author_count": support_df.height,
            "eligible_author_count": eligible_df.height,
        })
        logger.info(
            "Aggregated author support partition %s/%s: authors=%s eligible=%s",
            partition_id + 1,
            partition_count,
            f"{support_df.height:,}",
            f"{eligible_df.height:,}",
        )

    authors_path.mkdir(parents=True, exist_ok=False)
    authors_part_path = authors_path / "part-00000.parquet"
    eligible_parts = sorted(eligible_shards_path.glob("*.parquet"))
    if eligible_parts:
        author_vocabulary.add_author_indices(pl.scan_parquet(eligible_parts)).sink_parquet(
            authors_part_path,
            compression="zstd",
            maintain_order=True,
            engine="streaming",
        )
    else:
        author_vocabulary.empty_frame(
            author_vocabulary.AUTHOR_VOCABULARY_SCHEMA
        ).write_parquet(authors_part_path, compression="zstd")
    validation = author_vocabulary.validate_author_vocabulary(
        pl.scan_parquet(authors_part_path),
        min_training_feature_count=min_training_feature_count,
    )
    if validation["author_count"] != totals["eligible_author_count"]:
        raise ValueError("Published author vocabulary count does not match support filtering")
    return {
        **totals,
        "excluded_author_count": (
            totals["pre_threshold_author_count"] - totals["eligible_author_count"]
        ),
        "support_distribution": distribution,
        "author_partition_stats": partition_stats,
        "public_validation": validation,
    }


def _author_support_partition_paths(path: Path, partition_id: int) -> list[Path]:
    """Return rows assigned to one Stage 7 author-support partition."""

    partition_dir = Path(path) / f"_author_partition={partition_id}"
    return sorted(partition_dir.rglob("*.parquet")) if partition_dir.exists() else []


def attach_author_indices_to_usage(
    *,
    counted_paths: tuple[Path, Path, Path],
    indexed_paths: tuple[Path, Path, Path],
    authors_df: pl.DataFrame,
    partition_count: int,
) -> dict[str, int]:
    """Replace raw author DIDs with final vocabulary indices in usage rows."""

    for output_path in indexed_paths:
        output_path.mkdir(parents=True, exist_ok=False)
    totals = {
        "positive_feature_count": 0,
        "positive_unk_count": 0,
        "history_feature_count": 0,
        "history_unk_count": 0,
        "negative_feature_count": 0,
        "negative_unk_count": 0,
    }
    role_names = ("positive", "history", "negative")
    author_indices_df = authors_df.select("author_did", "author_idx")
    for partition_id in range(partition_count):
        for role, input_path, output_path in zip(
            role_names,
            counted_paths,
            indexed_paths,
        ):
            parts = _public_part(input_path, partition_id)
            if not parts:
                continue
            indexed_df = (
                pl.read_parquet(parts)
                .join(author_indices_df, on="author_did", how="left")
                .with_columns(
                    pl.col("author_idx")
                    .fill_null(AUTHOR_UNK_IDX)
                    .cast(pl.UInt32)
                )
                .drop("author_did")
            )
            indexed_df.write_parquet(
                output_path / f"part-{partition_id:05d}.parquet",
                compression="zstd",
            )
            totals[f"{role}_feature_count"] += indexed_df.height
            totals[f"{role}_unk_count"] += indexed_df.filter(
                pl.col("author_idx") == AUTHOR_UNK_IDX
            ).height
    return totals


def _query_partition_paths(path: Path, partition_id: int) -> list[Path]:
    """Return files assigned to one queried-user hash partition."""

    partition_dir = Path(path) / f"_user_partition={partition_id}"
    return sorted(partition_dir.rglob("*.parquet")) if partition_dir.exists() else []


def publish_query_artifacts(
    *,
    queries_lf: pl.LazyFrame,
    counted_positives_path: Path,
    counted_histories_path: Path,
    queries_path: Path,
    query_positives_path: Path,
    query_histories_path: Path,
    staging_path: Path,
    partition_count: int,
) -> dict[str, Any]:
    """Route by queried user, drop empty-positive queries, and rebuild histories.

    History items carry an explicit original position through URI filtering
    and popularity attachment. Sorting on that position before aggregation
    guarantees that URI, like-time, embedding, author, and count lists remain
    aligned. Queries with empty histories remain valid; queries with no
    surviving positive do not.
    """
    query_routes = staging_path / "queries"
    positive_routes = staging_path / "positives"
    history_routes = staging_path / "histories"
    sink_partitioned_parquet(
        queries_lf.with_columns(user_history.user_partition_expr(partition_count)),
        output_path=query_routes,
        key="_user_partition",
    )
    positive_parts = sorted(counted_positives_path.glob("*.parquet"))
    history_parts = sorted(counted_histories_path.glob("*.parquet"))
    if positive_parts:
        sink_partitioned_parquet(
            pl.scan_parquet(positive_parts).with_columns(
                user_history.user_partition_expr(partition_count)
            ),
            output_path=positive_routes,
            key="_user_partition",
        )
    else:
        positive_routes.mkdir(parents=True, exist_ok=False)
    if history_parts:
        sink_partitioned_parquet(
            pl.scan_parquet(history_parts).with_columns(
                user_history.user_partition_expr(partition_count)
            ),
            output_path=history_routes,
            key="_user_partition",
        )
    else:
        history_routes.mkdir(parents=True, exist_ok=False)

    for path in (queries_path, query_positives_path, query_histories_path):
        path.mkdir(parents=True, exist_ok=False)
    totals = {
        "input_query_count": 0,
        "retained_query_count": 0,
        "dropped_zero_positive_query_count": 0,
        "retained_positive_count": 0,
        "retained_history_item_count": 0,
    }
    retained_hours: set[Any] = set()
    for partition_id in range(partition_count):
        query_df = read_parquet_parts(
            _query_partition_paths(query_routes, partition_id),
            empty=dataset_hydration.empty_frame(dataset_hydration.QUERY_SCHEMA),
        )
        positive_df = read_parquet_parts(
            _query_partition_paths(positive_routes, partition_id),
            empty=dataset_hydration.empty_frame({
                "did": pl.String,
                "query_hour": dataset_hydration.UTC_DATETIME,
                "subject_uri": pl.String,
                "like_created_at": dataset_hydration.UTC_DATETIME,
                "emb_idx": pl.UInt32,
                "post_created_at": dataset_hydration.UTC_DATETIME,
                "author_idx": pl.UInt32,
                "prior_like_count": pl.UInt64,
            }),
        )
        history_df = read_parquet_parts(
            _query_partition_paths(history_routes, partition_id),
            empty=dataset_hydration.empty_frame({
                "did": pl.String,
                "query_hour": dataset_hydration.UTC_DATETIME,
                "_history_position": pl.UInt32,
                "subject_uri": pl.String,
                "like_created_at": dataset_hydration.UTC_DATETIME,
                "emb_idx": pl.UInt32,
                "post_created_at": dataset_hydration.UTC_DATETIME,
                "author_idx": pl.UInt32,
                "prior_like_count": pl.UInt64,
            }),
        )
        counts_df = positive_df.group_by("did", "query_hour").agg(
            pl.len().cast(pl.UInt32).alias("positive_count")
        )
        retained_queries_df = (
            query_df.drop("positive_count")
            .join(counts_df, on=["did", "query_hour"], how="inner")
            .select(dataset_hydration.QUERY_COLUMNS)
            .sort(["query_hour", "did"])
        )
        retained_keys = retained_queries_df.select("did", "query_hour")
        public_positives_df = (
            positive_df.join(retained_keys, on=["did", "query_hour"], how="semi")
            .select(dataset_hydration.QUERY_POSITIVE_COLUMNS)
            .sort(["query_hour", "did", "subject_uri"])
        )

        retained_history_items_df = history_df.join(
            retained_keys,
            on=["did", "query_hour"],
            how="semi",
        ).sort(["did", "query_hour", "_history_position"])
        history_lists_df = retained_history_items_df.group_by(
            "did",
            "query_hour",
            maintain_order=True,
        ).agg(
            pl.col("subject_uri").alias("history_subject_uris"),
            pl.col("like_created_at").alias("history_like_created_ats"),
            pl.col("emb_idx").alias("history_emb_indices"),
            pl.col("author_idx").alias("history_author_indices"),
            pl.col("prior_like_count").alias("history_prior_like_counts"),
        )
        public_histories_df = (
            retained_keys.join(
                history_lists_df,
                on=["did", "query_hour"],
                how="left",
            )
            .with_columns(
                pl.col("history_subject_uris").fill_null(
                    pl.lit([], dtype=pl.List(pl.String))
                ),
                pl.col("history_like_created_ats").fill_null(
                    pl.lit([], dtype=pl.List(dataset_hydration.UTC_DATETIME))
                ),
                pl.col("history_emb_indices").fill_null(
                    pl.lit([], dtype=pl.List(pl.UInt32))
                ),
                pl.col("history_author_indices").fill_null(
                    pl.lit([], dtype=pl.List(pl.UInt32))
                ),
                pl.col("history_prior_like_counts").fill_null(
                    pl.lit([], dtype=pl.List(pl.UInt64))
                ),
            )
            .select(dataset_hydration.QUERY_HISTORY_COLUMNS)
            .sort(["query_hour", "did"])
        )
        dataset_hydration.validate_query_histories(public_histories_df)
        dataset_hydration.validate_frame(
            retained_queries_df,
            dataset_hydration.QUERY_SCHEMA,
            key=["did", "query_hour"],
        )
        dataset_hydration.validate_frame(
            public_positives_df,
            dataset_hydration.QUERY_POSITIVE_SCHEMA,
            key=["did", "query_hour", "subject_uri"],
        )
        for df, path in (
            (retained_queries_df, queries_path),
            (public_positives_df, query_positives_path),
            (public_histories_df, query_histories_path),
        ):
            write_parquet_part_if_not_empty(
                df,
                path / f"part-{partition_id:05d}.parquet",
            )
        totals["input_query_count"] += query_df.height
        totals["retained_query_count"] += retained_queries_df.height
        totals["dropped_zero_positive_query_count"] += query_df.height - retained_queries_df.height
        totals["retained_positive_count"] += public_positives_df.height
        if public_histories_df.height:
            totals["retained_history_item_count"] += int(
                public_histories_df.select(
                    pl.col("history_subject_uris").list.len().sum()
                ).item()
                or 0
            )
        retained_hours.update(
            retained_queries_df.get_column("query_hour").unique().to_list()
        )

    for path, schema in (
        (queries_path, dataset_hydration.QUERY_SCHEMA),
        (query_positives_path, dataset_hydration.QUERY_POSITIVE_SCHEMA),
        (query_histories_path, dataset_hydration.QUERY_HISTORY_SCHEMA),
    ):
        ensure_typed_parquet_dataset(path, schema)
    if totals["retained_query_count"] == 0:
        raise ValueError("Dataset hydration produced no model-ready queries")
    return {**totals, "retained_query_hours": sorted(retained_hours)}


def publish_negative_candidates(
    *,
    counted_negatives_path: Path,
    retained_query_hours: list[Any],
    output_path: Path,
) -> dict[str, int]:
    """Publish hydrated negatives only for query hours that still survive."""

    parts = sorted(counted_negatives_path.glob("*.parquet"))
    output_path.mkdir(parents=True, exist_ok=False)
    if not parts or not retained_query_hours:
        ensure_typed_parquet_dataset(
            output_path,
            dataset_hydration.HOURLY_NEGATIVE_SCHEMA,
        )
        return {"retained_negative_count": 0, "retained_negative_post_count": 0}
    hours_df = pl.DataFrame(
        {"query_hour": retained_query_hours},
        schema={"query_hour": dataset_hydration.UTC_DATETIME},
    )
    final_lf = (
        pl.scan_parquet(parts)
        .join(hours_df.lazy(), on="query_hour", how="semi")
        .select(dataset_hydration.HOURLY_NEGATIVE_COLUMNS)
        .unique(["query_hour", "subject_uri"])
        .sort(["query_hour", "subject_uri"])
    )
    final_path = output_path / "part-00000.parquet"
    final_lf.sink_parquet(
        final_path,
        compression="zstd",
        maintain_order=True,
        engine="streaming",
    )
    final_scan = pl.scan_parquet(final_path)
    if final_scan.collect_schema() != pl.Schema(dataset_hydration.HOURLY_NEGATIVE_SCHEMA):
        raise ValueError("Unexpected hydrated negative-candidate schema")
    invalid_key_count = int(
        final_scan.filter(
            pl.col("query_hour").is_null() | pl.col("subject_uri").is_null()
        ).select(pl.len()).collect(engine="streaming").item()
    )
    duplicate_key_count = int(
        final_scan.group_by("query_hour", "subject_uri")
        .len()
        .filter(pl.col("len") > 1)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if invalid_key_count or duplicate_key_count:
        raise ValueError("Hydrated negative candidates have invalid or duplicate keys")
    counts = final_scan.select(
        pl.len().alias("row_count"),
        pl.col("subject_uri").n_unique().alias("post_count"),
    ).collect(engine="streaming").row(0, named=True)
    return {
        "retained_negative_count": int(counts["row_count"]),
        "retained_negative_post_count": int(counts["post_count"]),
    }


def summarize_author_index_usage_by_split(
    *,
    queries_path: Path,
    query_positives_path: Path,
    query_histories_path: Path,
    hourly_negative_candidates_path: Path,
) -> dict[str, dict[str, int]]:
    """Report known-versus-UNK model-facing occurrences for each split."""

    queries_lf = scan_parquet_artifact(queries_path).select(
        "did", "query_hour", "split"
    )
    role_frames = {
        "positive": (
            scan_parquet_artifact(query_positives_path)
            .join(queries_lf, on=["did", "query_hour"], how="inner")
            .select("split", "author_idx")
        ),
        "history": (
            scan_parquet_artifact(query_histories_path)
            .join(queries_lf, on=["did", "query_hour"], how="inner")
            .select(
                "split",
                pl.col("history_author_indices").alias("author_idx"),
            )
            .explode("author_idx", empty_as_null=True)
            .filter(pl.col("author_idx").is_not_null())
        ),
        "negative": (
            scan_parquet_artifact(hourly_negative_candidates_path)
            .join(
                queries_lf.select("query_hour", "split").unique(),
                on="query_hour",
                how="inner",
            )
            .select("split", "author_idx")
        ),
    }
    result: dict[str, dict[str, int]] = {}
    for role, role_lf in role_frames.items():
        rows = (
            role_lf.group_by("split")
            .agg(
                pl.len().alias("feature_count"),
                (pl.col("author_idx") == AUTHOR_UNK_IDX)
                .sum()
                .alias("unk_count"),
            )
            .collect(engine="streaming")
        )
        for row in rows.iter_rows(named=True):
            split_stats = result.setdefault(str(row["split"]), {})
            feature_count = int(row["feature_count"])
            unk_count = int(row["unk_count"])
            split_stats[f"{role}_feature_count"] = feature_count
            split_stats[f"{role}_known_count"] = feature_count - unk_count
            split_stats[f"{role}_unk_count"] = unk_count
    for split_stats in result.values():
        for role in role_frames:
            split_stats.setdefault(f"{role}_feature_count", 0)
            split_stats.setdefault(f"{role}_known_count", 0)
            split_stats.setdefault(f"{role}_unk_count", 0)
    return dict(sorted(result.items()))


def validate_public_bundle(
    *,
    embeddings_path: Path,
    posts_path: Path,
    queries_path: Path,
    query_positives_path: Path,
    query_histories_path: Path,
    hourly_negative_candidates_path: Path,
    authors_path: Path,
    min_author_training_feature_count: int,
    embedding_dim: int,
) -> dict[str, int]:
    """Validate final artifacts without collecting the complete dataset.

    Posts are already URI-hash partitioned and query artifacts are already
    DID-hash partitioned. Validating each matching group independently proves
    global key uniqueness because equal keys deterministically route to the
    same physical partition.
    """
    authors_validation = author_vocabulary.validate_author_vocabulary(
        scan_parquet_artifact(authors_path),
        min_training_feature_count=min_author_training_feature_count,
    )
    max_author_idx = authors_validation["author_table_num_rows"] - 1
    mmap = np.load(embeddings_path, mmap_mode="r")
    if mmap.dtype != np.float32 or mmap.ndim != 2 or mmap.shape[1] != embedding_dim:
        raise ValueError(f"Unexpected embeddings memmap shape/dtype: {mmap.shape} {mmap.dtype}")
    embedding_count = mmap.shape[0]
    del mmap

    expected_emb_idx = 0
    for part_path in sorted(posts_path.glob("part-*.parquet")):
        posts_df = pl.read_parquet(part_path)
        dataset_hydration.validate_frame(
            posts_df,
            dataset_hydration.POST_SCHEMA,
            key=["subject_uri"],
        )
        if posts_df.height and (
            posts_df.get_column("author_idx").min() < AUTHOR_UNK_IDX
            or posts_df.get_column("author_idx").max() > max_author_idx
        ):
            raise ValueError("Hydrated posts contain an invalid author index")
        if posts_df.height:
            actual_indices = posts_df.get_column("emb_idx").to_numpy()
            expected_indices = np.arange(
                expected_emb_idx,
                expected_emb_idx + posts_df.height,
                dtype=actual_indices.dtype,
            )
            if not np.array_equal(actual_indices, expected_indices):
                raise ValueError(
                    "Hydrated post embedding indices are not ordered, dense, and "
                    "memmap-aligned"
                )
        expected_emb_idx += posts_df.height
    if expected_emb_idx != embedding_count:
        raise ValueError("Hydrated post rows do not align with the embeddings memmap")

    query_part_names = {
        part.name
        for dataset_path in (queries_path, query_positives_path, query_histories_path)
        for part in dataset_path.glob("part-*.parquet")
    }
    query_count = 0
    positive_count = 0
    history_query_count = 0
    for part_name in sorted(query_part_names):
        query_part = queries_path / part_name
        positive_part = query_positives_path / part_name
        history_part = query_histories_path / part_name
        queries_df = (
            pl.read_parquet(query_part)
            if query_part.exists()
            else dataset_hydration.empty_frame(dataset_hydration.QUERY_SCHEMA)
        )
        positives_df = (
            pl.read_parquet(positive_part)
            if positive_part.exists()
            else dataset_hydration.empty_frame(dataset_hydration.QUERY_POSITIVE_SCHEMA)
        )
        histories_df = (
            pl.read_parquet(history_part)
            if history_part.exists()
            else dataset_hydration.empty_frame(dataset_hydration.QUERY_HISTORY_SCHEMA)
        )
        dataset_hydration.validate_frame(
            queries_df,
            dataset_hydration.QUERY_SCHEMA,
            key=["did", "query_hour"],
        )
        dataset_hydration.validate_frame(
            positives_df,
            dataset_hydration.QUERY_POSITIVE_SCHEMA,
            key=["did", "query_hour", "subject_uri"],
        )
        dataset_hydration.validate_query_histories(histories_df)
        query_keys = queries_df.select("did", "query_hour")
        if positives_df.join(query_keys, on=["did", "query_hour"], how="anti").height:
            raise ValueError("Hydrated positives contain orphan query keys")
        history_keys = histories_df.select("did", "query_hour")
        if history_keys.join(query_keys, on=["did", "query_hour"], how="anti").height:
            raise ValueError("Hydrated histories contain orphan query keys")
        if query_keys.join(history_keys, on=["did", "query_hour"], how="anti").height:
            raise ValueError("Hydrated queries are missing history rows")
        actual_positive_counts = positives_df.group_by("did", "query_hour").agg(
            pl.len().cast(pl.UInt32).alias("positive_count")
        )
        if queries_df.join(
            actual_positive_counts,
            on=["did", "query_hour", "positive_count"],
            how="anti",
        ).height:
            raise ValueError("Hydrated query positive counts are incorrect")
        for column in ("emb_idx", "history_emb_indices"):
            frame = positives_df if column == "emb_idx" else histories_df
            values = (
                frame.get_column(column)
                if column == "emb_idx"
                else frame.get_column(column).explode(empty_as_null=True)
            ).drop_nulls()
            if values.len() and values.max() >= embedding_count:
                raise ValueError("Hydrated query artifacts contain an invalid embedding index")
        positive_author_indices = positives_df.get_column("author_idx")
        history_author_indices = histories_df.get_column(
            "history_author_indices"
        ).explode(empty_as_null=True).drop_nulls()
        for values in (positive_author_indices, history_author_indices):
            if values.len() and (
                values.min() < AUTHOR_UNK_IDX
                or values.max() > max_author_idx
            ):
                raise ValueError("Hydrated query artifacts contain an invalid author index")
        query_count += queries_df.height
        positive_count += positives_df.height
        history_query_count += histories_df.height

    negatives_lf = scan_parquet_artifact(hourly_negative_candidates_path)
    if negatives_lf.collect_schema() != pl.Schema(dataset_hydration.HOURLY_NEGATIVE_SCHEMA):
        raise ValueError("Unexpected hydrated negative-candidate schema")
    negative_stats = negatives_lf.select(
        pl.len().alias("row_count"),
        pl.col("emb_idx").max().alias("max_emb_idx"),
        (
            pl.col("query_hour").is_null()
            | pl.col("subject_uri").is_null()
        ).sum().alias("invalid_key_count"),
        (
            pl.col("selection_source").is_null()
            | ~pl.col("selection_source").is_in(["popular", "random"])
        )
        .sum()
        .alias("invalid_source_count"),
    ).collect(engine="streaming").row(0, named=True)
    duplicate_negative_key_count = int(
        negatives_lf.group_by("query_hour", "subject_uri")
        .len()
        .filter(pl.col("len") > 1)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if (
        negative_stats["invalid_key_count"]
        or negative_stats["invalid_source_count"]
        or duplicate_negative_key_count
    ):
        raise ValueError("Hydrated negatives contain invalid keys or source labels")
    if (
        negative_stats["max_emb_idx"] is not None
        and negative_stats["max_emb_idx"] >= embedding_count
    ):
        raise ValueError("Hydrated negatives contain an invalid embedding index")
    negative_author_stats = negatives_lf.select(
        pl.col("author_idx").min().alias("min_author_idx"),
        pl.col("author_idx").max().alias("max_author_idx"),
    ).collect(engine="streaming").row(0, named=True)
    if negative_author_stats["min_author_idx"] is not None and (
        negative_author_stats["min_author_idx"] < AUTHOR_UNK_IDX
        or negative_author_stats["max_author_idx"] > max_author_idx
    ):
        raise ValueError("Hydrated negatives contain an invalid author index")
    return {
        **authors_validation,
        "embedding_count": expected_emb_idx,
        "query_count": query_count,
        "positive_count": positive_count,
        "history_query_count": history_query_count,
        "negative_count": int(negative_stats["row_count"]),
    }
