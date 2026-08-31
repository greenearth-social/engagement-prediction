"""Bounded process-pool execution for independent artifact partitions."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from typing import Any, Callable


PartitionResult = dict[str, Any]


def _validate_result(
    result: PartitionResult,
    *,
    expected_partition_id: int,
) -> PartitionResult:
    """Enforce the compact result contract before parent-side aggregation."""

    if not isinstance(result, dict) or "partition_id" not in result:
        raise ValueError("Partition worker results must contain partition_id")
    actual_partition_id = int(result["partition_id"])
    if actual_partition_id != expected_partition_id:
        raise ValueError(
            "Partition worker returned partition_id "
            f"{actual_partition_id}, expected {expected_partition_id}"
        )
    return result


def run_partition_jobs(
    *,
    worker: Callable[..., PartitionResult],
    worker_kwargs: list[dict[str, Any]],
    worker_count: int,
    on_result: Callable[[PartitionResult], None],
) -> tuple[list[PartitionResult], int]:
    """Run file-backed partition jobs with bounded spawn-based concurrency.

    Workers must be module-level callables and own disjoint output files. Only
    compact result dictionaries cross the process boundary. A one-worker path
    stays in the parent process for low-memory runs, debugging, and tests.
    Results are returned in partition-ID order regardless of completion order.
    """

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if not worker_kwargs:
        return [], 0

    partition_ids = [int(kwargs["partition_id"]) for kwargs in worker_kwargs]
    if len(set(partition_ids)) != len(partition_ids):
        raise ValueError("Partition jobs must have unique partition_id values")

    effective_worker_count = min(worker_count, len(worker_kwargs))
    results: list[PartitionResult] = []
    if effective_worker_count == 1:
        for kwargs in worker_kwargs:
            result = _validate_result(
                worker(**kwargs),
                expected_partition_id=int(kwargs["partition_id"]),
            )
            results.append(result)
            on_result(result)
    else:
        # Polars owns a native thread pool, so use spawn rather than forking an
        # already-initialized process. Each child reads and writes only its
        # assigned bounded partition.
        executor = ProcessPoolExecutor(
            max_workers=effective_worker_count,
            mp_context=multiprocessing.get_context("spawn"),
        )
        futures = {
            executor.submit(worker, **kwargs): int(kwargs["partition_id"])
            for kwargs in worker_kwargs
        }
        try:
            for future in as_completed(futures):
                result = _validate_result(
                    future.result(),
                    expected_partition_id=futures[future],
                )
                results.append(result)
                on_result(result)
        except BaseException:
            # Completed parts remain available in the caller's partial output,
            # but queued jobs should not continue writing after failure.
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    results.sort(key=lambda result: int(result["partition_id"]))
    return results, effective_worker_count
