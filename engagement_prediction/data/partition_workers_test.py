import pytest

from engagement_prediction.data import partition_workers


def _result_worker(*, partition_id: int, returned_partition_id: int | None = None):
    return {
        "partition_id": (
            partition_id if returned_partition_id is None else returned_partition_id
        )
    }


def test_empty_partition_job_set_uses_no_workers():
    callbacks = []
    results, effective_worker_count = partition_workers.run_partition_jobs(
        worker=_result_worker,
        worker_kwargs=[],
        worker_count=4,
        on_result=callbacks.append,
    )

    assert results == []
    assert effective_worker_count == 0
    assert callbacks == []


def test_serial_partition_results_are_sorted_and_worker_count_is_capped():
    callbacks = []
    results, effective_worker_count = partition_workers.run_partition_jobs(
        worker=_result_worker,
        worker_kwargs=[{"partition_id": 2}, {"partition_id": 0}],
        worker_count=1,
        on_result=callbacks.append,
    )

    assert [result["partition_id"] for result in results] == [0, 2]
    assert [result["partition_id"] for result in callbacks] == [2, 0]
    assert effective_worker_count == 1


def test_partition_job_ids_must_be_unique():
    with pytest.raises(ValueError, match="unique partition_id"):
        partition_workers.run_partition_jobs(
            worker=_result_worker,
            worker_kwargs=[{"partition_id": 1}, {"partition_id": 1}],
            worker_count=1,
            on_result=lambda _result: None,
        )


def test_worker_must_return_the_submitted_partition_id():
    with pytest.raises(ValueError, match="expected 1"):
        partition_workers.run_partition_jobs(
            worker=_result_worker,
            worker_kwargs=[{"partition_id": 1, "returned_partition_id": 2}],
            worker_count=1,
            on_result=lambda _result: None,
        )
