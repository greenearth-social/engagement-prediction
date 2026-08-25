from datetime import datetime, timezone

import pytest
import torch

from engagement_prediction.training.ranking import (
    calc_baseline_rank_metrics_for_batch,
    finalize_rank_metrics,
    finalize_zero_history_rank_metrics,
    rank_metric_sums_for_batch,
    ranking_rows_for_batch,
    stage_info_metric_lines,
    zero_history_rank_metric_sums_for_batch,
)


def test_rank_metrics_report_ndcg_and_map_without_recall():
    ranked_labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    sums, user_count = rank_metric_sums_for_batch(ranked_labels, [1, 2])
    metrics = finalize_rank_metrics(sums, user_count)

    assert metrics["ndcg@1"] == pytest.approx(0.5)
    assert metrics["ndcg@2"] == pytest.approx(0.8154648768)
    assert metrics["mean_average_precision"] == pytest.approx(0.75)
    assert not any("recall" in key for key in metrics)


def test_random_baseline_reports_ndcg_and_map_without_recall():
    labels = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

    sums, user_count = calc_baseline_rank_metrics_for_batch(labels, [1, 3])
    metrics = finalize_rank_metrics(sums, user_count)

    assert metrics["ndcg@1"] == pytest.approx(0.5)
    assert metrics["ndcg@3"] == pytest.approx(0.7906795144)
    assert metrics["mean_average_precision"] == pytest.approx(0.7083333731)
    assert not any("recall" in key for key in metrics)


def test_zero_history_metrics_and_stage_lines_have_no_recall():
    batch = {"history_mask": torch.tensor([[False, False], [True, False]])}
    ranked_labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    sums, user_count = zero_history_rank_metric_sums_for_batch(batch, ranked_labels, [1])
    metrics = finalize_zero_history_rank_metrics(sums, user_count)
    metrics["rank_metric_user_count"] = 2
    lines = stage_info_metric_lines({"train": metrics})

    assert metrics["zero_history_ndcg@1"] == pytest.approx(1.0)
    assert metrics["zero_history_mean_average_precision"] == pytest.approx(1.0)
    assert not any("recall" in key for key in metrics)
    assert not any("recall" in line for line in lines)


def test_ranking_rows_include_ndcg_without_recall():
    hour = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    batch = {
        "user_id": ["u1"],
        "bucket": hour,
        "history_mask": torch.tensor([[True, False]]),
    }
    scores = torch.tensor([[0.9, 0.1, 0.8]])
    labels = torch.tensor([[1.0, 0.0, 1.0]])

    rows = ranking_rows_for_batch(batch, scores, labels, [1, 2])

    assert rows[0]["ndcg@1"] == pytest.approx(1.0)
    assert rows[0]["ndcg@2"] == pytest.approx(1.0)
    assert not any("recall" in key for key in rows[0])
