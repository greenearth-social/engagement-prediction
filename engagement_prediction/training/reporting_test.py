from unittest.mock import Mock

import numpy as np
import pytest

from engagement_prediction.experiment_tracking.base import NoOpExperimentTracker
from engagement_prediction.training.reporting import (
    write_bst_training_history_plot,
    write_history_length_plots,
)


def test_write_bst_training_history_plot(tmp_path):
    output_path = tmp_path / "training_history.png"
    history = {
        "train_loss": [2.0, 1.0],
        "val_loss": [2.1, 1.1],
        "val_unseen_loss": [2.2, 1.2],
        "train_ndcg@30": [0.2, 0.3],
        "val_ndcg@30": [0.1, 0.2],
        "val_unseen_ndcg@30": [0.15, 0.25],
    }

    write_bst_training_history_plot(history, output_path, best_epoch=2)

    assert output_path.stat().st_size > 0


@pytest.fixture
def history_length_final_metrics():
    labels = ["0", "1", "2", "3–4", "5–8", "9–16", "17–32", ">32"]
    upper_bounds = [0, 1, 2, 4, 8, 16, 32, None]
    lower_bounds = [0, 1, 2, 3, 5, 9, 17, 33]
    metrics = {}
    for split, offset in (("val", 0), ("val_unseen_users", 1)):
        metrics[split] = {
            "history_length_breakdown": [
                {
                    "label": label,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "query_count": 0 if index == 2 else index + offset + 1,
                    "ndcg@1": None if index == 2 else 0.1 + offset * 0.1,
                    "ndcg@30": None if index == 2 else 0.4 + offset * 0.1,
                }
                for index, (label, lower_bound, upper_bound) in enumerate(
                    zip(labels, lower_bounds, upper_bounds)
                )
            ]
        }
    return metrics


def test_write_history_length_plots(tmp_path, history_length_final_metrics):
    import matplotlib.pyplot as plt

    tracker = Mock()
    figures_before = plt.get_fignums()
    output_paths = write_history_length_plots(
        final_metrics=history_length_final_metrics,
        metrics_top_ks=[1, 30],
        output_dir=tmp_path / "plots",
        best_epoch=7,
        tracker=tracker,
    )

    assert output_paths == {
        "ndcg@1": tmp_path / "plots" / "history_length_ndcg_at_1.png",
        "ndcg@30": tmp_path / "plots" / "history_length_ndcg_at_30.png",
    }
    assert tracker.log_plot.call_count == 2
    for k, call in zip((1, 30), tracker.log_plot.call_args_list):
        path = output_paths[f"ndcg@{k}"]
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert call.kwargs["title"] == "NDCG by history length"
        assert call.kwargs["series"] == f"ndcg@{k}"
        assert call.kwargs["iteration"] == 7
        metric_axis, count_axis = call.kwargs["figure"].axes
        assert "best epoch 7" in metric_axis.get_title()
        assert metric_axis.get_ylabel() == f"NDCG@{k}"
        assert metric_axis.get_ylim()[0] < 0
        assert metric_axis.get_ylim()[1] > 1
        assert all(0 <= tick <= 1 for tick in metric_axis.get_yticks())
        assert all(float(tick).is_integer() for tick in count_axis.get_yticks())
        assert [line.get_label() for line in metric_axis.lines] == [
            "Validation",
            "Validation Unseen Users",
        ]
        assert [tick.get_text() for tick in count_axis.get_xticklabels()] == [
            bucket["label"]
            for bucket in history_length_final_metrics["val"]["history_length_breakdown"]
        ]
        for index, split in enumerate(("val", "val_unseen_users")):
            buckets = history_length_final_metrics[split]["history_length_breakdown"]
            expected_values = [
                np.nan if bucket[f"ndcg@{k}"] is None else bucket[f"ndcg@{k}"]
                for bucket in buckets
            ]
            np.testing.assert_allclose(
                metric_axis.lines[index].get_ydata(), expected_values, equal_nan=True
            )
            assert metric_axis.lines[index].get_marker() == "o"
            assert [bar.get_height() for bar in count_axis.containers[index]] == [
                bucket["query_count"] for bucket in buckets
            ]
    assert plt.get_fignums() == figures_before


@pytest.mark.parametrize("tracker", [None, NoOpExperimentTracker()])
def test_write_history_length_plots_without_tracking(
    tmp_path, history_length_final_metrics, tracker
):
    paths = write_history_length_plots(
        final_metrics=history_length_final_metrics,
        metrics_top_ks=[30],
        output_dir=tmp_path,
        best_epoch=2,
        tracker=tracker,
    )

    assert paths["ndcg@30"].stat().st_size > 0
