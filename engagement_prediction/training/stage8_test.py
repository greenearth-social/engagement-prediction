import json
import logging

import pytest
import torch

from engagement_prediction.training.stage8 import (
    build_training_result_payload,
    build_training_summary,
    evaluate_listwise_splits,
    upload_reproducibility_artifacts,
    write_stage_info,
    write_training_result_files,
)


def _training_results():
    return {
        "primary_metric_name": "val_unseen_ndcg@30",
        "best_val_metric": 0.5,
        "best_val_loss": 1.25,
        "best_epoch": 2,
        "epochs_completed": 3,
        "stopped_early": True,
        "patience_counter": 2,
        "baseline_metrics": {"train": {"ndcg@30": 0.1}},
        "history": {"train_loss": [2.0, 1.5]},
    }


def test_evaluate_listwise_splits_uses_split_specific_limits():
    calls = []

    def epoch_runner(**kwargs):
        calls.append(kwargs)
        return 1.0, {"loss": 1.0, "ndcg@30": 0.25}, {}

    loaders = {"train": object(), "val": object(), "val_unseen_users": object()}
    results = evaluate_listwise_splits(
        model=torch.nn.Linear(1, 1),
        epoch_runner=epoch_runner,
        device="cpu",
        loaders=loaders,
        disable_progress=True,
        gradient_clip_max_norm=1.0,
        metrics_top_ks=[30],
        max_batches_by_split={"train": 4},
    )

    assert results == {
        split: {"loss": 1.0, "ndcg@30": 0.25} for split in loaders
    }
    assert [call["split_name"] for call in calls] == [
        "Final train",
        "Final val",
        "Final val_unseen_users",
    ]
    assert [call["max_batches"] for call in calls] == [4, None, None]
    assert all(call["train"] is False for call in calls)
    assert all(call["calc_baseline_metrics"] is False for call in calls)


def test_evaluate_listwise_splits_rejects_unknown_limit():
    with pytest.raises(ValueError, match="unknown splits"):
        evaluate_listwise_splits(
            model=torch.nn.Linear(1, 1),
            epoch_runner=lambda **kwargs: (0.0, {}, {}),
            device="cpu",
            loaders={"val": object()},
            disable_progress=True,
            gradient_clip_max_norm=1.0,
            metrics_top_ks=[30],
            max_batches_by_split={"train": 1},
        )


def test_result_payload_and_summary_preserve_model_specific_fields(tmp_path):
    final_metrics = {"val": {"loss": 1.0, "ndcg@30": 0.4}}
    result_payload = build_training_result_payload(
        training_results=_training_results(),
        final_metrics=final_metrics,
        split_query_counts={"train": 10, "val": 3},
        torchscript_export={"export_count": 2},
        author_map={"author_count": 5},
        clearml_publication={"status": "complete"},
        local_pipeline_runtime_seconds=4.0,
        runtime_seconds=5.0,
        extra_fields={"output_embedding_dim": 128},
    )
    summary = build_training_summary(
        training_config={"batch_size": 32},
        stage7_dir=tmp_path / "stage7",
        bundle_path=tmp_path / "bundle",
        model_config={"model_type": "two-tower"},
        result_payload=result_payload,
        outputs={"checkpoint_path": "checkpoint.pth"},
        runtime_seconds=5.0,
        extra_sections={"model_specific": {"enabled": True}},
    )

    assert result_payload["output_embedding_dim"] == 128
    assert result_payload["training_history"] == {"train_loss": [2.0, 1.5]}
    assert summary["results"]["output_embedding_dim"] == 128
    assert "training_history" not in summary["results"]
    assert summary["model_specific"] == {"enabled": True}

    with pytest.raises(ValueError, match="shared training-result keys"):
        build_training_result_payload(
            training_results=_training_results(),
            final_metrics=final_metrics,
            split_query_counts={},
            torchscript_export={},
            author_map={},
            clearml_publication={},
            local_pipeline_runtime_seconds=0.0,
            runtime_seconds=0.0,
            extra_fields={"best_epoch": 99},
        )


def test_common_result_files_and_stage_info_are_published(tmp_path):
    result_path = tmp_path / "training_results.json"
    summary_path = tmp_path / "summary.json"
    write_training_result_files(
        training_results_path=result_path,
        result_payload={"best_epoch": 2},
        summary_path=summary_path,
        summary={"runtime_seconds": 3.0},
    )
    stage_info_path = tmp_path / "stage_info.txt"
    write_stage_info(
        stage_info_path=stage_info_path,
        lines=["stage: train_test"],
        final_metrics={"val": {"loss": 1.25, "ndcg@30": 0.5}},
        primary_metric_key="ndcg@30",
    )

    assert json.loads(result_path.read_text()) == {"best_epoch": 2}
    assert json.loads(summary_path.read_text()) == {"runtime_seconds": 3.0}
    assert stage_info_path.read_text().splitlines() == [
        "stage: train_test",
        "val_loss: 1.250000",
        "val_ndcg@30: 0.500000",
    ]
    assert not list(tmp_path.glob("*.partial"))


def test_reproducibility_upload_is_shared_and_warns_on_failure(tmp_path, caplog):
    class Tracker:
        id = "task-id"

        def __init__(self):
            self.calls = []

        def log_file_artifact(self, name, path):
            self.calls.append((name, path))
            return name != "failed"

    tracker = Tracker()
    logger = logging.getLogger("stage8-test")
    with caplog.at_level(logging.WARNING):
        upload_reproducibility_artifacts(
            tracker=tracker,
            logger=logger,
            artifact_paths={"ok": tmp_path / "ok", "failed": tmp_path / "failed"},
        )

    assert [name for name, _ in tracker.calls] == ["ok", "failed"]
    assert "did not upload artifact 'failed'" in caplog.text
