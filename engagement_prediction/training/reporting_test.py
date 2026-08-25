from engagement_prediction.training.reporting import write_bst_training_history_plot


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
