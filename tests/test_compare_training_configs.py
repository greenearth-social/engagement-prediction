import json

from ops import compare_training_configs


def _write_config(parent_dir, run_name, config):
    run_dir = parent_dir / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "training_config.json").write_text(json.dumps(config) + "\n")


def test_compare_training_configs_prints_differences(tmp_path, capsys):
    parent_dir = tmp_path / "03_train"
    _write_config(
        parent_dir,
        "run_a",
        {
            "learning_rate": 0.001,
            "prediction_hidden_dims": [64, 32, 16],
            "bst_popularity_log_mean": 1.0,
            "bst_popularity_log_std": 2.0,
            "nested": {"enabled": True},
        },
    )
    _write_config(
        parent_dir,
        "run_b",
        {
            "learning_rate": 0.002,
            "prediction_hidden_dims": [64, 48, 16],
            "bst_popularity_log_mean": 3.0,
            "bst_popularity_log_std": 4.0,
            "new_key": "present",
            "nested": {"enabled": False},
        },
    )

    exit_code = compare_training_configs.main([
        "--parent-dir",
        str(parent_dir),
        "run_a",
        "run_b",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "learning_rate:" in captured.out
    assert "prediction_hidden_dims[1]:" in captured.out
    assert "nested.enabled:" in captured.out
    assert "new_key:" in captured.out
    assert "bst_popularity_log_mean" not in captured.out
    assert "bst_popularity_log_std" not in captured.out
    assert "<missing>" in captured.out


def test_compare_training_configs_prints_no_differences(tmp_path, capsys):
    parent_dir = tmp_path / "03_train"
    config = {"learning_rate": 0.001, "batch_size": 256}
    _write_config(parent_dir, "run_a", config)
    _write_config(parent_dir, "run_b", config)

    exit_code = compare_training_configs.main([
        "--parent-dir",
        str(parent_dir),
        "run_a",
        "run_b",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "No differences found."


def test_compare_training_configs_reports_missing_file(tmp_path, capsys):
    exit_code = compare_training_configs.main([
        "--parent-dir",
        str(tmp_path),
        "run_a",
        "run_b",
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Missing training config" in captured.err
