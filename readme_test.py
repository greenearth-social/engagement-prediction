from pathlib import Path


def test_readme_describes_current_pipeline_and_new_ranker_surface():
    readme = (Path(__file__).resolve().parent / "README.md").read_text()

    assert "engagement_prediction/stages/query_selection.py" in readme
    assert "engagement_prediction/data/ingex.py" in readme
    assert "utils/02_user_history/stage_generate_user_history.py" in readme
    assert "utils/03_train/stage_train_bst_ranker.py" in readme
    assert "utils/04_evaluate/stage_evaluate.py" in readme
    assert "engagement_prediction/pipeline/{core.py,dependencies.py,registry.py}" in readme
    assert "utils/pipeline/registry.py" not in readme
    assert "compare-rankers" in readme
    assert "--model-type bst-ranker" in readme
    assert "prior_like_age_hours_at_bucket_start" in readme
    assert "DIN" not in readme
    assert "stage_featurize.py" not in readme
    assert "stage_relevel_uniform.py" not in readme
