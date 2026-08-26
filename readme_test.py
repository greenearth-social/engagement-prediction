from pathlib import Path


def test_readme_describes_current_pipeline_and_new_ranker_surface():
    readme = (Path(__file__).resolve().parent / "README.md").read_text()

    assert "engagement_prediction/stages/source_metadata.py" in readme
    assert "engagement_prediction/stages/query_selection.py" in readme
    assert "engagement_prediction/stages/user_history.py" in readme
    assert "engagement_prediction/stages/negative_selection.py" in readme
    assert "engagement_prediction/stages/post_liker_history.py" in readme
    assert "engagement_prediction/stages/author_statistics.py" in readme
    assert "engagement_prediction/stages/dataset_hydration.py" in readme
    assert "engagement_prediction/stages/train_bst_ranker.py" in readme
    assert "engagement_prediction/stages/train_two_tower.py" in readme
    assert "engagement_prediction/data/datasets.py" in readme
    assert "engagement_prediction/data/ingex.py" in readme
    assert "engagement_prediction/data/source_metadata.py" in readme
    assert "engagement_prediction/pipeline/logging.py" in readme
    assert "engagement_prediction/training/runtime.py" in readme
    assert "engagement_prediction/training/bst_ranker.py" in readme
    assert "engagement_prediction/training/two_tower.py" in readme
    assert "engagement_prediction/models/two_tower.py" in readme
    assert "legacy/evaluation/" in readme
    assert "`legacy/`" in readme
    assert "engagement_prediction/pipeline/{core.py,dependencies.py,registry.py}" in readme
    assert "utils/" not in readme
    assert "compare-rankers" in readme
    assert "--model-type bst-ranker" in readme
    assert "--model-type two-tower" in readme
    assert "--output-embedding-dim" in readme
    assert "08_train_two_tower" in readme
    assert "hourly_candidates/" in readme
    assert "post_metadata/" in readme
    assert "--prior-00-source-metadata" in readme
    assert "--prior-03-post-selection" in readme
    assert "post_liker_events/" in readme
    assert "--prior-04-negative-selection" in readme
    assert "authors/" in readme
    assert "--prior-05-post-liker-history" in readme
    assert "embeddings.npy" in readme
    assert "hourly_negative_candidates/" in readme
    assert "--prior-06-author-statistics" in readme
    assert "--prior-07-dataset-hydration" in readme
    assert "train_mlp" not in readme
    assert "--prior-03-train" not in readme
    assert "--prior-01-get-data" not in readme
    assert "post_selection_partition_count" not in readme
    assert "DIN" not in readme
    assert "stage_featurize.py" not in readme
    assert "stage_relevel_uniform.py" not in readme
