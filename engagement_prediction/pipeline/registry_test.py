from engagement_prediction.pipeline import core, registry


def test_registry_contains_only_canonical_stages():
    assert set(registry.STAGE_SPECS) == {
        "source_metadata",
        "query_selection",
        "user_history",
        "post_selection",
        "negative_selection",
        "post_liker_history",
        "author_statistics",
        "dataset_hydration",
        "train_two_tower",
        "train_bst_ranker",
    }
    assert all(
        module_path.startswith("engagement_prediction/stages/")
        for module_path, _stage_folder in registry.STAGE_SPECS.values()
    )


def test_registered_stage_modules_load():
    for stage_name in registry.STAGE_SPECS:
        module_path, _stage_folder = registry.get_stage_spec(stage_name)

        assert callable(core.load_run_callable(module_path))
