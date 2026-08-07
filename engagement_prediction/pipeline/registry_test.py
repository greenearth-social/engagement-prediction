from engagement_prediction.pipeline import core, registry


def test_registered_stage_modules_load():
    for stage_name in registry.STAGE_SPECS:
        module_path, _stage_folder = registry.get_stage_spec(stage_name)

        assert callable(core.load_run_callable(module_path))
