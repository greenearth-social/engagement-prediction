from copy import deepcopy

import pytest

from ops import check_env_sync


@pytest.fixture
def environments():
    main = {
        "source": [
            {"name": "pypi", "url": "https://pypi.org/simple", "verify_ssl": True},
            {"name": "pytorch", "url": "https://download.pytorch.org/whl/cu121", "verify_ssl": True},
        ],
        "requires": {"python_full_version": "3.11.15"},
        "packages": {
            "numpy": "==2.4.3",
            "torch": {"version": "==2.5.1+cu121", "index": "pytorch"},
            "torchvision": {"version": "==0.20.1+cu121", "index": "pytorch"},
            "torchaudio": {"version": "==2.5.1+cu121", "index": "pytorch"},
        },
        "dev-packages": {"pytest": "==9.0.2"},
    }
    ci = deepcopy(main)
    ci["source"][1]["url"] = "https://download.pytorch.org/whl/cpu"
    for name in check_env_sync.TORCH_PACKAGES:
        ci["packages"][name]["version"] = main["packages"][name]["version"].replace("+cu121", "+cpu")
    main["packages"]["nvidia-nvjitlink-cu12"] = "==12.1.105"
    return main, ci


@pytest.mark.parametrize("include_variant", [True, False])
def test_accepts_gpu_cpu_wheels_without_mutating_inputs(environments, include_variant):
    main, ci = environments
    if not include_variant:
        for env in (main, ci):
            for name in check_env_sync.TORCH_PACKAGES:
                env["packages"][name]["version"] = env["packages"][name]["version"].split("+")[0]
    before = deepcopy(environments)

    assert check_env_sync.check_env_sync(main, ci) == []
    assert environments == before


@pytest.mark.parametrize(
    "section,name,value",
    [
        ("packages", "numpy", "==2.4.4"),
        ("packages", "pandas", "==3.0.1"),
        ("packages", "torch", {"version": "==2.6.0+cpu", "index": "pytorch"}),
        ("packages", "torch", {"version": "==2.5.1+cpu", "index": "pypi"}),
        ("packages", "torch", {"version": "==2.5.1+cu128", "index": "pytorch"}),
        ("dev-packages", "pytest", "==8.4.2"),
        ("requires", "python_full_version", "3.11.16"),
    ],
)
def test_rejects_dependency_and_python_drift(environments, section, name, value):
    main, ci = environments
    ci[section][name] = value

    assert check_env_sync.check_env_sync(main, ci)


@pytest.mark.parametrize("env_index", [0, 1])
def test_rejects_missing_shared_dependency(environments, env_index):
    del environments[env_index]["packages"]["numpy"]

    assert check_env_sync.check_env_sync(*environments)


@pytest.mark.parametrize("env_index", [0, 1])
def test_rejects_other_cuda_only_dependencies(environments, env_index):
    environments[env_index]["packages"]["nvidia-cublas-cu12"] = "==12.1.3.1"

    assert check_env_sync.check_env_sync(*environments)


@pytest.mark.parametrize("version", [None, "==12.9.86"])
def test_requires_original_gpu_nvjitlink_version(environments, version):
    main, ci = environments
    if version is None:
        del main["packages"]["nvidia-nvjitlink-cu12"]
    else:
        main["packages"]["nvidia-nvjitlink-cu12"] = version

    assert "Pipfile must pin nvidia-nvjitlink-cu12 to ==12.1.105." in check_env_sync.check_env_sync(main, ci)


@pytest.mark.parametrize("section", ["packages", "dev-packages"])
def test_rejects_nvjitlink_in_cpu_environment(environments, section):
    main, ci = environments
    ci[section]["nvidia-nvjitlink-cu12"] = "==12.1.105"

    assert "Pipfile.ci must not include nvidia-nvjitlink-cu12." in check_env_sync.check_env_sync(main, ci)


@pytest.mark.parametrize("url", ["https://download.pytorch.org/whl/cu121", "https://example.com/cpu"])
def test_rejects_wrong_cpu_source(environments, url):
    main, ci = environments
    ci["source"][1]["url"] = url

    assert any("pytorch source" in error for error in check_env_sync.check_env_sync(main, ci))


def test_rejects_other_source_changes(environments):
    main, ci = environments
    ci["source"][0]["verify_ssl"] = False

    assert "source mismatch between Pipfile and Pipfile.ci." in check_env_sync.check_env_sync(main, ci)


def test_main_reads_pipfiles_and_returns_failure_for_drift(tmp_path, monkeypatch, capsys):
    paths = [tmp_path / "Pipfile", tmp_path / "Pipfile.ci"]
    for path, variant in zip(paths, ("cu121", "cpu")):
        path.write_text(
            '[[source]]\nname = "pytorch"\n'
            f'url = "https://download.pytorch.org/whl/{variant}"\n'
            '[requires]\npython_full_version = "3.11.15"\n'
            '[packages]\n'
            + ('nvidia-nvjitlink-cu12 = "==12.1.105"\n' if variant == "cu121" else "")
        )
    monkeypatch.setattr(check_env_sync, "ENV_MAIN", paths[0])
    monkeypatch.setattr(check_env_sync, "ENV_CI", paths[1])

    assert check_env_sync.main() == 0
    paths[1].write_text(paths[1].read_text().replace("3.11.15", "3.11.16"))
    assert check_env_sync.main() == 1
    assert "requires mismatch" in capsys.readouterr().out
