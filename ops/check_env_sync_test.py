import json
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
    cpu = deepcopy(main)
    cpu["source"][1]["url"] = "https://download.pytorch.org/whl/cpu"
    for name in check_env_sync.TORCH_PACKAGES:
        cpu["packages"][name]["version"] = main["packages"][name]["version"].replace("+cu121", "+cpu")
    main["packages"]["nvidia-nvjitlink-cu12"] = "==12.1.105"
    return main, cpu


@pytest.mark.parametrize("include_variant", [True, False])
def test_accepts_gpu_cpu_wheels_without_mutating_inputs(environments, include_variant):
    main, cpu = environments
    if not include_variant:
        for env in (main, cpu):
            for name in check_env_sync.TORCH_PACKAGES:
                env["packages"][name]["version"] = env["packages"][name]["version"].split("+")[0]
    before = deepcopy(environments)

    assert check_env_sync.check_env_sync(main, cpu) == []
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
    main, cpu = environments
    cpu[section][name] = value

    assert check_env_sync.check_env_sync(main, cpu)


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
    main, cpu = environments
    if version is None:
        del main["packages"]["nvidia-nvjitlink-cu12"]
    else:
        main["packages"]["nvidia-nvjitlink-cu12"] = version

    assert "Pipfile must pin nvidia-nvjitlink-cu12 to ==12.1.105." in check_env_sync.check_env_sync(main, cpu)


@pytest.mark.parametrize("section", ["packages", "dev-packages"])
def test_rejects_nvjitlink_in_cpu_environment(environments, section):
    main, cpu = environments
    cpu[section]["nvidia-nvjitlink-cu12"] = "==12.1.105"

    assert "Pipfile.cpu must not include nvidia-nvjitlink-cu12." in check_env_sync.check_env_sync(main, cpu)


@pytest.mark.parametrize("url", ["https://download.pytorch.org/whl/cu121", "https://example.com/cpu"])
def test_rejects_wrong_cpu_source(environments, url):
    main, cpu = environments
    cpu["source"][1]["url"] = url

    assert any("pytorch source" in error for error in check_env_sync.check_env_sync(main, cpu))


def test_rejects_other_source_changes(environments):
    main, cpu = environments
    cpu["source"][0]["verify_ssl"] = False

    assert "source mismatch between Pipfile and Pipfile.cpu." in check_env_sync.check_env_sync(main, cpu)


@pytest.fixture
def locks():
    main = {
        "default": {
            "numpy": {"version": "==2.4.3"},
            "certifi": {"version": "==2026.7.22"},
            "torch": {"version": "==2.5.1+cu121", "hashes": ["sha256:gpu"]},
            "torchvision": {"version": "==0.20.1+cu121"},
            "torchaudio": {"version": "==2.5.1+cu121"},
        },
        "develop": {"pytest": {"version": "==9.0.2"}, "packaging": {"version": "==26.0"}},
    }
    cpu = deepcopy(main)
    for name in ("torch", "torchvision", "torchaudio"):
        cpu["default"][name]["version"] = main["default"][name]["version"].replace("+cu121", "+cpu")
    cpu["default"]["torch"]["hashes"] = ["sha256:cpu"]
    main["default"].update(
        {
            "nvidia-cublas-cu12": {"version": "==12.1.3.1"},
            "nvidia-cuda-cupti-cu12": {"version": "==12.1.105"},
            "nvidia-cuda-nvrtc-cu12": {"version": "==12.1.105"},
            "nvidia-cuda-runtime-cu12": {"version": "==12.1.105"},
            "nvidia-cudnn-cu12": {"version": "==9.1.0.70"},
            "nvidia-cufft-cu12": {"version": "==11.0.2.54"},
            "nvidia-curand-cu12": {"version": "==10.3.2.106"},
            "nvidia-cusolver-cu12": {"version": "==11.4.5.107"},
            "nvidia-cusparse-cu12": {"version": "==12.1.0.106"},
            "nvidia-nccl-cu12": {"version": "==2.21.5"},
            "nvidia-nvjitlink-cu12": {"version": "==12.1.105"},
            "nvidia-nvtx-cu12": {"version": "==12.1.105"},
            "triton": {"version": "==3.1.0"},
        }
    )
    return main, cpu


def test_accepts_expected_locked_builds_and_gpu_packages_without_mutation(locks):
    before = deepcopy(locks)

    assert check_env_sync.check_lock_sync(*locks) == []
    assert locks == before


@pytest.mark.parametrize("section,name", [("default", "certifi"), ("develop", "packaging")])
def test_rejects_locked_transitive_version_drift(locks, section, name):
    main, cpu = locks
    cpu[section][name]["version"] = "==999.0"

    assert any(f"{section} {name} version mismatch" in error for error in check_env_sync.check_lock_sync(main, cpu))


@pytest.mark.parametrize("lock_index", [0, 1])
@pytest.mark.parametrize("section,name", [("default", "numpy"), ("develop", "pytest")])
def test_rejects_missing_locked_shared_packages(locks, lock_index, section, name):
    del locks[lock_index][section][name]

    assert any(f"Lock {section} package mismatch" in error for error in check_env_sync.check_lock_sync(*locks))


@pytest.mark.parametrize(
    "lock_index,section,name",
    [
        (0, "default", "nvidia-unexpected-cu12"),
        (1, "default", "unexpected-package"),
        (1, "default", "nvidia-nvjitlink-cu12"),
        (1, "default", "triton"),
        (0, "develop", "nvidia-nvjitlink-cu12"),
        (1, "develop", "triton"),
    ],
)
def test_rejects_unexpected_locked_packages_and_cpu_cuda(locks, lock_index, section, name):
    locks[lock_index][section][name] = {"version": "==1.0"}

    assert check_env_sync.check_lock_sync(*locks)


@pytest.mark.parametrize("name", ["torch", "torchvision", "torchaudio"])
@pytest.mark.parametrize("version", ["==9.0+cpu", "==2.5.1+cu128", "==2.5.1"])
def test_rejects_locked_torch_base_or_build_drift(locks, name, version):
    main, cpu = locks
    cpu["default"][name]["version"] = version

    assert check_env_sync.check_lock_sync(main, cpu)


def test_rejects_matching_wrong_locked_torch_builds(locks):
    for lock in locks:
        lock["default"]["torch"]["version"] = "==2.5.1+cu128"

    assert any("unexpected wheel variant" in error for error in check_env_sync.check_lock_sync(*locks))


@pytest.mark.parametrize("drift", ["manifest", "lock"])
def test_main_reads_manifests_and_locks_and_fails_for_drift(tmp_path, monkeypatch, capsys, locks, drift):
    paths = [tmp_path / "Pipfile", tmp_path / "Pipfile.cpu"]
    for path, variant in zip(paths, ("cu121", "cpu")):
        path.write_text(
            '[[source]]\nname = "pytorch"\n'
            f'url = "https://download.pytorch.org/whl/{variant}"\n'
            '[requires]\npython_full_version = "3.11.15"\n'
            '[packages]\n'
            + ('nvidia-nvjitlink-cu12 = "==12.1.105"\n' if variant == "cu121" else "")
        )
    monkeypatch.setattr(check_env_sync, "ENV_MAIN", paths[0])
    monkeypatch.setattr(check_env_sync, "ENV_CPU", paths[1])
    lock_paths = [tmp_path / "Pipfile.lock", tmp_path / "Pipfile.cpu.lock"]
    for path, lock in zip(lock_paths, locks):
        path.write_text(json.dumps(lock))

    assert check_env_sync.main() == 0
    if drift == "manifest":
        paths[1].write_text(paths[1].read_text().replace("3.11.15", "3.11.16"))
        expected_error = "requires mismatch"
    else:
        locks[1]["default"]["certifi"]["version"] = "==999.0"
        lock_paths[1].write_text(json.dumps(locks[1]))
        expected_error = "certifi version mismatch"
    assert check_env_sync.main() == 1
    assert expected_error in capsys.readouterr().out
