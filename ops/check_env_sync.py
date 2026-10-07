#!/usr/bin/env python3
"""Keep GPU and CPU manifests and locks aligned except for PyTorch/CUDA builds."""
from __future__ import annotations

import json
import sys
import tomllib
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_MAIN = ROOT / "Pipfile"
ENV_CPU = ROOT / "Pipfile.cpu"
TORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}
GPU_ONLY_LOCK_PACKAGES = {
    "nvidia-cublas-cu12",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-cusparselt-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvtx-cu12",
    "triton",
}


def load_pipfile(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def check_env_sync(env_main: dict, env_cpu: dict) -> list[str]:
    errors = []
    normalized = []
    for filename, env, variant in (
        ("Pipfile", env_main, "cu124"),
        ("Pipfile.cpu", env_cpu, "cpu"),
    ):
        env = deepcopy(env)
        # Match the CUDA linker to the GPU runtime without adding it to CPU.
        if variant == "cu124":
            nvjitlink = env.get("packages", {}).pop("nvidia-nvjitlink-cu12", None)
            if nvjitlink != "==12.4.127":
                errors.append("Pipfile must pin nvidia-nvjitlink-cu12 to ==12.4.127.")
        elif any("nvidia-nvjitlink-cu12" in env.get(section, {}) for section in ("packages", "dev-packages")):
            errors.append("Pipfile.cpu must not include nvidia-nvjitlink-cu12.")
        sources = env.get("source", [])
        torch_sources = [source for source in sources if source["name"] == "pytorch"]
        if len(torch_sources) != 1:
            errors.append(f"{filename} must define one pytorch source.")
        for source in torch_sources:
            expected_url = f"https://download.pytorch.org/whl/{variant}"
            if source["url"] != expected_url:
                errors.append(f"{filename} pytorch source must use {expected_url}.")
            source["url"] = "https://download.pytorch.org/whl/"

        # Only the wheel build may differ; preserve all other package options
        # so changes to versions, extras, markers, or indexes still fail.
        for section in ("packages", "dev-packages"):
            for name, spec in env.get(section, {}).items():
                if name not in TORCH_PACKAGES:
                    continue
                if not isinstance(spec, dict) or spec.get("index") != "pytorch":
                    errors.append(f"{filename} {name} must use the pytorch index.")
                    continue
                version = spec.get("version", "")
                if "+" in version and not version.endswith(f"+{variant}"):
                    errors.append(f"{filename} {name} has an unexpected wheel variant: {version}.")
                spec["version"] = version.removesuffix(f"+{variant}")
        normalized.append(env)

    for section in ("source", "requires", "packages", "dev-packages"):
        if normalized[0].get(section) != normalized[1].get(section):
            errors.append(f"{section} mismatch between Pipfile and Pipfile.cpu.")
    return errors


def check_lock_sync(lock_main: dict, lock_cpu: dict) -> list[str]:
    errors = []
    for section in ("default", "develop"):
        normalized = []
        for filename, lock, variant in (
            ("Pipfile.lock", lock_main, "cu124"),
            ("Pipfile.cpu.lock", lock_cpu, "cpu"),
        ):
            versions = {name: spec["version"] for name, spec in lock.get(section, {}).items()}
            cuda_packages = versions.keys() & GPU_ONLY_LOCK_PACKAGES
            if variant == "cu124" and section == "default":
                for name in cuda_packages:
                    del versions[name]
            elif cuda_packages:
                errors.append(f"{filename} {section} has unexpected GPU packages: {sorted(cuda_packages)}.")
            for name in versions.keys() & TORCH_PACKAGES:
                version = versions[name]
                if not version.endswith(f"+{variant}"):
                    errors.append(f"{filename} {section} {name} has an unexpected wheel variant: {version}.")
                versions[name] = version.removesuffix(f"+{variant}")
            normalized.append(versions)

        # Wheel hashes legitimately differ. Compare resolved versions, including
        # transitive dependencies that do not appear directly in the manifests.
        main_versions, cpu_versions = normalized
        main_only = main_versions.keys() - cpu_versions.keys()
        cpu_only = cpu_versions.keys() - main_versions.keys()
        if main_only or cpu_only:
            errors.append(
                f"Lock {section} package mismatch: GPU only {sorted(main_only)}; CPU only {sorted(cpu_only)}."
            )
        for name in sorted(main_versions.keys() & cpu_versions.keys()):
            if main_versions[name] != cpu_versions[name]:
                errors.append(
                    f"Lock {section} {name} version mismatch: GPU {main_versions[name]}; CPU {cpu_versions[name]}."
                )
    return errors


def main() -> int:
    errors = check_env_sync(load_pipfile(ENV_MAIN), load_pipfile(ENV_CPU))
    lock_main = json.loads(ENV_MAIN.with_suffix(".lock").read_text())
    lock_cpu = json.loads(Path(f"{ENV_CPU}.lock").read_text())
    errors.extend(check_lock_sync(lock_main, lock_cpu))
    if errors:
        for error in errors:
            print(f"❌ {error}")
        return 1
    print("✅ GPU and CPU manifests and lockfiles are in sync (except PyTorch/CUDA requirements).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
