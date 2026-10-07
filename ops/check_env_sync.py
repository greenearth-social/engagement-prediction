#!/usr/bin/env python3
"""Keep GPU and CI Pipfiles aligned except for their PyTorch/CUDA requirements."""
from __future__ import annotations

import sys
import tomllib
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_MAIN = ROOT / "Pipfile"
ENV_CI = ROOT / "Pipfile.ci"
TORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}


def load_pipfile(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def check_env_sync(env_main: dict, env_ci: dict) -> list[str]:
    errors = []
    normalized = []
    for filename, env, variant in (
        ("Pipfile", env_main, "cu121"),
        ("Pipfile.ci", env_ci, "cpu"),
    ):
        env = deepcopy(env)
        # Preserve the existing CUDA linker version without adding it to CPU CI.
        if variant == "cu121":
            nvjitlink = env.get("packages", {}).pop("nvidia-nvjitlink-cu12", None)
            if nvjitlink != "==12.1.105":
                errors.append("Pipfile must pin nvidia-nvjitlink-cu12 to ==12.1.105.")
        elif any("nvidia-nvjitlink-cu12" in env.get(section, {}) for section in ("packages", "dev-packages")):
            errors.append("Pipfile.ci must not include nvidia-nvjitlink-cu12.")
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
            errors.append(f"{section} mismatch between Pipfile and Pipfile.ci.")
    return errors


def main() -> int:
    errors = check_env_sync(load_pipfile(ENV_MAIN), load_pipfile(ENV_CI))
    if errors:
        for error in errors:
            print(f"❌ {error}")
        return 1
    print("✅ Pipfile and Pipfile.ci are in sync (except PyTorch/CUDA requirements).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
