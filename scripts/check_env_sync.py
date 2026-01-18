#!/usr/bin/env python3
"""
Verify that environment.yml and environment.ci.yml stay aligned, allowing only the
expected CI differences (no nvidia channel, no CUDA package).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
ENV_MAIN = ROOT / "environment.yml"
ENV_CI = ROOT / "environment.ci.yml"

# Differences we intentionally allow between local and CI envs
ALLOWED_CHANNELS_ABSENT_IN_CI = {"nvidia"}
ALLOWED_DEPS_ABSENT_IN_CI = {"pytorch-cuda=12.1"}


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_deps(env: dict) -> Tuple[set[str], set[str]]:
    """Return (conda_deps, pip_deps) from an env spec."""
    conda_deps: set[str] = set()
    pip_deps: set[str] = set()
    for entry in env.get("dependencies", []):
        if isinstance(entry, str):
            conda_deps.add(entry)
        elif isinstance(entry, dict) and "pip" in entry:
            pip_entries: Iterable[str] = entry.get("pip", [])
            pip_deps.update(pip_entries)
    return conda_deps, pip_deps


def normalize_channels(channels: Iterable[str]) -> set[str]:
    return set(channels)


def main() -> int:
    env_main = load_yaml(ENV_MAIN)
    env_ci = load_yaml(ENV_CI)

    channels_main = normalize_channels(env_main.get("channels", []))
    channels_ci = normalize_channels(env_ci.get("channels", []))

    extra_channels_in_main = channels_main - channels_ci - ALLOWED_CHANNELS_ABSENT_IN_CI
    extra_channels_in_ci = channels_ci - channels_main

    conda_main, pip_main = split_deps(env_main)
    conda_ci, pip_ci = split_deps(env_ci)

    missing_in_ci = (conda_main - conda_ci) - ALLOWED_DEPS_ABSENT_IN_CI
    extra_in_ci = conda_ci - conda_main

    pip_diff_main = pip_main - pip_ci
    pip_diff_ci = pip_ci - pip_main

    errors = []
    if extra_channels_in_main or extra_channels_in_ci:
        errors.append(
            f"Channel mismatch. Extra in environment.yml (excluding allowed): {sorted(extra_channels_in_main)}; "
            f"extra in environment.ci.yml: {sorted(extra_channels_in_ci)}"
        )
    if missing_in_ci or extra_in_ci:
        errors.append(
            f"Conda dependency mismatch. Missing in CI (excluding allowed): {sorted(missing_in_ci)}; "
            f"extra in CI: {sorted(extra_in_ci)}"
        )
    if pip_diff_main or pip_diff_ci:
        errors.append(
            f"Pip dependency mismatch. Missing in CI: {sorted(pip_diff_main)}; extra in CI: {sorted(pip_diff_ci)}"
        )

    if errors:
        for err in errors:
            print(f"❌ {err}")
        return 1

    print("✅ environment.yml and environment.ci.yml are in sync (except allowed differences).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
