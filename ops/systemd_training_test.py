import os
from pathlib import Path
import shlex
import subprocess

import pytest


PREPARE_CHECKOUT = Path(__file__).parent / "systemd" / "prepare-checkout.sh"


def git(path, *args):
    return subprocess.run(
        ["git", *args], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def commit(path, filename, contents):
    (path / filename).write_text(contents)
    git(path, "add", filename)
    git(path, "commit", "-m", f"Update {filename}")
    return git(path, "rev-parse", "HEAD")


@pytest.fixture
def checkouts(tmp_path, monkeypatch):
    # Real local repositories exercise Git's fast-forward behavior without SSH
    # or the developer's identity, signing, and hook configuration.
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Training test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "training-test@example.invalid")

    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "--initial-branch=main")
    commit(origin, ".gitignore", "config.local.yml\n.venv/\n")
    commit(origin, "model.txt", "initial\n")
    checkout = tmp_path / "training"
    git(tmp_path, "clone", "--branch", "main", str(origin), str(checkout))
    return origin, checkout


def prepare(checkout):
    return subprocess.run(
        ["bash", str(PREPARE_CHECKOUT.resolve())],
        cwd=checkout,
        capture_output=True,
        text=True,
    )


def test_fast_forwards_to_fetched_main_and_logs_revision(checkouts):
    origin, checkout = checkouts
    expected = commit(origin, "model.txt", "new main\n")

    result = prepare(checkout)

    assert result.returncode == 0, result.stderr
    assert git(checkout, "rev-parse", "HEAD") == expected
    assert (checkout / "model.txt").read_text() == "new main\n"
    assert expected in result.stdout


def test_accepts_current_main_with_ignored_config_and_environment(checkouts):
    _, checkout = checkouts
    expected = git(checkout, "rev-parse", "HEAD")
    (checkout / "config.local.yml").write_text("dry_run: true\n")
    (checkout / ".venv").mkdir()
    (checkout / ".venv" / "marker").write_text("environment\n")

    result = prepare(checkout)

    assert result.returncode == 0, result.stderr
    assert git(checkout, "rev-parse", "HEAD") == expected
    assert (checkout / "config.local.yml").read_text() == "dry_run: true\n"
    assert (checkout / ".venv" / "marker").read_text() == "environment\n"


@pytest.mark.parametrize("change", ["tracked", "staged", "untracked"])
def test_rejects_dirty_checkout_without_updating_or_discarding_changes(checkouts, change):
    origin, checkout = checkouts
    expected = git(checkout, "rev-parse", "HEAD")
    commit(origin, "new.txt", "remote update\n")
    filename = "local.txt" if change == "untracked" else "model.txt"
    (checkout / filename).write_text("local work\n")
    if change == "staged":
        git(checkout, "add", filename)

    result = prepare(checkout)

    assert result.returncode != 0
    assert git(checkout, "rev-parse", "HEAD") == expected
    assert (checkout / filename).read_text() == "local work\n"
    assert not (checkout / "new.txt").exists()


@pytest.mark.parametrize("branch", ["experiment", "detached"])
def test_rejects_other_branches_and_detached_head(checkouts, branch):
    origin, checkout = checkouts
    expected = git(checkout, "rev-parse", "HEAD")
    commit(origin, "model.txt", "remote update\n")
    if branch == "detached":
        git(checkout, "checkout", "--detach")
    else:
        git(checkout, "checkout", "-b", branch)

    result = prepare(checkout)

    assert result.returncode != 0
    assert git(checkout, "rev-parse", "HEAD") == expected


@pytest.mark.parametrize("diverged", [False, True])
def test_rejects_local_commits_even_if_main_is_only_ahead(checkouts, diverged):
    origin, checkout = checkouts
    expected = commit(checkout, "local.txt", "local commit\n")
    if diverged:
        commit(origin, "remote.txt", "remote commit\n")

    result = prepare(checkout)

    assert result.returncode != 0
    assert git(checkout, "rev-parse", "HEAD") == expected
    assert (checkout / "local.txt").read_text() == "local commit\n"


def test_fetch_failure_stops_without_using_stale_origin_main(checkouts, tmp_path):
    _, checkout = checkouts
    expected = git(checkout, "rev-parse", "HEAD")
    git(checkout, "remote", "set-url", "origin", str(tmp_path / "missing-origin"))

    result = prepare(checkout)

    assert result.returncode != 0
    assert git(checkout, "rev-parse", "HEAD") == expected


def test_service_updates_then_syncs_before_foreground_training():
    lines = (PREPARE_CHECKOUT.parent / "bst-ranker-training.service").read_text().splitlines()
    commands = [line.split("=", 1) for line in lines if line.startswith("ExecStart")]

    assert [directive for directive, _ in commands] == ["ExecStartPre", "ExecStartPre", "ExecStart"]
    assert commands[0][1] == "/usr/local/libexec/bst-ranker-training-prepare"
    assert shlex.split(commands[1][1])[1:] == ["sync", "--dev"]
    training = shlex.split(commands[2][1])
    assert training[1:5] == ["run", "python", "-u", "cli.py"]
    # The service must remain active throughout long training runs so the timer
    # cannot start another run while the first one is still using the checkout.
    assert "--no-background" in training
    assert "Type=oneshot" in lines
    assert "TimeoutStartSec=infinity" in lines
    assert "Restart=no" in lines
