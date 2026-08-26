import builtins

import pytest

import cli


def test_compare_rankers_parser_accepts_disabled_command():
    raw = cli.build_parser().parse_args([
        "compare-rankers",
        "--model", "tt:two-tower:/tmp/two_tower.pth",
        "--model", "bst:bst-ranker:/tmp/bst.pth",
        "--splits", "val", "holdout_unseen_users",
    ])

    assert raw.command == "compare-rankers"
    assert raw.model == [
        "tt:two-tower:/tmp/two_tower.pth",
        "bst:bst-ranker:/tmp/bst.pth",
    ]
    assert raw.splits == ["val", "holdout_unseen_users"]


def test_compare_rankers_fails_before_importing_deleted_utils(monkeypatch):
    raw = cli.build_parser().parse_args([
        "compare-rankers",
        "--model", "tt:two-tower:/tmp/two_tower.pth",
    ])
    real_import = builtins.__import__
    attempted_utils_imports = []

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "utils" or name.startswith("utils."):
            attempted_utils_imports.append(name)
            raise AssertionError(f"disabled compare command imported {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit, match="07_dataset_hydration"):
        cli.cmd_compare_rankers(raw)

    assert attempted_utils_imports == []
