#!/usr/bin/env python3
"""Compare training_config.json files from two Stage 3 training runs.

Usage:
    python ops/compare_training_configs.py 20260708_191249_6b900eb1 20260709_120000_abcd1234
    python ops/compare_training_configs.py --parent-dir /mnt/data/dave/outputs/artifacts/03_train run_a run_b
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, NamedTuple


DEFAULT_PARENT_DIR = Path("/mnt/data/dave/outputs/artifacts/03_train")
MISSING = object()
IGNORED_KEYS = {
    "bst_popularity_log_mean",
    "bst_popularity_log_std",
}


class Difference(NamedTuple):
    path: str
    left: Any
    right: Any


def resolve_run_dir(run: str, parent_dir: Path) -> Path:
    candidate = Path(run).expanduser()
    if candidate.is_absolute() or len(candidate.parts) > 1:
        return candidate
    return Path(parent_dir).expanduser() / candidate


def load_training_config(run: str, parent_dir: Path) -> tuple[dict[str, Any], Path]:
    run_dir = resolve_run_dir(run, parent_dir)
    config_path = run_dir / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing training config: {config_path}")
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Expected JSON object in {config_path}")
    return config, config_path


def join_key(path: str, key: str) -> str:
    if not path:
        return key
    return f"{path}.{key}"


def join_index(path: str, index: int) -> str:
    if not path:
        return f"[{index}]"
    return f"{path}[{index}]"


def compare_json(left: Any, right: Any, path: str = "") -> Iterable[Difference]:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key in IGNORED_KEYS:
                continue
            child_path = join_key(path, str(key))
            if key not in left:
                yield Difference(child_path, MISSING, right[key])
            elif key not in right:
                yield Difference(child_path, left[key], MISSING)
            else:
                yield from compare_json(left[key], right[key], child_path)
        return

    if isinstance(left, list) and isinstance(right, list):
        max_len = max(len(left), len(right))
        for index in range(max_len):
            child_path = join_index(path, index)
            if index >= len(left):
                yield Difference(child_path, MISSING, right[index])
            elif index >= len(right):
                yield Difference(child_path, left[index], MISSING)
            else:
                yield from compare_json(left[index], right[index], child_path)
        return

    if left != right:
        yield Difference(path or "<root>", left, right)


def format_value(value: Any) -> str:
    if value is MISSING:
        return "<missing>"
    return json.dumps(value, sort_keys=True)


def print_differences(differences: list[Difference], left_name: str, right_name: str) -> None:
    if not differences:
        print("No differences found.")
        return

    print(f"Found {len(differences)} difference(s):")
    for difference in differences:
        print(f"{difference.path}:")
        print(f"  {left_name}: {format_value(difference.left)}")
        print(f"  {right_name}: {format_value(difference.right)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two training_config.json files")
    parser.add_argument("left_run", help="First Stage 3 run folder name")
    parser.add_argument("right_run", help="Second Stage 3 run folder name")
    parser.add_argument(
        "--parent-dir",
        type=Path,
        default=DEFAULT_PARENT_DIR,
        help=f"Parent directory containing Stage 3 run folders (default: {DEFAULT_PARENT_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        left_config, _ = load_training_config(args.left_run, args.parent_dir)
        right_config, _ = load_training_config(args.right_run, args.parent_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    differences = list(compare_json(left_config, right_config))
    print_differences(differences, args.left_run, args.right_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
