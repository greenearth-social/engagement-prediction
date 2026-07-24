#!/usr/bin/env python3
"""Verify every file in a restored Power Likers portability export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    index = json.loads(args.index.read_text())
    package_prefix = f"power_likers/exports/{index['run_id']}/"
    bad: list[str] = []
    for entry in index["files"]:
        destination = entry["destination"]
        if not destination.startswith(package_prefix):
            bad.append(f"unexpected package prefix: {destination}")
            continue
        path = args.root / destination.removeprefix(package_prefix)
        if not path.is_file():
            bad.append(f"missing: {path}")
        elif digest(path) != entry["sha256"]:
            bad.append(f"checksum mismatch: {path}")
    if bad:
        raise SystemExit("Export verification failed:\n- " + "\n- ".join(bad))
    print(f"Verified {len(index['files'])} files from export {index['run_id']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
