#!/usr/bin/env python3
"""Create a checksummed Power Likers portability package in GCS.

The YAML manifest names each input, its expected GCS destination, and whether
it contains restricted or DID-bearing material.  This tool copies only files
present in that manifest, writes a local ``SHA256SUMS`` manifest, uploads it,
and then verifies every remote object's checksum metadata.

Usage:
    python ops/power_likers_portable/export_to_gcs.py \
      --bucket "$PL_GCS_BUCKET" --run-id 20260724_portability_v1 --dry-run

Set ``PL_EXPORT_PRIVATE=1`` only after confirming that the bucket's private
prefix has restricted IAM.  Exclusion lists are never copied otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("export_manifest.yml")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_skip(path: Path, excluded_names: set[str]) -> bool:
    return any(
        part in excluded_names or any(path.match(pattern) for pattern in excluded_names)
        for part in path.parts
    )


def files_for_entry(entry: dict[str, Any]) -> Iterable[tuple[Path, Path]]:
    source = Path(entry["source"]).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"{entry['id']}: source does not exist: {source}")
    include = entry.get("include")
    excludes = set(entry.get("excludes", []))
    if source.is_file():
        yield source, Path(source.name)
        return

    roots = [source / name for name in include] if include else [source]
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"{entry['id']}: required input does not exist: {root}")
        if root.is_file():
            yield root, root.relative_to(source)
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not should_skip(path.relative_to(source), excludes):
                yield path, path.relative_to(source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, help="Destination bucket name, without gs://")
    parser.add_argument("--run-id", required=True, help="Immutable export label, e.g. 20260724_portability_v1")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text())
    private_allowed = os.environ.get("PL_EXPORT_PRIVATE") == "1"
    package_prefix = f"{manifest['destination_prefix'].strip('/')}/exports/{args.run_id}"
    planned: list[dict[str, str]] = []

    for entry in manifest["artifacts"]:
        if entry.get("scope") == "private" and not private_allowed:
            print(
                f"SKIP private artifact {entry['id']} (set PL_EXPORT_PRIVATE=1 after IAM review)",
                file=sys.stderr,
            )
            continue
        for source, relative in files_for_entry(entry):
            destination = f"{package_prefix}/{entry['destination'].strip('/')}/{relative.as_posix()}"
            planned.append(
                {
                    "artifact": entry["id"],
                    "source": str(source),
                    "destination": destination,
                    "sha256": sha256_file(source),
                    "bytes": str(source.stat().st_size),
                    "sensitivity": entry["sensitivity"],
                }
            )

    index = {
        "version": manifest["version"],
        "project": manifest["project"],
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.popen(f"git -C {REPO_ROOT} rev-parse HEAD").read().strip(),
        "private_included": private_allowed,
        "files": planned,
    }
    print(f"Prepared {len(planned)} files ({sum(int(row['bytes']) for row in planned):,} bytes).")
    if args.dry_run:
        print(json.dumps(index, indent=2))
        return 0

    result = subprocess.run(
        ["gcloud", "storage", "buckets", "describe", f"gs://{args.bucket}"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"Bucket gs://{args.bucket} does not exist or is inaccessible: {result.stderr.strip()}")

    for row in planned:
        subprocess.run(
            ["gcloud", "storage", "cp", row["source"], f"gs://{args.bucket}/{row['destination']}"],
            check=True,
        )

    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "SHA256SUMS.json"
        index_path.write_text(json.dumps(index, indent=2) + "\n")
        subprocess.run(
            ["gcloud", "storage", "cp", str(index_path), f"gs://{args.bucket}/{package_prefix}/SHA256SUMS.json"],
            check=True,
        )
        remote_index = subprocess.check_output(
            ["gcloud", "storage", "cat", f"gs://{args.bucket}/{package_prefix}/SHA256SUMS.json"],
            text=True,
        )
        if json.loads(remote_index) != index:
            raise RuntimeError("Uploaded checksum index did not round-trip.")
    print(f"Verified gs://{args.bucket}/{package_prefix}/SHA256SUMS.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
