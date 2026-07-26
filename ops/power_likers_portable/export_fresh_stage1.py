#!/usr/bin/env python3
"""Export one verified fresh Stage-1 substrate to its isolated GCS prefix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

BUCKET = "power-likers-research-gcp-vox-wschulz"
PROJECT = "gcp-vox"
PREFIX = "fresh_2026q3"
COUNT_COLUMNS = ["user_surrogate_id", "like_count"]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def remote_digest(uri: str) -> str:
    process = subprocess.Popen(
        ["gcloud", "--project", PROJECT, "storage", "cat", uri], stdout=subprocess.PIPE
    )
    assert process.stdout is not None
    hasher = hashlib.sha256()
    for chunk in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
        hasher.update(chunk)
    if process.wait() != 0:
        raise RuntimeError(f"Could not stream {uri} for checksum verification")
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--extra", type=Path, action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("CLOUDSDK_CONFIG"):
        raise RuntimeError("CLOUDSDK_CONFIG must point to an isolated configuration directory")
    project = subprocess.check_output(["gcloud", "config", "get-value", "project"], text=True).strip()
    if project != PROJECT:
        raise RuntimeError(f"isolated gcloud project must be {PROJECT}, got {project!r}")
    stage_dir = args.stage_dir.resolve()
    if not (stage_dir / "stage1_manifest.json").is_file():
        raise FileNotFoundError("stage1_manifest.json is required before export")
    # The generated export manifest is intentionally not an input to itself.
    candidates = sorted(
        path for path in stage_dir.iterdir()
        if path.is_file() and path.name != "fresh_export_manifest.json"
    ) + args.extra
    files: list[Path] = []
    for path in candidates:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if "salt" in path.name.lower() or path.name == ".population_counts_raw.parquet":
            raise ValueError(f"refusing to export secret/raw population file: {path}")
        if path.name.startswith("per_user_like_counts_"):
            if pq.ParquetFile(path).schema_arrow.names != COUNT_COLUMNS:
                raise ValueError(f"privacy schema violation in {path.name}")
        files.append(path)
    root = f"gs://{BUCKET}/{PREFIX}/exports/{args.run_id}"
    entries = [{"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)} for path in files]
    manifest = {
        "version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "bucket": BUCKET, "prefix": f"{PREFIX}/exports/{args.run_id}",
        "project": PROJECT, "stage_manifest_sha256": digest(stage_dir / "stage1_manifest.json"),
        "files": entries,
    }
    export_manifest = stage_dir / "fresh_export_manifest.json"
    export_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    for path in files + [export_manifest]:
        subprocess.run(["gcloud", "--project", PROJECT, "storage", "cp", str(path), f"{root}/{path.name}"], check=True)
    for entry in entries:
        uri = f"{root}/{entry['name']}"
        if remote_digest(uri) != entry["sha256"]:
            raise RuntimeError(f"remote SHA256 mismatch: {uri}")
    if remote_digest(f"{root}/fresh_export_manifest.json") != digest(export_manifest):
        raise RuntimeError("remote export manifest checksum mismatch")
    print(json.dumps({"verified": len(entries), "root": root}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
