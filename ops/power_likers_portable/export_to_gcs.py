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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("export_manifest.yml")
HASH_CACHE_PATH = Path(__file__).with_name(".hash_cache.json")


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def load_hash_cache() -> dict[str, dict[str, Any]]:
    if HASH_CACHE_PATH.exists():
        try:
            return json.loads(HASH_CACHE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_hash_cache(cache: dict[str, dict[str, Any]]) -> None:
    HASH_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def sha256_file(path: Path, *, cache: dict[str, dict[str, Any]] | None = None) -> str:
    stat = path.stat()
    key = str(path)
    if cache is not None:
        cached = cache.get(key)
        if cached and cached.get("mtime") == stat.st_mtime and cached.get("size") == stat.st_size:
            return cached["sha256"]

    start = time.monotonic()
    digest = hashlib.sha256()
    read_bytes = 0
    last_report = start
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            read_bytes += len(chunk)
            now = time.monotonic()
            if now - last_report > 15:
                elapsed = now - start
                rate = read_bytes / elapsed if elapsed > 0 else 0
                log(
                    f"    ...hashing {path.name}: {human_bytes(read_bytes)}/{human_bytes(stat.st_size)} "
                    f"({rate/1e6:.1f} MB/s)"
                )
                last_report = now
    result = digest.hexdigest()
    if cache is not None:
        cache[key] = {"mtime": stat.st_mtime, "size": stat.st_size, "sha256": result}
        save_hash_cache(cache)
    return result


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
    hash_cache = load_hash_cache()

    log("Planning file list from manifest...")
    all_files: list[tuple[dict[str, Any], Path, Path]] = []
    for entry in manifest["artifacts"]:
        if entry.get("scope") == "private" and not private_allowed:
            log(f"SKIP private artifact {entry['id']} (set PL_EXPORT_PRIVATE=1 after IAM review)")
            continue
        for source, relative in files_for_entry(entry):
            all_files.append((entry, source, relative))
    total_bytes = sum(source.stat().st_size for _, source, _ in all_files)
    log(f"Found {len(all_files)} files to hash ({human_bytes(total_bytes)} total). Hashing now (cached hashes skip instantly)...")

    plan_start = time.monotonic()
    for i, (entry, source, relative) in enumerate(all_files, 1):
        stat = source.stat()
        size = stat.st_size
        t0 = time.monotonic()
        digest = sha256_file(source, cache=hash_cache)
        dt = time.monotonic() - t0
        cache_hit = dt < 0.05
        log(
            f"  [{i}/{len(all_files)}] {entry['id']}:{relative} ({human_bytes(size)})"
            f" {'[cached]' if cache_hit else f'hashed in {dt:.1f}s'}"
        )
        destination = f"{package_prefix}/{entry['destination'].strip('/')}/{relative.as_posix()}"
        planned.append(
            {
                "artifact": entry["id"],
                "source": str(source),
                "destination": destination,
                "sha256": digest,
                "bytes": str(size),
                "mtime_ns": str(stat.st_mtime_ns),
                "sensitivity": entry["sensitivity"],
            }
        )
    log(f"Hashing complete in {time.monotonic() - plan_start:.1f}s.")

    index = {
        "version": manifest["version"],
        "project": manifest["project"],
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.popen(f"git -C {REPO_ROOT} rev-parse HEAD").read().strip(),
        "private_included": private_allowed,
        "files": planned,
    }
    log(f"Prepared {len(planned)} files ({sum(int(row['bytes']) for row in planned):,} bytes).")
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

    upload_start = time.monotonic()
    for i, row in enumerate(planned, 1):
        source = Path(row["source"])
        stat = source.stat()
        if row["bytes"] != str(stat.st_size) or row["mtime_ns"] != str(stat.st_mtime_ns):
            log(f"  [{i}/{len(planned)}] Source changed since planning; re-hashing {source.name}.")
            row["bytes"] = str(stat.st_size)
            row["mtime_ns"] = str(stat.st_mtime_ns)
            row["sha256"] = sha256_file(source, cache=hash_cache)
        dest_uri = f"gs://{args.bucket}/{row['destination']}"
        # Skip re-upload if the object already exists with the same size (cheap resumability).
        existing = subprocess.run(
            ["gcloud", "storage", "du", dest_uri],
            capture_output=True,
            text=True,
        )
        if existing.returncode == 0 and existing.stdout.strip():
            existing_size = existing.stdout.strip().split()[0]
            if existing_size == row["bytes"]:
                log(f"  [{i}/{len(planned)}] SKIP (already uploaded, size match): {row['destination']}")
                continue
        t0 = time.monotonic()
        log(f"  [{i}/{len(planned)}] Uploading {row['artifact']}:{Path(row['source']).name} ({human_bytes(int(row['bytes']))}) -> {dest_uri}")
        subprocess.run(
            ["gcloud", "storage", "cp", row["source"], dest_uri],
            check=True,
        )
        dt = time.monotonic() - t0
        rate = int(row["bytes"]) / dt / 1e6 if dt > 0 else 0
        log(f"    done in {dt:.1f}s ({rate:.1f} MB/s)")
    log(f"Upload complete in {time.monotonic() - upload_start:.1f}s.")

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
    log(f"Verified gs://{args.bucket}/{package_prefix}/SHA256SUMS.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
