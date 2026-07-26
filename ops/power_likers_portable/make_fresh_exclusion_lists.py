#!/usr/bin/env python3
"""Create private fresh-substrate R2 DID exclusions with provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--likes-core", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--stage1-manifest", type=Path, default=None)
    parser.add_argument("--min-likes", type=int, default=5)
    parser.add_argument("--percentiles", type=int, nargs="+", default=[10, 20, 30, 40])
    args = parser.parse_args()
    if args.min_likes < 1:
        raise ValueError("--min-likes must be positive")
    likes = args.likes_core.resolve()
    if not likes.is_file():
        raise FileNotFoundError(likes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = (
        pl.scan_parquet(likes)
        .select(pl.col("did").cast(pl.String))
        .group_by("did")
        .len(name="n_likes")
        .filter(pl.col("n_likes") >= args.min_likes)
        .sort(["n_likes", "did"], descending=[True, False])
        .collect()
    )
    if not counts.height:
        raise RuntimeError("No users meet the fresh R2 min-likes threshold")
    files: list[dict[str, object]] = []
    for percentile in args.percentiles:
        if percentile <= 0 or percentile >= 100:
            raise ValueError(f"invalid percentile: {percentile}")
        n_drop = int(counts.height * percentile / 100)
        if n_drop < 1:
            raise RuntimeError(f"top-{percentile}% rounds to zero users")
        path = args.out_dir / f"exclude_top{percentile}pct.parquet"
        counts.head(n_drop).select("did").write_parquet(path, compression="zstd")
        files.append({
            "name": path.name, "percentile": percentile, "users": n_drop,
            "minimum_n_likes": int(counts.head(n_drop)["n_likes"].min()),
            "bytes": path.stat().st_size, "sha256": digest(path),
        })
    cumulative = counts["n_likes"].cum_sum()
    lorenz_index = next(index for index, value in enumerate(cumulative) if value >= cumulative[-1] * 0.5)
    lorenz_path = args.out_dir / "exclude_lorenz50.parquet"
    counts.head(lorenz_index).select("did").write_parquet(lorenz_path, compression="zstd")
    files.append({
        "name": lorenz_path.name, "percentile": lorenz_index / counts.height * 100,
        "users": lorenz_index, "minimum_n_likes": int(counts.head(lorenz_index)["n_likes"].min()),
        "bytes": lorenz_path.stat().st_size, "sha256": digest(lorenz_path),
    })
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "definition": "Historical R2: top DID counts from final capped/post-join likes_core, retaining users with at least min_likes.",
        "likes_core": str(likes), "likes_core_sha256": digest(likes),
        "stage1_manifest": str(args.stage1_manifest) if args.stage1_manifest else None,
        "stage1_manifest_sha256": digest(args.stage1_manifest) if args.stage1_manifest else None,
        "min_likes": args.min_likes, "eligible_users": counts.height,
        "eligible_likes": int(counts["n_likes"].sum()), "files": files,
        "privacy": "DID parquets are private host-only artifacts; this manifest contains no DIDs.",
    }
    (args.out_dir / "exclusion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
