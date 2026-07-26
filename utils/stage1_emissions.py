"""Privacy-safe Stage-1 population emissions and provenance helpers."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SURROGATE_COLUMN = "user_surrogate_id"
COUNT_COLUMN = "like_count"
_DISALLOWED_COLUMNS = {"did", "handle", "salt"}


def create_run_salt(run_timestamp: str) -> tuple[bytes, Path]:
    """Create a host-only run salt; it is intentionally outside output artifacts."""
    salt_dir = Path(os.environ.get(
        "STAGE1_SURROGATE_SALT_DIR",
        "/mnt/data/wm.s.schulz/private/stage1-fresh2026q3-salts",
    ))
    salt_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        salt_dir.chmod(0o700)
    except OSError:
        pass
    salt_path = salt_dir / f"{run_timestamp}.salt"
    if salt_path.exists():
        raise FileExistsError(f"Refusing to reuse Stage-1 surrogate salt: {salt_path}")
    salt = os.urandom(32)
    fd = os.open(salt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(salt)
    return salt, salt_path


def surrogate_id(did: str, salt: bytes) -> str:
    return hashlib.blake2b(did.encode("utf-8"), key=salt, digest_size=32).hexdigest()


def _validate_count_schema(columns: Iterable[str]) -> None:
    actual = set(columns)
    if actual != {SURROGATE_COLUMN, COUNT_COLUMN}:
        raise ValueError(f"count emission must contain only surrogate/count columns, got {sorted(actual)}")
    leaked = actual & _DISALLOWED_COLUMNS
    if leaked:
        raise ValueError(f"privacy violation in count emission: {sorted(leaked)}")


def write_surrogate_counts_from_parquet(source: Path, destination: Path, salt: bytes) -> None:
    """Convert a DID/count parquet to surrogate/count parquet in bounded batches."""
    source_file = pq.ParquetFile(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for batch in source_file.iter_batches(columns=["did", COUNT_COLUMN], batch_size=250_000):
            dids = batch.column(0).to_pylist()
            counts = batch.column(1)
            table = pa.table({
                SURROGATE_COLUMN: pa.array([surrogate_id(did, salt) for did in dids], type=pa.string()),
                COUNT_COLUMN: counts,
            })
            _validate_count_schema(table.column_names)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        empty = pa.table({SURROGATE_COLUMN: pa.array([], type=pa.string()), COUNT_COLUMN: pa.array([], type=pa.int64())})
        pq.write_table(empty, destination, compression="zstd")


def write_surrogate_counts_from_rows(dids: list[str], counts: list[int], destination: Path, salt: bytes) -> None:
    table = pa.table({
        SURROGATE_COLUMN: pa.array([surrogate_id(did, salt) for did in dids], type=pa.string()),
        COUNT_COLUMN: pa.array(counts, type=pa.int64()),
    })
    _validate_count_schema(table.column_names)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="zstd")


def _concentration_summary(count_path: Path, *, lorenz_points: int = 1001) -> dict[str, Any]:
    values = pq.read_table(count_path, columns=[COUNT_COLUMN]).column(COUNT_COLUMN).to_numpy(zero_copy_only=False).astype(np.float64)
    if len(values) == 0 or float(values.sum()) <= 0:
        return {"n_users": int(len(values)), "n_likes": int(values.sum()), "gini": 0.0, "lorenz_curve": [], "top_user_share_to_like_share": []}
    values.sort()
    cumulative = np.cumsum(values)
    total = float(cumulative[-1])
    user_share = np.linspace(0.0, 1.0, lorenz_points)
    indices = np.minimum((user_share * len(values)).astype(np.int64), len(values))
    lorenz = np.zeros(lorenz_points, dtype=np.float64)
    nonzero = indices > 0
    lorenz[nonzero] = cumulative[indices[nonzero] - 1] / total
    gini = float((len(values) + 1 - 2 * cumulative.sum() / total) / len(values))
    top_table = []
    for share in [0.001, 0.005, 0.01, 0.05, 0.10, 0.20]:
        n_top = max(1, int(np.ceil(len(values) * share)))
        top_table.append({"top_user_share": share, "like_share": float(values[-n_top:].sum() / total)})
    return {
        "n_users": int(len(values)),
        "n_likes": int(total),
        "gini": gini,
        "lorenz_curve": [{"cumulative_user_share": float(x), "cumulative_like_share": float(y)} for x, y in zip(user_share, lorenz)],
        "top_user_share_to_like_share": top_table,
    }


def write_concentration(destination: Path, population_path: Path, sampled_pre_cap_path: Path) -> None:
    payload = {
        "definition": "Uncapped counts are measured before the per-user random cap.",
        "population_before_min_like_filter_or_sampling": _concentration_summary(population_path),
        "sampled_pre_cap": _concentration_summary(sampled_pre_cap_path),
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n")


def write_stage1_ledger(destination_json: Path, destination_md: Path, stats: dict[str, Any], outputs: dict[str, int]) -> None:
    likes = stats["likes"]
    inferences = stats.get("inferences", {})
    rows = [
        {"point": "P1", "step": "raw likes in window", "n_users": likes["n_users_initial"], "n_likes": likes["n_likes_initial"], "reason": "date window"},
        {"point": "P2", "step": "users with at least one like", "n_users": likes["n_users_initial"], "n_likes": likes["n_likes_initial"], "reason": "population baseline"},
        {"point": "P3", "step": "minimum-like eligibility", "n_users": likes["n_users_eligible_for_sampling"], "n_likes": likes["n_likes_eligible"], "reason": "min_likes_per_user"},
        {"point": "P4", "step": "deterministic user sample", "n_users": likes["n_users_sampled"], "n_likes": likes["n_likes_after_user_sample"], "reason": "max_liking_users"},
        {"point": "P5", "step": "sampled pre-cap counts", "n_users": likes["n_users_sampled"], "n_likes": likes["n_likes_after_user_sample"], "reason": "before per-user cap"},
        {"point": "P6", "step": "sampled post-cap counts", "n_users": likes["n_users_post_cap"], "n_likes": likes["n_likes_after_per_user_cap"], "reason": "max_likes_per_user"},
        {"point": "P7", "step": "join-verified likes_core", "n_users": likes["n_users_final"], "n_likes": outputs["likes_core_rows"], "reason": "post/embedding join and final deduplication"},
        {"point": "P8", "step": "posts_core", "n_users": None, "n_likes": outputs["posts_core_rows"], "reason": "liked posts plus negative sample"},
        {"point": "P9", "step": "inferences_core coverage", "n_users": None, "n_likes": outputs["inferences_core_rows"], "reason": "inference rows matching posts_core"},
    ]
    payload = {"version": 1, "rows": rows, "inference_coverage": inferences.get("coverage_pct"), "generated_at": datetime.now(timezone.utc).isoformat()}
    destination_json.write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Stage-1 P1–P9 attrition ledger", "", "| Point | Step | Users | Likes / rows | Reason |", "|---|---|---:|---:|---|"]
    lines.extend(f"| {row['point']} | {row['step']} | {row['n_users'] if row['n_users'] is not None else ''} | {row['n_likes']} | {row['reason']} |" for row in rows)
    destination_md.write_text("\n".join(lines) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_stage1_manifest(destination: Path, out_dir: Path, parameters: dict[str, Any], source_commit: str, run_timestamp: str) -> None:
    artifacts = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != destination.name:
            artifacts.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    payload = {
        "version": 1,
        "run_timestamp": run_timestamp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "parameters": parameters,
        "privacy": {"surrogate_salt": "host-only; not present in this directory or manifest", "count_artifacts": "surrogate IDs and counts only"},
        "artifacts": artifacts,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n")


def current_git_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
