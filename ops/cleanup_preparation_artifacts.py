#!/usr/bin/env python3
"""Retire large preparation payloads from one completed Stage 7's ancestry.

Run offline, after deciding that no preparation run needs these shared inputs.
Dry runs write nothing. Cleanup preserves provenance and does not change prior
selection or mark ancestors as unavailable for subsequent preparation runs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import stat
import sys
from typing import Any, Sequence
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


ANCESTOR_FOLDERS = (
    "00_source_metadata", "01_query_selection", "02_user_history",
    "03_post_selection", "04_negative_selection", "05_post_liker_history",
    "06_author_statistics",
)

# Exact producer-owned payload names only. Outer bundles contain source JSON
# manifests, so neither they nor entire ancestor stage directories are removed.
PAYLOAD_NAMES = {
    "00_source_metadata": ("source_metadata_{suffix}/post_metadata",),
    "01_query_selection": (
        "queries_{suffix}.parquet", "query_positives_{suffix}.parquet",
        "_candidate_query_counts_{suffix}.parquet",
    ),
    "02_user_history": ("query_histories_{suffix}", "history_post_uris_{suffix}"),
    "03_post_selection": (
        "post_universe_{suffix}/posts", "post_universe_{suffix}/required_posts",
        "post_universe_{suffix}/candidate_sources",
        "post_universe_{suffix}/missing_required_posts",
    ),
    "05_post_liker_history": (
        "post_liker_histories_{suffix}/post_liker_events",
        "post_liker_histories_{suffix}/post_liker_posts",
    ),
}


@dataclass(frozen=True)
class CleanupCandidate:
    stage_folder: str
    path: Path
    size_bytes: int
    allocated_bytes: int
    file_count: int


@dataclass(frozen=True)
class CleanupPlan:
    stage7_dir: Path
    lineage: dict[str, Path]
    candidates: tuple[CleanupCandidate, ...]

    @property
    def total_bytes(self) -> int:
        return sum(candidate.size_bytes for candidate in self.candidates)

    @property
    def total_allocated_bytes(self) -> int:
        return sum(candidate.allocated_bytes for candidate in self.candidates)


def _regular_files(path: Path) -> list[Path]:
    """Inspect a tree without following symlinks, including broken links."""
    files: list[Path] = []
    pending = [path]
    while pending:
        current = pending.pop()
        mode = current.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"Refusing symlink in artifact payload: {current}")
        if stat.S_ISDIR(mode):
            pending.extend(sorted(current.iterdir()))
        elif stat.S_ISREG(mode):
            files.append(current)
        else:
            raise ValueError(f"Refusing non-regular artifact file: {current}")
    return files


def _inspect_candidate(stage_folder: str, stage_dir: Path, path: Path) -> CleanupCandidate | None:
    relative = path.relative_to(stage_dir)
    current = stage_dir
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"Refusing symlink in cleanup path: {current}")
    if not path.exists():
        return None
    files = _regular_files(path)
    size_bytes = 0
    allocated_bytes = 0
    for file in files:
        if file.suffix != ".parquet":
            raise ValueError(f"Unexpected non-Parquet file in cleanup payload: {file}")
        info = file.stat()
        size_bytes += info.st_size
        allocated_bytes += info.st_blocks * 512
    return CleanupCandidate(stage_folder, path, size_bytes, allocated_bytes, len(files))


def _validate_stage7(stage7_dir: Path) -> None:
    """Validate the retained index and Parquet headers, including older schemas."""
    import polars as pl
    from engagement_prediction.evaluation.artifacts import resolve_stage7_artifact

    artifact = resolve_stage7_artifact(stage7_dir)
    if artifact.root != stage7_dir:
        raise ValueError("--stage7 must identify the completed stage directory, not its bundle")
    bundle = artifact.bundle_path
    if bundle.parent != stage7_dir or (stage7_dir / bundle.name).is_symlink():
        raise ValueError("Stage 7 must retain its own bundle, not a symlink to other artifacts")
    _regular_files(bundle)
    index_path = bundle / "loader_index"
    metadata = json.loads((index_path / "format.json").read_text())
    entries = [metadata["embedding"], *metadata["arrays"].values(), *metadata["arrow_tables"].values()]
    for split in metadata["splits"].values():
        entries.extend(split["arrays"].values())
        entries.extend(split["arrow_tables"].values())
    for entry in entries:
        if not (index_path / entry["path"]).resolve().is_relative_to(bundle):
            raise ValueError("Stage 7 loader files must be contained in its retained bundle")
    datasets = [
        "posts", "queries", "query_positives", "query_histories",
        "hourly_negative_candidates", "authors",
    ]
    if artifact.loader_index_validation["format_version"] >= 2:
        datasets.append("post_liker_users")
    for name in datasets:
        parts = sorted((bundle / name).rglob("*.parquet"))
        if not parts:
            raise ValueError(f"Stage 7 is missing its retained {name} Parquet dataset")
        for part in parts:
            pl.read_parquet_schema(part)


def build_cleanup_plan(stage7_dir: Path) -> CleanupPlan:
    """Validate recorded ancestry and inventory only the known large payloads."""
    from engagement_prediction.pipeline.core import Context
    from engagement_prediction.pipeline.dependencies import load_stage_manifest
    from engagement_prediction.pipeline.lineage import resolve_recorded_stage_lineage

    stage7_dir = Path(stage7_dir).expanduser().resolve()
    _validate_stage7(stage7_dir)
    context = Context(
        run_dir=stage7_dir,
        artifacts_dir=stage7_dir.parent.parent,
        runs_dir=stage7_dir.parent.parent,
        prior_outputs={"07_dataset_hydration": stage7_dir},
    )
    lineage = resolve_recorded_stage_lineage(
        context,
        terminal_stage_folder="07_dataset_hydration",
        ancestor_stage_folders=ANCESTOR_FOLDERS,
    )
    directories = list(lineage.values())
    for index, directory in enumerate(directories):
        for other in directories[index + 1:]:
            if directory.is_relative_to(other) or other.is_relative_to(directory):
                raise ValueError("Stage directories in cleanup lineage must not overlap")
    candidates: list[CleanupCandidate] = []
    for folder in ANCESTOR_FOLDERS:
        stage_dir = lineage[folder]
        manifest = load_stage_manifest(stage_dir)
        if (
            manifest.get("stage_folder") != folder
            or manifest.get("stage_key") != folder.split("_", 1)[1]
            or manifest.get("status", "complete") != "complete"
            or manifest.get("stage_run_id", stage_dir.name) != stage_dir.name
        ):
            raise ValueError(f"Expected completed {folder} ancestor: {stage_dir}")
        for template in PAYLOAD_NAMES.get(folder, ()):
            path = stage_dir / template.format(suffix=stage_dir.name)
            candidate = _inspect_candidate(folder, stage_dir, path)
            if candidate is not None:
                candidates.append(candidate)
    return CleanupPlan(stage7_dir, lineage, tuple(candidates))


def _write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def apply_cleanup(plan: CleanupPlan) -> Path:
    """Delete a preflighted plan and retain an audit report, even after errors."""
    for directory in plan.lineage.values():
        if directory.resolve() != directory:
            raise ValueError(f"Cleanup lineage directory became a symlink: {directory}")
    _validate_stage7(plan.stage7_dir)
    # Recheck every candidate before the first deletion: a plan may have been
    # inspected before a path was replaced or a new file appeared in a dataset.
    current_candidates = []
    for candidate in plan.candidates:
        current = _inspect_candidate(
            candidate.stage_folder, plan.lineage[candidate.stage_folder], candidate.path
        )
        if current is not None and current != candidate:
            raise ValueError(f"Cleanup payload changed since planning: {candidate.path}")
        current_candidates.append(current)
    started_at = datetime.now(timezone.utc)
    report_path = plan.stage7_dir / (
        f"cleanup_preparation_report_{started_at:%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}.json"
    )
    report: dict[str, Any] = {
        "stage7_dir": str(plan.stage7_dir),
        "lineage": {folder: str(path) for folder, path in plan.lineage.items()},
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "status": "in_progress",
        "candidates": [{
            "stage_folder": candidate.stage_folder,
            "path": str(candidate.path),
            "size_bytes": candidate.size_bytes,
            "allocated_bytes": candidate.allocated_bytes,
            "file_count": candidate.file_count,
            "status": "pending" if current is not None else "already_absent",
        } for candidate, current in zip(plan.candidates, current_candidates)],
    }
    _write_report(report_path, report)
    try:
        for candidate, record in zip(plan.candidates, report["candidates"]):
            if record["status"] == "already_absent":
                continue
            if candidate.path.is_dir():
                shutil.rmtree(candidate.path)
            else:
                candidate.path.unlink()
            record["status"] = "deleted"
            _write_report(report_path, report)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        print(f"Cleanup interrupted; audit report: {report_path}", file=sys.stderr)
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_report(report_path, report)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage7", type=Path, required=True, help="Completed Stage 7 stage directory or run-view symlink")
    parser.add_argument("--apply", action="store_true", help="Delete the listed upstream payloads and write an audit report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print("Validating retained Stage 7 data and recorded ancestry...", flush=True)
        plan = build_cleanup_plan(args.stage7)
        print(f"Stage 7: {plan.stage7_dir}")
        for candidate in plan.candidates:
            print(f"{candidate.allocated_bytes / 2**30:8.3f} GiB allocated | {candidate.file_count:5d} files | {candidate.path}")
        print(
            f"Total: {len(plan.candidates)} payloads, {plan.total_bytes:,} file bytes, "
            f"{plan.total_allocated_bytes / 2**30:.3f} GiB allocated (estimated reclaimable space)"
        )
        print("Stages 4 and 6, all lineage metadata, and Stage 7 data are retained.")
        print("Run offline: these ancestors may be shared by other preparation runs.")
        if args.apply:
            report_path = apply_cleanup(plan)
            print(f"Cleanup complete. Audit report: {report_path}")
        else:
            print("Dry run: nothing changed. Use --apply to delete these payloads.")
        return 0
    except Exception as exc:
        print(f"Cleanup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
