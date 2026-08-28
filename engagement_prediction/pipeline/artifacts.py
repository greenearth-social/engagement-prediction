"""Atomic lifecycle helpers for partitioned pipeline artifact bundles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import polars as pl

from engagement_prediction.data.parquet import (
    ensure_typed_parquet_dataset,
    validate_parquet_part_schemas,
)
from engagement_prediction.pipeline.core import Context


def _child_path(parent: Path, relative_name: str) -> Path:
    relative_path = Path(relative_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Artifact paths must be relative children: {relative_name}")
    return parent / relative_path


@dataclass(frozen=True)
class PartialArtifactBundle:
    """Own one public partial bundle and its disposable staging directory.

    Failed work is intentionally retained under the partial paths for diagnosis.
    The final bundle appears only after all declared Parquet datasets have a
    physical, schema-correct part and successful-run staging has been removed.
    """

    output_dir: Path
    final_path: Path
    partial_path: Path
    staging_path: Path
    dataset_schemas: Mapping[str, dict[str, pl.DataType]]

    @classmethod
    def create(
        cls,
        *,
        output_dir: Path,
        bundle_name: str,
        staging_name: str,
        dataset_schemas: Mapping[str, dict[str, pl.DataType]],
    ) -> "PartialArtifactBundle":
        output_dir = Path(output_dir)
        final_path = _child_path(output_dir, bundle_name)
        partial_path = _child_path(output_dir, f"{bundle_name}.partial")
        staging_path = _child_path(output_dir, staging_name)
        conflicts = [
            path
            for path in (
                final_path,
                partial_path,
                staging_path,
                output_dir / "summary.json",
                output_dir / "summary.json.partial",
                output_dir / "stage_info.txt",
                output_dir / "stage_info.txt.partial",
            )
            if path.exists()
        ]
        if conflicts:
            raise FileExistsError(
                "Artifact publication paths already exist: "
                + ", ".join(str(path) for path in conflicts)
            )
        partial_path.mkdir(parents=True, exist_ok=False)
        try:
            staging_path.mkdir(parents=True, exist_ok=False)
        except Exception:
            partial_path.rmdir()
            raise
        return cls(
            output_dir=output_dir,
            final_path=final_path,
            partial_path=partial_path,
            staging_path=staging_path,
            dataset_schemas=dict(dataset_schemas),
        )

    def public_path(self, relative_name: str) -> Path:
        """Return a path under the not-yet-published public bundle."""

        return _child_path(self.partial_path, relative_name)

    def final_public_path(self, relative_name: str) -> Path:
        """Return the corresponding path under the final public bundle."""

        return _child_path(self.final_path, relative_name)

    def staging_output_path(self, relative_name: str) -> Path:
        """Return a path under the disposable successful-run staging root."""

        return _child_path(self.staging_path, relative_name)

    def publish(
        self,
        *,
        summary: Mapping[str, Any],
        stage_info: str,
    ) -> dict[str, int]:
        """Validate declared datasets, remove staging, and publish atomically."""

        part_counts: dict[str, int] = {}
        for relative_name, schema in self.dataset_schemas.items():
            dataset_path = self.public_path(relative_name)
            ensure_typed_parquet_dataset(dataset_path, schema)
            part_counts[relative_name] = validate_parquet_part_schemas(
                dataset_path,
                schema,
            )

        summary_path = self.output_dir / "summary.json"
        stage_info_path = self.output_dir / "stage_info.txt"
        summary_partial_path = self.output_dir / "summary.json.partial"
        stage_info_partial_path = self.output_dir / "stage_info.txt.partial"
        summary_partial_path.write_text(
            json.dumps(dict(summary), indent=2, sort_keys=True) + "\n"
        )
        stage_info_partial_path.write_text(
            stage_info if stage_info.endswith("\n") else stage_info + "\n"
        )

        bundle_published = False
        summary_published = False
        stage_info_published = False
        try:
            if self.final_path.exists():
                raise FileExistsError(
                    f"Refusing to replace existing artifact bundle: {self.final_path}"
                )
            shutil.rmtree(self.staging_path)
            self.partial_path.replace(self.final_path)
            bundle_published = True
            summary_partial_path.replace(summary_path)
            summary_published = True
            stage_info_partial_path.replace(stage_info_path)
            stage_info_published = True
        except Exception:
            # The stage manifest is written only after this method returns. Put
            # any already-promoted files back under their diagnostic partial
            # names so a failed publication never looks complete.
            if stage_info_published and stage_info_path.exists():
                stage_info_path.replace(stage_info_partial_path)
            if summary_published and summary_path.exists():
                summary_path.replace(summary_partial_path)
            if bundle_published and self.final_path.exists():
                self.final_path.replace(self.partial_path)
            raise
        return part_counts


def complete_stage_artifacts(
    *,
    context: Context,
    stage_key: str,
    stage_folder: str,
    result: Any,
    args: Any,
) -> dict[str, object]:
    """Record a successful stage result and write its completion manifest."""

    if not isinstance(result, dict):
        raise RuntimeError(f"Stage '{stage_key}' must return an artifact dictionary")
    output_dir_value = result.get("output_dir")
    if output_dir_value is None:
        raise RuntimeError(f"Stage '{stage_key}' did not return an output_dir")
    output_dir = Path(output_dir_value)
    if not output_dir.is_dir():
        raise FileNotFoundError(
            f"Stage '{stage_key}' output directory does not exist: {output_dir}"
        )
    artifact_values = result.get("artifacts") or {}
    if not isinstance(artifact_values, dict):
        raise RuntimeError(f"Stage '{stage_key}' artifacts must be a dictionary")

    context.record_artifact(stage_key, output_dir, extras=artifact_values)
    context.finalize_stage(
        stage_key=stage_key,
        stage_folder=stage_folder,
        output_dir=output_dir,
        args=args,
        argv=getattr(args, "_argv", None),
    )
    return result
