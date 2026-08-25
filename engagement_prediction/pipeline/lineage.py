"""Resolve and validate the complete recorded ancestry of a stage artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline.dependencies import load_stage_manifest


def _stage_label(stage_folder: str) -> str:
    """Convert a numbered artifact folder such as ``03_post_selection`` to ``Stage 3``."""

    number = stage_folder.split("_", maxsplit=1)[0]
    try:
        return f"Stage {int(number)}"
    except ValueError:
        return f"stage '{stage_folder}'"


def resolve_recorded_stage_lineage(
    context: Context,
    *,
    terminal_stage_folder: str,
    ancestor_stage_folders: Sequence[str],
) -> dict[str, Path]:
    """Resolve a terminal artifact and verify its complete transitive lineage.

    New data-stage manifests record every selected ancestor, not only their
    direct input. This helper uses the terminal manifest as the authoritative
    chain, rejects conflicting explicit pins, and then checks that every
    intermediate manifest records the same earlier ancestors.

    The returned mapping contains ancestors in the caller-supplied order plus
    ``terminal_stage_folder``. Every resolved path is also recorded on the
    context so the current stage's manifest preserves the full lineage.
    """

    ancestors = tuple(ancestor_stage_folders)
    if len(set(ancestors)) != len(ancestors):
        raise ValueError("ancestor_stage_folders must be unique")
    if terminal_stage_folder in ancestors:
        raise ValueError("terminal_stage_folder cannot also be an ancestor")

    terminal_dir = context.resolve_prior_output(
        terminal_stage_folder,
        prior_path=context.prior_outputs.get(terminal_stage_folder),
    )
    terminal_inputs = load_stage_manifest(terminal_dir).get("inputs", {})
    missing = [folder for folder in ancestors if not terminal_inputs.get(folder)]
    if missing:
        rerun_hint = ""
        if "00_source_metadata" in missing:
            rerun_hint = (
                " This artifact predates the required Stage 00 lineage; rerun "
                "the pipeline from source_metadata."
            )
        raise ValueError(
            f"{_stage_label(terminal_stage_folder)} artifact '{terminal_dir}' must "
            "record its complete aligned ancestry; missing "
            f"{', '.join(missing)}.{rerun_hint}"
        )

    resolved: dict[str, Path] = {}
    for stage_folder in ancestors:
        recorded = Path(terminal_inputs[stage_folder]).resolve()
        explicit = context.prior_outputs.get(stage_folder)
        if explicit is not None and Path(explicit).resolve() != recorded:
            raise ValueError(
                f"Pinned {stage_folder} artifact '{Path(explicit).resolve()}' does "
                f"not match {_stage_label(terminal_stage_folder)} lineage '{recorded}'"
            )
        resolved[stage_folder] = context.resolve_prior_output(
            stage_folder,
            prior_path=recorded,
        )

    # An intermediate stage may itself be internally inconsistent even when
    # the terminal manifest lists the expected paths. Verify every transitive
    # edge instead of trusting only the terminal artifact's flattened inputs.
    for index, stage_folder in enumerate(ancestors):
        earlier_ancestors = ancestors[:index]
        if not earlier_ancestors:
            continue
        artifact_dir = resolved[stage_folder]
        recorded_inputs = load_stage_manifest(artifact_dir).get("inputs", {})
        for earlier_folder in earlier_ancestors:
            recorded = recorded_inputs.get(earlier_folder)
            if not recorded or Path(recorded).resolve() != resolved[earlier_folder]:
                raise ValueError(
                    f"{_stage_label(terminal_stage_folder)} lineage is invalid: "
                    f"{artifact_dir} records a different {earlier_folder} ancestor"
                )

    resolved[terminal_stage_folder] = terminal_dir
    return resolved
