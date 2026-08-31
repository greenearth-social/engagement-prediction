"""Reusable offline evaluation for canonical engagement models."""

from .artifacts import (
    ModelArtifact,
    Stage7Artifact,
    resolve_model_artifact,
    resolve_stage7_artifact,
    validate_comparison_contract,
)
from .comparison import ComparisonResult, ComparisonSettings, run_model_comparison

__all__ = [
    "ComparisonResult",
    "ComparisonSettings",
    "ModelArtifact",
    "Stage7Artifact",
    "resolve_model_artifact",
    "resolve_stage7_artifact",
    "run_model_comparison",
    "validate_comparison_contract",
]
