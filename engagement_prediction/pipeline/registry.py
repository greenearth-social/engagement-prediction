#!/usr/bin/env python3

"""Map canonical stage keys to entrypoint files and artifact-folder names.

Stages remain loaded by absolute file path to preserve the pipeline's existing
execution contract.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional

from .core import ROOT, Context, load_run_callable


# Stage specs: stage_key -> (relative_file_path_from_root, stage_folder_name)
STAGE_SPECS: Dict[str, Tuple[str, str]] = {
    'source_metadata': ("engagement_prediction/stages/source_metadata.py",       "00_source_metadata"),
    'query_selection': ("engagement_prediction/stages/query_selection.py",     "01_query_selection"),
    'user_history':    ("engagement_prediction/stages/user_history.py",        "02_user_history"),
    'post_selection':  ("engagement_prediction/stages/post_selection.py",      "03_post_selection"),
    'negative_selection': ("engagement_prediction/stages/negative_selection.py", "04_negative_selection"),
    'post_liker_history': ("engagement_prediction/stages/post_liker_history.py", "05_post_liker_history"),
    'author_statistics': ("engagement_prediction/stages/author_statistics.py",   "06_author_statistics"),
    'dataset_hydration': ("engagement_prediction/stages/dataset_hydration.py",   "07_dataset_hydration"),
    'train_two_tower': ("engagement_prediction/stages/train_two_tower.py",     "08_train_two_tower"),
    'train_bst_ranker': ("engagement_prediction/stages/train_bst_ranker.py",   "08_train_bst_ranker"),
}


def get_stage_spec(stage_name: str) -> Tuple[Path, str]:
    if stage_name not in STAGE_SPECS:
        raise KeyError(f"Unknown stage '{stage_name}'")
    rel_path, folder = STAGE_SPECS[stage_name]
    return (ROOT / rel_path).resolve(), folder


def run_stage(stage_name: str, context: Context, args) -> Dict[str, object]:
    module_path, folder = get_stage_spec(stage_name)
    run_fn = load_run_callable(module_path)
    # Each stage script is responsible for creating a timestamped subdir under
    # the canonical artifact store and returning its path.
    context.begin_stage(stage_name, folder)
    result = run_fn(context, args)
    # Expect: {'output_dir': Path, 'artifacts': {...}}
    out_dir = result.get('output_dir') if isinstance(result, dict) else None
    if out_dir is None:
        raise RuntimeError(f"Stage '{stage_name}' did not return an output_dir")
    context.record_artifact(stage_name, Path(out_dir), extras=(result.get('artifacts') or {}))
    context.finalize_stage(
        stage_key=stage_name,
        stage_folder=folder,
        output_dir=Path(out_dir),
        args=args,
        argv=getattr(args, "_argv", None),
    )
    return result
