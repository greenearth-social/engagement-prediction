#!/usr/bin/env python3

"""
Parse a sweep YAML config and emit a TSV plan + bash variable file.

Used by ``run_cap_arch_sweep.sh`` to translate a human-readable sweep config
into machine-readable instructions, without forcing the bash script to parse
YAML directly.

Outputs (written to stdout when run with --emit-plan, or printed as bash
``key=value`` lines when run with --emit-vars):

Plan (TSV, one cell per line):
  PREP\t<cap_label>\t<cap_value_or_NONE>\t<run_dir>
  TRAIN\t<phase>\t<cap_label>\t<run_dir>\t<run_tag>\t<seed>\t<json_args>

Where:
- cap_label: ``cap_inf`` for null cap, else ``cap_<N>``
- run_dir: ``<sweep_root>/<cap_label>``
- phase: ``mlp`` (parallelisable) or ``tt`` (sequential)
- json_args: JSON-encoded dict of cli flags for this cell, passed verbatim

Variables (bash-evalable):
  SWEEP_NAME=...
  INGESTION_RUN=...
  EPOCHS=...
  ...
  CAPS_LABELS=(cap_inf cap_50 ...)
  CAPS_VALUES=(NONE 50 ...)
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except ImportError:
    sys.stderr.write(
        "ERROR: PyYAML is required.  Install it via the eng-pred conda env "
        "(it is already declared in environment.yml) or `pip install pyyaml`.\n"
    )
    sys.exit(2)


REQUIRED_KEYS = {"sweep_name", "ingestion_run", "caps", "architectures", "seeds"}
DEFAULTS = {
    "epochs": 100,
    "batch_size": 4096,
    "patience": 20,
    "max_parallel_mlp": 2,
    "extra_cli_args": [],
}


def _cap_label(cap: Any) -> str:
    if cap is None:
        return "cap_inf"
    return f"cap_{int(cap)}"


def _validate(cfg: Dict[str, Any]) -> None:
    missing = REQUIRED_KEYS - set(cfg.keys())
    if missing:
        raise SystemExit(f"Sweep YAML missing required keys: {sorted(missing)}")
    if not isinstance(cfg["caps"], list) or not cfg["caps"]:
        raise SystemExit("`caps` must be a non-empty list (use [null] for no cap).")
    if not isinstance(cfg["architectures"], list) or not cfg["architectures"]:
        raise SystemExit("`architectures` must be a non-empty list.")
    if not isinstance(cfg["seeds"], list) or not cfg["seeds"]:
        raise SystemExit("`seeds` must be a non-empty list of ints.")
    for i, arch in enumerate(cfg["architectures"]):
        if not isinstance(arch, dict) or "model_type" not in arch:
            raise SystemExit(
                f"architectures[{i}] must be a mapping with at least `model_type`"
            )
        if arch["model_type"] not in {"mlp", "two-tower"}:
            raise SystemExit(
                f"architectures[{i}].model_type must be 'mlp' or 'two-tower' "
                f"(got {arch['model_type']!r})"
            )


def _arch_run_tag_suffix(arch: Dict[str, Any]) -> str:
    parts: List[str] = [arch["model_type"]]
    if "user_encoder" in arch:
        parts.append(str(arch["user_encoder"]))
    if arch.get("user_summarization"):
        parts.append(str(arch["user_summarization"]))
    return "_".join(parts).replace("-", "_")


def _arch_cli_args(arch: Dict[str, Any], ema_alpha_default: float) -> Dict[str, Any]:
    """Translate an architecture spec into per-cell CLI flag values."""
    args: Dict[str, Any] = {"model_type": arch["model_type"]}
    if "user_encoder" in arch:
        args["user_encoder"] = arch["user_encoder"]
    if arch.get("user_encoder") == "summarized":
        if arch.get("user_summarization"):
            args["user_summarization"] = arch["user_summarization"]
        if arch.get("user_summarization") == "ema":
            args["ema_alpha"] = float(arch.get("ema_alpha", ema_alpha_default))
    return args


def emit_plan(cfg: Dict[str, Any], sweep_root: Path) -> None:
    ema_alpha_default = float(cfg.get("ema_alpha", 0.1))
    epochs = int(cfg.get("epochs", DEFAULTS["epochs"]))
    batch_size = int(cfg.get("batch_size", DEFAULTS["batch_size"]))
    patience = int(cfg.get("patience", DEFAULTS["patience"]))

    # PREP rows: one per cap level.
    for cap in cfg["caps"]:
        label = _cap_label(cap)
        cell_dir = sweep_root / label
        print(
            "PREP",
            label,
            "NONE" if cap is None else int(cap),
            str(cell_dir),
            sep="\t",
        )

    # TRAIN rows: one per (cap, arch, seed) cell.
    for cap in cfg["caps"]:
        cap_label = _cap_label(cap)
        cell_dir = sweep_root / cap_label
        for arch in cfg["architectures"]:
            arch_args = _arch_cli_args(arch, ema_alpha_default)
            arch_args["epochs"] = epochs
            arch_args["batch_size"] = batch_size
            arch_args["patience"] = patience
            for extra in cfg.get("extra_cli_args", []):
                pass  # see _emit_vars: passed via shell, not per-cell

            tag_suffix = _arch_run_tag_suffix(arch)
            phase = "mlp" if arch["model_type"] == "mlp" else "tt"
            for seed in cfg["seeds"]:
                run_tag = f"{tag_suffix}_seed{seed}_{cap_label}"
                cell_args = dict(arch_args)
                cell_args["random_seed"] = int(seed)
                cell_args["run_tag"] = run_tag
                print(
                    "TRAIN",
                    phase,
                    cap_label,
                    str(cell_dir),
                    run_tag,
                    int(seed),
                    json.dumps(cell_args),
                    sep="\t",
                )


def emit_vars(cfg: Dict[str, Any], sweep_root: Path, plan_path: Path) -> None:
    print(f"SWEEP_NAME={shlex.quote(str(cfg['sweep_name']))}")
    print(f"INGESTION_RUN={shlex.quote(str(cfg['ingestion_run']))}")
    print(f"SWEEP_ROOT={shlex.quote(str(sweep_root))}")
    print(f"PLAN_PATH={shlex.quote(str(plan_path))}")
    print(f"EPOCHS={int(cfg.get('epochs', DEFAULTS['epochs']))}")
    print(f"BATCH_SIZE={int(cfg.get('batch_size', DEFAULTS['batch_size']))}")
    print(f"PATIENCE={int(cfg.get('patience', DEFAULTS['patience']))}")
    print(f"MAX_PARALLEL_MLP={int(cfg.get('max_parallel_mlp', DEFAULTS['max_parallel_mlp']))}")
    extra = cfg.get("extra_cli_args", [])
    print(f"EXTRA_CLI_ARGS={shlex.quote(json.dumps(extra))}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("config", help="Path to sweep YAML config")
    p.add_argument("--sweep-root", required=True,
                   help="Absolute path of the sweep root directory")
    p.add_argument("--plan-path", required=True,
                   help="Where the plan file will be written")
    p.add_argument("--mode", choices=["plan", "vars"], required=True)
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    _validate(cfg)
    sweep_root = Path(args.sweep_root)

    if args.mode == "plan":
        emit_plan(cfg, sweep_root)
    elif args.mode == "vars":
        emit_vars(cfg, sweep_root, Path(args.plan_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
