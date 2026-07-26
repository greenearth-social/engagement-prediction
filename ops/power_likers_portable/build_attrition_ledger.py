#!/usr/bin/env python3
"""Build one auditable P1-P18 attrition ledger from replay artifacts.

The frozen Stage-1 summary is treated as immutable evidence for P1-P9.  Each
completed cell's ``paper_quality/cell_manifest.json`` and
``holdout_strata.json`` contributes P10-P18 rows, avoiding hand-maintained
population tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def nested_numbers(value: object, prefix: str = "") -> dict[str, float | int]:
    """Flatten numeric leaves so evolving Stage-1 summaries remain readable."""
    found: dict[str, float | int] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            found.update(nested_numbers(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        found[prefix] = value
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-summary", required=True, type=Path)
    parser.add_argument("--cells-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    stage1 = json.loads(args.stage1_summary.read_text())
    rows: list[dict[str, object]] = [
        {
            "point": "P1-P9",
            "stage": "01_get_data",
            "filter": key,
            "value": value,
            "source": str(args.stage1_summary),
        }
        for index, (key, value) in enumerate(sorted(nested_numbers(stage1).items()), start=1)
    ]
    manifests = sorted(args.cells_root.glob("**/paper_quality/cell_manifest.json"))
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        strata_path = manifest_path.with_name("holdout_strata.json")
        strata = json.loads(strata_path.read_text()) if strata_path.exists() else {}
        rows.extend(
            [
                {
                    "point": "P10",
                    "stage": "02_target_posts",
                    "filter": "excluded training users",
                    "value": manifest["n_excluded_users"],
                    "source": str(manifest_path),
                },
                {
                    "point": "P12",
                    "stage": "04_train",
                    "filter": "validation AUC",
                    "value": manifest["best_val_auc"],
                    "source": str(manifest_path),
                },
                {
                    "point": "P13",
                    "stage": "05_evaluate",
                    "filter": "holdout rows",
                    "value": sum(item["rows"] for item in strata.get("strata", [])),
                    "source": str(strata_path),
                },
                {
                    "point": "P18",
                    "stage": "remedy_sweep",
                    "filter": "cell manifest",
                    "value": manifest["random_seed"],
                    "source": str(manifest_path),
                },
            ]
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage1_summary": str(args.stage1_summary),
        "n_cells": len(manifests),
        "rows": rows,
    }
    (args.out_dir / "attrition_ledger.json").write_text(json.dumps(payload, indent=2) + "\n")
    markdown = [
        "# Attrition ledger",
        "",
        "| Point | Stage | Filter / measure | Value | Source |",
        "|---|---|---|---:|---|",
    ]
    markdown.extend(
        f"| {row['point']} | {row['stage']} | {row['filter']} | {row['value']} | `{row['source']}` |"
        for row in rows
    )
    (args.out_dir / "attrition_ledger.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps({"n_rows": len(rows), "n_cells": len(manifests)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
