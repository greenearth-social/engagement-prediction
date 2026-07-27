#!/usr/bin/env bash
# Harvest the completed nine-condition MLP matrix without training F1 or
# two-tower cells. Run GPU scoring before CPU-only report generation so the
# accelerator becomes available as early as possible.
set -euo pipefail
shopt -s nullglob

repo="${PL_REPO_ROOT:-$HOME/power-likers/code/engagement-prediction}"
stage1="${PL_STAGE1_ROOT:-$HOME/power-likers/stage1/0015_stage1_fixc_v2_20260512_054231}"
exclusions="${PL_EXCLUSIONS_DIR:-$HOME/power-likers/private/exclusions}"
out="${PL_FULL_MATRIX_OUT:-$HOME/power-likers/full_matrix/harvest_mlp}"
source_commit="${PL_SOURCE_COMMIT:-unknown}"
phase="${PL_HARVEST_PHASE:-all}"
# This harvest needs directional uncertainty for 45 cells promptly.  The
# emitted JSON records the actual count, so selected results can later be
# re-emitted at a higher Monte Carlo precision without changing the estimand.
bootstrap_repetitions="${PL_BOOTSTRAP_REPETITIONS:-200}"

[[ "$phase" == "all" || "$phase" == "gpu" || "$phase" == "cpu" || "$phase" == "dry-run" ]] || {
  echo "PL_HARVEST_PHASE must be all, gpu, cpu, or dry-run; got $phase" >&2
  exit 64
}
[[ -d "$stage1/01_get_data" ]] || {
  echo "Missing Stage-1 substrate: $stage1" >&2
  exit 66
}
mkdir -p "$out/reports"
cd "$repo"

conditions=(
  $'R1_cap5\tsweep_04_longer_window\tcap_5\t'
  $'R1_cap10\tsweep_04_longer_window\tcap_10\t'
  $'R2_drop10\tsweep_05a_remedy_R2_drop10\tcap_inf\texclude_top10pct.parquet'
  $'R2_drop20\tsweep_05b_remedy_R2_drop20\tcap_inf\texclude_top20pct.parquet'
  $'R2_drop30\tsweep_05d_remedy_R2_drop30\tcap_inf\texclude_top30pct.parquet'
  $'R2_drop40\tsweep_05e_remedy_R2_drop40\tcap_inf\texclude_top40pct.parquet'
  $'R3_ipw\tsweep_05c_remedy_R3_ipw\tcap_inf\t'
  $'R3_ipw_sqrt\tsweep_05f_remedy_R3_ipw_sqrt\tcap_inf\t'
  $'R3_ipw_inv\tsweep_05g_remedy_R3_ipw_inv\tcap_inf\t'
)

select_completed_cell() {
  local train_root="$1" pattern="$2" candidate
  for candidate in "$train_root"/04_train/$pattern; do
    if [[ -f "$candidate/stage_info.txt" \
      && -f "$candidate/training_config.json" \
      && -f "$candidate/predictions/holdout_unseen_users.parquet" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

baseline_root="$stage1/sweep_04_longer_window/cap_inf"
likes="$(find "$stage1/01_get_data" -name 'likes_core_*.parquet' -print -quit)"
[[ -n "$likes" ]] || { echo "Missing likes_core parquet" >&2; exit 66; }
summary_csv="$out/reports/fixed_cohort_auc.csv"

baseline_for_seed() {
  select_completed_cell "$baseline_root" "*_mlp_*_seed${1}_cap_inf"
}

remedy_for_seed() {
  local sweep="$1" cap="$2" seed="$3"
  select_completed_cell "$stage1/$sweep/$cap" "*_mlp_*_seed${seed}_*"
}

record_prediction_hash() {
  local prediction="$1"
  [[ -s "$prediction" ]] || { echo "Missing or empty prediction: $prediction" >&2; exit 1; }
  sha256sum "$prediction" > "${prediction}.sha256"
}

score_on_baseline() {
  local condition="$1" sweep="$2" cap="$3" seed="$4"
  local baseline remedy scored prediction
  baseline="$(baseline_for_seed "$seed")" || {
    echo "Missing verified baseline MLP seed $seed" >&2; return 1;
  }
  remedy="$(remedy_for_seed "$sweep" "$cap" "$seed")" || {
    echo "Missing verified $condition MLP seed $seed" >&2; return 1;
  }
  scored="$remedy/paper_quality/on_baseline_substrate"
  prediction="$scored/holdout_unseen_users.parquet"
  mkdir -p "$scored"
  echo "=== $(date -Is) GPU score $condition seed $seed ===" | tee -a "$out/gpu_harvest.log"
  python3 scripts/run_holdout_pred.py "$remedy" --holdout-type unseen_users --device cuda \
    --substrate-run-dir "$baseline_root" --output-dir "$scored" 2>&1 | tee -a "$out/gpu_harvest.log"
  record_prediction_hash "$prediction"
}

cpu_harvest_cell() {
  local condition="$1" sweep="$2" cap="$3" seed="$4" exclusion="${5:-}"
  local baseline remedy eval_dir scored
  baseline="$(baseline_for_seed "$seed")" || {
    echo "Missing verified baseline MLP seed $seed" >&2; return 1;
  }
  remedy="$(remedy_for_seed "$sweep" "$cap" "$seed")" || {
    echo "Missing verified $condition MLP seed $seed" >&2; return 1;
  }
  scored="$remedy/paper_quality/on_baseline_substrate"
  [[ -s "$scored/holdout_unseen_users.parquet" && -f "$scored/holdout_unseen_users.parquet.sha256" ]] || {
    echo "Missing verified baseline-substrate prediction for $condition seed $seed" >&2
    return 1
  }
  python3 scripts/run_sweep_eval.py "$baseline_root" --max-workers 1
  python3 scripts/run_sweep_eval.py "$(dirname "$(dirname "$remedy")")" --max-workers 1
  eval_dir="$(ls -dt "$remedy"/evals/* | head -1)"
  python3 ops/power_likers_portable/generate_paper_quality_artifacts.py \
    --run-dir "$(dirname "$(dirname "$remedy")")" --eval-dir "$eval_dir" \
    --predictions "$remedy/predictions/holdout_unseen_users.parquet" \
    --likes-core "$likes" --out-dir "$remedy/paper_quality"
  PL_SOURCE_COMMIT="$source_commit" python3 ops/power_likers_portable/fixed_cohort_auc.py \
    --baseline "$baseline/predictions/holdout_unseen_users.parquet" \
    --remedy "$scored/holdout_unseen_users.parquet" --likes-core "$likes" \
    --out "$scored/fixed_cohort_auc.json" --condition "$condition" \
    --architecture mlp --model-seed "$seed" --summary-csv "$summary_csv" \
    --bootstrap-repetitions "$bootstrap_repetitions" \
    ${exclusion:+--exclusion-file "$exclusions/$exclusion"}
  python3 ops/power_likers_portable/emit_cell_manifest.py \
    --cell-dir "$remedy" --predictions "$remedy/predictions/holdout_unseen_users.parquet" \
    --likes-core "$likes" --substrate-id "$(basename "$stage1")" \
    ${exclusion:+--exclusion-file "$exclusions/$exclusion"}
}

is_report_for_condition() {
  local report="$1" condition="$2"
  python3 - "$report" "$condition" <<'PY'
import json
import sys

provenance = json.load(open(sys.argv[1])).get("provenance", {})
raise SystemExit(
    0 if provenance.get("condition") == sys.argv[2]
    and provenance.get("architecture") == "mlp" else 1
)
PY
}

write_matrix_scope() {
  python3 - "$out/reports/matrix_scope.json" "$source_commit" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

destination, source_commit = sys.argv[1:]
included = [
    "R1_cap5", "R1_cap10", "R2_drop10", "R2_drop20", "R2_drop30",
    "R2_drop40", "R3_ipw", "R3_ipw_sqrt", "R3_ipw_inv",
]
two_tower = [
    {"condition": condition, "architecture": "two_tower", "seeds": [1, 2, 3]}
    for condition in (
        "baseline", "R2_drop30", "R2_drop40", "R3_ipw_sqrt",
        "R3_ipw_inv", "F1_footprint126",
    )
]
payload = {
    "architecture_scope": "mlp",
    "harvest_scope": "nine_completed_conditions",
    "source_commit": source_commit,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "primary_architecture": "mlp",
    "included_mlp_conditions": included,
    "deferred_f1_cells": [
        {"condition": "F1_footprint126", "architecture": "mlp", "seed": seed}
        for seed in range(1, 6)
    ],
    "deferred_f1_reason": (
        "F1 training was intentionally stopped before any verified holdout "
        "prediction existed so the completed nine-condition matrix could be "
        "harvested and interpreted first."
    ),
    "deferred_two_tower_cells": two_tower,
    "deferred_two_tower_cell_count": 18,
    "deferred_two_tower_reason": (
        "Two-tower training and scoring are deferred pending the MLP-primary "
        "report."
    ),
}
Path(destination).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

if [[ "$phase" == "dry-run" ]]; then
  completed=0
  for row in "${conditions[@]}"; do
    IFS=$'\t' read -r condition sweep cap exclusion <<< "$row"
    for seed in 1 2 3 4 5; do
      baseline="$(baseline_for_seed "$seed")" || {
        echo "Missing verified baseline MLP seed $seed" >&2; exit 1;
      }
      remedy="$(remedy_for_seed "$sweep" "$cap" "$seed")" || {
        echo "Missing verified $condition MLP seed $seed" >&2; exit 1;
      }
      printf '%s\tseed=%s\tbaseline=%s\tremedy=%s\n' \
        "$condition" "$seed" "$(basename "$baseline")" "$(basename "$remedy")"
      ((completed += 1))
    done
  done
  [[ "$completed" == 45 ]] || {
    echo "Expected 45 completed MLP remedy cells, found $completed" >&2
    exit 1
  }
  echo "Dry run verified 45 completed MLP remedy cells; F1 and two-tower excluded."
  exit 0
fi

if [[ "$phase" == "all" || "$phase" == "gpu" ]]; then
  for row in "${conditions[@]}"; do
    IFS=$'\t' read -r condition sweep cap exclusion <<< "$row"
    for seed in 1 2 3 4 5; do
      score_on_baseline "$condition" "$sweep" "$cap" "$seed"
    done
  done
fi

if [[ "$phase" == "all" || "$phase" == "cpu" ]]; then
  for seed in 1 2 3 4 5; do
    baseline="$(baseline_for_seed "$seed")" || {
      echo "Missing verified baseline MLP seed $seed" >&2; exit 1;
    }
    python3 scripts/run_sweep_eval.py "$baseline_root" --max-workers 1
    eval_dir="$(ls -dt "$baseline"/evals/* | head -1)"
    python3 ops/power_likers_portable/generate_paper_quality_artifacts.py \
      --run-dir "$baseline_root" --eval-dir "$eval_dir" \
      --predictions "$baseline/predictions/holdout_unseen_users.parquet" \
      --likes-core "$likes" --out-dir "$baseline/paper_quality"
  done
  for row in "${conditions[@]}"; do
    IFS=$'\t' read -r condition sweep cap exclusion <<< "$row"
    for seed in 1 2 3 4 5; do
      cpu_harvest_cell "$condition" "$sweep" "$cap" "$seed" "$exclusion"
    done
  done
  for row in "${conditions[@]}"; do
    IFS=$'\t' read -r condition _ <<< "$row"
    reports=()
    while IFS= read -r report; do
      is_report_for_condition "$report" "$condition" && reports+=("$report")
    done < <(find "$stage1" -path "*/paper_quality/on_baseline_substrate/fixed_cohort_auc.json" -print)
    (( ${#reports[@]} == 5 )) || {
      echo "Expected five MLP reports for $condition, found ${#reports[@]}" >&2
      exit 1
    }
    python3 ops/power_likers_portable/paired_auc_report.py "${reports[@]}" \
      --architecture mlp --out "$out/reports/${condition}_typical_paired_auc.json"
  done
  stage_summary="$(find "$stage1/01_get_data" -name summary.json -print -quit)"
  [[ -n "$stage_summary" ]] || {
    echo "Missing required Stage-1 summary.json under $stage1/01_get_data; " \
      "refusing to claim an attrition ledger was emitted." >&2
    exit 66
  }
  python3 ops/power_likers_portable/build_attrition_ledger.py \
    --stage1-summary "$stage_summary" --cells-root "$stage1" --out-dir "$out/reports"
  write_matrix_scope
fi

echo "MLP harvest phase $phase complete: $out" | tee -a "$out/harvest.log"
