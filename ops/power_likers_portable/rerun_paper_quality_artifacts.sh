#!/usr/bin/env bash
# Rebuild D1/D2 artifacts after post-processor fixes without rerunning models,
# synthetic-feed scoring, or paired AUC.  This intentionally covers only the
# completed MLP matrix on the frozen Stage-1 substrate.
set -euo pipefail
shopt -s nullglob

repo="${PL_REPO_ROOT:-$HOME/power-likers/code/engagement-prediction}"
stage1="${PL_STAGE1_ROOT:-$HOME/power-likers/stage1/0015_stage1_fixc_v2_20260512_054231}"
log_dir="${PL_PAPER_QUALITY_RERUN_LOG_DIR:-$stage1/paper_quality_rerun_logs}"
python_bin="${PL_PYTHON_BIN:-python3}"

mkdir -p "$log_dir"
cd "$repo"

likes="$(find "$stage1/01_get_data" -name 'likes_core_*.parquet' -print -quit)"
[[ -n "$likes" ]] || { echo "Missing likes_core parquet under $stage1" >&2; exit 66; }

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

process_cell() {
  local label="$1" run_dir="$2" cell="$3" eval_dir
  eval_dir="$(ls -dt "$cell"/evals/* | head -1)"
  [[ -n "$eval_dir" ]] || {
    echo "Missing synthetic-feed eval for $label ($cell)" >&2
    return 1
  }
  echo "=== $(date -Is) $label $(basename "$cell") ===" | tee -a "$log_dir/rerun.log"
  "$python_bin" ops/power_likers_portable/generate_paper_quality_artifacts.py \
    --run-dir "$run_dir" --eval-dir "$eval_dir" \
    --predictions "$cell/predictions/holdout_unseen_users.parquet" \
    --likes-core "$likes" --out-dir "$cell/paper_quality" \
    2>&1 | tee -a "$log_dir/rerun.log"
}

baseline_root="$stage1/sweep_04_longer_window/cap_inf"
for seed in 1 2 3 4 5; do
  cell="$(select_completed_cell "$baseline_root" "*_mlp_*_seed${seed}_cap_inf")" || {
    echo "Missing baseline MLP seed $seed" >&2; exit 1;
  }
  process_cell "baseline_seed${seed}" "$baseline_root" "$cell"
done

conditions=(
  $'R1_cap5\tsweep_04_longer_window\tcap_5\t'
  $'R1_cap10\tsweep_04_longer_window\tcap_10\t'
  $'R2_drop10\tsweep_05a_remedy_R2_drop10\tcap_inf\t'
  $'R2_drop20\tsweep_05b_remedy_R2_drop20\tcap_inf\t'
  $'R2_drop30\tsweep_05d_remedy_R2_drop30\tcap_inf\t'
  $'R2_drop40\tsweep_05e_remedy_R2_drop40\tcap_inf\t'
  $'R3_ipw\tsweep_05c_remedy_R3_ipw\tcap_inf\t'
  $'R3_ipw_sqrt\tsweep_05f_remedy_R3_ipw_sqrt\tcap_inf\t'
  $'R3_ipw_inv\tsweep_05g_remedy_R3_ipw_inv\tcap_inf\t'
)

for row in "${conditions[@]}"; do
  IFS=$'\t' read -r condition sweep cap <<< "$row"
  run_dir="$stage1/$sweep/$cap"
  for seed in 1 2 3 4 5; do
    cell="$(select_completed_cell "$run_dir" "*_mlp_*_seed${seed}_*")" || {
      echo "Missing $condition MLP seed $seed" >&2; exit 1;
    }
    process_cell "${condition}_seed${seed}" "$run_dir" "$cell"
  done
done

echo "Paper-quality rerun complete: $stage1" | tee -a "$log_dir/rerun.log"
