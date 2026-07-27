#!/usr/bin/env bash
# Produce cohort-pinned synthetic-feed bias artifacts for the completed
# old-data MLP matrix.  One GPU worker is intentional: synthetic feed scoring
# holds substantial pool/history state in memory.
set -euo pipefail
shopt -s nullglob

repo="${PL_REPO_ROOT:-$HOME/power-likers/code/engagement-prediction}"
stage1="${PL_STAGE1_ROOT:-$HOME/power-likers/stage1/0015_stage1_fixc_v2_20260512_054231}"
out="${PL_FIXED_BIAS_OUT:-$HOME/power-likers/full_matrix/fixed_cohort_bias_$(date -u +%Y%m%dT%H%M%SZ)}"
device="${PL_FIXED_BIAS_DEVICE:-cuda}"

mkdir -p "$out"
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

baseline_root="$stage1/sweep_04_longer_window/cap_inf"
pool_baseline="$(select_completed_cell "$baseline_root" "*_mlp_*_seed1_cap_inf")" || {
  echo "Missing baseline MLP seed 1" >&2; exit 1;
}
pool_eval="$(ls -dt "$pool_baseline"/evals/* | head -1)"
[[ -n "$pool_eval" ]] || { echo "Missing baseline synthetic-feed eval" >&2; exit 1; }

run_cell() {
  local condition="$1" seed="$2" cell="$3" predictions="$4" output="$5"
  if [[ -f "$output/fixed_cohort_bias_manifest.json" ]]; then
    echo "[$condition seed $seed] already complete — skipping"
    return 0
  fi
  echo "=== $(date -Is) $condition seed $seed ===" | tee -a "$out/fixed_bias.log"
  python3 ops/power_likers_portable/fixed_cohort_synthetic_feed.py \
    --train-cell-dir "$cell" --baseline-run-dir "$baseline_root" \
    --baseline-eval-dir "$pool_eval" --predictions "$predictions" \
    --likes-core "$likes" --out-dir "$output" --condition "$condition" \
    --model-seed "$seed" --device "$device" \
    2>&1 | tee -a "$out/fixed_bias.log"
}

for seed in 1 2 3 4 5; do
  baseline="$(select_completed_cell "$baseline_root" "*_mlp_*_seed${seed}_cap_inf")" || {
    echo "Missing baseline MLP seed $seed" >&2; exit 1;
  }
  run_cell "baseline" "$seed" "$baseline" \
    "$baseline/predictions/holdout_unseen_users.parquet" \
    "$baseline/paper_quality/fixed_cohort_bias"
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
  run_root="$stage1/$sweep/$cap"
  for seed in 1 2 3 4 5; do
    cell="$(select_completed_cell "$run_root" "*_mlp_*_seed${seed}_*")" || {
      echo "Missing $condition MLP seed $seed" >&2; exit 1;
    }
    scored="$cell/paper_quality/on_baseline_substrate"
    run_cell "$condition" "$seed" "$cell" \
      "$scored/holdout_unseen_users.parquet" \
      "$scored/fixed_cohort_bias"
  done
done

echo "Fixed-cohort bias matrix complete: $out" | tee -a "$out/fixed_bias.log"
