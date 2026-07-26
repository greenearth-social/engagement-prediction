#!/usr/bin/env bash
# Execute the preregistered, paper-cited Power Likers matrix on the portable VM.
#
# This script is restart-safe at the cell level: run_cap_arch_sweep.sh skips
# existing prediction parquets and run_sweep_eval.py skips existing bias
# exports.  All model scoring is sequential after fitting to keep peak RSS
# bounded.
set -euo pipefail

repo="${PL_REPO_ROOT:-$HOME/power-likers/code/engagement-prediction}"
stage1="${PL_STAGE1_ROOT:-$HOME/power-likers/stage1/0015_stage1_fixc_v2_20260512_054231}"
exclusions="${PL_EXCLUSIONS_DIR:-$HOME/power-likers/private/exclusions}"
archive_sweeps="${PL_SWEEPS_ROOT:-$HOME/power-likers/code/sweeps}"
out="${PL_FULL_MATRIX_OUT:-$HOME/power-likers/full_matrix}"
source_commit="${PL_SOURCE_COMMIT:-unknown}"
mkdir -p "$out/configs" "$out/reports"
cd "$repo"

[[ -d "$stage1/01_get_data" && -d "$archive_sweeps" ]] || {
  echo "Missing Stage-1 substrate or archived sweep configs." >&2
  exit 66
}

write_config() {
  local source="$1" label="$2" architecture="$3" seeds="$4" caps="$5"
  python3 - "$archive_sweeps/$source.yml" "$out/configs/$label.yml" \
    "$stage1" "$exclusions" "$architecture" "$seeds" "$caps" <<'PY'
import json, sys
from pathlib import Path
import yaml

source, destination, stage1, exclusions, architecture, seeds, caps = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text())
config["ingestion_run"] = stage1
config["architectures"] = [
    item for item in config["architectures"]
    if item["model_type"] == architecture
]
assert len(config["architectures"]) == 1, (source, architecture)
config["seeds"] = [int(seed) for seed in seeds.split(",") if seed]
config["caps"] = [None if cap == "inf" else int(cap) for cap in caps.split(",")]
config["max_parallel_mlp"] = int(
    __import__("os").environ.get("PL_MAX_PARALLEL_MLP", "2")
)
extra = config.setdefault("extra_cli_args", [])
for old, replacement in {
    "/mnt/data/wm.s.schulz/modules/engagement-prediction/analyses/remedies/artifacts/exclude_top10pct.parquet": f"{exclusions}/exclude_top10pct.parquet",
    "/mnt/data/wm.s.schulz/modules/engagement-prediction/analyses/remedies/artifacts/exclude_top20pct.parquet": f"{exclusions}/exclude_top20pct.parquet",
    "/mnt/data/wm.s.schulz/modules/engagement-prediction/analyses/remedies/artifacts/exclude_top30pct.parquet": f"{exclusions}/exclude_top30pct.parquet",
    "/mnt/data/wm.s.schulz/modules/engagement-prediction/analyses/remedies/artifacts/exclude_top40pct.parquet": f"{exclusions}/exclude_top40pct.parquet",
}.items():
    extra[:] = [replacement if item == old else item for item in extra]
if "--experiment-tracker" not in extra:
    extra.extend(["--experiment-tracker", "none"])
Path(destination).write_text(yaml.safe_dump(config, sort_keys=False))
PY
}

run_config() {
  local name="$1"
  echo "=== $(date -Is) running $name ===" | tee -a "$out/full_matrix.log"
  bash run_cap_arch_sweep.sh "$out/configs/$name.yml" 2>&1 | tee -a "$out/full_matrix.log"
}

# MLP: five paired seeds for every paper-cited condition. Existing portable
# baseline/R2 seed 1 are not retrained.
write_config 04_longer_window baseline_mlp mlp 2,3,4,5 inf
write_config 04_longer_window r1_mlp mlp 1,2,3,4,5 5,10
write_config 05a_remedy_R2_drop10 r2d10_mlp mlp 1,2,3,4,5 inf
write_config 05b_remedy_R2_drop20 r2d20_mlp mlp 1,2,3,4,5 inf
write_config 05d_remedy_R2_drop30 r2d30_mlp mlp 2,3,4,5 inf
write_config 05e_remedy_R2_drop40 r2d40_mlp mlp 1,2,3,4,5 inf
write_config 05c_remedy_R3_ipw r3log_mlp mlp 1,2,3,4,5 inf
write_config 05f_remedy_R3_ipw_sqrt r3sqrt_mlp mlp 1,2,3,4,5 inf
write_config 05g_remedy_R3_ipw_inv r3inv_mlp mlp 1,2,3,4,5 inf
write_config 05h_remedy_F1_footprint126 f1_mlp mlp 1,2,3,4,5 inf

for config in baseline_mlp r1_mlp r2d10_mlp r2d20_mlp r2d30_mlp r2d40_mlp r3log_mlp r3sqrt_mlp r3inv_mlp f1_mlp; do
  run_config "$config"
done

# Table A5 robustness cells: exactly three two-tower seeds for the six
# conditions reported in the manuscript.
write_config 04_longer_window baseline_tt two-tower 1,2,3 inf
write_config 05d_remedy_R2_drop30 r2d30_tt two-tower 1,2,3 inf
write_config 05e_remedy_R2_drop40 r2d40_tt two-tower 1,2,3 inf
write_config 05f_remedy_R3_ipw_sqrt r3sqrt_tt two-tower 1,2,3 inf
write_config 05g_remedy_R3_ipw_inv r3inv_tt two-tower 1,2,3 inf
write_config 05h_remedy_F1_footprint126 f1_tt two-tower 1,2,3 inf
for config in baseline_tt r2d30_tt r2d40_tt r3sqrt_tt r3inv_tt f1_tt; do
  run_config "$config"
done

likes="$(find "$stage1/01_get_data" -name 'likes_core_*.parquet' -print -quit)"
baseline_root="$stage1/sweep_04_longer_window/cap_inf"
baseline_cells="$baseline_root/04_train"
summary_csv="$out/reports/fixed_cohort_auc.csv"

for seed in 1 2 3 4 5; do
  baseline="$(ls -dt "$baseline_cells"/*_mlp_*_seed"${seed}"_cap_inf 2>/dev/null | head -1)"
  [[ -n "$baseline" ]] || { echo "Missing baseline MLP seed $seed" >&2; exit 1; }
  python3 scripts/run_sweep_eval.py "$baseline_root" --max-workers 1
  eval_dir="$(ls -dt "$baseline"/evals/* | head -1)"
  python3 ops/power_likers_portable/generate_paper_quality_artifacts.py \
    --run-dir "$baseline_root" --eval-dir "$eval_dir" \
    --predictions "$baseline/predictions/holdout_unseen_users.parquet" \
    --likes-core "$likes" --out-dir "$baseline/paper_quality"
done

pair_condition() {
  local condition="$1" arch="$2" sweep="$3" cap="$4" seed="$5" exclusion="${6:-}"
  local baseline remedy eval_dir
  baseline="$(ls -dt "$baseline_cells"/*_"${arch}"_*_seed"${seed}"_cap_inf 2>/dev/null | head -1)"
  remedy="$(ls -dt "$stage1/$sweep/$cap/04_train"/*_"${arch}"_*_seed"${seed}"_* 2>/dev/null | head -1)"
  [[ -n "$baseline" && -n "$remedy" ]] || {
    echo "Missing cell for $condition/$arch/seed$seed" >&2
    return 1
  }
  # Eval is per cap dir and idempotently skips completed cells.  Restricting
  # it here avoids accidentally evaluating unrelated historical sweep cells.
  python3 scripts/run_sweep_eval.py "$(dirname "$(dirname "$baseline")")" --max-workers 1
  python3 scripts/run_sweep_eval.py "$(dirname "$(dirname "$remedy")")" --max-workers 1
  eval_dir="$(ls -dt "$remedy"/evals/* | head -1)"
  python3 ops/power_likers_portable/generate_paper_quality_artifacts.py \
    --run-dir "$(dirname "$(dirname "$remedy")")" --eval-dir "$eval_dir" \
    --predictions "$remedy/predictions/holdout_unseen_users.parquet" \
    --likes-core "$likes" --out-dir "$remedy/paper_quality"
  local scored="$remedy/paper_quality/on_baseline_substrate"
  mkdir -p "$scored"
  python3 scripts/run_holdout_pred.py "$remedy" --holdout-type unseen_users --device cuda \
    --substrate-run-dir "$baseline_root" --output-dir "$scored"
  PL_SOURCE_COMMIT="$source_commit" python3 ops/power_likers_portable/fixed_cohort_auc.py \
    --baseline "$baseline/predictions/holdout_unseen_users.parquet" \
    --remedy "$scored/holdout_unseen_users.parquet" --likes-core "$likes" \
    --out "$scored/fixed_cohort_auc.json" --condition "$condition" \
    --architecture "$arch" --model-seed "$seed" --summary-csv "$summary_csv" \
    ${exclusion:+--exclusion-file "$exclusions/$exclusion"}
  python3 ops/power_likers_portable/emit_cell_manifest.py \
    --cell-dir "$remedy" --predictions "$remedy/predictions/holdout_unseen_users.parquet" \
    --likes-core "$likes" --substrate-id "$(basename "$stage1")" \
    ${exclusion:+--exclusion-file "$exclusions/$exclusion"}
}

# Explicitly enumerate only the conditions chosen for the paper.
for seed in 1 2 3 4 5; do
  pair_condition R1_cap5 mlp sweep_04_longer_window cap_5 "$seed"
  pair_condition R1_cap10 mlp sweep_04_longer_window cap_10 "$seed"
  pair_condition R2_drop10 mlp sweep_05a_remedy_R2_drop10 cap_inf "$seed" exclude_top10pct.parquet
  pair_condition R2_drop20 mlp sweep_05b_remedy_R2_drop20 cap_inf "$seed" exclude_top20pct.parquet
  pair_condition R2_drop30 mlp sweep_05d_remedy_R2_drop30 cap_inf "$seed" exclude_top30pct.parquet
  pair_condition R2_drop40 mlp sweep_05e_remedy_R2_drop40 cap_inf "$seed" exclude_top40pct.parquet
  pair_condition R3_ipw mlp sweep_05c_remedy_R3_ipw cap_inf "$seed"
  pair_condition R3_ipw_sqrt mlp sweep_05f_remedy_R3_ipw_sqrt cap_inf "$seed"
  pair_condition R3_ipw_inv mlp sweep_05g_remedy_R3_ipw_inv cap_inf "$seed"
  pair_condition F1_footprint126 mlp sweep_05h_remedy_F1_footprint126 cap_inf "$seed"
done
for seed in 1 2 3; do
  pair_condition R2_drop30 two_tower sweep_05d_remedy_R2_drop30 cap_inf "$seed" exclude_top30pct.parquet
  pair_condition R2_drop40 two_tower sweep_05e_remedy_R2_drop40 cap_inf "$seed" exclude_top40pct.parquet
  pair_condition R3_ipw_sqrt two_tower sweep_05f_remedy_R3_ipw_sqrt cap_inf "$seed"
  pair_condition R3_ipw_inv two_tower sweep_05g_remedy_R3_ipw_inv cap_inf "$seed"
  pair_condition F1_footprint126 two_tower sweep_05h_remedy_F1_footprint126 cap_inf "$seed"
done

for condition in R1_cap5 R1_cap10 R2_drop10 R2_drop20 R2_drop30 R2_drop40 R3_ipw R3_ipw_sqrt R3_ipw_inv F1_footprint126; do
  mapfile -t reports < <(find "$stage1" -path "*/paper_quality/on_baseline_substrate/fixed_cohort_auc.json" -print)
  selected=()
  for report in "${reports[@]}"; do
    [[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("provenance",{}).get("condition",""))' "$report")" == "$condition" ]] && selected+=("$report")
  done
  (( ${#selected[@]} )) && python3 ops/power_likers_portable/paired_auc_report.py "${selected[@]}" \
    --out "$out/reports/${condition}_typical_paired_auc.json"
done

stage_summary="$(find "$stage1/01_get_data" -name summary.json -print -quit)"
[[ -n "$stage_summary" ]] && python3 ops/power_likers_portable/build_attrition_ledger.py \
  --stage1-summary "$stage_summary" --cells-root "$stage1" --out-dir "$out/reports"

echo "Full matrix complete: $out" | tee -a "$out/full_matrix.log"
