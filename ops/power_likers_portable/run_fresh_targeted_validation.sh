#!/usr/bin/env bash
# Run exactly one fresh baseline/R2-drop30 MLP paired fixed-cohort gate.
set -euo pipefail

repo="${PL_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
stage1="${PL_FRESH_STAGE1_ROOT:-$repo/outputs/fresh2026q3_full_20260726_1815}"
exclusions="${PL_FRESH_EXCLUSIONS_DIR:-/mnt/data/wm.s.schulz/private/fresh2026q3_r2_exclusions_20260726}"
out="${PL_FRESH_VALIDATION_OUT:-$stage1/fresh_targeted_validation}"
baseline_config="${PL_BASELINE_CONFIG:-$repo/sweeps/04_longer_window.yml}"
r2_config="${PL_R2_CONFIG:-$repo/sweeps/05d_remedy_R2_drop30.yml}"
python_bin="${PL_PYTHON_BIN:-python3}"
if [[ -x "$python_bin" ]]; then
  export PATH="$(dirname "$python_bin"):$PATH"
fi

[[ -d "$repo" && -d "$stage1/01_get_data" && -f "$exclusions/exclude_top30pct.parquet" ]] || {
  echo "Missing fresh repo, Stage-1 substrate, or private fresh R2 exclusions." >&2
  exit 66
}
mkdir -p "$out/configs" "$out/logs"
log="$out/validation.log"
gpu_log="$out/gpu_sidecar.log"
sidecar_pid=""
cleanup() { [[ -n "$sidecar_pid" ]] && kill "$sidecar_pid" 2>/dev/null || true; }
trap cleanup EXIT
(
  while true; do date -Is; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader; sleep 30; done
) >> "$gpu_log" 2>&1 &
sidecar_pid=$!

cd "$repo"
"$python_bin" - "$baseline_config" "$r2_config" "$stage1" "$exclusions/exclude_top30pct.parquet" "$out" <<'PY'
import sys
from pathlib import Path
import yaml

baseline, r2, stage1, exclusion, out = map(Path, sys.argv[1:])
def set_arg(items, flag, value):
    if flag in items:
        items[items.index(flag) + 1] = value
    else:
        items.extend([flag, value])
for source, name, exclude in ((baseline, "baseline", None), (r2, "r2_drop30", exclusion)):
    config = yaml.safe_load(source.read_text())
    config["ingestion_run"] = str(stage1)
    config["caps"] = [None]
    config["architectures"] = [item for item in config["architectures"] if item["model_type"] == "mlp"][:1]
    config["seeds"] = [1]
    config["max_parallel_mlp"] = 1
    extra = list(config.get("extra_cli_args", []))
    for flag, value in (("--train-start", "2026-05-22"), ("--val-start", "2026-07-07"), ("--holdout-start", "2026-07-12"), ("--holdout-user-fraction", "0.0909")):
        set_arg(extra, flag, value)
    if exclude is not None:
        set_arg(extra, "--exclude-users-file", str(exclude))
    if "--experiment-tracker" not in extra:
        extra.extend(["--experiment-tracker", "none"])
    config["extra_cli_args"] = extra
    (out / "configs" / f"{name}.yml").write_text(yaml.safe_dump(config, sort_keys=False))
PY

printf '[%s] baseline start\n' "$(date -Is)" | tee -a "$log"
bash run_cap_arch_sweep.sh "$out/configs/baseline.yml" 2>&1 | tee -a "$log"
printf '[%s] R2-drop30 start\n' "$(date -Is)" | tee -a "$log"
bash run_cap_arch_sweep.sh "$out/configs/r2_drop30.yml" 2>&1 | tee -a "$log"

baseline_cell="$(ls -dt "$stage1"/sweep_04_longer_window/cap_inf/04_train/*_mlp_summarized_ema_seed1_cap_inf | head -1)"
r2_cell="$(ls -dt "$stage1"/sweep_05d_remedy_R2_drop30/cap_inf/04_train/*_mlp_summarized_ema_seed1_cap_inf | head -1)"
[[ -n "$baseline_cell" && -n "$r2_cell" ]] || { echo "Missing targeted validation train cell" >&2; exit 1; }
baseline_substrate="$(dirname "$(dirname "$baseline_cell")")"
python3 scripts/run_holdout_pred.py "$baseline_cell" --holdout-type unseen_users --device cuda 2>&1 | tee -a "$log"
python3 scripts/run_holdout_pred.py "$r2_cell" --holdout-type unseen_users --device cuda \
  --substrate-run-dir "$baseline_substrate" --output-dir "$out/r2_on_baseline_substrate" 2>&1 | tee -a "$log"
likes="$(find "$stage1/01_get_data" -name 'likes_core_*.parquet' -print -quit)"
python3 ops/power_likers_portable/fixed_cohort_auc.py \
  --baseline "$baseline_cell/predictions/holdout_unseen_users.parquet" \
  --remedy "$out/r2_on_baseline_substrate/holdout_unseen_users.parquet" \
  --likes-core "$likes" --out "$out/fixed_cohort_auc.json" 2>&1 | tee -a "$log"

python3 - "$stage1" "$exclusions" "$out" "$baseline_cell" "$r2_cell" <<'PY'
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
stage1, exclusions, out, baseline, remedy = map(Path, sys.argv[1:])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
manifest={
  'created_at': datetime.now(timezone.utc).isoformat(),
  'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
  'stage1_root': str(stage1),
  'stage1_manifest_sha256': sha(next((stage1/'01_get_data').glob('*/stage1_manifest.json'))),
  'exclusion_manifest_sha256': sha(exclusions/'exclusion_manifest.json'),
  'exclusion_top30_sha256': sha(exclusions/'exclude_top30pct.parquet'),
  'baseline_config_sha256': sha(out/'configs/baseline.yml'),
  'r2_config_sha256': sha(out/'configs/r2_drop30.yml'),
  'baseline_cell': str(baseline), 'r2_cell': str(remedy),
  'fixed_cohort_report': str(out/'fixed_cohort_auc.json'),
}
(out/'validation_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2))
PY
printf '[%s] fresh targeted validation complete: %s\n' "$(date -Is)" "$out" | tee -a "$log"
