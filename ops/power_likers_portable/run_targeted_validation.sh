#!/usr/bin/env bash
# Run the low-cost gate before committing to the complete Power Likers matrix.
#
# Expected layout after bootstrap:
#   ~/power-likers/code/engagement-prediction
#   ~/power-likers/stage1/<run>/01_get_data/<stamp>
#   ~/power-likers/private/exclusions/exclude_top30pct.parquet
#
# The R2 checkpoint is deliberately re-scored on the baseline substrate.  That
# gives both models exactly the same holdout users and prediction rows, which
# the legacy native-R2 AUC did not guarantee.
set -euo pipefail

repo="${PL_REPO_ROOT:-$HOME/power-likers/code/engagement-prediction}"
stage1="${PL_STAGE1_ROOT:-$HOME/power-likers/stage1/0015_stage1_fixc_v2_20260512_054231}"
exclusions="${PL_EXCLUSIONS_DIR:-$HOME/power-likers/private/exclusions}"
out="${PL_VALIDATION_OUT:-$HOME/power-likers/validation}"
# `04_longer_window.yml` exists only in the archived GPU sweep package, which
# is restored alongside the repo at ~/power-likers/code/sweeps.
baseline_config="${PL_BASELINE_CONFIG:-$HOME/power-likers/code/sweeps/04_longer_window.yml}"
r2_config="${PL_R2_CONFIG:-$HOME/power-likers/code/sweeps/05d_remedy_R2_drop30.yml}"

[[ -d "$repo" && -d "$stage1/01_get_data" && -f "$exclusions/exclude_top30pct.parquet" ]] || {
  echo "Missing repo, Stage-1 substrate, or private R2 exclusion list." >&2
  exit 66
}

cd "$repo"
mkdir -p "$out"
if git rev-parse HEAD > "$out/code_commit.txt" 2>/dev/null; then
  cat "$out/code_commit.txt"
else
  # Portability exports deliberately omit .git. Preserve the commit recorded
  # in the signed export index instead of failing the validation gate.
  python - "$HOME/power-likers/SHA256SUMS.json" > "$out/code_commit.txt" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text())["source_commit"])
PY
  cat "$out/code_commit.txt"
fi
sha256sum "$baseline_config" "$r2_config" "$exclusions/exclude_top30pct.parquet" > "$out/input_sha256sums.txt"

# Use a one-seed copy of each archived YAML.  The original configs reference
# GPU-local paths, so replace only the portable Stage-1 and exclusion paths.
python - "$baseline_config" "$r2_config" "$stage1" "$exclusions/exclude_top30pct.parquet" "$out" <<'PY'
import sys
from pathlib import Path
import yaml

baseline, r2, stage1, exclusion, out = map(Path, sys.argv[1:])
for source, name in ((baseline, "baseline"), (r2, "r2_drop30")):
    config = yaml.safe_load(source.read_text())
    config["ingestion_run"] = str(stage1)
    config["caps"] = [None]
    config["architectures"] = [config["architectures"][0]]  # MLP only
    config["seeds"] = [1]
    extra = config.get("extra_cli_args", [])
    config["extra_cli_args"] = [
        str(exclusion) if value == "/mnt/data/wm.s.schulz/modules/engagement-prediction/analyses/remedies/artifacts/exclude_top30pct.parquet" else value
        for value in extra
    ]
    if "--experiment-tracker" not in config["extra_cli_args"]:
        config["extra_cli_args"].extend(["--experiment-tracker", "none"])
    (out / f"{name}.yml").write_text(yaml.safe_dump(config, sort_keys=False))
PY

bash run_cap_arch_sweep.sh "$out/baseline.yml"
bash run_cap_arch_sweep.sh "$out/r2_drop30.yml"

baseline_cell="$(ls -dt "$stage1"/sweep_04_longer_window/cap_inf/04_train/*_mlp_summarized_ema_seed1_cap_inf | head -1)"
r2_cell="$(ls -dt "$stage1"/sweep_05d_remedy_R2_drop30/cap_inf/04_train/*_mlp_summarized_ema_seed1_cap_inf | head -1)"
baseline_substrate="$(dirname "$(dirname "$baseline_cell")")"

# Native baseline predictions define the cohort.  The R2 model is scored on
# that same substrate, to its own output directory, so it cannot overwrite
# its native predictions.
python scripts/run_holdout_pred.py "$baseline_cell" --holdout-type unseen_users --device cuda
python scripts/run_holdout_pred.py "$r2_cell" --holdout-type unseen_users --device cuda \
  --substrate-run-dir "$baseline_substrate" --output-dir "$out/r2_on_baseline_substrate"

likes="$(find "$stage1/01_get_data" -name 'likes_core_*.parquet' -print -quit)"
python ops/power_likers_portable/fixed_cohort_auc.py \
  --baseline "$baseline_cell/predictions/holdout_unseen_users.parquet" \
  --remedy "$out/r2_on_baseline_substrate/holdout_unseen_users.parquet" \
  --likes-core "$likes" \
  --out "$out/fixed_cohort_auc.json"

echo "Validation outputs: $out"
