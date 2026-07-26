#!/usr/bin/env bash
# Bounded fresh MLP baseline/R2-drop30 replication, seeds 2-5 by default.
set -euo pipefail
repo="${PL_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
stage1="${PL_FRESH_STAGE1_ROOT:-$repo/outputs/fresh2026q3_full_20260726_1815}"
exclusions="${PL_FRESH_EXCLUSIONS_DIR:-/mnt/data/wm.s.schulz/private/fresh2026q3_r2_exclusions_20260726}"
out="${PL_FRESH_REPLICATION_OUT:-$stage1/fresh_five_seed_replication_20260726_2320}"
python_bin="${PL_PYTHON_BIN:-python3}"
seeds="${PL_SEEDS:-2 3 4 5}"
[[ -x "$python_bin" ]] && export PATH="$(dirname "$python_bin"):$PATH"
[[ -d "$stage1/01_get_data" && -f "$exclusions/exclude_top30pct.parquet" ]] || { echo 'fresh inputs missing' >&2; exit 66; }
mkdir -p "$out"/{configs,logs,pairs,tmp}
log="$out/replication.log"; gpu_log="$out/gpu_sidecar.log"; sidecar=''
cleanup(){ [[ -n "$sidecar" ]] && kill "$sidecar" 2>/dev/null || true; }
trap cleanup EXIT
(while true; do date -Is; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader; sleep 30; done) >> "$gpu_log" 2>&1 & sidecar=$!
cd "$repo"
make_config(){
  local kind="$1" seed="$2"
  local config="$out/configs/${kind}_seed${seed}.yml"
  python3 - "$repo/sweeps/06a_baseline_cap_inf_seeds45.yml" "$repo/sweeps/05d_remedy_R2_drop30.yml" "$stage1" "$exclusions/exclude_top30pct.parquet" "$kind" "$seed" "$config" <<'PY'
import sys
from pathlib import Path
import yaml
base, r2, stage, exclusion, kind, seed, dest = map(Path, sys.argv[1:])
seed=int(seed); source=base if str(kind)=='baseline' else r2
cfg=yaml.safe_load(source.read_text())
cfg['sweep_name']=f'fresh_replication_{kind}_seed{seed}'
cfg['ingestion_run']=str(stage); cfg['caps']=[None]
cfg['architectures']=[x for x in cfg['architectures'] if x['model_type']=='mlp'][:1]
cfg['seeds']=[seed]; cfg['max_parallel_mlp']=1
extra=list(cfg.get('extra_cli_args', []))
def set_arg(flag,value):
 if flag in extra: extra[extra.index(flag)+1]=value
 else: extra.extend([flag,value])
for flag,value in [('--train-start','2026-05-22'),('--val-start','2026-07-07'),('--holdout-start','2026-07-12'),('--holdout-user-fraction','0.0909'),('--num-dataloader-workers','0'),('--experiment-tracker','none')]: set_arg(flag,value)
if str(kind)=='r2': set_arg('--exclude-users-file',str(exclusion))
cfg['extra_cli_args']=extra
dest.write_text(yaml.safe_dump(cfg,sort_keys=False))
PY
}
sha(){ sha256sum "$1" | awk '{print $1}'; }
run_pair(){
  local seed="$1" manifest="$out/pairs/pair_seed${seed}.json"
  if [[ -f "$manifest" ]] && python3 - "$manifest" <<'PY'
import json,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text())
assert all(Path(p).is_file() for p in x['required_files'])
PY
  then echo "[$(date -Is)] seed $seed validated resume skip" | tee -a "$log"; return; fi
  make_config baseline "$seed"; make_config r2 "$seed"
  echo "[$(date -Is)] seed $seed baseline" | tee -a "$log"
  bash run_cap_arch_sweep.sh "$out/configs/baseline_seed${seed}.yml" 2>&1 | tee -a "$log"
  echo "[$(date -Is)] seed $seed R2-drop30" | tee -a "$log"
  bash run_cap_arch_sweep.sh "$out/configs/r2_seed${seed}.yml" 2>&1 | tee -a "$log"
  local basecell r2cell basesub r2pred likes report
  basecell="$(ls -dt "$stage1"/sweep_fresh_replication_baseline_seed${seed}/cap_inf/04_train/*_mlp_summarized_ema_seed${seed}_cap_inf | head -1)"
  r2cell="$(ls -dt "$stage1"/sweep_fresh_replication_r2_seed${seed}/cap_inf/04_train/*_mlp_summarized_ema_seed${seed}_cap_inf | head -1)"
  basesub="$(dirname "$(dirname "$basecell")")"; r2pred="$out/pairs/r2_seed${seed}_on_baseline"
  python3 scripts/run_holdout_pred.py "$basecell" --holdout-type unseen_users --device cuda 2>&1 | tee -a "$log"
  python3 scripts/run_holdout_pred.py "$r2cell" --holdout-type unseen_users --device cuda --substrate-run-dir "$basesub" --output-dir "$r2pred" 2>&1 | tee -a "$log"
  likes="$(rg --files "$stage1/01_get_data" | rg 'likes_core_.*\\.parquet$' | awk 'NR==1')"
  report="$out/pairs/fixed_cohort_seed${seed}.json"
  python3 - "$basecell/predictions/holdout_unseen_users.parquet" "$r2pred/holdout_unseen_users.parquet" <<'PY'
import sys,polars as pl
b,r=map(pl.read_parquet,sys.argv[1:])
keys=['did','post_id','y_true']
assert b.height and b.height==r.height, (b.height,r.height)
assert b.select(keys).equals(r.select(keys)), 'paired keys differ in order or content'
PY
  python3 ops/power_likers_portable/fixed_cohort_auc.py --baseline "$basecell/predictions/holdout_unseen_users.parquet" --remedy "$r2pred/holdout_unseen_users.parquet" --likes-core "$likes" --out "$report" 2>&1 | tee -a "$log"
  python3 - "$manifest" "$seed" "$basecell" "$r2cell" "$report" "$basecell/checkpoints/engagement_model_best.pth" "$r2cell/checkpoints/engagement_model_best.pth" "$basecell/predictions/holdout_unseen_users.parquet" "$r2pred/holdout_unseen_users.parquet" "$out/configs/baseline_seed${seed}.yml" "$out/configs/r2_seed${seed}.yml" <<'PY'
import hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
m,seed,*files=map(Path,sys.argv[1:]); required=[str(x) for x in files[2:]]
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
d={'seed':int(seed),'created_at':datetime.now(timezone.utc).isoformat(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'baseline_cell':str(files[0]),'r2_cell':str(files[1]),'required_files':required,'sha256':{str(p):sha(p) for p in map(Path,required)}}
m.write_text(json.dumps(d,indent=2)+'\n')
PY
}
for seed in $seeds; do run_pair "$seed"; done
reports=("$stage1/fresh_targeted_validation_20260726_2235/fixed_cohort_auc.json")
for seed in $seeds; do reports+=("$out/pairs/fixed_cohort_seed${seed}.json"); done
python3 ops/power_likers_portable/paired_auc_report.py "${reports[@]}" --stratum typical --out "$out/paired_auc_typical_5seed.json" 2>&1 | tee -a "$log"
printf '[%s] completed bounded five-seed replication\n' "$(date -Is)" | tee -a "$log"
