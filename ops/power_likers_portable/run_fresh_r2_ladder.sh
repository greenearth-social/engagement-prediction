#!/usr/bin/env bash
# Fresh MLP-only R2 dose-response, with seed-matched baseline-substrate scoring.
set -euo pipefail
repo="${PL_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
stage="${PL_FRESH_STAGE1_ROOT:-$repo/outputs/fresh2026q3_full_20260726_1815}"
private="${PL_FRESH_EXCLUSIONS_DIR:-/mnt/data/wm.s.schulz/private/fresh2026q3_r2_exclusions_20260726}"
out="${PL_R2_LADDER_OUT:-$stage/fresh_r2_ladder_20260727_0210}"
conditions="${PL_R2_CONDITIONS:-10 20 40}"
python_bin="${PL_PYTHON_BIN:-python3}"
[[ -x "$python_bin" ]] && export PATH="$(dirname "$python_bin"):$PATH"
[[ -d "$stage/01_get_data" && -d "$private" ]] || { echo 'fresh inputs missing' >&2; exit 66; }
mkdir -p "$out"/{configs,pairs,logs,tmp}
log="$out/ladder.log"
cd "$repo"
source_yml(){ case "$1" in 10) echo "$repo/sweeps/05a_remedy_R2_drop10.yml";;20) echo "$repo/sweeps/05b_remedy_R2_drop20.yml";;40) echo "$repo/sweeps/05e_remedy_R2_drop40.yml";;*) exit 64;;esac; }
base_cell(){ local s="$1"; if [[ "$s" == 1 ]]; then ls -dt "$stage"/sweep_fresh_targeted_baseline/cap_inf/04_train/*_mlp_summarized_ema_seed1_cap_inf | awk 'NR==1'; else ls -dt "$stage"/sweep_fresh_replication_baseline_seed${s}/cap_inf/04_train/*_mlp_summarized_ema_seed${s}_cap_inf | awk 'NR==1'; fi; }
make_config(){ local pct="$1" seed="$2"; local cfg="$out/configs/r2_drop${pct}_seed${seed}.yml"; python3 - "$(source_yml "$pct")" "$stage" "$private/exclude_top${pct}pct.parquet" "$pct" "$seed" "$cfg" <<'PY'
import sys
from pathlib import Path
import yaml
src,stage,exclude,pct,seed,dest=map(Path,sys.argv[1:]); seed=int(str(seed)); pct=str(pct)
c=yaml.safe_load(src.read_text()); c['sweep_name']=f'fresh_r2_drop{pct}_seed{seed}'; c['ingestion_run']=str(stage); c['caps']=[None]; c['architectures']=[x for x in c['architectures'] if x['model_type']=='mlp'][:1]; c['seeds']=[seed]; c['max_parallel_mlp']=1
extra=list(c.get('extra_cli_args',[]))
def set_arg(k,v):
 if k in extra: extra[extra.index(k)+1]=v
 else: extra.extend([k,v])
for k,v in [('--train-start','2026-05-22'),('--val-start','2026-07-07'),('--holdout-start','2026-07-12'),('--holdout-user-fraction','0.0909'),('--exclude-users-file',str(exclude)),('--num-dataloader-workers','0'),('--experiment-tracker','none')]: set_arg(k,v)
c['extra_cli_args']=extra; dest.write_text(yaml.safe_dump(c,sort_keys=False))
PY
}
validate_resume(){ python3 - "$1" <<'PY'
import hashlib,json,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text())
for raw,want in x['sha256'].items():
 p=Path(raw)
 if not p.is_file(): raise SystemExit(1)
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if h!=want: raise SystemExit(1)
PY
}
run_seed(){ local pct="$1" seed="$2" man="$out/pairs/drop${pct}_seed${seed}.json"; [[ -f "$man" ]] && validate_resume "$man" && { echo "[$(date -Is)] drop${pct} seed${seed} resume skip" | tee -a "$log"; return; }
make_config "$pct" "$seed"; echo "[$(date -Is)] drop${pct} seed${seed} training" | tee -a "$log"; bash run_cap_arch_sweep.sh "$out/configs/r2_drop${pct}_seed${seed}.yml" 2>&1 | tee -a "$log"
local r2 base substrate pred report likes
r2="$(ls -dt "$stage"/sweep_fresh_r2_drop${pct}_seed${seed}/cap_inf/04_train/*_mlp_summarized_ema_seed${seed}_cap_inf | awk 'NR==1')"; base="$(base_cell "$seed")"; substrate="$(dirname "$(dirname "$base")")"; pred="$out/pairs/drop${pct}_seed${seed}_on_baseline"; report="$out/pairs/fixed_cohort_drop${pct}_seed${seed}.json"; likes="$(ls "$stage"/01_get_data/*/likes_core_*.parquet | awk 'NR==1')"
python3 scripts/run_holdout_pred.py "$r2" --holdout-type unseen_users --device cuda --substrate-run-dir "$substrate" --output-dir "$pred" 2>&1 | tee -a "$log"
python3 - "$base/predictions/holdout_unseen_users.parquet" "$pred/holdout_unseen_users.parquet" <<'PY'
import sys,polars as pl
b,r=map(pl.read_parquet,sys.argv[1:]); k=['did','post_id','y_true']; assert b.height==r.height and b.height>0; assert b.select(k).equals(r.select(k)), 'paired keys mismatch'
PY
python3 ops/power_likers_portable/fixed_cohort_auc.py --baseline "$base/predictions/holdout_unseen_users.parquet" --remedy "$pred/holdout_unseen_users.parquet" --likes-core "$likes" --out "$report" 2>&1 | tee -a "$log"
python3 - "$man" "$repo" "$base" "$r2" "$report" "$pred/holdout_unseen_users.parquet" "$out/configs/r2_drop${pct}_seed${seed}.yml" <<'PY'
import hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
m,repo,base,r2,*files=map(Path,sys.argv[1:]); files=[base/'checkpoints/engagement_model_best.pth',r2/'checkpoints/engagement_model_best.pth',base/'predictions/holdout_unseen_users.parquet',*files]
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert all(p.is_file() for p in files)
m.write_text(json.dumps({'created_at':datetime.now(timezone.utc).isoformat(),'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'baseline_cell':str(base),'r2_cell':str(r2),'sha256':{str(p):sha(p) for p in files}},indent=2)+'\n')
PY
}
for pct in $conditions; do for seed in 1 2 3 4 5; do run_seed "$pct" "$seed"; done; done
