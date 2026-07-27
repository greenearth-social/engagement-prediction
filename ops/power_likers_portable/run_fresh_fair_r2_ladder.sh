#!/usr/bin/env bash
# Resumable five-seed fair R2 ladder: every arm trains on equal real train/val N.
set -euo pipefail
repo="${PL_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
stage="${PL_FRESH_STAGE1_ROOT:-$repo/outputs/fresh2026q3_full_20260726_1815}"
private="${PL_FRESH_EXCLUSIONS_DIR:-/mnt/data/wm.s.schulz/private/fresh2026q3_r2_exclusions_20260726}"
out="${PL_FAIR_R2_OUT:-$stage/fresh_fair_r2_ladder_$(date +%Y%m%d_%H%M)}"
conditions="${PL_FAIR_R2_CONDITIONS:-10 20 30 40}"
seeds="${PL_FAIR_R2_SEEDS:-1 2 3 4 5}"
train_target="${PL_FAIR_R2_TRAIN_ROWS_TARGET:-95000}"; val_target="${PL_FAIR_R2_VAL_ROWS_TARGET:-4000}"
python_bin="${PL_PYTHON_BIN:-python3}"; [[ -x "$python_bin" ]] && export PATH="$(dirname "$python_bin"):$PATH"
[[ -d "$stage/01_get_data" && -d "$private" ]] || { echo "fresh inputs missing" >&2; exit 66; }
mkdir -p "$out"/{configs,pairs}; log="$out/ladder.log"; cd "$repo"
source_yml(){ case "$1" in baseline) echo "$repo/sweeps/06a_baseline_cap_inf_seeds45.yml";;10) echo "$repo/sweeps/05a_remedy_R2_drop10.yml";;20) echo "$repo/sweeps/05b_remedy_R2_drop20.yml";;30) echo "$repo/sweeps/05d_remedy_R2_drop30.yml";;40) echo "$repo/sweeps/05e_remedy_R2_drop40.yml";;*) exit 64;;esac; }
cell(){ local c="$1" s="$2" p; [[ "$c" == baseline ]] && p="sweep_fresh_fair_baseline_seed$s" || p="sweep_fresh_fair_r2_drop${c}_seed$s"; ls -dt "$stage/$p/cap_inf/04_train"/*_mlp_summarized_ema_seed"$s"_cap_inf 2>/dev/null | awk 'NR==1'; }
full(){ local s="$1"; if [[ "$s" == 1 ]]; then ls -dt "$stage/sweep_fresh_targeted_baseline/cap_inf/04_train"/*_mlp_summarized_ema_seed1_cap_inf|awk 'NR==1'; else ls -dt "$stage/sweep_fresh_replication_baseline_seed$s/cap_inf/04_train"/*_mlp_summarized_ema_seed"$s"_cap_inf|awk 'NR==1'; fi; }
config(){ local c="$1" s="$2" f="$out/configs/${c}_seed$s.yml"; "$python_bin" - "$(source_yml "$c")" "$stage" "$private" "$c" "$s" "$f" "$train_target" "$val_target" <<'PY'
import sys
from pathlib import Path
import yaml
src,stage,private,c,s,d,train_n,val_n=map(Path,sys.argv[1:]); c=str(c); s=int(str(s)); x=yaml.safe_load(src.read_text())
x['sweep_name']=f"fresh_fair_{'baseline' if c=='baseline' else 'r2_drop'+c}_seed{s}"; x['ingestion_run']=str(stage); x['caps']=[None]; x['architectures']=[a for a in x['architectures'] if a['model_type']=='mlp'][:1]; x['seeds']=[s]; x['max_parallel_mlp']=1
e=list(x.get('extra_cli_args',[]))
def setarg(k,v):
 if k in e:e[e.index(k)+1]=v
 else:e.extend([k,v])
for k,v in [('--train-start','2026-05-22'),('--val-start','2026-07-07'),('--holdout-start','2026-07-12'),('--holdout-user-fraction','0.0909'),('--train-rows-target',str(train_n)),('--val-rows-target',str(val_n)),('--row-subsample-seed','42'),('--num-dataloader-workers','0'),('--experiment-tracker','none')]:setarg(k,v)
if c!='baseline':setarg('--exclude-users-file',str(private/f'exclude_top{c}pct.parquet'))
x['extra_cli_args']=e; d.write_text(yaml.safe_dump(x,sort_keys=False))
PY
}
assert_counts(){ "$python_bin" - "$1" "$train_target" "$val_target" <<'PY'
import sys,polars as pl
from pathlib import Path
f=sorted((Path(sys.argv[1])/'02_target_posts').glob('*/target_posts_*.parquet'))[-1]; g=pl.scan_parquet(f).group_by('split').len().collect(); d=dict(zip(g['split'],g['len'])); assert d.get('train')==int(sys.argv[2]) and d.get('val')==int(sys.argv[3]),d; print(d)
PY
}
sha_manifest(){ "$python_bin" - "$1" "$2" "$3" "$4" "$5" <<'PY'
import sys,json,hashlib
from pathlib import Path
m,base,model,pred,cfg=map(Path,sys.argv[1:]); ps=[base/'checkpoints/engagement_model_best.pth',model/'checkpoints/engagement_model_best.pth',pred,cfg]
def h(p):
 x=hashlib.sha256(); x.update(p.read_bytes()); return x.hexdigest()
assert all(p.is_file() for p in ps),ps;m.write_text(json.dumps({'sha256':{str(p):h(p) for p in ps}},indent=2)+'\n')
PY
}
run(){ local c="$1"; local s="$2"; local man="$out/pairs/${c}_seed$s.json"; [[ -f "$man" ]] && return; config "$c" "$s"; echo "[$(date -Is)] $c seed$s training"|tee -a "$log"; bash run_cap_arch_sweep.sh "$out/configs/${c}_seed$s.yml" 2>&1|tee -a "$log"; local model; model="$(cell "$c" "$s")"; [[ -n "$model" ]]||exit 1; assert_counts "$(dirname "$(dirname "$model")")"; local fbase substrate pred likes; fbase="$(full "$s")"; substrate="$(dirname "$(dirname "$fbase")")"; likes="$(ls "$stage"/01_get_data/*/likes_core_*.parquet|head -1)"; if [[ "$c" == baseline ]]; then pred="$out/pairs/baseline_fair_seed${s}_on_full/holdout_unseen_users.parquet"; else pred="$out/pairs/drop${c}_seed${s}_on_full/holdout_unseen_users.parquet"; fi; mkdir -p "$(dirname "$pred")"; python3 scripts/run_holdout_pred.py "$model" --holdout-type unseen_users --device cuda --substrate-run-dir "$substrate" --output-dir "$(dirname "$pred")" 2>&1|tee -a "$log"; if [[ "$c" == baseline ]]; then "$python_bin" ops/power_likers_portable/fixed_cohort_auc.py --baseline "$fbase/predictions/holdout_unseen_users.parquet" --remedy "$pred" --likes-core "$likes" --out "$out/pairs/full_vs_fair_seed$s.json"|tee -a "$log"; else local bpred="$out/pairs/baseline_fair_seed${s}_on_full/holdout_unseen_users.parquet"; "$python_bin" ops/power_likers_portable/fixed_cohort_auc.py --baseline "$bpred" --remedy "$pred" --likes-core "$likes" --out "$out/pairs/fixed_cohort_drop${c}_seed$s.json"|tee -a "$log"; fi; sha_manifest "$man" "$fbase" "$model" "$pred" "$out/configs/${c}_seed$s.yml"; }
for s in $seeds; do run baseline "$s"; for c in $conditions; do run "$c" "$s"; done; done
