#!/usr/bin/env bash
#
# Cap × architecture × seed training sweep.
#
# Modeled on `run_training_sweep.sh` but with a two-level loop:
#   - outer loop: --effective-likes-cap levels (Stage 2 + Stage 3 are re-run
#                 once per cap level, sharing a single Stage 1 ingestion via
#                 a symlink so we don't re-download from GCS).
#   - inner loop: model_type × user_encoder × seed cells, with MLPs running
#                 in parallel (up to MAX_PARALLEL_MLP) and two-tower runs
#                 sequentially.
#
# YAML-driven: see sweeps/01_cross_arch_validation.yml for the schema.
#
# Usage (survives SSH disconnect):
#   tmux new-session -d -s cap-arch-sweep './run_cap_arch_sweep.sh sweeps/01_cross_arch_validation.yml'
#   tmux attach -t cap-arch-sweep
#
# Each cap level lives at:
#   <ingestion_run>/sweep_<sweep_name>/<cap_label>/
#     01_get_data -> ../../01_get_data            (symlink, shared ingestion)
#     02_target_posts/<ts>_<cap_label>/...
#     03_user_history/<ts>_<cap_label>/...
#     04_train/<ts>_<run_tag>/...
#       evals/<ts>/...
#
set -euo pipefail

# ── Args ───────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <sweep_config.yml>" >&2
  exit 64
fi

CONFIG_PATH="$1"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Sweep config not found: $CONFIG_PATH" >&2
  exit 66
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_PATH_ABS="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"

# ── Parse YAML via Python helper ───────────────────────────────────────
TMP_DIR="$(mktemp -d)"
SIDECAR_PID=""
cleanup() {
  if [[ -n "${SIDECAR_PID}" ]]; then
    kill "$SIDECAR_PID" 2>/dev/null || true
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT
PLAN_PATH="$TMP_DIR/plan.tsv"

# We need INGESTION_RUN and SWEEP_NAME first to compute SWEEP_ROOT.
INITIAL_VARS="$(python3 scripts/_emit_sweep_plan.py \
  "$CONFIG_PATH_ABS" \
  --sweep-root /tmp/dummy --plan-path "$PLAN_PATH" --mode vars)"
eval "$INITIAL_VARS"

# Resolve INGESTION_RUN to absolute path (relative paths are interpreted
# from the eng_pred_repo root, since that's where the script runs).
if [[ "${INGESTION_RUN:0:1}" != "/" ]]; then
  INGESTION_RUN_ABS="$SCRIPT_DIR/$INGESTION_RUN"
else
  INGESTION_RUN_ABS="$INGESTION_RUN"
fi
if [[ ! -d "$INGESTION_RUN_ABS" ]]; then
  echo "Ingestion run not found: $INGESTION_RUN_ABS" >&2
  echo "  Expected a directory containing 01_get_data/<ts>/..." >&2
  exit 66
fi
if [[ ! -d "$INGESTION_RUN_ABS/01_get_data" ]]; then
  echo "Ingestion run has no 01_get_data subdir: $INGESTION_RUN_ABS" >&2
  exit 66
fi

SWEEP_ROOT="$INGESTION_RUN_ABS/sweep_${SWEEP_NAME}"
mkdir -p "$SWEEP_ROOT"

# Re-emit the plan with the real sweep root path.
python3 scripts/_emit_sweep_plan.py \
  "$CONFIG_PATH_ABS" \
  --sweep-root "$SWEEP_ROOT" --plan-path "$PLAN_PATH" --mode plan > "$PLAN_PATH"

# ── Logging ────────────────────────────────────────────────────────────
LOG_DIR="$SWEEP_ROOT/sweep_logs"
mkdir -p "$LOG_DIR"
SWEEP_LOG="$LOG_DIR/sweep_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SWEEP_LOG"; }

N_PREP=$(awk -F'\t' '$1=="PREP"' "$PLAN_PATH" | wc -l | tr -d ' ')
N_MLP=$(awk -F'\t' '$1=="TRAIN" && $2=="mlp"' "$PLAN_PATH" | wc -l | tr -d ' ')
N_TT=$(awk -F'\t' '$1=="TRAIN" && $2=="tt"' "$PLAN_PATH" | wc -l | tr -d ' ')

log "════════════════════════════════════════════════════════════════"
log "Cap × arch sweep: $SWEEP_NAME"
log "  ingestion_run:   $INGESTION_RUN_ABS"
log "  sweep_root:      $SWEEP_ROOT"
log "  prep cells:      $N_PREP"
log "  MLP cells:       $N_MLP (parallel up to $MAX_PARALLEL_MLP)"
log "  Two-tower cells: $N_TT (sequential)"
log "  epochs=$EPOCHS  batch_size=$BATCH_SIZE  patience=$PATIENCE"
log "════════════════════════════════════════════════════════════════"

# Save a copy of the config and the plan into the sweep root so the
# canonical record of what we ran lives next to the artifacts.
cp "$CONFIG_PATH_ABS" "$SWEEP_ROOT/config.yml"
cp "$PLAN_PATH" "$SWEEP_ROOT/plan.tsv"

# ── Memory sidecar ─────────────────────────────────────────────────────
# Launches a background process that snapshots `free -h` + top-RSS
# processes every 30 s into $LOG_DIR/mem_<ts>.log.  See
# 260428_like_biases/jobs/0006_sweep02_memory_prep.md for context.
# Killed on EXIT via the cleanup trap above.
if [[ -x "$SCRIPT_DIR/scripts/mem_sidecar.sh" ]]; then
  "$SCRIPT_DIR/scripts/mem_sidecar.sh" "$LOG_DIR" 30 &
  SIDECAR_PID=$!
  log "Memory sidecar PID: $SIDECAR_PID -> $LOG_DIR/mem_*.log"
else
  log "Memory sidecar script not found/executable; continuing without it."
fi

# ── Helper: extract a JSON arg into bash ──────────────────────────────
json_get() {
  # Usage: json_get '<json_string>' '<key>' [default]
  python3 - "$1" "$2" "${3:-}" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
key = sys.argv[2]
default = sys.argv[3]
val = data.get(key, default if default else None)
sys.stdout.write("" if val is None else str(val))
PY
}

# ── Phase A: per-cap data prep (Stage 2 + Stage 3) ─────────────────────
# We run prep sequentially per cap to keep memory usage bounded; each
# prep run is moderate (Stage 2 negative sampling + Stage 3 history join).
log ""
log "╔══════════════════════════════════════════════════════════════╗"
log "║  Phase A: per-cap data prep (target_posts + user_history)   ║"
log "╚══════════════════════════════════════════════════════════════╝"

FAILED_PREP=0
while IFS=$'\t' read -r KIND CAP_LABEL CAP_VALUE CELL_DIR _REST; do
  [[ "$KIND" != "PREP" ]] && continue
  mkdir -p "$CELL_DIR"

  # Symlink Stage 1 into the cap-specific run dir so cli.py finds it via
  # `select_prior_output(use_latest=True)` without re-ingesting.
  if [[ ! -e "$CELL_DIR/01_get_data" ]]; then
    ln -s "$INGESTION_RUN_ABS/01_get_data" "$CELL_DIR/01_get_data"
  fi

  PREP_LOG="$LOG_DIR/prep_${CAP_LABEL}.log"
  log "[$CAP_LABEL] Prep starting (cap=$CAP_VALUE)"

  CMD=(
    python3 cli.py
    --output-dir "$CELL_DIR"
    --start-from target_posts --stop-after user_history
  )
  if [[ "$CAP_VALUE" != "NONE" ]]; then
    CMD+=(--effective-likes-cap "$CAP_VALUE")
  fi

  # Pass through any extra CLI args from the YAML (e.g. --val-start, etc.).
  if [[ -n "$EXTRA_CLI_ARGS" && "$EXTRA_CLI_ARGS" != "[]" ]]; then
    while IFS= read -r EXTRA; do
      [[ -n "$EXTRA" ]] && CMD+=("$EXTRA")
    done < <(python3 -c "import json,sys; [print(a) for a in json.loads(sys.argv[1])]" "$EXTRA_CLI_ARGS")
  fi

  if "${CMD[@]}" > "$PREP_LOG" 2>&1; then
    log "[$CAP_LABEL] Prep ✓ (see $PREP_LOG)"
  else
    EC=$?
    log "[$CAP_LABEL] Prep ✗ exit=$EC (see $PREP_LOG)"
    (( FAILED_PREP++ )) || true
  fi
done < "$PLAN_PATH"

if (( FAILED_PREP > 0 )); then
  log "ABORT: $FAILED_PREP prep cell(s) failed; not proceeding to training."
  exit 1
fi

# ── Helper: launch one training cell ──────────────────────────────────
run_train_cell() {
  local CELL_DIR="$1"
  local RUN_TAG="$2"
  local JSON_ARGS="$3"
  local RUN_LOG="$4"
  local STATUS_FILE="$5"
  local CAP_LABEL="$6"

  local MODEL_TYPE
  local USER_ENCODER
  local USER_SUMM
  local EMA_ALPHA
  local SEED
  local EPOCHS_LOCAL
  local BATCH_SIZE_LOCAL
  local PATIENCE_LOCAL

  MODEL_TYPE="$(json_get "$JSON_ARGS" model_type)"
  USER_ENCODER="$(json_get "$JSON_ARGS" user_encoder)"
  USER_SUMM="$(json_get "$JSON_ARGS" user_summarization)"
  EMA_ALPHA="$(json_get "$JSON_ARGS" ema_alpha)"
  SEED="$(json_get "$JSON_ARGS" random_seed)"
  EPOCHS_LOCAL="$(json_get "$JSON_ARGS" epochs)"
  BATCH_SIZE_LOCAL="$(json_get "$JSON_ARGS" batch_size)"
  PATIENCE_LOCAL="$(json_get "$JSON_ARGS" patience)"

  local CMD=(
    python3 cli.py
    --output-dir "$CELL_DIR"
    --start-from train
    --model-type "$MODEL_TYPE"
    --run-tag "$RUN_TAG"
    --random-seed "$SEED"
    --epochs "$EPOCHS_LOCAL"
    --batch-size "$BATCH_SIZE_LOCAL"
    --patience "$PATIENCE_LOCAL"
  )

  # When skip_holdout_pred_during_train is active, stop at train so Stage 5 eval
  # does not run before Phase C.5 has produced the holdout parquets.
  # Stage 5 eval (Phase D) becomes a manual step to run after Phase C.5 completes.
  if [[ "${SKIP_HOLDOUT_PRED_DURING_TRAIN:-0}" == "1" ]]; then
    CMD+=(--stop-after train --skip-holdout-pred)
  else
    CMD+=(--stop-after evaluate)
  fi

  if [[ -n "$USER_ENCODER" ]]; then
    CMD+=(--user-encoder "$USER_ENCODER")
  fi
  if [[ "$USER_ENCODER" == "summarized" && -n "$USER_SUMM" ]]; then
    CMD+=(--user-summarization "$USER_SUMM")
    if [[ "$USER_SUMM" == "ema" && -n "$EMA_ALPHA" ]]; then
      CMD+=(--ema-alpha "$EMA_ALPHA")
    fi
  fi

  log "[$CAP_LABEL/$RUN_TAG] Launching: ${CMD[*]}"
  if "${CMD[@]}" > "$RUN_LOG" 2>&1; then
    echo "0" > "$STATUS_FILE"
    log "[$CAP_LABEL/$RUN_TAG] ✓ Completed"
  else
    local EC=$?
    echo "$EC" > "$STATUS_FILE"
    log "[$CAP_LABEL/$RUN_TAG] ✗ exit=$EC (see $RUN_LOG)"
  fi
}

# ── Phase B: MLP training cells (parallel) ────────────────────────────
log ""
log "╔══════════════════════════════════════════════════════════════╗"
log "║  Phase B: MLP training cells (parallel, max $MAX_PARALLEL_MLP)              ║"
log "╚══════════════════════════════════════════════════════════════╝"

PASSED=0
FAILED=0
MLP_PIDS=()
MLP_STATUS=()
MLP_LABELS=()
RUNNING=0

while IFS=$'\t' read -r KIND PHASE CAP_LABEL CELL_DIR RUN_TAG SEED JSON_ARGS; do
  [[ "$KIND" != "TRAIN" || "$PHASE" != "mlp" ]] && continue

  RUN_LOG="$LOG_DIR/run_${CAP_LABEL}_${RUN_TAG}.log"
  STATUS_FILE="$LOG_DIR/.status_${CAP_LABEL}_${RUN_TAG}"

  while (( RUNNING >= MAX_PARALLEL_MLP )); do
    wait -n 2>/dev/null || true
    RUNNING=0
    mapfile -t CURRENT < <(jobs -pr)
    for pid in "${MLP_PIDS[@]}"; do
      if printf '%s\n' "${CURRENT[@]}" | grep -qw -- "$pid"; then
        (( RUNNING++ )) || true
      fi
    done
  done

  run_train_cell "$CELL_DIR" "$RUN_TAG" "$JSON_ARGS" "$RUN_LOG" "$STATUS_FILE" "$CAP_LABEL" &
  MLP_PIDS+=($!)
  MLP_STATUS+=("$STATUS_FILE")
  MLP_LABELS+=("$CAP_LABEL/$RUN_TAG")
  (( RUNNING++ )) || true
done < "$PLAN_PATH"

log ""
log "Waiting for MLP cells to finish..."
for pid in "${MLP_PIDS[@]}"; do
  wait "$pid" 2>/dev/null || true
done

for i in "${!MLP_STATUS[@]}"; do
  if [[ -f "${MLP_STATUS[$i]}" ]] && [[ "$(cat "${MLP_STATUS[$i]}")" == "0" ]]; then
    (( PASSED++ )) || true
  else
    (( FAILED++ )) || true
  fi
  rm -f "${MLP_STATUS[$i]}"
done

log "MLP phase complete: $PASSED passed, $FAILED failed"

# ── Phase C: Two-tower training cells (sequential) ─────────────────────
log ""
log "╔══════════════════════════════════════════════════════════════╗"
log "║  Phase C: Two-tower training cells (sequential)             ║"
log "╚══════════════════════════════════════════════════════════════╝"

while IFS=$'\t' read -r KIND PHASE CAP_LABEL CELL_DIR RUN_TAG SEED JSON_ARGS; do
  [[ "$KIND" != "TRAIN" || "$PHASE" != "tt" ]] && continue

  RUN_LOG="$LOG_DIR/run_${CAP_LABEL}_${RUN_TAG}.log"
  STATUS_FILE="$LOG_DIR/.status_${CAP_LABEL}_${RUN_TAG}"

  log ""
  log "──────────────────────────────────────────────"
  log "[$CAP_LABEL/$RUN_TAG] Two-tower starting"
  log "──────────────────────────────────────────────"

  run_train_cell "$CELL_DIR" "$RUN_TAG" "$JSON_ARGS" "$RUN_LOG" "$STATUS_FILE" "$CAP_LABEL"
  if [[ -f "$STATUS_FILE" ]] && [[ "$(cat "$STATUS_FILE")" == "0" ]]; then
    (( PASSED++ )) || true
  else
    (( FAILED++ )) || true
  fi
  rm -f "$STATUS_FILE"
done < "$PLAN_PATH"

# ── Phase C.5: Holdout prediction (sequential) ────────────────────────
# Only runs when --skip-holdout-pred was passed during training (else the
# train stages already wrote predictions/holdout_*.parquet).  Sequential
# across all cells to keep RSS bounded; idempotent — skips cells that
# already have holdout parquets.
if [[ "${SKIP_HOLDOUT_PRED_DURING_TRAIN:-0}" == "1" ]]; then
  log ""
  log "╔══════════════════════════════════════════════════════════════╗"
  log "║  Phase C.5: Holdout prediction (sequential)                 ║"
  log "╚══════════════════════════════════════════════════════════════╝"

  HOLDOUT_OK=0
  HOLDOUT_FAIL=0
  HOLDOUT_SKIPPED=0
  while IFS=$'\t' read -r KIND PHASE CAP_LABEL CELL_DIR RUN_TAG SEED JSON_ARGS; do
    [[ "$KIND" != "TRAIN" ]] && continue

    # Resolve the actual training cell directory (timestamp-prefixed).
    TRAIN_CELL_DIR="$(ls -dt "$CELL_DIR/04_train"/*_"${RUN_TAG}" 2>/dev/null | head -1 || true)"
    if [[ -z "$TRAIN_CELL_DIR" || ! -d "$TRAIN_CELL_DIR" ]]; then
      log "[$CAP_LABEL/$RUN_TAG] holdout-pred SKIP — no train cell dir found"
      (( HOLDOUT_SKIPPED++ )) || true
      continue
    fi

    PRED_PARQ="$TRAIN_CELL_DIR/predictions/holdout_unseen_users.parquet"
    if [[ -f "$PRED_PARQ" ]]; then
      log "[$CAP_LABEL/$RUN_TAG] holdout-pred already present, skipping"
      (( HOLDOUT_SKIPPED++ )) || true
      continue
    fi

    HOLDOUT_LOG="$LOG_DIR/holdout_${CAP_LABEL}_${RUN_TAG}.log"
    log "[$CAP_LABEL/$RUN_TAG] Running holdout-pred -> $HOLDOUT_LOG"
    if python3 scripts/run_holdout_pred.py "$TRAIN_CELL_DIR" > "$HOLDOUT_LOG" 2>&1; then
      log "[$CAP_LABEL/$RUN_TAG] holdout-pred ✓"
      (( HOLDOUT_OK++ )) || true
    else
      log "[$CAP_LABEL/$RUN_TAG] holdout-pred ✗ (see $HOLDOUT_LOG)"
      (( HOLDOUT_FAIL++ )) || true
    fi
  done < "$PLAN_PATH"

  log "Phase C.5 complete: ok=$HOLDOUT_OK skipped=$HOLDOUT_SKIPPED failed=$HOLDOUT_FAIL"
  if (( HOLDOUT_FAIL > 0 )); then
    (( FAILED += HOLDOUT_FAIL )) || true
  fi
fi

# ── Summary ────────────────────────────────────────────────────────────
TOTAL=$(( N_MLP + N_TT ))
log ""
log "════════════════════════════════════════════════════════════════"
log "Sweep complete: $PASSED passed, $FAILED failed out of $TOTAL"
log "Sweep root: $SWEEP_ROOT"
log "Logs:       $LOG_DIR"
log "════════════════════════════════════════════════════════════════"

exit "$FAILED"
