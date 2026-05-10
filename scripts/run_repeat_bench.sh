#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GPU_REPO="$SUITE_ROOT/gpu_repo"
CONFIG_FILE="${1:-$SUITE_ROOT/configs/default.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

RUN_DIR="$SUITE_ROOT/runs"
DUMP_DIR="$RUN_DIR/dumps/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs/$RUN_NAME"
SUMMARY_DIR="$RUN_DIR/summaries/$RUN_NAME"
REPORT_DIR="$RUN_DIR/reports/$RUN_NAME"
mkdir -p "$DUMP_DIR" "$LOG_DIR" "$SUMMARY_DIR" "$REPORT_DIR"

DUMP_PATH="$DUMP_DIR/${RUN_NAME}.pkl"
REPEAT_LOG="$LOG_DIR/${RUN_NAME}_repeat.log"
REPEAT_JSON="$SUMMARY_DIR/${RUN_NAME}_repeat.json"
DUMP_JSON="$SUMMARY_DIR/${RUN_NAME}_dump.json"
PREPROCESS_LOG="$LOG_DIR/${RUN_NAME}_preprocess.log"
DUMP_LOG="$LOG_DIR/${RUN_NAME}_dump.log"

# 如果 dump 不存在，先生成
if [[ ! -f "$DUMP_PATH" ]]; then
  "${PYTHON_BIN}" "$GPU_REPO/pipelines/run_guess6_full_pipeline.py" \
    --python "$PYTHON_BIN" \
    --n "$N" \
    --q "$Q" \
    --dist "$DIST" \
    --dist_param "$DIST_PARAM" \
    --lats_per_dim "$LATS_PER_DIM" \
    --inst_per_lat "$INST_PER_LAT" \
    --n_guess_coord "$N_GUESS_COORD" \
    --beta_pre "$BETA_PRE" \
    --n_slicer_coord "$N_SLICER_COORD" \
    --delta_slicer_coord "$DELTA_SLICER_COORD" \
    --params "$PARAMS" \
    --dump_path "$DUMP_PATH" \
    --preprocess_log "$PREPROCESS_LOG" \
    --dump_log "$DUMP_LOG" \
    --dump_json "$DUMP_JSON" \
    --skip_gpu --skip_cpu
fi

"${PYTHON_BIN}" "$GPU_REPO/benchmarks/bench_replay_repeat.py" \
  --python "$PYTHON_BIN" \
  --run-prog "$GPU_REPO/run_prog_hyb.py" \
  --dump "$DUMP_PATH" \
  --gpu-runs "$GPU_RUNS" \
  --cpu-runs "$CPU_RUNS" \
  --gpu-warmup "$GPU_WARMUP" \
  --validate-batches "$VALIDATE_BATCHES" \
  --json-out "$REPEAT_JSON" | tee "$REPEAT_LOG"

echo "[done] run_repeat_bench completed"
echo "[repeat json] $REPEAT_JSON"
