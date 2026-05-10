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
GPU_LOG="$LOG_DIR/${RUN_NAME}_gpu.log"
CPU_LOG="$LOG_DIR/${RUN_NAME}_cpu.log"
PREPROCESS_LOG="$LOG_DIR/${RUN_NAME}_preprocess.log"
DUMP_LOG="$LOG_DIR/${RUN_NAME}_dump.log"
COMPARE_LOG="$REPORT_DIR/${RUN_NAME}_compare.txt"
SUMMARY_JSON="$SUMMARY_DIR/${RUN_NAME}_summary.json"
DUMP_JSON="$SUMMARY_DIR/${RUN_NAME}_dump.json"
GPU_JSON="$SUMMARY_DIR/${RUN_NAME}_gpu.json"
CPU_JSON="$SUMMARY_DIR/${RUN_NAME}_cpu.json"

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
  --gpu_log "$GPU_LOG" \
  --cpu_log "$CPU_LOG" \
  --preprocess_log "$PREPROCESS_LOG" \
  --dump_log "$DUMP_LOG" \
  --dump_json "$DUMP_JSON" \
  --gpu_json "$GPU_JSON" \
  --cpu_json "$CPU_JSON" \
  --validate_batches "$VALIDATE_BATCHES" \
  --validate_atol "$VALIDATE_ATOL" \
  --validate_rtol "$VALIDATE_RTOL"

# 若流水线脚本没有产出 JSON，可由 collect 工具兜底统一汇总
"${PYTHON_BIN}" "$GPU_REPO/tools/compare_gpu_cpu_scores.py" \
  --cpu_log "$CPU_LOG" \
  --gpu_log "$GPU_LOG" | tee "$COMPARE_LOG"

"${PYTHON_BIN}" "$SUITE_ROOT/tools/collect_replay_summary.py" \
  --gpu-log "$GPU_LOG" \
  --cpu-log "$CPU_LOG" \
  --compare-log "$COMPARE_LOG" \
  --output "$SUMMARY_JSON"

echo "[done] run_once completed"
echo "[summary] $SUMMARY_JSON"
