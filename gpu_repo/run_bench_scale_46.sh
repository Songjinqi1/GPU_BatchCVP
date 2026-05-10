#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${HOME}/research/g6k_hybrid-ac_artifact"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

DUMP_DIR="results/dumps/bench_scale_46"
LOG_DIR="results/logs/bench_scale_46"
mkdir -p "${DUMP_DIR}" "${LOG_DIR}"

GLOBAL_CONSTS="global_consts.py"
BACKUP_CONSTS="${GLOBAL_CONSTS}.benchbak"

# ===== fixed benchmark params =====
N=140
Q=3329
DIST="binomial"
DIST_PARAM="3"
N_GUESS=1
BETA_PRE=46
N_SLICER=47
DELTA_SLICER=0

# lats_per_dim -> jobs = 200 * lats_per_dim
SCALES=(1 2 4 8 16)

# ===== helpers =====
restore_consts() {
    if [[ -f "${BACKUP_CONSTS}" ]]; then
        mv "${BACKUP_CONSTS}" "${GLOBAL_CONSTS}"
    fi
}

set_backend_flag() {
    local mode="$1"   # True / False
    if ! grep -q '^HYB_GPU_BACKEND_ENABLE = ' "${GLOBAL_CONSTS}"; then
        echo "[ERROR] Cannot find HYB_GPU_BACKEND_ENABLE in ${GLOBAL_CONSTS}"
        exit 1
    fi
    sed -i "s/^HYB_GPU_BACKEND_ENABLE = .*/HYB_GPU_BACKEND_ENABLE = ${mode}/" "${GLOBAL_CONSTS}"
}

run_one_scale() {
    local L="$1"
    local JOBS=$((200 * L))

    local DUMP_PATH="${DUMP_DIR}/batch_${N}_${DIST}_${N_SLICER}_L${L}.pkl"
    local CPU_LOG="${LOG_DIR}/batch_${N}_${DIST}_${N_SLICER}_L${L}_cpu.txt"
    local GPU_LOG="${LOG_DIR}/batch_${N}_${DIST}_${N_SLICER}_L${L}_gpu.txt"
    local CMP_LOG="${LOG_DIR}/batch_${N}_${DIST}_${N_SLICER}_L${L}_compare.txt"

    echo "============================================================"
    echo "[BENCH] lats_per_dim=${L}  -> approx jobs=${JOBS}"
    echo "============================================================"

    echo "[1/4] dump candidates -> ${DUMP_PATH}"
    "${PYTHON_BIN}" run_prog_hyb.py \
      --n "${N}" \
      --q "${Q}" \
      --dist "${DIST}" \
      --dist_param "${DIST_PARAM}" \
      --n_guess_coord "${N_GUESS}" \
      --beta_pre "${BETA_PRE}" \
      --n_slicer_coord "${N_SLICER}" \
      --delta_slicer_coord "${DELTA_SLICER}" \
      --lats_per_dim "${L}" \
      --dump_batch_candidates \
      --batch_dump_path "${DUMP_PATH}"

    echo "[2/4] CPU replay -> ${CPU_LOG}"
    set_backend_flag False
    "${PYTHON_BIN}" run_prog_hyb.py \
      --replay_batch_candidates "${DUMP_PATH}" \
      | tee "${CPU_LOG}"

    echo "[3/4] GPU replay -> ${GPU_LOG}"
    set_backend_flag True
    "${PYTHON_BIN}" run_prog_hyb.py \
      --replay_batch_candidates "${DUMP_PATH}" \
      | tee "${GPU_LOG}"

    echo "[4/4] compare -> ${CMP_LOG}"
    "${PYTHON_BIN}" compare_gpu_cpu_scores.py \
      --cpu_log "${CPU_LOG}" \
      --gpu_log "${GPU_LOG}" \
      | tee "${CMP_LOG}"

    echo "[DONE] scale L=${L}, approx jobs=${JOBS}"
    echo
}

# ===== main =====
cp "${GLOBAL_CONSTS}" "${BACKUP_CONSTS}"
trap restore_consts EXIT

echo "[INFO] Project dir: ${PROJECT_DIR}"
echo "[INFO] Python: $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "[INFO] Dump dir: ${DUMP_DIR}"
echo "[INFO] Log dir : ${LOG_DIR}"
echo

for L in "${SCALES[@]}"; do
    run_one_scale "${L}"
done

echo "============================================================"
echo "[ALL DONE] Finished scales: ${SCALES[*]}"
echo "Results are under:"
echo "  ${DUMP_DIR}"
echo "  ${LOG_DIR}"
echo "============================================================"