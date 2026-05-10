import numpy as np

global BKZ_MAX_LOOPS
global BKZ_SIEVING_CROSSOVER
global N_SIEVE_THREADS
global DTYPE
global N_MAX_SLICER_ITERATIONS
global NRAND_FACTOR
global EPS2
global NPROJ_TESTS, HYB_PROJ_THRESHOLD
global SATURATION_SCALAR

BKZ_MAX_LOOPS = 2
BKZ_SIEVING_CROSSOVER = 55
N_SIEVE_THREADS = 5
N_MAX_SLICER_ITERATIONS = 400
NRAND_FACTOR = 10.0
DTYPE = np.float64
EPS2 = 1.0001
NPROJ_TESTS = 2048
HYB_PROJ_THRESHOLD = 0.93
SATURATION_SCALAR = 1.0

# ===== Cross-instance batch scheduler =====
global HYB_BATCH_SCHED_ENABLE
global HYB_BATCH_MAX_TARGETS
global HYB_BATCH_VERBOSE

HYB_BATCH_SCHED_ENABLE = True
# 当前 3050Ti 4GB 下，16384 已经在多轮 repeat 中比 8192 更适合作为默认值。
HYB_BATCH_MAX_TARGETS = 16384
HYB_BATCH_VERBOSE = False
# =========================================

# ===== Projected batch backend =====
global HYB_GPU_BACKEND_ENABLE
global HYB_GPU_BACKEND_VERBOSE
global HYB_GPU_PROJ_ELL_MODE
global HYB_GPU_PROJ_TAU_SCALE
global HYB_GPU_ALLOW_FALLBACK
global HYB_GPU_SYNC_TIMING

HYB_GPU_BACKEND_ENABLE = True
HYB_GPU_BACKEND_VERBOSE = False
HYB_GPU_PROJ_ELL_MODE = "chosen_delta"
HYB_GPU_PROJ_TAU_SCALE = 1.0
HYB_GPU_ALLOW_FALLBACK = True

# True：分项计时更准确，适合 profiling；False：只看 walltime/throughput 时可尝试。
HYB_GPU_SYNC_TIMING = True
# =======================================

# ===== Blocked exact Babai (projected NP) =====
global HYB_GPU_NP_IMPL
global HYB_GPU_NP_BLOCK_ROWS

# "loop"                -> 原始 CuPy 逐层循环版
# "blocked_exact"       -> 块化精确 Babai 版
# "blocked_exact_fused" -> RawKernel fused panel 版
HYB_GPU_NP_IMPL = "blocked_exact_fused"
HYB_GPU_NP_BLOCK_ROWS = 32
# ==============================================

# ===== GPU T staging / layout optimization =====
global HYB_GPU_DIRECT_STAGE_T
global HYB_GPU_USE_ROWMAJOR_T
global HYB_GPU_T_LAYOUT_DEBUG
global HYB_GPU_T_LAYOUT_DEBUG_MAX_PRINT

# True: v2 column-major provider 返回的列切片直接 stage 到 pinned host buffer。
HYB_GPU_DIRECT_STAGE_T = True

# True: 若 v3 dump 中存在 all_targets_matrix_rowmajor，则优先使用 row-major target path。
HYB_GPU_USE_ROWMAJOR_T = True

HYB_GPU_T_LAYOUT_DEBUG = False
HYB_GPU_T_LAYOUT_DEBUG_MAX_PRINT = 8
# =====================================

# ===== Guess / candidate generation (legacy path) =====
global HYB_CAND_MIN_PER_BRANCH
global HYB_CAND_MAX_PER_BRANCH
global HYB_CAND_USE_SQRT_KEYNUM
global HYB_CAND_DEDUP

HYB_CAND_MIN_PER_BRANCH = 64
HYB_CAND_MAX_PER_BRANCH = 64
HYB_CAND_USE_SQRT_KEYNUM = False
HYB_CAND_DEDUP = True
# ======================================================

# ===== Deterministic candidate enumerator =====
global HYB_ENUMERATOR_ENABLE
global HYB_ENUMERATOR_CHUNK_SIZE
global HYB_WRONG_TARGET_COUNT
global HYB_CORRECT_TARGET_COUNT

HYB_ENUMERATOR_ENABLE = True
HYB_ENUMERATOR_CHUNK_SIZE = 4096
HYB_WRONG_TARGET_COUNT = 64
HYB_CORRECT_TARGET_COUNT = 64
# =============================================
