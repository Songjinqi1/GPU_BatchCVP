import os
import time
from collections import Counter

import numpy as np

try:
    import cupy as cp
    CUPY_IMPORT_OK = True
    CUPY_IMPORT_ERROR = None
except Exception as e:
    cp = None
    CUPY_IMPORT_OK = False
    CUPY_IMPORT_ERROR = repr(e)

# 可选读取 global_consts；读不到就走默认值
try:
    from global_consts import HYB_GPU_NP_IMPL
except Exception:
    HYB_GPU_NP_IMPL = "loop"

try:
    from global_consts import HYB_GPU_NP_BLOCK_ROWS
except Exception:
    HYB_GPU_NP_BLOCK_ROWS = 8


def _env_bool(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "on", "y")


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


try:
    from global_consts import HYB_GPU_T_LAYOUT_DEBUG
except Exception:
    HYB_GPU_T_LAYOUT_DEBUG = _env_bool("HYB_GPU_T_LAYOUT_DEBUG", False)

# Environment variables override global_consts for quick experiments.
HYB_GPU_NP_IMPL = os.environ.get("HYB_GPU_NP_IMPL", HYB_GPU_NP_IMPL)
HYB_GPU_NP_BLOCK_ROWS = _env_int("HYB_GPU_NP_BLOCK_ROWS", HYB_GPU_NP_BLOCK_ROWS)

HYB_GPU_T_LAYOUT_DEBUG_MAX_PRINT = _env_int("HYB_GPU_T_LAYOUT_DEBUG_MAX_PRINT", 8)

# If enabled, GPU path accepts non-contiguous provider views and copies them
# directly into the pinned staging buffer. This avoids one intermediate
# np.ascontiguousarray copy for column-sliced all_targets_matrix blocks.
try:
    from global_consts import HYB_GPU_DIRECT_STAGE_T
except Exception:
    HYB_GPU_DIRECT_STAGE_T = _env_bool("HYB_GPU_DIRECT_STAGE_T", True)
HYB_GPU_DIRECT_STAGE_T = _env_bool("HYB_GPU_DIRECT_STAGE_T", HYB_GPU_DIRECT_STAGE_T)

# Row-major target layout path for v3 dumps. If enabled and a row-major
# provider is available, GPU staging uses T_rows with shape (m, d), which
# is C-contiguous for contiguous job ranges. The projected matrix is still
# materialized as Y_gpu with shape (ell, m), so the nearest-plane code remains
# unchanged.
try:
    from global_consts import HYB_GPU_USE_ROWMAJOR_T
except Exception:
    HYB_GPU_USE_ROWMAJOR_T = _env_bool("HYB_GPU_USE_ROWMAJOR_T", False)
HYB_GPU_USE_ROWMAJOR_T = _env_bool("HYB_GPU_USE_ROWMAJOR_T", HYB_GPU_USE_ROWMAJOR_T)

_T_LAYOUT_DEBUG_COUNTER = 0


def _short_exc(e, maxlen=300):
    s = repr(e)
    if len(s) > maxlen:
        s = s[:maxlen] + "..."
    return s


def gpu_runtime_info():
    info = {
        "cupy_import_ok": CUPY_IMPORT_OK,
        "cupy_import_error": CUPY_IMPORT_ERROR,
        "gpu_device_count": 0,
        "gpu_available": False,
        "device_names": [],
    }

    if not CUPY_IMPORT_OK:
        return info

    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
        info["gpu_device_count"] = ndev
        info["gpu_available"] = ndev > 0
        names = []
        for i in range(ndev):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            names.append(str(name))
        info["device_names"] = names
    except Exception as e:
        info["gpu_available"] = False
        info["gpu_runtime_error"] = _short_exc(e)

    return info


# =========================================================
# Target block provider: serve batch-ready target matrices
# =========================================================

class TargetBlockProvider:
    """
    Provide T blocks directly from a global contiguous all_targets_matrix.

    all_targets_matrix: shape (d, total_jobs), float64 contiguous
    all_job_ids      : list[int], same column order
    """
    def __init__(self, all_targets_matrix=None, all_job_ids=None):
        if all_targets_matrix is None or all_job_ids is None:
            self.all_targets_matrix = None
            self.all_job_ids = None
            self.job_id_to_col = None
            return

        T = np.ascontiguousarray(np.asarray(all_targets_matrix, dtype=np.float64))
        if T.ndim != 2:
            raise ValueError(f"all_targets_matrix must be 2D, got shape={T.shape}")

        self.all_targets_matrix = T
        self.all_job_ids = [int(x) for x in all_job_ids]
        if T.shape[1] != len(self.all_job_ids):
            raise ValueError(
                f"all_targets_matrix second dim {T.shape[1]} != len(all_job_ids) {len(self.all_job_ids)}"
            )

        self.job_id_to_col = {jid: idx for idx, jid in enumerate(self.all_job_ids)}

    def enabled(self):
        return self.all_targets_matrix is not None and self.job_id_to_col is not None

    def get_block(self, batch_jobs):
        """
        Return T_block with shape (d, m), float64 contiguous.
        If batch jobs correspond to contiguous columns, return a view.
        Otherwise return a contiguous gathered copy.
        """
        if not self.enabled():
            return None

        if len(batch_jobs) == 0:
            return np.zeros((0, 0), dtype=np.float64)

        cols = [self.job_id_to_col[int(job.job_id)] for job in batch_jobs]
        start = cols[0]
        contiguous = True
        for k, c in enumerate(cols):
            if c != start + k:
                contiguous = False
                break

        if contiguous:
            return self.all_targets_matrix[:, start:start + len(cols)]

        return np.ascontiguousarray(self.all_targets_matrix[:, cols], dtype=np.float64)


class RowMajorTargetBlockProvider:
    """
    Provide target blocks from a row-major global target matrix.

    all_targets_matrix_rowmajor: shape (total_jobs, d), float64 C-contiguous
    all_job_ids                : list[int], same row order

    For contiguous batch jobs, get_rows_block() returns a row slice view with
    shape (m, d). This is the layout we want for lower host staging overhead.
    """
    def __init__(self, all_targets_matrix_rowmajor=None, all_job_ids=None):
        if all_targets_matrix_rowmajor is None or all_job_ids is None:
            self.all_targets_matrix_rowmajor = None
            self.all_job_ids = None
            self.job_id_to_row = None
            return

        T = np.ascontiguousarray(np.asarray(all_targets_matrix_rowmajor, dtype=np.float64))
        if T.ndim != 2:
            raise ValueError(
                f"all_targets_matrix_rowmajor must be 2D, got shape={T.shape}"
            )

        self.all_targets_matrix_rowmajor = T
        self.all_job_ids = [int(x) for x in all_job_ids]
        if T.shape[0] != len(self.all_job_ids):
            raise ValueError(
                f"rowmajor first dim {T.shape[0]} != len(all_job_ids) {len(self.all_job_ids)}"
            )

        self.job_id_to_row = {jid: idx for idx, jid in enumerate(self.all_job_ids)}

    def enabled(self):
        return (
            self.all_targets_matrix_rowmajor is not None
            and self.job_id_to_row is not None
        )

    def get_rows_block(self, batch_jobs):
        """
        Return T_rows with shape (m, d), float64.
        If batch jobs correspond to contiguous rows, return a C-contiguous view.
        Otherwise return a contiguous gathered copy.
        """
        if not self.enabled():
            return None

        if len(batch_jobs) == 0:
            return np.zeros((0, 0), dtype=np.float64)

        rows = [self.job_id_to_row[int(job.job_id)] for job in batch_jobs]
        start = rows[0]
        contiguous = True
        for k, r in enumerate(rows):
            if r != start + k:
                contiguous = False
                break

        if contiguous:
            return self.all_targets_matrix_rowmajor[start:start + len(rows), :]

        return np.ascontiguousarray(self.all_targets_matrix_rowmajor[rows, :], dtype=np.float64)


# =========================================================
# Reusable GPU input workspace for lower H2D overhead
# =========================================================

_GPU_INPUT_WORKSPACE_CACHE = {}


class _GpuInputWorkspace:
    """
    Reusable input buffers:
      - pinned host staging buffer
      - reusable device input buffer T_gpu
    Shape convention:
      T_host / T_gpu: (d, max_cols)
    """
    def __init__(self, d: int, max_cols: int):
        self.d = int(d)
        self.max_cols = int(max_cols)

        nbytes = self.d * self.max_cols * np.dtype(np.float64).itemsize

        # pinned host memory
        self._pinned_mem = cp.cuda.alloc_pinned_memory(nbytes)
        numel = self.d * self.max_cols
        arr = np.frombuffer(self._pinned_mem, dtype=np.float64, count=numel)
        if arr.size != numel:
            raise RuntimeError(
                f"pinned buffer mismatch: got {arr.size}, need {numel}"
            )
        self.T_host = arr.reshape(self.d, self.max_cols)

        # reusable device buffer
        self.T_gpu = cp.empty((self.d, self.max_cols), dtype=cp.float64)


def _gpu_input_workspace_key(d: int, max_cols: int):
    return (int(d), int(max_cols))


def get_gpu_input_workspace(d: int, max_cols: int) -> _GpuInputWorkspace:
    key = _gpu_input_workspace_key(d, max_cols)
    ws = _GPU_INPUT_WORKSPACE_CACHE.get(key)
    if ws is None:
        ws = _GpuInputWorkspace(d=d, max_cols=max_cols)
        _GPU_INPUT_WORKSPACE_CACHE[key] = ws
    return ws


def stage_T_block_to_gpu(T: np.ndarray, sync_timing: bool = True):
    """
    Stage one T block to reusable GPU input buffer.

    Important optimization:
      We intentionally do NOT force np.ascontiguousarray(T) here.
      For v2 replay, TargetBlockProvider often returns a non-contiguous
      column slice of the global (d, total_jobs) matrix. The old path first
      materialized a contiguous temporary T and then copied T again into
      pinned staging memory. This function copies the possibly non-contiguous
      view directly into the pinned staging buffer, removing one host-side copy.

    Input:
      T: shape (d, m), float64 array or view

    Returns:
      T_gpu_view: shape (d, m) device view
      stage_info: dict with host_copy_sec / h2d_copy_sec / input layout flags
    """
    T_arr = np.asarray(T, dtype=np.float64)
    if T_arr.ndim != 2:
        raise ValueError(f"T must be 2D, got shape={T_arr.shape}")

    d, m = T_arr.shape
    ws = get_gpu_input_workspace(d=d, max_cols=m)

    info = {
        "host_copy_sec": 0.0,
        "h2d_copy_sec": 0.0,
        "stage_input_c_contiguous": bool(T_arr.flags["C_CONTIGUOUS"]),
        "stage_input_f_contiguous": bool(T_arr.flags["F_CONTIGUOUS"]),
        "stage_input_strides": tuple(int(x) for x in T_arr.strides),
    }

    # host view -> pinned host staging
    # This is the only required host copy before H2D. If T_arr is a
    # non-contiguous provider view, NumPy copies directly into pinned memory.
    t0 = time.perf_counter()
    ws.T_host[:, :m] = T_arr
    info["host_copy_sec"] = time.perf_counter() - t0

    # pinned host -> device
    _sync_if_needed(sync_timing)
    t0 = time.perf_counter()
    ws.T_gpu[:, :m].set(ws.T_host[:, :m])
    _sync_if_needed(sync_timing)
    info["h2d_copy_sec"] = time.perf_counter() - t0

    return ws.T_gpu[:, :m], info



# =========================================================
# Reusable row-major GPU input workspace for v3 target layout
# =========================================================

_GPU_ROWS_INPUT_WORKSPACE_CACHE = {}


class _GpuRowsInputWorkspace:
    """
    Reusable row-major input buffers:
      - pinned host staging buffer
      - reusable device input buffer T_rows_gpu

    Shape convention:
      T_rows_host / T_rows_gpu: (max_rows, d)
    """
    def __init__(self, max_rows: int, d: int):
        self.max_rows = int(max_rows)
        self.d = int(d)

        nbytes = self.max_rows * self.d * np.dtype(np.float64).itemsize

        self._pinned_mem = cp.cuda.alloc_pinned_memory(nbytes)
        numel = self.max_rows * self.d
        arr = np.frombuffer(self._pinned_mem, dtype=np.float64, count=numel)
        if arr.size != numel:
            raise RuntimeError(
                f"row-major pinned buffer mismatch: got {arr.size}, need {numel}"
            )
        self.T_rows_host = arr.reshape(self.max_rows, self.d)
        self.T_rows_gpu = cp.empty((self.max_rows, self.d), dtype=cp.float64)


def _gpu_rows_input_workspace_key(max_rows: int, d: int):
    return (int(max_rows), int(d))


def get_gpu_rows_input_workspace(max_rows: int, d: int) -> _GpuRowsInputWorkspace:
    key = _gpu_rows_input_workspace_key(max_rows, d)
    ws = _GPU_ROWS_INPUT_WORKSPACE_CACHE.get(key)
    if ws is None:
        ws = _GpuRowsInputWorkspace(max_rows=max_rows, d=d)
        _GPU_ROWS_INPUT_WORKSPACE_CACHE[key] = ws
    return ws


def stage_T_rows_to_gpu(T_rows: np.ndarray, sync_timing: bool = True):
    """
    Stage one row-major T block to reusable GPU input buffer.

    Input:
      T_rows: shape (m, d), preferably C-contiguous float64

    Returns:
      T_rows_gpu_view: shape (m, d) device view
      stage_info: dict with host_copy_sec / h2d_copy_sec / input layout flags
    """
    T_arr = np.asarray(T_rows, dtype=np.float64)
    if T_arr.ndim != 2:
        raise ValueError(f"T_rows must be 2D, got shape={T_arr.shape}")

    m, d = T_arr.shape
    ws = get_gpu_rows_input_workspace(max_rows=m, d=d)

    info = {
        "host_copy_sec": 0.0,
        "h2d_copy_sec": 0.0,
        "stage_input_c_contiguous": bool(T_arr.flags["C_CONTIGUOUS"]),
        "stage_input_f_contiguous": bool(T_arr.flags["F_CONTIGUOUS"]),
        "stage_input_strides": tuple(int(x) for x in T_arr.strides),
    }

    t0 = time.perf_counter()
    ws.T_rows_host[:m, :d] = T_arr
    info["host_copy_sec"] = time.perf_counter() - t0

    _sync_if_needed(sync_timing)
    t0 = time.perf_counter()
    ws.T_rows_gpu[:m, :d].set(ws.T_rows_host[:m, :d])
    _sync_if_needed(sync_timing)
    info["h2d_copy_sec"] = time.perf_counter() - t0

    return ws.T_rows_gpu[:m, :d], info
def batched_targets_to_matrix(batch_jobs):
    """
    Legacy fallback path.
    Stack targets into T with shape (d, m).
    """
    if len(batch_jobs) == 0:
        return np.zeros((0, 0), dtype=np.float64)

    mats = []
    base_dim = None
    for job in batch_jobs:
        v = np.asarray(job.target, dtype=np.float64).reshape(-1)
        if base_dim is None:
            base_dim = len(v)
        elif len(v) != base_dim:
            raise ValueError(
                f"inconsistent target dimension in batch: got {len(v)} vs expected {base_dim}"
            )
        mats.append(v)

    T = np.stack(mats, axis=1)
    return np.ascontiguousarray(T, dtype=np.float64)


def _maybe_print_t_layout(tag: str, T: np.ndarray, note: str = ""):
    """Print a small number of target-layout debug lines when enabled."""
    global _T_LAYOUT_DEBUG_COUNTER
    if not HYB_GPU_T_LAYOUT_DEBUG:
        return
    if _T_LAYOUT_DEBUG_COUNTER >= HYB_GPU_T_LAYOUT_DEBUG_MAX_PRINT:
        return
    _T_LAYOUT_DEBUG_COUNTER += 1
    try:
        print(
            f"[T layout] {tag} shape={tuple(T.shape)} dtype={T.dtype} "
            f"C={bool(T.flags['C_CONTIGUOUS'])} "
            f"F={bool(T.flags['F_CONTIGUOUS'])} "
            f"strides={T.strides} {note}"
        )
    except Exception as e:
        print(f"[T layout] failed to print layout: {_short_exc(e)}")


def get_T_matrix_for_jobs(batch_jobs, target_block_provider=None, ensure_contiguous=True):
    """
    Preferred path:
      1) Use target_block_provider if available
      2) Fallback to legacy batched_targets_to_matrix

    When ensure_contiguous=True, return the old C-contiguous (d, m) matrix.
    When ensure_contiguous=False, return the provider view directly when
    possible. The GPU path can then copy that view directly into pinned memory,
    avoiding an intermediate host copy.
    """
    if target_block_provider is not None and target_block_provider.enabled():
        T = target_block_provider.get_block(batch_jobs)
        if T is not None:
            T_arr = np.asarray(T, dtype=np.float64)
            _maybe_print_t_layout("provider_raw", T_arr)

            if not ensure_contiguous:
                _maybe_print_t_layout("provider_direct_stage_view", T_arr, note="direct_to_pinned")
                return T_arr

            if T_arr.flags["C_CONTIGUOUS"]:
                return T_arr
            T_contig = np.ascontiguousarray(T_arr, dtype=np.float64)
            _maybe_print_t_layout("provider_contiguous_copy", T_contig, note="from provider_raw")
            return T_contig

    T = batched_targets_to_matrix(batch_jobs)
    _maybe_print_t_layout("legacy_stacked", T)
    return T



def get_T_rows_for_jobs(batch_jobs, rowmajor_target_block_provider=None):
    """Return row-major targets T_rows with shape (m, d) if provider is available."""
    if rowmajor_target_block_provider is None or not rowmajor_target_block_provider.enabled():
        return None
    T_rows = rowmajor_target_block_provider.get_rows_block(batch_jobs)
    if T_rows is None:
        return None
    T_rows_arr = np.asarray(T_rows, dtype=np.float64)
    _maybe_print_t_layout("rowmajor_provider_raw", T_rows_arr)
    return T_rows_arr
def batched_nearest_plane_cpu(R22, Y):
    ell, m = Y.shape
    U = np.zeros((ell, m), dtype=np.float64)
    E = np.zeros((ell, m), dtype=np.float64)

    for i in range(ell - 1, -1, -1):
        rhs = Y[i].copy()
        if i < ell - 1:
            rhs = rhs - R22[i, i + 1:] @ U[i + 1:, :]
        ui = np.rint(rhs / R22[i, i])
        U[i, :] = ui
        E[i, :] = rhs - R22[i, i] * ui
    return U, E


def batched_nearest_plane_gpu(R22_gpu, Y_gpu):
    ell, m = Y_gpu.shape
    U_gpu = cp.zeros((ell, m), dtype=cp.float64)
    E_gpu = cp.zeros((ell, m), dtype=cp.float64)

    for i in range(ell - 1, -1, -1):
        rhs = Y_gpu[i]
        if i < ell - 1:
            rhs = rhs - R22_gpu[i, i + 1:] @ U_gpu[i + 1:, :]
        else:
            rhs = rhs.copy()
        ui = cp.rint(rhs / R22_gpu[i, i])
        U_gpu[i, :] = ui
        E_gpu[i, :] = rhs - R22_gpu[i, i] * ui
    return U_gpu, E_gpu


def build_block_partitions(ell: int, block_rows: int):
    if block_rows <= 0:
        raise ValueError(f"block_rows must be positive, got {block_rows}")
    parts = []
    r1 = ell
    while r1 > 0:
        r0 = max(0, r1 - block_rows)
        parts.append((r0, r1))
        r1 = r0
    return parts


def blocked_babai_panel_gpu(Rjj_gpu, Yj_gpu):
    b, m = Yj_gpu.shape
    Uj_gpu = cp.zeros((b, m), dtype=cp.float64)
    Ej_gpu = cp.zeros((b, m), dtype=cp.float64)

    for local_i in range(b - 1, -1, -1):
        rhs = Yj_gpu[local_i]
        if local_i < b - 1:
            rhs = rhs - Rjj_gpu[local_i, local_i + 1:] @ Uj_gpu[local_i + 1:, :]
        else:
            rhs = rhs.copy()
        uj = cp.rint(rhs / Rjj_gpu[local_i, local_i])
        Uj_gpu[local_i, :] = uj
        Ej_gpu[local_i, :] = rhs - Rjj_gpu[local_i, local_i] * uj

    return Uj_gpu, Ej_gpu


def update_upper_rows_gpu(Rtopj_gpu, Uj_gpu, Ytop_gpu):
    Ytop_gpu -= Rtopj_gpu @ Uj_gpu


# =========================================================
# Optimized blocked_exact nearest-plane path
# =========================================================

_NP_WORKSPACE_CACHE = {}


class _NearestPlaneWorkspace:
    """
    Reusable workspace for blocked_exact nearest-plane.

    Shape convention:
      U_panel / E_panel: (max_block_rows, max_m)
      norms           : (max_m,)
    """
    def __init__(self, block_rows: int, max_m: int):
        self.block_rows = int(block_rows)
        self.max_m = int(max_m)

        self.U_panel = cp.empty((self.block_rows, self.max_m), dtype=cp.float64)
        self.E_panel = cp.empty((self.block_rows, self.max_m), dtype=cp.float64)
        self.norms = cp.empty((self.max_m,), dtype=cp.float64)


def _np_workspace_key(block_rows: int, max_m: int):
    return (int(block_rows), int(max_m))


def get_np_workspace(block_rows: int, max_m: int) -> _NearestPlaneWorkspace:
    key = _np_workspace_key(block_rows, max_m)
    ws = _NP_WORKSPACE_CACHE.get(key)
    if ws is None:
        ws = _NearestPlaneWorkspace(block_rows=block_rows, max_m=max_m)
        _NP_WORKSPACE_CACHE[key] = ws
    return ws


def blocked_babai_panel_gpu_into(Rjj_gpu, Yj_gpu, Uj_gpu, Ej_gpu):
    """
    Equivalent to blocked_babai_panel_gpu, but writes into preallocated buffers.

    Correctness-preserving:
      - same reverse local_i order
      - same rhs computation
      - same cp.rint rounding
      - same Ej definition
    """
    b, m = Yj_gpu.shape

    for local_i in range(b - 1, -1, -1):
        rhs = Yj_gpu[local_i]
        if local_i < b - 1:
            rhs = rhs - Rjj_gpu[local_i, local_i + 1:] @ Uj_gpu[local_i + 1:b, :m]
        else:
            # Keep the original branch behavior. This avoids alias surprises and
            # preserves the old implementation's operation shape.
            rhs = rhs.copy()

        uj = cp.rint(rhs / Rjj_gpu[local_i, local_i])
        Uj_gpu[local_i, :m] = uj
        Ej_gpu[local_i, :m] = rhs - Rjj_gpu[local_i, local_i] * uj



_FUSED_PANEL_KERNEL = None


def get_fused_panel_kernel():
    """Compile/cache one local-row fused Babai panel kernel."""
    global _FUSED_PANEL_KERNEL
    if _FUSED_PANEL_KERNEL is not None:
        return _FUSED_PANEL_KERNEL

    code = r"""
extern "C" __global__
void babai_panel_row_fused(
    const double* __restrict__ R22,
    const double* __restrict__ Ywork,
    double* __restrict__ Uj,
    double* __restrict__ Ej,
    const int ell,
    const int m,
    const int r0,
    const int b,
    const int local_i
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= m) {
        return;
    }

    int global_i = r0 + local_i;
    double rhs = Ywork[global_i * m + col];

    for (int k = local_i + 1; k < b; ++k) {
        double rik = R22[global_i * ell + (r0 + k)];
        rhs -= rik * Uj[k * m + col];
    }

    double diag = R22[global_i * ell + global_i];
    double uj = rint(rhs / diag);

    Uj[local_i * m + col] = uj;
    Ej[local_i * m + col] = rhs - diag * uj;
}
""";
    _FUSED_PANEL_KERNEL = cp.RawKernel(code, "babai_panel_row_fused")
    return _FUSED_PANEL_KERNEL


def blocked_babai_panel_gpu_fused_into(R22_gpu, work_Y_gpu, r0, r1, Uj_gpu, Ej_gpu):
    """
    Fused panel implementation for blocked_exact_fused.

    Correctness boundary:
      - local_i is still processed from b-1 down to 0;
      - each row still rounds immediately after its rhs is computed;
      - only the CuPy expression sequence is fused into one kernel per local_i.
    """
    b = int(r1 - r0)
    m = int(work_Y_gpu.shape[1])
    ell = int(R22_gpu.shape[0])

    kernel = get_fused_panel_kernel()
    threads = 256
    blocks = (m + threads - 1) // threads

    for local_i in range(b - 1, -1, -1):
        kernel(
            (blocks,),
            (threads,),
            (R22_gpu, work_Y_gpu, Uj_gpu, Ej_gpu, ell, m, int(r0), b, int(local_i)),
        )

def batched_nearest_plane_gpu_blocked_exact_norms(
    R22_gpu,
    Y_gpu,
    block_rows=8,
    sync_timing=True,
    fused_panel=False,
):
    """
    Optimized blocked_exact path for the normal scoring path.

    Compared with the original implementation:
      1) does not allocate full U_gpu
      2) does not allocate full E_gpu
      3) accumulates residual norms panel by panel
      4) reuses panel workspace
      5) updates Y_gpu in-place because caller does not reuse raw Y_gpu

    Mathematical behavior:
      The Babai recursion order and rounding rule are unchanged.
    """
    if block_rows <= 0:
        raise ValueError(f"block_rows must be positive, got {block_rows}")

    ell, m = Y_gpu.shape

    # Original code used: work_Y_gpu = Y_gpu.copy()
    # In _run_group_gpu_projected, Y_gpu is only used by nearest-plane after GEMM.
    # Therefore we can use it as the mutable work buffer and avoid one full copy.
    # If future debug code needs the raw Q2T@T matrix after this call, change this
    # line back to: work_Y_gpu = Y_gpu.copy()
    work_Y_gpu = Y_gpu

    parts = build_block_partitions(ell, block_rows)

    ws = get_np_workspace(block_rows=block_rows, max_m=m)
    norms_gpu = ws.norms[:m]
    norms_gpu.fill(0.0)

    np_panel_sec = 0.0
    np_update_gemm_sec = 0.0
    np_norm_sec = 0.0

    for (r0, r1) in parts:
        b = r1 - r0

        Rjj_gpu = R22_gpu[r0:r1, r0:r1]
        Yj_gpu = work_Y_gpu[r0:r1, :]

        Uj_gpu = ws.U_panel[:b, :m]
        Ej_gpu = ws.E_panel[:b, :m]

        if sync_timing:
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()

        if fused_panel:
            blocked_babai_panel_gpu_fused_into(
                R22_gpu=R22_gpu,
                work_Y_gpu=work_Y_gpu,
                r0=r0,
                r1=r1,
                Uj_gpu=Uj_gpu,
                Ej_gpu=Ej_gpu,
            )
        else:
            blocked_babai_panel_gpu_into(
                Rjj_gpu=Rjj_gpu,
                Yj_gpu=Yj_gpu,
                Uj_gpu=Uj_gpu,
                Ej_gpu=Ej_gpu,
            )

        if sync_timing:
            cp.cuda.Stream.null.synchronize()
        np_panel_sec += time.perf_counter() - t0

        # Accumulate current panel residual norm directly. This replaces writing
        # every Ej block into a full E_gpu followed by cp.sum(E_gpu * E_gpu, axis=0).
        if sync_timing:
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()

        norms_gpu += cp.sum(Ej_gpu * Ej_gpu, axis=0)

        if sync_timing:
            cp.cuda.Stream.null.synchronize()
        np_norm_sec += time.perf_counter() - t0

        # Use current panel Uj to update upper rows, exactly as original blocked_exact.
        if r0 > 0:
            Rtopj_gpu = R22_gpu[0:r0, r0:r1]
            Ytop_gpu = work_Y_gpu[0:r0, :]

            if sync_timing:
                cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()

            update_upper_rows_gpu(Rtopj_gpu, Uj_gpu, Ytop_gpu)

            if sync_timing:
                cp.cuda.Stream.null.synchronize()
            np_update_gemm_sec += time.perf_counter() - t0

    detail = {
        "np_panel_sec": np_panel_sec,
        "np_update_gemm_sec": np_update_gemm_sec,
        "np_norm_sec": np_norm_sec,
        "num_blocks": len(parts),
        "block_rows": int(block_rows),
        "optimized_norms_direct": True,
        "inplace_Y": True,
        "fused_panel": bool(fused_panel),
    }
    return norms_gpu, detail


def batched_nearest_plane_gpu_blocked_exact(
    R22_gpu,
    Y_gpu,
    block_rows=8,
    sync_timing=True,
):
    """
    Compatibility wrapper preserving the old return type: U_gpu, E_gpu, detail.

    The optimized scoring path is batched_nearest_plane_gpu_blocked_exact_norms().
    Keep this function for any external/debug caller that still expects full U/E.
    """
    if block_rows <= 0:
        raise ValueError(f"block_rows must be positive, got {block_rows}")

    ell, m = Y_gpu.shape
    work_Y_gpu = Y_gpu.copy()
    U_gpu = cp.zeros((ell, m), dtype=cp.float64)
    E_gpu = cp.zeros((ell, m), dtype=cp.float64)

    parts = build_block_partitions(ell, block_rows)

    np_panel_sec = 0.0
    np_update_gemm_sec = 0.0

    for (r0, r1) in parts:
        Rjj_gpu = R22_gpu[r0:r1, r0:r1]
        Yj_gpu = work_Y_gpu[r0:r1, :]

        if sync_timing:
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()

        Uj_gpu, Ej_gpu = blocked_babai_panel_gpu(Rjj_gpu, Yj_gpu)

        if sync_timing:
            cp.cuda.Stream.null.synchronize()
        np_panel_sec += time.perf_counter() - t0

        U_gpu[r0:r1, :] = Uj_gpu
        E_gpu[r0:r1, :] = Ej_gpu

        if r0 > 0:
            Rtopj_gpu = R22_gpu[0:r0, r0:r1]
            Ytop_gpu = work_Y_gpu[0:r0, :]

            if sync_timing:
                cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()

            update_upper_rows_gpu(Rtopj_gpu, Uj_gpu, Ytop_gpu)

            if sync_timing:
                cp.cuda.Stream.null.synchronize()
            np_update_gemm_sec += time.perf_counter() - t0

    detail = {
        "np_panel_sec": np_panel_sec,
        "np_update_gemm_sec": np_update_gemm_sec,
        "num_blocks": len(parts),
        "block_rows": int(block_rows),
        "compat_wrapper": True,
    }
    return U_gpu, E_gpu, detail

def group_jobs(batch_jobs):
    groups = {}
    for job in batch_jobs:
        chosen_delta = job.meta.get("chosen_delta")
        key = (job.lat_index, chosen_delta)
        groups.setdefault(key, []).append(job)
    return groups


def _format_results_for_group(jobs, proj_norm_sq, backend_name, ell, fallback_reason=None):
    results = []
    for j, job in enumerate(jobs):
        results.append({
            "job_id": job.job_id,
            "lat_index": job.lat_index,
            "exp_index": job.exp_index,
            "branch": job.branch,
            "candidate_index": job.candidate_index,
            "accepted": True,
            "score": float(proj_norm_sq[j]),
            "backend": backend_name,
            "ell": int(ell),
            "fallback_reason": fallback_reason,
        })
    return results


def _sync_if_needed(sync_timing: bool):
    if sync_timing and cp is not None:
        cp.cuda.Stream.null.synchronize()


def _run_group_cpu_projected(
    jobs,
    Q2T,
    R22,
    ell,
    stats,
    verbose=False,
    reason=None,
    target_block_provider=None,
    rowmajor_target_block_provider=None,
):
    t0 = time.perf_counter()
    if target_block_provider is None and rowmajor_target_block_provider is not None and rowmajor_target_block_provider.enabled():
        T_rows = get_T_rows_for_jobs(jobs, rowmajor_target_block_provider=rowmajor_target_block_provider)
        T = np.ascontiguousarray(T_rows.T, dtype=np.float64)
    else:
        T = get_T_matrix_for_jobs(jobs, target_block_provider=target_block_provider)
    Y = Q2T @ T
    _, E = batched_nearest_plane_cpu(R22, Y)
    proj_norm_sq = np.sum(E * E, axis=0)
    dt = time.perf_counter() - t0

    stats["cpu_groups"] += 1
    stats["cpu_jobs"] += len(jobs)
    stats["cpu_time_sec"] += dt
    if reason is not None:
        stats["fallback_reason_counter"][reason] += len(jobs)

    if verbose:
        msg = f"[projected_batch_backend] CPU projected group_jobs={len(jobs)} ell={ell} cpu_time={dt:.6f}s"
        if reason is not None:
            msg += f" reason={reason}"
        print(msg)

    return _format_results_for_group(
        jobs=jobs,
        proj_norm_sq=proj_norm_sq,
        backend_name="cpu_projected",
        ell=ell,
        fallback_reason=reason,
    ), dt


def _run_group_gpu_projected(
    jobs,
    projection_cache,
    group_key,
    ell,
    stats,
    verbose=False,
    sync_timing=True,
    target_block_provider=None,
    rowmajor_target_block_provider=None,
):
    np_impl = HYB_GPU_NP_IMPL
    np_block_rows = int(HYB_GPU_NP_BLOCK_ROWS)

    stage = {
        "host_build_T_sec": 0.0,
        "h2d_copy_sec": 0.0,
        "gemm_sec": 0.0,
        "nearest_plane_sec": 0.0,
        "d2h_copy_sec": 0.0,
        "gpu_group_total_sec": 0.0,
        "np_panel_sec": 0.0,
        "np_update_gemm_sec": 0.0,
        "np_norm_sec": 0.0,
        "np_impl": np_impl,
        "np_block_rows": np_block_rows,
        "T_c_contiguous": None,
        "T_f_contiguous": None,
        "T_strides": None,
        "T_direct_stage": bool(HYB_GPU_DIRECT_STAGE_T),
        "target_layout": "colmajor",
        "rowmajor_requested": bool(HYB_GPU_USE_ROWMAJOR_T),
        "rowmajor_used": False,
        "rowmajor_gemm_mode": None,
        "stage_input_c_contiguous": None,
        "stage_input_f_contiguous": None,
        "stage_input_strides": None,
    }

    t_group0 = time.perf_counter()

    # 1) host build T / T_rows
    use_rowmajor = (
        bool(HYB_GPU_USE_ROWMAJOR_T)
        and rowmajor_target_block_provider is not None
        and rowmajor_target_block_provider.enabled()
    )

    t0 = time.perf_counter()
    if use_rowmajor:
        T_rows = get_T_rows_for_jobs(
            jobs,
            rowmajor_target_block_provider=rowmajor_target_block_provider,
        )
        if T_rows is None:
            use_rowmajor = False
        else:
            stage["target_layout"] = "rowmajor"
            stage["rowmajor_used"] = True
            stage["T_c_contiguous"] = bool(T_rows.flags["C_CONTIGUOUS"])
            stage["T_f_contiguous"] = bool(T_rows.flags["F_CONTIGUOUS"])
            stage["T_strides"] = tuple(int(x) for x in T_rows.strides)

    if not use_rowmajor:
        T = get_T_matrix_for_jobs(
            jobs,
            target_block_provider=target_block_provider,
            ensure_contiguous=not HYB_GPU_DIRECT_STAGE_T,
        )
        stage["target_layout"] = "colmajor"
        stage["rowmajor_used"] = False
        stage["T_c_contiguous"] = bool(T.flags["C_CONTIGUOUS"])
        stage["T_f_contiguous"] = bool(T.flags["F_CONTIGUOUS"])
        stage["T_strides"] = tuple(int(x) for x in T.strides)

    stage["host_build_T_sec"] = time.perf_counter() - t0

    # 2) H2D via pinned host staging + reusable device input buffer
    Q2T_gpu, R22_gpu = projection_cache.ensure_gpu_arrays(group_key, cp)
    if use_rowmajor:
        T_rows_gpu, stage_copy = stage_T_rows_to_gpu(T_rows, sync_timing=sync_timing)
    else:
        T_gpu, stage_copy = stage_T_block_to_gpu(T, sync_timing=sync_timing)

    stage["host_build_T_sec"] += stage_copy["host_copy_sec"]
    stage["h2d_copy_sec"] = stage_copy["h2d_copy_sec"]
    stage["stage_input_c_contiguous"] = stage_copy.get("stage_input_c_contiguous")
    stage["stage_input_f_contiguous"] = stage_copy.get("stage_input_f_contiguous")
    stage["stage_input_strides"] = stage_copy.get("stage_input_strides")

    # 3) GEMM
    _sync_if_needed(sync_timing)
    t0 = time.perf_counter()
    if use_rowmajor:
        # T_rows_gpu: (m, d).  Q2T_gpu.T is a transpose view with shape (d, ell).
        # This is mathematically equivalent to Q2T_gpu @ T_gpu.  We materialize
        # the legacy (ell, m) layout before Babai because the fused RawKernel
        # assumes contiguous row-major indexing over Y_gpu.
        Y_rows_gpu = T_rows_gpu @ Q2T_gpu.T
        Y_gpu = cp.ascontiguousarray(Y_rows_gpu.T)
        stage["rowmajor_gemm_mode"] = "T_rows_at_Q2T_T_then_transpose"
    else:
        Y_gpu = Q2T_gpu @ T_gpu
        stage["rowmajor_gemm_mode"] = None
    _sync_if_needed(sync_timing)
    stage["gemm_sec"] = time.perf_counter() - t0

    # 4) nearest plane
    _sync_if_needed(sync_timing)
    t0 = time.perf_counter()

    if np_impl in ("blocked_exact", "blocked_exact_fused"):
        norms_gpu, np_detail = batched_nearest_plane_gpu_blocked_exact_norms(
            R22_gpu=R22_gpu,
            Y_gpu=Y_gpu,
            block_rows=np_block_rows,
            sync_timing=sync_timing,
            fused_panel=(np_impl == "blocked_exact_fused"),
        )
        stage["np_panel_sec"] = float(np_detail.get("np_panel_sec", 0.0))
        stage["np_update_gemm_sec"] = float(np_detail.get("np_update_gemm_sec", 0.0))
        stage["np_norm_sec"] = float(np_detail.get("np_norm_sec", 0.0))
    elif np_impl == "loop":
        _, E_gpu = batched_nearest_plane_gpu(R22_gpu, Y_gpu)
        norms_gpu = cp.sum(E_gpu * E_gpu, axis=0)
    else:
        raise ValueError(f"Unknown HYB_GPU_NP_IMPL={np_impl!r}")

    _sync_if_needed(sync_timing)
    stage["nearest_plane_sec"] = time.perf_counter() - t0

    # 5) D2H
    _sync_if_needed(sync_timing)
    t0 = time.perf_counter()
    proj_norm_sq = cp.asnumpy(norms_gpu)
    _sync_if_needed(sync_timing)
    stage["d2h_copy_sec"] = time.perf_counter() - t0

    stage["gpu_group_total_sec"] = time.perf_counter() - t_group0

    stats["gpu_groups"] += 1
    stats["gpu_jobs"] += len(jobs)
    stats["gpu_time_sec"] += stage["gpu_group_total_sec"]

    stats["host_build_T_sec"] += stage["host_build_T_sec"]
    stats["h2d_copy_sec"] += stage["h2d_copy_sec"]
    stats["gemm_sec"] += stage["gemm_sec"]
    stats["nearest_plane_sec"] += stage["nearest_plane_sec"]
    stats["d2h_copy_sec"] += stage["d2h_copy_sec"]
    stats["np_panel_sec"] += stage["np_panel_sec"]
    stats["np_update_gemm_sec"] += stage["np_update_gemm_sec"]
    stats["np_norm_sec"] += stage["np_norm_sec"]
    if stage.get("rowmajor_used"):
        stats["rowmajor_groups"] += 1
        stats["rowmajor_jobs"] += len(jobs)
        stats["target_layout_counter"]["rowmajor"] += len(jobs)
    else:
        stats["colmajor_groups"] += 1
        stats["colmajor_jobs"] += len(jobs)
        stats["target_layout_counter"]["colmajor"] += len(jobs)

    if verbose:
        msg = (
            f"[gpu stage] group={group_key} jobs={len(jobs)} ell={ell} "
            f"np_impl={stage['np_impl']} "
        )
        if stage["np_impl"] in ("blocked_exact", "blocked_exact_fused"):
            msg += (
                f"block_rows={stage['np_block_rows']} "
                f"NP_panel={stage['np_panel_sec']:.6f}s "
                f"NP_update={stage['np_update_gemm_sec']:.6f}s "
                f"NP_norm={stage.get('np_norm_sec', 0.0):.6f}s "
            )
        msg += (
            f"T={stage['host_build_T_sec']:.6f}s "
            f"H2D={stage['h2d_copy_sec']:.6f}s "
            f"GEMM={stage['gemm_sec']:.6f}s "
            f"NP={stage['nearest_plane_sec']:.6f}s "
            f"D2H={stage['d2h_copy_sec']:.6f}s "
            f"TOTAL={stage['gpu_group_total_sec']:.6f}s"
        )
        if HYB_GPU_T_LAYOUT_DEBUG:
            msg += (
                f" T_C={stage['T_c_contiguous']} "
                f"T_F={stage['T_f_contiguous']} "
                f"T_strides={stage['T_strides']} "
                f"layout={stage['target_layout']} "
                f"rowmajor={stage['rowmajor_used']} "
                f"direct_stage={stage['T_direct_stage']} "
                f"stage_C={stage['stage_input_c_contiguous']} "
                f"stage_strides={stage['stage_input_strides']}"
            )
        print(msg)

    return _format_results_for_group(
        jobs=jobs,
        proj_norm_sq=proj_norm_sq,
        backend_name="gpu_projected",
        ell=ell,
        fallback_reason=None,
    ), stage["gpu_group_total_sec"], stage


def projected_batch_backend(
    batch_jobs,
    projection_cache,
    target_block_provider=None,
    rowmajor_target_block_provider=None,
    use_gpu=True,
    allow_fallback=True,
    tau_scale=1.0,
    verbose=False,
    sync_timing=True,
):
    runtime = gpu_runtime_info()

    stats = {
        "requested_backend": "gpu_projected" if use_gpu else "cpu_projected",
        "gpu_backend_enabled": use_gpu,
        "cupy_import_ok": runtime["cupy_import_ok"],
        "cupy_import_error": runtime.get("cupy_import_error"),
        "gpu_device_count": runtime.get("gpu_device_count", 0),
        "device_names": runtime.get("device_names", []),
        "gpu_groups": 0,
        "cpu_groups": 0,
        "gpu_jobs": 0,
        "cpu_jobs": 0,
        "gpu_time_sec": 0.0,
        "cpu_time_sec": 0.0,
        "host_build_T_sec": 0.0,
        "h2d_copy_sec": 0.0,
        "gemm_sec": 0.0,
        "nearest_plane_sec": 0.0,
        "d2h_copy_sec": 0.0,
        "np_panel_sec": 0.0,
        "np_update_gemm_sec": 0.0,
        "np_norm_sec": 0.0,
        "rowmajor_groups": 0,
        "rowmajor_jobs": 0,
        "colmajor_groups": 0,
        "colmajor_jobs": 0,
        "target_layout_counter": Counter(),
        "fallback_reason_counter": Counter(),
        "group_debug": [],
    }

    grouped = group_jobs(batch_jobs)
    results = []

    for group_key, jobs in grouped.items():
        cache_item = projection_cache.get(group_key)
        Q2T = cache_item["Q2T"]
        R22 = cache_item["R22"]
        ell = cache_item["ell"]

        group_info = {
            "group_key": group_key,
            "num_jobs": len(jobs),
            "ell": int(ell),
            "backend": None,
            "reason": None,
            "time_sec": None,
            "stage": None,
        }

        # CPU explicit mode
        if not use_gpu:
            group_results, dt = _run_group_cpu_projected(
                jobs=jobs,
                Q2T=Q2T,
                R22=R22,
                ell=ell,
                stats=stats,
                verbose=verbose,
                reason=None,
                target_block_provider=target_block_provider,
                rowmajor_target_block_provider=rowmajor_target_block_provider,
            )
            group_info["backend"] = "cpu_projected"
            group_info["reason"] = None
            group_info["time_sec"] = dt
            stats["group_debug"].append(group_info)
            results.extend(group_results)
            continue

        # GPU explicit mode
        if runtime["gpu_available"]:
            try:
                group_results, dt, stage = _run_group_gpu_projected(
                    jobs=jobs,
                    projection_cache=projection_cache,
                    group_key=group_key,
                    ell=ell,
                    stats=stats,
                    verbose=verbose,
                    sync_timing=sync_timing,
                    target_block_provider=target_block_provider,
                    rowmajor_target_block_provider=rowmajor_target_block_provider,
                )
                group_info["backend"] = "gpu_projected"
                group_info["reason"] = None
                group_info["time_sec"] = dt
                group_info["stage"] = stage
                stats["group_debug"].append(group_info)
                results.extend(group_results)
                continue

            except Exception as e:
                reason = "gpu_exception: " + _short_exc(e)
                if not allow_fallback:
                    raise RuntimeError(
                        f"GPU projected backend failed for group={group_key}: {reason}"
                    ) from e

                group_results, dt = _run_group_cpu_projected(
                    jobs=jobs,
                    Q2T=Q2T,
                    R22=R22,
                    ell=ell,
                    stats=stats,
                    verbose=verbose,
                    reason=reason,
                    target_block_provider=target_block_provider,
                    rowmajor_target_block_provider=rowmajor_target_block_provider,
                )
                group_info["backend"] = "cpu_projected"
                group_info["reason"] = reason
                group_info["time_sec"] = dt
                stats["group_debug"].append(group_info)
                results.extend(group_results)
                continue

        # GPU requested but unavailable
        if not runtime["cupy_import_ok"]:
            reason = "cupy_import_failed"
        elif runtime.get("gpu_device_count", 0) <= 0:
            reason = "no_gpu_device"
        else:
            reason = "gpu_unavailable_unknown"

        if not allow_fallback:
            raise RuntimeError(
                f"GPU projected backend requested but unavailable for group={group_key}: {reason}"
            )

        group_results, dt = _run_group_cpu_projected(
            jobs=jobs,
            Q2T=Q2T,
            R22=R22,
            ell=ell,
            stats=stats,
            verbose=verbose,
            reason=reason,
            target_block_provider=target_block_provider,
            rowmajor_target_block_provider=rowmajor_target_block_provider,
        )
        group_info["backend"] = "cpu_projected"
        group_info["reason"] = reason
        group_info["time_sec"] = dt
        stats["group_debug"].append(group_info)
        results.extend(group_results)

    stats["fallback_reason_counter"] = dict(stats["fallback_reason_counter"])
    stats["target_layout_counter"] = dict(stats["target_layout_counter"])

    return {
        "results": results,
        "stats": stats,
    }