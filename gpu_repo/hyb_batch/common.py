from __future__ import annotations

import hashlib
import numpy as np

from batch_scheduler import BatchScheduler
from projection_cache import ProjectionCache


def compute_target_hash(x: np.ndarray) -> str:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def build_all_targets_matrix_from_scheduler(scheduler: BatchScheduler):
    if len(scheduler.jobs) == 0:
        return np.zeros((0, 0), dtype=np.float64), []

    mats = []
    all_job_ids = []
    base_dim = None

    for job in scheduler.jobs:
        v = np.ascontiguousarray(np.asarray(job.target, dtype=np.float64).reshape(-1))
        if base_dim is None:
            base_dim = v.shape[0]
        elif v.shape[0] != base_dim:
            raise ValueError(
                f"inconsistent target dimension in scheduler: got {v.shape[0]} vs expected {base_dim}"
            )
        mats.append(v)
        all_job_ids.append(int(job.job_id))

    all_targets_matrix = np.stack(mats, axis=1)
    all_targets_matrix = np.ascontiguousarray(all_targets_matrix, dtype=np.float64)
    return all_targets_matrix, all_job_ids


def serialize_projection_cache(projection_cache: ProjectionCache) -> dict:
    out = {}
    for key, item in projection_cache._cache.items():
        out[key] = {
            "Q2T": np.asarray(item["Q2T"], dtype=np.float64),
            "R22": np.asarray(item["R22"], dtype=np.float64),
            "ell": int(item["ell"]),
        }
    return out


def deserialize_projection_cache(serialized: dict) -> ProjectionCache:
    cache = ProjectionCache()
    for key, item in serialized.items():
        cache.put(
            key,
            {
                "Q2T": np.asarray(item["Q2T"], dtype=np.float64),
                "R22": np.asarray(item["R22"], dtype=np.float64),
                "ell": int(item["ell"]),
            },
        )
    return cache


def enrich_results_with_job_info(batch_results, scheduler: BatchScheduler):
    job_map = {job.job_id: job for job in scheduler.jobs}
    enriched = []
    for row in batch_results:
        row = dict(row)
        job = job_map[int(row["job_id"])]
        row["target_hash"] = job.target_hash
        row["target_dim"] = job.target_dim
        enriched.append(row)
    return enriched


def compact_summary_for_print(summary: dict) -> dict:
    keep_keys = [
        "replay_source",
        "gpu_backend_enabled",
        "hyb_batch_max_targets",
        "hyb_gpu_np_impl",
        "hyb_gpu_np_block_rows",
        "hyb_gpu_use_rowmajor_t",
        "dump_version",
        "target_layout_counter",
        "rowmajor_groups",
        "rowmajor_jobs",
        "colmajor_groups",
        "colmajor_jobs",
        "hyb_enumerator_enable",
        "hyb_enum_chunk_size",
        "hyb_wrong_target_count",
        "hyb_correct_target_count",
        "projection_cache_size",
        "batch_total_jobs",
        "batch_num_batches",
        "batch_avg_size",
        "batch_total_accepted",
        "batch_walltime_total_sec",
        "gpu_groups",
        "cpu_groups",
        "gpu_jobs",
        "cpu_jobs",
        "gpu_time_sec",
        "cpu_time_sec",
        "gpu_throughput",
        "gpu_cost_per_job",
        "cpu_throughput",
        "cpu_cost_per_job",
        "throughput_ratio_gpu_over_cpu",
        "host_build_T_sec",
        "h2d_copy_sec",
        "gemm_sec",
        "nearest_plane_sec",
        "d2h_copy_sec",
        "np_panel_sec",
        "np_update_gemm_sec",
        "np_norm_sec",
        "validation_enabled",
        "validation_batches_requested",
        "validation_batches_run",
        "validation_exact_equal",
        "validation_max_abs_err",
        "validation_max_rel_err",
        "backend_counter",
        "gpu_fallback_reasons",
    ]
    out = {k: summary.get(k) for k in keep_keys if k in summary}

    if "gpu_runtime_info" in summary:
        gri = summary["gpu_runtime_info"]
        out["gpu_runtime_info_brief"] = {
            "cupy_import_ok": gri.get("cupy_import_ok"),
            "gpu_available": gri.get("gpu_available"),
            "gpu_device_count": gri.get("gpu_device_count"),
            "device_names": gri.get("device_names"),
        }

    return out
