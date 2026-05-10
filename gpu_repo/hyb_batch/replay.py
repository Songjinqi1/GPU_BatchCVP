from __future__ import annotations

import pickle
from time import perf_counter

from batch_scheduler import BatchScheduler
from projection_cache import ProjectionCache
from gpu_projected_batch_backend import (
    projected_batch_backend,
    gpu_runtime_info,
    TargetBlockProvider,
    RowMajorTargetBlockProvider,
)
from candidate_enumerator import compute_throughput_stats
from global_consts import *

from .common import deserialize_projection_cache, enrich_results_with_job_info
from .dump import prepare_batch_material
from .config import ReplayConfig, BatchRuntimeConfig, HybridBaseParams, write_json_summary


def _empty_validation_summary(enabled=False, requested=0):
    return {
        "validation_enabled": bool(enabled),
        "validation_batches_requested": int(requested),
        "validation_batches_run": 0,
        "validation_exact_equal": None if enabled else True,
        "validation_max_abs_err": 0.0,
        "validation_max_rel_err": 0.0,
    }


def _new_backend_stats_agg():
    return {
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
        "target_layout_counter": {},
        "fallback_reason_counter": {},
        "group_debug": [],
    }


def _accumulate_backend_stats(backend_stats_agg, batch_stats):
    backend_stats_agg["gpu_groups"] += batch_stats.get("gpu_groups", 0)
    backend_stats_agg["cpu_groups"] += batch_stats.get("cpu_groups", 0)
    backend_stats_agg["gpu_jobs"] += batch_stats.get("gpu_jobs", 0)
    backend_stats_agg["cpu_jobs"] += batch_stats.get("cpu_jobs", 0)
    backend_stats_agg["gpu_time_sec"] += batch_stats.get("gpu_time_sec", 0.0)
    backend_stats_agg["cpu_time_sec"] += batch_stats.get("cpu_time_sec", 0.0)
    backend_stats_agg["host_build_T_sec"] += batch_stats.get("host_build_T_sec", 0.0)
    backend_stats_agg["h2d_copy_sec"] += batch_stats.get("h2d_copy_sec", 0.0)
    backend_stats_agg["gemm_sec"] += batch_stats.get("gemm_sec", 0.0)
    backend_stats_agg["nearest_plane_sec"] += batch_stats.get("nearest_plane_sec", 0.0)
    backend_stats_agg["d2h_copy_sec"] += batch_stats.get("d2h_copy_sec", 0.0)
    backend_stats_agg["np_panel_sec"] += batch_stats.get("np_panel_sec", 0.0)
    backend_stats_agg["np_update_gemm_sec"] += batch_stats.get("np_update_gemm_sec", 0.0)
    backend_stats_agg["np_norm_sec"] += batch_stats.get("np_norm_sec", 0.0)
    backend_stats_agg["rowmajor_groups"] += batch_stats.get("rowmajor_groups", 0)
    backend_stats_agg["rowmajor_jobs"] += batch_stats.get("rowmajor_jobs", 0)
    backend_stats_agg["colmajor_groups"] += batch_stats.get("colmajor_groups", 0)
    backend_stats_agg["colmajor_jobs"] += batch_stats.get("colmajor_jobs", 0)
    for lk, lv in batch_stats.get("target_layout_counter", {}).items():
        backend_stats_agg["target_layout_counter"][lk] = (
            backend_stats_agg["target_layout_counter"].get(lk, 0) + lv
        )
    backend_stats_agg["group_debug"].extend(batch_stats.get("group_debug", []))
    for k, v in batch_stats.get("fallback_reason_counter", {}).items():
        backend_stats_agg["fallback_reason_counter"][k] = (
            backend_stats_agg["fallback_reason_counter"].get(k, 0) + v
        )


def validate_projected_backend_once(
    batch_jobs,
    projection_cache,
    target_block_provider=None,
    rowmajor_target_block_provider=None,
    atol=1e-8,
    rtol=1e-8,
    max_print=10,
    sync_timing=True,
):
    gpu_out = projected_batch_backend(
        batch_jobs=batch_jobs,
        projection_cache=projection_cache,
        target_block_provider=target_block_provider,
        rowmajor_target_block_provider=rowmajor_target_block_provider,
        use_gpu=True,
        allow_fallback=False,
        tau_scale=HYB_GPU_PROJ_TAU_SCALE,
        verbose=False,
        sync_timing=sync_timing,
    )

    cpu_out = projected_batch_backend(
        batch_jobs=batch_jobs,
        projection_cache=projection_cache,
        target_block_provider=target_block_provider,
        rowmajor_target_block_provider=rowmajor_target_block_provider,
        use_gpu=False,
        allow_fallback=False,
        tau_scale=HYB_GPU_PROJ_TAU_SCALE,
        verbose=False,
        sync_timing=sync_timing,
    )

    gpu_scores = {int(r["job_id"]): float(r["score"]) for r in gpu_out["results"]}
    cpu_scores = {int(r["job_id"]): float(r["score"]) for r in cpu_out["results"]}

    if set(gpu_scores.keys()) != set(cpu_scores.keys()):
        missing_gpu = sorted(set(cpu_scores.keys()) - set(gpu_scores.keys()))[:max_print]
        missing_cpu = sorted(set(gpu_scores.keys()) - set(cpu_scores.keys()))[:max_print]
        raise RuntimeError(
            f"[validate projected backend] job_id set mismatch: "
            f"missing_gpu={missing_gpu}, missing_cpu={missing_cpu}"
        )

    max_abs = 0.0
    max_rel = 0.0
    bad = []
    for jid in sorted(gpu_scores.keys()):
        g = gpu_scores[jid]
        c = cpu_scores[jid]
        abs_err = abs(g - c)
        rel_err = abs_err / max(1.0, abs(c))
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        if not (abs_err <= atol + rtol * abs(c)) and len(bad) < max_print:
            bad.append((jid, c, g, abs_err, rel_err))

    exact_equal = len(bad) == 0
    print(
        "[validate projected backend] "
        f"jobs={len(batch_jobs)} exact_equal={exact_equal} "
        f"max_abs_err={max_abs:.6e} max_rel_err={max_rel:.6e} "
        f"atol={atol:.1e} rtol={rtol:.1e}"
    )
    if not exact_equal:
        print("[validate projected backend] first bad examples:")
        for jid, c, g, ae, re in bad:
            print(
                f"  job_id={jid} cpu={c:.17e} gpu={g:.17e} "
                f"abs_err={ae:.6e} rel_err={re:.6e}"
            )
        raise RuntimeError("[validate projected backend] CPU/GPU score mismatch")

    return {
        "exact_equal": exact_equal,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "jobs": len(batch_jobs),
    }


def replay_batch_candidate_dump(
    dump_path: str | ReplayConfig,
    validate_projected_batches: int = 0,
    validate_atol: float = 1e-8,
    validate_rtol: float = 1e-8,
):
    json_out = None
    if isinstance(dump_path, ReplayConfig):
        cfg = dump_path
        dump_path = cfg.dump_path
        validate_projected_batches = cfg.validate_projected_batches
        validate_atol = cfg.validate_atol
        validate_rtol = cfg.validate_rtol
        json_out = cfg.json_out
        globals()["HYB_BATCH_MAX_TARGETS"] = cfg.runtime.max_targets
        globals()["HYB_GPU_BACKEND_ENABLE"] = cfg.runtime.gpu_backend_enable
        globals()["HYB_GPU_BACKEND_VERBOSE"] = cfg.runtime.gpu_backend_verbose
        globals()["HYB_GPU_PROJ_TAU_SCALE"] = cfg.runtime.gpu_proj_tau_scale
        globals()["HYB_GPU_ALLOW_FALLBACK"] = cfg.runtime.gpu_allow_fallback
        globals()["HYB_GPU_SYNC_TIMING"] = cfg.runtime.gpu_sync_timing
        globals()["HYB_GPU_NP_IMPL"] = cfg.runtime.gpu_np_impl
        globals()["HYB_GPU_NP_BLOCK_ROWS"] = cfg.runtime.gpu_np_block_rows
        globals()["HYB_GPU_USE_ROWMAJOR_T"] = cfg.runtime.gpu_use_rowmajor_t
        globals()["HYB_ENUMERATOR_ENABLE"] = cfg.runtime.enumerator_enable
        globals()["HYB_ENUMERATOR_CHUNK_SIZE"] = cfg.runtime.enum_chunk_size
        globals()["HYB_WRONG_TARGET_COUNT"] = cfg.runtime.wrong_target_count
        globals()["HYB_CORRECT_TARGET_COUNT"] = cfg.runtime.correct_target_count

    with open(dump_path, "rb") as f:
        dump = pickle.load(f)

    dump_version = int(dump.get("dump_version", -1))
    if dump_version not in (1, 2, 3):
        raise ValueError(f"Unsupported dump_version: {dump.get('dump_version')}")

    scheduler = BatchScheduler.from_serializable(dump["scheduler"])
    scheduler.max_targets = HYB_BATCH_MAX_TARGETS
    projection_cache = deserialize_projection_cache(dump["projection_cache"])
    per_instance_info = dump.get("per_instance_info", [])
    gpu_info = gpu_runtime_info()

    if dump_version >= 2 and "all_targets_matrix" in dump and "all_job_ids" in dump:
        target_block_provider = TargetBlockProvider(
            all_targets_matrix=dump["all_targets_matrix"],
            all_job_ids=dump["all_job_ids"],
        )
    else:
        target_block_provider = None

    if dump_version >= 3 and "all_targets_matrix_rowmajor" in dump and "all_job_ids" in dump:
        rowmajor_target_block_provider = RowMajorTargetBlockProvider(
            all_targets_matrix_rowmajor=dump["all_targets_matrix_rowmajor"],
            all_job_ids=dump["all_job_ids"],
        )
    else:
        rowmajor_target_block_provider = None

    all_batch_results = []
    batch_sizes = []
    num_batches = 0
    backend_stats_agg = _new_backend_stats_agg()
    validation_summary = _empty_validation_summary(
        enabled=validate_projected_batches > 0,
        requested=validate_projected_batches,
    )

    batch_wall_start = perf_counter()
    for batch_jobs in scheduler.iter_batches():
        num_batches += 1
        batch_sizes.append(len(batch_jobs))

        if validate_projected_batches > 0 and num_batches <= validate_projected_batches:
            val = validate_projected_backend_once(
                batch_jobs=batch_jobs,
                projection_cache=projection_cache,
                target_block_provider=target_block_provider,
                rowmajor_target_block_provider=rowmajor_target_block_provider,
                atol=validate_atol,
                rtol=validate_rtol,
                sync_timing=HYB_GPU_SYNC_TIMING,
            )
            validation_summary["validation_batches_run"] += 1
            validation_summary["validation_exact_equal"] = (
                validation_summary["validation_exact_equal"] is not False and bool(val["exact_equal"])
            )
            validation_summary["validation_max_abs_err"] = max(
                validation_summary["validation_max_abs_err"], float(val["max_abs_err"])
            )
            validation_summary["validation_max_rel_err"] = max(
                validation_summary["validation_max_rel_err"], float(val["max_rel_err"])
            )

        backend_output = projected_batch_backend(
            batch_jobs=batch_jobs,
            projection_cache=projection_cache,
            target_block_provider=target_block_provider,
            rowmajor_target_block_provider=rowmajor_target_block_provider,
            use_gpu=HYB_GPU_BACKEND_ENABLE,
            allow_fallback=HYB_GPU_ALLOW_FALLBACK,
            tau_scale=HYB_GPU_PROJ_TAU_SCALE,
            verbose=HYB_GPU_BACKEND_VERBOSE,
            sync_timing=HYB_GPU_SYNC_TIMING,
        )
        batch_results = enrich_results_with_job_info(
            batch_results=backend_output["results"],
            scheduler=scheduler,
        )
        batch_stats = backend_output["stats"]
        all_batch_results.extend(batch_results)
        _accumulate_backend_stats(backend_stats_agg, batch_stats)

    batch_wall_total = perf_counter() - batch_wall_start
    accepted_count = sum(1 for r in all_batch_results if r.get("accepted", False))
    backend_counter = {}
    for r in all_batch_results:
        b = r.get("backend", "unknown")
        backend_counter[b] = backend_counter.get(b, 0) + 1

    gpu_tp = compute_throughput_stats(
        total_jobs=backend_stats_agg["gpu_jobs"],
        elapsed_sec=backend_stats_agg["gpu_time_sec"],
    )
    cpu_tp = compute_throughput_stats(
        total_jobs=backend_stats_agg["cpu_jobs"],
        elapsed_sec=backend_stats_agg["cpu_time_sec"],
    )

    return {
        "replay_source": dump_path,
        "dump_version": dump_version,
        "gpu_backend_enabled": HYB_GPU_BACKEND_ENABLE,
        "hyb_batch_max_targets": HYB_BATCH_MAX_TARGETS,
        "hyb_gpu_np_impl": HYB_GPU_NP_IMPL,
        "hyb_gpu_np_block_rows": HYB_GPU_NP_BLOCK_ROWS,
        "hyb_gpu_use_rowmajor_t": HYB_GPU_USE_ROWMAJOR_T,
        "hyb_enumerator_enable": HYB_ENUMERATOR_ENABLE,
        "hyb_enum_chunk_size": HYB_ENUMERATOR_CHUNK_SIZE,
        "hyb_wrong_target_count": HYB_WRONG_TARGET_COUNT,
        "hyb_correct_target_count": HYB_CORRECT_TARGET_COUNT,
        "gpu_runtime_info": gpu_info,
        "projection_cache_size": len(projection_cache),
        "projection_cache_debug": projection_cache.describe(),
        "batch_total_jobs": scheduler.num_jobs(),
        "batch_num_batches": num_batches,
        "batch_avg_size": (sum(batch_sizes) / len(batch_sizes)) if batch_sizes else 0.0,
        "batch_total_accepted": accepted_count,
        "batch_walltime_total_sec": batch_wall_total,
        "gpu_groups": backend_stats_agg["gpu_groups"],
        "cpu_groups": backend_stats_agg["cpu_groups"],
        "gpu_jobs": backend_stats_agg["gpu_jobs"],
        "cpu_jobs": backend_stats_agg["cpu_jobs"],
        "gpu_time_sec": backend_stats_agg["gpu_time_sec"],
        "cpu_time_sec": backend_stats_agg["cpu_time_sec"],
        "gpu_throughput": gpu_tp["throughput"],
        "gpu_cost_per_job": gpu_tp["cost_per_job"],
        "cpu_throughput": cpu_tp["throughput"],
        "cpu_cost_per_job": cpu_tp["cost_per_job"],
        "throughput_ratio_gpu_over_cpu": (
            gpu_tp["throughput"] / cpu_tp["throughput"] if cpu_tp["throughput"] > 0 else None
        ),
        "host_build_T_sec": backend_stats_agg["host_build_T_sec"],
        "h2d_copy_sec": backend_stats_agg["h2d_copy_sec"],
        "gemm_sec": backend_stats_agg["gemm_sec"],
        "nearest_plane_sec": backend_stats_agg["nearest_plane_sec"],
        "d2h_copy_sec": backend_stats_agg["d2h_copy_sec"],
        "np_panel_sec": backend_stats_agg["np_panel_sec"],
        "np_update_gemm_sec": backend_stats_agg["np_update_gemm_sec"],
        "np_norm_sec": backend_stats_agg["np_norm_sec"],
        "rowmajor_groups": backend_stats_agg["rowmajor_groups"],
        "rowmajor_jobs": backend_stats_agg["rowmajor_jobs"],
        "colmajor_groups": backend_stats_agg["colmajor_groups"],
        "colmajor_jobs": backend_stats_agg["colmajor_jobs"],
        "target_layout_counter": backend_stats_agg["target_layout_counter"],
        **validation_summary,
        "backend_counter": backend_counter,
        "gpu_fallback_reasons": backend_stats_agg["fallback_reason_counter"],
        "gpu_group_debug": backend_stats_agg["group_debug"],
        "per_instance_info": per_instance_info,
        "all_batch_results": all_batch_results,
    }
    if json_out:
        summary["json_summary_path"] = write_json_summary(summary, json_out)
    return summary


def run_experiment_batch(params, delta_slicer_coord=0, runtime: BatchRuntimeConfig | None = None):
    if isinstance(params, HybridBaseParams):
        params = params.to_legacy_params()
    if runtime is not None:
        globals()["HYB_BATCH_MAX_TARGETS"] = runtime.max_targets
        globals()["HYB_GPU_BACKEND_ENABLE"] = runtime.gpu_backend_enable
        globals()["HYB_GPU_BACKEND_VERBOSE"] = runtime.gpu_backend_verbose
        globals()["HYB_GPU_PROJ_TAU_SCALE"] = runtime.gpu_proj_tau_scale
        globals()["HYB_GPU_ALLOW_FALLBACK"] = runtime.gpu_allow_fallback
        globals()["HYB_GPU_SYNC_TIMING"] = runtime.gpu_sync_timing
        globals()["HYB_GPU_NP_IMPL"] = runtime.gpu_np_impl
        globals()["HYB_GPU_NP_BLOCK_ROWS"] = runtime.gpu_np_block_rows
        globals()["HYB_GPU_USE_ROWMAJOR_T"] = runtime.gpu_use_rowmajor_t
        globals()["HYB_ENUMERATOR_ENABLE"] = runtime.enumerator_enable
        globals()["HYB_ENUMERATOR_CHUNK_SIZE"] = runtime.enum_chunk_size
        globals()["HYB_WRONG_TARGET_COUNT"] = runtime.wrong_target_count
        globals()["HYB_CORRECT_TARGET_COUNT"] = runtime.correct_target_count
    scheduler, projection_cache, per_instance_info = prepare_batch_material(
        params=params,
        delta_slicer_coord=delta_slicer_coord,
    )
    gpu_info = gpu_runtime_info()
    all_batch_results = []
    batch_sizes = []
    num_batches = 0
    backend_stats_agg = _new_backend_stats_agg()

    batch_wall_start = perf_counter()
    for batch_jobs in scheduler.iter_batches():
        num_batches += 1
        batch_sizes.append(len(batch_jobs))
        backend_output = projected_batch_backend(
            batch_jobs=batch_jobs,
            projection_cache=projection_cache,
            target_block_provider=None,
            use_gpu=HYB_GPU_BACKEND_ENABLE,
            allow_fallback=HYB_GPU_ALLOW_FALLBACK,
            tau_scale=HYB_GPU_PROJ_TAU_SCALE,
            verbose=HYB_GPU_BACKEND_VERBOSE,
            sync_timing=HYB_GPU_SYNC_TIMING,
        )
        batch_results = enrich_results_with_job_info(
            batch_results=backend_output["results"],
            scheduler=scheduler,
        )
        batch_stats = backend_output["stats"]
        all_batch_results.extend(batch_results)
        _accumulate_backend_stats(backend_stats_agg, batch_stats)

    batch_wall_total = perf_counter() - batch_wall_start
    accepted_count = sum(1 for r in all_batch_results if r.get("accepted", False))
    backend_counter = {}
    for r in all_batch_results:
        b = r.get("backend", "unknown")
        backend_counter[b] = backend_counter.get(b, 0) + 1

    gpu_tp = compute_throughput_stats(
        total_jobs=backend_stats_agg["gpu_jobs"],
        elapsed_sec=backend_stats_agg["gpu_time_sec"],
    )
    cpu_tp = compute_throughput_stats(
        total_jobs=backend_stats_agg["cpu_jobs"],
        elapsed_sec=backend_stats_agg["cpu_time_sec"],
    )

    return {
        "gpu_backend_enabled": HYB_GPU_BACKEND_ENABLE,
        "hyb_batch_max_targets": HYB_BATCH_MAX_TARGETS,
        "hyb_gpu_np_impl": HYB_GPU_NP_IMPL,
        "hyb_gpu_np_block_rows": HYB_GPU_NP_BLOCK_ROWS,
        "hyb_gpu_use_rowmajor_t": HYB_GPU_USE_ROWMAJOR_T,
        "hyb_enumerator_enable": HYB_ENUMERATOR_ENABLE,
        "hyb_enum_chunk_size": HYB_ENUMERATOR_CHUNK_SIZE,
        "hyb_wrong_target_count": HYB_WRONG_TARGET_COUNT,
        "hyb_correct_target_count": HYB_CORRECT_TARGET_COUNT,
        "gpu_runtime_info": gpu_info,
        "projection_cache_size": len(projection_cache),
        "projection_cache_debug": projection_cache.describe(),
        "batch_total_jobs": scheduler.num_jobs(),
        "batch_num_batches": num_batches,
        "batch_avg_size": (sum(batch_sizes) / len(batch_sizes)) if batch_sizes else 0.0,
        "batch_total_accepted": accepted_count,
        "batch_walltime_total_sec": batch_wall_total,
        "gpu_groups": backend_stats_agg["gpu_groups"],
        "cpu_groups": backend_stats_agg["cpu_groups"],
        "gpu_jobs": backend_stats_agg["gpu_jobs"],
        "cpu_jobs": backend_stats_agg["cpu_jobs"],
        "gpu_time_sec": backend_stats_agg["gpu_time_sec"],
        "cpu_time_sec": backend_stats_agg["cpu_time_sec"],
        "gpu_throughput": gpu_tp["throughput"],
        "gpu_cost_per_job": gpu_tp["cost_per_job"],
        "cpu_throughput": cpu_tp["throughput"],
        "cpu_cost_per_job": cpu_tp["cost_per_job"],
        "throughput_ratio_gpu_over_cpu": (
            gpu_tp["throughput"] / cpu_tp["throughput"] if cpu_tp["throughput"] > 0 else None
        ),
        "host_build_T_sec": backend_stats_agg["host_build_T_sec"],
        "h2d_copy_sec": backend_stats_agg["h2d_copy_sec"],
        "gemm_sec": backend_stats_agg["gemm_sec"],
        "nearest_plane_sec": backend_stats_agg["nearest_plane_sec"],
        "d2h_copy_sec": backend_stats_agg["d2h_copy_sec"],
        "np_panel_sec": backend_stats_agg["np_panel_sec"],
        "np_update_gemm_sec": backend_stats_agg["np_update_gemm_sec"],
        "np_norm_sec": backend_stats_agg["np_norm_sec"],
        "rowmajor_groups": backend_stats_agg["rowmajor_groups"],
        "rowmajor_jobs": backend_stats_agg["rowmajor_jobs"],
        "colmajor_groups": backend_stats_agg["colmajor_groups"],
        "colmajor_jobs": backend_stats_agg["colmajor_jobs"],
        "target_layout_counter": backend_stats_agg["target_layout_counter"],
        "backend_counter": backend_counter,
        "gpu_fallback_reasons": backend_stats_agg["fallback_reason_counter"],
        "gpu_group_debug": backend_stats_agg["group_debug"],
        "per_instance_info": per_instance_info,
        "all_batch_results": all_batch_results,
    }
