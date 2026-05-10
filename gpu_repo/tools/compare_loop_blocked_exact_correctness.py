import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import math
import pickle
from collections import Counter

import numpy as np

import gpu_projected_batch_backend as gpb
from batch_scheduler import BatchScheduler
from run_prog_hyb import deserialize_projection_cache


def run_one_mode(
    dump_path: str,
    np_impl: str,
    np_block_rows: int,
    batch_max_targets: int,
    verbose: bool = False,
):
    """
    在同一份 dump 上运行一次 projected backend，
    返回:
      rows_by_job: {job_id: row}
      summary: dict
    """
    with open(dump_path, "rb") as f:
        dump = pickle.load(f)

    scheduler = BatchScheduler.from_serializable(dump["scheduler"])
    scheduler.max_targets = batch_max_targets
    projection_cache = deserialize_projection_cache(dump["projection_cache"])

    # 直接覆盖 gpu_projected_batch_backend 模块里的实现开关
    gpb.HYB_GPU_NP_IMPL = np_impl
    gpb.HYB_GPU_NP_BLOCK_ROWS = np_block_rows

    all_rows = []
    batch_sizes = []
    batch_count = 0
    gpu_groups = 0
    cpu_groups = 0
    gpu_jobs = 0
    cpu_jobs = 0
    gpu_time_sec = 0.0
    cpu_time_sec = 0.0
    host_build_T_sec = 0.0
    h2d_copy_sec = 0.0
    gemm_sec = 0.0
    nearest_plane_sec = 0.0
    d2h_copy_sec = 0.0
    np_panel_sec = 0.0
    np_update_gemm_sec = 0.0
    fallback_counter = Counter()

    for batch_jobs in scheduler.iter_batches():
        batch_count += 1
        batch_sizes.append(len(batch_jobs))

        out = gpb.projected_batch_backend(
            batch_jobs=batch_jobs,
            projection_cache=projection_cache,
            use_gpu=True,
            allow_fallback=False,
            tau_scale=1.0,
            verbose=verbose,
            sync_timing=True,
        )

        rows = out["results"]
        stats = out["stats"]

        all_rows.extend(rows)

        gpu_groups += stats.get("gpu_groups", 0)
        cpu_groups += stats.get("cpu_groups", 0)
        gpu_jobs += stats.get("gpu_jobs", 0)
        cpu_jobs += stats.get("cpu_jobs", 0)
        gpu_time_sec += stats.get("gpu_time_sec", 0.0)
        cpu_time_sec += stats.get("cpu_time_sec", 0.0)
        host_build_T_sec += stats.get("host_build_T_sec", 0.0)
        h2d_copy_sec += stats.get("h2d_copy_sec", 0.0)
        gemm_sec += stats.get("gemm_sec", 0.0)
        nearest_plane_sec += stats.get("nearest_plane_sec", 0.0)
        d2h_copy_sec += stats.get("d2h_copy_sec", 0.0)
        np_panel_sec += stats.get("np_panel_sec", 0.0)
        np_update_gemm_sec += stats.get("np_update_gemm_sec", 0.0)

        for k, v in stats.get("fallback_reason_counter", {}).items():
            fallback_counter[k] += v

    rows_by_job = {}
    dup_job_ids = []
    for row in all_rows:
        jid = int(row["job_id"])
        if jid in rows_by_job:
            dup_job_ids.append(jid)
        rows_by_job[jid] = row

    summary = {
        "np_impl": np_impl,
        "np_block_rows": np_block_rows,
        "batch_max_targets": batch_max_targets,
        "num_rows": len(all_rows),
        "num_unique_jobs": len(rows_by_job),
        "dup_job_ids": dup_job_ids,
        "batch_count": batch_count,
        "batch_avg_size": (sum(batch_sizes) / len(batch_sizes)) if batch_sizes else 0.0,
        "gpu_groups": gpu_groups,
        "cpu_groups": cpu_groups,
        "gpu_jobs": gpu_jobs,
        "cpu_jobs": cpu_jobs,
        "gpu_time_sec": gpu_time_sec,
        "cpu_time_sec": cpu_time_sec,
        "host_build_T_sec": host_build_T_sec,
        "h2d_copy_sec": h2d_copy_sec,
        "gemm_sec": gemm_sec,
        "nearest_plane_sec": nearest_plane_sec,
        "d2h_copy_sec": d2h_copy_sec,
        "np_panel_sec": np_panel_sec,
        "np_update_gemm_sec": np_update_gemm_sec,
        "fallback_counter": dict(fallback_counter),
    }
    return rows_by_job, summary


def compare_rows(loop_rows, blk_rows, top_k=10):
    loop_ids = set(loop_rows.keys())
    blk_ids = set(blk_rows.keys())

    only_loop = sorted(loop_ids - blk_ids)
    only_blk = sorted(blk_ids - loop_ids)
    common = sorted(loop_ids & blk_ids)

    accepted_mismatch = []
    score_diffs = []

    exact_equal_count = 0
    abs_errs = []
    rel_errs = []

    for jid in common:
        a = loop_rows[jid]
        b = blk_rows[jid]

        if bool(a.get("accepted")) != bool(b.get("accepted")):
            accepted_mismatch.append(jid)

        sa = float(a["score"])
        sb = float(b["score"])
        abs_err = abs(sa - sb)
        rel_err = abs_err / max(abs(sa), 1e-18)

        if sa == sb:
            exact_equal_count += 1

        abs_errs.append(abs_err)
        rel_errs.append(rel_err)
        score_diffs.append({
            "job_id": jid,
            "loop_score": sa,
            "blocked_score": sb,
            "abs_err": abs_err,
            "rel_err": rel_err,
            "loop_accepted": bool(a.get("accepted")),
            "blocked_accepted": bool(b.get("accepted")),
        })

    score_diffs.sort(key=lambda x: x["abs_err"], reverse=True)

    out = {
        "loop_only_count": len(only_loop),
        "blocked_only_count": len(only_blk),
        "common_count": len(common),
        "accepted_mismatch_count": len(accepted_mismatch),
        "accepted_mismatch_job_ids": accepted_mismatch[:top_k],
        "exact_equal_count": exact_equal_count,
        "max_abs_err": max(abs_errs) if abs_errs else math.nan,
        "mean_abs_err": float(np.mean(abs_errs)) if abs_errs else math.nan,
        "median_abs_err": float(np.median(abs_errs)) if abs_errs else math.nan,
        "max_rel_err": max(rel_errs) if rel_errs else math.nan,
        "mean_rel_err": float(np.mean(rel_errs)) if rel_errs else math.nan,
        "median_rel_err": float(np.median(rel_errs)) if rel_errs else math.nan,
        "top_abs_err_rows": score_diffs[:top_k],
    }
    return out


def print_summary(title, summary):
    print(f"=== {title} ===")
    for k, v in summary.items():
        if k == "dup_job_ids":
            print(f"{k}: {v[:10]}{' ...' if len(v) > 10 else ''}")
        else:
            print(f"{k}: {v}")
    print()


def print_compare(cmp_result):
    print("=== Loop vs Blocked Exact(16) Correctness Compare ===")
    for k in [
        "loop_only_count",
        "blocked_only_count",
        "common_count",
        "accepted_mismatch_count",
        "accepted_mismatch_job_ids",
        "exact_equal_count",
        "max_abs_err",
        "mean_abs_err",
        "median_abs_err",
        "max_rel_err",
        "mean_rel_err",
        "median_rel_err",
    ]:
        print(f"{k}: {cmp_result[k]}")

    print("\nTop absolute score differences:")
    for row in cmp_result["top_abs_err_rows"]:
        print(
            f"job_id={row['job_id']}, "
            f"loop={row['loop_score']:.15f}, "
            f"blocked={row['blocked_score']:.15f}, "
            f"abs_err={row['abs_err']:.15e}, "
            f"rel_err={row['rel_err']:.15e}, "
            f"loop_acc={row['loop_accepted']}, "
            f"blk_acc={row['blocked_accepted']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare correctness of loop vs blocked_exact(16) on the same dump."
    )
    parser.add_argument(
        "--dump",
        required=True,
        help="Path to batch_candidates dump pkl",
    )
    parser.add_argument(
        "--batch_max_targets",
        type=int,
        default=4096,
        help="Replay batch max targets",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Show top-k score differences",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose backend output",
    )
    args = parser.parse_args()

    loop_rows, loop_summary = run_one_mode(
        dump_path=args.dump,
        np_impl="loop",
        np_block_rows=8,
        batch_max_targets=args.batch_max_targets,
        verbose=args.verbose,
    )

    blk_rows, blk_summary = run_one_mode(
        dump_path=args.dump,
        np_impl="blocked_exact",
        np_block_rows=16,
        batch_max_targets=args.batch_max_targets,
        verbose=args.verbose,
    )

    cmp_result = compare_rows(loop_rows, blk_rows, top_k=args.top_k)

    print_summary("Loop summary", loop_summary)
    print_summary("BlockedExact(16) summary", blk_summary)
    print_compare(cmp_result)


if __name__ == "__main__":
    main()