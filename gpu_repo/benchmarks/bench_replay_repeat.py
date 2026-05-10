#!/usr/bin/env python3
"""
Repeat benchmark wrapper for run_prog_hyb.py replay mode.

Purpose:
  - avoid judging GPU/CPU from one noisy replay run;
  - optionally warm up GPU once;
  - collect median/min/max for important timing fields.

Examples:
  python bench_replay_repeat.py \
    --dump results/dumps/guess6/strict_L1_inst2048_guess6_jobs262144_v2.pkl \
    --gpu-runs 5 --cpu-runs 3 --gpu-warmup 1

  HYB_GPU_NP_IMPL=blocked_exact_fused python bench_replay_repeat.py \
    --dump results/dumps/guess6/strict_L1_inst2048_guess6_jobs262144_v2.pkl \
    --gpu-runs 5 --cpu-runs 3 --gpu-warmup 1 --validate-batches 1
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = REPO_ROOT.parent


TIMING_KEYS = [
    "batch_walltime_total_sec",
    "gpu_time_sec",
    "cpu_time_sec",
    "gpu_throughput",
    "cpu_throughput",
    "host_build_T_sec",
    "h2d_copy_sec",
    "gemm_sec",
    "nearest_plane_sec",
    "d2h_copy_sec",
    "np_panel_sec",
    "np_update_gemm_sec",
    "np_norm_sec",
]


def parse_summary_from_stdout(stdout: str) -> Dict[str, Any]:
    """Find the last printed Python dict summary in run_prog_hyb.py output.

    run_prog_hyb.py prints a Python dict. Some numeric fields can be
    printed as bare `inf` when the opposite backend has no jobs, e.g.
    `cpu_cost_per_job: inf` during GPU-only runs. ast.literal_eval cannot
    parse bare `inf`, so we fall back to a restricted eval that only exposes
    inf/nan.
    """
    last_candidate = None

    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue

        last_candidate = line

        try:
            obj = ast.literal_eval(line)
        except Exception:
            try:
                obj = eval(
                    line,
                    {"__builtins__": {}},
                    {"inf": float("inf"), "nan": float("nan")},
                )
            except Exception:
                continue

        if isinstance(obj, dict) and ("replay_source" in obj or "gpu_time_sec" in obj):
            return obj

    debug_tail = "\n".join(stdout.splitlines()[-30:])
    raise RuntimeError(
        "Could not parse summary dict from run_prog_hyb.py output. "
        f"Last dict-like candidate was: {last_candidate!r}\n"
        f"--- stdout tail ---\n{debug_tail}"
    )


def run_one(
    *,
    python_bin: str,
    run_prog: str,
    dump: str,
    backend: str,
    validate_batches: int,
    extra_env: Optional[Dict[str, str]] = None,
    print_stdout: bool = False,
) -> Dict[str, Any]:
    if backend not in ("gpu", "cpu"):
        raise ValueError(f"backend must be gpu/cpu, got {backend!r}")

    cmd = [
        python_bin,
        run_prog,
        "--dump",
        dump,
        "--validate_projected_batches",
        str(validate_batches if backend == "gpu" else 0),
    ]
    if backend == "gpu":
        cmd.append("--force_gpu_backend")
    else:
        cmd.append("--force_cpu_backend")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(REPO_ROOT),
        check=False,
    )
    if print_stdout or proc.returncode != 0:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with return code {proc.returncode}: {' '.join(cmd)}")

    summary = parse_summary_from_stdout(proc.stdout)
    summary["_backend_requested"] = backend
    return summary


def fmt(x: Any) -> str:
    if isinstance(x, float):
        if x == float("inf"):
            return "inf"
        return f"{x:.6g}"
    return str(x)


def median_or_none(vals: List[float]) -> Optional[float]:
    return statistics.median(vals) if vals else None


def summarize(label: str, rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    print(f"\n[{label}] runs={len(rows)}")
    if not rows:
        return {}

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for key in TIMING_KEYS:
        vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
        if not vals:
            continue
        out[key] = {
            "median": median_or_none(vals),
            "min": min(vals),
            "max": max(vals),
        }

    print("key                         median        min           max")
    print("-" * 66)
    for key in TIMING_KEYS:
        if key not in out:
            continue
        v = out[key]
        print(f"{key:<27} {fmt(v['median']):>12} {fmt(v['min']):>12} {fmt(v['max']):>12}")
    return out


def print_per_run(label: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = [
        "gpu_time_sec" if label == "GPU" else "cpu_time_sec",
        "batch_walltime_total_sec",
        "host_build_T_sec",
        "gemm_sec",
        "nearest_plane_sec",
        "np_panel_sec",
        "gpu_throughput" if label == "GPU" else "cpu_throughput",
    ]
    print(f"\n[{label}] per-run key fields")
    print("run " + " ".join(f"{k:>22}" for k in keys))
    for i, row in enumerate(rows, 1):
        print(f"{i:>3} " + " ".join(f"{fmt(row.get(k, '')):>22}" for k in keys))


def main() -> int:
    ap = argparse.ArgumentParser(description="Repeat replay benchmark for GPU/CPU projected backend.")
    ap.add_argument("--dump", default="runs/dumps/guess6/strict_L1_inst2048_guess6_jobs262144_v2.pkl")
    ap.add_argument("--run-prog", default=str(REPO_ROOT / "pipelines" / "replay_batch_candidates.py"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--gpu-runs", type=int, default=5)
    ap.add_argument("--cpu-runs", type=int, default=3)
    ap.add_argument("--gpu-warmup", type=int, default=1)
    ap.add_argument("--validate-batches", type=int, default=0, help="GPU validation batches for the first measured GPU run only.")
    ap.add_argument("--print-stdout", action="store_true")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--skip-cpu", action="store_true")
    args = ap.parse_args()

    dump_path = (SUITE_ROOT / args.dump).resolve() if not Path(args.dump).is_absolute() else Path(args.dump)
    if not dump_path.exists():
        raise FileNotFoundError(f"dump not found: {dump_path}")

    gpu_rows: List[Dict[str, Any]] = []
    cpu_rows: List[Dict[str, Any]] = []

    if not args.skip_gpu:
        for i in range(args.gpu_warmup):
            print(f"[warmup GPU] {i + 1}/{args.gpu_warmup}")
            run_one(
                python_bin=args.python,
                run_prog=args.run_prog,
                dump=str(dump_path),
                backend="gpu",
                validate_batches=0,
                print_stdout=args.print_stdout,
            )

        for i in range(args.gpu_runs):
            vb = args.validate_batches if i == 0 else 0
            print(f"[measure GPU] {i + 1}/{args.gpu_runs} validate_batches={vb}")
            row = run_one(
                python_bin=args.python,
                run_prog=args.run_prog,
                dump=str(dump_path),
                backend="gpu",
                validate_batches=vb,
                print_stdout=args.print_stdout,
            )
            gpu_rows.append(row)

    if not args.skip_cpu:
        for i in range(args.cpu_runs):
            print(f"[measure CPU] {i + 1}/{args.cpu_runs}")
            row = run_one(
                python_bin=args.python,
                run_prog=args.run_prog,
                dump=str(dump_path),
                backend="cpu",
                validate_batches=0,
                print_stdout=args.print_stdout,
            )
            cpu_rows.append(row)

    print_per_run("GPU", gpu_rows)
    print_per_run("CPU", cpu_rows)
    gpu_summary = summarize("GPU", gpu_rows)
    cpu_summary = summarize("CPU", cpu_rows)

    if gpu_summary and cpu_summary:
        gtp = gpu_summary.get("gpu_throughput", {}).get("median")
        ctp = cpu_summary.get("cpu_throughput", {}).get("median")
        gt = gpu_summary.get("gpu_time_sec", {}).get("median")
        ct = cpu_summary.get("cpu_time_sec", {}).get("median")
        print("\n[GPU vs CPU median]")
        if gtp and ctp and ctp > 0:
            print(f"throughput_ratio_gpu_over_cpu = {gtp / ctp:.6f}")
        if gt and ct and ct > 0:
            print(f"time_ratio_gpu_over_cpu = {gt / ct:.6f}")

    result = {
        "gpu_rows": gpu_rows,
        "cpu_rows": cpu_rows,
        "gpu_summary": gpu_summary,
        "cpu_summary": cpu_summary,
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"\nJSON saved to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
