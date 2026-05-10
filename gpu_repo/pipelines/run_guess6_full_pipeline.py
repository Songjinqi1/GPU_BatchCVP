#!/usr/bin/env python3
"""
One-shot runner for the guess=6 strict_L1 inst_per_lat=2048 projected-batch benchmark.

Default steps:
  1) preprocessing.py
  2) run_prog_hyb.py --dump_batch_candidates
  3) GPU replay, optionally validating the first N batches against CPU
  4) CPU replay

This script assumes you have replaced run_prog_hyb.py with the validated version
that supports --force_gpu_backend, --force_cpu_backend, and
--validate_projected_batches.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = REPO_ROOT.parent


def run_and_tee(cmd: list[str], log_path: Optional[Path] = None, env: Optional[dict[str, str]] = None, cwd: Optional[Path] = None) -> None:
    print("\n" + "=" * 100)
    print("[run]", " ".join(cmd))
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[log] {log_path}")
    print("=" * 100, flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(cwd) if cwd else None,
    )

    assert proc.stdout is not None
    f = None
    try:
        if log_path is not None:
            f = log_path.open("w", encoding="utf-8")
        for line in proc.stdout:
            print(line, end="")
            if f is not None:
                f.write(line)
                f.flush()
    finally:
        if f is not None:
            f.close()

    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run preprocessing, v2 dump generation, GPU replay, and CPU replay for the guess=6 benchmark."
    )

    p.add_argument("--python", default=sys.executable, help="Python executable to use.")

    p.add_argument("--n", type=int, default=140)
    p.add_argument("--q", type=int, default=3329)
    p.add_argument("--dist", default="binomial")
    p.add_argument("--dist_param", type=float, default=3)
    p.add_argument("--nthreads", type=int, default=5)
    p.add_argument("--lats_per_dim", type=int, default=1)
    p.add_argument("--inst_per_lat", type=int, default=2048)

    p.add_argument("--n_guess_coord", type=int, default=6)
    p.add_argument("--beta_pre", type=int, default=46)
    p.add_argument("--n_slicer_coord", type=int, default=47)
    p.add_argument("--delta_slicer_coord", type=int, default=0)

    p.add_argument(
        "--params",
        default="[(140,6,46)]",
        help='Value passed to preprocessing.py --params, for example "[(140,6,46)]".',
    )

    p.add_argument(
        "--dump_path",
        default="results/dumps/guess6/strict_L1_inst2048_guess6_jobs262144_v2.pkl",
        help="Path for the generated/replayed v2 dump.",
    )
    p.add_argument(
        "--gpu_log",
        default="results/logs/guess6/strict_L1_inst2048_guess6_jobs262144_v2_gpu.txt",
        help="GPU replay log path.",
    )
    p.add_argument(
        "--cpu_log",
        default="results/logs/guess6_cpu/strict_L1_inst2048_guess6_jobs262144_v2_cpu.txt",
        help="CPU replay log path.",
    )
    p.add_argument(
        "--preprocess_log",
        default="results/logs/guess6/preprocess_strict_L1_inst2048_guess6.txt",
        help="Preprocessing log path.",
    )
    p.add_argument(
        "--dump_log",
        default="results/logs/guess6/dump_strict_L1_inst2048_guess6_jobs262144_v2.txt",
        help="Dump-generation log path.",
    )
    p.add_argument("--dump_json", default=None, help="Optional JSON summary path for dump step.")
    p.add_argument("--gpu_json", default=None, help="Optional JSON summary path for GPU replay.")
    p.add_argument("--cpu_json", default=None, help="Optional JSON summary path for CPU replay.")

    p.add_argument(
        "--validate_batches",
        type=int,
        default=1,
        help="Validate the first N GPU replay batches against CPU. Use 0 to disable.",
    )
    p.add_argument("--validate_atol", type=float, default=1e-8)
    p.add_argument("--validate_rtol", type=float, default=1e-8)

    p.add_argument("--skip_preprocess", action="store_true")
    p.add_argument("--skip_dump", action="store_true")
    p.add_argument("--skip_gpu", action="store_true")
    p.add_argument("--skip_cpu", action="store_true")

    return p


def main() -> None:
    args = get_parser().parse_args()

    dump_path = (SUITE_ROOT / args.dump_path).resolve() if not Path(args.dump_path).is_absolute() else Path(args.dump_path)
    gpu_log = (SUITE_ROOT / args.gpu_log).resolve() if not Path(args.gpu_log).is_absolute() else Path(args.gpu_log)
    cpu_log = (SUITE_ROOT / args.cpu_log).resolve() if not Path(args.cpu_log).is_absolute() else Path(args.cpu_log)
    preprocess_log = (SUITE_ROOT / args.preprocess_log).resolve() if not Path(args.preprocess_log).is_absolute() else Path(args.preprocess_log)
    dump_log = (SUITE_ROOT / args.dump_log).resolve() if not Path(args.dump_log).is_absolute() else Path(args.dump_log)

    dump_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_log.parent.mkdir(parents=True, exist_ok=True)
    cpu_log.parent.mkdir(parents=True, exist_ok=True)

    py = args.python

    if not args.skip_preprocess:
        preprocess_cmd = [
            py,
            str(REPO_ROOT / "preprocessing.py"),
            "--q", str(args.q),
            "--dist", str(args.dist),
            "--dist_param", str(args.dist_param),
            "--nthreads", str(args.nthreads),
            "--lats_per_dim", str(args.lats_per_dim),
            "--inst_per_lat", str(args.inst_per_lat),
            "--params", str(args.params),
            "--recompute_instance",
        ]
        run_and_tee(preprocess_cmd, preprocess_log, cwd=REPO_ROOT)

    if not args.skip_dump:
        dump_cmd = [
            py,
            str(REPO_ROOT / "pipelines" / "dump_batch_candidates.py"),
            "--n", str(args.n),
            "--q", str(args.q),
            "--dist", str(args.dist),
            "--dist_param", str(args.dist_param),
            "--nthreads", str(args.nthreads),
            "--n_guess_coord", str(args.n_guess_coord),
            "--beta_pre", str(args.beta_pre),
            "--n_slicer_coord", str(args.n_slicer_coord),
            "--delta_slicer_coord", str(args.delta_slicer_coord),
            "--lats_per_dim", str(args.lats_per_dim),
            "--output", str(dump_path),
        ]
        if args.dump_json:
            dump_cmd += ["--json-out", str((SUITE_ROOT / args.dump_json).resolve() if not Path(args.dump_json).is_absolute() else Path(args.dump_json))]
        run_and_tee(dump_cmd, dump_log, cwd=REPO_ROOT)

    if not args.skip_gpu:
        gpu_cmd = [
            py,
            str(REPO_ROOT / "pipelines" / "replay_batch_candidates.py"),
            "--dump", str(dump_path),
            "--force_gpu_backend",
        ]
        if args.gpu_json:
            gpu_cmd += ["--json-out", str((SUITE_ROOT / args.gpu_json).resolve() if not Path(args.gpu_json).is_absolute() else Path(args.gpu_json))]
        if args.validate_batches > 0:
            gpu_cmd += [
                "--validate_projected_batches", str(args.validate_batches),
                "--validate_projected_atol", str(args.validate_atol),
                "--validate_projected_rtol", str(args.validate_rtol),
            ]
        run_and_tee(gpu_cmd, gpu_log, cwd=REPO_ROOT)

    if not args.skip_cpu:
        cpu_cmd = [
            py,
            str(REPO_ROOT / "pipelines" / "replay_batch_candidates.py"),
            "--dump", str(dump_path),
            "--force_cpu_backend",
        ]
        if args.cpu_json:
            cpu_cmd += ["--json-out", str((SUITE_ROOT / args.cpu_json).resolve() if not Path(args.cpu_json).is_absolute() else Path(args.cpu_json))]
        run_and_tee(cpu_cmd, cpu_log, cwd=REPO_ROOT)

    print("\n[done] Full pipeline completed.")
    print(f"[dump] {dump_path}")
    if not args.skip_gpu:
        print(f"[gpu log] {gpu_log}")
    if not args.skip_cpu:
        print(f"[cpu log] {cpu_log}")


if __name__ == "__main__":
    main()
