#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))




def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay dumped batch candidates with CPU or GPU projected backend.")
    p.add_argument("--dump", required=True)
    p.add_argument("--validate_projected_batches", type=int, default=0)
    p.add_argument("--validate_projected_atol", type=float, default=1e-8)
    p.add_argument("--validate_projected_rtol", type=float, default=1e-8)
    p.add_argument("--json-out", default=None)
    p.add_argument("--force_cpu_backend", action="store_true")
    p.add_argument("--force_gpu_backend", action="store_true")
    return p


def main() -> int:
    args = get_parser().parse_args()
    from hyb_batch.common import compact_summary_for_print
    import hyb_batch.replay as replay_mod
    from hyb_batch.replay import replay_batch_candidate_dump
    from hyb_batch.config import BatchRuntimeConfig, ReplayConfig

    if args.force_cpu_backend and args.force_gpu_backend:
        raise ValueError("--force_cpu_backend and --force_gpu_backend cannot be used together")
    if args.force_cpu_backend:
        replay_mod.HYB_GPU_BACKEND_ENABLE = False
    if args.force_gpu_backend:
        replay_mod.HYB_GPU_BACKEND_ENABLE = True

    runtime = BatchRuntimeConfig(gpu_backend_enable=replay_mod.HYB_GPU_BACKEND_ENABLE)
    cfg = ReplayConfig(
        dump_path=args.dump,
        runtime=runtime,
        validate_projected_batches=args.validate_projected_batches,
        validate_atol=args.validate_projected_atol,
        validate_rtol=args.validate_projected_rtol,
        json_out=args.json_out,
    )
    summary = replay_batch_candidate_dump(cfg)
    print(compact_summary_for_print(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
