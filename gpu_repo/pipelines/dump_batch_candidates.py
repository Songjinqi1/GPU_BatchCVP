#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pickle


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build and dump batch candidates for projected backend replay.")
    p.add_argument("--nthreads", type=int, required=True)
    p.add_argument("--lats_per_dim", type=int, default=1)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--q", type=int, required=True)
    p.add_argument("--dist", type=str, required=True)
    p.add_argument("--dist_param", type=float, required=True)
    p.add_argument("--beta_pre", type=int, required=True)
    p.add_argument("--n_guess_coord", type=int, required=True)
    p.add_argument("--n_slicer_coord", type=int, required=True)
    p.add_argument("--delta_slicer_coord", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--json-out", default=None)
    return p


def main() -> int:
    args = get_parser().parse_args()
    from hyb_batch.dump import build_batch_candidate_dump
    from hyb_batch.config import HybridBaseParams, DumpBuildConfig, write_json_summary

    base = HybridBaseParams(
        nthreads=args.nthreads,
        n=args.n,
        q=args.q,
        dist=args.dist,
        dist_param=args.dist_param,
        beta_pre=args.beta_pre,
        n_guess_coord=args.n_guess_coord,
        n_slicer_coord=args.n_slicer_coord,
        lats_per_dim=args.lats_per_dim,
        verbose=args.verbose,
    )
    cfg = DumpBuildConfig(base=base, delta_slicer_coord=args.delta_slicer_coord, json_out=args.json_out)
    dump = build_batch_candidate_dump(cfg)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(dump, f)
    if args.json_out:
        write_json_summary({
            "mode": "dump",
            "output": str(out),
            "dump_version": dump.get("dump_version"),
            "all_job_ids_count": len(dump.get("all_job_ids", [])),
            "config": cfg.to_dict(),
        }, args.json_out)
    print(f"batch candidates dumped to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
