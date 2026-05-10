import warnings

warnings.filterwarnings("ignore", message=".*Dimension of lattice is larger than.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated as an API..*")

import argparse
import os
import pickle
import sys
from copy import copy
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_consts import *

try:
    from multiprocess import Pool
except ModuleNotFoundError:
    from multiprocessing import Pool

out_path = "lwe_instances/reduced_lattices/"


def _load_attack_runtime_deps():
    import time
    from time import perf_counter
    import numpy as np
    from fpylll import FPLLL, IntegerMatrix
    from fpylll.util import gaussian_heuristic
    from g6k.siever import Siever
    from g6k.siever_params import SieverParams

    from utils import test_vect_proj, from_canonical_scaled
    from sample import centeredBinomial, ternaryDist, Distribution
    from sparse_dist import sparse_distribution
    from preprocessing import load_lwe
    from hybrid_attack import alg_3_debug_v2

    FPLLL.set_random_seed(0x1337)
    FPLLL.set_precision(210)
    return {
        "time": time,
        "perf_counter": perf_counter,
        "np": np,
        "IntegerMatrix": IntegerMatrix,
        "gaussian_heuristic": gaussian_heuristic,
        "Siever": Siever,
        "SieverParams": SieverParams,
        "test_vect_proj": test_vect_proj,
        "from_canonical_scaled": from_canonical_scaled,
        "centeredBinomial": centeredBinomial,
        "ternaryDist": ternaryDist,
        "Distribution": Distribution,
        "sparse_distribution": sparse_distribution,
        "load_lwe": load_lwe,
        "alg_3_debug_v2": alg_3_debug_v2,
    }


def _apply_backend_override(force_cpu_backend: bool, force_gpu_backend: bool):
    import hyb_batch.replay as replay_mod

    if force_cpu_backend and force_gpu_backend:
        raise ValueError("--force_cpu_backend and --force_gpu_backend cannot be used together")
    if force_cpu_backend:
        replay_mod.HYB_GPU_BACKEND_ENABLE = False
    if force_gpu_backend:
        replay_mod.HYB_GPU_BACKEND_ENABLE = True


def run_experiment(lat_index, params, stats_dict, delta_slicer_coord=0):
    deps = _load_attack_runtime_deps()
    time = deps["time"]
    perf_counter = deps["perf_counter"]
    np = deps["np"]
    IntegerMatrix = deps["IntegerMatrix"]
    gaussian_heuristic = deps["gaussian_heuristic"]
    Siever = deps["Siever"]
    SieverParams = deps["SieverParams"]
    test_vect_proj = deps["test_vect_proj"]
    from_canonical_scaled = deps["from_canonical_scaled"]
    centeredBinomial = deps["centeredBinomial"]
    ternaryDist = deps["ternaryDist"]
    Distribution = deps["Distribution"]
    sparse_distribution = deps["sparse_distribution"]
    load_lwe = deps["load_lwe"]
    alg_3_debug_v2 = deps["alg_3_debug_v2"]

    nthreads = params["nthreads"]
    n, q, dist, dist_param = params["n"], params["q"], params["dist"], params["dist_param"]
    n_guess_coord, n_slicer_coord = params["n_guess_coord"], params["n_slicer_coord"]
    beta_pre = params["beta_pre"]
    verbose = params["verbose"]

    A, q, bse = load_lwe(params)

    Binit = [[int(0) for _ in range(2 * n)] for _ in range(2 * n)]
    for i in range(n):
        Binit[i][i] = int(q)
    for i in range(n, 2 * n):
        Binit[i][i] = 1
    for i in range(n, 2 * n):
        for j in range(n):
            Binit[i][j] = int(A[i - n, j])

    filename_g6kdump = f"g6kdump_{n}_{q}_{dist}_{dist_param:.04f}_{lat_index}_{n_guess_coord}_{n_slicer_coord}_{beta_pre}.pkl"
    g6k = Siever.restore_from_file(out_path + filename_g6kdump)

    param_sieve = SieverParams()
    param_sieve["threads"] = nthreads
    param_sieve["otf_lift"] = False
    param_sieve["saturation_ratio"] = 0.7
    g6k.params = param_sieve

    G = g6k.M
    G.update_gso()

    match dist:
        case "binomial":
            dist_param = int(dist_param)
            distrib = centeredBinomial(dist_param)
        case "ternary":
            distrib = ternaryDist(dist_param)
        case "ternary_sparse":
            distrib = sparse_distribution(n, n_guess_coord, int(dist_param), Distribution({-1: 0.5, 1: 0.5}))
        case _:
            raise NotImplementedError(f"Bad distribution")

    chosen_delta = n_slicer_coord
    for delta in range(n_slicer_coord, n_slicer_coord + delta_slicer_coord + 1):
        lens = test_vect_proj(G, delta, NPROJ_TESTS, distrib)
        est_norm = np.percentile(lens, 50)
        if est_norm <= HYB_PROJ_THRESHOLD:
            chosen_delta = delta
            break

    overhead_tsieve = time.perf_counter()
    assert chosen_delta <= G.d, f"Too many slicer coords: {chosen_delta}>{G.d}"
    G = g6k.M
    g6k = Siever(G, param_sieve)
    g6k.initialize_local(g6k.M.d - chosen_delta, max(g6k.M.d - chosen_delta, g6k.M.d - 50), g6k.M.d)
    print("Running bdgl2...")
    then = time.perf_counter()
    g6k(alg="bdgl2")
    while g6k.ll < g6k.l:
        g6k.extend_left()
        g6k(alg="bdgl2")
    print(f"pump done in {time.perf_counter() - then}")

    H11 = g6k.M.B
    overhead_tsieve = time.perf_counter() - overhead_tsieve
    n_slicer_coord = chosen_delta
    gh_sub = gaussian_heuristic(G.r()[-n_slicer_coord:])
    print(f"Sieving-1 done in {perf_counter() - then}")

    ex_cntr = 0
    for (b, s, e) in bse:
        ex_cntr += 1
        print(f"running exp # {ex_cntr}")
        ex_timer = perf_counter()
        assert all((s @ A + e) % q == b), f"wrong lwe instance! {((A @ s + e) % q, b)}"
        answer = np.concatenate([b - e, s])
        t = np.concatenate([b, n * [0]])
        e_ = np.concatenate([e, -s])[:-n_guess_coord]
        e_ = from_canonical_scaled(G, e_, offset=n_slicer_coord, scale_fact=gh_sub)
        dist_sq_bnd = e_ @ e_
        dist_bnd = dist_sq_bnd ** 0.5
        B = IntegerMatrix.from_matrix(Binit)
        tracer = {}
        iter_v = alg_3_debug_v2(
            g6k, H11, B, t, n_guess_coord, dist, dist_param, s,
            dist_sq_bnd=EPS2 * dist_sq_bnd, nthreads=nthreads,
            tracer_alg3=tracer, verbose=verbose,
        )
        guess_cntr = 0
        sli_succ = False
        v2 = None
        for v in iter_v:
            if v is None:
                v = np.array(len(answer) * [0])
            guess_cntr += 1
            v2 = v
            sli_succ = all(answer == v2)
            if sli_succ:
                print("Success in experiment!")
                break
        if not sli_succ:
            print(f"Fail @{(lat_index, ex_cntr)}")
        fail_reason = "other" if guess_cntr < 1 else "parasites"
        walltime = tracer["wrong_guess_time_alg3"] + tracer["wrong_guess_time_alg2"]
        walltime_observed = perf_counter() - ex_timer
        stats_dict[(n, lat_index, n_slicer_coord, n_guess_coord, ex_cntr)] = {
            "walltime": walltime,
            "dist_bnd": dist_bnd,
            "succ": sli_succ,
            "fail_reason": None if sli_succ else fail_reason,
            "key_num": tracer["key_num"],
            "g6k_len": len(g6k),
            "g6k_dim": g6k.r - g6k.l,
            "wrong_guess_time_alg3": tracer["wrong_guess_time_alg3"],
            "correct_guess_time_alg3": tracer["correct_guess_time_alg3"],
            "wrong_guess_time_alg2": tracer["wrong_guess_time_alg2"],
            "correct_guess_time_alg2": tracer["correct_guess_time_alg2"],
            "walltime_observed": walltime_observed,
            "overhead_tsieve": overhead_tsieve,
        }
    return stats_dict


def get_parser():
    parser = argparse.ArgumentParser(description="Hybrid attack / batch dump / replay compatibility entrypoint.")
    parser.add_argument("--nthreads", default=N_SIEVE_THREADS, type=int, help="Threads per slicer.")
    parser.add_argument("--nworkers", default=1, type=int, help="Workers for experiments.")
    parser.add_argument("--lats_per_dim", default=1, type=int, help="Number of lattices.")
    parser.add_argument("--n", default=125, type=int, help="LWE dimension")
    parser.add_argument("--q", default=3329, type=int, help="LWE modulus")
    parser.add_argument("--dist", default="binomial", type=str, help="LWE distribution")
    parser.add_argument("--dist_param", default=2.0, type=float, help="LWE distribution's parameter (as float)")
    parser.add_argument("--beta_pre", default=46, type=int, help="BKZ blocksize.")
    parser.add_argument("--n_guess_coord", default=2, type=int, help="Number of guessing coordinates")
    parser.add_argument("--n_slicer_coord", default=47, type=int, help="Minimal dimension of slicer.")
    parser.add_argument("--delta_slicer_coord", default=3, type=int, help="Maximal dimension of slicer will be n_slicer_coord+delta_slicer_coord.")
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("--batch_prototype", action="store_true", help="Run cross-instance batch prototype")
    parser.add_argument("--dump_batch_candidates", action="store_true", help="Generate and dump batch candidates to a pickle file, without projected scoring.")
    parser.add_argument("--replay_batch_candidates", "--dump", type=str, default=None, help="Replay a dumped batch-candidates pickle and run projected backend on it.")
    parser.add_argument("--batch_dump_path", type=str, default=None, help="Explicit output path for dumped batch candidates.")
    parser.add_argument("--validate_projected_batches", type=int, default=int(os.environ.get("HYB_VALIDATE_PROJECTED_BATCHES", "0")))
    parser.add_argument("--validate_projected_atol", type=float, default=float(os.environ.get("HYB_VALIDATE_PROJECTED_ATOL", "1e-8")))
    parser.add_argument("--validate_projected_rtol", type=float, default=float(os.environ.get("HYB_VALIDATE_PROJECTED_RTOL", "1e-8")))
    parser.add_argument("--force_cpu_backend", action="store_true")
    parser.add_argument("--force_gpu_backend", action="store_true")
    parser.add_argument("--json-out", type=str, default=None, help="Write dump/replay compact JSON summary to this path.")
    return parser


def main() -> int:
    args = get_parser().parse_args()

    if args.dump_batch_candidates or args.replay_batch_candidates is not None or args.batch_prototype:
        from hyb_batch import compact_summary_for_print
        from hyb_batch.config import HybridBaseParams, BatchRuntimeConfig, DumpBuildConfig, ReplayConfig, write_json_summary
        from hyb_batch.dump import build_batch_candidate_dump
        from hyb_batch.replay import replay_batch_candidate_dump, run_experiment_batch

        _apply_backend_override(args.force_cpu_backend, args.force_gpu_backend)
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
        runtime = BatchRuntimeConfig()
        if args.force_cpu_backend:
            runtime.gpu_backend_enable = False
        if args.force_gpu_backend:
            runtime.gpu_backend_enable = True

        if args.dump_batch_candidates:
            dump = build_batch_candidate_dump(DumpBuildConfig(base=base, delta_slicer_coord=args.delta_slicer_coord, json_out=args.json_out))
            dump_path = args.batch_dump_path or f"batch_candidates_{args.n}_{args.dist}_{args.dist_param:0.4f}_{args.n_guess_coord}_{args.beta_pre}_{args.n_slicer_coord + args.delta_slicer_coord}.pkl"
            with open(dump_path, "wb") as f:
                pickle.dump(dump, f)
            if args.json_out:
                write_json_summary({"mode": "dump", "output": dump_path, "dump_version": dump.get("dump_version")}, args.json_out)
            print(f"batch candidates dumped to: {dump_path}")
            return 0

        if args.replay_batch_candidates is not None:
            summary = replay_batch_candidate_dump(ReplayConfig(
                dump_path=args.replay_batch_candidates,
                runtime=runtime,
                validate_projected_batches=args.validate_projected_batches,
                validate_atol=args.validate_projected_atol,
                validate_rtol=args.validate_projected_rtol,
                json_out=args.json_out,
            ))
            print(compact_summary_for_print(summary))
            return 0

        summary = run_experiment_batch(base, delta_slicer_coord=args.delta_slicer_coord, runtime=runtime)
        if args.json_out:
            summary["json_summary_path"] = write_json_summary(summary, args.json_out)
        print(compact_summary_for_print(summary))
        return 0

    params = {
        "nthreads": args.nthreads,
        "n": args.n,
        "dist": args.dist,
        "dist_param": args.dist_param,
        "q": args.q,
        "n_guess_coord": args.n_guess_coord,
        "n_slicer_coord": args.n_slicer_coord,
        "beta_pre": args.beta_pre,
        "verbose": args.verbose,
        "lats_per_dim": args.lats_per_dim,
    }
    output = []
    pool = Pool(processes=args.nworkers)
    tasks = []
    for lat_index in range(args.lats_per_dim):
        output.append({})
        params["seed"] = (lat_index, 0)
        tasks.append(pool.apply_async(run_experiment, (lat_index, copy(params), output[lat_index], args.delta_slicer_coord)))

    stats_dict_agr = {}
    for t in tasks:
        stats_dict_agr.update(t.get())
    print(stats_dict_agr)
    filename = f"tph_{args.n}_{args.dist}_{args.dist_param:0.4f}_{args.n_guess_coord}_{args.beta_pre}_{args.n_slicer_coord + args.delta_slicer_coord}.pkl"
    with open(filename, "wb") as file:
        pickle.dump(stats_dict_agr, file)
    print("Results dumped to " + filename)
    pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
