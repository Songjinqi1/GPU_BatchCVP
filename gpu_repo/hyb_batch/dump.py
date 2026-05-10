from __future__ import annotations

from dataclasses import asdict
import numpy as np
from fpylll import IntegerMatrix, GSO
from fpylll.util import gaussian_heuristic
from g6k.siever import Siever
from g6k.siever_params import SieverParams

from utils import *
from sample import *
from sparse_dist import sparse_distribution
from preprocessing import load_lwe
from batch_scheduler import BatchScheduler
from hybrid_attack import build_target_candidates_debug_v2
from projection_cache import ProjectionCache, integer_matrix_to_numpy, build_projected_block
from candidate_enumerator import InstanceEnumerationConfig, stream_instance_candidates
from global_consts import *

from .common import (
    compute_target_hash,
    build_all_targets_matrix_from_scheduler,
    serialize_projection_cache,
)
from .config import DumpBuildConfig, HybridBaseParams

inp_path = "lwe_instances/saved_lattices/"
out_path = "lwe_instances/reduced_lattices/"


def prepare_batch_material(params, delta_slicer_coord=0):
    if isinstance(params, HybridBaseParams):
        params = params.to_legacy_params()
    nthreads = params["nthreads"]
    n, q, dist, dist_param = params["n"], params["q"], params["dist"], params["dist_param"]
    latnum = params["lats_per_dim"]
    n_guess_coord, n_slicer_coord = params["n_guess_coord"], params["n_slicer_coord"]

    scheduler = BatchScheduler(max_targets=HYB_BATCH_MAX_TARGETS)
    projection_cache = ProjectionCache()
    per_instance_info = []

    for lat_index in range(latnum):
        params_local = dict(params)
        params_local["seed"] = (lat_index, 0)

        A, q, bse = load_lwe(params_local)

        Binit = [[int(0) for _ in range(2 * n)] for _ in range(2 * n)]
        for i in range(n):
            Binit[i][i] = int(q)
        for i in range(n, 2 * n):
            Binit[i][i] = 1
        for i in range(n, 2 * n):
            for j in range(n):
                Binit[i][j] = int(A[i - n, j])

        filename_g6kdump = (
            f"g6kdump_{n}_{q}_{dist}_{dist_param:.04f}_{lat_index}_"
            f"{n_guess_coord}_{n_slicer_coord}_{params['beta_pre']}.pkl"
        )
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
                dist_param_local = int(dist_param)
                distrib = centeredBinomial(dist_param_local)
            case "ternary":
                distrib = ternaryDist(dist_param)
            case "ternary_sparse":
                distrib = sparse_distribution(
                    n, n_guess_coord, int(dist_param), Distribution({-1: 0.5, 1: 0.5})
                )
            case _:
                raise NotImplementedError("Bad distribution")

        chosen_delta = n_slicer_coord
        for delta in range(n_slicer_coord, n_slicer_coord + delta_slicer_coord + 1):
            lens = test_vect_proj(G, delta, NPROJ_TESTS, distrib)
            est_norm = np.percentile(lens, 50)
            if est_norm <= HYB_PROJ_THRESHOLD:
                chosen_delta = delta
                break

        g6k = Siever(G, param_sieve)
        g6k.initialize_local(
            g6k.M.d - chosen_delta,
            max(g6k.M.d - chosen_delta, g6k.M.d - 50),
            g6k.M.d,
        )
        g6k(alg="bdgl2")
        while g6k.ll < g6k.l:
            g6k.extend_left()
            g6k(alg="bdgl2")

        cache_key = (lat_index, chosen_delta)
        if not projection_cache.has(cache_key):
            basis_np = integer_matrix_to_numpy(g6k.M.B)
            ell = chosen_delta
            Q2T, R22 = build_projected_block(basis_np, ell)
            projection_cache.put(
                cache_key,
                {
                    "basis_np": basis_np,
                    "Q2T": Q2T,
                    "R22": R22,
                    "ell": ell,
                },
            )

        gh_sub = gaussian_heuristic(G.r()[-chosen_delta:])

        ex_cntr = 0
        for (b, s, e) in bse:
            ex_cntr += 1

            t = np.concatenate([b, n * [0]])
            e_ = np.concatenate([e, -s])[:-n_guess_coord]
            e_ = from_canonical_scaled(G, e_, offset=chosen_delta, scale_fact=gh_sub)
            dist_sq_bnd = e_ @ e_

            B = IntegerMatrix.from_matrix(Binit)

            if not HYB_ENUMERATOR_ENABLE:
                tracer = {}
                cand_info = build_target_candidates_debug_v2(
                    B=B,
                    target=t,
                    n_guess_coord=n_guess_coord,
                    dist=dist,
                    dist_param=dist_param,
                    s=s,
                    g6k_len=len(g6k),
                    tracer=tracer,
                )

                for i, tc in enumerate(cand_info["wrong_candidates"]):
                    tc = np.ascontiguousarray(np.asarray(tc, dtype=np.float64).reshape(-1))
                    scheduler.add_job(
                        lat_index=lat_index,
                        exp_index=ex_cntr,
                        branch="wrong",
                        candidate_index=i,
                        target=tc,
                        dist_sq_bnd=dist_sq_bnd,
                        meta={"chosen_delta": chosen_delta},
                        target_hash=compute_target_hash(tc),
                        target_dim=len(tc),
                    )

                for i, tc in enumerate(cand_info["correct_candidates"]):
                    tc = np.ascontiguousarray(np.asarray(tc, dtype=np.float64).reshape(-1))
                    scheduler.add_job(
                        lat_index=lat_index,
                        exp_index=ex_cntr,
                        branch="correct",
                        candidate_index=i,
                        target=tc,
                        dist_sq_bnd=dist_sq_bnd,
                        meta={"chosen_delta": chosen_delta},
                        target_hash=compute_target_hash(tc),
                        target_dim=len(tc),
                    )

                per_instance_info.append({
                    "lat_index": lat_index,
                    "exp_index": ex_cntr,
                    "g6k_dim": g6k.r - g6k.l,
                    "g6k_len": len(g6k),
                    "key_num": tracer.get("key_num"),
                    "wrong_candidates": len(cand_info["wrong_candidates"]),
                    "correct_candidates": len(cand_info["correct_candidates"]),
                    "chosen_delta": chosen_delta,
                    "enum_total_space": None,
                    "enum_chunk_size": None,
                })

            else:
                t1_np = np.asarray(t[:-n_guess_coord], dtype=np.int64)
                t2_np = np.asarray(t[-n_guess_coord:], dtype=np.int64)

                H12_rows = [
                    list(row)[: 2 * n - n_guess_coord]
                    for row in Binit[2 * n - n_guess_coord:]
                ]
                H12_np = np.asarray(H12_rows, dtype=np.int64).T

                true_etilde2 = np.array(-s[-n_guess_coord:], dtype=np.int64)

                enum_cfg = InstanceEnumerationConfig(
                    dist=dist,
                    dist_param=dist_param,
                    guess_dim=n_guess_coord,
                    wrong_target_count=HYB_WRONG_TARGET_COUNT,
                    correct_target_count=HYB_CORRECT_TARGET_COUNT,
                    chunk_size=HYB_ENUMERATOR_CHUNK_SIZE,
                    include_true_guess_in_correct=True,
                    exclude_true_guess_from_wrong=True,
                )

                wrong_count = 0
                correct_count = 0
                enum_total_space = None

                for chunk in stream_instance_candidates(
                    H12_np=H12_np,
                    t1_np=t1_np,
                    t2_np=t2_np,
                    true_etilde2=true_etilde2,
                    config=enum_cfg,
                ):
                    if enum_total_space is None:
                        enum_total_space = int(chunk.total_space)

                    chunk_targets = np.asarray(chunk.t1_chunk, dtype=np.float64)
                    for local_i in range(len(chunk_targets)):
                        tc = np.ascontiguousarray(
                            np.asarray(chunk_targets[local_i], dtype=np.float64).reshape(-1)
                        )

                        if chunk.branch == "wrong":
                            cand_idx = wrong_count
                            wrong_count += 1
                        else:
                            cand_idx = correct_count
                            correct_count += 1

                        scheduler.add_job(
                            lat_index=lat_index,
                            exp_index=ex_cntr,
                            branch=chunk.branch,
                            candidate_index=cand_idx,
                            target=tc,
                            dist_sq_bnd=dist_sq_bnd,
                            meta={
                                "chosen_delta": chosen_delta,
                                "enum_index_start": chunk.index_start,
                                "enum_total_space": chunk.total_space,
                            },
                            target_hash=compute_target_hash(tc),
                            target_dim=len(tc),
                        )

                per_instance_info.append({
                    "lat_index": lat_index,
                    "exp_index": ex_cntr,
                    "g6k_dim": g6k.r - g6k.l,
                    "g6k_len": len(g6k),
                    "key_num": enum_total_space,
                    "wrong_candidates": wrong_count,
                    "correct_candidates": correct_count,
                    "chosen_delta": chosen_delta,
                    "enum_total_space": enum_total_space,
                    "enum_chunk_size": HYB_ENUMERATOR_CHUNK_SIZE,
                })

    return scheduler, projection_cache, per_instance_info


def build_batch_candidate_dump(params, delta_slicer_coord=0):
    if isinstance(params, DumpBuildConfig):
        delta_slicer_coord = params.delta_slicer_coord
        params = params.base
    scheduler, projection_cache, per_instance_info = prepare_batch_material(
        params=params,
        delta_slicer_coord=delta_slicer_coord,
    )

    all_targets_matrix, all_job_ids = build_all_targets_matrix_from_scheduler(scheduler)
    all_targets_matrix_rowmajor = np.ascontiguousarray(all_targets_matrix.T, dtype=np.float64)

    return {
        "dump_version": 3,
        "params": asdict(params),
        "delta_slicer_coord": int(delta_slicer_coord),
        "scheduler": scheduler.to_serializable(),
        "projection_cache": serialize_projection_cache(projection_cache),
        "per_instance_info": per_instance_info,
        "all_targets_matrix": all_targets_matrix,
        "all_targets_matrix_rowmajor": all_targets_matrix_rowmajor,
        "all_job_ids": all_job_ids,
    }
