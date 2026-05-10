from experiments.lwe_gen import *

import sys, os
import time
from time import perf_counter
from math import ceil
from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *
from global_consts import *
from sparse_dist import sparse_distribution

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

import pickle
from sample import *

from preprocessing import load_lwe
from hybrid_estimator.batchCVP import batchCVPP_cost

approx_fact = 1.0001
max_nsampl = 2**31 - 1
inp_path = "lwe_instances/saved_lattices/"
out_path = "lwe_instances/reduced_lattices/"


def alg_2_batched(g6k, target_candidates, dist_sq_bnd=1.0, nthreads=N_SIEVE_THREADS, tracer_alg2=None):
    if tracer_alg2 is not None:
        startt = time.perf_counter()
        tracer_alg2["walltime"] = 0

    sieve_dim = g6k.r - g6k.l
    G = g6k.M
    gh_sub = gaussian_heuristic(G.r()[-sieve_dim:])
    B = G.B
    dim = G.d
    Gsub = GSO.Mat(G.B[:dim - sieve_dim], float_type=G.float_type)
    Gsub.update_gso()

    slicer = RandomizedSlicer(g6k)
    slicer.set_nthreads(nthreads)
    slicer.set_max_slicer_interations(N_MAX_SLICER_ITERATIONS)
    slicer.set_proj_error_bound(EPS2 * dist_sq_bnd)
    slicer.set_Nt(len(target_candidates))
    slicer.set_saturation_scalar(SATURATION_SCALAR)

    nrand_, _ = batchCVPP_cost(sieve_dim, 1, len(g6k) ** (1.0 / sieve_dim), 1)
    nrand = ceil(NRAND_FACTOR * (1.0 / nrand_) ** sieve_dim)

    t_gs_list = []
    t_gs_reduced_list = []
    shift_babai_c_list = []

    for target in target_candidates:
        t_gs = from_canonical_scaled(G, target, offset=sieve_dim, scale_fact=gh_sub)

        t_gs_non_scaled = G.from_canonical(target)[dim - sieve_dim:]
        shift_babai_c = list(G.babai(list(t_gs_non_scaled), start=dim - sieve_dim, gso=True))
        shift_babai = G.B.multiply_left((dim - sieve_dim) * [0] + list(shift_babai_c))
        t_gs_reduced = from_canonical_scaled(
            G,
            np.array(target, dtype=DTYPE) - shift_babai,
            offset=sieve_dim,
            scale_fact=gh_sub,
        )

        t_gs_list.append(t_gs)
        shift_babai_c_list.append(shift_babai_c)
        t_gs_reduced_list.append(t_gs_reduced)
        slicer.grow_db_with_target(t_gs_reduced, n_per_target=nrand)

    blocks = 2
    blocks = min(3, max(1, blocks))
    blocks = min(int(sieve_dim / 28), blocks)
    sp = g6k.params
    N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
    buckets = (
        sp["bdgl_bucket_size_factor"]
        * 2.0 ** ((blocks - 1.0) / (blocks + 1.0))
        * sp["bdgl_multi_hash"] ** ((2.0 * blocks) / (blocks + 1.0))
        * (N ** (blocks / (1.0 + blocks)))
    )
    buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
    buckets = max(buckets, 2 ** (blocks - 1))

    slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], False)

    iterator = slicer.itervalues_cdb_t(return_with_index=True)
    best_bab_01 = np.array(g6k.M.d * [0])
    attemptcntr = 0

    for tmp, index in iterator:
        out_gs_reduced = np.array(tmp, dtype=DTYPE)
        if (out_gs_reduced @ out_gs_reduced) > 1.00001 * dist_sq_bnd:
            break
        attemptcntr += 1

        min_norm_err_sq = float("inf")
        out_reduced = np.array(
            to_canonical_scaled(G, out_gs_reduced, offset=sieve_dim, scale_fact=gh_sub),
            dtype=DTYPE,
        )
        out_reduced = G.to_canonical(
            (G.d - sieve_dim) * [0] + list(G.from_canonical(out_reduced, start=G.d - sieve_dim)),
            start=0,
        )

        assert index is not None, "Impossible!"
        t = np.array(target_candidates[index], dtype=DTYPE)
        bab_01 = np.array(G.babai(t - out_reduced))
        solution_candidate = np.array(G.B.multiply_left(bab_01), dtype=DTYPE)
        diff = t - solution_candidate
        diff_nrm_sq = diff @ diff

        if diff_nrm_sq <= min_norm_err_sq:
            min_norm_err_sq = diff_nrm_sq
            best_bab_01 = bab_01
            if tracer_alg2 is not None:
                tracer_alg2["walltime"] = time.perf_counter() - startt
                tracer_alg2["len(target_candidates)"] = len(target_candidates)
                tracer_alg2["nrand"] = nrand
            yield best_bab_01

    return best_bab_01


def alg_3(g6k, B, H11, t, n_guess_coord, eta, dist_sq_bnd=1.0, nthreads=1, tracer_alg3=None):
    then_start = perf_counter()
    dim = B.nrows
    print(f"Lattice dimension: {dim}")

    t1, t2 = t[:-n_guess_coord], t[-n_guess_coord:]
    slicer = RandomizedSlicer(g6k)
    distrib = centeredBinomial(eta)
    nsampl = ceil(2 ** (distrib.entropy * n_guess_coord))
    print(f"nsampl: {nsampl}")
    nsampl = min(max_nsampl, nsampl)
    target_candidates = [t1]
    vtilde2s = [np.array(t2)]

    H12 = IntegerMatrix.from_matrix([list(b)[:dim - n_guess_coord] for b in B[dim - n_guess_coord:]])
    for times in range(nsampl):
        if times != 0 and times % 64 == 0:
            print(f"{times} guesses done out of {nsampl}", end=", ")
        etilde2 = np.array(distrib.sample(n_guess_coord), dtype=DTYPE)
        vtilde2 = np.array(t2, dtype=DTYPE) - etilde2
        vtilde2s.append(vtilde2)
        tmp = H12.multiply_left(vtilde2)

        t1_ = np.array(list(t1), dtype=DTYPE) - tmp
        target_candidates.append(t1_)
    print()

    ctilde1 = alg_2_batched(g6k, target_candidates, dist_sq_bnd, nthreads=nthreads, tracer_alg2=None)

    v1 = np.array(g6k.M.B[: len(ctilde1)].multiply_left(ctilde1))
    argminv = None
    minv = 10**12
    cntr = 0
    for vtilde2 in vtilde2s:
        tmp = H12.multiply_left(vtilde2)
        v2 = np.concatenate([(dim - n_guess_coord) * [0], vtilde2])
        v = np.concatenate([v1, n_guess_coord * [0]]) + v2 + np.concatenate(
            [np.array(H12.multiply_left(vtilde2)), n_guess_coord * [0]]
        )
        v_t = v - np.array(t)
        vv = v_t @ v_t
        if vv < minv:
            minv = vv
            argminv = v
        cntr += 1
    return argminv


def build_target_candidates_debug_v2(B, target, n_guess_coord, dist, dist_param, s, g6k_len, tracer=None):
    """
    只负责生成 wrong/correct 两类 target candidates，不做验证。
    第一阶段给跨实例 batch 调度使用。

    这一版的目标：
      1) 适度放开同一 lattice 下实际生成的候选数
      2) 让候选数随 n_guess_coord 增大而温和增长
      3) 保持 current pipeline 不变：仍返回 wrong/correct 两类候选
    """
    dim = B.nrows
    n = dim
    t1, t2 = target[:-n_guess_coord], target[-n_guess_coord:]

    if dist == "binomial":
        dist_param = int(dist_param)
        distrib = centeredBinomial(dist_param)
    elif dist == "ternary":
        distrib = ternaryDist(dist_param)
    elif dist == "ternary_sparse":
        distrib = sparse_distribution(
            n,
            n_guess_coord,
            int(dist_param),
            Distribution({-1: 0.5, 1: 0.5}),
        )
    else:
        raise ValueError("unsupported distribution")

    # 理论候选规模
    nsampl = ceil(2 ** (distrib.entropy * n_guess_coord))
    if tracer is not None:
        tracer["key_num"] = nsampl

    H12 = IntegerMatrix.from_matrix(
        [list(b)[: dim - n_guess_coord] for b in B[dim - n_guess_coord:]]
    )

    # ---------- 新的“适度放量”规则 ----------
    # 旧版：
    #   times = min(10, max(1, g6k_len))
    #
    # 新版：
    #   budget = max(min_cap, min(max_cap, max(2*g6k_len, ceil(sqrt(key_num)))))
    #
    # 直觉：
    #   - g6k_len 体现当前尾部格规模
    #   - sqrt(key_num) 让 guess 位数增大时，候选数也能温和上涨
    #   - max_cap 防止直接爆炸
    #
    min_cap = HYB_CAND_MIN_PER_BRANCH if "HYB_CAND_MIN_PER_BRANCH" in globals() else 16
    max_cap = HYB_CAND_MAX_PER_BRANCH if "HYB_CAND_MAX_PER_BRANCH" in globals() else 64
    use_sqrt_keynum = HYB_CAND_USE_SQRT_KEYNUM if "HYB_CAND_USE_SQRT_KEYNUM" in globals() else True
    do_dedup = HYB_CAND_DEDUP if "HYB_CAND_DEDUP" in globals() else True

    if use_sqrt_keynum:
        budget_core = max(2 * int(g6k_len), int(ceil(nsampl ** 0.5)))
    else:
        budget_core = 2 * int(g6k_len)

    times = max(min_cap, min(max_cap, budget_core))

    if tracer is not None:
        tracer["candidate_budget_per_branch"] = times

    def to_key(v):
        # 用 int64 规整成 hash key，便于去重
        return tuple(np.asarray(v, dtype=np.int64).tolist())

    true_etilde2 = np.array(-s[-n_guess_coord:], dtype=np.int64)

    # ===== wrong candidates =====
    wrong_candidates = []
    wrong_vtilde2s = []
    wrong_seen = set()

    attempts = 0
    max_attempts = max(10 * times, 100)

    while len(wrong_candidates) < times and attempts < max_attempts:
        attempts += 1

        etilde2 = np.array(distrib.sample(n_guess_coord), dtype=np.int64)

        # wrong branch: 尽量避开真实 guess
        if np.array_equal(etilde2, true_etilde2):
            continue

        vtilde2 = np.array(t2, dtype=np.int64) - etilde2
        key = to_key(vtilde2)

        if do_dedup and key in wrong_seen:
            continue

        wrong_seen.add(key)
        wrong_vtilde2s.append(vtilde2)

        tmp = np.array(H12.multiply_left(vtilde2))
        t1_ = np.array(list(t1), dtype=np.int64) - tmp
        wrong_candidates.append(np.asarray(t1_, dtype=np.float64))

    # 如果因为去重/分布问题没凑够，就放宽限制继续补齐，但仍尽量避免重复
    attempts = 0
    while len(wrong_candidates) < times and attempts < max_attempts:
        attempts += 1

        etilde2 = np.array(distrib.sample(n_guess_coord), dtype=np.int64)
        vtilde2 = np.array(t2, dtype=np.int64) - etilde2
        key = to_key(vtilde2)

        if do_dedup and key in wrong_seen:
            continue

        wrong_seen.add(key)
        wrong_vtilde2s.append(vtilde2)

        tmp = np.array(H12.multiply_left(vtilde2))
        t1_ = np.array(list(t1), dtype=np.int64) - tmp
        wrong_candidates.append(np.asarray(t1_, dtype=np.float64))

    # ===== correct candidates =====
    correct_candidates = []
    correct_vtilde2s = []
    correct_seen = set()

    # 先强制放入真实 guess，对应一个 guaranteed correct-style candidate
    vtilde2_true = np.array(t2, dtype=np.int64) - true_etilde2
    key_true = to_key(vtilde2_true)
    correct_seen.add(key_true)
    correct_vtilde2s.append(vtilde2_true)

    tmp = np.array(H12.multiply_left(vtilde2_true))
    t1_true = np.array(list(t1), dtype=np.int64) - tmp
    correct_candidates.append(np.asarray(t1_true, dtype=np.float64))

    attempts = 0
    while len(correct_candidates) < times and attempts < max_attempts:
        attempts += 1

        etilde2 = np.array(distrib.sample(n_guess_coord), dtype=np.int64)
        vtilde2 = np.array(t2, dtype=np.int64) - etilde2
        key = to_key(vtilde2)

        if do_dedup and key in correct_seen:
            continue

        correct_seen.add(key)
        correct_vtilde2s.append(vtilde2)

        tmp = np.array(H12.multiply_left(vtilde2))
        t1_ = np.array(list(t1), dtype=np.int64) - tmp
        correct_candidates.append(np.asarray(t1_, dtype=np.float64))

    if tracer is not None:
        tracer["wrong_candidates_generated"] = len(wrong_candidates)
        tracer["correct_candidates_generated"] = len(correct_candidates)

    return {
        "wrong_candidates": wrong_candidates,
        "wrong_vtilde2s": wrong_vtilde2s,
        "correct_candidates": correct_candidates,
        "correct_vtilde2s": correct_vtilde2s,
        "key_num": nsampl,
    }


def alg_3_debug_v2(g6k, H11, B, target, n_guess_coord, dist, dist_param, s, dist_sq_bnd=1.0, nthreads=1, tracer_alg3=None, verbose=False):
    # Emulates batch CVPP with guessing.
    then_start = perf_counter()
    gh_sub = gaussian_heuristic(g6k.M.r()[-(g6k.r - g6k.l):])
    dim = B.nrows
    n = dim
    if verbose:
        print(f"Lattice dimension: {dim}")

    t1, t2 = target[:-n_guess_coord], target[-n_guess_coord:]
    if dist == "binomial":
        distrib = centeredBinomial(dist_param)
    elif dist == "ternary":
        distrib = ternaryDist(dist_param)
    elif dist == "ternary_sparse":
        distrib = sparse_distribution(n, n_guess_coord, int(dist_param), Distribution({-1: 0.5, 1: 0.5}))
    else:
        raise ValueError("unsupported distribution")

    nsampl = ceil(2 ** (distrib.entropy * n_guess_coord))
    if tracer_alg3 is not None:
        tracer_alg3["key_num"] = nsampl

    H12 = IntegerMatrix.from_matrix([list(b)[:dim - n_guess_coord] for b in B[dim - n_guess_coord:]])
    sieve_dim = g6k.r - g6k.l

    from hybrid_estimator.batchCVP import batchCVPP_cost
    nrand_, _ = batchCVPP_cost(sieve_dim, 100, len(g6k) ** (1.0 / sieve_dim), 1)
    nrand = ceil(NRAND_FACTOR * (1.0 / nrand_) ** sieve_dim)
    if verbose:
        print(f"Number of batches: {ceil(len(g6k) / nrand)}")
    times = ceil(len(g6k) / nrand)

    tracer_alg2_correct, tracer_alg2_wrong = {}, {}

    # - - - BEGIN INCORRECT GUESS - - -
    wrong_guess_time_alg3_start = time.perf_counter()
    target_candidates = []
    vtilde2s = []
    wrong_guess_time_start = time.perf_counter()

    for cntr in range(times):
        if cntr != 0 and cntr % 1000 == 0 and verbose:
            print(f"{cntr} guesses done out of {nsampl}", end=", ")
        etilde2 = np.array(distrib.sample(n_guess_coord))
        vtilde2 = np.array(t2) - etilde2
        vtilde2s.append(vtilde2)

        tmp = np.array(H12.multiply_left(vtilde2))
        t1_ = np.array(list(t1)) - tmp
        target_candidates.append(t1_)
    print()

    it = alg_2_batched(g6k, target_candidates, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg2=tracer_alg2_wrong)
    ctilde1 = np.zeros(dim - n_guess_coord)
    for ctilde1 in it:
        break

    v1 = np.array(H11.multiply_left(ctilde1))
    argminv = None
    minv = 10**12
    cntr = 0
    for vtilde2 in vtilde2s:
        v2 = np.concatenate([(dim - n_guess_coord) * [0], vtilde2])
        babshift = np.concatenate([np.array(H12.multiply_left(vtilde2)), n_guess_coord * [0]])
        v = np.concatenate([v1, n_guess_coord * [0]]) + v2 + babshift

        v_t = v - np.array(target)
        vv = v_t @ v_t
        if vv < minv:
            minv = vv
            argminv = v
        cntr += 1

    wrong_guess_time = time.perf_counter() - wrong_guess_time_start
    wrong_guess_time_alg3 = time.perf_counter() - wrong_guess_time_alg3_start

    if tracer_alg3 is not None:
        tracer_alg3["wrong_guess_time_alg3"] = wrong_guess_time
        tracer_alg3["wrong_guess_time_alg2"] = tracer_alg2_wrong.get("walltime", 0)

    # - - - BEGIN CORRECT GUESS - - -
    target_candidates = []
    vtilde2s = []
    correct_guess_time_start = time.perf_counter()

    for i in range(times):
        if i != 0 and i % 1000 == 0:
            print(f"{i} guesses done out of {nsampl}", end=", ")
        if i > 0:
            etilde2 = np.array(distrib.sample(n_guess_coord))
        else:
            etilde2 = np.array(-s[-n_guess_coord:])
        vtilde2 = np.array(t2) - etilde2
        vtilde2s.append(vtilde2)

        tmp = np.array(H12.multiply_left(vtilde2))
        t1_ = np.array(list(t1)) - tmp
        target_candidates.append(t1_)

    it = alg_2_batched(g6k, target_candidates, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg2=tracer_alg2_correct)

    if tracer_alg3 is not None:
        tracer_alg3["correct_guess_time_alg3"] = 0
        tracer_alg3["correct_guess_time_alg2"] = 0

    for ctilde1 in it:
        v1 = np.array(H11.multiply_left(ctilde1))
        argminv_correct = None
        minv = 10**12
        cntr = 0
        for vtilde2 in vtilde2s:
            v2 = np.concatenate([(dim - n_guess_coord) * [0], vtilde2])
            babshift = np.concatenate([np.array(H12.multiply_left(vtilde2)), n_guess_coord * [0]])
            v = np.concatenate([v1, n_guess_coord * [0]]) + v2 + babshift

            v_t = v - np.array(target)
            vv = v_t @ v_t
            if vv < minv:
                minv = vv
                argminv_correct = v
                correct_guess_time = time.perf_counter() - correct_guess_time_start
                if tracer_alg3 is not None:
                    tracer_alg3["correct_guess_time_alg3"] = correct_guess_time
                    tracer_alg3["correct_guess_time_alg2"] = tracer_alg2_correct.get("walltime", 0)
                yield argminv_correct
            cntr += 1


def alg_3_debug(g6k, H11, B, target, n_guess_coord, dist, dist_param, dist_sq_bnd=1.0, nthreads=1, tracer_alg3=None):
    then_start = perf_counter()
    gh_sub = gaussian_heuristic(g6k.M.r()[-(g6k.r - g6k.l):])
    dim = B.nrows
    n = dim

    t1, t2 = target[:-n_guess_coord], target[-n_guess_coord:]
    if dist == "binomial":
        distrib = centeredBinomial(dist_param)
    elif dist == "ternary":
        distrib = ternaryDist(dist_param)
    elif dist == "ternary_sparse":
        distrib = sparse_distribution(n, n_guess_coord, int(dist_param), Distribution({-1: 0.5, 1: 0.5}))
    else:
        raise ValueError("unsupported distribution")

    nsampl = ceil(2 ** (distrib.entropy * n_guess_coord))
    if tracer_alg3 is not None:
        tracer_alg3["key_num"] = 0

    H12 = IntegerMatrix.from_matrix([list(b)[:dim - n_guess_coord] for b in B[dim - n_guess_coord:]])
    sieve_dim = g6k.r - g6k.l

    from hybrid_estimator.batchCVP import batchCVPP_cost
    nrand_, _ = batchCVPP_cost(sieve_dim, 100, len(g6k) ** (1.0 / sieve_dim), 1)
    nrand = ceil(NRAND_FACTOR * (1.0 / nrand_) ** sieve_dim)
    times = ceil(len(g6k) / nrand)

    tracer_alg2_correct, tracer_alg2_wrong = {}, {}
    correct_guess_time_start = time.perf_counter()

    for _ in range(ceil(nsampl / times)):
        target_candidates = []
        vtilde2s = []
        for cntr in range(times):
            if tracer_alg3 is not None:
                tracer_alg3["key_num"] += 1
            if cntr != 0 and cntr % 1000 == 0:
                print(f"{cntr} guesses done out of {nsampl}", end=", ")
            etilde2 = np.array(distrib.sample(n_guess_coord))
            vtilde2 = np.array(t2) - etilde2
            vtilde2s.append(vtilde2)
            tmp = np.array(H12.multiply_left(vtilde2))

            t1_ = np.array(list(t1)) - tmp
            target_candidates.append(t1_)

        it = alg_2_batched(g6k, target_candidates, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg2=tracer_alg2_correct)
        if tracer_alg3 is not None:
            tracer_alg3["correct_guess_time_alg3"] = 0
            tracer_alg3["correct_guess_time_alg2"] = 0

        for ctilde1 in it:
            v1 = np.array(H11.multiply_left(ctilde1))
            argminv_correct = None
            minv = 10**12
            cntr = 0
            for vtilde2 in vtilde2s:
                v2 = np.concatenate([(dim - n_guess_coord) * [0], vtilde2])
                babshift = np.concatenate([np.array(H12.multiply_left(vtilde2)), n_guess_coord * [0]])
                v = np.concatenate([v1, n_guess_coord * [0]]) + v2 + babshift

                v_t = v - np.array(target)
                vv = v_t @ v_t
                if vv < minv:
                    minv = vv
                    argminv_correct = v
                    correct_guess_time = time.perf_counter() - correct_guess_time_start
                    if tracer_alg3 is not None:
                        tracer_alg3["correct_guess_time_alg3"] = correct_guess_time
                        tracer_alg3["correct_guess_time_alg2"] = tracer_alg2_correct.get("walltime", 0)
                    yield argminv_correct
                cntr += 1