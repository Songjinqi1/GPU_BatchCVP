from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Dict, Any, Optional, Tuple

import math
import numpy as np


def support_for_distribution(dist: str, dist_param: float) -> np.ndarray:
    """
    Return the discrete support of the guess distribution.

    First engineering version:
      - supports binomial(k): centered binomial support [-k, ..., k]

    Returns
    -------
    np.ndarray, shape (base,), dtype=int16
    """
    if dist == "binomial":
        k = int(dist_param)
        return np.arange(-k, k + 1, dtype=np.int16)

    raise NotImplementedError(
        f"candidate_enumerator currently supports dist='binomial' only, got {dist!r}"
    )


def support_space_size(dist: str, dist_param: float, guess_dim: int) -> int:
    support = support_for_distribution(dist, dist_param)
    return int(len(support) ** int(guess_dim))


def decode_indices_to_vectors(
    indices: np.ndarray,
    support: np.ndarray,
    guess_dim: int,
) -> np.ndarray:
    """
    Decode integer indices into support^guess_dim vectors using mixed radix.

    Parameters
    ----------
    indices:
        shape (m,), integer indices in [0, len(support)^guess_dim)
    support:
        shape (base,), discrete support values
    guess_dim:
        number of guessed coordinates

    Returns
    -------
    vectors:
        shape (m, guess_dim), dtype=int16
    """
    base = int(len(support))
    idx = np.asarray(indices, dtype=np.int64).copy()
    m = len(idx)

    out = np.empty((m, guess_dim), dtype=np.int16)

    # least-significant digit on the right
    for j in range(guess_dim - 1, -1, -1):
        digit = idx % base
        out[:, j] = support[digit]
        idx //= base

    return out


def enumerate_vectors_chunked(
    support: np.ndarray,
    guess_dim: int,
    start: int,
    count: int,
    chunk_size: int,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Enumerate support^guess_dim vectors over [start, start+count), chunk by chunk.

    Yields
    ------
    (chunk_start_index, vectors_chunk)
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    total_space = int(len(support) ** int(guess_dim))
    if start < 0 or count < 0:
        raise ValueError(f"start/count must be nonnegative, got start={start}, count={count}")
    if start > total_space:
        raise ValueError(f"start={start} exceeds total_space={total_space}")

    end = min(start + count, total_space)
    cur = start
    while cur < end:
        nxt = min(cur + chunk_size, end)
        idx = np.arange(cur, nxt, dtype=np.int64)
        yield cur, decode_indices_to_vectors(idx, support, guess_dim)
        cur = nxt


@dataclass
class EnumerationPlan:
    dist: str
    dist_param: float
    guess_dim: int
    support_size: int
    total_space: int

    wrong_target_count: int
    correct_target_count: int

    wrong_index_start: int
    correct_index_start: int

    mode: str = "prefix_disjoint"


def build_enumeration_plan(
    dist: str,
    dist_param: float,
    guess_dim: int,
    wrong_target_count: int,
    correct_target_count: int,
    include_true_guess_in_correct: bool = True,
) -> EnumerationPlan:
    """
    Deterministic range plan.

    IMPORTANT:
      This function no longer means "scan exactly count raw indices".
      It now means "start scanning from these ranges, then continue forward until
      enough valid outputs are produced".

    wrong:
      starts at wrong_index_start

    correct:
      starts at correct_index_start
      if include_true_guess_in_correct=True, caller will explicitly prepend true guess
    """
    support = support_for_distribution(dist, dist_param)
    total_space = int(len(support) ** int(guess_dim))

    if wrong_target_count > total_space:
        raise ValueError(
            f"wrong_target_count={wrong_target_count} exceeds total_space={total_space}"
        )

    correct_extra = 1 if include_true_guess_in_correct else 0
    if correct_target_count - correct_extra > total_space:
        raise ValueError(
            f"correct_target_count={correct_target_count} too large for total_space={total_space}"
        )

    return EnumerationPlan(
        dist=dist,
        dist_param=dist_param,
        guess_dim=int(guess_dim),
        support_size=len(support),
        total_space=total_space,
        wrong_target_count=int(wrong_target_count),
        correct_target_count=int(correct_target_count),
        wrong_index_start=0,
        correct_index_start=int(wrong_target_count),
        mode="prefix_disjoint",
    )


def build_candidate_chunk_from_etilde2(
    H12_np: np.ndarray,         # shape (dim - guess_dim, guess_dim)
    t1_np: np.ndarray,          # shape (dim - guess_dim,)
    t2_np: np.ndarray,          # shape (guess_dim,)
    etilde2_chunk: np.ndarray,  # shape (m, guess_dim)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized candidate construction.

    vtilde2 = t2 - etilde2
    t1'     = t1 - H12 @ vtilde2

    Returns
    -------
    t1_chunk:
        shape (m, dim - guess_dim), float64
    vtilde2_chunk:
        shape (m, guess_dim), int64
    """
    etilde2_chunk = np.asarray(etilde2_chunk, dtype=np.int64)
    H12_np = np.asarray(H12_np, dtype=np.int64)
    t1_np = np.asarray(t1_np, dtype=np.int64).reshape(-1)
    t2_np = np.asarray(t2_np, dtype=np.int64).reshape(-1)

    vtilde2_chunk = t2_np[None, :] - etilde2_chunk
    Hv = vtilde2_chunk @ H12_np.T
    t1_chunk = t1_np[None, :] - Hv

    return np.asarray(t1_chunk, dtype=np.float64), vtilde2_chunk


@dataclass
class CandidateChunk:
    branch: str
    t1_chunk: np.ndarray
    vtilde2_chunk: np.ndarray
    etilde2_chunk: np.ndarray
    index_start: int
    index_end: int
    total_space: int


def stream_branch_chunks(
    branch: str,
    H12_np: np.ndarray,
    t1_np: np.ndarray,
    t2_np: np.ndarray,
    support: np.ndarray,
    guess_dim: int,
    start: int,
    target_count: int,
    chunk_size: int,
    true_etilde2: Optional[np.ndarray] = None,
    exclude_true_in_wrong: bool = True,
    prepend_true_in_correct: bool = False,
) -> Iterator[CandidateChunk]:
    """
    Strict-fill streamer for one branch.

    This version guarantees:
      - wrong branch emits exactly target_count valid outputs
      - correct branch emits exactly target_count valid outputs
        (counting the prepended true guess if enabled)

    as long as the target_count is feasible under total_space.
    """
    if branch not in ("wrong", "correct"):
        raise ValueError(f"invalid branch={branch!r}")
    if target_count < 0:
        raise ValueError(f"target_count must be nonnegative, got {target_count}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    total_space = int(len(support) ** int(guess_dim))
    produced = 0

    # correct branch: optionally emit true guess first
    if branch == "correct" and prepend_true_in_correct and true_etilde2 is not None and target_count > 0:
        et = np.asarray(true_etilde2, dtype=np.int64).reshape(1, -1)
        t1_chunk, vtilde2_chunk = build_candidate_chunk_from_etilde2(H12_np, t1_np, t2_np, et)
        yield CandidateChunk(
            branch="correct",
            t1_chunk=t1_chunk,
            vtilde2_chunk=vtilde2_chunk,
            etilde2_chunk=et,
            index_start=-1,
            index_end=0,
            total_space=total_space,
        )
        produced += 1

    if produced >= target_count:
        return

    cur = int(start)

    while produced < target_count and cur < total_space:
        remain = target_count - produced
        raw_count = min(chunk_size, total_space - cur)
        raw_idx = np.arange(cur, cur + raw_count, dtype=np.int64)
        raw_chunk = decode_indices_to_vectors(raw_idx, support, guess_dim)
        raw_start = cur
        cur += raw_count

        etilde2_chunk = raw_chunk

        if branch == "wrong" and exclude_true_in_wrong and true_etilde2 is not None:
            mask = ~np.all(etilde2_chunk == true_etilde2[None, :], axis=1)
            etilde2_chunk = etilde2_chunk[mask]

        if branch == "correct" and prepend_true_in_correct and true_etilde2 is not None:
            mask = ~np.all(etilde2_chunk == true_etilde2[None, :], axis=1)
            etilde2_chunk = etilde2_chunk[mask]

        if len(etilde2_chunk) == 0:
            continue

        if len(etilde2_chunk) > remain:
            etilde2_chunk = etilde2_chunk[:remain]

        t1_chunk, vtilde2_chunk = build_candidate_chunk_from_etilde2(
            H12_np, t1_np, t2_np, etilde2_chunk
        )

        # 这里的 index_start/index_end 表示“原始枚举流的扫描位置区间起点”
        # 对 strict-fill 来说，它不再精确等于有效产出位置，只作为调试元数据。
        yield CandidateChunk(
            branch=branch,
            t1_chunk=t1_chunk,
            vtilde2_chunk=vtilde2_chunk,
            etilde2_chunk=etilde2_chunk,
            index_start=raw_start,
            index_end=raw_start + raw_count,
            total_space=total_space,
        )

        produced += len(etilde2_chunk)

    if produced < target_count:
        raise RuntimeError(
            f"Unable to produce enough valid candidates for branch={branch!r}: "
            f"produced={produced}, target_count={target_count}, total_space={total_space}"
        )


@dataclass
class InstanceEnumerationConfig:
    dist: str
    dist_param: float
    guess_dim: int
    wrong_target_count: int
    correct_target_count: int
    chunk_size: int = 4096
    include_true_guess_in_correct: bool = True
    exclude_true_guess_from_wrong: bool = True


def stream_instance_candidates(
    H12_np: np.ndarray,
    t1_np: np.ndarray,
    t2_np: np.ndarray,
    true_etilde2: np.ndarray,
    config: InstanceEnumerationConfig,
) -> Iterator[CandidateChunk]:
    """
    Stream wrong chunks first, then correct chunks, for one instance.
    Guarantees strict target counts if feasible.
    """
    support = support_for_distribution(config.dist, config.dist_param)
    plan = build_enumeration_plan(
        dist=config.dist,
        dist_param=config.dist_param,
        guess_dim=config.guess_dim,
        wrong_target_count=config.wrong_target_count,
        correct_target_count=config.correct_target_count,
        include_true_guess_in_correct=config.include_true_guess_in_correct,
    )

    # wrong branch: strict-fill to wrong_target_count valid outputs
    yield from stream_branch_chunks(
        branch="wrong",
        H12_np=H12_np,
        t1_np=t1_np,
        t2_np=t2_np,
        support=support,
        guess_dim=config.guess_dim,
        start=plan.wrong_index_start,
        target_count=plan.wrong_target_count,
        chunk_size=config.chunk_size,
        true_etilde2=true_etilde2,
        exclude_true_in_wrong=config.exclude_true_guess_from_wrong,
        prepend_true_in_correct=False,
    )

    # correct branch: strict-fill to correct_target_count valid outputs
    yield from stream_branch_chunks(
        branch="correct",
        H12_np=H12_np,
        t1_np=t1_np,
        t2_np=t2_np,
        support=support,
        guess_dim=config.guess_dim,
        start=plan.correct_index_start,
        target_count=plan.correct_target_count,
        chunk_size=config.chunk_size,
        true_etilde2=true_etilde2,
        exclude_true_in_wrong=False,
        prepend_true_in_correct=config.include_true_guess_in_correct,
    )


def compute_throughput_stats(total_jobs: int, elapsed_sec: float) -> Dict[str, float]:
    """
    Return throughput metrics.

    throughput   = jobs / time
    cost_per_job = time / jobs
    """
    total_jobs = int(total_jobs)
    elapsed_sec = float(elapsed_sec)

    if total_jobs <= 0 or elapsed_sec <= 0:
        return {
            "throughput": 0.0,
            "cost_per_job": math.inf,
        }

    return {
        "throughput": total_jobs / elapsed_sec,
        "cost_per_job": elapsed_sec / total_jobs,
    }