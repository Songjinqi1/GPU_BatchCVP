from dataclasses import dataclass
from typing import Any, Iterator, List, Optional

import numpy as np


@dataclass
class TargetJob:
    job_id: int
    lat_index: int
    exp_index: int
    branch: str            # "wrong" or "correct"
    candidate_index: int
    target: Any            # np.ndarray
    dist_sq_bnd: float
    meta: dict
    target_hash: str = ""
    target_dim: int = 0

    def to_dict(self) -> dict:
        arr = np.asarray(self.target, dtype=np.float64).reshape(-1)
        return {
            "job_id": int(self.job_id),
            "lat_index": int(self.lat_index),
            "exp_index": int(self.exp_index),
            "branch": str(self.branch),
            "candidate_index": int(self.candidate_index),
            "target": arr,
            "dist_sq_bnd": float(self.dist_sq_bnd),
            "meta": dict(self.meta),
            "target_hash": str(self.target_hash),
            "target_dim": int(self.target_dim if self.target_dim else len(arr)),
        }

    @staticmethod
    def from_dict(d: dict) -> "TargetJob":
        arr = np.asarray(d["target"], dtype=np.float64).reshape(-1)
        return TargetJob(
            job_id=int(d["job_id"]),
            lat_index=int(d["lat_index"]),
            exp_index=int(d["exp_index"]),
            branch=str(d["branch"]),
            candidate_index=int(d["candidate_index"]),
            target=arr,
            dist_sq_bnd=float(d["dist_sq_bnd"]),
            meta=dict(d.get("meta", {})),
            target_hash=str(d.get("target_hash", "")),
            target_dim=int(d.get("target_dim", len(arr))),
        )


class BatchScheduler:
    def __init__(self, max_targets: int):
        self.max_targets = int(max_targets)
        self.jobs: List[TargetJob] = []
        self._next_job_id = 0

    def add_job(
        self,
        lat_index: int,
        exp_index: int,
        branch: str,
        candidate_index: int,
        target: Any,
        dist_sq_bnd: float,
        meta: Optional[dict] = None,
        target_hash: str = "",
        target_dim: Optional[int] = None,
    ) -> int:
        if meta is None:
            meta = {}

        arr = np.asarray(target, dtype=np.float64).reshape(-1)
        if target_dim is None:
            target_dim = len(arr)

        job = TargetJob(
            job_id=self._next_job_id,
            lat_index=int(lat_index),
            exp_index=int(exp_index),
            branch=str(branch),
            candidate_index=int(candidate_index),
            target=arr,
            dist_sq_bnd=float(dist_sq_bnd),
            meta=meta,
            target_hash=str(target_hash),
            target_dim=int(target_dim),
        )
        self.jobs.append(job)
        self._next_job_id += 1
        return job.job_id

    def clear(self) -> None:
        self.jobs = []
        self._next_job_id = 0

    def num_jobs(self) -> int:
        return len(self.jobs)

    def iter_batches(self) -> Iterator[List[TargetJob]]:
        for i in range(0, len(self.jobs), self.max_targets):
            yield self.jobs[i:i + self.max_targets]

    def describe(self) -> dict:
        return {
            "num_jobs": len(self.jobs),
            "max_targets": self.max_targets,
            "num_batches": (len(self.jobs) + self.max_targets - 1) // self.max_targets if self.max_targets > 0 else 0,
        }

    def to_serializable(self) -> dict:
        return {
            "max_targets": int(self.max_targets),
            "next_job_id": int(self._next_job_id),
            "jobs": [job.to_dict() for job in self.jobs],
        }

    @staticmethod
    def from_serializable(data: dict) -> "BatchScheduler":
        sched = BatchScheduler(max_targets=int(data["max_targets"]))
        sched.jobs = [TargetJob.from_dict(x) for x in data["jobs"]]
        sched._next_job_id = int(data.get("next_job_id", len(sched.jobs)))
        return sched