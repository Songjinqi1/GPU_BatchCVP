from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from global_consts import *


@dataclass
class HybridBaseParams:
    nthreads: int
    n: int
    q: int
    dist: str
    dist_param: float
    beta_pre: int
    n_guess_coord: int
    n_slicer_coord: int
    lats_per_dim: int = 1
    verbose: bool = False

    def to_legacy_params(self) -> Dict[str, Any]:
        return {
            "nthreads": self.nthreads,
            "n": self.n,
            "q": self.q,
            "dist": self.dist,
            "dist_param": self.dist_param,
            "beta_pre": self.beta_pre,
            "n_guess_coord": self.n_guess_coord,
            "n_slicer_coord": self.n_slicer_coord,
            "lats_per_dim": self.lats_per_dim,
            "verbose": self.verbose,
        }


@dataclass
class BatchRuntimeConfig:
    max_targets: int = HYB_BATCH_MAX_TARGETS
    batch_verbose: bool = HYB_BATCH_VERBOSE
    gpu_backend_enable: bool = HYB_GPU_BACKEND_ENABLE
    gpu_backend_verbose: bool = HYB_GPU_BACKEND_VERBOSE
    gpu_proj_tau_scale: float = HYB_GPU_PROJ_TAU_SCALE
    gpu_allow_fallback: bool = HYB_GPU_ALLOW_FALLBACK
    gpu_sync_timing: bool = HYB_GPU_SYNC_TIMING
    gpu_np_impl: str = HYB_GPU_NP_IMPL
    gpu_np_block_rows: int = HYB_GPU_NP_BLOCK_ROWS
    gpu_use_rowmajor_t: bool = HYB_GPU_USE_ROWMAJOR_T
    enumerator_enable: bool = HYB_ENUMERATOR_ENABLE
    enum_chunk_size: int = HYB_ENUMERATOR_CHUNK_SIZE
    wrong_target_count: int = HYB_WRONG_TARGET_COUNT
    correct_target_count: int = HYB_CORRECT_TARGET_COUNT

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DumpBuildConfig:
    base: HybridBaseParams
    delta_slicer_coord: int = 0
    json_out: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base": asdict(self.base),
            "delta_slicer_coord": self.delta_slicer_coord,
            "json_out": self.json_out,
        }


@dataclass
class ReplayConfig:
    dump_path: str
    runtime: BatchRuntimeConfig
    validate_projected_batches: int = 0
    validate_atol: float = 1e-8
    validate_rtol: float = 1e-8
    json_out: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dump_path": self.dump_path,
            "runtime": self.runtime.to_dict(),
            "validate_projected_batches": self.validate_projected_batches,
            "validate_atol": self.validate_atol,
            "validate_rtol": self.validate_rtol,
            "json_out": self.json_out,
        }


def write_json_summary(payload: Dict[str, Any], output_path: Optional[str]) -> Optional[str]:
    if not output_path:
        return None
    import json
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)
