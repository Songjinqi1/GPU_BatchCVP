"""Batch dump / replay submodules for hybrid projected backend."""

from .common import (
    compute_target_hash,
    build_all_targets_matrix_from_scheduler,
    serialize_projection_cache,
    deserialize_projection_cache,
    enrich_results_with_job_info,
    compact_summary_for_print,
)
from .config import (
    HybridBaseParams,
    BatchRuntimeConfig,
    DumpBuildConfig,
    ReplayConfig,
    write_json_summary,
)
from .dump import prepare_batch_material, build_batch_candidate_dump
from .replay import replay_batch_candidate_dump, run_experiment_batch

__all__ = [
    "HybridBaseParams",
    "BatchRuntimeConfig",
    "DumpBuildConfig",
    "ReplayConfig",
    "write_json_summary",
    "compute_target_hash",
    "build_all_targets_matrix_from_scheduler",
    "serialize_projection_cache",
    "deserialize_projection_cache",
    "enrich_results_with_job_info",
    "compact_summary_for_print",
    "prepare_batch_material",
    "build_batch_candidate_dump",
    "replay_batch_candidate_dump",
    "run_experiment_batch",
]
