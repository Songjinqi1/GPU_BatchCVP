from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = REPO_ROOT.parent

import argparse
import ast
import math
import statistics
from typing import Any, Dict, List


def _sanitize_for_literal_eval(text: str) -> str:
    import re
    text = re.sub(r':\s*-?inf\b', lambda m: m.group(0).replace('inf', '1e400'), text)
    text = re.sub(r':\s*nan\b', ': 0.0', text)
    return text


def extract_last_python_dict(text: str) -> Dict[str, Any]:
    start = None
    depth = 0
    last_block = None

    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    last_block = text[start:i + 1]

    if last_block is None:
        raise ValueError("未在日志中找到可解析的顶层 Python 字典。")

    try:
        return ast.literal_eval(_sanitize_for_literal_eval(last_block))
    except Exception as e:
        raise ValueError(f"找到字典文本，但解析失败: {e}") from e


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return extract_last_python_dict(text)


def format_float(x: Any) -> str:
    if x is None:
        return "None"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.12f}"
    return str(x)


def rows_to_map(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for r in rows:
        jid = int(r["job_id"])
        if jid in out:
            raise ValueError(f"发现重复 job_id: {jid}")
        out[jid] = r
    return out


def validate_rows(rows: List[Dict[str, Any]], role: str):
    none_scores = []
    missing_hash = []
    backends = {}
    for r in rows:
        b = r.get("backend", "unknown")
        backends[b] = backends.get(b, 0) + 1
        if r.get("score") is None:
            none_scores.append(int(r["job_id"]))
        if "target_hash" not in r:
            missing_hash.append(int(r["job_id"]))

    if none_scores:
        raise ValueError(
            f"{role} 日志中存在 score=None，不能用于数值一致性比较。"
            f" backend_counter={backends}, 前20个问题 job_id={none_scores[:20]}"
        )

    if missing_hash:
        raise ValueError(
            f"{role} 日志中缺少 target_hash，不能用于严格同输入对照。"
            f" 前20个问题 job_id={missing_hash[:20]}"
        )

    return backends


def compare_scores(cpu_rows: List[Dict[str, Any]], gpu_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cpu_backends = validate_rows(cpu_rows, "CPU")
    gpu_backends = validate_rows(gpu_rows, "GPU")

    cpu_map = rows_to_map(cpu_rows)
    gpu_map = rows_to_map(gpu_rows)

    cpu_ids = set(cpu_map.keys())
    gpu_ids = set(gpu_map.keys())

    only_cpu = sorted(cpu_ids - gpu_ids)
    only_gpu = sorted(gpu_ids - cpu_ids)
    common = sorted(cpu_ids & gpu_ids)

    if not common:
        raise ValueError("CPU 与 GPU 没有重叠的 job_id。")

    abs_errs = []
    rel_errs = []
    mismatched_meta = []
    exact_equal_count = 0

    for jid in common:
        c = cpu_map[jid]
        g = gpu_map[jid]

        meta_keys = [
            "lat_index",
            "exp_index",
            "branch",
            "candidate_index",
            "ell",
            "target_hash",
            "target_dim",
        ]
        for k in meta_keys:
            if c.get(k) != g.get(k):
                mismatched_meta.append(
                    {
                        "job_id": jid,
                        "field": k,
                        "cpu": c.get(k),
                        "gpu": g.get(k),
                    }
                )

        c_score = float(c["score"])
        g_score = float(g["score"])
        abs_err = abs(c_score - g_score)
        abs_errs.append(abs_err)

        denom = max(abs(c_score), 1e-15)
        rel_err = abs_err / denom
        rel_errs.append(rel_err)

        if c_score == g_score:
            exact_equal_count += 1

    out = {
        "cpu_backend_counter": cpu_backends,
        "gpu_backend_counter": gpu_backends,
        "cpu_job_count": len(cpu_rows),
        "gpu_job_count": len(gpu_rows),
        "common_job_count": len(common),
        "cpu_only_job_ids": only_cpu,
        "gpu_only_job_ids": only_gpu,
        "meta_mismatch_count": len(mismatched_meta),
        "meta_mismatches_preview": mismatched_meta[:20],
        "exact_equal_count": exact_equal_count,
        "max_abs_err": max(abs_errs),
        "mean_abs_err": statistics.mean(abs_errs),
        "median_abs_err": statistics.median(abs_errs),
        "max_rel_err": max(rel_errs),
        "mean_rel_err": statistics.mean(rel_errs),
        "median_rel_err": statistics.median(rel_errs),
    }

    pairs = []
    for jid in common:
        c_score = float(cpu_map[jid]["score"])
        g_score = float(gpu_map[jid]["score"])
        abs_err = abs(c_score - g_score)
        pairs.append(
            {
                "job_id": jid,
                "lat_index": cpu_map[jid]["lat_index"],
                "exp_index": cpu_map[jid]["exp_index"],
                "branch": cpu_map[jid]["branch"],
                "candidate_index": cpu_map[jid]["candidate_index"],
                "target_hash": cpu_map[jid]["target_hash"],
                "cpu_score": c_score,
                "gpu_score": g_score,
                "abs_err": abs_err,
            }
        )
    pairs.sort(key=lambda x: x["abs_err"], reverse=True)
    out["largest_abs_err_preview"] = pairs[:20]

    return out


def compare_times(cpu_summary: Dict[str, Any], gpu_summary: Dict[str, Any]) -> Dict[str, Any]:
    cpu_batch_wall = float(cpu_summary.get("batch_walltime_total_sec", float("nan")))
    gpu_batch_wall = float(gpu_summary.get("batch_walltime_total_sec", float("nan")))

    cpu_backend_time = float(cpu_summary.get("cpu_time_sec", float("nan")))
    gpu_backend_time = float(gpu_summary.get("gpu_time_sec", float("nan")))

    out = {
        "cpu_batch_walltime_total_sec": cpu_batch_wall,
        "gpu_batch_walltime_total_sec": gpu_batch_wall,
        "cpu_backend_time_sec": cpu_backend_time,
        "gpu_backend_time_sec": gpu_backend_time,
        "batch_walltime_speedup": (cpu_batch_wall / gpu_batch_wall) if gpu_batch_wall > 0 else float("nan"),
        "backend_time_speedup": (cpu_backend_time / gpu_backend_time) if gpu_backend_time > 0 else float("nan"),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare CPU vs GPU replay outputs.")
    parser.add_argument("--cpu_log", required=True, help="CPU log file path")
    parser.add_argument("--gpu_log", required=True, help="GPU log file path")
    args = parser.parse_args()

    cpu_summary = load_summary(args.cpu_log)
    gpu_summary = load_summary(args.gpu_log)

    cpu_rows = cpu_summary.get("all_batch_results")
    gpu_rows = gpu_summary.get("all_batch_results")
    have_score_data = isinstance(cpu_rows, list) and isinstance(gpu_rows, list)

    if have_score_data:
        score_cmp = compare_scores(cpu_rows, gpu_rows)
    time_cmp = compare_times(cpu_summary, gpu_summary)

    if have_score_data:
        print("=== CPU vs GPU Score Consistency ===")
        print()

        print("[Backend Counter]")
        print(f"{'cpu_backend_counter':28s}: {score_cmp['cpu_backend_counter']}")
        print(f"{'gpu_backend_counter':28s}: {score_cmp['gpu_backend_counter']}")
        print()

        print("[Job Coverage]")
        print(f"{'cpu_job_count':28s}: {score_cmp['cpu_job_count']}")
        print(f"{'gpu_job_count':28s}: {score_cmp['gpu_job_count']}")
        print(f"{'common_job_count':28s}: {score_cmp['common_job_count']}")
        print(f"{'cpu_only_job_count':28s}: {len(score_cmp['cpu_only_job_ids'])}")
        print(f"{'gpu_only_job_count':28s}: {len(score_cmp['gpu_only_job_ids'])}")
        print()

        print("[Metadata Consistency]")
        print(f"{'meta_mismatch_count':28s}: {score_cmp['meta_mismatch_count']}")
        if score_cmp["meta_mismatch_count"] > 0:
            print("meta mismatches preview:")
            for row in score_cmp["meta_mismatches_preview"]:
                print(
                    f"  job_id={row['job_id']}, field={row['field']}, "
                    f"cpu={row['cpu']}, gpu={row['gpu']}"
                )
        print()

        print("[Score Error]")
        print(f"{'exact_equal_count':28s}: {score_cmp['exact_equal_count']}")
        print(f"{'max_abs_err':28s}: {format_float(score_cmp['max_abs_err'])}")
        print(f"{'mean_abs_err':28s}: {format_float(score_cmp['mean_abs_err'])}")
        print(f"{'median_abs_err':28s}: {format_float(score_cmp['median_abs_err'])}")
        print(f"{'max_rel_err':28s}: {format_float(score_cmp['max_rel_err'])}")
        print(f"{'mean_rel_err':28s}: {format_float(score_cmp['mean_rel_err'])}")
        print(f"{'median_rel_err':28s}: {format_float(score_cmp['median_rel_err'])}")
        print()

        print("[Largest Score Differences Preview]")
        for row in score_cmp["largest_abs_err_preview"][:10]:
            print(
                f"job_id={row['job_id']}, lat={row['lat_index']}, exp={row['exp_index']}, "
                f"branch={row['branch']}, cand={row['candidate_index']}, "
                f"hash={row['target_hash'][:12]}, "
                f"cpu={row['cpu_score']:.12f}, gpu={row['gpu_score']:.12f}, "
                f"abs_err={row['abs_err']:.12f}"
            )
        print()

    print("=== CPU vs GPU Time Comparison ===")
    print()
    print(f"{'cpu_batch_walltime_total_sec':28s}: {format_float(time_cmp['cpu_batch_walltime_total_sec'])}")
    print(f"{'gpu_batch_walltime_total_sec':28s}: {format_float(time_cmp['gpu_batch_walltime_total_sec'])}")
    print(f"{'cpu_backend_time_sec':28s}: {format_float(time_cmp['cpu_backend_time_sec'])}")
    print(f"{'gpu_backend_time_sec':28s}: {format_float(time_cmp['gpu_backend_time_sec'])}")
    print(f"{'batch_walltime_speedup':28s}: {format_float(time_cmp['batch_walltime_speedup'])}")
    print(f"{'backend_time_speedup':28s}: {format_float(time_cmp['backend_time_speedup'])}")

    print()
    print("说明：speedup = CPU时间 / GPU时间，>1 表示 GPU 更快。")


if __name__ == "__main__":
    main()