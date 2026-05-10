from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = REPO_ROOT.parent

import argparse
import ast
import csv
import math
import os
import re
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def extract_last_python_dict(text: str) -> Dict[str, Any]:
    """
    从日志文本中提取最后一个顶层 Python dict。
    适配 batch_prototype 输出里那种：
    {'gpu_backend_enabled': True, ... 'all_batch_results': [...]}
    """
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
        return ast.literal_eval(last_block)
    except Exception as e:
        raise ValueError(f"找到字典文本，但解析失败: {e}") from e


def safe_mean(xs: List[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def safe_median(xs: List[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def summarize_scores(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    wrong_scores = [float(r["score"]) for r in rows if r.get("branch") == "wrong"]
    correct_scores = [float(r["score"]) for r in rows if r.get("branch") == "correct"]

    out = {
        "wrong_count": len(wrong_scores),
        "correct_count": len(correct_scores),
        "wrong_mean": safe_mean(wrong_scores),
        "wrong_median": safe_median(wrong_scores),
        "wrong_min": min(wrong_scores) if wrong_scores else float("nan"),
        "wrong_max": max(wrong_scores) if wrong_scores else float("nan"),
        "correct_mean": safe_mean(correct_scores),
        "correct_median": safe_median(correct_scores),
        "correct_min": min(correct_scores) if correct_scores else float("nan"),
        "correct_max": max(correct_scores) if correct_scores else float("nan"),
    }
    return out


def group_by_exp(rows: List[Dict[str, Any]]) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (int(r["lat_index"]), int(r["exp_index"]))
        groups[key].append(r)
    return groups


def analyze_rank_quality(rows: List[Dict[str, Any]], topk_list: List[int]) -> Dict[str, Any]:
    """
    对每个 (lat_index, exp_index)：
    - 按 score 从小到大排序
    - 看 correct 候选最早出现在第几名
    - 看 top-k 内是否包含 correct
    """
    groups = group_by_exp(rows)

    best_correct_ranks: List[int] = []
    worst_correct_ranks: List[int] = []
    avg_correct_rank_per_group: List[float] = []

    topk_hits = {k: 0 for k in topk_list}
    total_groups = 0

    detailed = []

    for key, items in sorted(groups.items()):
        total_groups += 1
        ordered = sorted(items, key=lambda x: float(x["score"]))
        correct_positions = [idx + 1 for idx, r in enumerate(ordered) if r.get("branch") == "correct"]

        if not correct_positions:
            # 理论上不该发生，但防御性保留
            best_rank = None
            worst_rank = None
            avg_rank = None
        else:
            best_rank = min(correct_positions)
            worst_rank = max(correct_positions)
            avg_rank = statistics.mean(correct_positions)

            best_correct_ranks.append(best_rank)
            worst_correct_ranks.append(worst_rank)
            avg_correct_rank_per_group.append(avg_rank)

            for k in topk_list:
                if best_rank <= k:
                    topk_hits[k] += 1

        detailed.append(
            {
                "lat_index": key[0],
                "exp_index": key[1],
                "num_items": len(items),
                "num_correct": sum(1 for r in items if r.get("branch") == "correct"),
                "num_wrong": sum(1 for r in items if r.get("branch") == "wrong"),
                "best_correct_rank": best_rank,
                "worst_correct_rank": worst_rank,
                "avg_correct_rank": avg_rank,
            }
        )

    summary = {
        "num_groups": total_groups,
        "best_correct_rank_mean": safe_mean(best_correct_ranks),
        "best_correct_rank_median": safe_median(best_correct_ranks),
        "best_correct_rank_min": min(best_correct_ranks) if best_correct_ranks else None,
        "best_correct_rank_max": max(best_correct_ranks) if best_correct_ranks else None,
        "avg_correct_rank_mean": safe_mean(avg_correct_rank_per_group),
        "avg_correct_rank_median": safe_median(avg_correct_rank_per_group),
        "topk_hits": topk_hits,
        "topk_hit_rate": {k: (topk_hits[k] / total_groups if total_groups else float("nan")) for k in topk_list},
        "per_group": detailed,
    }
    return summary


def branch_separation_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    简单衡量 wrong/correct 分布分离情况：
    - correct_mean - wrong_mean
    - correct_median - wrong_median
    若为负，表示 correct 候选整体更靠前（更小分数）
    """
    wrong_scores = [float(r["score"]) for r in rows if r.get("branch") == "wrong"]
    correct_scores = [float(r["score"]) for r in rows if r.get("branch") == "correct"]

    if not wrong_scores or not correct_scores:
        return {
            "mean_gap_correct_minus_wrong": float("nan"),
            "median_gap_correct_minus_wrong": float("nan"),
        }

    return {
        "mean_gap_correct_minus_wrong": safe_mean(correct_scores) - safe_mean(wrong_scores),
        "median_gap_correct_minus_wrong": safe_median(correct_scores) - safe_median(wrong_scores),
    }


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(x: Any) -> str:
    if x is None:
        return "None"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.6f}"
    return str(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze score distribution from batch_prototype log.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the saved stdout/stderr log text file from run_prog_hyb.py --batch_prototype",
    )
    parser.add_argument(
        "--export_prefix",
        default=None,
        help="Optional prefix for CSV exports. Example: results/score_analysis_140_binomial",
    )
    parser.add_argument(
        "--topk",
        default="1,3,5,10",
        help="Comma-separated top-k values for rank hit analysis, e.g. 1,3,5,10",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    obj = extract_last_python_dict(text)

    all_rows = obj.get("all_batch_results")
    if not isinstance(all_rows, list):
        raise ValueError("日志字典中没有找到 all_batch_results 列表。")

    summary_scores = summarize_scores(all_rows)
    separation = branch_separation_stats(all_rows)
    topk_list = [int(x.strip()) for x in args.topk.split(",") if x.strip()]
    rank_summary = analyze_rank_quality(all_rows, topk_list)

    print("=== Batch Prototype Score Analysis ===")
    print()
    print("[Meta]")
    for key in [
        "gpu_backend_enabled",
        "projection_cache_size",
        "batch_total_jobs",
        "batch_num_batches",
        "batch_avg_size",
        "batch_total_accepted",
    ]:
        if key in obj:
            print(f"{key:24s}: {obj[key]}")
    print()

    print("[Score Summary]")
    for k in [
        "wrong_count",
        "correct_count",
        "wrong_mean",
        "wrong_median",
        "wrong_min",
        "wrong_max",
        "correct_mean",
        "correct_median",
        "correct_min",
        "correct_max",
    ]:
        print(f"{k:24s}: {format_float(summary_scores[k])}")
    print()

    print("[Separation]")
    for k, v in separation.items():
        print(f"{k:24s}: {format_float(v)}")
    print("解释：如果 gap < 0，表示 correct 候选整体分数更小，更靠前。")
    print()

    print("[Ranking Quality]")
    print(f"{'num_groups':24s}: {rank_summary['num_groups']}")
    print(f"{'best_correct_rank_mean':24s}: {format_float(rank_summary['best_correct_rank_mean'])}")
    print(f"{'best_correct_rank_median':24s}: {format_float(rank_summary['best_correct_rank_median'])}")
    print(f"{'best_correct_rank_min':24s}: {format_float(rank_summary['best_correct_rank_min'])}")
    print(f"{'best_correct_rank_max':24s}: {format_float(rank_summary['best_correct_rank_max'])}")
    print(f"{'avg_correct_rank_mean':24s}: {format_float(rank_summary['avg_correct_rank_mean'])}")
    print(f"{'avg_correct_rank_median':24s}: {format_float(rank_summary['avg_correct_rank_median'])}")
    print()

    print("[Top-k Hit Rate]")
    for k in topk_list:
        hit = rank_summary["topk_hits"][k]
        rate = rank_summary["topk_hit_rate"][k]
        print(f"top-{k:<2d} hit groups          : {hit}/{rank_summary['num_groups']} ({rate:.4f})")
    print()

    print("[Per-group Best Rank Preview]")
    preview = rank_summary["per_group"][:10]
    for row in preview:
        print(
            f"lat={row['lat_index']}, exp={row['exp_index']}, "
            f"best_correct_rank={row['best_correct_rank']}, "
            f"avg_correct_rank={format_float(row['avg_correct_rank'])}"
        )

    if args.export_prefix:
        raw_csv = f"{args.export_prefix}_raw.csv"
        group_csv = f"{args.export_prefix}_per_group.csv"
        write_csv(raw_csv, all_rows)
        write_csv(group_csv, rank_summary["per_group"])
        print()
        print("[Export]")
        print(f"raw rows exported to      : {raw_csv}")
        print(f"per-group rows exported to: {group_csv}")


if __name__ == "__main__":
    main()