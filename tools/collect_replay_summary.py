#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict


def extract_last_python_dict(text: str) -> Dict[str, Any]:
    candidate = None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        candidate = line
        try:
            obj = ast.literal_eval(line)
        except Exception:
            try:
                obj = eval(line, {"__builtins__": {}}, {"inf": float("inf"), "nan": float("nan")})
            except Exception:
                continue
        if isinstance(obj, dict):
            return obj
    raise RuntimeError(f"No summary dict found. Last candidate: {candidate!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect replay summaries into one JSON file.")
    ap.add_argument("--gpu-log", required=True)
    ap.add_argument("--cpu-log", required=True)
    ap.add_argument("--compare-log", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    gpu_log = Path(args.gpu_log)
    cpu_log = Path(args.cpu_log)
    out = Path(args.output)

    gpu_summary = extract_last_python_dict(gpu_log.read_text(encoding="utf-8"))
    cpu_summary = extract_last_python_dict(cpu_log.read_text(encoding="utf-8"))

    payload = {
        "gpu_log": str(gpu_log),
        "cpu_log": str(cpu_log),
        "gpu_summary": gpu_summary,
        "cpu_summary": cpu_summary,
    }

    if args.compare_log:
        compare_log = Path(args.compare_log)
        payload["compare_log"] = str(compare_log)
        payload["compare_text"] = compare_log.read_text(encoding="utf-8")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
