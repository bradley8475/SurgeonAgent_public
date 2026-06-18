"""把 N 个独立 L2 run_dir 的 all_metrics.json 合到一个总 summary。

用法:
  uv run python scripts/aggregate_l2_runs.py <run_dir1> <run_dir2> ...
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.eval.run_l2 import aggregate, write_summary_md


def main() -> None:
    run_dirs = sys.argv[1:]
    if not run_dirs:
        print("usage: aggregate_l2_runs.py <run_dir1> <run_dir2> ...", file=sys.stderr)
        sys.exit(2)

    all_metrics: list[dict] = []
    for d in run_dirs:
        p = os.path.join(d, "all_metrics.json")
        if not os.path.exists(p):
            print(f"[skip] {p} missing", file=sys.stderr)
            continue
        with open(p) as f:
            ms = json.load(f)
        for m in ms:
            m["_source_run_dir"] = d
        all_metrics.extend(ms)

    if not all_metrics:
        print("no metrics found", file=sys.stderr)
        sys.exit(1)

    summary = aggregate(all_metrics)
    summary["source_run_dirs"] = run_dirs
    summary["n_total"] = len(all_metrics)

    out_dir = os.path.join(os.path.dirname(run_dirs[0]), "_aggregated")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "all_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_summary_md(summary, os.path.join(out_dir, "summary.md"))

    print(f"aggregated {len(all_metrics)} samples from {len(run_dirs)} runs")
    print(f"out dir: {out_dir}")
    print()
    for k, v in summary["overall"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
