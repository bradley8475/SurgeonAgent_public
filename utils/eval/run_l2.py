"""L2 driver: target image + retrieved init JSON -> writer/reviewer loop -> corrected JSON.

Mirrors `run_b1.py` shape (per-sample dir, by-bin aggregation, summary.md). The
LLM block is replaced with a writer/reviewer state machine from `utils.l2.loop`.

Usage:
  DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run python utils/eval/run_l2.py \
      [--limit N] [--samples ID1,ID2,...]

Output: results/l2_writer_reviewer/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime

import dotenv
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cad.api import load_pattern
from config.load_config import load_config
from utils.eval.metrics import (
    panel_iou_metrics,
    pass_loose,
    render_for_inspection,
    stitch_metrics,
    structural_metrics,
    try_load_pattern,
)
from utils.l2 import render as render_mod
from utils.l2.loop import run_one_sample
from utils.l2.reviewer import ReviewerSession
from utils.l2.writer import WriterSession
from utils.l2.writer_notools import WriterSessionNoTools
from utils.vector_index import ClipFaissIndex


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_META_PATH = os.path.join(PROJECT_ROOT, "splits", "test_meta.json")
WRITER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "l2_writer.yaml")
WRITER_NOTOOLS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "l2_writer_notools.yaml")
REVIEWER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "l2_reviewer.yaml")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "l2_writer_reviewer")
RESULTS_ROOT_NOTOOLS = os.path.join(PROJECT_ROOT, "results", "l2_writer_reviewer_notools")


def per_sample_dir(run_dir: str, sample_id: str) -> str:
    d = os.path.join(run_dir, "per_sample", sample_id)
    os.makedirs(d, exist_ok=True)
    return d


def run_one_test_sample(
    test_entry: dict,
    index: ClipFaissIndex,
    run_dir: str,
    writer: WriterSession,
    reviewer: ReviewerSession,
) -> dict:
    gt_path = test_entry["path"]
    sample_id = os.path.basename(os.path.dirname(gt_path))
    bin_name = test_entry["bin"]
    n_panels = test_entry["n_panels"]
    sd = per_sample_dir(run_dir, sample_id)
    metrics: dict = {"sample_id": sample_id, "bin": bin_name, "n_panels": n_panels, "gt_path": gt_path}
    t_start = time.time()

    dataset_target_img = os.path.join(os.path.dirname(gt_path), "panel_stitch.png")
    sd_target_png = os.path.join(sd, "target.png")
    sd_retrieved_png = os.path.join(sd, "retrieved.png")

    try:
        with open(gt_path) as f:
            gt_data = json.load(f)
        rr = render_mod.render_state(gt_data, sd_target_png)
        if not rr["ok"]:
            shutil.copy(dataset_target_img, sd_target_png)
            metrics["target_render_fallback"] = rr.get("error")
    except Exception as e:
        shutil.copy(dataset_target_img, sd_target_png)
        metrics["target_render_fallback"] = f"{type(e).__name__}: {e}"

    try:
        with Image.open(dataset_target_img) as im:
            emb = index.encode_image(im)
        sims, idxs = index.index.search(emb, 1)
        top_idx = int(idxs[0][0])
        top_sim = float(sims[0][0])
        init_json_path = index.ids[top_idx]
        with open(init_json_path) as f:
            init_data = json.load(f)
        rr_init = render_mod.render_state(init_data, sd_retrieved_png)
        if not rr_init["ok"]:
            init_dataset_png = os.path.join(os.path.dirname(init_json_path), "panel_stitch.png")
            shutil.copy(init_dataset_png, sd_retrieved_png)
            metrics["retrieved_render_fallback"] = rr_init.get("error")
        metrics["retrieval"] = {"init_path": init_json_path, "top1_sim": top_sim}
    except Exception as e:
        metrics["error"] = f"retrieval fail: {e}"
        with open(os.path.join(sd, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return metrics

    try:
        loop_result = run_one_sample(
            sample_id=sample_id,
            target_img_path=os.path.join(sd, "target.png"),
            init_state=init_data,
            out_dir=sd,
            writer=writer,
            reviewer=reviewer,
        )
    except Exception as e:
        metrics["error"] = f"loop fail: {e}"
        metrics["traceback"] = traceback.format_exc()
        with open(os.path.join(sd, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return metrics

    metrics["latency_sec"] = time.time() - t_start
    metrics["termination_reason"] = loop_result["termination_reason"]
    metrics["n_review_rounds"] = loop_result["n_review_rounds"]
    metrics["n_writer_inner_turns"] = loop_result["n_writer_inner_turns"]
    metrics["n_tool_calls"] = loop_result["n_tool_calls"]
    metrics["n_tool_errors"] = loop_result["n_tool_errors"]
    metrics["initial_render_ok"] = loop_result["initial_render_ok"]

    final_state = loop_result["final_state"]
    metrics["parsed"] = final_state is not None
    out_pattern = try_load_pattern(final_state)

    gt_pattern = load_pattern(gt_path)

    sm = structural_metrics(out_pattern, gt_pattern)
    metrics["structural"] = sm

    if sm["loadable"]:
        iou_metrics = panel_iou_metrics(out_pattern, gt_pattern)
        metrics["panel_iou"] = iou_metrics
        st_metrics = stitch_metrics(out_pattern, gt_pattern)
        metrics["stitch"] = st_metrics
        metrics["render"] = render_for_inspection(out_pattern, gt_pattern, sd)
    else:
        metrics["panel_iou"] = {"mean_iou_over_union": 0.0, "skipped": "not loadable"}
        metrics["stitch"] = {"f1": 0.0, "skipped": "not loadable"}

    metrics["pass_loose"] = pass_loose(sm)

    with open(os.path.join(sd, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def aggregate(all_metrics: list[dict]) -> dict:
    bins = defaultdict(list)
    for m in all_metrics:
        bins[m.get("bin", "unknown")].append(m)

    def agg(mset: list[dict]) -> dict:
        n = len(mset)
        if n == 0:
            return {}
        loadable = sum(1 for m in mset if m.get("structural", {}).get("loadable", False))
        parsed = sum(1 for m in mset if m.get("parsed", False))
        loose = sum(1 for m in mset if m.get("pass_loose", False))
        ious = [m.get("panel_iou", {}).get("mean_iou_over_union", 0.0) for m in mset if m.get("structural", {}).get("loadable", False)]
        stitch_f1s = [m.get("stitch", {}).get("f1", 0.0) for m in mset if m.get("structural", {}).get("loadable", False)]
        stitch_f1s = [x if x == x else 0.0 for x in stitch_f1s]
        recalls = [m.get("structural", {}).get("panel_name_recall", 0.0) for m in mset if m.get("structural", {}).get("loadable", False)]
        latencies = [m.get("latency_sec", 0.0) for m in mset if m.get("latency_sec")]
        sims = [m.get("retrieval", {}).get("top1_sim", 0.0) for m in mset if m.get("retrieval")]
        review_rounds = [m.get("n_review_rounds", 0) for m in mset if "n_review_rounds" in m]
        tool_calls = [m.get("n_tool_calls", 0) for m in mset if "n_tool_calls" in m]
        tool_errors = [m.get("n_tool_errors", 0) for m in mset if "n_tool_errors" in m]
        term_counts: dict = defaultdict(int)
        for m in mset:
            tr = m.get("termination_reason")
            if tr:
                term_counts[tr] += 1
        return {
            "n": n,
            "parse_rate": parsed / n,
            "load_rate": loadable / n,
            "pass_loose_rate": loose / n,
            "mean_panel_iou": (sum(ious) / len(ious)) if ious else 0.0,
            "mean_stitch_f1": (sum(stitch_f1s) / len(stitch_f1s)) if stitch_f1s else 0.0,
            "mean_panel_recall": (sum(recalls) / len(recalls)) if recalls else 0.0,
            "mean_latency_sec": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "mean_top1_retrieval_sim": (sum(sims) / len(sims)) if sims else 0.0,
            "mean_review_rounds": (sum(review_rounds) / len(review_rounds)) if review_rounds else 0.0,
            "mean_tool_calls": (sum(tool_calls) / len(tool_calls)) if tool_calls else 0.0,
            "mean_tool_errors": (sum(tool_errors) / len(tool_errors)) if tool_errors else 0.0,
            "termination_counts": dict(term_counts),
        }

    return {
        "overall": agg(all_metrics),
        "by_bin": {k: agg(v) for k, v in sorted(bins.items())},
    }


def write_summary_md(summary: dict, path: str) -> None:
    lines = ["# L2 (writer + reviewer) — Summary", ""]
    o = summary.get("overall", {})
    lines.append(f"**N = {o.get('n', 0)}** samples")
    lines.append("")
    lines.append("## Overall — quality")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k in ["parse_rate", "load_rate", "pass_loose_rate",
              "mean_panel_iou", "mean_stitch_f1", "mean_panel_recall",
              "mean_top1_retrieval_sim"]:
        v = o.get(k, 0.0)
        lines.append(f"| {k} | {v:.3f} |")
    lines.append("")
    lines.append("## Overall — L2 loop counters")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k in ["mean_review_rounds", "mean_tool_calls", "mean_tool_errors", "mean_latency_sec"]:
        v = o.get(k, 0.0)
        lines.append(f"| {k} | {v:.3f} |")
    tc = o.get("termination_counts", {})
    lines.append(f"| termination_counts | {json.dumps(tc)} |")
    lines.append("")
    lines.append("## By bin")
    lines.append("")
    bins = summary.get("by_bin", {})
    headers = ["bin", "n", "parse", "load", "pass", "iou", "stitch_f1", "rounds", "tools", "lat"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for bn in ["easy", "medium", "hard", "very_hard"]:
        b = bins.get(bn, {})
        lines.append("| " + " | ".join([
            bn,
            str(b.get("n", 0)),
            f"{b.get('parse_rate', 0):.2f}",
            f"{b.get('load_rate', 0):.2f}",
            f"{b.get('pass_loose_rate', 0):.2f}",
            f"{b.get('mean_panel_iou', 0):.3f}",
            f"{b.get('mean_stitch_f1', 0):.3f}",
            f"{b.get('mean_review_rounds', 0):.1f}",
            f"{b.get('mean_tool_calls', 0):.1f}",
            f"{b.get('mean_latency_sec', 0):.1f}s",
        ]) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None,
                        help="suffix appended to timestamp for run_dir + logging paths "
                             "(use a distinct value per concurrent process to avoid collisions)")
    parser.add_argument("--no-tools", action="store_true",
                        help="ablation: writer outputs full pattern.json each cycle "
                             "instead of calling L1 tools (reviewer loop unchanged)")
    args = parser.parse_args()

    dotenv.load_dotenv()
    writer_cfg_path = WRITER_NOTOOLS_CONFIG_PATH if args.no_tools else WRITER_CONFIG_PATH
    results_root = RESULTS_ROOT_NOTOOLS if args.no_tools else RESULTS_ROOT
    writer_cfg = load_config(writer_cfg_path)
    reviewer_cfg = load_config(REVIEWER_CONFIG_PATH)

    with open(TEST_META_PATH) as f:
        test_meta = json.load(f)

    if args.samples:
        wanted = set(args.samples.split(","))
        test_meta = [t for t in test_meta if os.path.basename(os.path.dirname(t["path"])) in wanted]
    elif args.limit:
        test_meta = test_meta[: args.limit]

    print(f"loading retrieval index...", flush=True)
    index = ClipFaissIndex()
    index.load()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.tag:
        timestamp = f"{timestamp}_{args.tag}"
    run_dir = os.path.join(results_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"run dir: {run_dir}", flush=True)
    print(f"variant: {'NO-TOOLS (ablation)' if args.no_tools else 'WITH-TOOLS (full L2)'}", flush=True)

    print(f"constructing writer + reviewer sessions...", flush=True)
    WriterCls = WriterSessionNoTools if args.no_tools else WriterSession
    writer = WriterCls(writer_cfg.agents.l2_writer, writer_cfg)
    reviewer = ReviewerSession(reviewer_cfg.agents.l2_reviewer, reviewer_cfg)
    writer.llm.logging_path = os.path.join(writer.llm.logging_path, timestamp)
    reviewer.llm.logging_path = os.path.join(reviewer.llm.logging_path, timestamp)
    os.makedirs(writer.llm.logging_path, exist_ok=True)
    os.makedirs(reviewer.llm.logging_path, exist_ok=True)
    print(f"writer tools: {writer.tool_names}", flush=True)
    print(f"writer llm logs: {writer.llm.logging_path}", flush=True)
    print(f"reviewer llm logs: {reviewer.llm.logging_path}", flush=True)

    print(f"running on {len(test_meta)} samples...", flush=True)
    all_metrics = []
    t0 = time.time()
    for i, t in enumerate(test_meta, 1):
        sample_id = os.path.basename(os.path.dirname(t["path"]))
        print(f"  [{i}/{len(test_meta)}] {sample_id} (bin={t['bin']}, n_panels={t['n_panels']})", flush=True)
        try:
            m = run_one_test_sample(t, index, run_dir, writer, reviewer)
        except Exception as e:
            m = {"sample_id": sample_id, "bin": t["bin"], "n_panels": t["n_panels"], "error": f"unhandled: {e}", "traceback": traceback.format_exc()}
        all_metrics.append(m)
        iou = m.get("panel_iou", {}).get("mean_iou_over_union", 0.0) if isinstance(m.get("panel_iou"), dict) else 0.0
        print(f"     load={m.get('structural', {}).get('loadable', False)} pass={m.get('pass_loose', False)} iou={iou:.3f} term={m.get('termination_reason', '?')} rounds={m.get('n_review_rounds', '?')} tools={m.get('n_tool_calls', '?')}", flush=True)

    summary = aggregate(all_metrics)
    summary["run_dir"] = run_dir
    summary["timestamp"] = timestamp
    summary["elapsed_sec"] = time.time() - t0

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_summary_md(summary, os.path.join(run_dir, "summary.md"))
    with open(os.path.join(run_dir, "all_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n=== done in {time.time()-t0:.1f}s ===", flush=True)
    print(f"results: {run_dir}", flush=True)
    print(f"\noverall:", flush=True)
    for k, v in summary["overall"].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
