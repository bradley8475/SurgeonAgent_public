"""B1 baseline driver: target image + retrieved init JSON → claude single-shot → corrected JSON.

用法：
  DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run python utils/eval/run_b1.py [--limit N] [--samples ID1,ID2,...]

输出在 results/b1_full_rewrite/<timestamp>/
"""

from __future__ import annotations

import argparse
import base64
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

from agent.llm import LLM
from cad.api import load_pattern
from config.load_config import load_config
from utils.eval.metrics import (
    panel_iou_metrics,
    parse_json_robust,
    pass_loose,
    render_for_inspection,
    stitch_metrics,
    structural_metrics,
    try_load_pattern,
)
from utils.vector_index import ClipFaissIndex


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_META_PATH = os.path.join(PROJECT_ROOT, "splits", "test_meta.json")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "b1_baseline.yaml")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "b1_full_rewrite")


def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


USER_PARTS = [
    ("text", "**Target image — this is what you must reproduce:**"),
    ("image", "<TARGET_IMG>"),
    ("text", "**Retrieved reference — the rendering of the initial pattern.json below:**"),
    ("image", "<INIT_IMG>"),
    ("text", "**Initial pattern.json (corresponds to the retrieved image — modify it to match the target):**\n```json\n<INIT_JSON>\n```\n\nCompare the two images, identify what differs, and output the corrected pattern.json."),
]


def build_user_message(target_img_b64: str, init_img_b64: str, init_json_text: str) -> dict:
    content = []
    for kind, payload in USER_PARTS:
        if kind == "text":
            content.append({"type": "text", "text": payload.replace("<INIT_JSON>", init_json_text)})
        else:
            b64 = target_img_b64 if payload == "<TARGET_IMG>" else init_img_b64
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return {"role": "user", "content": content}


def dump_prompt_human_readable(
    out_path: str,
    system_prompt: str,
    target_img_path: str,
    init_img_path: str,
    init_json_text: str,
) -> None:
    """Write a human-readable rendering of the prompt to a single text file.
    Images are shown as path references; full text is verbatim."""
    lines = []
    lines.append("=" * 80)
    lines.append("SYSTEM PROMPT")
    lines.append("=" * 80)
    lines.append(system_prompt)
    lines.append("")
    lines.append("=" * 80)
    lines.append("USER MESSAGE")
    lines.append("=" * 80)
    for kind, payload in USER_PARTS:
        if kind == "text":
            lines.append("[text]")
            lines.append(payload.replace("<INIT_JSON>", init_json_text))
        else:
            img_path = target_img_path if payload == "<TARGET_IMG>" else init_img_path
            lines.append(f"[image] {img_path}")
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def per_sample_dir(run_dir: str, sample_id: str) -> str:
    d = os.path.join(run_dir, "per_sample", sample_id)
    os.makedirs(d, exist_ok=True)
    return d


def run_one_sample(
    test_entry: dict,
    index: ClipFaissIndex,
    cfg,
    llm_cfg,
    run_dir: str,
) -> dict:
    """Process one test sample. Returns metrics dict."""
    gt_path = test_entry["path"]
    sample_id = os.path.basename(os.path.dirname(gt_path))
    bin_name = test_entry["bin"]
    n_panels = test_entry["n_panels"]
    sd = per_sample_dir(run_dir, sample_id)
    metrics: dict = {"sample_id": sample_id, "bin": bin_name, "n_panels": n_panels, "gt_path": gt_path}
    t_start = time.time()

    target_img = os.path.join(os.path.dirname(gt_path), "panel_stitch.png")
    shutil.copy(target_img, os.path.join(sd, "target.png"))

    # Retrieval
    try:
        with Image.open(target_img) as im:
            emb = index.encode_image(im)
        sims, idxs = index.index.search(emb, 1)
        top_idx = int(idxs[0][0])
        top_sim = float(sims[0][0])
        init_json_path = index.ids[top_idx]
        with open(init_json_path) as f:
            init_data = json.load(f)
        init_json_text = json.dumps(init_data, indent=2, ensure_ascii=False)
        with open(os.path.join(sd, "init.json"), "w") as f:
            f.write(init_json_text)
        metrics["retrieval"] = {
            "init_path": init_json_path,
            "top1_sim": top_sim,
        }
    except Exception as e:
        metrics["error"] = f"retrieval fail: {e}"
        with open(os.path.join(sd, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return metrics

    # LLM call (per-sample fresh LLM instance — cheap)
    try:
        llm = LLM("b1_baseline", llm_cfg, cfg)
        target_b64 = encode_image_b64(target_img)
        init_img_path = os.path.join(os.path.dirname(init_json_path), "panel_stitch.png")
        init_b64 = encode_image_b64(init_img_path)
        shutil.copy(init_img_path, os.path.join(sd, "retrieved.png"))
        user_msg = build_user_message(target_b64, init_b64, init_json_text)
        # Save human-readable prompt for inspection
        dump_prompt_human_readable(
            os.path.join(sd, "prompt.txt"),
            llm.system_prompt,
            target_img,
            init_img_path,
            init_json_text,
        )
        response = llm.generate_response([user_msg])
        raw_text = response.content if isinstance(response.content, str) else str(response.content)
        with open(os.path.join(sd, "raw_response.txt"), "w") as f:
            f.write(raw_text or "")
    except Exception as e:
        metrics["error"] = f"llm fail: {e}"
        metrics["traceback"] = traceback.format_exc()
        with open(os.path.join(sd, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return metrics

    metrics["latency_sec"] = time.time() - t_start
    metrics["raw_chars"] = len(raw_text or "")

    # Parse + load
    parsed = parse_json_robust(raw_text)
    metrics["parsed"] = parsed is not None
    if parsed is not None:
        with open(os.path.join(sd, "output.json"), "w") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
    out_pattern = try_load_pattern(parsed)

    # Load GT
    gt_pattern = load_pattern(gt_path)

    # Structural metrics
    sm = structural_metrics(out_pattern, gt_pattern)
    metrics["structural"] = sm

    # Per-panel IoU + stitch matching + render images for inspection (only if loadable)
    if sm["loadable"]:
        iou_metrics = panel_iou_metrics(out_pattern, gt_pattern)
        metrics["panel_iou"] = iou_metrics
        st_metrics = stitch_metrics(out_pattern, gt_pattern)
        metrics["stitch"] = st_metrics
        metrics["render"] = render_for_inspection(out_pattern, gt_pattern, sd)
        mean_iou = iou_metrics["mean_iou_over_union"]
        stitch_f1 = st_metrics["f1"] if st_metrics["f1"] == st_metrics["f1"] else 0.0  # nan-safe
    else:
        mean_iou = 0.0
        stitch_f1 = 0.0
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
        stitch_f1s = [x if x == x else 0.0 for x in stitch_f1s]  # nan-safe
        recalls = [m.get("structural", {}).get("panel_name_recall", 0.0) for m in mset if m.get("structural", {}).get("loadable", False)]
        latencies = [m.get("latency_sec", 0.0) for m in mset if m.get("latency_sec")]
        sims = [m.get("retrieval", {}).get("top1_sim", 0.0) for m in mset if m.get("retrieval")]
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
        }

    return {
        "overall": agg(all_metrics),
        "by_bin": {k: agg(v) for k, v in sorted(bins.items())},
    }


def write_summary_md(summary: dict, path: str) -> None:
    lines = ["# B1 Baseline — Summary", ""]
    o = summary.get("overall", {})
    lines.append(f"**N = {o.get('n', 0)}** samples")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k in ["parse_rate", "load_rate", "pass_loose_rate",
              "mean_panel_iou", "mean_stitch_f1", "mean_panel_recall",
              "mean_latency_sec", "mean_top1_retrieval_sim"]:  # noqa
        v = o.get(k, 0.0)
        lines.append(f"| {k} | {v:.3f} |")
    lines.append("")
    lines.append("## By bin")
    lines.append("")
    bins = summary.get("by_bin", {})
    headers = ["bin", "n", "parse", "load", "pass_loose", "panel_iou", "stitch_f1", "recall", "latency", "sim"]
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
            f"{b.get('mean_panel_recall', 0):.3f}",
            f"{b.get('mean_latency_sec', 0):.1f}s",
            f"{b.get('mean_top1_retrieval_sim', 0):.3f}",
        ]) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个 test 样本")
    parser.add_argument("--samples", type=str, default=None, help="逗号分隔的 sample_id（基于目录名），跳过其他")
    parser.add_argument("--config", type=str, default=CONFIG_PATH,
                        help="baseline yaml 路径，默认 config/b1_baseline.yaml")
    parser.add_argument("--tag", type=str, default=None,
                        help="run_dir / logging path 后缀（多进程并发时区分）")
    args = parser.parse_args()

    dotenv.load_dotenv()
    cfg = load_config(args.config)
    llm_cfg = cfg.agents.b1_baseline

    with open(TEST_META_PATH) as f:
        test_meta = json.load(f)

    if args.samples:
        wanted = set(args.samples.split(","))
        test_meta = [t for t in test_meta if os.path.basename(os.path.dirname(t["path"])) in wanted]
    elif args.limit:
        test_meta = test_meta[: args.limit]

    print(f"loading retrieval index...")
    index = ClipFaissIndex()
    index.load()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.tag:
        timestamp = f"{timestamp}_{args.tag}"
        # 给 LLM logging 子目录也加 tag，避免 4 进程的 input/output JSON 混在同一目录
        cfg.logging.path = os.path.join(cfg.logging.path, args.tag)
    run_dir = os.path.join(RESULTS_ROOT, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"run dir: {run_dir}")
    print(f"running on {len(test_meta)} samples...")

    all_metrics = []
    t0 = time.time()
    for i, t in enumerate(test_meta, 1):
        sample_id = os.path.basename(os.path.dirname(t["path"]))
        print(f"  [{i}/{len(test_meta)}] {sample_id} (bin={t['bin']}, n_panels={t['n_panels']})", flush=True)
        try:
            m = run_one_sample(t, index, cfg, llm_cfg, run_dir)
        except Exception as e:
            m = {"sample_id": sample_id, "bin": t["bin"], "n_panels": t["n_panels"], "error": f"unhandled: {e}", "traceback": traceback.format_exc()}
        all_metrics.append(m)
        ok = m.get("pass_loose", False)
        iou = m.get("panel_iou", {}).get("mean_iou_over_union", 0.0) if isinstance(m.get("panel_iou"), dict) else 0.0
        print(f"     parse={m.get('parsed', False)} load={m.get('structural', {}).get('loadable', False)} pass_loose={ok} iou={iou:.3f}")

    summary = aggregate(all_metrics)
    summary["run_dir"] = run_dir
    summary["timestamp"] = timestamp
    summary["elapsed_sec"] = time.time() - t0

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_summary_md(summary, os.path.join(run_dir, "summary.md"))
    with open(os.path.join(run_dir, "all_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n=== done in {time.time()-t0:.1f}s ===")
    print(f"results: {run_dir}")
    print(f"\noverall:")
    for k, v in summary["overall"].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
