"""L1 driver: incremental, diff-driven derivation of a typed Python tool API.

For each (A, B) pair sampled from train, the L1 designer LLM sees the cumulative toolset
and decides whether to reuse existing tools or write/update/delete tools, finishing with
submit_solution. The cumulative registry persists across pairs.

Stops when either:
- max_pairs reached
- last `saturation_window` pairs all added zero new tools

Usage:
  DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run python utils/l1/derive.py [--max-pairs N] [--saturation-window K] [--seed S]

Output: results/l1_derive/<timestamp>/{pair_<i>/tools,sequence.py,history.json}, final_tools/, summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime

import dotenv
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.llm import LLM
from config.load_config import load_config
from utils.l1 import derive_tools  # noqa: F401 — registers meta-tools as side effect
from utils.l1.derive_tools import (
    get_pair_history,
    get_registry_snapshot,
    is_submitted,
    render_toolset_for_prompt,
    set_active,
)
from utils.vector_index import ClipFaissIndex


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_TXT = os.path.join(PROJECT_ROOT, "splits", "train.txt")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "l1_designer.yaml")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "l1_derive")


def sample_pairs(n_pairs: int, seed: int) -> list[tuple[str, str, float]]:
    """Sample n_pairs (A, B) using CLIP retrieval — same setup as B1 / L2.

    For each anchor sample (the target B), retrieve the visually most similar train sample
    (the init A) via the existing ClipFaissIndex. This mirrors L2's runtime: at inference,
    L2 is given a target image + a retrieved init JSON. Training L1 on the same diff
    distribution makes the derived tools immediately useful for L2.

    Returns list of (A_json_path, B_json_path, retrieval_similarity).
    """
    index = ClipFaissIndex()
    index.load()
    indexed = list(index.ids)
    if len(indexed) < n_pairs + 1:
        raise ValueError(f"index has only {len(indexed)} entries; can't sample {n_pairs} pairs")

    rng = random.Random(seed)
    anchors = rng.sample(indexed, n_pairs)

    pairs: list[tuple[str, str, float]] = []
    for anchor_json in anchors:
        anchor_img = os.path.join(os.path.dirname(anchor_json), "panel_stitch.png")
        if not os.path.exists(anchor_img):
            continue
        with Image.open(anchor_img) as im:
            emb = index.encode_image(im)
        sims, idxs = index.index.search(emb, 2)
        # rank 0 is the anchor itself (retrieving from a pool that contains it); take rank 1
        ret_idx = int(idxs[0][1])
        ret_sim = float(sims[0][1])
        ret_json = index.ids[ret_idx]
        pairs.append((ret_json, anchor_json, ret_sim))
    if len(pairs) < n_pairs:
        raise ValueError(f"only built {len(pairs)}/{n_pairs} pairs (some anchors had no panel_stitch.png)")
    return pairs


def build_user_message(A: dict, B: dict) -> dict:
    toolset = render_toolset_for_prompt()
    body = (
        "## Current toolset\n\n"
        f"{toolset}\n\n"
        "## Pattern A (starting state)\n\n"
        "```json\n"
        f"{json.dumps(A, indent=2, ensure_ascii=False)}\n"
        "```\n\n"
        "## Pattern B (target state)\n\n"
        "```json\n"
        f"{json.dumps(B, indent=2, ensure_ascii=False)}\n"
        "```\n\n"
        "Express the transformation A → B. Reuse existing tools where possible; "
        "call `write_tool` / `update_tool` / `delete_tool` only when necessary. "
        "Finish by calling `submit_solution(call_sequence)`."
    )
    return {"role": "user", "content": body}


def run_one_pair(
    llm: LLM,
    pair_idx: int,
    A_path: str,
    B_path: str,
    run_dir: str,
) -> dict:
    pair_id = f"pair_{pair_idx:03d}"
    set_active(run_dir, pair_id)
    llm.reset_context()

    with open(A_path) as f:
        A = json.load(f)
    with open(B_path) as f:
        B = json.load(f)

    user_msg = build_user_message(A, B)
    print(f"[derive] -> calling LLM (user msg {len(user_msg['content'])} chars, may take 30-120s)...", flush=True)
    t0 = time.time()
    error = None
    try:
        llm.run([user_msg], max_rounds=30)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    print(f"[derive] <- LLM run done in {time.time() - t0:.1f}s", flush=True)

    elapsed = time.time() - t0
    history = get_pair_history()
    n_writes = sum(1 for op, _ in history if op == "write")
    n_updates = sum(1 for op, _ in history if op == "update")
    n_deletes = sum(1 for op, _ in history if op == "delete")
    return {
        "pair_id": pair_id,
        "A_path": A_path,
        "B_path": B_path,
        "submitted": is_submitted(),
        "new_tools": n_writes,
        "updates": n_updates,
        "deletes": n_deletes,
        "history": history,
        "registry_size_after": len(get_registry_snapshot()),
        "elapsed_sec": elapsed,
        "error": error,
    }


def write_final_snapshot(run_dir: str) -> str:
    """Dump the final cumulative registry as a single composite file at <run_dir>/final_tools.py."""
    from utils.l1.derive_tools import compose_file
    final_path = os.path.join(run_dir, "final_tools.py")
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(compose_file(get_registry_snapshot()))
    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--saturation-window", type=int, default=3, help="early stop if last K pairs added zero new tools")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-after", type=int, default=0,
                        help="if > 0: after derive, run validator on N hold-out pairs and write results to <run_dir>/validation/")
    args = parser.parse_args()

    dotenv.load_dotenv()
    cfg = load_config(CONFIG_PATH)
    llm_cfg = cfg.agents.l1_designer

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_ROOT, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"run dir: {run_dir}")

    llm = LLM("l1_designer", llm_cfg, cfg)
    # per-run log subdir so different derive runs don't pile into the same dir
    llm.logging_path = os.path.join(cfg.logging.path, "l1_designer", timestamp)
    os.makedirs(llm.logging_path, exist_ok=True)
    print(f"log dir: {llm.logging_path}")

    pairs = sample_pairs(args.max_pairs, args.seed)
    print(f"sampled {len(pairs)} pairs (seed={args.seed}, retrieval-based)")
    for i, (a, b, sim) in enumerate(pairs):
        print(f"  [{i}] sim={sim:.3f}  A={os.path.basename(os.path.dirname(a))}  B={os.path.basename(os.path.dirname(b))}")

    results = []
    recent_new = []
    t_start = time.time()
    stop_reason = "max_pairs"
    for i, (A_path, B_path, sim) in enumerate(pairs):
        print(f"\n[{i + 1}/{len(pairs)}] sim={sim:.3f}  A={os.path.basename(os.path.dirname(A_path))}  B={os.path.basename(os.path.dirname(B_path))}", flush=True)
        result = run_one_pair(llm, i, A_path, B_path, run_dir)
        result["retrieval_sim"] = sim
        results.append(result)
        print(
            f"  submitted={result['submitted']} "
            f"new={result['new_tools']} upd={result['updates']} del={result['deletes']} "
            f"reg_size={result['registry_size_after']} "
            f"elapsed={result['elapsed_sec']:.1f}s "
            f"error={result['error']}"
        )

        recent_new.append(result["new_tools"])
        if len(recent_new) > args.saturation_window:
            recent_new.pop(0)
        if len(recent_new) >= args.saturation_window and sum(recent_new) == 0:
            stop_reason = f"saturated (last {args.saturation_window} pairs added 0 new tools)"
            print(f"\n{stop_reason}; stopping early")
            break

    final_path = write_final_snapshot(run_dir)

    summary = {
        "timestamp": timestamp,
        "run_dir": run_dir,
        "final_tools_path": final_path,
        "stop_reason": stop_reason,
        "n_pairs_run": len(results),
        "final_registry_size": len(get_registry_snapshot()),
        "total_elapsed_sec": time.time() - t_start,
        "args": vars(args),
        "pairs": results,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== derive done ===")
    print(f"final tools: {summary['final_registry_size']} -> {final_path}")
    print(f"stop_reason: {stop_reason}")
    print(f"summary: {os.path.join(run_dir, 'summary.json')}")

    if args.validate_after > 0:
        print(f"\n=== chaining validator on {args.validate_after} hold-out pairs ===")
        from utils.l1.validator import run_validation
        val_dir = os.path.join(run_dir, "validation")
        os.makedirs(val_dir, exist_ok=True)
        # use a different seed so hold-out pairs don't overlap with derive pairs
        val_summary = run_validation(
            final_tools_path=final_path,
            n_pairs=args.validate_after,
            seed=args.seed + 1000,
            out_dir=val_dir,
        )
        summary["validation"] = {
            "n_pairs": val_summary["n_pairs"],
            "n_pass": val_summary["n_pass"],
            "completeness_rate": val_summary["completeness_rate"],
            "elapsed_sec": val_summary["elapsed_sec"],
            "out_dir": val_dir,
        }
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
