"""L2 orchestrator: writer <-> reviewer state machine.

Runs one sample end-to-end:
  set_active_pattern(init) -> render -> [writer cycle -> render -> reviewer] x N
until one of:
  - reviewer returns "ok"
  - n_review_rounds >= MAX_REVIEW_ROUNDS
  - submits_without_change >= NO_CHANGE_K

Persists per-sample artifacts under `out_dir/`:
  initial.json, final.json, trajectory.jsonl, writer_session.log,
  rounds/round_NNN_render.png, rounds/round_NNN_verdict.json
"""

from __future__ import annotations

import base64
import copy
import json
import os
import time

from utils.l1.validator import get_current_pattern, set_active_pattern
from utils.l2 import render as render_mod
from utils.l2.reviewer import ReviewerSession
from utils.l2.writer import WriterSession, render_log_for_human


MAX_REVIEW_ROUNDS = 8
MAX_TOOL_CALLS_PER_CYCLE = 8           # writer cap on normal fix-cycles
MAX_TOOL_CALLS_AFTER_REDO = 32         # lifted cap when last verdict was 'redo' — needs room for structural rebuild
MAX_INNER_TURNS = 8                     # writer backstop: max LLM round-trips per cycle
NO_CHANGE_K = 2


def _encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_one_sample(
    sample_id: str,
    target_img_path: str,
    init_state: dict,
    out_dir: str,
    writer: WriterSession,
    reviewer: ReviewerSession,
    max_review_rounds: int = MAX_REVIEW_ROUNDS,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_CYCLE,
    max_inner_turns: int = MAX_INNER_TURNS,
    no_change_k: int = NO_CHANGE_K,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rounds_dir = os.path.join(out_dir, "rounds")
    os.makedirs(rounds_dir, exist_ok=True)

    with open(os.path.join(out_dir, "initial.json"), "w", encoding="utf-8") as f:
        json.dump(init_state, f, indent=2, ensure_ascii=False)

    set_active_pattern(init_state)
    writer.reset_for_sample()

    target_b64 = _encode_image_b64(target_img_path)

    initial_render = render_mod.render_state(
        copy.deepcopy(get_current_pattern() or {}),
        os.path.join(rounds_dir, "round_000_pre_render.png"),
        debug_labels=True,
    )
    current_render_b64 = initial_render.get("b64")
    current_render_err = initial_render.get("error")

    n_review_rounds = 0
    submits_without_change = 0
    total_inner_turns = 0
    total_tool_calls = 0
    total_tool_errors = 0
    termination_reason: str | None = None
    reviewer_note: dict | None = None
    last_verdict: str | None = None
    t0 = time.time()

    while True:
        cycle_cap = MAX_TOOL_CALLS_AFTER_REDO if last_verdict == "redo" else max_tool_calls
        writer_result = writer.run_until_submit_or_cap(
            target_b64=target_b64,
            render_b64=current_render_b64,
            render_error=current_render_err,
            reviewer_note=reviewer_note,
            max_tool_calls=cycle_cap,
            max_inner_turns=max_inner_turns,
        )
        total_inner_turns += writer_result["n_inner_turns"]
        total_tool_calls += writer_result["n_tool_calls"]
        total_tool_errors += writer_result["n_errors"]
        if writer_result["state_changed"]:
            submits_without_change = 0
        else:
            submits_without_change += 1

        round_idx = n_review_rounds
        render_path = os.path.join(rounds_dir, f"round_{round_idx:03d}_render.png")
        rr = render_mod.render_state(copy.deepcopy(get_current_pattern() or {}), render_path, debug_labels=True)

        if not rr["ok"]:
            err = rr.get("error") or "unknown render failure"
            writer.trajectory.append({
                "kind": "error",
                "turn": writer._turn_counter,
                "text": f"render failed: {err}",
            })
            verdict_record = {
                "verdict": "fix",
                "fixes": [{"panel": "", "severity": "high", "instruction": f"current state did not render: {err}. Revert the last edits that broke it."}],
                "raw": "",
                "hallucinations": [],
                "skipped_review": True,
                "render_error": err,
            }
            current_render_b64 = None
            current_render_err = err
        else:
            verdict_record = reviewer.review(
                target_b64=target_b64,
                render_b64=rr["b64"],
                current_state=copy.deepcopy(get_current_pattern() or {}),
            )
            verdict_record["skipped_review"] = False
            current_render_b64 = rr["b64"]
            current_render_err = None

        with open(os.path.join(rounds_dir, f"round_{round_idx:03d}_verdict.json"), "w", encoding="utf-8") as f:
            json.dump(verdict_record, f, indent=2, ensure_ascii=False)
        with open(os.path.join(rounds_dir, f"round_{round_idx:03d}_writer.json"), "w", encoding="utf-8") as f:
            json.dump(writer_result, f, indent=2, ensure_ascii=False)

        writer.trajectory.append({
            "kind": "review",
            "turn": writer._turn_counter,
            "text": _summarize_verdict_for_trajectory(verdict_record),
        })

        n_review_rounds += 1

        if verdict_record["verdict"] == "ok":
            termination_reason = "reviewer_ok"
            break
        if n_review_rounds >= max_review_rounds:
            termination_reason = "max_rounds"
            break
        if submits_without_change >= no_change_k:
            termination_reason = "no_change"
            break

        reviewer_note = {"verdict": verdict_record["verdict"], "fixes": verdict_record["fixes"]}
        last_verdict = verdict_record["verdict"]

    elapsed = time.time() - t0
    final_state = copy.deepcopy(get_current_pattern() or {})

    with open(os.path.join(out_dir, "final.json"), "w", encoding="utf-8") as f:
        json.dump(final_state, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "trajectory.jsonl"), "w", encoding="utf-8") as f:
        for entry in writer.trajectory:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "writer_session.log"), "w", encoding="utf-8") as f:
        f.write(render_log_for_human(writer.trajectory))

    return {
        "final_state": final_state,
        "termination_reason": termination_reason or "unknown",
        "n_review_rounds": n_review_rounds,
        "n_writer_inner_turns": total_inner_turns,
        "n_tool_calls": total_tool_calls,
        "n_tool_errors": total_tool_errors,
        "latency_sec": elapsed,
        "initial_render_ok": initial_render["ok"],
    }


def _summarize_verdict_for_trajectory(v: dict) -> str:
    verdict = v["verdict"]
    fixes = v.get("fixes", []) or []
    if v.get("skipped_review"):
        return f"verdict={verdict} (REVIEW SKIPPED — render failed: {v.get('render_error', '?')})"
    if verdict == "ok":
        return "verdict=ok (reviewer accepted)"
    parts = [f"verdict={verdict}, {len(fixes)} fix item(s)"]
    for i, fx in enumerate(fixes, 1):
        parts.append(f"  {i}. [{fx.get('severity', '?')}] {fx.get('panel') or '(generic)'}: {fx.get('instruction', '')}")
    return "\n".join(parts)
