"""L1 validator: agentic reconstruction completeness on hold-out pairs.

Given a final_tools.py composite (from a derive run), expose each derived tool as
a callable in the agent's tool registry. The transcriber LLM agent calls them
iteratively, seeing the actual evolving state via tool responses, until it
either calls `transcribe_done` or stops emitting tool calls.

Usage:
  uv run python utils/l1/validator.py --final-tools <path> [--n-pairs 5] [--seed 1042]
  (or chained from derive.py via --validate-after N)
"""

from __future__ import annotations

import argparse
import ast
import copy as copy_mod
import functools
import inspect
import json
import os
import sys
import time
import traceback
from datetime import datetime

import dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.llm import LLM
from cad.api import load_pattern
from config.load_config import load_config
from tools.registry import ToolResponse
from tools.tools import tool_registry
from utils.eval.metrics import (
    panel_iou_metrics,
    stitch_metrics,
    structural_metrics,
    try_load_pattern,
)
from utils.l1.derive import sample_pairs


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRANSCRIBER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "l1_transcriber.yaml")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "l1_validate")

IOU_THRESHOLD = 0.95
STITCH_F1_THRESHOLD = 0.95
MAX_AGENT_ROUNDS = 60


# ---------- module state for the validator's agent loop ----------

_state = {
    "pattern": None,        # current dict; mutated by tool wrappers
    "done": False,          # set by transcribe_done
    "tool_call_log": [],    # [(tool_name, args_dict, status)] for audit
}


def set_active_pattern(initial: dict) -> None:
    """Reset module state for a new pair. Call before llm.run on each hold-out pair."""
    _state["pattern"] = copy_mod.deepcopy(initial)
    _state["done"] = False
    _state["tool_call_log"] = []


def get_current_pattern() -> dict | None:
    return _state["pattern"]


def get_tool_call_log() -> list:
    return list(_state["tool_call_log"])


# ---------- compact state summary returned by tool wrappers ----------


def _summarize_state() -> str:
    p = _state["pattern"]
    if not p:
        return "(no pattern set)"
    panels = p.get("pattern", {}).get("panels", {})
    stitches = p.get("pattern", {}).get("stitches", [])
    panel_lines = []
    for name in sorted(panels.keys()):
        n_edges = len(panels[name].get("edges", []))
        panel_lines.append(f"  {name}: {n_edges} edges")
    stitch_lines = []
    for i, s in enumerate(stitches):
        side0 = ",".join(f"{r.get('panel')}.{r.get('edge')}" for r in s.get("0", []))
        side1 = ",".join(f"{r.get('panel')}.{r.get('edge')}" for r in s.get("1", []))
        stitch_lines.append(f"  [{i}] {side0} <-> {side1}")
    return (
        f"panels ({len(panels)}):\n" + ("\n".join(panel_lines) if panel_lines else "  (none)") +
        f"\nstitches ({len(stitches)}):\n" + ("\n".join(stitch_lines) if stitch_lines else "  (none)")
    )


# ---------- wrapping derived tools as agent-callable tools ----------


def _wrap_derived_tool(func, name: str):
    """Wrap a derived tool (which takes `state: dict` as 1st arg) into an agent tool
    that operates on module-level _state["pattern"] and returns a ToolResponse.

    Wrappers return minimal `ok` / `error: ...` lines; the up-to-date pattern is
    re-injected into the user message at the top of every transcriber turn, so the
    tool response itself doesn't need to dump state."""
    sig = inspect.signature(func)
    new_params = [p for pn, p in sig.parameters.items() if pn != "state"]

    @functools.wraps(func)
    def wrapper(**kwargs):
        if _state["pattern"] is None:
            return ToolResponse(content="error: no active pattern (validator bug — set_active_pattern not called)")
        try:
            new_state = func(_state["pattern"], **kwargs)
            _state["pattern"] = new_state
            _state["tool_call_log"].append((name, kwargs, "ok"))
            return ToolResponse(content=f"ok: applied {name}")
        except Exception as e:
            err_msg = f"error: {type(e).__name__}: {e}"
            _state["tool_call_log"].append((name, kwargs, err_msg))
            return ToolResponse(content=err_msg)

    wrapper.__signature__ = sig.replace(parameters=new_params)
    wrapper.__name__ = name
    wrapper.__doc__ = func.__doc__ or f"Apply derived tool `{name}`."
    wrapper.__annotations__ = {k: v for k, v in func.__annotations__.items() if k != "state"}
    return wrapper


def register_derived_tools(tools_source: str) -> list[str]:
    """Exec tools_source, register each top-level function as an agent tool, return names."""
    namespace: dict = {}
    exec(tools_source, namespace)

    # Determine which names are derived tools (skip imports / helper classes)
    tree = ast.parse(tools_source)
    func_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]

    registered: list[str] = []
    for name in func_names:
        if name.startswith("_"):
            continue  # private helper, not an agent-exposed tool
        if name not in namespace or not callable(namespace[name]):
            continue
        wrapper = _wrap_derived_tool(namespace[name], name)
        tool_registry.register(wrapper)
        registered.append(name)
    return registered


# ---------- meta-tools for the transcriber ----------


@tool_registry.register
def transcribe_done(notes: str = "") -> ToolResponse:
    """Signal that you are done transforming A toward B. The validator will compare the current state to target B.

    Call this when you've applied all the edits you can, or when you decide further edits aren't possible
    with the available tools (e.g. only out-of-scope edits remain). `notes` is optional free-form text
    explaining what you did and what you couldn't do.
    """
    _state["done"] = True
    return ToolResponse(
        content=f"acknowledged. Validator will now compare current state to target B. Stop calling tools."
    )


# ---------- per-pair validation ----------


def _build_turn_message(target_B: dict, last_action: str) -> dict:
    """Build the user message for one transcriber turn.
    Always includes target B and current state freshly (no accumulated history).
    `last_action` summarizes what the previous turn's tool calls did (or "n/a" on turn 1)."""
    current = _state["pattern"]
    return {
        "role": "user",
        "content": (
            "## Target state (B)\n\n```json\n"
            f"{json.dumps(target_B, indent=2, ensure_ascii=False)}\n"
            "```\n\n"
            "## Current state (live)\n\n```json\n"
            f"{json.dumps(current, indent=2, ensure_ascii=False)}\n"
            "```\n\n"
            f"## Last action result\n{last_action}\n\n"
            "Apply the next tool call(s) to bring current closer to target. When current matches target "
            "(modulo arrangement/uvr), or only out-of-scope edits remain, call `transcribe_done`. "
            "Note: this user message is freshly built each turn — prior tool responses are not preserved, "
            "so trust the current JSON above as the only source of truth."
        ),
    }


def _run_transcriber_loop(llm: LLM, target_B: dict, max_rounds: int) -> tuple[int, int, str | None]:
    """Custom agent loop: per-turn fresh user msg with current+target+last_action.
    Returns (n_rounds_used, n_calls_total, error_msg_or_None)."""
    last_action = "n/a (turn 1 — no actions yet)"
    n_rounds = 0
    total_calls = 0
    while n_rounds < max_rounds:
        n_rounds += 1
        llm.reset_context()  # discard prior turns' history
        try:
            response = llm.generate_response([_build_turn_message(target_B, last_action)])
        except Exception as e:
            traceback.print_exc()
            return n_rounds, total_calls, f"{type(e).__name__}: {e}"

        if not response.tool_calls:
            return n_rounds, total_calls, None

        action_lines = []
        for tc in response.tool_calls:
            tname = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception as e:
                action_lines.append(f"{tname}(<unparseable args>) -> error: {e}")
                continue
            tr = llm.tool_registry.call_tool(tname, args)
            total_calls += 1
            line = tr.content.strip().splitlines()[0] if tr.content else "(no response)"
            arg_preview = ", ".join(f"{k}={v!r}" if isinstance(v, (str, int, float, bool)) else f"{k}=<...>" for k, v in args.items())
            action_lines.append(f"{tname}({arg_preview[:120]}) -> {line}")
            print(f"[validate]   {action_lines[-1]}", flush=True)

        if _state["done"]:
            return n_rounds, total_calls, None

        last_action = "\n".join(action_lines)

    return n_rounds, total_calls, f"max_rounds ({max_rounds}) exceeded"


def validate_one_pair(
    llm: LLM,
    idx: int,
    A_path: str,
    B_path: str,
    sim: float,
    out_dir: str,
) -> dict:
    pair_id = f"pair_{idx:03d}"
    pd = os.path.join(out_dir, pair_id)
    os.makedirs(pd, exist_ok=True)

    with open(A_path) as f:
        A = json.load(f)
    with open(B_path) as f:
        B = json.load(f)

    set_active_pattern(A)

    print(f"[validate] {pair_id} sim={sim:.3f} -> agentic transcribe (per-turn state injection)...", flush=True)
    t0 = time.time()
    n_rounds, total_calls, error = _run_transcriber_loop(llm, B, MAX_AGENT_ROUNDS)
    elapsed = time.time() - t0
    print(f"[validate] {pair_id}    rounds={n_rounds} calls={total_calls} error={error}", flush=True)

    log = get_tool_call_log()
    n_calls = len(log)
    n_ok = sum(1 for _, _, status in log if status == "ok")
    n_err = n_calls - n_ok

    with open(os.path.join(pd, "tool_call_log.json"), "w", encoding="utf-8") as f:
        json.dump([{"tool": t, "args": a, "status": s} for t, a, s in log], f, indent=2, ensure_ascii=False)

    result = get_current_pattern()
    with open(os.path.join(pd, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    metrics: dict = {
        "pair_id": pair_id, "A_path": A_path, "B_path": B_path,
        "retrieval_sim": sim, "transcriber_elapsed_sec": elapsed,
        "agent_rounds_error": error,
        "n_tool_calls": n_calls, "n_ok": n_ok, "n_errors": n_err,
        "submitted_done": _state["done"],
    }

    out_pattern = try_load_pattern(result)
    try:
        gt_pattern = load_pattern(B_path)
    except Exception as e:
        metrics["pass"] = False
        metrics["fail_reason"] = f"GT load failed: {e}"
        with open(os.path.join(pd, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return metrics

    sm = structural_metrics(out_pattern, gt_pattern)
    metrics["structural"] = sm

    if not sm.get("loadable"):
        metrics["pass"] = False
        metrics["fail_reason"] = "result not loadable by cad.api.load_pattern"
        with open(os.path.join(pd, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[validate] {pair_id} -> not loadable; calls={n_calls}({n_err} err)", flush=True)
        return metrics

    iou = panel_iou_metrics(out_pattern, gt_pattern)
    st = stitch_metrics(out_pattern, gt_pattern)
    metrics["panel_iou"] = iou
    metrics["stitch"] = st
    name_match = sm.get("panel_name_recall", 0) == 1.0 and sm.get("panel_name_precision", 0) == 1.0
    iou_score = iou.get("mean_iou_over_union", 0.0)
    f1_score = st.get("f1") or 0.0
    f1_score = 0.0 if f1_score != f1_score else f1_score
    metrics["pass"] = bool(name_match and iou_score >= IOU_THRESHOLD and f1_score >= STITCH_F1_THRESHOLD)
    if not metrics["pass"]:
        reasons = []
        if not name_match:
            reasons.append("panel-name mismatch")
        if iou_score < IOU_THRESHOLD:
            reasons.append(f"IoU {iou_score:.3f} < {IOU_THRESHOLD}")
        if f1_score < STITCH_F1_THRESHOLD:
            reasons.append(f"stitch F1 {f1_score:.3f} < {STITCH_F1_THRESHOLD}")
        metrics["fail_reason"] = "; ".join(reasons)

    with open(os.path.join(pd, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(
        f"[validate] {pair_id} -> pass={metrics['pass']} "
        f"calls={n_calls}({n_err} err) iou={iou_score:.3f} f1={f1_score:.3f}", flush=True
    )
    return metrics


# ---------- run a full validation ----------


def run_validation(
    final_tools_path: str,
    n_pairs: int,
    seed: int,
    out_dir: str,
) -> dict:
    cfg = load_config(TRANSCRIBER_CONFIG_PATH)
    llm_cfg = cfg.agents.l1_transcriber

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(final_tools_path) as f:
        tools_source = f.read()
    derived_names = register_derived_tools(tools_source)
    all_tool_names = derived_names + ["transcribe_done"]
    print(f"[validate] registered {len(derived_names)} derived tools + 1 meta-tool "
          f"(transcribe_done)", flush=True)

    llm = LLM("l1_transcriber", llm_cfg, cfg)
    llm.logging_path = os.path.join(cfg.logging.path, "l1_transcriber", timestamp)
    os.makedirs(llm.logging_path, exist_ok=True)
    llm.tools = llm.tool_registry.get_tool_schemas(all_tool_names)
    print(f"[validate] log dir: {llm.logging_path}", flush=True)
    print(f"[validate] tools exposed to agent: {all_tool_names}", flush=True)

    print(f"[validate] sampling {n_pairs} hold-out pairs (seed={seed})...", flush=True)
    pairs = sample_pairs(n_pairs, seed)
    for i, (a, b, s) in enumerate(pairs):
        print(f"  [{i}] sim={s:.3f}  A={os.path.basename(os.path.dirname(a))}  "
              f"B={os.path.basename(os.path.dirname(b))}", flush=True)

    results = []
    t_start = time.time()
    for i, (A_path, B_path, sim) in enumerate(pairs):
        m = validate_one_pair(llm, i, A_path, B_path, sim, out_dir)
        results.append(m)

    n_pass = sum(1 for m in results if m.get("pass", False))
    n = len(results)
    summary = {
        "n_pairs": n,
        "n_pass": n_pass,
        "completeness_rate": (n_pass / n) if n else 0.0,
        "elapsed_sec": time.time() - t_start,
        "iou_threshold": IOU_THRESHOLD,
        "stitch_f1_threshold": STITCH_F1_THRESHOLD,
        "final_tools_path": final_tools_path,
        "n_derived_tools": len(derived_names),
        "pairs": results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n=== validation done ===")
    print(f"completeness: {n_pass}/{n} = {summary['completeness_rate']:.2%} "
          f"(IoU≥{IOU_THRESHOLD} & stitch_F1≥{STITCH_F1_THRESHOLD} & panel-name match)")
    print(f"results: {out_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-tools", required=True, help="path to final_tools.py")
    parser.add_argument("--n-pairs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1042,
                        help="seed for hold-out pair sampling (different from derive seed)")
    args = parser.parse_args()
    dotenv.load_dotenv()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RESULTS_ROOT, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[validate] run dir: {out_dir}")
    run_validation(args.final_tools, args.n_pairs, args.seed, out_dir)


if __name__ == "__main__":
    main()
