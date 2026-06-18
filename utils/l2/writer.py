"""L2 writer agent: tool-driven multimodal editor for garment-pattern JSON.

Loads the L1 v1 toolset (`results/l1_derive_merged/v1/final_tools.py`) into the
global tool registry via `utils.l1.validator.register_derived_tools` and adds a
`submit_for_review` terminator. Each writer cycle rebuilds a fresh multimodal
user message (target image + current render + current JSON + trajectory log +
optional reviewer note) and runs a bounded inner tool-dispatch loop. State lives
in `utils.l1.validator._state` so the L1 wrappers' mutation flow is reused
verbatim.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from agent.llm import LLM
from tools.registry import ToolResponse
from tools.tools import tool_registry
from utils.l1.validator import (
    get_current_pattern,
    register_derived_tools,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1_TOOLS_PATH = os.path.join(PROJECT_ROOT, "results", "l1_derive_merged", "v1", "final_tools.py")

MAX_TOOL_CALLS_PER_CYCLE_DEFAULT = 8   # primary cap: each individual tool call counts as 1
MAX_INNER_TURNS_DEFAULT = 8             # backstop: max LLM round-trips (defensive against 0-tool-call loops)
ELISION_INLINE_CHAR_LIMIT = 100
BULK_TOOL_NAMES = {"add_panel", "replace_panel_geometry", "add_panel_vertex"}


_writer_state: dict = {"submitted": False, "submit_reason": ""}


def submit_for_review(reason: str = "") -> ToolResponse:
    """Signal that the current pattern is ready for visual review.

    Call this when the current rendering matches the target image. The reviewer
    will compare both images and either accept (`ok`), request fixes, or ask for
    a `redo`. Use a short `reason` to summarize what you did this cycle.
    """
    _writer_state["submitted"] = True
    _writer_state["submit_reason"] = reason or ""
    return ToolResponse(content="submitted; control returned to orchestrator")


_DERIVED_TOOL_NAMES: list[str] | None = None


def _ensure_tools_registered() -> list[str]:
    global _DERIVED_TOOL_NAMES
    if _DERIVED_TOOL_NAMES is not None:
        return _DERIVED_TOOL_NAMES
    with open(V1_TOOLS_PATH) as f:
        src = f.read()
    derived = register_derived_tools(src)
    tool_registry.register(submit_for_review)
    _DERIVED_TOOL_NAMES = derived
    return derived


def _summarize_args(args: dict, max_inline_len: int = ELISION_INLINE_CHAR_LIMIT) -> str:
    parts = []
    for k, v in args.items():
        s = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        if len(s) > max_inline_len:
            if isinstance(v, list):
                s = f"<list of {len(v)}>"
            elif isinstance(v, dict):
                s = f"<dict with {len(v)} keys>"
            else:
                s = s[:max_inline_len] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def _format_trajectory(entries: list[dict]) -> str:
    if not entries:
        return "(empty — this is your first turn this sample)"
    lines = []
    for e in entries:
        t = e["turn"]
        kind = e["kind"]
        if kind == "tool":
            lines.append(f"[T{t}] {e['tool']}({e['args_summary']}) -> {e['status']}")
        elif kind == "error":
            lines.append(f"[T{t}] ERROR: {e['text']}")
        elif kind == "submit":
            reason = e.get("text") or ""
            lines.append(f"[T{t}] submit_for_review({reason})")
        elif kind == "review":
            lines.append(f"[T{t}] REVIEW: {e['text']}")
    return "\n".join(lines)


def _format_reviewer_note(note: dict) -> str:
    verdict = note["verdict"]
    fixes = note.get("fixes", []) or []
    lines = [f"Verdict: {verdict}"]
    if verdict == "ok":
        lines.append("(no fixes — but this block is shown only when the orchestrator wants you to continue refining; treat as a soft 'fix anything obvious')")
        return "\n".join(lines)
    if not fixes:
        lines.append("(reviewer returned no fix items; use your own judgment)")
        return "\n".join(lines)
    lines.append("Address ALL of the following before re-submitting (severity order):")
    for i, fx in enumerate(fixes, 1):
        panel = fx.get("panel", "") or "(no specific panel)"
        sev = fx.get("severity", "?")
        instr = fx.get("instruction", "")
        lines.append(f"  {i}. [{sev}] panel={panel!r}: {instr}")
    return "\n".join(lines)


def _build_turn_message(
    target_b64: str,
    render_b64: str | None,
    render_error: str | None,
    current_state: dict,
    trajectory: list[dict],
    reviewer_note: dict | None,
) -> dict:
    content: list[dict] = []
    content.append({"type": "text", "text": "## Target image (immutable):"})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{target_b64}"}})
    if render_b64:
        content.append({"type": "text", "text": "## Current rendering (from your current JSON):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{render_b64}"}})
    else:
        content.append({"type": "text", "text": f"## Current rendering: **RENDER FAILED** — {render_error or 'unknown error'}. Inspect the JSON below and revert the recent edits that broke it before submitting."})
    content.append({"type": "text", "text": "## Current JSON (ground truth for state — trust this over your memory of prior calls):\n```json\n" + json.dumps(current_state, indent=2, ensure_ascii=False) + "\n```"})
    content.append({"type": "text", "text": "## Trajectory so far (oldest -> newest):\n" + _format_trajectory(trajectory)})
    if reviewer_note is not None:
        content.append({"type": "text", "text": "## Latest reviewer feedback:\n" + _format_reviewer_note(reviewer_note)})
    content.append({"type": "text", "text": "## Next step\nCall one or more L1 tools to edit the pattern, OR call `submit_for_review` if the current rendering matches the target."})
    return {"role": "user", "content": content}


def _state_signature(state: dict) -> str:
    """Cheap content hash of pattern dict for no-change detection."""
    return json.dumps(state, sort_keys=True, ensure_ascii=False)


class WriterSession:
    def __init__(self, llm_cfg, global_cfg, run_log_subdir: str | None = None):
        derived = _ensure_tools_registered()
        self.tool_names = derived + ["submit_for_review"]

        self.llm = LLM("l2_writer", llm_cfg, global_cfg)
        if run_log_subdir:
            self.llm.logging_path = os.path.join(global_cfg.logging.path, run_log_subdir, "l2_writer")
            os.makedirs(self.llm.logging_path, exist_ok=True)
        self.llm.tools = self.llm.tool_registry.get_tool_schemas(self.tool_names)

        self.trajectory: list[dict] = []
        self._turn_counter = 0

    def reset_for_sample(self) -> None:
        """Clear per-sample state. Call before each sample's first cycle.
        The caller is responsible for `set_active_pattern(initial)` on `_l1_state`.
        """
        self.trajectory = []
        self._turn_counter = 0
        _writer_state["submitted"] = False
        _writer_state["submit_reason"] = ""

    def run_until_submit_or_cap(
        self,
        target_b64: str,
        render_b64: str | None,
        render_error: str | None,
        reviewer_note: dict | None = None,
        max_tool_calls: int = MAX_TOOL_CALLS_PER_CYCLE_DEFAULT,
        max_inner_turns: int = MAX_INNER_TURNS_DEFAULT,
    ) -> dict:
        """One writer cycle: bounded tool-call loop until submit / tool-call cap / inner-turn cap.

        Each individual tool call (e.g. one set_panel_edge) counts as 1 toward `max_tool_calls`.
        Once the cap is reached mid-turn, the cycle ends immediately (force-submit) so the
        reviewer can weigh in before the writer keeps editing blind.
        """
        _writer_state["submitted"] = False
        _writer_state["submit_reason"] = ""

        sig_at_cycle_start = _state_signature(get_current_pattern() or {})

        n_inner_turns = 0
        n_tool_calls = 0
        n_errors = 0
        cap_hit = False
        last_reviewer_note = reviewer_note

        while n_inner_turns < max_inner_turns and not cap_hit:
            n_inner_turns += 1
            self._turn_counter += 1
            turn_id = self._turn_counter

            current = get_current_pattern() or {}
            msg = _build_turn_message(
                target_b64=target_b64,
                render_b64=render_b64,
                render_error=render_error,
                current_state=current,
                trajectory=self.trajectory,
                reviewer_note=last_reviewer_note,
            )
            last_reviewer_note = None  # show once at the top of the cycle only

            self.llm.reset_context()
            try:
                response = self.llm.generate_response([msg])
            except Exception as e:
                self.trajectory.append({"kind": "error", "turn": turn_id, "text": f"LLM call failed: {type(e).__name__}: {e}"})
                break

            tool_calls = response.tool_calls or []
            if not tool_calls:
                self.trajectory.append({"kind": "submit", "turn": turn_id, "text": "(implicit — model emitted no tool calls)"})
                _writer_state["submitted"] = True
                break

            for tc in tool_calls:
                tname = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except Exception as e:
                    self.trajectory.append({"kind": "error", "turn": turn_id, "tool": tname, "text": f"unparseable args: {e}"})
                    n_errors += 1
                    continue

                args_summary = _summarize_args(args)
                tr: ToolResponse = self.llm.tool_registry.call_tool(tname, args)
                content = (tr.content or "").strip()
                first_line = content.splitlines()[0] if content else "(no response)"
                status = first_line
                is_err = first_line.startswith("error:") or first_line.startswith("ERROR")

                if tname == "submit_for_review":
                    self.trajectory.append({"kind": "submit", "turn": turn_id, "text": args.get("reason", "")})
                else:
                    self.trajectory.append({"kind": "tool", "turn": turn_id, "tool": tname, "args_summary": args_summary, "status": status})
                    n_tool_calls += 1
                    if is_err:
                        n_errors += 1
                    if n_tool_calls >= max_tool_calls:
                        cap_hit = True
                        self.trajectory.append({"kind": "submit", "turn": turn_id, "text": f"(forced — hit max_tool_calls={max_tool_calls})"})
                        _writer_state["submitted"] = True
                        break

            if _writer_state["submitted"]:
                break

        sig_at_cycle_end = _state_signature(get_current_pattern() or {})
        changed = sig_at_cycle_start != sig_at_cycle_end

        return {
            "submitted": _writer_state["submitted"],
            "submit_reason": _writer_state["submit_reason"],
            "n_inner_turns": n_inner_turns,
            "n_tool_calls": n_tool_calls,
            "n_errors": n_errors,
            "cap_hit": cap_hit,
            "state_changed": changed,
        }


def render_log_for_human(trajectory: list[dict]) -> str:
    """Pretty-print the trajectory for the per-sample writer_session.log artifact."""
    if not trajectory:
        return "(empty)"
    return _format_trajectory(trajectory)
