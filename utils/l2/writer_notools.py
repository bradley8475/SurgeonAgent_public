"""L2 writer (no-tools ablation): each cycle = 1 LLM call returning full pattern.json.

Drop-in replacement for utils.l2.writer.WriterSession used by utils.l2.loop. The
public interface (reset_for_sample / run_until_submit_or_cap) matches the
tool-driven writer so loop.py and run_l2.py do not need to know which variant
is in use. Internally there are NO tools — each cycle issues one LLM round-trip,
parses the model's JSON response, and replaces `_l1_state["pattern"]` wholesale.

This isolates a single ablation variable: tool-based incremental editing vs.
free-form full-JSON rewriting, holding everything else (writer/reviewer loop,
retrieval init, model, reviewer, max rounds) constant.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agent.llm import LLM
from utils.eval.metrics import parse_json_robust
from utils.l1.validator import get_current_pattern, set_active_pattern


def _state_signature(state: dict) -> str:
    return json.dumps(state, sort_keys=True, ensure_ascii=False)


def _format_trajectory(entries: list[dict]) -> str:
    if not entries:
        return "(empty — this is your first turn this sample)"
    lines = []
    for e in entries:
        t = e["turn"]
        kind = e["kind"]
        if kind == "submit":
            reason = e.get("text") or ""
            lines.append(f"[T{t}] submitted full JSON ({reason})" if reason else f"[T{t}] submitted full JSON")
        elif kind == "error":
            lines.append(f"[T{t}] ERROR: {e['text']}")
        elif kind == "review":
            lines.append(f"[T{t}] REVIEW: {e['text']}")
    return "\n".join(lines)


def _format_reviewer_note(note: dict) -> str:
    verdict = note["verdict"]
    fixes = note.get("fixes", []) or []
    lines = [f"Verdict: {verdict}"]
    if verdict == "ok":
        lines.append("(soft 'fix anything obvious' — orchestrator asked you to keep refining)")
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
        content.append({"type": "text", "text": f"## Current rendering: **RENDER FAILED** — {render_error or 'unknown error'}. Inspect the JSON below; if your last output broke it, revert to a valid shape."})
    content.append({"type": "text", "text": "## Current JSON (ground truth for state — start from this):\n```json\n" + json.dumps(current_state, indent=2, ensure_ascii=False) + "\n```"})
    content.append({"type": "text", "text": "## Trajectory so far (oldest -> newest):\n" + _format_trajectory(trajectory)})
    if reviewer_note is not None:
        content.append({"type": "text", "text": "## Latest reviewer feedback:\n" + _format_reviewer_note(reviewer_note)})
    content.append({"type": "text", "text": "## Next step\nEmit the **complete** corrected pattern.json in a single ```json``` fence. No commentary."})
    return {"role": "user", "content": content}


class WriterSessionNoTools:
    """No-tools ablation variant of WriterSession. Same external interface."""

    def __init__(self, llm_cfg, global_cfg, run_log_subdir: str | None = None):
        self.tool_names: list[str] = []  # for parity with WriterSession (logged by run_l2)
        self.llm = LLM("l2_writer_notools", llm_cfg, global_cfg)
        if run_log_subdir:
            self.llm.logging_path = os.path.join(global_cfg.logging.path, run_log_subdir, "l2_writer_notools")
            os.makedirs(self.llm.logging_path, exist_ok=True)
        self.llm.tools = None

        self.trajectory: list[dict] = []
        self._turn_counter = 0

    def reset_for_sample(self) -> None:
        self.trajectory = []
        self._turn_counter = 0

    def run_until_submit_or_cap(
        self,
        target_b64: str,
        render_b64: str | None,
        render_error: str | None,
        reviewer_note: dict | None = None,
        max_tool_calls: int = 0,        # unused — there are no tools
        max_inner_turns: int = 1,       # unused — one LLM call per cycle by definition
    ) -> dict:
        sig_at_cycle_start = _state_signature(get_current_pattern() or {})

        self._turn_counter += 1
        turn_id = self._turn_counter

        current = get_current_pattern() or {}
        msg = _build_turn_message(
            target_b64=target_b64,
            render_b64=render_b64,
            render_error=render_error,
            current_state=current,
            trajectory=self.trajectory,
            reviewer_note=reviewer_note,
        )

        self.llm.reset_context()
        n_errors = 0
        try:
            response = self.llm.generate_response([msg])
        except Exception as e:
            self.trajectory.append({"kind": "error", "turn": turn_id, "text": f"LLM call failed: {type(e).__name__}: {e}"})
            return {
                "submitted": False,
                "submit_reason": "",
                "n_inner_turns": 1,
                "n_tool_calls": 0,
                "n_errors": 1,
                "cap_hit": False,
                "state_changed": False,
            }

        raw_text = response.content if isinstance(response.content, str) else str(response.content)
        parsed = parse_json_robust(raw_text or "")
        if parsed is None:
            self.trajectory.append({"kind": "error", "turn": turn_id, "text": f"could not parse JSON from {len(raw_text or '')} chars of response"})
            n_errors = 1
            submitted = True
            submit_reason = "parse failed — state unchanged"
        else:
            # Accept either {"pattern": {...}} or a bare {"panels": ..., "stitches": ...}
            if "pattern" not in parsed and ("panels" in parsed or "stitches" in parsed):
                new_state = {"pattern": parsed}
            else:
                new_state = parsed
            try:
                set_active_pattern(new_state)
                submit_reason = "full JSON replaced state"
            except Exception as e:
                self.trajectory.append({"kind": "error", "turn": turn_id, "text": f"set_active_pattern failed: {type(e).__name__}: {e}"})
                n_errors = 1
                submit_reason = "replace failed — state unchanged"
            submitted = True
            self.trajectory.append({"kind": "submit", "turn": turn_id, "text": submit_reason})

        sig_at_cycle_end = _state_signature(get_current_pattern() or {})
        return {
            "submitted": submitted,
            "submit_reason": submit_reason if submitted else "",
            "n_inner_turns": 1,
            "n_tool_calls": 0,
            "n_errors": n_errors,
            "cap_hit": False,
            "state_changed": sig_at_cycle_start != sig_at_cycle_end,
        }


def render_log_for_human(trajectory: list[dict]) -> str:
    if not trajectory:
        return "(empty)"
    return _format_trajectory(trajectory)
