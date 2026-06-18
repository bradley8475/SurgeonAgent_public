"""L2 reviewer agent: stateless multimodal critic.

One-shot per call. Compares target image to current rendering and emits a
bounded top-K (K=3) verdict via a single structured tool call. Panel-name
hallucinations are filtered post-hoc against the current JSON.
"""

from __future__ import annotations

import json
import os
import re

from agent.llm import LLM
from tools.registry import ToolResponse
from tools.tools import tool_registry


MAX_FIXES = 3

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_content_as_verdict(content: str) -> dict | None:
    """Fallback path: model returned the verdict as text content (often in a ```json``` fence)
    instead of a function call. Try several strategies to extract a {verdict, fixes_json} dict.
    Returns None if nothing parseable found."""
    if not content:
        return None
    candidates: list[str] = []
    m = _JSON_FENCE_RE.search(content)
    if m:
        candidates.append(m.group(1))
    candidates.append(content.strip())
    start = content.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(content)):
            c = content[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(content[start:i + 1])
                    break
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            return obj
    return None


_reviewer_state: dict = {"verdict": None, "fixes_json": None}


def submit_verdict(verdict: str, fixes_json: str) -> ToolResponse:
    """Submit a verdict on the current rendering vs target.

    `verdict`: one of "ok" / "fix" / "redo".
    `fixes_json`: JSON-encoded list (length 0-3) of {"panel": str, "severity": "high"|"medium"|"low", "instruction": str}.
    Severity order: high -> medium -> low. Use [] when verdict is "ok".
    """
    _reviewer_state["verdict"] = verdict
    _reviewer_state["fixes_json"] = fixes_json
    return ToolResponse(content=f"verdict={verdict} recorded")


_REGISTERED = False


def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    tool_registry.register(submit_verdict)
    _REGISTERED = True


def _build_review_message(target_b64: str, render_b64: str, current_state: dict, panel_names: list[str]) -> dict:
    content: list[dict] = [
        {"type": "text", "text": "## Target image:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{target_b64}"}},
        {"type": "text", "text": "## Current rendering (writer's current JSON, rendered):"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{render_b64}"}},
        {"type": "text", "text": "## Current JSON (for reference only — do not rewrite, just identify what's wrong):\n```json\n" + json.dumps(current_state, indent=2, ensure_ascii=False) + "\n```"},
        {"type": "text", "text": "## Available panel names (you MUST only use names from this list when verdict='fix'):\n" + json.dumps(panel_names)},
        {"type": "text", "text": "## Your task\nCompare the target to the current rendering. Call `submit_verdict` EXACTLY ONCE. No free text."},
    ]
    return {"role": "user", "content": content}


def _filter_fixes(fixes: list[dict], available_panels: set[str]) -> tuple[list[dict], list[str]]:
    """Drop items whose panel is not in available_panels (allow empty panel for 'redo' / generic).
    Returns (filtered, dropped_reasons)."""
    kept: list[dict] = []
    dropped: list[str] = []
    for fx in fixes:
        if not isinstance(fx, dict):
            dropped.append(f"non-dict item: {fx!r}")
            continue
        panel = fx.get("panel", "") or ""
        if panel and panel not in available_panels:
            dropped.append(f"unknown panel: {panel!r}")
            continue
        kept.append({
            "panel": panel,
            "severity": fx.get("severity", "medium"),
            "instruction": fx.get("instruction", ""),
        })
    return kept[:MAX_FIXES], dropped


_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _sort_by_severity(fixes: list[dict]) -> list[dict]:
    return sorted(fixes, key=lambda fx: _SEVERITY_RANK.get(fx.get("severity", "medium"), 1))


class ReviewerSession:
    def __init__(self, llm_cfg, global_cfg, run_log_subdir: str | None = None):
        _ensure_registered()
        self.llm = LLM("l2_reviewer", llm_cfg, global_cfg)
        if run_log_subdir:
            self.llm.logging_path = os.path.join(global_cfg.logging.path, run_log_subdir, "l2_reviewer")
            os.makedirs(self.llm.logging_path, exist_ok=True)
        self.llm.tools = self.llm.tool_registry.get_tool_schemas(["submit_verdict"])

    def review(self, target_b64: str, render_b64: str, current_state: dict) -> dict:
        """Returns {"verdict": str, "fixes": list[dict], "raw": str, "hallucinations": list[str]}."""
        panels = current_state.get("pattern", {}).get("panels", {})
        panel_names = sorted(panels.keys())

        _reviewer_state["verdict"] = None
        _reviewer_state["fixes_json"] = None

        self.llm.reset_context()
        try:
            response = self.llm.generate_response([_build_review_message(target_b64, render_b64, current_state, panel_names)])
        except Exception as e:
            return {
                "verdict": "fix",
                "fixes": [{"panel": "", "severity": "high", "instruction": f"reviewer LLM call failed ({type(e).__name__}: {e}); please continue refining"}],
                "raw": "",
                "hallucinations": [],
            }

        tool_calls = response.tool_calls or []
        parsed_via_content = False
        if tool_calls:
            tc = tool_calls[0]
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except Exception as e:
                return {
                    "verdict": "fix",
                    "fixes": [{"panel": "", "severity": "high", "instruction": f"reviewer emitted unparseable verdict args ({e}); please continue refining"}],
                    "raw": tc.function.arguments or "",
                    "hallucinations": [],
                }
            self.llm.tool_registry.call_tool(tc.function.name, args)
        else:
            # Content-fallback: model returned the verdict as text (often ```json``` fenced).
            obj = _parse_content_as_verdict(response.content or "")
            if obj is None:
                return {
                    "verdict": "fix",
                    "fixes": [{"panel": "", "severity": "high", "instruction": "reviewer returned no verdict (and content was unparseable); please continue refining"}],
                    "raw": response.content or "",
                    "hallucinations": [],
                }
            _reviewer_state["verdict"] = obj.get("verdict") or "fix"
            fj = obj.get("fixes_json")
            if isinstance(fj, list):
                fj = json.dumps(fj)
            _reviewer_state["fixes_json"] = fj if isinstance(fj, str) else json.dumps(obj.get("fixes") or [])
            parsed_via_content = True

        verdict = _reviewer_state["verdict"] or "fix"
        raw = _reviewer_state["fixes_json"] or "[]"
        try:
            fixes_raw = json.loads(raw)
            if not isinstance(fixes_raw, list):
                fixes_raw = []
        except Exception:
            fixes_raw = []

        available = set(panel_names)
        kept, dropped = _filter_fixes(fixes_raw, available)
        kept = _sort_by_severity(kept)

        if verdict == "fix" and not kept:
            kept = [{"panel": "", "severity": "high", "instruction": "reviewer referenced only unknown panels; please re-examine all panels"}]

        if verdict == "ok":
            kept = []

        return {
            "verdict": verdict,
            "fixes": kept,
            "raw": raw,
            "hallucinations": dropped,
            "parsed_via_content": parsed_via_content,
        }
