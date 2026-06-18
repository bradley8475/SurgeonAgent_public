"""Meta-tools the L1 designer agent calls to build up the tool registry incrementally.

Cross-pair state lives in module-level dicts — the driver must call `set_active(...)`
between pairs to update `pair_id` and clear the per-pair history. Sequential pairs only.
"""

from __future__ import annotations

import ast
import json
import os

from tools.registry import ToolResponse
from tools.tools import tool_registry


# --- per-run module state (driver mutates these between pairs) ---

_state = {
    "run_dir": None,        # output root for this derivation run
    "pair_id": None,        # current pair identifier
    "registry": {},         # cumulative {tool_name: source}; persists across pairs
    "pair_history": [],     # ops done THIS pair: [("write"|"update"|"delete", name), ...]
    "submitted": False,     # set by finish_pair; driver checks this to advance
}

FORBIDDEN_IMPORTS = ("cad", "numpy", "pandas", "scipy", "torch", "tensorflow", "PIL")

# Cap new tools per pair to prevent cold-start dumps. Updates and deletes are unbounded.
MAX_WRITES_PER_PAIR = 5

# Prelude prepended to every composite tools.py — the LLM must NOT redefine these.
# `copy` and `ToolError` are pre-provided in the runtime namespace.
PRELUDE = '''"""Auto-generated derived editing tools.

Prelude (pre-imported / pre-defined) is at the top; do not edit.
Below the prelude, each tool is delimited by a `# === <name> ===` marker.
"""
import copy


class ToolError(Exception):
    """Raised by any derived tool when its arguments violate a schema invariant
    (panel doesn't exist, edge index out of range, enum value not allowed, etc.)."""
    pass

'''

TOOL_DELIMITER = "# === {name} ===\n"


def set_active(run_dir: str, pair_id: str) -> None:
    """Driver calls this before each pair. Resets per-pair flags but keeps cumulative registry.
    Pre-creates <run_dir>/<pair_id>/ and writes the initial composite tools.py reflecting
    the registry state visible at the start of this pair."""
    _state["run_dir"] = run_dir
    _state["pair_id"] = pair_id
    _state["pair_history"] = []
    _state["submitted"] = False

    out_dir = os.path.join(run_dir, pair_id)
    os.makedirs(out_dir, exist_ok=True)
    _write_composite_file()
    print(f"[derive] pair {pair_id} ready; carrying over {len(_state['registry'])} tool(s)", flush=True)


def _composite_path() -> str:
    return os.path.join(_state["run_dir"], _state["pair_id"], "tools.py")


def compose_file(registry: dict[str, str]) -> str:
    """Build the composite tools.py content: prelude + each tool's source separated by markers."""
    parts = [PRELUDE]
    for name in sorted(registry.keys()):
        parts.append("\n" + TOOL_DELIMITER.format(name=name))
        parts.append(registry[name].rstrip() + "\n")
    return "".join(parts)


def _write_composite_file() -> None:
    """Rewrite the live tools.py from the current registry. Called after every CRUD op."""
    with open(_composite_path(), "w", encoding="utf-8") as f:
        f.write(compose_file(_state["registry"]))


def get_registry_snapshot() -> dict:
    """Driver reads this after a pair completes to snapshot the cumulative registry."""
    return dict(_state["registry"])


def get_pair_history() -> list:
    return list(_state["pair_history"])


def is_submitted() -> bool:
    return _state["submitted"]


def render_toolset_for_prompt() -> str:
    """Format the current toolset as `name(signature) — docstring` lines for the user message."""
    if not _state["registry"]:
        return "(no tools registered yet — you are on the first turn)"
    lines = []
    for name in sorted(_state["registry"].keys()):
        src = _state["registry"][name]
        try:
            tree = ast.parse(src)
            fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
            sig = ast.unparse(fn.args)
            doc = ast.get_docstring(fn) or "(no docstring)"
            doc_first_line = doc.strip().split("\n", 1)[0]
            lines.append(f"- {name}({sig}) — {doc_first_line}")
        except Exception as e:
            lines.append(f"- {name}(?) — (failed to introspect: {e})")
    return "\n".join(lines)


# --- validation helpers ---


FORBIDDEN_PARAM_NAMES = {"panels", "stitches", "panel_list", "stitch_list", "all_panels", "all_stitches"}


def _is_pattern_collection_target(target: ast.expr) -> bool:
    """True if target is `state["pattern"]["panels"]` or `state["pattern"]["stitches"]`."""
    if not isinstance(target, ast.Subscript):
        return False
    slc = target.slice
    if not (isinstance(slc, ast.Constant) and slc.value in ("panels", "stitches")):
        return False
    outer = target.value
    if not (isinstance(outer, ast.Subscript)
            and isinstance(outer.slice, ast.Constant)
            and outer.slice.value == "pattern"
            and isinstance(outer.value, ast.Name)
            and outer.value.id == "state"):
        return False
    return True


def _expr_references_pattern_collection(node: ast.expr) -> bool:
    """True if any subexpression refers to state['pattern']['panels'|'stitches']."""
    for n in ast.walk(node):
        if isinstance(n, ast.Subscript) and _is_pattern_collection_target(n):
            return True
    return False


def _is_forced_cleanup_assignment(value: ast.expr) -> bool:
    """True if the RHS is a derivation from the current state collection
    (filter / transform / slice), which we treat as a *forced cross-reference cleanup*
    rather than a wholesale replacement.

    Allowed shapes:
      - ListComp / GeneratorExp / DictComp / SetComp whose generator iterates over
        state['pattern']['panels'|'stitches']
      - Slice/subscript of state['pattern']['panels'|'stitches'] (e.g. [:i] + [...] + [i+1:])
      - Method call on state['pattern']['panels'|'stitches'] (.copy(), .items(), etc.)
      - BinOp involving the existing collection (e.g. list[:i] + [new] + list[i+1:])
    """
    if isinstance(value, (ast.ListComp, ast.GeneratorExp, ast.DictComp, ast.SetComp)):
        for gen in value.generators:
            if _expr_references_pattern_collection(gen.iter):
                return True
        return False
    # any other shape: allow if the expression somewhere references the existing collection
    return _expr_references_pattern_collection(value)


def _validate_tool_source(name: str, source: str) -> str | None:
    """Return error message string, or None if source is acceptable.

    Source must be a single top-level `def NAME(state, ...)` block. The runtime prelude
    provides `copy` and `ToolError` — tools must NOT redefine them or import anything.
    Atomicity is enforced: parameters named `panels`/`stitches`/etc. are rejected,
    and wholesale assignments like `state["pattern"]["stitches"] = X` are rejected.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"source does not parse as Python: {e}"

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return (
            "source must contain exactly one top-level `def` block — "
            "no imports, no class definitions, no helper functions at top level. "
            "`copy` and `ToolError` are pre-provided by the runtime prelude; just use them."
        )

    target = tree.body[0]
    if target.name != name:
        return f"top-level function is `{target.name}` but you said `name={name!r}`"

    if not target.args.args or target.args.args[0].arg != "state":
        return f"first parameter must be named `state` (got: {[a.arg for a in target.args.args]})"

    if ast.get_docstring(target) is None or not ast.get_docstring(target).strip():
        return f"`{name}` must have a non-empty docstring (subsequent turns and held-out transcriber depend on it)"

    # atomicity: forbidden parameter names that signal bulk-content tools
    for arg in target.args.args[1:]:
        if arg.arg in FORBIDDEN_PARAM_NAMES:
            return (
                f"parameter `{arg.arg}` represents bulk content (multiple panels/stitches) — "
                f"forbidden by the atomicity rule. Tools must edit at most one panel and at most "
                f"one stitch per call. Split into separate atomic tools."
            )

    # Atomicity: assignments to state["pattern"]["panels"|"stitches"] are only allowed
    # if the RHS is *derived from* the current collection (filter / transform / slice) —
    # i.e. a forced cross-reference cleanup, not a wholesale replacement.
    for node in ast.walk(target):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if _is_pattern_collection_target(tgt) and not _is_forced_cleanup_assignment(node.value):
                    slot = ast.unparse(tgt)
                    rhs_preview = ast.unparse(node.value)[:80]
                    return (
                        f"body contains a wholesale-replacement assignment `{slot} = {rhs_preview}` — "
                        f"the RHS does not derive from the current collection, so this is treated as "
                        f"wholesale replacement (forbidden). Acceptable forms include filtering "
                        f"(`state['pattern']['stitches'] = [s for s in state['pattern']['stitches'] if ...]`) "
                        f"or single-element mutation "
                        f"(`state['pattern']['panels'][name] = ...`, "
                        f"`state['pattern']['stitches'].append(...)`, `.pop(i)`)."
                    )

    return None


def _persist_snapshot(observations: str) -> str:
    """Tools are already on disk (written live by write_tool/update_tool/delete_tool).
    At pair finish, just dump observations.txt and history.json."""
    out_dir = os.path.join(_state["run_dir"], _state["pair_id"])
    if observations.strip():
        with open(os.path.join(out_dir, "observations.txt"), "w", encoding="utf-8") as f:
            f.write(observations)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(_state["pair_history"], f, indent=2, ensure_ascii=False)
    return out_dir


# --- meta-tools (registered with the global tool_registry; agents opt in via config) ---


def _post_submit_block(meta_tool_name: str) -> ToolResponse | None:
    """If finish_pair already accepted this pair, all subsequent meta-tool calls
    are rejected immediately. The model is supposed to emit a no-tool-call message and stop."""
    if _state["submitted"]:
        msg = (
            f"error: this pair is already finished; `{meta_tool_name}` rejected. "
            f"Stop calling tools — emit a brief confirmation message and finish."
        )
        print(f"[derive] {meta_tool_name} -> rejected (pair already finished)", flush=True)
        return ToolResponse(content=msg)
    return None


@tool_registry.register
def write_tool(name: str, source: str) -> ToolResponse:
    """Register a NEW editing tool. Errors if `name` already exists (use update_tool to modify).

    Source must:
    - parse as valid Python
    - define a function whose name matches `name`, taking `state: dict` as first param and returning a dict
    - import only `copy` and stdlib (no cad / numpy / pandas / etc.)
    - have a non-empty docstring describing the edit in semantic-token terms

    Args:
        name: function name (becomes the tool's identifier in the registry)
        source: full Python source defining the function (and any helper class like ToolError)
    """
    if (blocked := _post_submit_block("write_tool")) is not None:
        return blocked
    n_writes = sum(1 for op, _ in _state["pair_history"] if op == "write")
    if n_writes >= MAX_WRITES_PER_PAIR:
        msg = (
            f"error: per-pair write_tool budget exhausted ({n_writes}/{MAX_WRITES_PER_PAIR}). "
            f"Submit with the current toolset; later pairs will add what's missing. "
            f"Cold-start dump is exactly what this budget prevents."
        )
        print(f"[derive] write_tool(name={name!r}) -> budget exhausted", flush=True)
        return ToolResponse(content=msg)
    if name in _state["registry"]:
        msg = f"error: tool `{name}` already exists. Use update_tool to modify, or pick a different name."
        print(f"[derive] write_tool(name={name!r}, {len(source)} chars) -> {msg}", flush=True)
        return ToolResponse(content=msg)
    err = _validate_tool_source(name, source)
    if err:
        msg = f"error: {err}"
        print(f"[derive] write_tool(name={name!r}, {len(source)} chars) -> {msg}", flush=True)
        return ToolResponse(content=msg)
    _state["registry"][name] = source
    _state["pair_history"].append(("write", name))
    _write_composite_file()
    msg = f"registered: `{name}` ({n_writes + 1}/{MAX_WRITES_PER_PAIR} writes used this pair). Registry size: {len(_state['registry'])}."
    print(f"[derive] write_tool(name={name!r}, {len(source)} chars) -> {msg}", flush=True)
    return ToolResponse(content=msg)


@tool_registry.register
def update_tool(name: str, source: str) -> ToolResponse:
    """Replace the source of an existing tool. Errors if `name` doesn't exist.

    Use when an existing tool is almost right but needs generalization or a bug fix.
    Same validation rules as write_tool.

    Args:
        name: existing tool name
        source: full new Python source
    """
    if (blocked := _post_submit_block("update_tool")) is not None:
        return blocked
    if name not in _state["registry"]:
        msg = f"error: tool `{name}` does not exist. Use write_tool to add a new tool."
        print(f"[derive] update_tool(name={name!r}) -> {msg}", flush=True)
        return ToolResponse(content=msg)
    err = _validate_tool_source(name, source)
    if err:
        msg = f"error: {err}"
        print(f"[derive] update_tool(name={name!r}) -> {msg}", flush=True)
        return ToolResponse(content=msg)
    _state["registry"][name] = source
    _state["pair_history"].append(("update", name))
    _write_composite_file()
    msg = f"updated: `{name}`."
    print(f"[derive] update_tool(name={name!r}, {len(source)} chars) -> {msg}", flush=True)
    return ToolResponse(content=msg)


@tool_registry.register
def delete_tool(name: str) -> ToolResponse:
    """Remove a tool from the registry. Errors if `name` doesn't exist.

    Use sparingly — only when the tool is clearly subsumed by another or was a mistake.
    Past pairs' call sequences may still reference deleted tools (those will fail during reconstruction validation).

    Args:
        name: tool name to remove
    """
    if (blocked := _post_submit_block("delete_tool")) is not None:
        return blocked
    if name not in _state["registry"]:
        msg = f"error: tool `{name}` does not exist."
        print(f"[derive] delete_tool(name={name!r}) -> {msg}", flush=True)
        return ToolResponse(content=msg)
    del _state["registry"][name]
    _state["pair_history"].append(("delete", name))
    _write_composite_file()
    msg = f"deleted: `{name}`. Registry size: {len(_state['registry'])}."
    print(f"[derive] delete_tool(name={name!r}) -> {msg}", flush=True)
    return ToolResponse(content=msg)


@tool_registry.register
def view_tool(name: str) -> ToolResponse:
    """Return the full source code of an existing tool. Use to inspect implementation when the docstring isn't enough.

    Args:
        name: tool name to inspect
    """
    if (blocked := _post_submit_block("view_tool")) is not None:
        return blocked
    if name not in _state["registry"]:
        msg = f"error: tool `{name}` does not exist."
        print(f"[derive] view_tool(name={name!r}) -> {msg}", flush=True)
        return ToolResponse(content=msg)
    src = _state["registry"][name]
    print(f"[derive] view_tool(name={name!r}) -> returned {len(src)} chars", flush=True)
    return ToolResponse(content=src)


@tool_registry.register
def finish_pair(observations: str = "") -> ToolResponse:
    """Mark this pair complete. Your deliverable is the cumulative tool registry plus this pair's CRUD log.

    Call this when you've decided which tools to add/update/delete (if any) for this pair's diff.
    You are NOT required to demonstrate that the registered tools can transform A into B — the
    toolset is judged externally by a different agent on hold-out pairs you won't see. Your job
    here is purely to design good atomic editing primitives that capture the kinds of edits you
    see in this diff.

    Args:
        observations: optional free-form notes — e.g. "B differs from A by added cuff-skirt
            panels and replaced geometry on all main panels; I added add_panel + replace_panel.
            Per-stitch additions noticed but not yet tooled (next pair candidate)."
    """
    if (blocked := _post_submit_block("finish_pair")) is not None:
        return blocked
    out_dir = _persist_snapshot(observations)
    _state["submitted"] = True
    msg = (
        f"accepted. Registry size: {len(_state['registry'])}. "
        f"Snapshot at {out_dir}. "
        f"Reply with one short confirmation line; do not call any more tools."
    )
    print(f"[derive] finish_pair(observations={len(observations)} chars) -> accepted; pair done", flush=True)
    return ToolResponse(content=msg)
