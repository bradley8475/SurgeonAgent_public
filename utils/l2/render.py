"""Render the current L2 JSON state to a PNG for writer/reviewer consumption.

Failure-tolerant: any exception during Pattern.load or visualize is captured and
returned as {ok: False, error: "..."} so the orchestrator can route around it.
"""

from __future__ import annotations

import base64
import os
from typing import TypedDict

from cad.core import Pattern
from cad.utils.visualize import visualize


class RenderResult(TypedDict):
    ok: bool
    path: str | None
    b64: str | None
    error: str | None


def _check_dangling_stitch_refs(state: dict) -> list[str]:
    """Find stitch refs whose panel doesn't exist or whose edge index is out of range.
    Returns human-readable issue strings; empty list = all refs valid.

    Stitch storage uses keys "0" / "1" for the two sides (per L1 add_stitch / replace_stitch);
    each side is a list of {"panel": str, "edge": int} refs.
    """
    issues: list[str] = []
    panels = (state.get("pattern") or {}).get("panels") or {}
    stitches = (state.get("pattern") or {}).get("stitches") or []
    edge_counts = {name: len((p or {}).get("edges") or []) for name, p in panels.items()}
    for idx, st in enumerate(stitches):
        for side_key in ("0", "1"):
            for ref in (st or {}).get(side_key) or []:
                pname = (ref or {}).get("panel")
                pedge = (ref or {}).get("edge")
                if pname not in edge_counts:
                    issues.append(f"stitch[{idx}].side {side_key!r}: panel {pname!r} not found")
                elif not isinstance(pedge, int) or pedge < 0 or pedge >= edge_counts[pname]:
                    issues.append(
                        f"stitch[{idx}].side {side_key!r}: panel={pname!r} edge={pedge} "
                        f"is out of range ({pname!r} has {edge_counts[pname]} edges, valid 0..{edge_counts[pname]-1})"
                    )
    return issues


def render_state(state: dict, out_path: str, debug_labels: bool = False) -> RenderResult:
    """state is the full wrapper dict ({"pattern": {...}}). Renders panels sorted by
    name (matches metrics.py:_render_sorted) so the visual layout is comparable to
    the dataset's target.png.

    `debug_labels=True` overlays vertex indices (v0..vN with coords) and edge indices
    (e0..eN with bezier params when curved) on each panel. Use this for the writer's
    current-state render so it can correlate visual edges with JSON indices. NEVER
    enable on the target render — it would leak the answer numerically.
    """
    try:
        pattern = Pattern.load(state)
    except Exception as e:
        # Dangling stitch refs (common after `replace_panel_geometry` without
        # follow-up stitch cleanup) surface as IndexError inside cad. Translate
        # to an actionable message so the writer can call remove_stitch /
        # replace_stitch / set_stitch_reference on the specific bad refs.
        bad = _check_dangling_stitch_refs(state)
        if bad:
            shown = bad[:5]
            extra = f"  - ... and {len(bad) - 5} more" if len(bad) > 5 else ""
            details = "\n  - ".join(shown)
            msg = (
                f"Pattern.load failed due to dangling stitch refs "
                f"(likely from replace_panel_geometry / add_panel_vertex without follow-up). "
                f"Fix the listed stitches with remove_stitch / replace_stitch / set_stitch_reference:\n  - {details}"
                + (f"\n{extra}" if extra else "")
            )
            return RenderResult(ok=False, path=None, b64=None, error=msg)
        return RenderResult(ok=False, path=None, b64=None, error=f"Pattern.load: {type(e).__name__}: {e}")

    sorted_panels = sorted(pattern.panels, key=lambda p: p.name)
    saved = pattern.panels
    pattern.panels = sorted_panels
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        visualize(pattern, out_path, show_stitches=True, include_edge_labels=False, show_debug_labels=debug_labels)
    except Exception as e:
        pattern.panels = saved
        return RenderResult(ok=False, path=None, b64=None, error=f"visualize: {type(e).__name__}: {e}")
    finally:
        pattern.panels = saved

    try:
        with open(out_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    except Exception as e:
        return RenderResult(ok=False, path=out_path, b64=None, error=f"read PNG: {type(e).__name__}: {e}")

    return RenderResult(ok=True, path=out_path, b64=b64, error=None)
