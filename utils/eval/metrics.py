"""B1 baseline evaluation metrics.

提供：
- parse_json_robust(text)：从模型自由文本里提取 JSON
- try_load_pattern(data)：用 cad.api 加载，返回 Pattern 或 None
- structural_metrics(out_pattern, gt_pattern)：panel/edge/stitch 维度的结构匹配
- render_iou(out_pattern, gt_pattern, work_dir)：按 panel 名 sort 后渲染 panel-only 图，pixel mask IoU
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import numpy as np
from PIL import Image

from cad.core import Pattern
from cad.utils import pattern_panel_iou, stitch_match
from cad.utils.visualize import visualize


# ---------- JSON parsing ----------

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_json_robust(text: str) -> Optional[dict]:
    """Try several strategies to extract a JSON object from model output."""
    if not text:
        return None
    # 1. whole text is JSON
    s = text.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    # 2. ```json``` fence
    m = _JSON_FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 3. largest balanced {...} substring
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    continue
    return None


# ---------- Pattern loading ----------


def try_load_pattern(data: Optional[dict]) -> Optional[Pattern]:
    if data is None:
        return None
    try:
        return Pattern.load(data)
    except Exception:
        return None


# ---------- Structural metrics ----------


def structural_metrics(out: Optional[Pattern], gt: Pattern) -> dict:
    if out is None:
        return {
            "loadable": False,
            "panel_name_recall": 0.0,
            "panel_name_precision": 0.0,
            "panel_count_match": False,
            "stitch_count_match": False,
            "panel_vertex_count_match_rate": 0.0,
            "panel_edge_count_match_rate": 0.0,
        }

    out_names = {p.name for p in out.panels}
    gt_names = {p.name for p in gt.panels}
    inter = out_names & gt_names

    recall = len(inter) / len(gt_names) if gt_names else 0.0
    precision = len(inter) / len(out_names) if out_names else 0.0

    out_by_name = {p.name: p for p in out.panels}
    gt_by_name = {p.name: p for p in gt.panels}
    matched_v, matched_e, matched = 0, 0, 0
    for name in inter:
        op, gp = out_by_name[name], gt_by_name[name]
        ov = len(op.boundary.get_vertices())
        gv = len(gp.boundary.get_vertices())
        oe = len(op.boundary.edges)
        ge = len(gp.boundary.edges)
        if ov == gv:
            matched_v += 1
        if oe == ge:
            matched_e += 1
        matched += 1

    return {
        "loadable": True,
        "panel_name_recall": recall,
        "panel_name_precision": precision,
        "panel_count_match": len(out.panels) == len(gt.panels),
        "stitch_count_match": len(out.stitches) == len(gt.stitches),
        "panel_vertex_count_match_rate": matched_v / matched if matched else 0.0,
        "panel_edge_count_match_rate": matched_e / matched if matched else 0.0,
        "out_panel_count": len(out.panels),
        "out_stitch_count": len(out.stitches),
    }


# ---------- Render IoU ----------


def _render_sorted(pattern: Pattern, out_path: str) -> None:
    """Render with panels sorted by name (deterministic layout). Inspection-only — stitches included."""
    sorted_panels = sorted(pattern.panels, key=lambda p: p.name)
    saved = pattern.panels
    pattern.panels = sorted_panels
    try:
        visualize(pattern, out_path, show_stitches=True)
    finally:
        pattern.panels = saved


def _panel_mask(img_path: str, threshold: int = 240) -> np.ndarray:
    """Anything not near-white is considered panel pixel."""
    arr = np.array(Image.open(img_path).convert("L"))
    return arr < threshold


def render_for_inspection(
    out: Pattern,
    gt: Pattern,
    work_dir: str,
) -> dict:
    """Render the model output to PNG for visual inspection ONLY. Not used for IoU.
    GT visual reference is the dataset's target.png — no need to re-render."""
    os.makedirs(work_dir, exist_ok=True)
    result = {"out_render_path": os.path.join(work_dir, "output_render.png")}
    try:
        _render_sorted(out, result["out_render_path"])
    except Exception as e:
        result["out_render_error"] = str(e)
    return result


def stitch_metrics(out: Pattern, gt: Pattern, arc_threshold: float = 0.7) -> dict:
    """Stitch matching: ID exact then arc-length fuzzy fallback (cad.utils.stitch_match)."""
    r = stitch_match(out, gt, arc_threshold=arc_threshold)
    return {
        "f1": r["f1"],
        "precision": r["precision"],
        "recall": r["recall"],
        "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
        "id_matches": r["id_matches"],
        "arc_matches": r["arc_matches"],
        "n_out_stitches": r["n_stitches_a"],
        "n_gt_stitches": r["n_stitches_b"],
        "unmatched_out": r["unmatched_a"],
        "unmatched_gt": r["unmatched_b"],
    }


def panel_iou_metrics(out: Pattern, gt: Pattern) -> dict:
    """Per-panel IoU via cad's pattern_panel_iou — shape-only (scale-normalized).

    Each panel is independently rescaled to fill the rasterization canvas before
    IoU is computed, so absolute size mismatch is NOT penalized — only shape
    similarity is measured. Rationale: garment patterns vary by size (S/M/L);
    we evaluate whether the model got the panel *shape* right, not whether it
    matched a specific size.

    Symmetric: missing AND extra panels both penalized via union denominator.
    """
    r = pattern_panel_iou(out, gt, normalize_scale=True)
    return {
        "per_panel": r["per_panel"],
        "mean_iou_over_union": r["mean_iou_over_union"],
        "mean_iou_matched": r["mean_iou_matched"],
        "matched_names": r["matched_names"],
        "only_in_output": r["only_in_a"],
        "only_in_gt": r["only_in_b"],
    }


# ---------- Pass criteria ----------


def pass_loose(s: dict) -> bool:
    return (
        s.get("loadable", False)
        and s.get("panel_name_recall", 0) == 1.0
        and s.get("panel_name_precision", 0) == 1.0
    )
