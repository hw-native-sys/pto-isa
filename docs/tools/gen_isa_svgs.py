#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""Generate per-instruction SVG diagrams for PTO ISA docs.

Design goals:
- Use grid-based tiles to visualize the tiled data structure.
- Include a clear, per-instruction conceptual procedure (pseudocode).
- Keep diagrams tidy and consistent across instruction families.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "isa" / "manifest.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures" / "isa"


CANVAS_W = 1200
CANVAS_H = 720
MARGIN = 24
HEADER_Y = 46
HEADER_DIVIDER_Y = 104

DIAGRAM_TOP = HEADER_DIVIDER_Y + 24
EXPR_Y = DIAGRAM_TOP + 12
SRC_Y = DIAGRAM_TOP + 50
DST_Y = DIAGRAM_TOP + 232

PROC_BOX_Y = 488
PROC_PAD = 16

TILE_ROWS = 5
TILE_COLS = 5
CELL = 22

# One representative element used for callouts.
EX_R = 1
EX_C = 2

ARROW_PAD = 12

COLOR_BY_TEMPLATE = {
    "elementwise": ("#2D5BCE", "#EAF2FF"),
    "scalar": ("#1D8E63", "#E9F7F1"),
    "reduce_expand": ("#C46A1C", "#FFF4E8"),
    "memory": ("#6A47C4", "#F0EDFF"),
    "matmul": ("#1B7F91", "#E8F8FB"),
    "reshape_move": ("#515151", "#F5F5F5"),
    "complex": ("#C53A79", "#FDF0F6"),
    "sync": ("#A37000", "#FFF7D6"),
    "config": ("#4C8A25", "#EEF7E6"),
}


def _esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def load_manifest(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("instructions", [])
    if not isinstance(entries, list):
        raise ValueError("manifest 'instructions' must be a list")
    return entries


def _tile_cell_text(prefix: str, r: int, c: int) -> str:
    return f"{prefix}{r}{c}"


def _tile_width(cols: int) -> int:
    return cols * CELL


def _tile_height(rows: int) -> int:
    return rows * CELL


def _cell_center(x: int, y: int, r: int, c: int) -> Tuple[int, int]:
    cx = x + c * CELL + CELL // 2
    cy = y + r * CELL + CELL // 2
    return (cx, cy)


def _draw_text_lines(
    out: List[str],
    x: int,
    y: int,
    lines: Sequence[str],
    cls: str,
    line_height: int,
) -> None:
    out.append(f'<text x="{x}" y="{y}" class="{cls}" xml:space="preserve">')
    first = True
    for ln in lines:
        if first:
            out.append(f'  <tspan x="{x}" dy="0">{_esc(ln)}</tspan>')
            first = False
        else:
            out.append(f'  <tspan x="{x}" dy="{line_height}">{_esc(ln)}</tspan>')
    out.append("</text>")


def _draw_tile_grid(
    out: List[str],
    *,
    x: int,
    y: int,
    label: str,
    prefix: str,
    rows: int = TILE_ROWS,
    cols: int = TILE_COLS,
    highlight_cells: Iterable[Tuple[int, int]] = (),
    highlight_rows: Iterable[int] = (),
    highlight_cols: Iterable[int] = (),
    valid_box: Optional[Tuple[int, int]] = None,
    text_override: Optional[Dict[Tuple[int, int], str]] = None,
    accent: str,
) -> None:
    highlight = set(highlight_cells)
    highlight_r = set(highlight_rows)
    highlight_c = set(highlight_cols)
    text_override = text_override or {}

    w = _tile_width(cols)
    h = _tile_height(rows)

    if valid_box is None:
        # Use a schematic (not full) valid region to make masking visible in most diagrams.
        vr = rows - 1 if rows > 2 else rows
        vc = cols - 1 if cols > 2 else cols
        valid_box = (vr, vc)

    vr, vc = valid_box
    vr = max(0, min(rows, vr))
    vc = max(0, min(cols, vc))

    out.append(f'<text x="{x + w // 2}" y="{y - 10}" class="tileLabel" text-anchor="middle">{_esc(label)}</text>')
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="tileBorder" />')

    for r in range(rows):
        for c in range(cols):
            is_hl = (r, c) in highlight or r in highlight_r or c in highlight_c
            rx = x + c * CELL
            ry = y + r * CELL
            masked = (r >= vr) or (c >= vc)
            cls_parts = ["cell"]
            if masked:
                cls_parts.append("cellMasked")
            if is_hl:
                cls_parts.append("cellHL")
            cls = " ".join(cls_parts)
            if is_hl:
                out.append(f'<rect x="{rx}" y="{ry}" width="{CELL}" height="{CELL}" class="{cls}" stroke="{accent}" />')
            else:
                out.append(f'<rect x="{rx}" y="{ry}" width="{CELL}" height="{CELL}" class="{cls}" />')

            text = text_override.get((r, c))
            if text:
                cx, cy = _cell_center(x, y, r, c)
                out.append(
                    f'<text x="{cx}" y="{cy + 1}" class="cellText" text-anchor="middle" dominant-baseline="middle">{_esc(text)}</text>'
                )

    if vr and vc:
        out.append(f'<rect x="{x}" y="{y}" width="{vc * CELL}" height="{vr * CELL}" class="validBox" stroke="{accent}" />')

        # Indicate the valid extents (Rv/Cv) without hard-coding numeric sizes.
        out.append(f'<text x="{x + 4}" y="{y + vr * CELL - 4}" class="axisText">{_esc("Rv")}</text>')
        out.append(f'<text x="{x + vc * CELL - 4}" y="{y + 12}" class="axisText" text-anchor="end">{_esc("Cv")}</text>')

    # r/c axes indicator (schematic): row increases downward, col increases to the right.
    _ = prefix  # reserved for future per-cell callouts
    if rows >= 3 and cols >= 3:
        ax = x + 10
        ay = y + 10
        out.append(f'<path d="M {ax} {ay} L {ax + 34} {ay}" class="axisLine" marker-end="url(#axisArrow)" />')
        out.append(f'<path d="M {ax} {ay} L {ax} {ay + 34}" class="axisLine" marker-end="url(#axisArrow)" />')
        out.append(f'<text x="{ax + 38}" y="{ay + 4}" class="axisText">{_esc("c")}</text>')
        out.append(f'<text x="{ax - 2}" y="{ay + 38}" class="axisText" text-anchor="end">{_esc("r")}</text>')


def _draw_scalar_box(out: List[str], *, x: int, y: int, label: str, value: str, accent: str) -> None:
    w = 160
    h = 54
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" class="scalarBox" />')
    out.append(f'<text x="{x + 12}" y="{y + 20}" class="smallLabel">{_esc(label)}</text>')
    out.append(f'<text x="{x + 12}" y="{y + 42}" class="scalarValue" fill="{accent}">{_esc(value)}</text>')


def _draw_mem_row(
    out: List[str],
    *,
    x: int,
    y: int,
    label: str,
    prefix: str,
    cells: int = 12,
    highlight_idx: Optional[int] = None,
    accent: Optional[str] = None,
    text_override: Optional[Dict[int, str]] = None,
) -> None:
    _ = prefix
    text_override = text_override or {}
    cell_w = CELL
    w = cells * cell_w
    h = CELL
    out.append(f'<text x="{x + w // 2}" y="{y - 10}" class="tileLabel" text-anchor="middle">{_esc(label)}</text>')
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="tileBorder" />')
    for i in range(cells):
        rx = x + i * cell_w
        is_hl = highlight_idx is not None and i == highlight_idx
        cls = "cell"
        if is_hl:
            cls += " cellHL"
        if is_hl and accent:
            out.append(f'<rect x="{rx}" y="{y}" width="{cell_w}" height="{CELL}" class="{cls}" stroke="{accent}" />')
        else:
            out.append(f'<rect x="{rx}" y="{y}" width="{cell_w}" height="{CELL}" class="{cls}" />')
        text = text_override.get(i)
        if text:
            out.append(
                f'<text x="{rx + cell_w // 2}" y="{y + CELL // 2 + 1}" class="cellText" text-anchor="middle" dominant-baseline="middle">{_esc(text)}</text>'
            )


def _layout_row_lefts(center_x: int, widths: Sequence[int], gap: int) -> List[int]:
    if not widths:
        return []
    total = sum(widths) + gap * (len(widths) - 1)
    start = int(center_x - total / 2)
    xs: List[int] = []
    cur = start
    for w in widths:
        xs.append(cur)
        cur += w + gap
    return xs


def _cell_anchor_top(x: int, y: int, r: int, c: int) -> Tuple[int, int]:
    return (x + c * CELL + CELL // 2, y + r * CELL)


def _cell_anchor_bottom(x: int, y: int, r: int, c: int) -> Tuple[int, int]:
    return (x + c * CELL + CELL // 2, y + (r + 1) * CELL)


def _cell_anchor_left(x: int, y: int, r: int, c: int) -> Tuple[int, int]:
    return (x + c * CELL, y + r * CELL + CELL // 2)


def _cell_anchor_right(x: int, y: int, r: int, c: int) -> Tuple[int, int]:
    return (x + (c + 1) * CELL, y + r * CELL + CELL // 2)


def _tile_port_top(*, x: int, y: int, rows: int, cols: int, c: int) -> Tuple[int, int]:
    _ = rows
    _ = cols
    return (x + c * CELL + CELL // 2, y - ARROW_PAD)


def _tile_port_bottom(*, x: int, y: int, rows: int, cols: int, c: int) -> Tuple[int, int]:
    _ = cols
    return (x + c * CELL + CELL // 2, y + rows * CELL + ARROW_PAD)


def _scalar_port_bottom(*, x: int, y: int, w: int = 160, h: int = 54) -> Tuple[int, int]:
    return (x + w // 2, y + h + ARROW_PAD)


def _scalar_port_top(*, x: int, y: int, w: int = 160, h: int = 54) -> Tuple[int, int]:
    _ = h
    return (x + w // 2, y - ARROW_PAD)


def _draw_op_node(
    out: List[str],
    *,
    cx: int,
    cy: int,
    instr: str,
    accent: str,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Draw a circled-square op node and return (left, right, bottom) anchors."""
    max_chars_per_line = 6
    label_lines = [instr[i : i + max_chars_per_line] for i in range(0, len(instr), max_chars_per_line)]
    line_count = max(1, len(label_lines))

    if line_count == 1:
        font_px = 10
    elif line_count == 2:
        font_px = 8
    else:
        font_px = 7

    line_gap = font_px + 1
    text_w = max(len(line) for line in label_lines) * font_px * 0.62
    text_h = line_count * line_gap - 1
    side = int(max(24, text_w + 8, text_h + 8))
    side = min(side, 34)
    r = max(18, side // 2 + 5)

    out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" class="opCircle" stroke="{_esc(accent)}" />')
    out.append(
        f'<rect x="{cx - side // 2}" y="{cy - side // 2}" width="{side}" height="{side}" rx="6" class="opRect" stroke="{_esc(accent)}" />'
    )
    y0 = cy - ((line_count - 1) * line_gap) / 2
    for idx, line in enumerate(label_lines):
        y = y0 + idx * line_gap + 3
        out.append(
            f'<text x="{cx}" y="{y:.1f}" class="opText" font-size="{font_px}px" text-anchor="middle">{_esc(line)}</text>'
        )
    return ((cx - r, cy), (cx + r, cy), (cx, cy + r))


def _draw_binary_flow(
    out: List[str],
    *,
    instr: str,
    left_src: Tuple[int, int],
    right_src: Tuple[int, int],
    dst: Tuple[int, int],
    accent: str,
    op_cx: Optional[int] = None,
    op_cy: Optional[int] = None,
) -> None:
    """Route two sources through a circled-square op node into one destination."""
    # Ensure "left" and "right" are geometrically left/right for clear diagrams.
    l_src, r_src = (left_src, right_src) if left_src[0] <= right_src[0] else (right_src, left_src)
    dx, dy = dst

    if op_cx is None:
        op_cx = int((l_src[0] + r_src[0]) / 2)
    if op_cy is None:
        op_cy = int((max(l_src[1], r_src[1]) + dy) / 2)

    left, right, bottom = _draw_op_node(out, cx=op_cx, cy=op_cy, instr=instr, accent=accent)
    _draw_ortho_arrow(out, x1=l_src[0], y1=l_src[1], x2=left[0], y2=left[1], via_y=left[1], accent=accent)
    _draw_ortho_arrow(out, x1=r_src[0], y1=r_src[1], x2=right[0], y2=right[1], via_y=right[1], accent=accent)
    _draw_ortho_arrow(
        out,
        x1=bottom[0],
        y1=bottom[1],
        x2=dx,
        y2=dy,
        via_y=int((bottom[1] + dy) / 2),
        accent=accent,
    )


def _mem_anchor_top(x: int, y: int, i: int) -> Tuple[int, int]:
    return (x + i * CELL + CELL // 2, y - ARROW_PAD)


def _mem_anchor_bottom(x: int, y: int, i: int) -> Tuple[int, int]:
    return (x + i * CELL + CELL // 2, y + CELL + ARROW_PAD)


def _mem_anchor_left(x: int, y: int, i: int) -> Tuple[int, int]:
    return (x + i * CELL - ARROW_PAD, y + CELL // 2)


def _mem_anchor_right(x: int, y: int, i: int) -> Tuple[int, int]:
    return (x + (i + 1) * CELL + ARROW_PAD, y + CELL // 2)


def _draw_ortho_arrow(
    out: List[str],
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    accent: str,
    via_y: Optional[int] = None,
    via_x: Optional[int] = None,
) -> None:
    if via_y is None and via_x is None:
        via_y = int((y1 + y2) / 2)
    if via_y is not None and via_x is not None:
        raise ValueError("specify at most one of via_y/via_x")
    if via_y is not None:
        d = f"M {x1} {y1} L {x1} {via_y} L {x2} {via_y} L {x2} {y2}"
    else:
        assert via_x is not None
        d = f"M {x1} {y1} L {via_x} {y1} L {via_x} {y2} L {x2} {y2}"
    out.append(f'<path d="{d}" class="arrow" stroke="{accent}" marker-end="url(#arrow)" />')


def _begin_svg(instr: str, summary: str, template: str, accent: str, bg: str) -> List[str]:
    aria = f"{instr} tile operation diagram"
    out: List[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}" role="img" aria-label="{_esc(aria)}">'
    )
    out.append("<defs>")
    out.append(
        f'  <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L0,12 L12,6 z" fill="{_esc(accent)}"/></marker>'
    )
    out.append(
        '  <marker id="axisArrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L0,10 L10,5 z" fill="#64748b"/></marker>'
    )
    out.append("</defs>")
    out.append("<style>")
    out.append(
        "\n".join(
            [
                "svg { font-family: Arial, Helvetica, sans-serif; }",
                ".title { font-size: 30px; font-weight: 700; fill: #0f172a; }",
                ".subtitle { font-size: 14px; fill: #334155; }",
                ".meta { font-size: 12px; fill: #64748b; }",
                ".frame { fill: white; }",
                ".panel { fill: " + bg + "; stroke: #e2e8f0; stroke-width: 1.5; rx: 14; }",
                ".tileLabel { font-size: 14px; font-weight: 700; fill: #0f172a; }",
                ".tileBorder { fill: none; stroke: #475569; stroke-width: 1.5; }",
                ".cell { fill: #ffffff; stroke: #94a3b8; stroke-width: 1; }",
                ".cellMasked { fill: #e2e8f0; }",
                ".cellHL { stroke-width: 2; }",
                ".cellText { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 10px; fill: #0f172a; }",
                ".arrow { stroke-width: 2.5; fill: none; stroke-linejoin: round; stroke-linecap: round; }",
                ".axisLine { stroke: #64748b; stroke-width: 1.5; fill: none; }",
                ".axisText { font-size: 10px; fill: #64748b; font-weight: 700; }",
                ".opCircle { fill: #ffffff; stroke-width: 2; }",
                ".opRect { fill: #ffffff; stroke-width: 2; }",
                ".opText { font-size: 10px; font-weight: 800; fill: #0f172a; }",
                ".procBox { fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1.5; rx: 12; }",
                ".procTitle { font-size: 14px; font-weight: 700; fill: #0f172a; }",
                ".procText { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 12px; fill: #0f172a; }",
                ".smallLabel { font-size: 12px; fill: #334155; }",
                ".scalarBox { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1.5; }",
                ".scalarValue { font-size: 16px; font-weight: 700; }",
                ".validBox { fill: none; stroke-width: 2; stroke-dasharray: 6 4; }",
            ]
        )
    )
    out.append("</style>")

    out.append(f'<rect x="0" y="0" width="{CANVAS_W}" height="{CANVAS_H}" class="frame" />')
    out.append(
        f'<rect x="{MARGIN}" y="{MARGIN}" width="{CANVAS_W - 2 * MARGIN}" height="{CANVAS_H - 2 * MARGIN}" class="panel" />'
    )
    out.append(f'<text x="{MARGIN + 16}" y="{HEADER_Y}" class="title">{_esc(instr)}</text>')
    out.append(f'<text x="{MARGIN + 16}" y="{HEADER_Y + 26}" class="subtitle">{_esc(summary)}</text>')
    out.append(f'<text x="{MARGIN + 16}" y="{HEADER_Y + 46}" class="meta">Template: {_esc(template)}</text>')
    out.append(
        f'<text x="{CANVAS_W - MARGIN - 16}" y="{HEADER_Y + 46}" class="meta" text-anchor="end">Legend: outline=example; dashed=valid rows/cols (Rv,Cv); shaded=masked; r down / c right; ortho arrows=dataflow</text>'
    )
    out.append(
        f'<line x1="{MARGIN + 12}" y1="{HEADER_DIVIDER_Y}" x2="{CANVAS_W - MARGIN - 12}" y2="{HEADER_DIVIDER_Y}" stroke="#e2e8f0" stroke-width="1.5" />'
    )
    return out


def _end_svg(out: List[str]) -> str:
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _draw_procedure(out: List[str], *, lines: Sequence[str], accent: str) -> None:
    x = MARGIN + 16
    y = PROC_BOX_Y + 14
    w = CANVAS_W - 2 * (MARGIN + 16)
    h = CANVAS_H - PROC_BOX_Y - MARGIN - 14
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="procBox" />')
    out.append(f'<text x="{x + PROC_PAD}" y="{y + 26}" class="procTitle">Procedure (conceptual)</text>')
    _draw_text_lines(out, x + PROC_PAD, y + 52, lines, "procText", 16)
    out.append(
        f'<text x="{x + PROC_PAD}" y="{y + h - 16}" class="meta">Note: semantics apply to the valid region unless stated otherwise.</text>'
    )


def _elementwise_spec(instr: str) -> Tuple[List[str], str, List[str]]:
    unary = {
        "TABS": "abs(src)",
        "TEXP": "exp(src)",
        "TLOG": "log(src)",
        "TNEG": "-src",
        "TNOT": "~src",
        "TRECIP": "1/src",
        "TRELU": "relu(src)",
        "TRSQRT": "rsqrt(src)",
        "TSQRT": "sqrt(src)",
        "TCVT": "convert(src, roundMode)",
    }
    binary = {
        "TADD": "src0 + src1",
        "TSUB": "src0 - src1",
        "TMUL": "src0 * src1",
        "TDIV": "src0 / src1",
        "TMIN": "min(src0, src1)",
        "TMAX": "max(src0, src1)",
        "TAND": "src0 & src1",
        "TOR": "src0 | src1",
        "TXOR": "src0 ^ src1",
        "TSHL": "src0 << src1",
        "TSHR": "src0 >> src1",
        "TREM": "remainder(src0, src1)",
        "TFMOD": "fmod(src0, src1)",
    }
    ternary = {
        "TADDC": "src0 + src1 + src2",
        "TSUBC": "src0 - src1 + src2",
    }

    if instr in unary:
        expr = f"dst[r,c] = {unary[instr]}"
        proc = [
            "for r in 0..Rv-1:",
            "  for c in 0..Cv-1:",
            f"    {expr}",
        ]
        return (["src"], expr, proc)
    if instr in ternary:
        expr = f"dst[r,c] = {ternary[instr]}"
        proc = [
            "for r in 0..Rv-1:",
            "  for c in 0..Cv-1:",
            f"    {expr}",
        ]
        return (["src0", "src1", "src2"], expr, proc)
    if instr == "TSEL":
        expr = "dst[r,c] = (mask[r,c] != 0) ? src0[r,c] : src1[r,c]"
        proc = [
            "for r in 0..Rv-1:",
            "  for c in 0..Cv-1:",
            f"    {expr}",
        ]
        return (["mask", "src0", "src1"], expr, proc)
    if instr == "TCMP":
        expr = "dst_mask[r,c] = cmp(src0[r,c], src1[r,c])"
        proc = [
            "for r in 0..Rv-1:",
            "  for c in 0..Cv-1:",
            f"    {expr}",
        ]
        return (["src0", "src1"], expr, proc)
    if instr == "TPRELU":
        expr = "dst[r,c] = (x>0) ? x : slope*x"
        proc = [
            "for r in 0..Rv-1:",
            "  for c in 0..Cv-1:",
            "    x = src0[r,c]",
            "    slope = src1[r,c]",
            f"    {expr}",
        ]
        return (["src0", "src1"], expr, proc)
    if instr in binary:
        expr = f"dst[r,c] = {binary[instr]}"
        proc = [
            "for r in 0..Rv-1:",
            "  for c in 0..Cv-1:",
            f"    {expr}",
        ]
        return (["src0", "src1"], expr, proc)

    expr = "dst[r,c] = op(src...)[r,c]"
    proc = [
        "for r in 0..Rv-1:",
        "  for c in 0..Cv-1:",
        f"    {expr}",
    ]
    return (["src0", "src1"], expr, proc)


def _scalar_spec(instr: str) -> Tuple[List[str], str, List[str]]:
    tile_scalar = {
        "TADDS": "src[r,c] + s",
        "TSUBS": "src[r,c] - s",
        "TMULS": "src[r,c] * s",
        "TDIVS": "src[r,c] / s   (or s / src[r,c])",
        "TMAXS": "max(src[r,c], s)",
        "TMINS": "min(src[r,c], s)",
        "TANDS": "src[r,c] & s",
        "TORS": "src[r,c] | s",
        "TXORS": "src[r,c] ^ s",
        "TSHLS": "src[r,c] << s",
        "TSHRS": "src[r,c] >> s",
        "TFMODS": "fmod(src[r,c], s)",
        "TREMS": "remainder(src[r,c], s)",
    }

    if instr == "TEXPANDS":
        expr = "dst[r,c] = s"
        proc = [
            "for r in 0..Rv-1:",
            "  for c in 0..Cv-1:",
            f"    {expr}",
        ]
        return (["src(tile)"], expr, proc)
    if instr == "TCMPS":
        expr = "dst_mask[r,c] = cmp(src[r,c], s)"
        proc = [
            "for r in 0..Rv-1:",
            "  for c in 0..Cv-1:",
            f"    {expr}",
        ]
        return (["src(tile)"], expr, proc)
    if instr == "TSELS":
        expr = "dst = (selectMode) ? src0 : src1"
        proc = [
            "if selectMode:",
            "  dst = src0",
            "else:",
            "  dst = src1",
        ]
        return (["src0", "src1"], expr, proc)
    if instr == "TLRELU":
        expr = "dst[r,c] = (x>0) ? x : slope*x"
        proc = [
            "for r in 0..Rv-1:",
            "  for c in 0..Cv-1:",
            "    x = src[r,c]",
            f"    {expr}",
        ]
        return (["src(tile)"], expr, proc)
    if instr == "TADDSC":
        expr = "dst[r,c] = src0[r,c] + s + src1[r,c]"
        proc = [
            "for r in 0..Rv-1:",
            "  for c in 0..Cv-1:",
            f"    {expr}",
        ]
        return (["src0", "src1"], expr, proc)
    if instr == "TSUBSC":
        expr = "dst[r,c] = src0[r,c] - s + src1[r,c]"
        proc = [
            "for r in 0..Rv-1:",
            "  for c in 0..Cv-1:",
            f"    {expr}",
        ]
        return (["src0", "src1"], expr, proc)

    if instr in tile_scalar:
        expr = f"dst[r,c] = {tile_scalar[instr]}"
        proc = [
            "for r in 0..Rv-1:",
            "  for c in 0..Cv-1:",
            f"    {expr}",
        ]
        return (["src(tile)"], expr, proc)

    expr = "dst[r,c] = op(src[r,c], s)"
    proc = [
        "for r in 0..Rv-1:",
        "  for c in 0..Cv-1:",
        f"    {expr}",
    ]
    return (["src(tile)"], expr, proc)


def _reduce_expand_kind(instr: str) -> Tuple[str, str, str]:
    # Returns (mode, axis, op)
    # mode: reduce|expand|expand_op
    if instr in {"TROWSUM", "TROWMAX", "TROWMIN"}:
        return ("reduce", "row", instr.replace("TROW", "").lower())
    if instr in {"TCOLSUM", "TCOLMAX", "TCOLMIN"}:
        return ("reduce", "col", instr.replace("TCOL", "").lower())
    if instr in {"TROWEXPAND", "TCOLEXPAND"}:
        return ("expand", "row" if instr.startswith("TROW") else "col", "broadcast")
    if instr.startswith("TROWEXPAND"):
        return ("expand_op", "row", instr.replace("TROWEXPAND", "").lower() or "broadcast")
    if instr.startswith("TCOLEXPAND"):
        return ("expand_op", "col", instr.replace("TCOLEXPAND", "").lower() or "broadcast")
    return ("reduce_expand", "row", "op")


def _render_elementwise(instr: str, summary: str, accent: str, bg: str) -> str:
    inputs, expr, proc = _elementwise_spec(instr)
    out = _begin_svg(instr, summary, "elementwise", accent, bg)

    out.append(
        f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
    )

    tile_w = _tile_width(TILE_COLS)
    tile_h = _tile_height(TILE_ROWS)
    gap = 80
    y_src = SRC_Y
    y_dst = DST_Y

    if instr == "TSEL":
        prefixes = ["m", "a", "b"]
    else:
        prefixes = ["a", "b", "c"][: len(inputs)]

    xs = _layout_row_lefts(CANVAS_W // 2, [tile_w] * len(inputs), gap)
    for x, label, pfx in zip(xs, inputs, prefixes, strict=False):
        _draw_tile_grid(out, x=x, y=y_src, label=label, prefix=pfx, highlight_cells=[(EX_R, EX_C)], accent=accent)

    out_label = "dst(mask)" if instr == "TCMP" else "dst"
    out_prefix = "m" if instr == "TCMP" else "d"
    x_dst = (CANVAS_W - tile_w) // 2
    _draw_tile_grid(out, x=x_dst, y=y_dst, label=out_label, prefix=out_prefix, highlight_cells=[(EX_R, EX_C)], accent=accent)

    dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)

    # For binary ops, route through an explicit op node (circled square mnemonic).
    if len(inputs) == 2 and len(xs) == 2:
        s0x, s0y = _tile_port_bottom(x=xs[0], y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        s1x, s1y = _tile_port_bottom(x=xs[1], y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(s0x, s0y),
            right_src=(s1x, s1y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
    else:
        via_base = int((y_src + tile_h + y_dst) / 2)
        n = max(1, len(xs))
        for i, x in enumerate(xs):
            sx, sy = _tile_port_bottom(x=x, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
            via_y = via_base + int((i - (n - 1) / 2) * 14)
            _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)

    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _render_scalar(instr: str, summary: str, accent: str, bg: str) -> str:
    _inputs, expr, proc = _scalar_spec(instr)
    out = _begin_svg(instr, summary, "scalar", accent, bg)

    out.append(
        f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
    )

    tile_w = _tile_width(TILE_COLS)
    tile_h = _tile_height(TILE_ROWS)
    gap = 80
    y_src = SRC_Y
    y_dst = DST_Y

    src_labels = ["src0", "src1"] if instr in {"TADDSC", "TSUBSC", "TSELS"} else ["src"]
    src_prefixes = ["a", "b"] if len(src_labels) == 2 else ["a"]

    xs = _layout_row_lefts(CANVAS_W // 2, [tile_w] * len(src_labels), gap)
    for x, label, pfx in zip(xs, src_labels, src_prefixes, strict=False):
        _draw_tile_grid(out, x=x, y=y_src, label=label, prefix=pfx, highlight_cells=[(EX_R, EX_C)], accent=accent)

    scalar_label = "selectMode" if instr == "TSELS" else "scalar"
    scalar_value = "mode" if instr == "TSELS" else "s"
    scalar_x = CANVAS_W - MARGIN - 16 - 160
    scalar_y = y_src + 8
    _draw_scalar_box(out, x=scalar_x, y=scalar_y, label=scalar_label, value=scalar_value, accent=accent)

    out_label = "dst(mask)" if instr == "TCMPS" else "dst"
    out_prefix = "m" if instr == "TCMPS" else "d"
    x_dst = (CANVAS_W - tile_w) // 2
    _draw_tile_grid(out, x=x_dst, y=y_dst, label=out_label, prefix=out_prefix, highlight_cells=[(EX_R, EX_C)], accent=accent)

    dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    via_base = int((y_src + tile_h + y_dst) / 2)

    sources: List[Tuple[int, int]] = []
    for x in xs:
        sources.append(_tile_port_bottom(x=x, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C))
    sources.append(_scalar_port_bottom(x=scalar_x, y=scalar_y))

    if len(sources) == 2:
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=sources[0],
            right_src=sources[1],
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
    else:
        n = len(sources)
        for i, (sx, sy) in enumerate(sources):
            via_y = via_base + int((i - (n - 1) / 2) * 14)
            _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)

    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _render_reduce_expand(instr: str, summary: str, accent: str, bg: str) -> str:
    mode, axis, op = _reduce_expand_kind(instr)
    out = _begin_svg(instr, summary, "reduce_expand", accent, bg)

    tile_w = _tile_width(TILE_COLS)
    tile_h = _tile_height(TILE_ROWS)
    y_src = SRC_Y
    y_dst = DST_Y

    if mode == "reduce":
        if axis == "row":
            expr = f"dst[r,0] = {op}_c src[r,c]"
            proc = [
                "for r in 0..Rv-1:",
                f"  dst[r,0] = {op} over c=0..Cv-1 of src[r,c]",
            ]
            x_src = (CANVAS_W - tile_w) // 2
            x_dst = (CANVAS_W - _tile_width(1)) // 2
            _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", highlight_rows=[EX_R], accent=accent)
            _draw_tile_grid(
                out,
                x=x_dst,
                y=y_dst,
                label="dst (column vector)",
                prefix="d",
                rows=TILE_ROWS,
                cols=1,
                highlight_cells=[(EX_R, 0)],
                accent=accent,
            )
            sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
            dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=1, c=0)
        else:
            expr = f"dst[0,c] = {op}_r src[r,c]"
            proc = [
                "for c in 0..Cv-1:",
                f"  dst[0,c] = {op} over r=0..Rv-1 of src[r,c]",
            ]
            x_src = (CANVAS_W - tile_w) // 2
            dst_rows, dst_cols = 1, TILE_COLS
            dst_w, dst_h = _tile_width(dst_cols), _tile_height(dst_rows)
            x_dst = (CANVAS_W - dst_w) // 2
            y_dst2 = y_dst + (tile_h - dst_h) // 2
            _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", highlight_cols=[EX_C], accent=accent)
            _draw_tile_grid(
                out,
                x=x_dst,
                y=y_dst2,
                label="dst (row vector)",
                prefix="d",
                rows=dst_rows,
                cols=dst_cols,
                highlight_cells=[(0, EX_C)],
                accent=accent,
            )
            sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
            dx, dy = _tile_port_top(x=x_dst, y=y_dst2, rows=dst_rows, cols=dst_cols, c=EX_C)

        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        via_y = int((sy + dy) / 2)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if mode == "expand":
        if axis == "row":
            expr = "dst[r,c] = src[r,0]"
            proc = [
                "for r in 0..Rv-1:",
                "  v = src[r,0]",
                "  for c in 0..Cv-1:",
                "    dst[r,c] = v",
            ]
            x_src = (CANVAS_W - tile_w) // 2
            x_dst = (CANVAS_W - tile_w) // 2
            _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", highlight_cells=[(EX_R, 0)], accent=accent)
            _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst", prefix="d", highlight_rows=[EX_R], highlight_cells=[(EX_R, EX_C)], accent=accent)
            sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=0)
            dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        else:
            expr = "dst[r,c] = src[0,c]"
            proc = [
                "for c in 0..Cv-1:",
                "  v = src[0,c]",
                "  for r in 0..Rv-1:",
                "    dst[r,c] = v",
            ]
            x_src = (CANVAS_W - tile_w) // 2
            x_dst = (CANVAS_W - tile_w) // 2
            _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", highlight_cells=[(0, EX_C)], accent=accent)
            _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst", prefix="d", highlight_cols=[EX_C], highlight_cells=[(EX_R, EX_C)], accent=accent)
            sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
            dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)

        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        via_y = int((sy + dy) / 2)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    # expand_op
    if axis == "row":
        s_rows, s_cols = TILE_ROWS, 1
        s_h = _tile_height(s_rows)
        y_s = y_src
        s_cell = (EX_R, 0)
        scalar_ref = "s = src1[r,0]"
    else:
        s_rows, s_cols = 1, TILE_COLS
        s_h = _tile_height(s_rows)
        y_s = y_src + (tile_h - s_h) // 2
        s_cell = (0, EX_C)
        scalar_ref = "s = src1[0,c]"

    if op == "expdif":
        expr = "dst[r,c] = exp(src0[r,c] - s)"
    elif op == "add":
        expr = "dst[r,c] = src0[r,c] + s"
    elif op == "sub":
        expr = "dst[r,c] = src0[r,c] - s"
    elif op == "mul":
        expr = "dst[r,c] = src0[r,c] * s"
    elif op == "div":
        expr = "dst[r,c] = src0[r,c] / s"
    elif op == "max":
        expr = "dst[r,c] = max(src0[r,c], s)"
    elif op == "min":
        expr = "dst[r,c] = min(src0[r,c], s)"
    else:
        expr = "dst[r,c] = op(src0[r,c], s)"

    proc = [
        "for r in 0..Rv-1:",
        "  for c in 0..Cv-1:",
        f"    {scalar_ref}",
        f"    {expr}",
    ]

    widths = [tile_w, _tile_width(s_cols)]
    xs = _layout_row_lefts(CANVAS_W // 2, widths, 80)
    x_src0, x_src1 = xs[0], xs[1]
    x_dst = (CANVAS_W - tile_w) // 2

    _draw_tile_grid(out, x=x_src0, y=y_src, label="src0", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)
    _draw_tile_grid(
        out,
        x=x_src1,
        y=y_s,
        label="src1 (per-row)" if axis == "row" else "src1 (per-col)",
        prefix="s",
        rows=s_rows,
        cols=s_cols,
        highlight_cells=[s_cell],
        accent=accent,
    )
    _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent)

    out.append(
        f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
    )

    dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    a_x, a_y = _tile_port_bottom(x=x_src0, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    s_x, s_y = _tile_port_bottom(x=x_src1, y=y_s, rows=s_rows, cols=s_cols, c=s_cell[1])
    _draw_binary_flow(
        out,
        instr=instr,
        left_src=(a_x, a_y),
        right_src=(s_x, s_y),
        dst=(dx, dy),
        accent=accent,
        op_cx=CANVAS_W // 2,
    )

    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _render_memory(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "memory", accent, bg)
    tile_w = _tile_width(TILE_COLS)
    tile_h = _tile_height(TILE_ROWS)
    y_src = SRC_Y
    y_dst = DST_Y

    if instr in {"TLOAD", "TPREFETCH"}:
        expr = "dst[r,c] = GM[...]"
        proc = [
            "for r,c in valid(dst):",
            "  dst[r,c] = GM[base + (row0+r)*stride + (col0+c)]",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        mem_w = 12 * CELL
        x_mem = (CANVAS_W - mem_w) // 2
        y_mem = y_src + (tile_h - CELL) // 2
        _draw_mem_row(out, x=x_mem, y=y_mem, label="GlobalTensor / GM", prefix="g", highlight_idx=6, accent=accent)

        x_tile = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_tile, y=y_dst, label="dst tile", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent)

        sx, sy = _mem_anchor_bottom(x_mem, y_mem, 6)
        dx, dy = _tile_port_top(x=x_tile, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        via_y = int((sy + dy) / 2)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TSTORE":
        expr = "GM[...] = src[r,c]"
        proc = [
            "for r,c in valid(src):",
            "  GM[base + (row0+r)*stride + (col0+c)] = src[r,c]",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        x_src = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_src, y=y_src, label="src tile", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)

        mem_w = 12 * CELL
        x_mem = (CANVAS_W - mem_w) // 2
        y_mem = y_dst + (tile_h - CELL) // 2
        _draw_mem_row(out, x=x_mem, y=y_mem, label="GlobalTensor / GM", prefix="g", highlight_idx=6, accent=accent)

        sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        dx, dy = _mem_anchor_top(x_mem, y_mem, 6)
        via_y = int((sy + dy) / 2)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TSTORE_FP":
        expr = "GM[...] = quantize(src, fp)"
        proc = [
            "for r,c in valid(src):",
            "  q = quantize(src[r,c], fp[r,c])",
            "  GM[...] = q",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 80)
        x_src, x_fp = xs[0], xs[1]
        _draw_tile_grid(out, x=x_src, y=y_src, label="src (acc)", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)
        _draw_tile_grid(out, x=x_fp, y=y_src, label="fp/scale", prefix="s", highlight_cells=[(EX_R, EX_C)], accent=accent)

        mem_w = 12 * CELL
        x_mem = (CANVAS_W - mem_w) // 2
        y_mem = y_dst + (tile_h - CELL) // 2
        _draw_mem_row(out, x=x_mem, y=y_mem, label="GlobalTensor / GM", prefix="g", highlight_idx=6, accent=accent)

        dx, dy = _mem_anchor_top(x_mem, y_mem, 6)
        a_x, a_y = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        f_x, f_y = _tile_port_bottom(x=x_fp, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(a_x, a_y),
            right_src=(f_x, f_y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "MGATHER":
        expr = "dst[r,c] = mem[indexes[r,c]]"
        proc = [
            "for r,c in valid(dst):",
            "  idx = indexes[r,c]",
            "  dst[r,c] = mem[idx]",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        mem_w = 12 * CELL
        widths = [mem_w, tile_w]
        xs = _layout_row_lefts(CANVAS_W // 2, widths, 80)
        x_mem, x_idx = xs[0], xs[1]
        y_mem = y_src + (tile_h - CELL) // 2
        _draw_mem_row(out, x=x_mem, y=y_mem, label="GM / mem", prefix="g", highlight_idx=10, accent=accent)
        _draw_tile_grid(out, x=x_idx, y=y_src, label="idx tile", prefix="i", highlight_cells=[(EX_R, EX_C)], accent=accent)

        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst tile", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent)

        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        m_x, m_y = _mem_anchor_bottom(x_mem, y_mem, 10)
        i_x, i_y = _tile_port_bottom(x=x_idx, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(m_x, m_y),
            right_src=(i_x, i_y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "MSCATTER":
        expr = "mem[indexes[r,c]] = src[r,c]"
        proc = [
            "for r,c in valid(src):",
            "  idx = indexes[r,c]",
            "  mem[idx] = src[r,c]",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 80)
        x_src, x_idx = xs[0], xs[1]
        _draw_tile_grid(out, x=x_src, y=y_src, label="src tile", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)
        _draw_tile_grid(out, x=x_idx, y=y_src, label="idx tile", prefix="i", highlight_cells=[(EX_R, EX_C)], accent=accent)

        mem_w = 12 * CELL
        x_mem = (CANVAS_W - mem_w) // 2
        y_mem = y_dst + (tile_h - CELL) // 2
        _draw_mem_row(out, x=x_mem, y=y_mem, label="GM / mem", prefix="g", highlight_idx=10, accent=accent)

        dx, dy = _mem_anchor_top(x_mem, y_mem, 10)
        s_x, s_y = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        i_x, i_y = _tile_port_bottom(x=x_idx, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(s_x, s_y),
            right_src=(i_x, i_y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    _draw_procedure(out, lines=["(implementation-defined)"], accent=accent)
    return _end_svg(out)


def _render_matmul(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "matmul", accent, bg)

    tile_w = _tile_width(TILE_COLS)
    tile_h = _tile_height(TILE_ROWS)
    y_src = SRC_Y
    y_dst = DST_Y

    label_a = "A" if instr.startswith("TMATMUL") else "A (vec)"
    label_b = "B" if instr.startswith("TMATMUL") else "B (vec)"

    xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 120)
    x_a, x_b = xs[0], xs[1]
    x_c = (CANVAS_W - tile_w) // 2

    _draw_tile_grid(out, x=x_a, y=y_src, label=label_a, prefix="a", highlight_cells=[(EX_R, 1)], accent=accent)
    _draw_tile_grid(out, x=x_b, y=y_src, label=label_b, prefix="b", highlight_cells=[(1, EX_C)], accent=accent)
    _draw_tile_grid(out, x=x_c, y=y_dst, label="C / dst", prefix="c", highlight_cells=[(EX_R, EX_C)], accent=accent)

    if instr.startswith("TMATMUL"):
        expr = "C[i,j] = sum_k A[i,k] * B[k,j]"
    else:
        expr = "C[0,j] = sum_k A[0,k] * B[k,j]"

    out.append(
        f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
    )

    dx, dy = _tile_port_top(x=x_c, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    a_x, a_y = _tile_port_bottom(x=x_a, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=1)
    b_x, b_y = _tile_port_bottom(x=x_b, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    _draw_binary_flow(
        out,
        instr=instr,
        left_src=(a_x, a_y),
        right_src=(b_x, b_y),
        dst=(dx, dy),
        accent=accent,
        op_cx=CANVAS_W // 2,
    )

    proc = [
        "for i,j in output valid region:",
        "  acc = 0",
        "  for k in 0..K-1:",
        "    acc += A[i,k] * B[k,j]",
        "  C[i,j] = acc   (+ bias/acc, if applicable)",
    ]
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _render_reshape_move(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "reshape_move", accent, bg)

    tile_w = _tile_width(TILE_COLS)
    tile_h = _tile_height(TILE_ROWS)
    y_src = SRC_Y
    y_dst = DST_Y

    if instr.startswith("TEXTRACT"):
        expr = "dst = slice(src, offset)"
        proc = [
            "for r,c in valid(dst):",
            "  dst[r,c] = src[r + row_off, c + col_off]",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        x_src = (CANVAS_W - tile_w) // 2
        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", valid_box=(3, 3), highlight_cells=[(0, 0)], accent=accent)
        _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst (window)", prefix="d", highlight_cells=[(0, 0)], accent=accent)
        sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=0)
        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=0)
        via_y = int((sy + dy) / 2)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr.startswith("TINSERT"):
        expr = "dst[off + (r,c)] = src[r,c]"
        proc = [
            "for r,c in valid(src):",
            "  dst[r + row_off, c + col_off] = src[r,c]",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 120)
        x_dst_in, x_src_win = xs[0], xs[1]
        x_dst_out = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_dst_in, y=y_src, label="dst (in)", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent)
        _draw_tile_grid(out, x=x_src_win, y=y_src, label="src (window)", prefix="a", valid_box=(2, 2), highlight_cells=[(0, 0)], accent=accent)
        _draw_tile_grid(out, x=x_dst_out, y=y_dst, label="dst (out)", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent)
        dx, dy = _tile_port_top(x=x_dst_out, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        d_x, d_y = _tile_port_bottom(x=x_dst_in, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        s_x, s_y = _tile_port_bottom(x=x_src_win, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=0)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(d_x, d_y),
            right_src=(s_x, s_y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr.startswith("TFILLPAD"):
        expr = "dst = pad(src, pad_value)"
        proc = [
            "for r,c in full tile domain:",
            "  if (r,c) in valid(src):",
            "    dst[r,c] = src[r,c]",
            "  else:",
            "    dst[r,c] = pad_value",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        x_src = (CANVAS_W - tile_w) // 2
        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_src, y=y_src, label="src (valid)", prefix="a", valid_box=(3, 3), highlight_cells=[(EX_R, EX_C)], accent=accent)
        _draw_tile_grid(
            out, x=x_dst, y=y_dst, label="dst (padded)", prefix="d", valid_box=(3, 3), highlight_cells=[(3, 3)], accent=accent
        )
        sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        via_y = int((sy + dy) / 2)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TTRANS":
        expr = "dst[r,c] = src[c,r]"
        proc = [
            "for r,c in valid(dst):",
            "  dst[r,c] = src[c,r]",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        x_src = (CANVAS_W - tile_w) // 2
        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)
        _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst", prefix="d", highlight_cells=[(EX_C, EX_R)], accent=accent)
        sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_R)
        via_y = int((sy + dy) / 2)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr.startswith("TSUBVIEW"):
        expr = "dst = subview(src, rI, cI)"
        proc = [
            "for r,c in valid(dst):",
            "  dst[r,c] = src[r + rI, c + cI]",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        x_src = (CANVAS_W - tile_w) // 2
        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(
            out,
            x=x_src,
            y=y_src,
            label="src (rI = 0 and cI = 0)",
            prefix="a",
            valid_box=(5, 5),
            highlight_cells=[(0, 0)],
            accent=accent)
        _draw_tile_grid(
            out,
            x=x_dst,
            y=y_dst,
            label="dst (subtile)",
            valid_box=(3, 3),
            prefix="d",
            highlight_cells=[(0, 0)],
            accent=accent
        )
        sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=0)
        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=0)
        via_y = int((sy + dy) / 2)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    # Default: movement/reshape
    expr = "dst = move/reshape(src)"
    proc = [
        "for r,c in valid(dst):",
        "  dst[r,c] = transform(src[r,c])   (layout/location dependent)",
    ]
    out.append(
        f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
    )
    x_src = (CANVAS_W - tile_w) // 2
    x_dst = (CANVAS_W - tile_w) // 2
    _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)
    _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent)
    sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
    via_y = int((sy + dy) / 2)
    _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _render_complex(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "complex", accent, bg)

    tile_w = _tile_width(TILE_COLS)
    tile_h = _tile_height(TILE_ROWS)
    y_src = SRC_Y
    y_dst = DST_Y

    if instr == "TCI":
        expr = "dst[r,c] = base + r*stride + c"
        proc = [
            "for r in 0..Rv-1:",
            "  for c in 0..Cv-1:",
            f"    {expr}",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        scalar_gap = 40
        scalar_y = y_src + 8
        x_base = (CANVAS_W - (160 * 2 + scalar_gap)) // 2
        x_stride = x_base + 160 + scalar_gap
        _draw_scalar_box(out, x=x_base, y=scalar_y, label="base", value="base", accent=accent)
        _draw_scalar_box(out, x=x_stride, y=scalar_y, label="stride", value="stride", accent=accent)

        override: Dict[Tuple[int, int], str] = {(EX_R, EX_C): "base+..."}
        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst", prefix="d", text_override=override, highlight_cells=[(EX_R, EX_C)], accent=accent)
        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        sources = [_scalar_port_bottom(x=x_base, y=scalar_y), _scalar_port_bottom(x=x_stride, y=scalar_y)]
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=sources[0],
            right_src=sources[1],
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TTRI":
        expr = "mask[r,c] = (r >= c) ? 1 : 0"
        proc = [
            "for r in 0..Rv-1:",
            "  for c in 0..Cv-1:",
            f"    {expr}",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        scalar_x = (CANVAS_W - 160) // 2
        scalar_y = y_src + 8
        _draw_scalar_box(out, x=scalar_x, y=scalar_y, label="pattern", value="r>=c", accent=accent)

        override = {(r, c): ("1" if r >= c else "0") for r in range(TILE_ROWS) for c in range(TILE_COLS)}
        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_dst, y=y_dst, label="mask tile", prefix="m", text_override=override, highlight_cells=[(EX_R, EX_C)], accent=accent)
        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        sx, sy = _scalar_port_bottom(x=scalar_x, y=scalar_y)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=int((sy + dy) / 2), accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr in {"TGATHER", "TGATHERB"}:
        expr = "dst[r,c] = src0[ indices[r,c] ]"
        proc = [
            "for r,c in valid(dst):",
            "  k = indices[r,c]",
            "  dst[r,c] = src0[k]",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 120)
        x_src0, x_idx = xs[0], xs[1]
        idx_text = {(EX_R, EX_C): "k"}
        _draw_tile_grid(out, x=x_src0, y=y_src, label="src0", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)
        _draw_tile_grid(out, x=x_idx, y=y_src, label="indices", prefix="i", highlight_cells=[(EX_R, EX_C)], text_override=idx_text, accent=accent)

        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent)

        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        a_x, a_y = _tile_port_bottom(x=x_src0, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        i_x, i_y = _tile_port_bottom(x=x_idx, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(a_x, a_y),
            right_src=(i_x, i_y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TSCATTER":
        expr = "dst[ idx[r,c], c ] = src[r,c]"
        proc = [
            "for r,c in valid(src):",
            "  rr = idx[r,c]",
            "  dst[rr, c] = src[r,c]",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 120)
        x_src, x_idx = xs[0], xs[1]
        idx_text = {(EX_R, EX_C): "rr"}
        _draw_tile_grid(out, x=x_src, y=y_src, label="src", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)
        _draw_tile_grid(out, x=x_idx, y=y_src, label="row idx", prefix="i", highlight_cells=[(EX_R, EX_C)], text_override=idx_text, accent=accent)

        x_dst = (CANVAS_W - tile_w) // 2
        dst_r = min(TILE_ROWS - 2, EX_R + 1) if TILE_ROWS > 2 else EX_R
        _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst", prefix="d", highlight_cells=[(dst_r, EX_C)], accent=accent)

        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        s_x, s_y = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        i_x, i_y = _tile_port_bottom(x=x_idx, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(s_x, s_y),
            right_src=(i_x, i_y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TSORT32":
        expr = "dst[i,k] = src[i, pi_i(k)] ; idx = pi"
        proc = [
            "for each row i:",
            "  (dst_row, idx_row) = sort_with_indices(src_row)",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        x_src = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_src, y=y_src, label="src (block)", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)

        xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 120)
        x_dst, x_idx = xs[0], xs[1]
        _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst (sorted)", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent)
        _draw_tile_grid(out, x=x_idx, y=y_dst, label="idx (perm)", prefix="p", highlight_cells=[(EX_R, EX_C)], accent=accent)

        s_x, s_y = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        d1_x, d1_y = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        d2_x, d2_y = _tile_port_top(x=x_idx, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        via_base = int((y_src + tile_h + y_dst) / 2)
        _draw_ortho_arrow(out, x1=s_x, y1=s_y, x2=d1_x, y2=d1_y, via_y=via_base - 10, accent=accent)
        _draw_ortho_arrow(out, x1=s_x, y1=s_y, x2=d2_x, y2=d2_y, via_y=via_base + 10, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TMRGSORT":
        expr = "dst = merge(src0, src1, ...)"
        proc = [
            "dst = merge(sorted_lists...)",
            "(ordering/format are implementation-defined)",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 120)
        x0, x1 = xs[0], xs[1]
        _draw_tile_grid(out, x=x0, y=y_src, label="src0 (sorted)", prefix="a", accent=accent)
        _draw_tile_grid(out, x=x1, y=y_src, label="src1 (sorted)", prefix="b", accent=accent)
        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst (merged)", prefix="d", highlight_cells=[(EX_R, EX_C)], accent=accent)
        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        a_x, a_y = _tile_port_bottom(x=x0, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        b_x, b_y = _tile_port_bottom(x=x1, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(a_x, a_y),
            right_src=(b_x, b_y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TQUANT":
        expr = "dst, exp, max, scaling = quantize(src, mode)"
        proc = [
            "max = max(abs(src))",
            "scaling = compute_scaling(max, mode)",
            "for r,c in valid(src):",
            "  (dst[r,c], exp[r,c]) = quantize(src[r,c], scaling, mode)",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        x_src = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_src, y=y_src, label="src (fp32)", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)

        # Outputs: exp tile, dst tile, scaling/max (vector).
        scale_rows, scale_cols = 1, TILE_COLS
        scale_h = _tile_height(scale_rows)
        y_scale = y_dst + (tile_h - scale_h) // 2
        widths = [tile_w, tile_w, tile_w]
        xs = _layout_row_lefts(CANVAS_W // 2, widths, 80)
        x_exp, x_dst, x_scale = xs[0], xs[1], xs[2]
        _draw_tile_grid(out, x=x_exp, y=y_dst, label="exp tile", prefix="e", highlight_cells=[(EX_R, EX_C)], accent=accent)
        _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst (quant)", prefix="q", highlight_cells=[(EX_R, EX_C)], accent=accent)
        _draw_tile_grid(out, x=x_scale, y=y_scale, label="max/scale", prefix="s", rows=scale_rows, cols=scale_cols, accent=accent)

        s_x, s_y = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        d_x, d_y = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        e_x, e_y = _tile_port_top(x=x_exp, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        m_x, m_y = _tile_port_top(x=x_scale, y=y_scale, rows=scale_rows, cols=scale_cols, c=EX_C)
        via_base = int((y_src + tile_h + y_dst) / 2)
        _draw_ortho_arrow(out, x1=s_x, y1=s_y, x2=d_x, y2=d_y, via_y=via_base - 12, accent=accent)
        _draw_ortho_arrow(out, x1=s_x, y1=s_y, x2=e_x, y2=e_y, via_y=via_base + 0, accent=accent)
        _draw_ortho_arrow(out, x1=s_x, y1=s_y, x2=m_x, y2=m_y, via_y=via_base + 12, accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr.startswith("TPART"):
        op = instr.replace("TPART", "").lower()
        if op == "add":
            body = "src0 + src1"
        elif op == "mul":
            body = "src0 * src1"
        elif op == "max":
            body = "max(src0, src1)"
        elif op == "min":
            body = "min(src0, src1)"
        else:
            body = "op(src0, src1)"
        expr = f"dst[r,c] = partial({body})   (validity-dependent)"
        proc = [
            "for r,c in valid(dst):",
            "  if defined(src0[r,c]) and defined(src1[r,c]):",
            f"    dst[r,c] = {body}",
            "  elif only one is defined:",
            "    dst[r,c] = that defined value",
            "  else:",
            "    dst[r,c] = implementation-defined",
        ]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        xs = _layout_row_lefts(CANVAS_W // 2, [tile_w, tile_w], 120)
        x0, x1 = xs[0], xs[1]
        _draw_tile_grid(out, x=x0, y=y_src, label="src0 (valid A)", prefix="a", valid_box=(3, 3), highlight_cells=[(EX_R, EX_C)], accent=accent)
        _draw_tile_grid(out, x=x1, y=y_src, label="src1 (valid B)", prefix="b", valid_box=(2, 4), highlight_cells=[(EX_R, EX_C)], accent=accent)
        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_dst, y=y_dst, label="dst (valid D)", prefix="d", valid_box=(3, 3), highlight_cells=[(EX_R, EX_C)], accent=accent)
        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        a_x, a_y = _tile_port_bottom(x=x0, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        b_x, b_y = _tile_port_bottom(x=x1, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(a_x, a_y),
            right_src=(b_x, b_y),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "TPRINT":
        expr = "print(src)   (implementation-defined formatting)"
        proc = [expr]
        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )
        x_src = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(out, x=x_src, y=y_src, label="src tile", prefix="a", highlight_cells=[(EX_R, EX_C)], accent=accent)
        box_x = (CANVAS_W - 160) // 2
        box_y = y_dst + 20
        _draw_scalar_box(out, x=box_x, y=box_y, label="side effect", value="print/log", accent=accent)
        sx, sy = _tile_port_bottom(x=x_src, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        dx, dy = _scalar_port_top(x=box_x, y=box_y)
        _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=int((sy + dy) / 2), accent=accent)
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    _draw_procedure(out, lines=["(implementation-defined)"], accent=accent)
    return _end_svg(out)


def _render_sync(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "sync", accent, bg)
    expr = "TSYNC establishes ordering: producer -> consumer"
    out.append(
        f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
    )

    box_w = 520
    box_h = 92
    box_x = (CANVAS_W - box_w) // 2
    y_prod = SRC_Y + 6
    y_cons = DST_Y + 6

    def stage_box(y: int, title: str, detail: str) -> None:
        out.append(
            f'<rect x="{box_x}" y="{y}" width="{box_w}" height="{box_h}" rx="14" fill="#ffffff" stroke="{_esc(accent)}" stroke-width="2"/>'
        )
        out.append(f'<text x="{box_x + box_w // 2}" y="{y + 38}" text-anchor="middle" class="tileLabel">{_esc(title)}</text>')
        out.append(
            f'<text x="{box_x + box_w // 2}" y="{y + 64}" text-anchor="middle" class="smallLabel">{_esc(detail)}</text>'
        )

    stage_box(y_prod, "Producer stage", "ops tagged as producer_class")
    stage_box(y_cons, "Consumer stage", "ops tagged as consumer_class (after TSYNC)")

    src_x, src_y = (box_x + box_w // 2, y_prod + box_h)
    dst_x, dst_y = (box_x + box_w // 2, y_cons)
    via_x = box_x + box_w + 56
    _draw_ortho_arrow(out, x1=src_x, y1=src_y, x2=dst_x, y2=dst_y, via_x=via_x, accent=accent)

    proc = [
        "TSYNC(producer_class, consumer_class)",
        "1) Let P be all earlier ops issued with class=producer_class.",
        "2) TSYNC waits until P are complete (or until their events are satisfied).",
        "3) For all later ops with class=consumer_class: observe results of P.",
        "Ordering: P happens-before consumer_class ops after TSYNC.",
    ]
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def _render_config(instr: str, summary: str, accent: str, bg: str) -> str:
    out = _begin_svg(instr, summary, "config", accent, bg)
    tile_w = _tile_width(TILE_COLS)
    tile_h = _tile_height(TILE_ROWS)
    y_src = SRC_Y
    y_dst = DST_Y

    def draw_state_box(*, x: int, y: int, title: str, lines: Sequence[str]) -> Tuple[int, int]:
        w = 520
        h = 92
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#ffffff" stroke="{_esc(accent)}" stroke-width="2"/>')
        out.append(f'<text x="{x + 18}" y="{y + 34}" class="tileLabel">{_esc(title)}</text>')
        # Two lines is enough for these config diagrams; keep text tidy.
        _draw_text_lines(out, x + 18, y + 56, lines[:2], "smallLabel", 18)
        return (x + w // 2, y)

    if instr == "TASSIGN":
        expr = "tile.bind(address)   (implementation-defined mapping)"
        proc = [
            "TASSIGN(tile, address, ...waitEvents)",
            "1) address := immediate/scalar (implementation-defined base).",
            "2) Bind tile handle to an on-chip address range starting at address.",
            "3) Subsequent memory ops on this tile use the bound mapping.",
        ]

        out.append(
            f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
        )

        widths = [tile_w, 160]
        xs = _layout_row_lefts(CANVAS_W // 2, widths, 220)
        x_tile = xs[0]
        x_addr = xs[1]
        scalar_y = y_src + 8

        _draw_tile_grid(
            out,
            x=x_tile,
            y=y_src,
            label="tile handle (unbound)",
            prefix="t",
            highlight_cells=[(EX_R, EX_C)],
            accent=accent,
        )
        _draw_scalar_box(out, x=x_addr, y=scalar_y, label="address", value="0x....", accent=accent)

        x_dst = (CANVAS_W - tile_w) // 2
        _draw_tile_grid(
            out,
            x=x_dst,
            y=y_dst,
            label="tile handle (bound)",
            prefix="t",
            highlight_cells=[(EX_R, EX_C)],
            accent=accent,
        )

        dx, dy = _tile_port_top(x=x_dst, y=y_dst, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        sx, sy = _tile_port_bottom(x=x_tile, y=y_src, rows=TILE_ROWS, cols=TILE_COLS, c=EX_C)
        ax, ay = _scalar_port_bottom(x=x_addr, y=scalar_y)
        _draw_binary_flow(
            out,
            instr=instr,
            left_src=(sx, sy),
            right_src=(ax, ay),
            dst=(dx, dy),
            accent=accent,
            op_cx=CANVAS_W // 2,
        )
        _draw_procedure(out, lines=proc, accent=accent)
        return _end_svg(out)

    if instr == "SETFMATRIX":
        expr = "set FMATRIX state (used by later ops)"
        proc = [
            "SETFMATRIX(value, ...waitEvents)",
            "1) Update FMATRIX register/state.",
            "2) Ordering: update takes effect before dependent ops.",
            "3) Affects subsequent IMG2COL / layout-sensitive operations.",
        ]
        scalar_label = "FMATRIX"
        scalar_value = "set"
        state_lines = ["FMATRIX state updated", "consulted by later ops"]
    else:
        mode = "HF32" if instr == "SETTF32MODE" else "TF32" if instr == "SETTF32MODE" else "mode"
        expr = f"set transform mode ({mode})"
        proc = [
            f"{instr}(enable/mode, ...waitEvents)",
            "1) Update backend transform/rounding mode (implementation-defined).",
            "2) Ordering: update takes effect before dependent ops.",
            "3) Affects subsequent GEMV/MATMUL or conversion paths (if applicable).",
        ]
        scalar_label = "mode"
        scalar_value = "enable/mode"
        state_lines = ["backend mode state updated", "used by later ops"]

    out.append(
        f'<text x="{CANVAS_W // 2}" y="{EXPR_Y}" class="subtitle" text-anchor="middle" fill="{_esc(accent)}">{_esc(expr)}</text>'
    )

    scalar_x = CANVAS_W - MARGIN - 16 - 160
    scalar_y = y_src + 8
    _draw_scalar_box(out, x=scalar_x, y=scalar_y, label=scalar_label, value=scalar_value, accent=accent)

    state_x = (CANVAS_W - 520) // 2
    state_y = y_dst + 6
    dx, dy = draw_state_box(x=state_x, y=state_y, title="Execution state", lines=state_lines)
    sx, sy = (scalar_x + 160 // 2, scalar_y + 54)
    via_y = int((sy + dy) / 2)
    _draw_ortho_arrow(out, x1=sx, y1=sy, x2=dx, y2=dy, via_y=via_y, accent=accent)
    _draw_procedure(out, lines=proc, accent=accent)
    return _end_svg(out)


def render_svg(entry: Dict[str, object]) -> str:
    instr = str(entry.get("instruction", "UNKNOWN")).strip()
    template = str(entry.get("diagram_template", "elementwise")).strip()
    summary = str(entry.get("summary_en", "")).strip()

    accent, bg = COLOR_BY_TEMPLATE.get(template, COLOR_BY_TEMPLATE["elementwise"])

    if template == "elementwise":
        return _render_elementwise(instr, summary, accent, bg)
    if template == "scalar":
        return _render_scalar(instr, summary, accent, bg)
    if template == "reduce_expand":
        return _render_reduce_expand(instr, summary, accent, bg)
    if template == "memory":
        return _render_memory(instr, summary, accent, bg)
    if template == "matmul":
        return _render_matmul(instr, summary, accent, bg)
    if template == "reshape_move":
        return _render_reshape_move(instr, summary, accent, bg)
    if template == "complex":
        return _render_complex(instr, summary, accent, bg)
    if template == "sync":
        return _render_sync(instr, summary, accent, bg)
    if template == "config":
        return _render_config(instr, summary, accent, bg)

    # Fallback
    out = _begin_svg(instr, summary, template, accent, bg)
    _draw_procedure(out, lines=["(diagram template not implemented)"], accent=accent)
    return _end_svg(out)


def check_svgs(entries: List[Dict[str, object]], output_dir: Path) -> List[str]:
    errors: List[str] = []
    for e in entries:
        instr = str(e.get("instruction", "")).strip()
        if not instr:
            continue
        svg = output_dir / f"{instr}.svg"
        if not svg.exists():
            errors.append(f"missing svg: {svg}")
            continue
        data = svg.read_text(encoding="utf-8", errors="ignore")
        if "<svg" not in data:
            errors.append(f"invalid svg: {svg}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate per-instruction SVG diagrams")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    entries = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        errors = check_svgs(entries, args.output_dir)
        if errors:
            for err in errors:
                print(f"ERROR: {err}")
            return 1
        print("OK: all instruction SVG files are present.")
        return 0

    for e in entries:
        instr = str(e.get("instruction", "")).strip()
        if not instr:
            continue
        out_path = args.output_dir / f"{instr}.svg"
        out_path.write_text(render_svg(e), encoding="utf-8")

    print(f"Generated {len(entries)} SVG files in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
