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

import html
import json
from dataclasses import dataclass
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
    "comm": ("#8A4B12", "#FFF3E7"),
}


@dataclass(frozen=True)
class TileGridContext:
    x: int
    y: int
    accent: str


@dataclass(frozen=True)
class TileCellState:
    r: int
    c: int
    masked: bool
    is_hl: bool


@dataclass(frozen=True)
class TextLinesSpec:
    x: int
    y: int
    lines: Sequence[str]
    cls: str
    line_height: int


def _esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def _append_svg_text(
    out: List[str],
    *,
    x: object,
    y: object,
    cls: str,
    text: object,
    text_anchor: Optional[str] = None,
    dominant_baseline: Optional[str] = None,
    fill: Optional[str] = None,
    font_size: Optional[str] = None,
) -> None:
    attrs = [f'x="{x}"', f'y="{y}"', f'class="{cls}"']
    if text_anchor:
        attrs.append(f'text-anchor="{text_anchor}"')
    if dominant_baseline:
        attrs.append(f'dominant-baseline="{dominant_baseline}"')
    if fill:
        attrs.append(f'fill="{fill}"')
    if font_size:
        attrs.append(f'font-size="{font_size}"')
    out.append(f"<text {' '.join(attrs)}>{_esc(text)}</text>")


def _append_rect(
    out: List[str],
    *,
    x: object,
    y: object,
    width: object,
    height: object,
    rx: Optional[object] = None,
    cls: Optional[str] = None,
    fill: Optional[str] = None,
    stroke: Optional[str] = None,
    stroke_width: Optional[object] = None,
) -> None:
    attrs = [f'x="{x}"', f'y="{y}"', f'width="{width}"', f'height="{height}"']
    if rx is not None:
        attrs.append(f'rx="{rx}"')
    if cls:
        attrs.append(f'class="{cls}"')
    if fill:
        attrs.append(f'fill="{fill}"')
    if stroke:
        attrs.append(f'stroke="{_esc(stroke)}"')
    if stroke_width is not None:
        attrs.append(f'stroke-width="{stroke_width}"')
    out.append(f"<rect {' '.join(attrs)}/>")


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


def _draw_text_lines(out: List[str], spec: TextLinesSpec) -> None:
    out.append(f'<text x="{spec.x}" y="{spec.y}" class="{spec.cls}" xml:space="preserve">')
    first = True
    for ln in spec.lines:
        if first:
            out.append(f'  <tspan x="{spec.x}" dy="0">{_esc(ln)}</tspan>')
            first = False
        else:
            out.append(f'  <tspan x="{spec.x}" dy="{spec.line_height}">{_esc(ln)}</tspan>')
    out.append("</text>")


def _normalize_valid_box(rows: int, cols: int, valid_box: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if valid_box is None:
        vr = rows - 1 if rows > 2 else rows
        vc = cols - 1 if cols > 2 else cols
    else:
        vr, vc = valid_box
    return (max(0, min(rows, vr)), max(0, min(cols, vc)))


def _draw_tile_cell(out: List[str], ctx: TileGridContext, state: TileCellState) -> None:
    rx = ctx.x + state.c * CELL
    ry = ctx.y + state.r * CELL
    cls_parts = ["cell"]
    if state.masked:
        cls_parts.append("cellMasked")
    if state.is_hl:
        cls_parts.append("cellHL")
    cls = " ".join(cls_parts)
    if state.is_hl:
        out.append(f'<rect x="{rx}" y="{ry}" width="{CELL}" height="{CELL}" class="{cls}" stroke="{ctx.accent}" />')
    else:
        out.append(f'<rect x="{rx}" y="{ry}" width="{CELL}" height="{CELL}" class="{cls}" />')


def _draw_tile_text(out: List[str], ctx: TileGridContext, r: int, c: int, text: str) -> None:
    cx, cy = _cell_center(ctx.x, ctx.y, r, c)
    _append_svg_text(out, x=cx, y=cy + 1, cls="cellText", text=text, text_anchor="middle", dominant_baseline="middle")


def _draw_tile_valid_box(out: List[str], ctx: TileGridContext, valid_box: Tuple[int, int]) -> None:
    vr, vc = valid_box
    if not (vr and vc):
        return
    out.append(
        f'<rect x="{ctx.x}" y="{ctx.y}" width="{vc * CELL}" height="{vr * CELL}" '
        f'class="validBox" stroke="{ctx.accent}" />'
    )
    out.append(f'<text x="{ctx.x + 4}" y="{ctx.y + vr * CELL - 4}" class="axisText">{_esc("Rv")}</text>')
    out.append(
        f'<text x="{ctx.x + vc * CELL - 4}" y="{ctx.y + 12}" class="axisText" text-anchor="end">{_esc("Cv")}</text>'
    )


def _draw_tile_axes(out: List[str], *, x: int, y: int, rows: int, cols: int) -> None:
    if rows < 3 or cols < 3:
        return
    ax = x + 10
    ay = y + 10
    out.append(f'<path d="M {ax} {ay} L {ax + 34} {ay}" class="axisLine" marker-end="url(#axisArrow)" />')
    out.append(f'<path d="M {ax} {ay} L {ax} {ay + 34}" class="axisLine" marker-end="url(#axisArrow)" />')
    out.append(f'<text x="{ax + 38}" y="{ay + 4}" class="axisText">{_esc("c")}</text>')
    out.append(f'<text x="{ax - 2}" y="{ay + 38}" class="axisText" text-anchor="end">{_esc("r")}</text>')


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
    vr, vc = _normalize_valid_box(rows, cols, valid_box)
    ctx = TileGridContext(x=x, y=y, accent=accent)

    out.append(f'<text x="{x + w // 2}" y="{y - 10}" class="tileLabel" text-anchor="middle">{_esc(label)}</text>')
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="tileBorder" />')

    for r in range(rows):
        for c in range(cols):
            is_hl = (r, c) in highlight or r in highlight_r or c in highlight_c
            masked = (r >= vr) or (c >= vc)
            _draw_tile_cell(out, ctx, TileCellState(r=r, c=c, masked=masked, is_hl=is_hl))
            text = text_override.get((r, c))
            if text:
                _draw_tile_text(out, ctx, r, c, text)

    _draw_tile_valid_box(out, ctx, (vr, vc))
    _ = prefix  # reserved for future per-cell callouts
    _draw_tile_axes(out, x=x, y=y, rows=rows, cols=cols)


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
            _append_svg_text(
                out,
                x=rx + cell_w // 2,
                y=y + CELL // 2 + 1,
                cls="cellText",
                text=text,
                text_anchor="middle",
                dominant_baseline="middle",
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
    out: List[str], *, cx: int, cy: int, instr: str, accent: str
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Draw a circled-square op node and return (left, right, bottom) anchors."""
    max_chars_per_line = 6
    label_lines = []
    for i in range(0, len(instr), max_chars_per_line):
        line_end = i + max_chars_per_line
        label_lines.append(instr[i:line_end])
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
    _append_rect(out, x=cx - side // 2, y=cy - side // 2, width=side, height=side, rx=6, cls="opRect", stroke=accent)
    y0 = cy - ((line_count - 1) * line_gap) / 2
    for idx, line in enumerate(label_lines):
        y = y0 + idx * line_gap + 3
        _append_svg_text(
            out, x=cx, y=f"{y:.1f}", cls="opText", text=line, font_size=f"{font_px}px", text_anchor="middle"
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
    _draw_ortho_arrow(out, x1=bottom[0], y1=bottom[1], x2=dx, y2=dy, via_y=int((bottom[1] + dy) / 2), accent=accent)


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
        if via_x is None:
            raise ValueError("via_x must be specified when via_y is not specified")
        d = f"M {x1} {y1} L {via_x} {y1} L {via_x} {y2} L {x2} {y2}"
    out.append(f'<path d="{d}" class="arrow" stroke="{accent}" marker-end="url(#arrow)" />')


def _append_svg_defs(out: List[str], accent: str) -> None:
    out.append("<defs>")
    out.append(
        f'  <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" '
        f'orient="auto"><path d="M0,0 L0,12 L12,6 z" fill="{_esc(accent)}"/></marker>'
    )
    out.append(
        '  <marker id="axisArrow" markerWidth="10" markerHeight="10" refX="8" refY="5" '
        'orient="auto"><path d="M0,0 L0,10 L10,5 z" fill="#64748b"/></marker>'
    )
    out.append("</defs>")


def _svg_style_lines(bg: str) -> List[str]:
    return [
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
        ".cellText { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "
        "'Liberation Mono', 'Courier New', monospace; font-size: 10px; fill: #0f172a; }",
        ".arrow { stroke-width: 2.5; fill: none; stroke-linejoin: round; stroke-linecap: round; }",
        ".axisLine { stroke: #64748b; stroke-width: 1.5; fill: none; }",
        ".axisText { font-size: 10px; fill: #64748b; font-weight: 700; }",
        ".opCircle { fill: #ffffff; stroke-width: 2; }",
        ".opRect { fill: #ffffff; stroke-width: 2; }",
        ".opText { font-size: 10px; font-weight: 800; fill: #0f172a; }",
        ".procBox { fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1.5; rx: 12; }",
        ".procTitle { font-size: 14px; font-weight: 700; fill: #0f172a; }",
        ".procText { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "
        "'Liberation Mono', 'Courier New', monospace; font-size: 12px; fill: #0f172a; }",
        ".smallLabel { font-size: 12px; fill: #334155; }",
        ".scalarBox { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1.5; }",
        ".scalarValue { font-size: 16px; font-weight: 700; }",
        ".validBox { fill: none; stroke-width: 2; stroke-dasharray: 6 4; }",
    ]


def _append_svg_style(out: List[str], bg: str) -> None:
    out.append("<style>")
    out.append("\n".join(_svg_style_lines(bg)))
    out.append("</style>")


def _append_svg_header(out: List[str], instr: str, summary: str, template: str) -> None:
    out.append(f'<rect x="0" y="0" width="{CANVAS_W}" height="{CANVAS_H}" class="frame" />')
    _append_rect(out, x=MARGIN, y=MARGIN, width=CANVAS_W - 2 * MARGIN, height=CANVAS_H - 2 * MARGIN, cls="panel")
    out.append(f'<text x="{MARGIN + 16}" y="{HEADER_Y}" class="title">{_esc(instr)}</text>')
    out.append(f'<text x="{MARGIN + 16}" y="{HEADER_Y + 26}" class="subtitle">{_esc(summary)}</text>')
    out.append(f'<text x="{MARGIN + 16}" y="{HEADER_Y + 46}" class="meta">Template: {_esc(template)}</text>')
    _append_svg_text(
        out,
        x=CANVAS_W - MARGIN - 16,
        y=HEADER_Y + 46,
        cls="meta",
        text="Legend: outline=example; dashed=valid rows/cols (Rv,Cv); shaded=masked; "
        "r down / c right; ortho arrows=dataflow",
        text_anchor="end",
    )
    out.append(
        f'<line x1="{MARGIN + 12}" y1="{HEADER_DIVIDER_Y}" '
        f'x2="{CANVAS_W - MARGIN - 12}" y2="{HEADER_DIVIDER_Y}" '
        'stroke="#e2e8f0" stroke-width="1.5" />'
    )


def _begin_svg(instr: str, summary: str, template: str, accent: str, bg: str) -> List[str]:
    aria = f"{instr} tile operation diagram"
    out: List[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" '
        f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" role="img" aria-label="{_esc(aria)}">'
    )
    _append_svg_defs(out, accent)
    _append_svg_style(out, bg)
    _append_svg_header(out, instr, summary, template)
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
    _draw_text_lines(out, TextLinesSpec(x=x + PROC_PAD, y=y + 52, lines=lines, cls="procText", line_height=16))
    _append_svg_text(
        out,
        x=x + PROC_PAD,
        y=y + h - 16,
        cls="meta",
        text="Note: semantics apply to the valid region unless stated otherwise.",
    )


def _draw_expr(out: List[str], expr: str, accent: str) -> None:
    _append_svg_text(out, x=CANVAS_W // 2, y=EXPR_Y, cls="subtitle", text=expr, text_anchor="middle", fill=accent)


UNARY_ELEMENTWISE_EXPR = {
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

BINARY_ELEMENTWISE_EXPR = {
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

TERNARY_ELEMENTWISE_SPEC = {
    "TMADD": ("src0 * dst(old) + src1", ["src0", "dst(old)", "src1"]),
    "TMULADDDST": ("src0 * src1 + dst(old)", ["src0", "src1", "dst(old)"]),
}


def _elementwise_proc(expr: str) -> List[str]:
    return ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]


def _elementwise_return(inputs: List[str], expr: str) -> Tuple[List[str], str, List[str]]:
    return (inputs, expr, _elementwise_proc(expr))


def _elementwise_special_spec(instr: str) -> Optional[Tuple[List[str], str, List[str]]]:
    if instr == "TSEL":
        expr = "dst[r,c] = (mask[r,c] != 0) ? src0[r,c] : src1[r,c]"
        return _elementwise_return(["mask", "src0", "src1"], expr)
    if instr == "TCMP":
        expr = "dst_mask[r,c] = cmp(src0[r,c], src1[r,c])"
        return _elementwise_return(["src0", "src1"], expr)
    if instr == "TPRELU":
        expr = "dst[r,c] = (x>0) ? x : slope*x"
        proc = _elementwise_proc(expr)
        proc[2:2] = ["    x = src0[r,c]", "    slope = src1[r,c]"]
        return (["src0", "src1"], expr, proc)
    return None


def _elementwise_spec(instr: str) -> Tuple[List[str], str, List[str]]:
    if instr in UNARY_ELEMENTWISE_EXPR:
        expr = f"dst[r,c] = {UNARY_ELEMENTWISE_EXPR[instr]}"
        return _elementwise_return(["src"], expr)
    if instr in TERNARY_ELEMENTWISE_SPEC:
        op_expr, op_inputs = TERNARY_ELEMENTWISE_SPEC[instr]
        return _elementwise_return(op_inputs, f"dst[r,c] = {op_expr}")
    special = _elementwise_special_spec(instr)
    if special is not None:
        return special
    if instr in BINARY_ELEMENTWISE_EXPR:
        expr = f"dst[r,c] = {BINARY_ELEMENTWISE_EXPR[instr]}"
        return _elementwise_return(["src0", "src1"], expr)

    expr = "dst[r,c] = op(src...)[r,c]"
    return _elementwise_return(["src0", "src1"], expr)


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
        proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]
        return (["src(tile)"], expr, proc)
    if instr == "TCMPS":
        expr = "dst_mask[r,c] = cmp(src[r,c], s)"
        proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]
        return (["src(tile)"], expr, proc)
    if instr == "TSELS":
        expr = "dst = (selectMode) ? src0 : src1"
        proc = ["if selectMode:", "  dst = src0", "else:", "  dst = src1"]
        return (["src0", "src1"], expr, proc)
    if instr == "TLRELU":
        expr = "dst[r,c] = (x>0) ? x : slope*x"
        proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", "    x = src[r,c]", f"    {expr}"]
        return (["src(tile)"], expr, proc)
    if instr in tile_scalar:
        expr = f"dst[r,c] = {tile_scalar[instr]}"
        proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]
        return (["src(tile)"], expr, proc)

    expr = "dst[r,c] = op(src[r,c], s)"
    proc = ["for r in 0..Rv-1:", "  for c in 0..Cv-1:", f"    {expr}"]
    return (["src(tile)"], expr, proc)


def _reduce_expand_kind(instr: str) -> Tuple[str, str, str]:
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
