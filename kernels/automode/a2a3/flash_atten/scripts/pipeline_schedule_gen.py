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

"""
Generate a simple pipeline schedule SVG for FA (qk -> p -> pv -> gu) with configurable
cycle ratios, stage counts, and preload length.

Usage:
  python pipeline_schedule_gen.py \
    --tiles 10 \
    --preload 4 \
    --cycle-qk 10 --cycle-p 16 --cycle-pv 10 --cycle-gu 6 \
    --slot-width 8 \
    --output fa_pipeline_generated.svg

Rules encoded:
- compute_qk / compute_pv share cube core (no overlap between qk and pv on cube).
- compute_p / compute_gu share vector core (no overlap between p and gu on vector).
- Cube and vector cores can run in parallel (qk || p, pv || p, etc.).
- Data dependency per tile: qk(i) -> p(i) -> pv(i) -> gu(i).
- Preload: user can specify how many initial qk/p steps are issued before pv starts.

This is a compact planner, not a full simulator; it greedily schedules respecting
resource exclusivity and per-tile dependencies. Modify as needed for deeper fidelity.
"""

from dataclasses import dataclass
from typing import List, Tuple
import argparse
import math

@dataclass
class Stage:
    name: str
    core: str  # "cube" or "vector"
    cycles: int  # compute cycles (excluding IO)

    def total(self, io_overhead: int) -> int:
        # Model TLOAD + COMPUTE + TSTORE, each IO side costs io_overhead; compute in between
        return self.cycles + 2 * io_overhead

@dataclass
class Task:
    stage: Stage
    tile: int
    start: int
    end: int

    def duration(self) -> int:
        return self.end - self.start


def schedule_stage(
    tile: int,
    stage: Stage,
    dep_ready: int,
    io_load_overhead: int,
    io_store_overhead: int,
    comp_free: int,
    io_load_free: dict,
    io_store_free: dict,
    pending_comp: dict,
    tasks: List[Task],
    store_enabled: bool = True,
) -> Tuple[int, int]:
    # split into load/compute/store; IO serialized per core; at most 2 loads ahead of compute per stage (ping/pong)
    load_stage = Stage(f"{stage.name}_load", stage.core, io_load_overhead)
    comp_stage = Stage(f"{stage.name}_comp", stage.core, stage.cycles)
    store_stage = Stage(f"{stage.name}_store", stage.core, io_store_overhead)

    base_name = stage.name
    pending = pending_comp.setdefault(base_name, [])

    def retire(t: int):
        while pending and pending[0] <= t:
            pending.pop(0)

    # enforce ping/pong (<=2 outstanding) and serialized IO per core
    load_start = max(dep_ready, io_load_free[stage.core])
    retire(load_start)
    while len(pending) >= 2:
        load_start = max(load_start, pending[0])
        retire(load_start)
    load_end = load_start + io_load_overhead
    io_load_free[stage.core] = load_end
    tasks.append(Task(load_stage, tile, load_start, load_end))

    comp_start = max(load_end, comp_free)
    comp_end = comp_start + stage.cycles
    comp_free = comp_end
    pending.append(comp_end)
    pending.sort()
    tasks.append(Task(comp_stage, tile, comp_start, comp_end))

    if store_enabled:
        store_start = max(comp_end, io_store_free[stage.core])
        store_end = store_start + io_store_overhead
        io_store_free[stage.core] = store_end
        tasks.append(Task(store_stage, tile, store_start, store_end))
        return store_end, comp_free

    # When store is disabled (e.g., GU accumulates and only final tile stores), end at compute finish
    return comp_end, comp_free


def schedule_pipeline(
    num_tiles: int,
    preload: int,
    cyc_qk: int,
    cyc_p: int,
    cyc_pv: int,
    cyc_gu: int,
    io_load_overhead: int,
    io_store_overhead: int,
) -> List[Task]:
    # Define stages
    qk = Stage("qk", "cube", cyc_qk)
    p = Stage("p", "vector", cyc_p)
    pv = Stage("pv", "cube", cyc_pv)
    gu = Stage("gu", "vector", cyc_gu)

    tasks: List[Task] = []

    cube_comp_free = 0
    vec_comp_free = 0
    io_load_free = {"cube": 0, "vector": 0}
    io_store_free = {"cube": 0, "vector": 0}
    pending_comp: dict[str, List[int]] = {}

    preload = max(0, min(preload, num_tiles))  # tiles of qk/p to issue before first pv is allowed

    # Track completion per tile
    p_done = [0] * num_tiles

    pv_queue: List[int] = []  # tiles whose P is done and waiting for PV

    for t in range(num_tiles):
        # QK scheduling (cube: load/comp/store)
        qk_done, cube_comp_free = schedule_stage(
            t, qk, 0, io_load_overhead, io_store_overhead, cube_comp_free, io_load_free, io_store_free, pending_comp, tasks, True
        )

        # P scheduling (depends on qk(t) done; vector load/comp/store)
        p_start_dep = qk_done
        p_done_t, vec_comp_free = schedule_stage(
            t, p, p_start_dep, io_load_overhead, io_store_overhead, vec_comp_free, io_load_free, io_store_free, pending_comp, tasks, True
        )
        p_done[t] = p_done_t

        # enqueue PV work for this tile
        pv_queue.append(t)

        # schedule one PV/GU when allowed
        if len(pv_queue) >= max(1, preload + 1):
            tile_idx = pv_queue.pop(0)
            pv_start_dep = p_done[tile_idx]
            pv_done, cube_comp_free = schedule_stage(
                tile_idx, pv, pv_start_dep, io_load_overhead, io_store_overhead, cube_comp_free, io_load_free, io_store_free, pending_comp, tasks, True
            )

            gu_start_dep = pv_done
            _, vec_comp_free = schedule_stage(
                tile_idx, gu, gu_start_dep, io_load_overhead, io_store_overhead, vec_comp_free, io_load_free, io_store_free, pending_comp,
                tasks, tile_idx == num_tiles - 1
            )

    # Flush any remaining PV/GU after all qk/p issued
    while pv_queue:
        tile_idx = pv_queue.pop(0)
        pv_start_dep = p_done[tile_idx]
        pv_done, cube_comp_free = schedule_stage(
            tile_idx, pv, pv_start_dep, io_load_overhead, io_store_overhead, cube_comp_free, io_load_free, io_store_free, pending_comp, tasks, True
        )

        gu_start_dep = pv_done
        _, vec_comp_free = schedule_stage(
            tile_idx, gu, gu_start_dep, io_load_overhead, io_store_overhead, vec_comp_free, io_load_free, io_store_free, pending_comp,
            tasks, tile_idx == num_tiles - 1
        )

    return tasks


def to_svg(tasks: List[Task], slot_width: int, output: str):
    # layout rows
    rows = {"qk": 60, "p": 120, "pv": 180, "gu": 240, "cube": 320, "vector": 380}
    height = 440
    # compute max time
    tmax = max(task.end for task in tasks)
    width = int(120 + slot_width * tmax + 40)

    colors = {
        "qk": ("#1f77b4", "#1a4f7a"),
        "p": ("#ff9933", "#c66f0a"),
        "pv": ("#2ca25f", "#1f7a47"),
        "gu": ("#9467bd", "#6c3f8b"),
    }

    def base(stage_name: str) -> str:
        return stage_name.split("_")[0]

    def color_for(stage_name: str) -> Tuple[str, str]:
        return colors.get(base(stage_name), ("#888", "#555"))

    mini_h = 12
    mini_gap = 4

    def y_for(task: Task) -> float:
        band_top = rows[base(task.stage.name)]
        suffix = task.stage.name.split("_")[1] if "_" in task.stage.name else "comp"
        if suffix == "load":
            offset = 0
        elif suffix == "comp":
            offset = mini_h + mini_gap
        else:  # store
            offset = 2 * (mini_h + mini_gap)
        return band_top + offset

    def rect(task: Task) -> str:
        x = 120 + slot_width * task.start
        w = slot_width * (task.end - task.start)
        y = y_for(task)
        fill, stroke = color_for(task.stage.name)
        label = f"{task.stage.name}({task.tile})"
        return (
            f'  <rect x="{x}" y="{y}" width="{w}" height="{mini_h}" fill="{fill}" '
            f'rx="3" stroke="{stroke}" stroke-width="1" />\n'
            f'  <text x="{x + 4}" y="{y + mini_h - 3:.1f}" font-family="Arial" font-size="10" fill="#fff">{label}</text>'
        )

    svg_parts: List[str] = []
    svg_parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg_parts.append(f'  <rect width="{width}" height="{height}" fill="#fff" />')
    svg_parts.append(
        "  <defs>\n"
        "    <marker id=\"arrowhead\" markerWidth=\"10\" markerHeight=\"7\" refX=\"10\" refY=\"3.5\" orient=\"auto\">\n"
        "      <polygon points=\"0 0 10 3.5 0 7\" fill=\"#333\" />\n"
        "    </marker>\n"
        "    <marker id=\"arrowhead-intra\" markerWidth=\"10\" markerHeight=\"7\" refX=\"10\" refY=\"3.5\" orient=\"auto\">\n"
        "      <polygon points=\"0 0 10 3.5 0 7\" fill=\"#1b75d1\" />\n"
        "    </marker>\n"
        "  </defs>"
    )
    svg_parts.append('  <text x="20" y="28" font-family="Arial" font-size="16" font-weight="bold" fill="#222">FA Pipeline Schedule (generated)</text>')

    # grid (light weight)
    svg_parts.append('  <g stroke="#f2f2f2" stroke-width="1">')
    for t in range(0, tmax + 1, 10):
        x = 120 + slot_width * t
        svg_parts.append(f'    <line x1="{x}" y1="40" x2="{x}" y2="280" />')
    svg_parts.append('  </g>')

    # row labels
    svg_parts.append('  <text x="20" y="80" font-family="Arial" font-size="13" fill="#222">compute_qk (cube)</text>')
    svg_parts.append('  <text x="20" y="140" font-family="Arial" font-size="13" fill="#222">compute_p (vector)</text>')
    svg_parts.append('  <text x="20" y="200" font-family="Arial" font-size="13" fill="#222">compute_pv (cube)</text>')
    svg_parts.append('  <text x="20" y="260" font-family="Arial" font-size="13" fill="#222">compute_gu (vector)</text>')
    svg_parts.append('  <text x="20" y="340" font-family="Arial" font-size="13" fill="#222">cube timeline (qk+pv)</text>')
    svg_parts.append('  <text x="20" y="400" font-family="Arial" font-size="13" fill="#222">vector timeline (p+gu)</text>')

    # blocks
    for task in tasks:
        svg_parts.append(rect(task))

    # helper to reuse coloring for merged timelines
    def rect_at(task: Task, y_row: int) -> str:
        # Keep merged timelines slim with the same sub-stage offsets
        y = y_row + (y_for(task) - rows[base(task.stage.name)])
        x = 120 + slot_width * task.start
        w = slot_width * (task.end - task.start)
        fill, stroke = color_for(task.stage.name)
        label = f"{task.stage.name}({task.tile})"
        return (
            f'  <rect x="{x}" y="{y}" width="{w}" height="{mini_h}" fill="{fill}" '
            f'rx="3" stroke="{stroke}" stroke-width="1" />\n'
            f'  <text x="{x + 4}" y="{y + mini_h - 3:.1f}" font-family="Arial" font-size="10" fill="#fff">{label}</text>'
        )

    # merged cube timeline (qk + pv) and vector timeline (p + gu)
    for task in sorted(tasks, key=lambda t: (t.start, t.end)):
        if task.stage.core == "cube":
            svg_parts.append(rect_at(task, rows["cube"]))
        else:
            svg_parts.append(rect_at(task, rows["vector"]))

    # arrows for dependencies per tile (qk->p, p->pv, pv->gu) using store boundaries
    def find(stage_name: str, tile: int) -> Task:
        for tk in tasks:
            if tk.stage.name == stage_name and tk.tile == tile:
                return tk
        return None

    for tile in range(max(tk.tile for tk in tasks) + 1):
        y_offset = tile * 3  # stagger arrows vertically per tile to avoid overlap
        # cross-stage deps
        qk_store = find("qk_store", tile)
        p_load = find("p_load", tile)
        if qk_store and p_load:
            x1 = 120 + slot_width * ((qk_store.start + qk_store.end) / 2)
            x2 = 120 + slot_width * p_load.start
            y1 = y_for(qk_store) + mini_h / 2 + y_offset
            y2 = y_for(p_load) + mini_h / 2 + y_offset
            svg_parts.append(
                f'  <path d="M{x1:.1f} {y1} L{x1:.1f} {y2-4} L{x2} {y2-4}" stroke="#333" stroke-width="1.5" fill="none" marker-end="url(#arrowhead)" />'
            )
        p_store = find("p_store", tile)
        pv_load = find("pv_load", tile)
        if p_store and pv_load:
            x1 = 120 + slot_width * ((p_store.start + p_store.end) / 2)
            x2 = 120 + slot_width * pv_load.start
            y1 = y_for(p_store) + mini_h / 2 + y_offset
            y2 = y_for(pv_load) + mini_h / 2 + y_offset
            svg_parts.append(
                f'  <path d="M{x1:.1f} {y1} L{x1:.1f} {y2-4} L{x2} {y2-4}" stroke="#333" stroke-width="1.5" fill="none" marker-end="url(#arrowhead)" />'
            )
        pv_store = find("pv_store", tile)
        gu_load = find("gu_load", tile)
        if pv_store and gu_load:
            x1 = 120 + slot_width * ((pv_store.start + pv_store.end) / 2)
            x2 = 120 + slot_width * gu_load.start
            y1 = y_for(pv_store) + mini_h / 2 + y_offset
            y2 = y_for(gu_load) + mini_h / 2 + y_offset
            svg_parts.append(
                f'  <path d="M{x1:.1f} {y1} L{x1:.1f} {y2-4} L{x2} {y2-4}" stroke="#333" stroke-width="1.5" fill="none" marker-end="url(#arrowhead)" />'
            )

        # intra-stage deps (load -> comp -> store) in blue
        for base_name in ("qk", "p", "pv", "gu"):
            load = find(f"{base_name}_load", tile)
            comp = find(f"{base_name}_comp", tile)
            store = find(f"{base_name}_store", tile)
            if load and comp and load.start != load.end and comp.start != comp.end:
                x1 = 120 + slot_width * load.end
                x2 = 120 + slot_width * comp.start
                y1 = y_for(load) + mini_h / 2 + y_offset
                y2 = y_for(comp) + mini_h / 2 + y_offset
                svg_parts.append(
                    f'  <path d="M{x1:.1f} {y1} L{x2} {y2}" stroke="#1b75d1" stroke-width="1.2" fill="none" marker-end="url(#arrowhead-intra)" />'
                )
            if comp and store and comp.start != comp.end and store.start != store.end:
                x1 = 120 + slot_width * comp.end
                x2 = 120 + slot_width * store.start
                y1 = y_for(comp) + mini_h / 2 + y_offset
                y2 = y_for(store) + mini_h / 2 + y_offset
                svg_parts.append(
                    f'  <path d="M{x1:.1f} {y1} L{x2} {y2}" stroke="#1b75d1" stroke-width="1.2" fill="none" marker-end="url(#arrowhead-intra)" />'
                )

    svg_parts.append('</svg>')

    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
    print(f"Wrote {output} (width={width}, height={height}, tmax={tmax})")


def dump_schedule(tasks: List[Task]):
    print("\nPipeline execution (start -> end cycles):")
    by_tile: dict[int, List[Task]] = {}
    for t in tasks:
        by_tile.setdefault(t.tile, []).append(t)

    for tile in sorted(by_tile):
        print(f"Tile {tile}:")
        for task in sorted(by_tile[tile], key=lambda x: x.start):
            print(
                f"  {task.stage.name:<10} {task.start:>4} -> {task.end:>4} "
                f"({task.duration()} cycles) on {task.stage.core}"
            )


def main():
    ap = argparse.ArgumentParser(description="Generate FA pipeline schedule SVG")
    ap.add_argument("--tiles", type=int, default=10, help="Total S1/Cube_S1 tiles")
    ap.add_argument("--preload", type=int, default=0, help="Tiles to issue before first PV is allowed (0 = allow PV0 as soon as p0 is done)")
    ap.add_argument("--cycle-qk", type=int, default=10, help="Cycles for qk stage")
    ap.add_argument("--cycle-p", type=int, default=16, help="Cycles for p stage")
    ap.add_argument("--cycle-pv", type=int, default=10, help="Cycles for pv stage")
    ap.add_argument("--cycle-gu", type=int, default=6, help="Cycles for gu stage")
    ap.add_argument("--io-load-overhead", type=int, default=10, help="IO overhead cycles per stage for TLOAD")
    ap.add_argument("--io-store-overhead", type=int, default=10, help="IO overhead cycles per stage for TSTORE")
    ap.add_argument("--slot-width", type=int, default=8, help="Pixels per cycle in SVG")
    ap.add_argument("--output", type=str, default="fa_pipeline_generated.svg", help="Output SVG path")
    args = ap.parse_args()

    tasks = schedule_pipeline(
        num_tiles=args.tiles,
        preload=args.preload,
        cyc_qk=args.cycle_qk,
        cyc_p=args.cycle_p,
        cyc_pv=args.cycle_pv,
        cyc_gu=args.cycle_gu,
        io_load_overhead=args.io_load_overhead,
        io_store_overhead=args.io_store_overhead,
    )
    dump_schedule(tasks)
    to_svg(tasks, slot_width=args.slot_width, output=args.output)


if __name__ == "__main__":
    main()
