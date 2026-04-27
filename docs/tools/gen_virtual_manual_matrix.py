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

"""Generate an instruction-family matrix from the ISA manifest.

This script is intentionally source-synchronized with:
- docs/isa/manifest.yaml
- include/pto/common/pto_instr.hpp
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "isa" / "manifest.yaml"
DEFAULT_HEADER = REPO_ROOT / "include" / "pto" / "common" / "pto_instr.hpp"
DEFAULT_OUT_EN = REPO_ROOT / "build" / "docs" / "generated" / "instruction-family-matrix.md"
DEFAULT_OUT_ZH = REPO_ROOT / "build" / "docs" / "generated" / "instruction-family-matrix_zh.md"

CATEGORY_ZH = {
    "Synchronization": "同步",
    "Manual / Resource Binding": "手动 / 资源绑定",
    "Elementwise (Tile-Tile)": "逐元素（Tile-Tile）",
    "Tile-Scalar / Tile-Immediate": "Tile-标量 / Tile-立即数",
    "Axis Reduce / Expand": "轴归约 / 扩展",
    "Padding": "填充",
    "Memory (GM <-> Tile)": "内存（GM <-> Tile）",
    "Matrix Multiply": "矩阵乘",
    "Data Movement / Layout": "数据搬运 / 布局",
    "Complex": "复杂指令",
}


def load_manifest(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("instructions", [])
    if not isinstance(entries, list):
        raise ValueError("manifest 'instructions' must be a list")
    out: List[Dict[str, object]] = []
    seen: set[str] = set()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("each manifest instruction must be an object")
        instr = str(item.get("instruction", "")).strip()
        if not instr:
            raise ValueError("manifest has empty instruction name")
        if instr in seen:
            raise ValueError(f"duplicate instruction in manifest: {instr}")
        seen.add(instr)
        out.append(item)
    return out


def parse_header_instr(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    names = re.findall(r"PTO_INST\s+(?:void|RecordEvent)\s+([A-Z][A-Z0-9_]+)\s*\(", text)
    out: List[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def group_by_category(entries: Iterable[Dict[str, object]]) -> OrderedDict[str, List[Dict[str, object]]]:
    grouped: OrderedDict[str, List[Dict[str, object]]] = OrderedDict()
    for item in entries:
        cat = str(item.get("category", "Uncategorized")).strip() or "Uncategorized"
        grouped.setdefault(cat, []).append(item)
    return grouped


def category_count_table(grouped: OrderedDict[str, List[Dict[str, object]]], zh: bool) -> List[str]:
    lines: List[str] = []
    if zh:
        lines.append("| 分类 | 指令数量 |")
    else:
        lines.append("| Category | Instruction Count |")
    lines.append("|---|---:|")
    for cat, items in grouped.items():
        cat_name = CATEGORY_ZH.get(cat, cat) if zh else cat
        lines.append(f"| {cat_name} | {len(items)} |")
    lines.append(f"| {'总计' if zh else 'Total'} | {sum(len(v) for v in grouped.values())} |")
    return lines


def matrix_rows(grouped: OrderedDict[str, List[Dict[str, object]]], zh: bool) -> List[str]:
    lines: List[str] = []
    if zh:
        lines.append("| 分类 | 指令 | 图示模板 | 操作数契约 | 语义页面 |")
    else:
        lines.append("| Category | Instruction | Diagram Template | Operand Contract | Semantic Page |")
    lines.append("|---|---|---|---|---|")
    for cat, items in grouped.items():
        cat_name = CATEGORY_ZH.get(cat, cat) if zh else cat
        for item in items:
            instr = str(item["instruction"])
            diagram = str(item.get("diagram_template", "")).strip() or "-"
            operands = item.get("operands", [])
            if isinstance(operands, list) and operands:
                op_text = ", ".join(str(x) for x in operands)
            else:
                op_text = "-"
            page = f"docs/isa/{instr}_zh.md" if zh else f"docs/isa/{instr}.md"
            lines.append(f"| {cat_name} | `{instr}` | `{diagram}` | `{op_text}` | `{page}` |")
    return lines


def render_en(entries: List[Dict[str, object]], header_instrs: List[str]) -> str:
    grouped = group_by_category(entries)
    manifest_set = {str(item["instruction"]) for item in entries}
    header_set = set(header_instrs)
    missing_in_manifest = sorted(header_set - manifest_set)
    extra_in_manifest = sorted(manifest_set - header_set)

    lines: List[str] = []
    lines.append("# Appendix D. Instruction Family Matrix")
    lines.append("")
    lines.append("## D.1 Scope")
    lines.append("")
    lines.append("This appendix is generated from `docs/isa/manifest.yaml` and provides a source-synchronized matrix of PTO virtual instruction families.")
    lines.append("")
    lines.append("## D.2 Coverage summary")
    lines.append("")
    lines.extend(category_count_table(grouped, zh=False))
    lines.append("")
    lines.append("## D.3 Header synchronization status")
    lines.append("")
    lines.append(f"- Header inventory source: `include/pto/common/pto_instr.hpp` ({len(header_instrs)} unique instruction APIs)")
    lines.append(f"- Manifest inventory source: `docs/isa/manifest.yaml` ({len(entries)} entries)")
    lines.append(f"- Missing in manifest: {', '.join(missing_in_manifest) if missing_in_manifest else 'none'}")
    lines.append(f"- Present in manifest but missing in header: {', '.join(extra_in_manifest) if extra_in_manifest else 'none'}")
    lines.append("")
    lines.append("## D.4 Family matrix")
    lines.append("")
    lines.extend(matrix_rows(grouped, zh=False))
    lines.append("")
    lines.append("## D.5 Notes")
    lines.append("")
    lines.append("- Per-instruction semantics remain canonical in `docs/isa/*.md`.")
    lines.append("- This appendix is a taxonomy and coverage matrix, not a replacement for per-op normative semantics.")
    lines.append("")
    return "\n".join(lines)


def render_zh(entries: List[Dict[str, object]], header_instrs: List[str]) -> str:
    grouped = group_by_category(entries)
    manifest_set = {str(item["instruction"]) for item in entries}
    header_set = set(header_instrs)
    missing_in_manifest = sorted(header_set - manifest_set)
    extra_in_manifest = sorted(manifest_set - header_set)

    lines: List[str] = []
    lines.append("# 附录 D. 指令族矩阵")
    lines.append("")
    lines.append("## D.1 范围")
    lines.append("")
    lines.append("本附录由 `docs/isa/manifest.yaml` 自动生成，用于给出 PTO 虚拟指令族的源同步矩阵。")
    lines.append("")
    lines.append("## D.2 覆盖统计")
    lines.append("")
    lines.extend(category_count_table(grouped, zh=True))
    lines.append("")
    lines.append("## D.3 头文件同步状态")
    lines.append("")
    lines.append(f"- 头文件清单来源：`include/pto/common/pto_instr.hpp`（{len(header_instrs)} 个唯一指令 API）")
    lines.append(f"- Manifest 清单来源：`docs/isa/manifest.yaml`（{len(entries)} 条目）")
    lines.append(f"- 头文件有但 manifest 缺失：{', '.join(missing_in_manifest) if missing_in_manifest else '无'}")
    lines.append(f"- manifest 有但头文件缺失：{', '.join(extra_in_manifest) if extra_in_manifest else '无'}")
    lines.append("")
    lines.append("## D.4 指令族矩阵")
    lines.append("")
    lines.extend(matrix_rows(grouped, zh=True))
    lines.append("")
    lines.append("## D.5 说明")
    lines.append("")
    lines.append("- 逐条指令语义仍以 `docs/isa/*_zh.md` 为准。")
    lines.append("- 本附录用于分类与覆盖追踪，不替代逐条指令的规范化语义描述。")
    lines.append("")
    return "\n".join(lines)


def write_or_check(path: Path, content: str, check: bool) -> List[str]:
    errors: List[str] = []
    if check:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != content:
            errors.append(f"out of date: {path}")
    else:
        path.write_text(content, encoding="utf-8")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PTO ISA instruction-family matrix")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--out-en", type=Path, default=DEFAULT_OUT_EN)
    parser.add_argument("--out-zh", type=Path, default=DEFAULT_OUT_ZH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    entries = load_manifest(args.manifest)
    header_instrs = parse_header_instr(args.header)

    args.out_en.parent.mkdir(parents=True, exist_ok=True)
    args.out_zh.parent.mkdir(parents=True, exist_ok=True)

    errors: List[str] = []
    errors.extend(write_or_check(args.out_en, render_en(entries, header_instrs), args.check))
    errors.extend(write_or_check(args.out_zh, render_zh(entries, header_instrs), args.check))

    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1

    if args.check:
        print("OK: virtual manual matrix files are synchronized.")
    else:
        print("Generated virtual manual matrix appendices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
