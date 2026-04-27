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

"""Generate ISA index documents from docs/isa/manifest.yaml.

Manifest format is YAML-compatible JSON:
{
  "instructions": [
    {
      "instruction": "TADD",
      "category": "Elementwise (Tile-Tile)",
      "summary_en": "Elementwise add of two tiles.",
      "summary_zh": "两个 Tile 的逐元素加法。",
      "diagram_template": "elementwise",
      "operands": ["dst", "src0", "src1"],
      "notes": []
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "isa" / "manifest.yaml"
DEFAULT_ISA_README = REPO_ROOT / "docs" / "isa" / "README.md"
DEFAULT_ISA_README_ZH = REPO_ROOT / "docs" / "isa" / "README_zh.md"
DEFAULT_PTOISA = REPO_ROOT / "docs" / "PTOISA.md"
DEFAULT_PTOISA_ZH = REPO_ROOT / "docs" / "PTOISA_zh.md"

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
    instructions = data.get("instructions", [])
    if not isinstance(instructions, list):
        raise ValueError("manifest 'instructions' must be a list")

    seen: set[str] = set()
    out: List[Dict[str, object]] = []
    for item in instructions:
        if not isinstance(item, dict):
            raise ValueError("each manifest instruction must be an object")
        instr = str(item.get("instruction", "")).strip()
        if not instr:
            raise ValueError("manifest entry missing instruction name")
        if instr in seen:
            raise ValueError(f"duplicate instruction in manifest: {instr}")
        seen.add(instr)
        out.append(item)
    return out


def group_by_category(entries: List[Dict[str, object]]) -> OrderedDict[str, List[Dict[str, object]]]:
    grouped: OrderedDict[str, List[Dict[str, object]]] = OrderedDict()
    for e in entries:
        cat = str(e.get("category", "Uncategorized"))
        grouped.setdefault(cat, []).append(e)
    return grouped


def render_isa_readme(entries: List[Dict[str, object]]) -> str:
    grouped = group_by_category(entries)
    lines: List[str] = []
    lines.append('<p align="center">')
    lines.append('  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />')
    lines.append("</p>")
    lines.append("")
    lines.append("# PTO ISA Reference")
    lines.append("")
    lines.append("This directory contains the per-instruction reference for the PTO Tile Lib ISA.")
    lines.append("")
    lines.append("- Source of truth (C++ intrinsics): `include/pto/common/pto_instr.hpp`")
    lines.append("- Common conventions (operands, events, modifiers): `docs/isa/conventions.md`")
    lines.append("")
    for cat, cat_entries in grouped.items():
        lines.append(f"## {cat}")
        for e in cat_entries:
            instr = str(e["instruction"])
            summary = str(e.get("summary_en", "")).strip()
            suffix = f" - {summary}" if summary else ""
            lines.append(f"- [{instr}]({instr}.md){suffix}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_isa_readme_zh(entries: List[Dict[str, object]]) -> str:
    grouped = group_by_category(entries)
    lines: List[str] = []
    lines.append('<p align="center">')
    lines.append('  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />')
    lines.append("</p>")
    lines.append("")
    lines.append("# PTO ISA 参考")
    lines.append("")
    lines.append("本目录是 PTO Tile Lib ISA 的指令参考（每条指令一页）。")
    lines.append("")
    lines.append("- 权威来源：`include/pto/common/pto_instr.hpp`")
    lines.append("- 通用约定（操作数、事件、修饰符）：`docs/isa/conventions_zh.md`")
    lines.append("")
    for cat, cat_entries in grouped.items():
        lines.append(f"## {CATEGORY_ZH.get(cat, cat)}")
        for e in cat_entries:
            instr = str(e["instruction"])
            summary = str(e.get("summary_zh", "")).strip()
            suffix = f" - {summary}" if summary else ""
            lines.append(f"- [{instr}]({instr}_zh.md){suffix}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_ptoisa(entries: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# PTO ISA Overview")
    lines.append("")
    lines.append("This page is the source-synchronized ISA index generated from `docs/isa/manifest.yaml`.")
    lines.append("")
    lines.append("## Docs Contents")
    lines.append("")
    lines.append("| Area | Page | Description |")
    lines.append("|---|---|---|")
    lines.append("| Overview | [`docs/README.md`](README.md) | PTO ISA guide entry point and navigation. |")
    lines.append("| Overview | [`docs/PTOISA.md`](PTOISA.md) | This page (overview + full instruction index). |")
    lines.append("| ISA reference | [`docs/isa/README.md`](isa/README.md) | Per-instruction reference directory index. |")
    lines.append("| ISA reference | [`docs/isa/conventions.md`](isa/conventions.md) | Shared notation, operands, events, and modifiers. |")
    lines.append("| Syntax and operands | [`docs/isa/syntax-and-operands/assembly-model.md`](isa/syntax-and-operands/assembly-model.md) | PTO-AS spelling inside the PTO ISA syntax model. |")
    lines.append("| Source of truth | [`include/pto/common/pto_instr.hpp`](reference/pto-intrinsics-header.md) | C++ intrinsic API (authoritative). |")
    lines.append("")
    lines.append("## Instruction Index (All PTO Instructions)")
    lines.append("")
    lines.append("| Category | Instruction | Description |")
    lines.append("|---|---|---|")
    for e in entries:
        cat = str(e.get("category", ""))
        instr = str(e["instruction"])
        summary = str(e.get("summary_en", "")).strip()
        lines.append(f"| {cat} | [`{instr}`](isa/{instr}.md) | {summary} |")
    lines.append("")
    return "\n".join(lines)


def render_ptoisa_zh(entries: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# PTO ISA 概述")
    lines.append("")
    lines.append("本文档为根据 `docs/isa/manifest.yaml` 自动生成的 ISA 索引。")
    lines.append("")
    lines.append("## 文档目录")
    lines.append("")
    lines.append("| 领域 | 页面 | 描述 |")
    lines.append("|---|---|---|")
    lines.append("| 概述 | [`docs/README_zh.md`](README_zh.md) | PTO ISA 指南入口与导航。 |")
    lines.append("| 概述 | [`docs/PTOISA_zh.md`](PTOISA_zh.md) | 本页（概述 + 全量指令索引）。 |")
    lines.append("| ISA 参考 | [`docs/isa/README_zh.md`](isa/README_zh.md) | 每条指令参考目录。 |")
    lines.append("| ISA 参考 | [`docs/isa/conventions_zh.md`](isa/conventions_zh.md) | 通用符号、操作数、事件与修饰符。 |")
    lines.append("| 语法与操作数 | [`docs/isa/syntax-and-operands/assembly-model_zh.md`](isa/syntax-and-operands/assembly-model_zh.md) | PTO ISA 语法模型中的 PTO-AS 拼写。 |")
    lines.append("| 权威源 | [`include/pto/common/pto_instr.hpp`](reference/pto-intrinsics-header_zh.md) | C++ intrinsic API（权威来源）。 |")
    lines.append("")
    lines.append("## 指令索引（全部 PTO 指令）")
    lines.append("")
    lines.append("| 分类 | 指令 | 描述 |")
    lines.append("|---|---|---|")
    for e in entries:
        cat = CATEGORY_ZH.get(str(e.get("category", "")), str(e.get("category", "")))
        instr = str(e["instruction"])
        summary = str(e.get("summary_zh", "")).strip()
        lines.append(f"| {cat} | [`{instr}`](isa/{instr}_zh.md) | {summary} |")
    lines.append("")
    return "\n".join(lines)


def write_or_check(path: Path, content: str, check: bool) -> List[str]:
    errors: List[str] = []
    if check:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != content:
            errors.append(f"out of date: {path}")
        return errors
    path.write_text(content, encoding="utf-8")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ISA index files from manifest")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    entries = load_manifest(args.manifest)

    errors: List[str] = []
    errors += write_or_check(DEFAULT_ISA_README, render_isa_readme(entries), args.check)
    errors += write_or_check(DEFAULT_ISA_README_ZH, render_isa_readme_zh(entries), args.check)
    errors += write_or_check(DEFAULT_PTOISA, render_ptoisa(entries), args.check)
    errors += write_or_check(DEFAULT_PTOISA_ZH, render_ptoisa_zh(entries), args.check)

    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1
    if args.check:
        print("OK: ISA index files are synchronized with manifest.")
    else:
        print("Generated ISA index files from manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
