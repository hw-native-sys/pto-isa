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
    "Cross-core Communication": "核间通信",
}


REMOVED_INTERFACES_EN = [
    "`TADDC`",
    "`TAddDeqRelu`",
    "`TADDReluConv`",
    "`TADDSC`",
    "`TFUSEDMULADDRELU`",
    "`TGET_SCALE_ADDR`",
    "`TPairReduceSum`",
    "`TSUBC`",
    "`TSUBRELU`",
    "`TSUBRELUCONV`",
    "`TSUBSC`",
    "`TSUBVIEW`",
    "`TSYNC`",
]

REMOVED_INTERFACES_ZH = [
    "`TADDC`",
    "`TAddDeqRelu`",
    "`TADDReluConv`",
    "`TADDSC`",
    "`TFUSEDMULADDRELU`",
    "`TGET_SCALE_ADDR`",
    "`TPairReduceSum`",
    "`TSUBC`",
    "`TSUBRELU`",
    "`TSUBRELUCONV`",
    "`TSUBSC`",
    "`TSUBVIEW`",
    "`TSYNC`",
]

COMMUNICATION_ENTRIES_EN = [
    ("TPUT", "Remote write: transfer local data to remote NPU memory (GM → UB → GM)."),
    ("TGET", "Remote read: read remote NPU data to local memory (GM → UB → GM)."),
    ("TPUT_ASYNC", "Asynchronous remote write (local GM → DMA engine → remote GM)."),
    ("TGET_ASYNC", "Asynchronous remote read (remote GM → DMA engine → local GM)."),
    ("TNOTIFY", "Send flag notification to remote NPU."),
    ("TWAIT", "Blocking wait until signal(s) meet comparison condition."),
    ("TTEST", "Non-blocking test if signal(s) meet comparison condition."),
    ("TGATHER", "Gather data from all ranks and concatenate along DIM_3."),
    ("TSCATTER", "Scatter data to all ranks by splitting along DIM_3."),
    ("TREDUCE", "Gather and reduce data from all ranks element-wise to local."),
    ("TBROADCAST", "Broadcast data from current NPU to all ranks."),
]

COMMUNICATION_ENTRIES_ZH = [
    ("TPUT", "远程写：将本地数据传输到远端 NPU 内存（GM → UB → GM）。"),
    ("TGET", "远程读：将远端 NPU 数据读取到本地内存（GM → UB → GM）。"),
    ("TPUT_ASYNC", "异步远程写（本地 GM → DMA 引擎 → 远端 GM）。"),
    ("TGET_ASYNC", "异步远程读（远端 GM → DMA 引擎 → 本地 GM）。"),
    ("TNOTIFY", "向远端 NPU 发送标志通知。"),
    ("TWAIT", "阻塞等待，直到信号满足比较条件。"),
    ("TTEST", "非阻塞检测信号是否满足比较条件。"),
    ("TGATHER", "从所有 rank 收集数据并沿 DIM_3 拼接。"),
    ("TSCATTER", "将数据沿 DIM_3 拆分并分发到所有 rank。"),
    ("TREDUCE", "从所有 rank 收集数据并逐元素归约到本地。"),
    ("TBROADCAST", "将当前 NPU 的数据广播到所有 rank。"),
]


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


def append_isa_readme_header(lines: List[str]) -> None:
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


def append_isa_readme_header_zh(lines: List[str]) -> None:
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


def append_removed_interfaces_en(lines: List[str]) -> None:
    lines.append("## Removed interfaces and migration notes")
    lines.append("")
    lines.append("The current ISA reference no longer exposes the following legacy instruction interfaces:")
    lines.append("")
    for item in REMOVED_INTERFACES_EN:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Migration guidance:")
    lines.append("")
    lines.append(
        "- Effective version: this cleanup takes effect in PTO ISA v9.2.0. The compatibility window for "
        "these legacy wrappers is closed; no public wrapper is retained."
    )
    lines.append(
        "- Replace `TSYNC(events...)` with ordinary event-based ordering: pass the event object to the "
        "consumer intrinsic, or call `WaitAllEvents(events...)` before the consumer when an explicit wait "
        "is required."
    )
    lines.append(
        "- `TSUBVIEW` is not a public ISA replacement. In-repository implementations may use "
        "`pto::detail::PtoSubTileView` as an internal helper; external code should express the data view "
        "through supported tile construction and public data movement APIs."
    )
    lines.append(
        "- Replace ternary/scalar fused arithmetic forms with the corresponding primitive arithmetic sequence, "
        "such as `TADD`, `TSUB`, `TADDS`, `TSUBS`, `TMUL`, `TMADD`, and `TRELU`."
    )
    lines.append("- Rename legacy fused multiply-add APIs: `TFUSEDMULADD` to `TMADD`.")
    lines.append(
        "- Replace fused add/ReLU/convert or add/dequant/ReLU forms with explicit arithmetic, "
        "conversion/dequantization, and `TRELU` steps."
    )
    lines.append(
        "- Replace `TPairReduceSum` with the supported row/column reduction primitives that match the target layout."
    )
    lines.append(
        "- Do not call `TGET_SCALE_ADDR`; for AUTO-mode MX tests, bind the scale tile address from the data tile "
        "in test code before invoking the MX matmul primitive."
    )
    lines.append("")


def append_removed_interfaces_zh(lines: List[str]) -> None:
    lines.append("## 删除接口与迁移说明")
    lines.append("")
    lines.append("当前 ISA 参考不再公开以下历史指令接口：")
    lines.append("")
    for item in REMOVED_INTERFACES_ZH:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("迁移建议：")
    lines.append("")
    lines.append(
        "- 生效版本：本次清理自 PTO ISA v9.2.0 起生效；这些历史 wrapper 的兼容窗口关闭，不再保留公开 wrapper。"
    )
    lines.append(
        "- 将 `TSYNC(events...)` 替换为普通 event 顺序表达：把 event 对象传给消费端 intrinsic，"
        "或在确需显式等待时调用 `WaitAllEvents(events...)`。"
    )
    lines.append(
        "- `TSUBVIEW` 不提供公开 ISA 替代接口。仓内实现可使用 `pto::detail::PtoSubTileView` 作为内部 helper；"
        "外部代码应通过受支持的 Tile 构造和公开数据搬运 API 表达数据视图。"
    )
    lines.append(
        "- 将三元/标量融合算术形式替换为对应基础算术序列，例如 `TADD`、`TSUB`、`TADDS`、`TSUBS`、"
        "`TMUL`、`TMADD` 和 `TRELU`。"
    )
    lines.append("- 将历史融合乘加接口重命名：`TFUSEDMULADD` 改为 `TMADD`。")
    lines.append("- 将融合 add/ReLU/convert 或 add/dequant/ReLU 形式拆分为显式算术、转换/反量化和 `TRELU` 步骤。")
    lines.append("- 将 `TPairReduceSum` 替换为与目标 layout 匹配的现有行/列归约原语。")
    lines.append(
        "- 不再调用 `TGET_SCALE_ADDR`；AUTO 模式 MX 测试可在调用 MX matmul 原语前，"
        "在测试代码中根据数据 Tile 绑定 scale Tile 地址。"
    )
    lines.append("")


def append_isa_categories(lines: List[str], grouped: OrderedDict[str, List[Dict[str, object]]]) -> None:
    for cat, cat_entries in grouped.items():
        lines.append(f"## {cat}")
        for e in cat_entries:
            instr = str(e["instruction"])
            summary = str(e.get("summary_en", "")).strip()
            suffix = f" - {summary}" if summary else ""
            lines.append(f"- [{instr}]({instr}.md){suffix}")
        lines.append("")


def append_isa_categories_zh(lines: List[str], grouped: OrderedDict[str, List[Dict[str, object]]]) -> None:
    for cat, cat_entries in grouped.items():
        lines.append(f"## {CATEGORY_ZH.get(cat, cat)}")
        for e in cat_entries:
            instr = str(e["instruction"])
            summary = str(e.get("summary_zh", "")).strip()
            suffix = f" - {summary}" if summary else ""
            lines.append(f"- [{instr}]({instr}_zh.md){suffix}")
        lines.append("")


def render_isa_readme(entries: List[Dict[str, object]]) -> str:
    grouped = group_by_category(entries)
    lines: List[str] = []
    append_isa_readme_header(lines)
    append_removed_interfaces_en(lines)
    append_isa_categories(lines, grouped)
    return "\n".join(lines).rstrip() + "\n"


def render_isa_readme_zh(entries: List[Dict[str, object]]) -> str:
    grouped = group_by_category(entries)
    lines: List[str] = []
    append_isa_readme_header_zh(lines)
    append_removed_interfaces_zh(lines)
    append_isa_categories_zh(lines, grouped)
    return "\n".join(lines).rstrip() + "\n"


def render_ptoisa(entries: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# PTO ISA Overview")
    lines.append("")
    lines.append(
        "This page is the source-synchronized ISA index generated from `docs/isa/manifest.yaml` and the "
        "communication ISA reference."
    )
    lines.append("")
    lines.append("## Docs Contents")
    lines.append("")
    lines.append("| Area | Page | Description |")
    lines.append("|---|---|---|")
    lines.append("| Overview | [`docs/README.md`](README.md) | PTO ISA guide entry point and navigation. |")
    lines.append("| Overview | [`docs/PTOISA.md`](PTOISA.md) | This page (overview + full instruction index). |")
    lines.append(
        "| ISA reference | [`docs/isa/README.md`](isa/README.md) | Per-instruction reference directory index. |"
    )
    lines.append(
        "| ISA reference | [`docs/isa/conventions.md`](isa/conventions.md) | Shared notation, operands, "
        "events, and modifiers. |"
    )
    lines.append(
        "| Communication ISA | [`docs/isa/comm/README.md`](isa/comm/README.md) | Per-instruction "
        "communication ISA reference. |"
    )
    lines.append(
        "| Source of truth | [`include/pto/common/pto_instr.hpp`](reference/pto-intrinsics-header.md) | "
        "C++ intrinsic API (authoritative). |"
    )
    lines.append("| PTO Auto Mode | [`docs/auto_mode/README.md`](README.md) | PTO auto mode guide entry point. |")
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
    for instr, summary in COMMUNICATION_ENTRIES_EN:
        lines.append(f"| Communication | [`{instr}`](isa/comm/{instr}.md) | {summary} |")
    lines.append("")
    return "\n".join(lines)


def render_ptoisa_zh(entries: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# PTO ISA 概述")
    lines.append("")
    lines.append("本文档为根据 `docs/isa/manifest.yaml` 和通信 ISA 参考自动生成的 ISA 索引。")
    lines.append("")
    lines.append("## 文档目录")
    lines.append("")
    lines.append("| 领域 | 页面 | 描述 |")
    lines.append("|---|---|---|")
    lines.append("| 概述 | [`docs/README_zh.md`](README_zh.md) | PTO ISA 指南入口与导航。 |")
    lines.append("| 概述 | [`docs/PTOISA_zh.md`](PTOISA_zh.md) | 本页（概述 + 全量指令索引）。 |")
    lines.append("| ISA 参考 | [`docs/isa/README_zh.md`](isa/README_zh.md) | 每条指令参考目录。 |")
    lines.append(
        "| ISA 参考 | [`docs/isa/conventions_zh.md`](isa/conventions_zh.md) | 通用符号、操作数、事件与修饰符。 |"
    )
    lines.append("| 通信 ISA | [`docs/isa/comm/README_zh.md`](isa/comm/README_zh.md) | 每条通信 ISA 指令参考。 |")
    lines.append(
        "| 权威源 | [`include/pto/common/pto_instr.hpp`](reference/pto-intrinsics-header_zh.md) | "
        "C++ intrinsic API（权威来源）。 |"
    )
    lines.append("| PTO auto 模式 | [`docs/auto_mode/README_zh.md`](README_zh.md) | PTO auto模式文档入口 |")
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
    for instr, summary in COMMUNICATION_ENTRIES_ZH:
        lines.append(f"| 通信 | [`{instr}`](isa/comm/{instr}_zh.md) | {summary} |")
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
