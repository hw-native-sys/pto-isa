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

"""Normalize PTO ISA instruction pages and generate Chinese counterparts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "docs" / "isa" / "manifest.yaml"
ISA_DIR = REPO_ROOT / "docs" / "isa"
PTO_ISA_LEVEL_TABLE = Path.home() / "pto-isa.txt"

_LANG_LINK_LINE_RE = re.compile(r"^>\s*(?:Chinese|English)\s+version:\s*.*$", re.IGNORECASE)
_INSTR_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_CODE_BLOCK_RE = re.compile(r"```(?:[A-Za-z0-9_+-]+)?\n(.*?)\n```", re.DOTALL)

_LEVEL_ALIAS = {
    # Family variants that share the same L1/L2 row in the latest PTO-AS table.
    "TGEMV_ACC": "TGEMV",
    "TGEMV_BIAS": "TGEMV",
}

_EXPLICIT_FALLBACK_FORMS: Dict[str, Dict[str, str]] = {
    "TEXTRACT_FP": {
        "level1": "%dst = pto.textract_fp %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>",
        "level2": "pto.textract_fp ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)",
    },
    "TFILLPAD_EXPAND": {
        "level1": "%dst = pto.tfillpad_expand %src : !pto.tile<...> -> !pto.tile<...>",
        "level2": "pto.tfillpad_expand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)",
    },
    "TFILLPAD_INPLACE": {
        "level1": "%dst = pto.tfillpad_inplace %src : !pto.tile<...> -> !pto.tile<...>",
        "level2": "pto.tfillpad_inplace ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)",
    },
    "TIMG2COL": {
        "level1": "%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>",
        "level2": "pto.timg2col ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)",
    },
    "TINSERT_FP": {
        "level1": "%dst = pto.tinsert_fp %src, %fp, %idxrow, %idxcol : (!pto.tile<...>, !pto.tile<...>, dtype, dtype) -> !pto.tile<...>",
        "level2": "pto.tinsert_fp ins(%src, %fp, %idxrow, %idxcol : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)",
    },
    "TQUANT": {
        "level1": "%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>",
        "level2": "pto.tquant ins(%src, %qp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)",
    },
    "SETFMATRIX": {
        "level1": "pto.setfmatrix %cfg : !pto.fmatrix_config -> ()",
        "level2": "pto.setfmatrix ins(%cfg : !pto.fmatrix_config) outs()",
    },
    "TTRI": {
        "level1": "%dst = pto.ttri %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>",
        "level2": "pto.ttri ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)",
    },
}


def load_manifest() -> List[Dict[str, object]]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return data["instructions"]


def _normalize_cell(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(line.rstrip() for line in lines)


def load_level_formats(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}

    formats: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quotechar='"')
        for row in reader:
            if len(row) < 4:
                continue
            raw_name = row[1].strip()
            if not raw_name or raw_name == "PTO":
                continue

            name = re.sub(r"\(.*\)$", "", raw_name).strip().upper()
            if not _INSTR_NAME_RE.fullmatch(name):
                continue

            level1 = _normalize_cell(row[2])
            level2 = _normalize_cell(row[3])
            notes = _normalize_cell(row[4]) if len(row) > 4 else ""
            if not (level1 or level2 or notes):
                continue

            formats[name] = {"level1": level1, "level2": level2, "notes": notes}

    for dst, src in _LEVEL_ALIAS.items():
        if dst not in formats and src in formats:
            formats[dst] = dict(formats[src])
    return formats


def _split_sections(text: str) -> Dict[str, str]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, re.MULTILINE))
    out: Dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.end() + 1
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        out[name] = text[start:end]
    return out


def _fallback_level1(instr: str, assembly_body: str) -> str:
    explicit = _EXPLICIT_FALLBACK_FORMS.get(instr, {}).get("level1")
    if explicit:
        return explicit

    for block in _CODE_BLOCK_RE.findall(assembly_body):
        candidate = _normalize_cell(block)
        if candidate:
            # Normalize shorthand op form (`tadd`) to the explicit PTO-AS style (`pto.tadd`).
            candidate = re.sub(r"(^\s*[%@][^=\n]*=\s*)([a-z][a-z0-9_.]+)\b", r"\1pto.\2", candidate, flags=re.MULTILINE)
            candidate = re.sub(r"(^\s*)([a-z][a-z0-9_.]+)\b", r"\1pto.\2", candidate, flags=re.MULTILINE)
            return candidate
    return f"%dst = pto.{instr.lower()} ..."


def _to_tile_buf_types(type_expr: str) -> str:
    return re.sub(r"!pto\.tile<", "!pto.tile_buf<", type_expr)


def _render_dps_from_level1(level1: str) -> str:
    lines: List[str] = []
    for raw_line in level1.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("//"):
            continue

        typed = re.match(
            r"(?:(?P<res>[%@][^=\n]*?)\s*=\s*)?pto\.(?P<op>[a-z][a-z0-9_.]*)\s*(?P<args>.*?)\s*:\s*(?P<ins>.+?)\s*->\s*(?P<outs>.+)$",
            line,
        )
        if typed:
            res = (typed.group("res") or "").strip()
            op = typed.group("op").strip()
            args = typed.group("args").strip()
            in_types = _to_tile_buf_types(typed.group("ins").strip())
            out_types_raw = typed.group("outs").strip()
            out_types = _to_tile_buf_types(out_types_raw)

            if args:
                ins_clause = f"{args} : {in_types}"
            else:
                ins_clause = f": {in_types}"

            if out_types_raw == "()":
                lines.append(f"pto.{op} ins({ins_clause}) outs()")
            else:
                out_name = res if res else "%dst"
                lines.append(f"pto.{op} ins({ins_clause}) outs({out_name} : {out_types})")
            continue

        semi_typed = re.match(
            r"(?:(?P<res>[%@][^=\n]*?)\s*=\s*)?pto\.(?P<op>[a-z][a-z0-9_.]*)\s*(?P<args>.*?)\s*:\s*(?P<ins>.+)$",
            line,
        )
        if semi_typed:
            res = (semi_typed.group("res") or "").strip()
            op = semi_typed.group("op").strip()
            args = semi_typed.group("args").strip()
            in_types = _to_tile_buf_types(semi_typed.group("ins").strip())

            if args:
                ins_clause = f"{args} : {in_types}"
            else:
                ins_clause = f": {in_types}"

            if res:
                lines.append(f"pto.{op} ins({ins_clause}) outs({res} : !pto.tile_buf<...>)")
            else:
                lines.append(f"pto.{op} ins({ins_clause}) outs()")
            continue

        untyped = re.match(r"(?:(?P<res>[%@][^=\n]*?)\s*=\s*)?pto\.(?P<op>[a-z][a-z0-9_.]*)\s*(?P<args>.*)$", line)
        if untyped:
            op = untyped.group("op").strip()
            args = untyped.group("args").strip()
            res = (untyped.group("res") or "").strip()
            if args and res:
                lines.append(f"pto.{op} ins({args}) outs({res} : !pto.tile_buf<...>)")
            elif args:
                lines.append(f"pto.{op} ins({args}) outs()")
            elif res:
                lines.append(f"pto.{op} ins() outs({res} : !pto.tile_buf<...>)")
            else:
                lines.append(f"pto.{op} ins() outs()")

    return "\n".join(lines).strip()


def _fallback_level2(instr: str, level1: str) -> str:
    explicit = _EXPLICIT_FALLBACK_FORMS.get(instr, {}).get("level2")
    if explicit:
        return explicit

    synthesized = _render_dps_from_level1(level1)
    if synthesized:
        return synthesized

    return f"pto.{instr.lower()} ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)"


def _sync_level2_from_table(level_formats: Dict[str, Dict[str, str]]) -> str:
    segments: List[str] = []
    for name in ("RECORD_EVENT", "WAIT_EVENT", "BARRIER"):
        item = level_formats.get(name)
        if not item:
            continue
        body = item.get("level2", "").strip()
        note = item.get("notes", "").strip()
        if body:
            segments.append(body)
        if note:
            segments.append(f"// {note}")
    return "\n".join(segments).strip()


def _resolve_level_formats(instr: str, assembly_body: str, level_formats: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    if instr == "TSYNC":
        level1 = "// Level 1 (SSA) does not support explicit synchronization primitives."
        level2 = _sync_level2_from_table(level_formats)
        if not level2:
            level2 = "pto.record_event[src_op, dst_op, eventID]\npto.wait_event[src_op, dst_op, eventID]\npto.barrier(op)\n\nNote: for the current PTO-DSL front-end flow, prefer sync-free source plus `ptoas --enable-insert-sync`."
        return {"level1": level1, "level2": level2}

    item = level_formats.get(instr, {})
    level1 = item.get("level1", "").strip()
    level2 = item.get("level2", "").strip()

    if not level1:
        level1 = _fallback_level1(instr, assembly_body)
    if not level2:
        level2 = _fallback_level2(instr, level1)
    return {"level1": level1, "level2": level2}


def ensure_ir_level_syntax(instr: str, text: str, level_formats: Dict[str, Dict[str, str]]) -> str:
    section_map = _split_sections(text)
    body = section_map.get("Assembly Syntax")
    if body is None:
        return text

    base = re.split(r"^###\s+IR Level 1 \(SSA\)\s*$", body, maxsplit=1, flags=re.MULTILINE)[0].rstrip()
    formats = _resolve_level_formats(instr, base, level_formats)

    level_block = (
        "\n\n### IR Level 1 (SSA)\n\n"
        "```text\n"
        f"{formats['level1']}\n"
        "```\n\n"
        "### IR Level 2 (DPS)\n\n"
        "```text\n"
        f"{formats['level2']}\n"
        "```\n"
    )

    new_body = (base + level_block).rstrip() + "\n"
    start_marker = re.compile(r"^##\s+Assembly Syntax\s*$", re.MULTILINE)
    start_match = start_marker.search(text)
    if not start_match:
        return text
    start = start_match.end() + 1

    next_heading = re.search(r"^##\s+.+$", text[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(text)
    return text[:start] + new_body + text[end:]


def template_new_page(instr: str, summary: str) -> str:
    if instr == "TGEMV_MX":
        return """# TGEMV_MX

## Introduction

GEMV with scaling tiles for mixed-precision / quantized matrix-vector compute on supported targets.

This instruction family extends `TGEMV` with additional scale operands (mx path). Accumulator and scale handling are target-dependent.

## Math Interpretation

Conceptually (base GEMV path):

$$
\\mathrm{C}_{0,j} = \\sum_{k=0}^{K-1} \\mathrm{A}_{0,k} \\cdot \\mathrm{B}_{k,j}
$$

For `TGEMV_MX`, scale tiles participate in implementation-defined mixed-precision reconstruction / scaling. The architectural contract is that output corresponds to the target-defined mx GEMV semantics.

## Assembly Syntax

PTO-AS form: see `docs/isa/syntax-and-operands/assembly-model.md`.

Schematic form:

```text
%acc = tgemv.mx %a, %a_scale, %b, %b_scale : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix,
                              TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &... events);
```

Additional overloads support accumulation/bias variants and `AccPhase` selection.

## Constraints

- Uses backend-specific mx legality checks for data types, tile locations, fractal/layout combinations, and scaling formats.
- Scale tile compatibility and accumulator promotion are implementation-defined by target backend.
- For portability, validate the exact `(A, B, scaleA, scaleB, C)` type tuple and tile layout against target implementation constraints.

## Examples

For practical usage patterns, see:

- `docs/isa/TMATMUL_MX.md`
- `docs/isa/TGEMV.md`
"""

    if instr == "TPARTMUL":
        return """# TPARTMUL

## Introduction

Partial elementwise multiply with implementation-defined handling of mismatched valid regions.

## Math Interpretation

For each element `(i, j)` in the destination valid region:

$$
\\mathrm{dst}_{i,j} =
\\begin{cases}
\\mathrm{src0}_{i,j} \\cdot \\mathrm{src1}_{i,j} & \\text{if both inputs are defined at } (i,j) \\\\\n+\\mathrm{src0}_{i,j} & \\text{if only src0 is defined at } (i,j) \\\\\n+\\mathrm{src1}_{i,j} & \\text{if only src1 is defined at } (i,j)
\\end{cases}
$$

## Assembly Syntax

PTO-AS form: see `docs/isa/syntax-and-operands/assembly-model.md`.

Synchronous form:

```text
%dst = tpartmul %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Constraints

- Element type/layout legality follows backend checks and is analogous to `TPARTADD` / `TPARTMAX` / `TPARTMIN`.
- Destination valid region defines the result domain.
- Partial-validity handling is implementation-defined for unsupported shape combinations.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TPARTMUL(dst, src0, src1);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  TPARTMUL(dst, src0, src1);
}
```
"""

    if instr == "SETTF32MODE":
        return """# SETTF32MODE

## Introduction

Configure HF32 transform mode (implementation-defined).

This instruction controls backend-specific HF32 transformation behavior used by supported compute paths.

## Math Interpretation

No direct tensor arithmetic is produced by this instruction. It updates target mode state used by subsequent instructions.

## Assembly Syntax

PTO-AS form: see `docs/isa/syntax-and-operands/assembly-model.md`.

Schematic form:

```text
settf32mode {enable = true, mode = ...}
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <bool isEnable, RoundMode hf32TransMode = RoundMode::CAST_ROUND, typename... WaitEvents>
PTO_INST RecordEvent SETTF32MODE(WaitEvents &... events);
```

## Constraints

- Available only when the corresponding backend capability macro is enabled.
- Exact mode values and hardware behavior are target-defined.
- This instruction has control-state side effects and should be ordered appropriately relative to dependent compute instructions.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_enable_hf32() {
  SETTF32MODE<true, RoundMode::CAST_ROUND>();
}
```
"""

    if instr == "SETTF32MODE":
        return """# SETTF32MODE

## Introduction

Configure TF32 transform mode (implementation-defined).

This instruction controls backend-specific TF32 transformation behavior used by supported compute paths.

## Math Interpretation

No direct tensor arithmetic is produced by this instruction. It updates target mode state used by subsequent instructions.

## Assembly Syntax

PTO-AS form: see `docs/isa/syntax-and-operands/assembly-model.md`.

Schematic form:

```text
settf32mode {enable = true, mode = ...}
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <bool isEnable, RoundMode tf32TransMode = RoundMode::CAST_ROUND, typename... WaitEvents>
PTO_INST RecordEvent SETTF32MODE(WaitEvents &... events);
```

## Constraints

- Available only when the corresponding backend capability macro is enabled.
- Exact mode values and hardware behavior are target-defined.
- This instruction has control-state side effects and should be ordered appropriately relative to dependent compute instructions.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_enable_tf32() {
  SETTF32MODE<true, RoundMode::CAST_ROUND>();
}
```
"""

    return f"""# {instr}

## Introduction

{summary}

## Math Interpretation

Semantics are instruction-specific. Unless stated otherwise, behavior is defined over the destination valid region.

## Assembly Syntax

PTO-AS form: see `docs/isa/syntax-and-operands/assembly-model.md`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`.

## Constraints

Refer to backend-specific legality checks for data type/layout/location/shape constraints.

## Examples

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.
"""


def ensure_top_block(instr: str, text: str) -> str:
    svg_token = f"../figures/isa/{instr}.svg"
    if svg_token in text:
        return text

    lines = text.splitlines()
    if not lines:
        return text

    insert_at = 1
    if len(lines) > 1 and lines[1].strip() == "":
        insert_at = 2

    block = [
        "## Tile Operation Diagram",
        "",
        f"![{instr} tile operation]({svg_token})",
        "",
    ]

    out = lines[:insert_at] + block + lines[insert_at:]
    return "\n".join(out).rstrip() + "\n"


def ensure_required_sections(instr: str, text: str) -> str:
    required = [
        ("Introduction", "## Introduction\n\nRefer to the authoritative summary in `docs/isa/manifest.yaml`.\n"),
        (
            "Math Interpretation",
            "## Math Interpretation\n\nUnless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.\n",
        ),
        (
            "Assembly Syntax",
            "## Assembly Syntax\n\nPTO-AS form: see `docs/isa/syntax-and-operands/assembly-model.md`.\n",
        ),
        (
            "C++ Intrinsic",
            "## C++ Intrinsic\n\nDeclared in `include/pto/common/pto_instr.hpp`.\n",
        ),
        (
            "Constraints",
            "## Constraints\n\nType/layout/location/shape legality is backend-dependent; treat implementation-specific notes as normative for that backend.\n",
        ),
        (
            "Examples",
            "## Examples\n\nSee related examples in `docs/isa/` and `docs/coding/tutorials/`.\n",
        ),
    ]

    out = text.rstrip() + "\n"
    for sec, fallback in required:
        pattern = re.compile(rf"^##\s+{re.escape(sec)}\s*$", re.MULTILINE)
        if not pattern.search(out):
            out += "\n" + fallback
    return out


def _strip_language_links(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if _LANG_LINK_LINE_RE.match(line.strip()):
            continue
        if "_zh.md" in line:
            continue
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def _extract_sections(md: str) -> Dict[str, str]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", md, re.MULTILINE))
    out: Dict[str, str] = {}
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end() + 1
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip("\n")
        out[title] = body
    return out


def _translate_zh_line_segment(seg: str) -> str:
    # Best-effort phrase mapping for common boilerplate and headings.
    table = [
        ("### IR Level 1 (SSA)", "### AS Level 1（SSA）"),
        ("### IR Level 2 (DPS)", "### AS Level 2（DPS）"),
        ("PTO-AS form: see ", "PTO-AS 形式：参见 "),
        ("Declared in ", "声明于 "),
        ("Level 1 (SSA) does not support explicit synchronization primitives.", "Level 1（SSA）不支持显式同步原语。"),
        (" in the valid region:", " 在有效区域内："),
        (" in the valid region.", " 在有效区域内。"),
        ("Synchronous form:", "同步形式："),
        ("### Auto", "### 自动（Auto）"),
        ("### Manual", "### 手动（Manual）"),
        ("Index-based gather (conceptual):", "基于索引的 gather（概念性定义）："),
        ("Mask-pattern gather:", "基于掩码模式的 gather："),
        ("Mask-pattern gather is", "掩码模式 gather 属于"),
        ("Exact index interpretation and bounds behavior are implementation-defined.", "索引解释方式与越界行为为实现定义。"),
        ("Implementation checks", "实现检查"),
        ("Valid region", "有效区域"),
        ("Runtime valid checks", "运行期有效区域检查"),
        ("Bounds / validity", "边界 / 有效性"),
        ("Data types", "数据类型"),
        ("Tile layout", "Tile 布局"),
        ("Tile shape/layout constraint", "Tile 形状/布局约束"),
        ("DType consistency", "数据类型一致性"),
        ("Recommended", "推荐"),
        ("To be removed", "将移除"),
        ("See related examples in `docs/isa/` and `docs/coding/tutorials/`.", "更多用法示例参见 `docs/isa/` 与 `docs/coding/tutorials/`。"),
        ("For each element", "对每个元素"),
        ("For each source element", "对每个源元素"),
        ("For each", "对每个"),
        ("Unless otherwise specified", "除非另有说明"),
        ("Semantics are instruction-specific.", "语义随指令而变化。"),
    ]
    for k, v in table:
        seg = seg.replace(k, v)
    return seg


def _translate_md_to_zh(md: str) -> str:
    out: List[str] = []
    in_code = False
    in_math = False
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if stripped.startswith("$$"):
            in_math = not in_math
            out.append(line)
            continue
        if in_code or in_math:
            out.append(line)
            continue

        # Translate outside inline code spans.
        parts = re.split(r"(`[^`]*`)", line)
        for i, p in enumerate(parts):
            if p.startswith("`") and p.endswith("`"):
                continue
            parts[i] = _translate_zh_line_segment(p)
        out.append("".join(parts))

    return "\n".join(out).rstrip()


def build_zh_page(instr: str, summary_zh: str, en_text: str) -> str:
    sections = _extract_sections(_strip_language_links(en_text))
    intro_en = sections.get("Introduction", "").strip()
    math_en = sections.get("Math Interpretation", "").strip()
    asm_en = sections.get("Assembly Syntax", "").strip()
    cpp_en = sections.get("C++ Intrinsic", "").strip()
    cons_en = sections.get("Constraints", "").strip()
    ex_en = sections.get("Examples", "").strip()

    lines: List[str] = []
    lines.append(f"# {instr}")
    lines.append("")
    lines.append("## 指令示意图")
    lines.append("")
    lines.append(f"![{instr} tile operation](../figures/isa/{instr}.svg)")
    lines.append("")
    lines.append("## 简介")
    lines.append("")
    lines.append(summary_zh.strip() or f"{instr} 指令。")
    if intro_en:
        intro_lines = [ln for ln in intro_en.splitlines() if ln.strip()]
        is_substantive = len(intro_lines) > 2 or any(ln.lstrip().startswith(("-", "*")) for ln in intro_lines) or "```" in intro_en
        if is_substantive:
            lines.append("")
            lines.append(_translate_md_to_zh(intro_en))
    lines.append("")
    lines.append("## 数学语义")
    lines.append("")
    if math_en:
        lines.append(_translate_md_to_zh(math_en))
    else:
        lines.append("该指令的数学语义为指令相关定义。除非另有说明，语义仅在有效区域内定义。")
    lines.append("")
    lines.append("## 汇编语法")
    lines.append("")
    if asm_en:
        lines.append(_translate_md_to_zh(asm_en))
    else:
        lines.append("PTO-AS 形式：参见 `docs/isa/syntax-and-operands/assembly-model.md`。")
    lines.append("")
    lines.append("## C++ 内建接口")
    lines.append("")
    if cpp_en:
        lines.append(_translate_md_to_zh(cpp_en))
    else:
        lines.append("接口声明位于 `include/pto/common/pto_instr.hpp`。")
    lines.append("")
    lines.append("## 约束")
    lines.append("")
    if cons_en:
        lines.append(_translate_md_to_zh(cons_en))
    else:
        lines.append("类型/布局/位置/形状等合法性通常依赖后端实现。若行为依赖具体后端，文档会标注为“实现定义”。")
    lines.append("")
    lines.append("## 示例")
    lines.append("")
    if ex_en:
        lines.append(_translate_md_to_zh(ex_en))
    else:
        lines.append("参见 `docs/isa/` 与 `docs/coding/tutorials/`。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    entries = load_manifest()
    level_formats = load_level_formats(PTO_ISA_LEVEL_TABLE)
    for e in entries:
        instr = str(e["instruction"])
        summary_en = str(e.get("summary_en", f"{instr} instruction."))
        summary_zh = str(e.get("summary_zh", f"{instr} 指令。"))

        en_path = ISA_DIR / f"{instr}.md"
        if not en_path.exists():
            en_path.write_text(template_new_page(instr, summary_en), encoding="utf-8")

        text = _strip_language_links(en_path.read_text(encoding="utf-8", errors="ignore"))
        text = ensure_top_block(instr, text)
        text = ensure_required_sections(instr, text)
        text = ensure_ir_level_syntax(instr, text, level_formats)
        en_path.write_text(text, encoding="utf-8")

        zh_path = ISA_DIR / f"{instr}_zh.md"
        zh_path.write_text(build_zh_page(instr, summary_zh, text), encoding="utf-8")

    print(
        f"Normalized English ISA pages and generated Chinese counterparts for {len(entries)} instructions. "
        f"Loaded PTO-AS level table entries: {len(level_formats)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
