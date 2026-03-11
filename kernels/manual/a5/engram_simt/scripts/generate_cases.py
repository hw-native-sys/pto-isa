#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import argparse
import json
from pathlib import Path
from typing import List, Dict

DEFAULT_CONFIGS = [(128, 1, 65536, "T64K"), (256, 4, 65536, "T64K"), (512, 1, 65536, "T64K"), (1024, 1, 65536, "T64K")]

PERF_DIMS = [128, 256, 512, 1024]
PERF_BLOCKS = [1, 4, 16, 64]
PERF_TABLES = [(65536, "T64K"), (262144, "T256K"), (1048576, "T1M")]
PERF_PATTERNS = ["RAND", "SEQ", "SAME", "STRIDE"]


def _default_cases() -> List[Dict]:
    return [{"emb_dim": e, "block_size": b, "table_rows": t, "tag": tag} for (e, b, t, tag) in DEFAULT_CONFIGS]


def _perf_cases() -> List[Dict]:
    cases = []
    for e in PERF_DIMS:
        for b in PERF_BLOCKS:
            for t, ttag in PERF_TABLES:
                for p in PERF_PATTERNS:
                    cases.append({"emb_dim": e, "block_size": b, "table_rows": t, "tag": f"{ttag}_{p}"})
    return cases


def _case_name(case: Dict) -> str:
    return f"E{case['emb_dim']}_B{case['block_size']}_{case['tag']}"


def _render_macro(cases: List[Dict]) -> str:
    lines = ["#define ENGRAM_FOR_EACH_CASE(MACRO) \\"]
    for idx, case in enumerate(cases):
        suffix = " \\" if idx + 1 != len(cases) else ""
        line = f"    MACRO({case['emb_dim']}, {case['block_size']}, {case['table_rows']}, {case['tag']}){suffix}"
        lines.append(line)
    return "\n".join(lines)


def _render_header(cases: List[Dict]) -> str:
    macro_block = _render_macro(cases)
    array_entries = []
    for case in cases:
        name = _case_name(case)
        array_entries.append(
            f'    {{{case["emb_dim"]}, {case["block_size"]}, {case["table_rows"]}, "{case["tag"]}", "{name}"}}'
        )
    array_block = ",\n".join(array_entries)

    return f"""#pragma once

// clang-format off
#include <cstddef>

{macro_block}

struct GeneratedEngramCase {{
    int embDim;
    int blockSize;
    int tableRows;
    const char *tag;
    const char *name;
}};

static constexpr GeneratedEngramCase kGeneratedEngramCases[] = {{
{array_block}
}};
static constexpr std::size_t kGeneratedEngramCasesCount = sizeof(kGeneratedEngramCases) / sizeof(kGeneratedEngramCases[0]);
// clang-format on
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Engram SIMT case header/JSON")
    parser.add_argument(
        "--perf-analysis",
        action="store_true",
        help="Generate all perf-analysis configs (4D x 4B x 3T x 4P = 192 configs)",
    )
    parser.add_argument(
        "--cases",
        action="append",
        default=None,
        help="Case entry: EMB_DIM,BLOCK_SIZE,TABLE_ROWS,TAG (repeat for multiple)",
    )
    parser.add_argument(
        "--output-header",
        default=str(Path(__file__).resolve().parent.parent / "build" / "generated_cases.h"),
        help="Output header path",
    )
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parent.parent / "build" / "generated_cases.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    if args.cases:
        cases = []
        for raw in args.cases:
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) != 4:
                raise ValueError(f"Expected EMB_DIM,BLOCK_SIZE,TABLE_ROWS,TAG, got '{raw}'")
            cases.append(
                {"emb_dim": int(parts[0]), "block_size": int(parts[1]), "table_rows": int(parts[2]), "tag": parts[3]}
            )
    elif args.perf_analysis:
        cases = _perf_cases()
    else:
        cases = _default_cases()

    header_text = _render_header(cases)
    header_path = Path(args.output_header)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(header_text)

    json_payload = [{"name": _case_name(c), **c} for c in cases]
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_payload, indent=2))

    print(f"[INFO] Wrote {header_path}")
    print(f"[INFO] Wrote {json_path}")
    print(f"[INFO] {len(cases)} case configs generated:")
    for c in cases:
        print(f"  - {_case_name(c)} (D={c['emb_dim']}, B={c['block_size']}, T={c['table_rows']})")


if __name__ == "__main__":
    main()
