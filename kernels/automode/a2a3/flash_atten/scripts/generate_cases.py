#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""
Generate TFA case configuration and emit a shared header/JSON for host/kernel build.

Usage examples:
    python3 generate_cases.py --cases "128,128,1024,128,256" \
            --cases "128,512,2048,128,256"

    # Override cube-side preload depth (defaults to 4)
    python3 generate_cases.py --qk-preload 6

Each --cases entry format: HEAD_SIZE,S0,S1,CUBE_S0[,TILE_S1]
CUBE_S1 is fixed at 128; TILE_S1 defaults to 256 if omitted.
Defaults replicate the previous hard-coded set if --cases is omitted.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

TILE_S1_DEFAULT = 256
QK_PRELOAD_DEFAULT = 4

DEFAULT_CASES = [
    (128, 128, 1024, 128, TILE_S1_DEFAULT, False),
    (128, 128, 2048, 128, TILE_S1_DEFAULT, False),
    (128, 128, 8192, 128, TILE_S1_DEFAULT, False),
    (128, 512, 1024, 128, TILE_S1_DEFAULT, False),
    (128, 512, 2048, 128, TILE_S1_DEFAULT, False),
    (128, 512, 8192, 128, TILE_S1_DEFAULT, False),
]


def _parse_case_entry(raw: str, qk_preload: int, causal_mask: bool) -> Dict[str, int]:
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    if len(parts) not in (4, 5):
        raise ValueError(f"Expected 4 or 5 comma-separated values (HEAD_SIZE,S0,S1,CUBE_S0[,TILE_S1]), got '{raw}'")
    head, s0, s1, cube_s0 = map(int, parts[:4])
    tile_s1 = int(parts[4]) if len(parts) == 5 else TILE_S1_DEFAULT
    return {
        "head_size": head,
        "s0": s0,
        "s1": s1,
        "cube_s0": cube_s0,
        "cube_s1": 128,
        "tile_s1": tile_s1,
        "qk_preload": qk_preload,
        "causal_mask": int(causal_mask),
    }


def _default_cases(qk_preload: int) -> List[Dict[str, int]]:
    return [
        {
            "head_size": head,
            "s0": s0,
            "s1": s1,
            "cube_s0": cube_s0,
            "cube_s1": 128,
            "tile_s1": tile_s1,
            "qk_preload": qk_preload,
            "causal_mask": int(causal_mask),
        }
        for (head, s0, s1, cube_s0, tile_s1, causal_mask) in DEFAULT_CASES
    ]


def _case_name(case: Dict[str, int]) -> str:
    return f"case_float_H_{case['head_size']}_S0_{case['s0']}_S1_{case['s1']}"


def _normalize_case(case: Dict[str, int]) -> Dict[str, int]:
    if case["qk_preload"] < 1:
        raise ValueError("qk_preload must be >= 1")

    # Ensure cube_s0 does not exceed s0 and divides evenly; otherwise set cube_s0 = s0
    if case["cube_s0"] > case["s0"] or case["s0"] % case["cube_s0"] != 0:
        case["cube_s0"] = case["s0"]

    # Fix cube_s1 to 128 and ensure divisibility
    if case["cube_s1"] != 128:
        case["cube_s1"] = 128
    if case["s1"] % case["cube_s1"] != 0:
        raise ValueError("S1 must be divisible by CUBE_S1 (128)")

    # Ensure TILE_S1 divides S1 and is a multiple of CUBE_S1
    if case["tile_s1"] % case["cube_s1"] != 0:
        raise ValueError("TILE_S1 must be divisible by CUBE_S1 (128)")
    if case["s1"] % case["tile_s1"] != 0:
        raise ValueError("S1 must be divisible by TILE_S1")

    return case


def _render_macro(cases: List[Dict[str, int]]) -> str:
    lines = ["#define TFA_FOR_EACH_CASE(MACRO) \\"]
    for idx, case in enumerate(cases):
        causal_mask = str("true" if bool(case["causal_mask"]) else "false")
        suffix = " \\" if idx + 1 != len(cases) else ""
        line = f"    MACRO({case['s0']}, {case['head_size']}, {case['s1']}, {case['cube_s0']}, {case['cube_s1']}, {case['tile_s1']}, {case['qk_preload']}, {causal_mask}){suffix}"
        lines.append(line)
    return "\n".join(lines)


def _render_header(cases: List[Dict[str, int]]) -> str:
    macro_block = _render_macro(cases)
    array_entries = []
    for case in cases:
        array_entries.append(
            "    {" + ", ".join(
                [
                    str(case["s0"]),
                    str(case["head_size"]),
                    str(case["s1"]),
                    str(case["cube_s0"]),
                    str(case["cube_s1"]),
                    str(case["tile_s1"]),
                    str(case["qk_preload"]),
                    str("true" if bool(case["causal_mask"]) else "false"),
                    f'"{_case_name(case)}"',
                ]
            ) + "}"
        )
    array_block = ",\n".join(array_entries)

    return f"""#pragma once
// Auto-generated by scripts/generate_cases.py. Do not edit manually.
// clang-format off
#include <cstddef>

{macro_block}

struct GeneratedTfaCase {{
    int s0;
    int head_size;
    int s1;
    int cube_s0;
    int cube_s1;
    int tile_s1;
    int qk_preload;
    bool causal_mask;
    const char *name;
}};

static constexpr GeneratedTfaCase kGeneratedTfaCases[] = {{
{array_block}
}};
static constexpr std::size_t kGeneratedTfaCasesCount = sizeof(kGeneratedTfaCases) / sizeof(kGeneratedTfaCases[0]);
// clang-format on
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TFA case header/JSON")
    parser.add_argument(
        "--cases",
        action="append",
        default=None,
        help="Case entry in the format HEAD_SIZE,S0,S1,CUBE_S0[,TILE_S1] (repeat for multiple entries; CUBE_S1 fixed at 128)",
    )
    parser.add_argument(
        "--qk-preload",
        type=int,
        default=QK_PRELOAD_DEFAULT,
        help="qkPreloadNum (cube pipeline preload depth) applied to all generated cases",
    )
    parser.add_argument(
        "--output-header",
        default=str((Path(__file__).resolve().parent.parent / "build" / "generated_cases.h")),
        help="Output header path (default: kernels/fa_performance/build/generated_cases.h)",
    )
    parser.add_argument(
        "--output-json",
        default=str((Path(__file__).resolve().parent.parent / "build" / "generated_cases.json")),
        help="Output JSON path (default: kernels/fa_performance/build/generated_cases.json)",
    )
    parser.add_argument(
        "--causal-mask",
        default=False,
        help="Enable causal mask",
    )
    args = parser.parse_args()

    if args.cases:
        cases = [_normalize_case(_parse_case_entry(entry, args.qk_preload, args.causal_mask)) for entry in args.cases]
    else:
        cases = [_normalize_case(case) for case in _default_cases(args.qk_preload)]

    header_text = _render_header(cases)
    header_path = Path(args.output_header)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(header_text)

    json_payload = [
        {
            "name": _case_name(case),
            **case,
        }
        for case in cases
    ]
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_payload, indent=2))

    print(f"[INFO] Wrote {header_path}")
    print(f"[INFO] Wrote {json_path}")
    print("[INFO] Cases generated:")
    for case in json_payload:
        print(f"  - {case['name']} (H={case['head_size']}, S0={case['s0']}, S1={case['s1']}, CUBE_S0={case['cube_s0']}, CUBE_S1={case['cube_s1']}, TILE_S1={case['tile_s1']}, QK_PRELOAD={case['qk_preload']}, CAUSAL_MASK={case['causal_mask']})")


if __name__ == "__main__":
    main()
