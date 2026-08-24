#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
"""Validate the A5 TSCATTER destination-initialization dependency.

``InitUBBuffer`` initializes the destination with vector stores.  Both indexed
and mask scatter then update that same destination with vector stores, so the
required dependency is VST-to-VST.  A VST-to-VLD barrier adds an unrelated
dependency and changes every mask-scatter kernel's AIV code generation.
"""

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]


def extract_body(source: str, signature: str) -> str:
    match = re.search(signature + r"\s*\{", source)
    if match is None:
        raise AssertionError(f"function signature not found: {signature}")

    begin = source.find("{", match.start())
    depth = 0
    for pos in range(begin, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[begin : pos + 1]
    raise AssertionError(f"unterminated function body: {signature}")


def main() -> None:
    source = (REPO_ROOT / "include" / "pto" / "npu" / "a5" / "TScatter.hpp").read_text()
    init_body = extract_body(source, r"PTO_INTERNAL\s+void\s+InitUBBuffer\([^)]*\)")

    assert "mem_bar(VST_VLD)" not in init_body, (
        "TSCATTER destination initialization must not introduce a VST-to-VLD dependency"
    )
    assert init_body.count("mem_bar(VST_VST)") == 1, (
        "TSCATTER destination initialization must keep exactly one VST-to-VST dependency"
    )
    assert init_body.rfind("vsts(") < init_body.index("mem_bar(VST_VST)"), (
        "the VST-to-VST dependency must follow all destination initialization stores"
    )
    print("A5 TSCATTER initialization barrier validation passed")


if __name__ == "__main__":
    main()
