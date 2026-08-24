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
"""Validate A5 TPUSH/TPOP sub-block dispatch in the legacy overloads.

The overloads without an explicit ``subBlockId`` must not evaluate
``get_subblockid()`` for ``TILE_NO_SPLIT``.  Reading the vector sub-block ID on
that path changes both AIC and AIV code generation and can deadlock C2V/V2C
pipes.  Split calls keep the historical implicit-ID behavior, while the
three-argument overloads continue to forward the caller-provided ID.
"""

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]


def extract_body(source: str, signature: str) -> str:
    """Return a function body, including braces, using balanced-brace parsing."""
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


def compact(source: str) -> str:
    return re.sub(r"\s+", "", source)


def validate_operation(filename: str, operation: str, endpoint: str) -> None:
    source = (REPO_ROOT / "include" / "pto" / "npu" / "a5" / filename).read_text()

    implicit = extract_body(
        source,
        rf"PTO_INTERNAL\s+[^\n]+\s+{operation}_IMPL\(Pipe&\s+pipe,\s+Tile\w+&\s+tile\)",
    )
    normalized = compact(implicit)
    no_split = "ifconstexpr(Split==TileSplitAxis::TILE_NO_SPLIT){"
    split_else = "}else{"
    assert no_split in normalized, f"{operation}: missing compile-time TILE_NO_SPLIT branch"

    no_split_body, split_body = normalized.split(no_split, 1)[1].split(split_else, 1)
    assert "get_subblockid()" not in no_split_body, f"{operation}: TILE_NO_SPLIT evaluates get_subblockid()"
    assert f"{endpoint}<Tile" in no_split_body and ",tile,0);" in no_split_body, (
        f"{operation}: TILE_NO_SPLIT must pass literal subBlockId 0"
    )
    assert "get_subblockid()" in split_body, f"{operation}: split path lost implicit subBlockId"
    assert implicit.count("get_subblockid()") == 1, f"{operation}: unexpected implicit subBlockId reads"

    explicit = extract_body(
        source,
        rf"PTO_INTERNAL\s+[^\n]+\s+{operation}_IMPL\(Pipe&\s+pipe,\s+Tile\w+&\s+tile,\s+int32_t\s+subBlockId\)",
    )
    assert "get_subblockid()" not in explicit, f"{operation}: explicit overload must use its argument"
    assert ", tile, subBlockId)" in explicit, f"{operation}: explicit overload does not forward subBlockId"


def main() -> None:
    validate_operation("TPush.hpp", "TPUSH", "pipe.prod.templatepush")
    validate_operation("TPop.hpp", "TPOP", "pipe.cons.templatepop")
    print("A5 TPUSH/TPOP sub-block dispatch validation passed")


if __name__ == "__main__":
    main()
