#!/usr/bin/env python3
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


from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

_BF16_PROBE_SOURCE = """#include <stdfloat>
int main() {
    std::bfloat16_t x = std::bfloat16_t(1.0f);
    static_assert(sizeof(std::bfloat16_t) == 2);
    return static_cast<float>(x) > 0.0f ? 0 : 1;
}
"""


def _iter_bfloat16_candidates(explicit: Optional[str] = None) -> Iterable[str]:
    seen: set[str] = set()

    def emit(candidate: Optional[str]) -> Iterable[str]:
        if not candidate:
            return
        resolved = shutil.which(candidate) or candidate
        if resolved in seen:
            return
        seen.add(resolved)
        yield resolved

    for item in emit(explicit):
        yield item
    for item in emit(os.environ.get("CXX")):
        yield item
    for item in emit("g++-15"):
        yield item
    for item in emit("g++-14"):
        yield item
    for item in emit("g++"):
        yield item
    for item in emit("clang++-19"):
        yield item
    for item in emit("clang++-18"):
        yield item
    for item in emit("clang++-17"):
        yield item
    for item in emit("clang++"):
        yield item
    for item in emit("c++"):
        yield item


def compiler_supports_bfloat16(cxx: str) -> bool:
    compiler = shutil.which(cxx) or cxx
    if not Path(compiler).exists():
        return False

    with tempfile.TemporaryDirectory(prefix="pto_bf16_probe_") as tmpdir:
        src = Path(tmpdir) / "probe.cpp"
        out = Path(tmpdir) / "probe.out"
        src.write_text(_BF16_PROBE_SOURCE, encoding="utf-8")
        result = subprocess.run(
            [compiler, "-std=c++23", str(src), "-o", str(out)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return result.returncode == 0


def detect_bfloat16_cxx(explicit: Optional[str] = None) -> str:
    for candidate in _iter_bfloat16_candidates(explicit):
        if compiler_supports_bfloat16(candidate):
            return candidate
    raise RuntimeError(
        "Could not find a compiler with std::bfloat16_t support. "
        "Pass --cxx/--compiler explicitly, or install a C++23 compiler such as g++-14+."
    )


def derive_cc_from_cxx(cxx: Optional[str]) -> Optional[str]:
    if not cxx:
        return None
    compiler = shutil.which(cxx) or cxx
    name = Path(compiler).name
    directory = str(Path(compiler).parent)

    gcc_match = re.fullmatch(r"g\+\+(-\d+)?", name)
    if gcc_match:
        cc_name = f"gcc{gcc_match.group(1) or ''}"
        cc_path = shutil.which(cc_name)
        if cc_path:
            return cc_path
        candidate = Path(directory) / cc_name
        return str(candidate) if candidate.exists() else None

    clang_match = re.fullmatch(r"clang\+\+(-\d+)?", name)
    if clang_match:
        cc_name = f"clang{clang_match.group(1) or ''}"
        cc_path = shutil.which(cc_name)
        if cc_path:
            return cc_path
        candidate = Path(directory) / cc_name
        return str(candidate) if candidate.exists() else None

    if name == "c++":
        return shutil.which("cc") or None

    return None
