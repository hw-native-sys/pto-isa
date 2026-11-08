#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compile the Linx target umbrella through the Linx CPU-sim backend."""

import argparse
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "linx_target.o"
    command = [
        args.compiler,
        "-std=c++20",
        "-DPTO_LINX_CPU_SIM",
        "-D__cpu_sim__",
        "-I" + str(root / "include"),
        "-c",
        str(root / "tests/compile/linx_target.cpp"),
        "-o",
        str(output),
    ]
    result = subprocess.run(command, cwd=root, check=False)
    if result.returncode != 0:
        return result.returncode
    if not output.is_file() or output.stat().st_size == 0:
        print(f"compiler did not produce a non-empty object: {output}")
        return 1
    print(f"Linx CPU-sim target compile passed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
