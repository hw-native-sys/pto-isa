#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run Linx CPU correctness/contract tests or compile with a real Linx toolchain."""

import argparse
import os
from pathlib import Path
import subprocess


def run(command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--compiler")
    parser.add_argument("--sysroot", type=Path)
    parser.add_argument("--compiler-arg", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compiler = args.compiler or (os.environ.get("LINX_CXX") if args.mode == "npu" else "c++")
    if not compiler:
        parser.error("NPU compilation requires --compiler or LINX_CXX pointing to the Linx compiler")
    command = [compiler, "-std=c++20", "-O2", "-I" + str(root / "include")]
    command.extend(args.compiler_arg)
    if args.mode == "npu":
        command.extend(["--target=linx64v5-unknown-linux-musl", "-mlxbc", "-fenable-matrix", "-D__linx"])
        sysroot = args.sysroot or os.environ.get("LINX_SYSROOT")
        if sysroot:
            command.extend(["--sysroot=" + str(sysroot), "-nostdinc++", "-isystem", str(Path(sysroot) / "usr/include/c++/v1")])
        output = args.output_dir / "linx_device.o"
        run(command + ["-c", str(root / "tests/compile/linx_device.cpp"), "-o", str(output)])
        if not output.is_file() or not output.stat().st_size:
            raise RuntimeError("Linx compiler did not emit an object")
        rejection = subprocess.run(
            command + ["-DPTO_LINX_TEST_REJECT_TRANSPOSE", "-fsyntax-only", str(root / "tests/compile/linx_device.cpp")],
            capture_output=True, text=True,
        )
        if rejection.returncode == 0 or "TTRANS is retired" not in rejection.stderr:
            print(rejection.stderr)
            raise RuntimeError("expected the native Linx TTRANS capability diagnostic")
        print(f"Linx device compilation and capability rejection passed: {output}")
        return 0

    command.extend(["-DPTO_LINX_CPU_SIM", "-D__cpu_sim__"])
    binary = (args.output_dir / "linx_cpu_sim").resolve()
    run(command + [str(root / "tests/compile/linx_target.cpp"), "-o", str(binary)])
    run([str(binary)])
    # Every failure must match our contract diagnostic, not an unrelated syntax error.
    cases = [
        ("conflicting_device", ["-D__linx"], "", "mutually exclusive"),
        ("conflicting_cann", ["-D__CCE_AICORE__"], "", "cannot be combined"),
        ("conflicting_arch", ["-D__NPU_ARCH__=2201"], "", "exactly one NPU target"),
        ("missing_sim", ["-U__cpu_sim__"], "", "requires the Linx __cpu_sim__ backend"),
        ("pipe_event", [], "T a,b,c; int e=0; pto::TADD(c,a,b,e);", "does not support PTO pipe events"),
        ("cube_layout", [], "pto::VecTileM32<float,32,8> a,b,c; pto::TADD(c,a,b);", "unboxed RowMajor/ColMajor"),
        ("precision", [], "T a,b,c; pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(c,a,b);", "default math algorithm"),
    ]
    for name, flags, body, diagnostic in cases:
        source = args.output_dir / (name + ".cpp")
        source.write_text(
            "#include <pto/pto-inst.hpp>\n"
            "using T = pto::Tile<pto::TileType::Vec,float,4,8>;\n"
            "void test() {" + body + "}\n"
        )
        result = subprocess.run(command + flags + ["-fsyntax-only", str(source)], capture_output=True, text=True)
        if result.returncode == 0 or diagnostic not in result.stderr:
            print(result.stderr)
            raise RuntimeError(f"{name}: expected diagnostic {diagnostic!r}")
    # Reverse public-header include order is also supported.
    source = args.output_dir / "include_order.cpp"
    source.write_text("#include <pto/pto-inst.hpp>\n#include <pto/common/type.hpp>\n#include <pto/common/pto_tile.hpp>\n")
    run(command + ["-fsyntax-only", str(source)])
    print(f"Linx CPU-sim: numerical tests, {len(cases)} rejection tests and include-order check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
