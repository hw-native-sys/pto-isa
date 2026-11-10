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


def expect_rejection(command, source, diagnostic):
    result = subprocess.run(command + ["-fsyntax-only", str(source)], capture_output=True, text=True)
    if result.returncode == 0 or diagnostic not in result.stderr:
        print(result.stderr)
        raise RuntimeError(f"{source.name}: expected diagnostic {diagnostic!r}")


def check_multi_tu(command, output_dir, *, device=False):
    # The public header must link in a normal application, even at -O0.
    sources = [output_dir / "odr_a.cpp", output_dir / "odr_b.cpp"]
    sources[0].write_text(
        "#include <pto/pto-inst.hpp>\n"
        "const char* layoutName() { return pto::layout_type_to_str(pto::LayoutEnum::kRowMajor); }\n"
    )
    sources[1].write_text(
        "#include <pto/pto-inst.hpp>\n"
        "extern const char* layoutName();\n"
        "int main() { return layoutName()[0] != pto::layout_type_to_str(pto::LayoutEnum::kRowMajor)[0]; }\n"
    )
    if device:
        # Relocatable linking needs no runtime or hardware and catches strong
        # duplicate header symbols across independently compiled device objects.
        objects = [source.with_suffix(".o") for source in sources]
        for source, obj in zip(sources, objects):
            run(command + ["-O0", "-c", str(source), "-o", str(obj)])
        run(command + ["-nostdlib", "-r", *map(str, objects), "-o", str(output_dir / "linx_combined.o")])
    else:
        binary = (output_dir / "linx_multi_tu").resolve()
        run(command + ["-O0", *map(str, sources), "-o", str(binary)])
        run([str(binary)])


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
            command.extend(
                ["--sysroot=" + str(sysroot), "-nostdinc++", "-isystem", str(Path(sysroot) / "usr/include/c++/v1")]
            )
        output = args.output_dir / "linx_device.o"
        run(command + ["-c", str(root / "tests/compile/linx_device.cpp"), "-o", str(output)])
        if not output.is_file() or not output.stat().st_size:
            raise RuntimeError("Linx compiler did not emit an object")
        expect_rejection(
            command + ["-DPTO_LINX_TEST_REJECT_TRANSPOSE"], root / "tests/compile/linx_device.cpp", "TTRANS is retired"
        )
        for operation in ("TLOAD", "TSTORE"):
            for stride in (0, 2, -2, -1):
                source = args.output_dir / f"{operation.lower()}_stride_{stride}.cpp"
                source.write_text(
                    "#include <pto/pto-inst.hpp>\nusing namespace pto;\n"
                    "using T = Tile<TileType::Vec,float,4,8>;\n"
                    f"using S = Stride<64,64,64,16,{stride}>;\n"
                    "using G = GlobalTensor<float,Shape<1,1,1,4,8>,S>;\n"
                    "void test(float* p, int stride) { T tile; G gm(p, {}, "
                    + ("S(stride)" if stride == -1 else "S{}")
                    + "); "
                    + ("TLOAD(tile, gm);" if operation == "TLOAD" else "TSTORE(gm, tile);")
                    + " }\n"
                )
                expect_rejection(command, source, f"Linx device {operation} requires a static GM column stride of 1")
        check_multi_tu(command, args.output_dir, device=True)
        print(f"Linx device object/relocatable-link tests and 9 capability rejections passed: {output}")
        return 0

    command.extend(["-DPTO_LINX_CPU_SIM", "-D__cpu_sim__"])
    binary = (args.output_dir / "linx_cpu_sim").resolve()
    run(command + [str(root / "tests/compile/linx_target.cpp"), "-o", str(binary)])
    run([str(binary)])
    conversion = (args.output_dir / "linx_conversion").resolve()
    run(command + [str(root / "tests/compile/linx_conversion.cpp"), "-o", str(conversion)])
    for direction in ("to-nz", "to-col"):
        run([str(conversion), direction], timeout=30)
    check_multi_tu(command, args.output_dir)
    # Every failure must match our contract diagnostic, not an unrelated syntax error.
    cases = [
        ("conflicting_device", ["-D__linx"], "", "mutually exclusive"),
        ("conflicting_cann", ["-D__CCE_AICORE__"], "", "cannot be combined"),
        ("conflicting_arch", ["-D__NPU_ARCH__=2201"], "", "exactly one NPU target"),
        ("missing_sim", ["-U__cpu_sim__"], "", "requires the Linx __cpu_sim__ backend"),
        ("pipe_event", [], "T a,b,c; int e=0; pto::TADD(c,a,b,e);", "does not support PTO pipe events"),
        ("cube_layout", [], "pto::VecTileM32<float,32,8> a,b,c; pto::TADD(c,a,b);", "unboxed RowMajor/ColMajor"),
        (
            "dynamic_valid",
            [],
            "pto::Tile<pto::TileType::Vec,float,4,8,pto::BLayout::RowMajor,-1,-1> a(3,5),b(3,5),c(3,5); pto::TADD(c,a,b);",
            "requires static valid shapes",
        ),
        (
            "oversized_valid",
            [],
            "pto::Tile<pto::TileType::Vec,float,4,8,pto::BLayout::RowMajor,5,8> a,b,c; pto::TADD(c,a,b);",
            "valid shape exceeds physical shape",
        ),
        ("precision", [], "T a,b,c; pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(c,a,b);", "default math algorithm"),
    ]
    for name, flags, body, diagnostic in cases:
        source = args.output_dir / (name + ".cpp")
        source.write_text(
            "#include <pto/pto-inst.hpp>\n"
            "using T = pto::Tile<pto::TileType::Vec,float,4,8>;\n"
            "void test() {" + body + "}\n"
        )
        expect_rejection(command + flags, source, diagnostic)
    # Reverse public-header include order is also supported.
    source = args.output_dir / "include_order.cpp"
    source.write_text(
        "#include <pto/pto-inst.hpp>\n#include <pto/common/type.hpp>\n#include <pto/common/pto_tile.hpp>\n"
    )
    run(command + ["-fsyntax-only", str(source)])
    print(
        f"Linx CPU-sim: numerical/internal-conversion/multi-TU tests, {len(cases)} rejection tests and include-order check passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
