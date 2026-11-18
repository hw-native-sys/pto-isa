#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Execute identical Linx kernels on CPU and gfrun against independent goldens.

Both native header routes are tested. A matching failure in the original route
remains a failure, never a pass. Output sentinels and canaries detect skipped
stores and writes beyond the expected output.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "tests/cpu/linx"
FORMATS = {"float": "f", "uint32_t": "I", "int32_t": "i", "uint8_t": "B", "int8_t": "b", "__half": "e", "uint16_t": "H"}


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def product32(values):
    result = 1.0
    for value in values:
        result = f32(result * value)
    return result


def declaration(ctype, name, values):
    if ctype.startswith("__fp"):
        return (
            declaration("uint8_t", name + "_bytes", values)
            + f"auto* const {name} = reinterpret_cast<{ctype}*>({name}_bytes);\n"
        )

    def literal(value):
        if ctype.startswith("__fp"):
            return "{" + str(value) + "}"
        if ctype == "uint32_t":
            return str(value & 0xFFFFFFFF) + "u"
        return repr(value)

    return f"alignas(128) {ctype} {name}[] = {{" + ",".join(map(literal, values)) + "};\n"


@dataclass
class Case:
    name: str
    source: str
    inputs: str
    call: str
    expected: list
    dtype: str = "float"
    pes: int = 1
    instructions: tuple = ()
    kernel: str = ""
    profile: str = ""


def cases():
    result = []
    x = [((i * 7) % 16 + 1) / 4 for i in range(128)]
    y = [(i % 8 + 1) / 4 for i in range(128)]
    inputs = declaration("float", "x", x) + declaration("float", "y", y)
    binary = {
        "TADD": lambda a, b: a + b,
        "TSUB": lambda a, b: a - b,
        "TMUL": lambda a, b: a * b,
        "TDIV": lambda a, b: a / b,
        "TREM": math.fmod,
        "TMAX": max,
        "TMIN": min,
        "TEXPDIF": lambda a, b: math.exp(a - b),
    }
    for op, reference in binary.items():
        result.append(Case(op, "vector", inputs, f"k_{op}(x,y,result);", [f32(reference(a, b)) for a, b in zip(x, y)]))
        if op != "TEXPDIF":
            result.append(Case(op + "S", "vector", inputs, f"k_{op}S(x,result);", [f32(reference(a, 2)) for a in x]))
    unary = {
        "TABS": abs,
        "TNEG": lambda a: -a,
        "TEXP": math.exp,
        "TLOG": lambda a: f32(f32(math.log2(a)) * f32(math.log(2))),
        "TRECIP": lambda a: 1 / a,
        "TSQRT": math.sqrt,
        "TRSQRT": lambda a: 1 / f32(math.sqrt(a)),
        "TRELU": lambda a: max(a, 0),
    }
    for op, reference in unary.items():
        result.append(Case(op, "vector", inputs, f"k_{op}(x,result);", [f32(reference(a)) for a in x]))
    result.append(Case("TFMA", "vector", inputs, "k_fma(x,y,result);", [f32(a * b + a) for a, b in zip(x, y)]))
    u = [0xDEAD0000 + i for i in range(128)]
    v = [i % 37 for i in range(128)]
    uints = declaration("uint32_t", "x", u) + declaration("uint32_t", "y", v)
    bitwise = {
        "TAND": lambda a, b: a & b,
        "TOR": lambda a, b: a | b,
        "TXOR": lambda a, b: a ^ b,
        "TSHL": lambda a, b: (a << (b % 32)) & 0xFFFFFFFF,
        "TSHR": lambda a, b: a >> (b % 32),
    }
    for op, reference in bitwise.items():
        result.append(
            Case(op, "vector", uints, f"k_{op}(x,y,result);", [reference(a, b) for a, b in zip(u, v)], "uint32_t")
        )
    for axis in ("ROW", "COL"):
        groups = [x[r * 32 : (r + 1) * 32] for r in range(4)] if axis == "ROW" else [x[c::32] for c in range(32)]
        for op, reference in {
            "SUM": sum,
            "PROD": product32,
            "MAX": max,
            "MIN": min,
            "ARGMAX": lambda g: g.index(max(g)),
            "ARGMIN": lambda g: g.index(min(g)),
        }.items():
            name = "T" + axis + op
            dtype = "uint32_t" if "ARG" in op else "float"
            result.append(Case(name, "vector", inputs, f"k_{name}(x,result);", [reference(g) for g in groups], dtype))
    golden = [a + sum(x[(i // 32) * 32 : (i // 32 + 1) * 32]) for i, a in enumerate(x)]
    golden += [a * sum(x[i % 32 :: 32]) for i, a in enumerate(x)]
    result.append(Case("broadcast", "vector", inputs, "k_broadcast(x,result);", golden))
    result.append(
        Case(
            "regions",
            "vector",
            declaration("int32_t", "x", list(range(256))) + declaration("int32_t", "y", [-i for i in range(32)]),
            "k_regions(x,y,result);",
            [i - i // 8 for i in range(256)],
            "int32_t",
        )
    )
    operations = [
        lambda a, b: b,
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
        max,
        min,
        lambda a, b: math.exp(a - b),
    ]
    for column in (False, True):
        for op, reference in enumerate(operations):
            result.append(
                Case(
                    f"broadcast_{'col' if column else 'row'}_{op}",
                    "vector",
                    inputs,
                    f"k_broadcast_ops({op}, {str(column).lower()}, x, y, result);",
                    [f32(reference(a, y[i % 32 if column else i // 32])) for i, a in enumerate(x)],
                )
            )
    result.append(
        Case("compare_move", "vector", inputs, "k_compare_move(x,y,result);", [min(a, b) for a, b in zip(x, y)])
    )
    data = list(range(512))
    inputs = declaration("float", "x", data)
    result.append(
        Case(
            "local_assembly",
            "storage",
            inputs,
            "localAssembly(x,result);",
            [data[(c // 4) * 128 + r * 4 + c % 4] for r in range(32) for c in range(8)],
        )
    )
    result.append(
        Case(
            "shared_load",
            "storage",
            inputs,
            "sharedLoad(x,result);",
            [data[r * 32 + c] for pe in range(4) for r in range(4) for c in range(32)],
            pes=4,
        )
    )
    result.append(
        Case("shared_single_producer", "storage", inputs, "sharedSingleProducer(x,result);", data[384:] * 2, pes=4)
    )
    words = [0xAABB0000 + i for i in range(32)]
    golden = (
        [0x55440000 | i for i in range(32)]
        + [0xBB00] * 32
        + [words[max(0, i - 1)] for i in range(32)]
        + [0x66666666] * 32
    )
    result.append(
        Case("rearrange", "storage", declaration("uint32_t", "x", words), "rearrange(x,result);", golden, "uint32_t")
    )
    result.append(
        Case("gpr_bits", "storage", "", "makeBits(result);", [13 if i % 4 == 0 else 0 for i in range(128)], "uint8_t")
    )
    image = [i % 13 - 6 for i in range(192)]
    weight = [i % 7 - 3 for i in range(24)]
    inputs = declaration("float", "x", image) + declaration("float", "w", weight)
    result.append(
        Case(
            "image_nhwc",
            "transform",
            inputs,
            "imageND(x,result);",
            [image[r * 3 + c] if c < 3 else 0 for r in range(16) for c in range(8)],
            pes=4,
        )
    )
    result.append(
        Case(
            "image_nchw",
            "transform",
            declaration("float", "x", [image[i * 3 + c] for c in range(3) for i in range(16)]),
            "imageDN(x,result);",
            [image[r * 3 + c] if c < 3 else 0 for r in range(16) for c in range(8)],
        )
    )
    result.append(
        Case(
            "weights",
            "transform",
            inputs,
            "weights(w,result);",
            [weight[r * 3 + c] if c < 3 else 0 for r in range(8) for c in range(8)],
        )
    )
    result.append(
        Case(
            "matrix_slots",
            "transform",
            inputs,
            "slots(x,w,result);",
            [sum(image[r * 3 + k] * weight[n * 3 + k] for k in range(3)) for r in range(64) for n in range(8)],
            pes=4,
        )
    )
    a = [(i % 11 - 5) / 8 for i in range(512)]
    b = [(i % 7 - 3) / 4 for i in range(64)]
    inputs = declaration("float", "x", a) + declaration("float", "y", b)
    result.append(
        Case(
            "matrix_acc_scale",
            "matrix",
            inputs,
            "matrixAccScale(x,y,result);",
            [2 + sum(a[r * 8 + k] * b[k * 8 + n] for k in range(8)) for r in range(32) for n in range(8)],
        )
    )
    result.append(
        Case(
            "matrix_shared",
            "matrix",
            inputs,
            "matrixShared(x,y,result);",
            [sum(a[r * 8 + k] * b[n * 8 + k] for k in range(8)) for r in range(64) for n in range(8)],
            pes=4,
        )
    )
    mx = declaration("__fp8_e4m3", "x", [0x3C] * 512) + declaration("__fp8_e5m2", "y", [0x40] * 256)
    result.append(Case("matrix_mx", "matrix", mx, "matrixMX(x,y,result);", [768] * 128 + [770] * 128 + [771] * 128))
    hif = declaration("__fp4_hif4x2", "x", [0x44] * 512) + declaration("__fp4_hif4x2", "y", [0x44] * 256)
    result.append(Case("matrix_hif4", "matrix", hif, "matrixHif4(x,y,result);", [256] * 128))
    half = declaration("__half", "x", [1] * 32) + declaration("__half", "y", [2] * 256)
    result.append(
        Case(
            "gemv", "matrix", half, "gemv(x,y,result);", [64] * 8 + [66] * 8 + [67] * 8 + [64] * 8 + [66] * 8 + [67] * 8
        )
    )
    golden = [sum(a[r * 8 + k] * b[k * 8 + n] for k in range(8)) for r in range(32) for n in range(8)]
    golden = [value if value >= 0 else -0.0 for value in golden]
    row = [max(golden[r * 8 : (r + 1) * 8]) for r in range(32)]
    result.append(
        Case(
            "matrix_aux", "matrix", inputs, "matrixAux(x,y,result,result+256,result+288);", golden + row + row, "__half"
        )
    )
    operands = [(i * 7 + 5) % 17 for i in range(32)]
    memory = [(i * 11 + 3) % 29 for i in range(32)]
    inputs = (
        declaration("uint32_t", "memory", memory)
        + declaration("uint32_t", "value", operands)
        + declaration("int32_t", "offsets", [i * 4 for i in range(32)])
    )
    atomic = [
        lambda a, b: b,
        max,
        min,
        lambda a, b: a + b,
        lambda a, b: 0 if a >= b else a + 1,
        lambda a, b: b if a == 0 or a > b else a - 1,
        lambda a, b: a & b,
        lambda a, b: a | b,
        lambda a, b: a ^ b,
    ]
    for op, reference in enumerate(atomic):
        result.append(
            Case(
                f"gather_atomic_{op}",
                "indexed",
                inputs,
                f"gatherAtomic({op},memory,offsets,value,result); for(unsigned i=0;i<32;++i) result[32+i]=memory[i];",
                memory + [reference(a, b) for a, b in zip(memory, operands)],
                "uint32_t",
            )
        )
    for op in range(9):
        golden = [atomic[op + 1](a, b) if op < 8 else a + 1 for a, b in zip(memory, operands)]
        result.append(
            Case(
                f"scatter_atomic_{op}",
                "indexed",
                inputs,
                f"scatterAtomic({op},memory,offsets,value); for(unsigned i=0;i<32;++i) result[i]=memory[i];",
                golden,
                "uint32_t",
            )
        )
        if op < 8:
            result.append(
                Case(
                    f"store_atomic_{op}",
                    "indexed",
                    inputs,
                    f"atomicStore({op},memory,value); for(unsigned i=0;i<32;++i) result[i]=memory[i];",
                    golden,
                    "uint32_t",
                )
            )
    expected = [value if i % 2 else 999 for i, value in enumerate(memory)]
    result.append(
        Case(
            "compare_exchange",
            "indexed",
            inputs + declaration("uint32_t", "expected", expected),
            "compareExchange(memory,offsets,expected,value,result); for(unsigned i=0;i<32;++i) result[32+i]=memory[i];",
            memory + [operands[i] if i % 2 else value for i, value in enumerate(memory)],
            "uint32_t",
        )
    )
    inputs = declaration("uint32_t", "memory", memory) + declaration(
        "int32_t", "offsets", [(i - 16) * 4 for i in range(32)]
    )
    result.append(
        Case(
            "indexed_mask",
            "indexed",
            inputs + declaration("uint8_t", "mask_bytes", [int(i >= 16) for i in memory]),
            "regularIndexed(memory+16,offsets,mask_bytes,result);",
            memory + [i if i >= 16 else 0 for i in memory] + [i if i >= 16 else 93 for i in memory],
            "uint32_t",
        )
    )
    inputs = (
        declaration("uint32_t", "memory", [1] * 128)
        + declaration("uint32_t", "value", list(range(128)))
        + declaration("int64_t", "offsets", [i * 4 for i in range(128)])
    )
    result.append(
        Case(
            "predicate_atomic",
            "indexed",
            inputs,
            "predicateAtomic(memory,offsets,value,result); for(unsigned i=0;i<128;++i) result[128+i]=memory[i];",
            [i if 32 <= i < 96 else 99 for i in range(128)] + [i + 1 if 32 <= i < 96 else 1 for i in range(128)],
            "uint32_t",
        )
    )
    result.append(
        Case(
            "predicate_regions",
            "indexed",
            declaration("uint32_t", "x", list(range(256))),
            "predicateRegions(x,result);",
            [i if i >= 128 else 999 for i in range(256)],
            "uint32_t",
        )
    )
    result.append(
        Case(
            "select_bits",
            "boundaries",
            declaration("uint32_t", "x", list(range(128))),
            "selectAndBits(x,result);",
            [i if i >= 64 else 999 for i in range(128)]
            + [(~i) & 0xFFFFFFFF for i in range(128)]
            + [(((i & 15) | 0x100) ^ 0x10) * 2 for i in range(128)],
            "uint32_t",
        )
    )
    indices = [(i // 32 + i % 2) % 4 for i in range(128)]
    scattered = [0] * 128
    for i, idx in enumerate(indices):
        scattered[idx * 32 + i % 32] = i
    result.append(
        Case(
            "tile_indexed",
            "boundaries",
            declaration("float", "x", list(range(128))) + declaration("uint16_t", "offsets", indices),
            "indexedRows(x,offsets,result);",
            [idx * 32 + i % 32 for i, idx in enumerate(indices)] + scattered,
        )
    )
    result.append(
        Case(
            "dynamic_shape",
            "boundaries",
            declaration("float", "x", list(range(128))),
            "dynamic(x,result,3,17);",
            [i + 3 if i // 32 < 3 and i % 32 < 17 else 93 for i in range(128)],
        )
    )
    result.append(
        Case(
            "core_gmov",
            "native_api",
            declaration("float", "x", list(range(128))),
            "collective(x,result);",
            [((p + 1) % 4) * 32 + i for p in range(4) for i in range(32)],
            pes=4,
        )
    )
    result.append(
        Case(
            "fp16_convert",
            "native_api",
            declaration("float", "x", [i / 4 - 3 for i in range(32)]),
            "conversion(x,result);",
            [i / 4 - 3 for i in range(32)],
            "__half",
        )
    )
    rounding = [
        [2, 4, -2, -4, 2, -2],
        [2, 3, -2, -3, 2, -2],
        [2, 3, -3, -4, 2, -3],
        [3, 4, -2, -3, 3, -2],
        [3, 4, -3, -4, 2, -2],
        [3, 3, -3, -3, 3, -3],
        [3, 4, -2, -3, 2, -2],
    ]
    result.append(
        Case(
            "scalar_rounding",
            "native_api",
            declaration("float", "x", [2.5, 3.5, -2.5, -3.5, 2.2, -2.2]),
            "scalarRounding(x,result);",
            sum(rounding, []),
            "int32_t",
        )
    )
    result.append(Case("packed_bf16", "native_api", "", "packedBf16(result);", [0x40003F80, 0x40804000], "uint32_t"))
    for case in list(result):
        if case.name.startswith(("gather_atomic_", "scatter_atomic_")):
            result.append(
                Case(
                    case.name + "_unbounded",
                    case.source,
                    case.inputs,
                    case.call.replace("gatherAtomic(", "gatherAtomic<false>(").replace(
                        "scatterAtomic(", "scatterAtomic<false>("
                    ),
                    case.expected,
                    case.dtype,
                )
            )
    return result


def program(case, upstream=False):
    # Reuse the acceptance kernels; changing backend is compile-only.
    kernel = case.kernel or (SOURCES / (case.source + ".cpp")).read_text()
    # Internal linkage avoids codegen of unrelated kernels with compiler limitations.
    kernel = re.sub(r"^void (\w+)\(", r"static void \1(", kernel, flags=re.M)
    if upstream:
        kernel = kernel.replace("<pto/pto-inst.hpp>", "<common/pto_tileop.hpp>")
    text = "#define LINX_TEST_KERNEL_ONLY\n" + kernel + "\n" + case.inputs
    text += declaration(case.dtype, "result", [93] * (len(case.expected) + 16))
    text += """
#ifdef __CPU_SIM
#include <cstdio>
#endif
int main() {
"""
    if case.pes == 4:
        text += "#ifdef __CPU_SIM\npto::linx_cpu::runCore([&] {" + case.call + "});\n#else\n" + case.call + "\n#endif\n"
    else:
        text += case.call + "\n"
    text += "#ifdef __CPU_SIM\nif (std::fwrite(result,1,sizeof(result),stdout)!=sizeof(result)) return 1;\n#endif\nreturn 0;\n}\n"
    return text


def execute(command, log, cwd=None, timeout=90):
    with log.open("wb") as stream:
        try:
            proc = subprocess.run(
                list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, cwd=cwd, timeout=timeout
            )
            return {"command": list(map(str, command)), "returncode": proc.returncode, "log": str(log)}
        except subprocess.TimeoutExpired:
            return {"command": list(map(str, command)), "returncode": None, "log": str(log), "error": "timeout"}


def compare(actual, expected, dtype):
    if len(actual) != len(expected):
        return {"passed": False, "actual_bytes": len(actual), "expected_bytes": len(expected)}
    fmt = FORMATS[dtype]
    size = struct.calcsize(fmt)
    mismatches = []
    for offset in range(0, len(actual), size):
        if actual[offset : offset + size] != expected[offset : offset + size]:
            got = struct.unpack_from("<" + fmt, actual, offset)[0]
            want = struct.unpack_from("<" + fmt, expected, offset)[0]
            mismatches.append({"index": offset // size, "actual": got, "expected": want})
    return {"passed": not mismatches, "mismatches": len(mismatches), "examples": mismatches[:8]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--linx-compiler", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--runtime-dir", type=Path, help="Directory containing compatible target runtime libraries")
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--case", action="append")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--instruction-suite", action="store_true", help="Audit every active instruction API name")
    parser.add_argument(
        "--legacy-cases", dest="instruction_suite", action="store_false", help="Replay the historical 130 cases"
    )
    parser.set_defaults(instruction_suite=True)
    parser.add_argument(
        "--routes", nargs="+", choices=("cpu", "device", "upstream"), default=["cpu", "device", "upstream"]
    )
    parser.add_argument(
        "--device-include", type=Path, help="Diagnostic snapshot include directory; recorded in evidence"
    )
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout < 1:
        parser.error("--jobs and --timeout must be positive")
    args.output_dir = args.output_dir.resolve()
    args.model_root = args.model_root.resolve()
    args.upstream = args.upstream.resolve()
    args.linx_compiler = args.linx_compiler.resolve()
    if args.output_dir.exists():
        parser.error("--output-dir must be fresh")
    args.output_dir.mkdir(parents=True)
    if os.name == "posix":
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    version = subprocess.check_output([str(args.linx_compiler), "--version"], text=True)
    if "clang version 15." not in version:
        parser.error("the SuperNPUBench startup used here requires the LinxV5 Clang 15 toolchain")
    subprocess.run(["python3", ROOT / "tests/script/sync_linx_headers.py", "--upstream", args.upstream], check=True)
    resource_dir = subprocess.check_output([str(args.linx_compiler), "-print-resource-dir"], text=True).strip()
    # SuperNPUBench _start, with Clang 15's byte-offset spelling of C.SETRET.
    start = args.output_dir / "start.s"
    start.write_text(
        ".global _start\n.type _start,@function\n.text\n_start:\n bstart.std call main\n c.setret 2, ->ra\n c.bstart.sys fall\n addi zero, 0x5e, ->x1\n acrc 1\n c.bstop\n"
    )
    if args.instruction_suite:
        from linx_instruction_cases import instruction_cases, summarize, verify_catalog

        verify_catalog(args.compiler, args.linx_compiler, args.output_dir)
        available = instruction_cases()
    else:
        available = cases()
    selected = [c for c in available if not args.case or c.name in args.case]
    if not selected:
        parser.error("no cases selected")

    def run(case):
        folder = args.output_dir / case.name
        folder.mkdir()
        fmt = FORMATS[case.dtype]
        expected = b"".join(struct.pack("<" + fmt, v) for v in case.expected + [93] * 16)
        (folder / "golden.bin").write_bytes(expected)
        record = {
            "name": case.name,
            "dtype": case.dtype,
            "elements": len(case.expected),
            "pes": case.pes,
            "instructions": case.instructions,
            "profile": case.profile,
            "routes": {},
        }
        record["golden_sha256"] = hashlib.sha256(expected).hexdigest()
        for route in args.routes:
            source = folder / (route + ".cpp")
            source.write_text(program(case, route == "upstream"))
            binary = folder / (route + ".elf")
            if route == "cpu":
                command = [
                    args.compiler,
                    "-std=c++23",
                    "-O2",
                    "-ffp-contract=off",
                    "-pthread",
                    "-D__CPU_SIM",
                    "-DPTO_CPU_SIM_TARGET_LINX",
                    "-I",
                    ROOT / "include",
                    source,
                    "-o",
                    binary,
                ]
            else:
                command = [
                    args.linx_compiler,
                    "-std=c++20",
                    "-mlxbc",
                    "-fenable-matrix",
                    "-O2",
                    "-mllvm",
                    "-enable-all-vector-as-tilereg=true",
                    "-ffunction-sections",
                    "-fdata-sections",
                    "-nostdlib++",
                    "-nostartfiles",
                    "-static",
                    "-Wl,--gc-sections",
                    start,
                ]
                command += (
                    ["-I", (args.device_include or ROOT / "include").resolve()]
                    if route == "device"
                    else ["-isystem", Path(resource_dir) / "include/tileop-api", "-I", args.upstream / "include"]
                )
                if args.runtime_dir:
                    command += ["-L", args.runtime_dir.resolve()]
                command += [source, "-o", binary]
            item = {"build": execute(command, folder / (route + "-build.log"))}
            item["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            record["routes"][route] = item
            if item["build"]["returncode"] != 0:
                item["passed"] = False
                log = Path(item["build"]["log"]).read_text(errors="replace")
                item["compiler_crash"] = any(
                    marker in log for marker in ("PLEASE submit a bug report", "frontend command failed", "Stack dump:")
                )
                blocked = (
                    item["build"]["returncode"] == 1 and "PTO Linx experimental:" in log and not item["compiler_crash"]
                )
                item["status"] = "blocked" if blocked else "compile_failure"
                continue
            dump = folder / (route + ".bin")
            if route == "cpu":
                item["run"] = execute([binary], dump, timeout=args.timeout)
            else:
                symbols = subprocess.check_output(["readelf", "-sW", str(binary)], text=True)
                address = next(
                    int(line.split()[1], 16)
                    for line in symbols.splitlines()
                    if line.split() and line.split()[-1] == "result"
                )
                command = [
                    args.model_root / "bin/gfrun",
                    "-f",
                    binary,
                    "-s",
                    f"softcore.multiThreadNum={case.pes}",
                    "--dump-memory",
                    f"0x{address:x}:{len(expected)}:{dump}",
                ]
                item["run"] = execute(command, folder / (route + "-run.log"), cwd=args.model_root, timeout=args.timeout)
                log = Path(item["run"]["log"]).read_text(errors="replace")
                item["completed"] = bool(re.search(r"Reach the End of Benchmark! R2 = 0\s", log))
                item["model_error"] = bool(re.search(r"(?i)exception|fatal|error:|validation fail|illegal", log))
            if dump.exists():
                item["golden"] = compare(dump.read_bytes(), expected, case.dtype)
            item["passed"] = (
                item["run"]["returncode"] == 0
                and item.get("completed", True)
                and not item.get("model_error", False)
                and item.get("golden", {}).get("passed", False)
            )
            item["status"] = (
                "pass"
                if item["passed"]
                else "execution_failure"
                if item["run"]["returncode"] != 0 or not item.get("completed", True) or item.get("model_error", False)
                else "value_mismatch"
            )
        for route in ("device", "upstream"):
            a, b = folder / "cpu.bin", folder / (route + ".bin")
            if route in record["routes"] and a.exists() and b.exists():
                record["routes"][route]["cpu_parity"] = compare(b.read_bytes(), a.read_bytes(), case.dtype)
        record["passed"] = all(r["passed"] for r in record["routes"].values())
        (folder / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            case.name
            + ": "
            + ", ".join(r + "=" + ("PASS" if v["passed"] else "FAIL") for r, v in record["routes"].items()),
            flush=True,
        )
        return record

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        records = list(executor.map(run, selected))
    tracked_sources = list((ROOT / "include/pto/cpu/linx").glob("*.hpp")) + list(SOURCES.glob("*.cpp"))
    tracked_sources += [Path(__file__), ROOT / "tests/script/sync_linx_headers.py"]
    tracked_sources += list((ROOT / "include/pto/npu/linx").rglob("*.hpp"))
    if args.instruction_suite:
        tracked_sources += [ROOT / "tests/cpu/linx_instruction_cases.py", SOURCES / "instruction_catalog.json"]
    result = {
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tracked_sources)
        },
        "compiler": version,
        "host_compiler": subprocess.check_output([args.compiler, "--version"], text=True),
        "device_include": str((args.device_include or ROOT / "include").resolve()),
        "device_header_hashes": {
            str(p.relative_to(args.device_include or ROOT / "include")): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((args.device_include or ROOT / "include").glob("pto/npu/linx/**/*.hpp"))
        },
        "model_revision": subprocess.check_output(
            ["git", "-C", str(args.model_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "upstream_revision": subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        "cases": records,
        "passed": all(r["passed"] for r in records),
    }
    if args.instruction_suite:
        result["instruction_matrix"] = summarize(records, args.routes)
        (args.output_dir / "instruction-matrix.json").write_text(
            json.dumps(result["instruction_matrix"], indent=2) + "\n"
        )
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
