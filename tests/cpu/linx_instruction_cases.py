# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Instruction-owned profiles: no credit for unreachable calls in a failed kernel.

The denominator is public active instruction API names, not assembly sites or
hardware opcodes. Explicit-dimension and associated spellings remain separately
visible. A name passes only when every listed profile passes; a rejection is
never numerical execution success. Helpers and retired declarations are separate.
"""

from dataclasses import replace
import json
import math
import re
import subprocess
import struct

from test_linx_models import Case, SOURCES, cases, declaration, f32

CATALOG = json.loads((SOURCES / "instruction_catalog.json").read_text())
PREAMBLE = "#include <pto/pto-inst.hpp>\n#include <cstdint>\nusing namespace pto;\n"


def verify_catalog(compiler, linx_compiler, output):
    """Check both preprocessed public surfaces, including generated wrappers."""
    from test_linx_models import ROOT

    source = output / "catalog.cpp"
    source.write_text(PREAMBLE)
    expected = set(sum(CATALOG.values(), []))
    evidence = {}
    for route, command in (
        ("cpu", [compiler, "-std=c++23", "-D__CPU_SIM", "-DPTO_CPU_SIM_TARGET_LINX"]),
        ("device", [str(linx_compiler), "-std=c++20", "-mlxbc"]),
    ):
        text = subprocess.check_output(command + ["-I", str(ROOT / "include"), "-E", str(source)], text=True)
        names = set(re.findall(r"\b(?:void|auto|decltype\(auto\))\s+([A-Z][A-Z0-9_]+)\s*\(", text))
        evidence[route] = {
            "declared": sorted(names),
            "missing": sorted(expected - names),
            "unclassified": sorted(names - expected),
        }
    (output / "catalog-audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if any(e["missing"] or e["unclassified"] for e in evidence.values()):
        raise ValueError("Instruction catalog drift: " + str(evidence))


def isolated(source, function):
    """Keep the selected kernel and its common declarations, not unrelated bodies."""
    text = (SOURCES / (source + ".cpp")).read_text()
    first = re.search(r"^(?:template <bool Bounded = true>\n)?(?:static )?void \w+\(", text, re.M).start()
    match = re.search(r"^(?:template <bool Bounded = true>\n)?(?:static )?void " + function + r"\(", text, re.M)
    start = text.index("{", match.end())
    end, depth = start + 1, 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[:first] + text[match.start() : end] + "\n"


def make(name, instructions, body, expected, dtype="float", inputs="", pes=1, profile=""):
    return Case(
        name,
        "",
        "",
        "probe(result);",
        expected,
        dtype,
        pes,
        tuple(instructions),
        PREAMBLE + inputs + f"\nvoid probe({dtype}* output) {{\n" + body + "\n}\n",
        profile,
    )


def instruction_cases():
    result = []
    mapping = {
        "compare_move": ["TCMP", "TSEL", "TMOV", "TPREFETCH"],
        "local_assembly": ["TLOAD_ASS"],
        "shared_load": ["TLOAD", "TMOV_S2L_BROADCAST"],
        "shared_single_producer": ["TLOAD"],
        "rearrange": ["TPACK", "TUNPACK", "TSHUF", "TPERMUTE", "TEXPANDS"],
        "gpr_bits": ["TGPR2T"],
        "image_nhwc": ["TIMG2COL"],
        "image_nchw": ["TIMG2COL_SPART", "TMOV_S2L_EXTRACT"],
        "weights": ["TLOAD"],
        "matrix_slots": ["TIMG2COL_SPART_SLOT", "TLOAD_SLOT", "TMATMUL_SLOT"],
        "matrix_acc_scale": ["TMATMUL_ACC"],
        "matrix_shared": ["TMATMUL"],
        "matrix_hif4": ["TMATMUL_MX"],
        "gemv": ["TGEMV", "TGEMV_ACC", "TGEMV_BIAS", "TGEMV_MX", "TGEMV_MX_ACC", "TGEMV_MX_BIAS"],
        "matrix_aux": ["TMATMUL"],
        "compare_exchange": ["MGATHER_CAS"],
        "indexed_mask": ["MGATHER", "MGATHER_MASK", "MSCATTER_MASK"],
        "select_bits": ["TCMPS", "TSELS", "TNOT", "TANDS", "TORS", "TXORS", "TSHLS", "TSHRS"],
        "tile_indexed": ["TGATHER", "TSCATTER"],
        "dynamic_shape": ["TLOAD", "TADDS", "TSTORE"],
        "core_gmov": ["GMOV"],
        "fp16_convert": ["TCVT"],
        "regions": ["TADD", "TPARTVIEW", "TASSEMBLY"],
        "predicate_regions": ["predicate::subview", "predicate::concatenate"],
        "predicate_atomic": ["predicate::atomic_add"],
        "scalar_rounding": ["linx_cvt"],
        "packed_bf16": ["blkv_bf16x2"],
    }
    gather = "EXCH MAX MIN ADD INC DEC AND OR XOR".split()
    scatter = "MAX MIN ADD INC DEC AND OR XOR POPC".split()
    for c in cases():
        if c.name in ("broadcast", "matrix_mx"):
            continue
        names = mapping.get(c.name)
        if c.name in CATALOG["instruction"]:
            names = [c.name]
        if c.name.startswith("broadcast_"):
            _, axis, op = c.name.split("_")
            names = ["T" + axis.upper() + "EXPAND" + ["", "ADD", "SUB", "MUL", "DIV", "MAX", "MIN", "EXPDIF"][int(op)]]
        for prefix, ops in (("gather_atomic_", gather), ("scatter_atomic_", scatter), ("store_atomic_", scatter)):
            if c.name.startswith(prefix):
                op = ops[int(c.name[len(prefix) :].split("_")[0])]
                names = [
                    (
                        "MGATHER_"
                        if prefix.startswith("gather")
                        else "MSCATTER_"
                        if prefix.startswith("scatter")
                        else "TSTORE_ATOMIC_"
                    )
                    + op
                ]
        if not names:
            raise ValueError("Unowned case: " + c.name)
        function = re.match(r"(\w+)", c.call)[1]
        result.append(
            replace(
                c,
                instructions=tuple(names),
                kernel=isolated(c.source, function),
                profile=c.name + "; see retained kernel/input/golden",
            )
        )

    # Isolate ACC and BIAS so an earlier failure cannot hide the later operation.
    original = next(c for c in cases() if c.name == "matrix_mx")
    kernel = isolated("matrix", "matrixMX")
    for op, value in (("TMATMUL_MX", 768), ("TMATMUL_MX_ACC", 770), ("TMATMUL_MX_BIAS", 771)):
        k = kernel
        for other in ("TMATMUL_MX", "TMATMUL_MX_ACC", "TMATMUL_MX_BIAS"):
            if other != op:
                k = re.sub(r"^    " + other + r"\([^\n]+\n", "", k, flags=re.M)
        k = re.sub(r"^    TSTORE\([^\n]+\n", "", k, flags=re.M)
        k = k.rsplit("}", 1)[0] + "    TSTORE(md, d);\n}\n"
        result.append(
            replace(
                original,
                name=op + "_isolated",
                kernel=k,
                expected=[value] * 128,
                instructions=(op,),
                profile="E4M3/E5M2 + E8M0, M16 16x8, isolated",
            )
        )

    # Basic GM/TLSU transport. Values encode position, detecting swaps and truncation.
    vals = [i / 4 - 17 for i in range(128)]
    inputs = declaration("float", "x", vals)
    body = "VecTileM32<float,32,4> a; global_tensor<float,RowMajor<32,4>> in(x), out(output); TLOAD_CUBE(a,in); TSTORE_CUBE(out,a);"
    result.append(
        make("cube_transport", ["TLOAD_CUBE", "TSTORE_CUBE"], body, vals, inputs=inputs, profile="M32 FP32 32x4")
    )
    c = next(c for c in result if c.name == "fp16_convert")
    result.append(replace(c, name="TCVT_T", kernel=c.kernel.replace("TCVT(", "TCVT_T("), instructions=("TCVT_T",)))
    result.append(
        make(
            "local_publish",
            ["TMOV_L2S_INSERT", "TMOV_L2S_PUBLISH"],
            """
using T=Tile<Location::Vec,float,1,32>; T a,b;
global_tensor<float,RowMajor<1,32>> in(x),out(output+get_thread_idx()*32);
TLOAD(a,in); auto s=TMOV_L2S_INSERT<8>(a); TMOV_L2S_PUBLISH<8>(s,a);
TMOV_S2L_BROADCAST(b,s); TSTORE(out,b);
""",
            vals[:32] * 4,
            inputs=declaration("float", "x", vals[:32]),
            pes=4,
            profile="singleton PE0 publication; four consumers",
        )
    )
    result.append(
        Case(
            "shared_nd",
            "storage",
            declaration("float", "x", list(range(512))),
            "sharedND(x,result);",
            [((pe + 1) % 4) * 128 + i for pe in range(4) for i in range(128)],
            "float",
            4,
            ("TMOV_CUBE_TO_SHARED_ND", "TMOV_SHARED_ND_TO_CUBE"),
            isolated("storage", "sharedND"),
            "M32 FP32 / Shared ND, four producers, cyclic subview",
        )
    )
    # The model test uses a legal single producer to avoid conflating publication with SPART.
    result.append(
        make(
            "TSTORE_PART",
            ["TSTORE_PART"],
            """
using T=Tile<Location::Vec,float,1,32>;
global_tensor<float,RowMajor<1,32>> in(x), out(output);
auto s=TLOAD<T,8>(in); TSTORE_PART<8>(out,s);
""",
            vals[:32],
            inputs=declaration("float", "x", vals[:32]),
            pes=4,
            profile="Shared singleton FP32, PEMask=8",
        )
    )
    for op in ("TCI", "TCI_2D", "TCI_ROW", "TCI_COL", "TTRI"):
        if op == "TCI":
            ty = "Tile<Location::Vec,int32_t,1,32>"
            shape = "1,32"
            call = "TCI(d,7)"
            want = list(range(7, 39))
        elif op == "TTRI":
            ty = "Tile<Location::Vec,int32_t,4,32>"
            shape = "4,32"
            call = "TTRI(d)"
            want = [int(r >= c) for r in range(4) for c in range(32)]
        else:
            ty = "CubeTileM32<int32_t,32,4>"
            shape = "32,4"
            call = {
                "TCI_2D": "TCI_2D<decltype(d),int32_t,-1,1>(d,7)",
                "TCI_ROW": "TCI_ROW(d,7)",
                "TCI_COL": "TCI_COL(d,7)",
            }[op]
            want = [
                7 + (-r + c if op == "TCI_2D" else r if op == "TCI_ROW" else c) for r in range(32) for c in range(4)
            ]
        result.append(
            make(
                op,
                [op],
                f"{ty} d; {call}; global_tensor<int32_t,RowMajor<{shape}>> out(output); TSTORE(out,d);",
                want,
                "int32_t",
                profile=ty,
            )
        )
    # Plain scatter is independent of the masked/atomic forms.
    result.append(
        make(
            "MSCATTER",
            ["MSCATTER"],
            """
Tile<Location::Vec,uint32_t,1,32> a; Tile<Location::Vec,int32_t,1,32> i;
global_tensor<uint32_t,RowMajor<1,32>> in(x); global_tensor<int32_t,RowMajor<1,32>> ix(idx);
TLOAD(a,in); TLOAD(i,ix); global_tensor<uint32_t,RowMajor<1,32>> out(output); MSCATTER(out,a,i);
""",
            list(reversed(range(32))),
            "uint32_t",
            declaration("uint32_t", "x", list(range(32)))
            + declaration("int32_t", "idx", [(31 - i) * 4 for i in range(32)]),
            profile="U32, unique S32 reverse byte offsets",
        )
    )
    result.append(
        make(
            "TMATMUL_BIAS",
            ["TMATMUL_BIAS"],
            """
CubeTileM32<float,32,8> a; CubeTileN8<float,8,8> b; CubeBias<float,8> bias;
CubeAccumulatorM32<float,32,8> d;
float av[256],bv[64],cv[8]; for(auto& v:av)v=2; for(auto& v:bv)v=3; for(auto& v:cv)v=5;
global_tensor<float,RowMajor<32,8>> am(av); global_tensor<float,RowMajor<8,8>> bm(bv);
global_tensor<float,RowMajor<1,8>> cm(cv); TLOAD(a,am); TLOAD(b,bm); TLOAD(bias,cm);
TMATMUL_BIAS(d,a,b,bias); global_tensor<float,RowMajor<32,8>> out(output); TSTORE(out,d);
""",
            [53] * 256,
            profile="FP32 M32/N8, constant dot plus bias",
        )
    )
    for op in CATALOG["explicit_dimensions"]:
        c = next(c for c in result if c.name == op[:-2])
        k = c.kernel.replace(op[:-2] + "(d, a, b)", op + "<decltype(d),32,4>(d, a, b)")
        result.append(replace(c, name=op, kernel=k, instructions=(op,), profile="explicit columns=32, rows=4"))
    result.extend(legacy_cases())
    result.extend(associated_cases())
    reload = next(c for c in result if c.name == "TLOAD_CUBE_ASS")
    result.append(
        replace(
            reload,
            name="TLOAD_ASS_reload",
            instructions=("TLOAD_ASS",),
            kernel=reload.kernel.replace("TLOAD_CUBE_ASS(", "TLOAD_ASS("),
        )
    )
    result.extend(numeric_cases())
    covered = {op for c in result for op in c.instructions}
    missing = set(sum((CATALOG[k] for k in ("instruction", "explicit_dimensions", "associated")), [])) - covered
    if missing:
        raise ValueError("Missing instruction profiles: " + ", ".join(sorted(missing)))
    return result


def legacy_cases():
    result = []
    for n in (2, 4):
        for layout in ("ND2NZ", "ND2ZN", "DN2ZN"):
            name = f"TLOAD{n}_{layout}"
            # Legacy boxed NZ/ZN shapes, each tile contains 16x16 FP16 values.
            nz = layout == "ND2NZ" or (layout == "DN2ZN" and n == 2)
            cube = "SLayout::RowMajor" if nz else "SLayout::ColMajor"
            outer = "BLayout::ColMajor" if nz else "BLayout::RowMajor"
            gm = "ColMajor" if layout == "DN2ZN" else "RowMajor"
            rows, cols = (16, 16 * n) if layout == "ND2NZ" else (16 * n, 16)
            vals = list(range(rows * cols))
            kernel = f"using T=Tile<Location::Vec,__half,16,16,{outer},16,16,{cube}>; "
            kernel += (
                "T " + ",".join(f"d{i}" for i in range(n)) + f"; global_tensor<__half,{gm}<{rows},{cols}>> in(x);\n"
            )
            kernel += name + "(" + ",".join(f"d{i}" for i in reversed(range(n))) + ",in);\n"
            want = []
            for j in range(n):
                kernel += f"global_tensor<__half,RowMajor<16,16>> o{j}(output+{j * 256}); TSTORE(o{j},d{j});\n"
                for r in range(16):
                    for c in range(16):
                        rr, cc = (r, c + j * 16) if layout == "ND2NZ" else (r + j * 16, c)
                        want.append(vals[cc * rows + rr if gm == "ColMajor" else rr * cols + cc])
            result.append(
                make(
                    name,
                    [name],
                    kernel,
                    want,
                    "__half",
                    declaration("__half", "x", vals),
                    profile="legacy boxed FP16, distinct coordinate values",
                )
            )
    result.append(
        make(
            "TSTORE2_DN2DN",
            ["TSTORE2_DN2DN"],
            """
using T=Tile<Location::Vec,float,32,4,BLayout::ColMajor>;
T a,b; TEXPANDS(a,2.f); TEXPANDS(b,3.f);
global_tensor<float,ColMajor<64,4>> out(output); TSTORE2_DN2DN(out,b,a);
""",
            ([2] * 32 + [3] * 32) * 4,
            profile="ColMajor FP32, two distinct source tiles",
        )
    )
    return result


def associated_cases():
    """Two disjoint 128-byte slots; inspect both after LAST publishes the parent."""
    result = []
    unary = {
        "TABS": abs,
        "TNEG": lambda a: -a,
        "TEXP": math.exp,
        "TLOG": lambda a: f32(f32(math.log2(a)) * f32(math.log(2))),
        "TRECIP": lambda a: 1 / a,
        "TSQRT": math.sqrt,
        "TRSQRT": lambda a: 1 / f32(math.sqrt(a)),
        "TRELU": lambda a: max(a, 0),
        "TNOT": lambda a: ~a & 0xFFFFFFFF,
    }
    binary = {
        "TADD": lambda a, b: a + b,
        "TSUB": lambda a, b: a - b,
        "TMUL": lambda a, b: a * b,
        "TDIV": lambda a, b: a / b,
        "TREM": math.fmod,
        "TMAX": max,
        "TMIN": min,
        "TAND": lambda a, b: a & b,
        "TOR": lambda a, b: a | b,
        "TXOR": lambda a, b: a ^ b,
        "TSHL": lambda a, b: (a << b) & 0xFFFFFFFF,
        "TSHR": lambda a, b: a >> b,
    }
    for op in CATALOG["associated"]:
        if op == "TLOAD_ASS":
            continue
        if op == "TLOAD_CUBE_ASS":
            result.append(
                make(
                    op,
                    [op],
                    """
VecTileM32<float,32,4> a;
global_tensor<float,RowMajor<32,4>> in(x),initial(x+128),out(output);
TLOAD(a,initial); TLOAD_CUBE_ASS(a,in); TSTORE(out,a);
""",
                    list(range(128)),
                    inputs=declaration("float", "x", list(range(256))),
                    profile="M32 FP32 associated reload; initial data differs from replacement",
                )
            )
            continue
        if op == "TPACK_ASS":
            vals = [i % 4 + 1 for i in range(32)]
            result.append(
                make(
                    op,
                    [op],
                    """
using P=VecTileM32<uint32_t,32,2>; P parent;
VecTileM32<uint32_t,32,1> a,b;
global_tensor<uint32_t,RowMajor<32,1>> in(x),inb(y);
TLOAD(a,in); TLOAD(b,inb);
range::Assemble<P,2,true,false,0,0,1> first(parent);
TPACK(first,a,b,0x0202); auto last=range::assemble_last<1,1>(parent);
TPACK_ASS(last,a,b,0x0202);
global_tensor<uint32_t,RowMajor<32,2>> out(output); TSTORE(out,parent);
""",
                    [v for x in vals for v in [(2 << 16) | x] * 2],
                    "uint32_t",
                    declaration("uint32_t", "x", vals) + declaration("uint32_t", "y", [2] * 32),
                    profile="M32 U32 disjoint 128B TPACK INIT/LAST slots",
                )
            )
            continue
        base = op[:-4]
        uint = base in (
            "TNOT",
            "TAND",
            "TOR",
            "TXOR",
            "TSHL",
            "TSHR",
            "TANDS",
            "TORS",
            "TXORS",
            "TSHLS",
            "TSHRS",
            "TCMP",
            "TCMPS",
            "TPACK",
        )
        ty = "uint32_t" if uint else "float"
        x = [i % 4 + 1 for i in range(32)]
        y = [2] * 32
        call = ""
        want = []
        extra = ""
        if base in unary:
            call = f"{op}(last,a);"
            want = [unary[base](v) for v in x]
        elif base in binary:
            call = f"{op}(last,a,b);"
            want = [binary[base](v, 2) for v in x]
        elif base.endswith("S") and base[:-1] in binary:
            call = f"{op}(last,a,{ty}(2));"
            want = [binary[base[:-1]](v, 2) for v in x]
        elif base in ("TLOAD", "TLOAD_CUBE"):
            call = f"{op}(last,in);"
            want = x
        elif base == "TFMA":
            call = f"{op}(last,a,b,a);"
            want = [v * 2 + v for v in x]
        elif base == "TCVT":
            call = f"{op}(last,a);"
            want = x
        elif base in ("TCMP", "TCMPS"):
            call = f"{op}<CmpMode::GE>(last,a," + ("2u" if base == "TCMPS" else "b") + ");"
            want = [sum(int(v >= 2) << i for i, v in enumerate(x))] + [0] * 31
        elif base == "TPACK":
            call = f"{op}(last,a,b,0x0202);"
            want = [(2 << 16) | v for v in x]
        elif "EXPAND" in base:
            tail = base.split("EXPAND")[1]
            call = f"{op}(last," + ("b" if not tail else "a,b") + ");"
            fn = (
                (lambda a, b: b)
                if not tail
                else (lambda a, b: math.exp(a - b))
                if tail == "EXPDIF"
                else binary["T" + tail]
            )
            want = [fn(v, 2) for v in x]
        elif base.startswith(("TROW", "TCOL")):
            fn = {"SUM": sum, "PROD": lambda v: math.prod(v), "MAX": max, "MIN": min}
            tail = base[4:]
            call = f"{op}(last,a);"
            want = [fn[tail](x)] + [0] * 31 if base.startswith("TROW") else x
        else:
            raise ValueError(op)
        # A full parent stores two physical slots; each producer has one valid row.
        body = f"using T=Tile<Location::Vec,{ty},1,64,BLayout::RowMajor,1,32>; T a,b,parent;\n"
        body += f"global_tensor<{ty},RowMajor<1,32>> in(x), iny(y); TLOAD(a,in); TLOAD(b,iny);\n"
        body += (
            "range::Assemble<T,2,true,false,0,0,1> first(parent); TLOAD(first,in); auto last=range::assemble_last<1,1>(parent);\n"
            + extra
            + call
            + "\n"
        )
        body += f"Tile<Location::Vec,{ty},1,64> full; full.data()=parent.data(); global_tensor<{ty},RowMajor<1,64>> out(output); TSTORE(out,full);"
        result.append(
            make(
                op,
                [op],
                body,
                x + [f32(v) if ty == "float" else v for v in want],
                ty,
                declaration(ty, "x", x) + declaration(ty, "y", y),
                profile="RowMajor 1x32, disjoint INIT/LAST 128B slots, 256B parent",
            )
        )
    return result


def numeric_cases():
    """Discriminating low-precision inputs, not exact-value arithmetic only.

    RSQRT is explicitly a pinned-model profile (typed SQRT then DIV). The
    pinned specification's single-round reference is a separate open contract.
    """
    from test_linx_models import FORMATS

    FORMATS["uint16_t"] = "H"
    result = []

    def half(x):
        return struct.unpack("<e", struct.pack("<e", x))[0]

    def hbits(x):
        return struct.unpack("<H", struct.pack("<e", x))[0]

    def bf(x):
        bits = struct.unpack("<I", struct.pack("<f", x))[0]
        return struct.unpack("<f", struct.pack("<I", (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000))[0]

    def bbits(x):
        return struct.unpack("<I", struct.pack("<f", x))[0] >> 16

    for label, ty, rounder, bits in (("fp16", "__half", half, hbits), ("bf16", "__bf16", bf, bbits)):
        pairs = []
        for i in range(1, 6000):
            b = rounder(-i / 8192)
            want = rounder(f32(math.exp(rounder(1 - b))))
            if want != rounder(math.exp(1 - b)):
                pairs.append((1, b, want))
                if len(pairs) == 128:
                    break
        inputs = declaration("uint16_t", "x", [bits(a) for a, b, v in pairs]) + declaration(
            "uint16_t", "y", [bits(b) for a, b, v in pairs]
        )
        body = f"""
using T=Tile<Location::Vec,{ty},4,32>; T a,b,d;
global_tensor<{ty},RowMajor<4,32>> mx(reinterpret_cast<{ty}*>(x)),my(reinterpret_cast<{ty}*>(y)),out(reinterpret_cast<{ty}*>(output));
TLOAD(a,mx); TLOAD(b,my); TEXPDIF(d,a,b); TSTORE(out,d);
"""
        result.append(
            make(
                "TEXPDIF_" + label,
                ["TEXPDIF"],
                body,
                [bits(v) for a, b, v in pairs],
                "uint16_t",
                inputs,
                profile="same-type typed SUB then FP32 EXP then output rounding; 128 discriminating lanes",
            )
        )
        vals = []
        for raw in range(0x3000, 0x5000):
            value = (
                struct.unpack("<e", struct.pack("<H", raw))[0]
                if label == "fp16"
                else struct.unpack("<f", struct.pack("<I", raw << 16))[0]
            )
            if rounder(1 / rounder(math.sqrt(value))) != rounder(1 / math.sqrt(value)):
                vals.append(value)
                if len(vals) == 128:
                    break
        body = body.replace("TLOAD(b,my); TEXPDIF(d,a,b);", "TRSQRT(d,a);")
        inputs = declaration("uint16_t", "x", [bits(v) for v in vals]) + declaration("uint16_t", "y", [0] * 128)
        result.append(
            make(
                "TRSQRT_" + label,
                ["TRSQRT"],
                body,
                [bits(rounder(1 / rounder(math.sqrt(v)))) for v in vals],
                "uint16_t",
                inputs,
                profile="pinned model: typed SQRT then DIV; differs from pinned spec; 128 discriminating lanes",
            )
        )
    return result


def summarize(records, routes):
    matrix = []
    catalog = {name: category for category, names in CATALOG.items() for name in names if category != "retired"}
    catalog.update({op: "auxiliary" for c in records for op in c["instructions"] if op not in catalog})
    for name, category in sorted(catalog.items()):
        profiles = [c for c in records if name in c["instructions"]]
        item = {"instruction": name, "category": category, "cases": [c["name"] for c in profiles], "routes": {}}
        for route in routes:
            statuses = [c["routes"][route]["status"] for c in profiles]
            status = (
                "not_tested"
                if not statuses
                else "pass"
                if all(s == "pass" for s in statuses)
                else "blocked"
                if all(s in ("blocked", "pass") for s in statuses)
                else "fail"
            )
            item["routes"][route] = {
                "status": status,
                "passing_profiles": statuses.count("pass"),
                "total_profiles": len(statuses),
                "profile_statuses": statuses,
            }
        matrix.append(item)
    summary = {}
    for category in ("instruction", "explicit_dimensions", "associated", "helper", "auxiliary"):
        rows = [r for r in matrix if r["category"] == category]
        summary[category] = {
            route: {
                "passed": sum(r["routes"][route]["status"] == "pass" for r in rows),
                "total": len(rows),
                "blocked": sum(r["routes"][route]["status"] == "blocked" for r in rows),
                "not_tested": sum(r["routes"][route]["status"] == "not_tested" for r in rows),
            }
            for route in routes
        }
    return {
        "counting_unit": "public API name; all owned profiles must pass; helpers/retired excluded",
        "summary": summary,
        "rows": matrix,
    }
