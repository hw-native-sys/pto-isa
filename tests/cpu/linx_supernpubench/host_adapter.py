# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Explicit source-portability adaptations for the pinned SuperNPUBench tree.

This is a build adapter, not a benchmark implementation. Every replacement is
recorded in a patch; numerical kernels and golden checks remain in upstream.
"""

from pathlib import Path
import difflib
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "script"))
from sync_linx_headers import scopes


def prepare(source, destination):
    shutil.copytree(source, destination, dirs_exist_ok=True)
    changes = []
    for path in sorted(destination.rglob("*")):
        if path.suffix not in (".cpp", ".hpp", ".h"):
            continue
        original = path.read_text()
        text = original
        # GCC does not implement GNU C range designators in C++. Initialize the
        # same raw array, before main, without changing its layout or contents.
        pattern = r"(static uint16_t xbits\[PM \* PN\].*?)= \{\[0 \.\.\. PM \* PN - 1\] = (0x[0-9a-fA-F]+)\};"
        text = re.sub(
            pattern,
            lambda m: m[1] + "= {};\nstatic const bool xbits_initialized = [] {\n"
            "    for (auto &x : xbits) x = " + m[2] + ";\n    return true;\n}();",
            text,
        )
        if path.name in ("hashtable_lookup.cpp", "hashtable_lookup_dataset.cpp"):
            # libstdc++'s thread headers expose POSIX ::sync, which collides
            # with the benchmark's unqualified anonymous-namespace variable.
            text = re.sub(r"\bsync\b", "lookup_sync", text)
        replacements = {}
        if path.name == "histogram_cumsum_m32.hpp":
            replacements = {
                "group_tmuls_cell": "auto view = pto::linx_cpu::CarrierTile<I32Tile, decltype(parent.data())>(parent.data(), cell); TMULS(dst, view, scalar);",
                "group_tadd_cell_pair": "auto a = pto::linx_cpu::CarrierTile<I32Tile, decltype(parent.data())>(parent.data(), cell0); auto b = pto::linx_cpu::CarrierTile<I32Tile, decltype(parent.data())>(parent.data(), cell1); pto::linx_cpu::binary<0>(dst, a, b);",
                "group_tadd_cell_tail": "auto a = pto::linx_cpu::CarrierTile<I32Tile, decltype(parent.data())>(parent.data(), cell0); pto::linx_cpu::binary<0>(dst, a, addend);",
                "tstore_i32_strided": "global_tensor<int32_t, RowMajor<-1, -1>> memory(base, kLane, strideBytes / sizeof(int32_t)); TSTORE(memory, src);",
            }
            replacements["group_tmuls_cell"] = replacements["group_tmuls_cell"].replace(
                "TMULS(dst, view, scalar)", "pto::linx_cpu::scalar<2>(dst, view, scalar)"
            )
        if path.name == "qli_pto_opt_dynamic.hpp":
            replacements = {"TCOLSUMX": "TCOLSUM(dst, src);"}
        if path.name.startswith("hashtable_lookup_tileop_"):
            replacements = {
                "gather_masked": "pto::linx_cpu::gatherGpr(dst, base, offset, exec_mask);",
                "select_i32": "pto::linx_cpu::selectGpr(dst, mask, yes, no);",
                "equal_keys": "mask = pto::linx_cpu::compareGpr<CmpMode::EQ>(a, b);",
                "empty_keys": "pto::linx_cpu::assignCompareGpr<CmpMode::EQ, false>(mask, a, UINT64_MAX);",
            }
            assembly, _, _ = scopes(text)
            for item in reversed(assembly):
                owner = item["owner"]
                if not owner or owner["name"] != "lookup":
                    continue
                body = text[item["start"] : item["end"]]
                if "MGATHER" in body:
                    replacement = "pto::linx_cpu::gatherGpr(values, reinterpret_cast<const char*>(table) + 8, result, ~uint64_t{0});"
                elif "TSTORE" in body:
                    replacement = "global_tensor<int32_t, RowMajor<32, 1>> memory(dst); TSTORE(memory, result);"
                else:
                    raise ValueError(f"Unrecognized hash emitter: {body}")
                text = text[: item["start"]] + replacement + text[item["end"] :]
        if replacements:
            assembly, _, _ = scopes(text)
            seen = set()
            for item in reversed(assembly):
                owner = item["owner"]
                if owner and owner["name"] in replacements:
                    text = text[: item["start"]] + replacements[owner["name"]] + text[item["end"] :]
                    seen.add(owner["name"])
            if seen != replacements.keys():
                raise ValueError(f"Unrecognized native emitter in {path}: {seen}")
        if path.name == "conv2d_img2col_dyn.hpp":
            # Empty =Sr asm represents compiler-assigned Shared registers. Name
            # those two slots explicitly for a host compiler without S registers.
            text = text.replace(
                "      } else {\n        // The two Shared handles",
                "        pto::linx_cpu::publishSlot<0, 8>(tA);\n"
                "        pto::linx_cpu::publishSlot<1, 8>(tW);\n"
                "      } else {\n        // The two Shared handles",
            )
            text = text.replace('asm volatile("" : "=Sr"(tA.handle_ref()));', "pto::linx_cpu::acquireSlot<0>(tA);")
            text = text.replace('asm volatile("" : "=Sr"(tW.handle_ref()));', "pto::linx_cpu::acquireSlot<1>(tW);")
        if path.name == "qli_check_opt_dynamic.cpp":
            text = text.replace('__asm__ __volatile__("B.HINT TRACE.begin\\n" : : :);', "(void)0;")
            text = text.replace('__asm__ __volatile__("B.HINT TRACE.end\\n" : : :);', "(void)0;")
            # Scalar as well as Tile accesses use the explicitly supplied GM map.
            for address in (
                "SRCQ_ADDR",
                "SRCK_ADDR",
                "SRCW_ADDR",
                "SRCSQ_ADDR",
                "SRCSK_ADDR",
                "OUT_SCORES",
                "OUT_INDICES",
                "KEY_SCRATCH",
            ):
                text = re.sub(
                    r"reinterpret_cast<([^>]+)>\(" + address + r"\)",
                    lambda m: "reinterpret_cast<"
                    + m[1]
                    + ">(pto::linx_cpu::globalAddress(reinterpret_cast<void*>("
                    + address
                    + "), 0, 1))",
                    text,
                )
        if path.name == "moe_combine_mt_dyn.cpp":
            # Publish all PEs' reset/input writes before PE0 toggles windowState.
            # The upstream test relies on a lockstep startup and otherwise races.
            text = text.replace("        BENCHSTART;", "        pto::linx_cpu::syncCore();\n        BENCHSTART;")
        if text != original:
            path.write_text(text)
            rel = str(path.relative_to(destination))
            changes.extend(
                difflib.unified_diff(original.splitlines(True), text.splitlines(True), "a/" + rel, "b/" + rel)
            )
    (destination.parent / "host-adapter.patch").write_text("".join(changes))


if __name__ == "__main__":
    prepare(Path(sys.argv[1]), Path(sys.argv[2]))
