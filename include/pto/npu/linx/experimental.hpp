/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_LINX_EXPERIMENTAL_HPP
#define PTO_LINX_EXPERIMENTAL_HPP

// Deliberately no bypass macro. Diagnostic baseline tests use a separate snapshot.
// Dependent guards diagnose before lowering where the pinned compiler can crash
// at -O0. Other emitters use a code-generation error, including under NDEBUG.
namespace pto::linx_experimental {
#define PTO_LINX_REJECT(Name, Message)           \
    __attribute__((error(Message))) void Name(); \
    template <typename T>                        \
    void Name()                                  \
    {                                            \
        static_assert(sizeof(T) == 0, Message);  \
    }
PTO_LINX_REJECT(
    bounded_atomic, "PTO Linx experimental: bounded indexed atomics are disabled (TileOP #317 / Model #1252); use the "
                    "validated unbounded overload only with valid addresses. See docs/linx-tile-instruction-support.md")
PTO_LINX_REJECT(
    atomic_store, "PTO Linx experimental: TSTORE_ATOMIC_* is disabled: the pinned model performs overwrite instead of "
                  "reduction (Model #1119). See docs/linx-tile-instruction-support.md")
PTO_LINX_REJECT(
    region, "PTO Linx experimental: region and associated TEPL execution is disabled pending descriptor validation "
            "(Model #1253/#1254); use plain tiles. See docs/linx-tile-instruction-support.md")
PTO_LINX_REJECT(
    predicate, "PTO Linx experimental: Predicate region/atomic execution is disabled (Model #1131/#1254); use "
               "validated plain tile operations. See docs/linx-tile-instruction-support.md")
PTO_LINX_REJECT(
    gpr, "PTO Linx experimental: TGPR2T is disabled pending bundle/schema alignment (TileOP #318). See "
         "docs/linx-tile-instruction-support.md")
PTO_LINX_REJECT(
    image, "PTO Linx experimental: local TIMG2COL is disabled: empty-shard spill can fail (LLVM #164); use the "
           "validated Shared SPART profile. See docs/linx-tile-instruction-support.md")
PTO_LINX_REJECT(
    mx_acc, "PTO Linx experimental: TMATMUL_MX_ACC is disabled pending source-order repair (TileOP #319). See "
            "docs/linx-tile-instruction-support.md")
PTO_LINX_REJECT(
    legacy, "PTO Linx experimental: legacy multi-Tile transport and TSTORE_PART are disabled with the pinned "
            "assembler. See docs/linx-tile-instruction-support.md")
PTO_LINX_REJECT(
    scalar, "PTO Linx experimental: scalar/packed conversion and BF16 arithmetic are disabled pending compiler "
            "validation (LLVM #165/#166). See docs/linx-tile-instruction-support.md")
#undef PTO_LINX_REJECT
} // namespace pto::linx_experimental

#endif
