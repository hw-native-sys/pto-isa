/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_PERF_SIM_LATENCY_HPP
#define PTO_PERF_SIM_LATENCY_HPP

#include "recorder.hpp"
#include <pto/common/type.hpp>
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <tuple>
#include <type_traits>

namespace pto::perf_sim {

// ── Compile-time TileType extraction (returns Ctrl for non-tiles) ──

template <typename T>
constexpr pto::TileType SafeGetTileLoc()
{
    if constexpr (requires { T::Loc; })
        return T::Loc;
    else
        return pto::TileType::Ctrl;
}

inline bool IsOneOf(const std::string &opcode, std::initializer_list<const char *> names)
{
    return std::any_of(names.begin(), names.end(), [&](const char *name) { return opcode == name; });
}

// ── Static opcode → PipeStage lookup (for non-polymorphic instructions) ──

inline PipeStage StaticPipeStageLookup(const std::string &opcode)
{
    if (IsOneOf(opcode, {"TSYNC", "TRESHAPE", "TASSIGN", "TPRINT", "TSUBVIEW", "TALLOC", "TFREE", "TPUSH", "TPOP"})) {
        return PipeStage::Scalar;
    }
    if (IsOneOf(opcode, {"TMATMUL", "TMATMUL_ACC", "TMATMUL_BIAS", "TGEMV", "TGEMV_ACC", "TGEMV_BIAS"})) {
        return PipeStage::Matrix;
    }
    if (opcode == "TIMG2COL" || opcode == "TTRANS") {
        return PipeStage::MTE1;
    }
    if (opcode == "TLOAD" || opcode == "TPREFETCH") {
        return PipeStage::MTE2_AIV;
    }
    if (opcode == "TSTORE") {
        return PipeStage::MTE3;
    }
    return PipeStage::Vector;
}

inline PipeStage ResolveOnChipMoveStage(pto::TileType dst_loc, pto::TileType src_loc)
{
    if (src_loc == pto::TileType::Acc) {
        return PipeStage::Fixpipe;
    }
    if (dst_loc == pto::TileType::Vec) {
        return src_loc == pto::TileType::Mat ? PipeStage::MTE1 : PipeStage::Vector;
    }
    return src_loc == pto::TileType::Vec ? PipeStage::MTE3 : PipeStage::MTE1;
}

// ── Dynamic resolution: opcode + TileTypes → PipeStage ──

inline PipeStage ResolvePipeStage(const std::string &opcode, pto::TileType dst_loc, pto::TileType src_loc)
{
    // TLOAD: GM/L2 → tile. dst tile determines MTE2_AIV (→UB) or MTE2_AIC (→L1)
    if (opcode == "TLOAD") {
        return (dst_loc == pto::TileType::Vec) ? PipeStage::MTE2_AIV : PipeStage::MTE2_AIC;
    }
    // TPREFETCH: always GM→UB (AIV side)
    if (opcode == "TPREFETCH") {
        return PipeStage::MTE2_AIV;
    }
    // TSTORE: tile → GM/L2. src tile determines component:
    //   Vec(UB)→GM: MTE3, Mat(L1)→GM: MTE3, Acc(L0C)→GM: Fixpipe
    if (opcode == "TSTORE") {
        if (src_loc == pto::TileType::Acc)
            return PipeStage::Fixpipe;
        return PipeStage::MTE3;
    }
    // TMOV: on-chip transfer, routing depends on dst+src buffer types
    //   dst=Vec(UB): src=Vec→Vector, src=Mat(L1)→MTE1, src=Acc(L0C)→Fixpipe
    //   dst=Mat(L1): src=Acc(L0C)→Fixpipe, src=Mat→MTE1, src=Vec(UB)→MTE3
    if (opcode == "TMOV") {
        return ResolveOnChipMoveStage(dst_loc, src_loc);
    }
    // TEXTRACT: sub-tile extract, same routing as TMOV (data movement
    // depends on src/dst buffer types)
    //   e.g. L1→L0A/L0B = MTE1, L0C→UB = Fixpipe, UB→UB = Vector
    if (opcode == "TEXTRACT") {
        return ResolveOnChipMoveStage(dst_loc, src_loc);
    }
    // TTRANS: L1→L1 transpose → MTE1
    if (opcode == "TTRANS")
        return PipeStage::MTE1;

    return StaticPipeStageLookup(opcode);
}

// ── Variadic dispatch: extract TileTypes from macro arguments ──

inline PipeStage ResolvePipeStageArgs(const char *opcode)
{
    return ResolvePipeStage(opcode, pto::TileType::Ctrl, pto::TileType::Ctrl);
}

template <typename First, typename... Rest>
inline PipeStage ResolvePipeStageArgs(const char *opcode, First &, const Rest &...)
{
    using DstType = std::remove_cv_t<std::remove_reference_t<First>>;
    constexpr auto dst_loc = SafeGetTileLoc<DstType>();
    if constexpr (sizeof...(Rest) > 0) {
        using SrcType = std::remove_cv_t<std::remove_reference_t<std::tuple_element_t<0, std::tuple<Rest...>>>>;
        constexpr auto src_loc = SafeGetTileLoc<SrcType>();
        return ResolvePipeStage(opcode, dst_loc, src_loc);
    } else {
        return ResolvePipeStage(opcode, dst_loc, pto::TileType::Ctrl);
    }
}

// ── Instructions that need CPU simulation (iterative convergence depends on values) ──

inline bool NeedsCpuSim(const std::string &opcode)
{
    static const std::unordered_map<std::string, bool> lut = {
        {"TEXP", true}, {"TLOG", true}, {"TSQRT", true}, {"TRSQRT", true}, {"TDIV", true}, {"TDIVS", true},
    };
    auto it = lut.find(opcode);
    return it != lut.end() ? it->second : false;
}

// ── Backward-compatible alias ──
inline PipeStage OpToPipeStage(const std::string &opcode) __attribute__((deprecated));
inline PipeStage OpToPipeStage(const std::string &opcode)
{
    return StaticPipeStageLookup(opcode);
}

} // namespace pto::perf_sim

#endif
