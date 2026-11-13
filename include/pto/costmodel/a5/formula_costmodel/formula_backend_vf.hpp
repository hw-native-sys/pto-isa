/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_A5_FORMULA_BACKEND_VF_HPP
#define PTO_MOCKER_A5_FORMULA_BACKEND_VF_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string_view>

#include <pto/costmodel/a5/formula_costmodel/formula_params_generated.hpp>

namespace pto::mocker::lightweight::a5::fit {

inline constexpr double kRowLoopStartupCycles = 18.0;
inline constexpr uint64_t kFirstStartupLoop = 1;
inline constexpr uint64_t kSecondStartupLoop = 2;

struct VfCurveKey {
    std::string_view op;
    std::string_view src_dtype;
    std::string_view dst_dtype;
    ShapePath shape_path = ShapePath::Path1D;
    std::string_view vf_impl_kind;
    TailKind tail_kind = TailKind::Full;
    std::string_view op_params;
};

inline uint64_t RoundToCycles(double value)
{
    if (value <= 0.0) {
        return 0;
    }
    return static_cast<uint64_t>(std::llround(value));
}

inline bool TryLookupVfFormulaParam(const VfCurveKey& key, const VfFormulaParam*& param)
{
    for (const auto& entry : kVfFormulaParamTable) {
        if (entry.op == key.op && entry.src_dtype == key.src_dtype && entry.dst_dtype == key.dst_dtype &&
            entry.shape_path == key.shape_path && entry.vf_impl_kind == key.vf_impl_kind &&
            entry.tail_kind == key.tail_kind && entry.op_params == key.op_params) {
            param = &entry;
            return true;
        }
    }
    return false;
}

inline uint64_t EvalVfFormula(
    const VfFormulaParam& param, uint64_t loop_count, uint64_t outer_iter, uint64_t inner_iter, uint64_t tail_count,
    uint64_t row_inner_iter, uint64_t init_repeats = 0)
{
    double value = 0.0;
    switch (param.formula_kind) {
        case FormulaKind::CubeThroughputFloor:
            // Cube callers supply padded M, K and N, not vector loop counts.
            // Parameters: cycles/MAC, minimum effective M*N area, cycle floor, startup.
            // The utilization floor applies to the output tile, not to every
            // M row independently; increasing M can compensate for a small N.
            value = std::max(
                        param.param2,
                        param.param0 * static_cast<double>(outer_iter) *
                            std::max(param.param1, static_cast<double>(loop_count) * static_cast<double>(inner_iter))) +
                    param.param3;
            break;
        case FormulaKind::ScatterPatternRow:
            // The destination clear and the source row/repeat expansion are separate VFs.
            value = param.param0 + param.param1 * static_cast<double>(outer_iter) +
                    param.param2 * static_cast<double>(loop_count) + param.param3 * static_cast<double>(init_repeats);
            break;
        case FormulaKind::Linear:
            value = param.param0 * static_cast<double>(loop_count) + param.param1;
            break;
        case FormulaKind::LinearSmallLoop:
            value = loop_count <= static_cast<uint64_t>(param.param3) ?
                        param.param2 :
                        param.param0 * static_cast<double>(loop_count) + param.param1;
            break;
        case FormulaKind::LinearTwoStartup:
            value = loop_count <= kFirstStartupLoop  ? param.param2 :
                    loop_count <= kSecondStartupLoop ? param.param3 :
                                                       param.param0 * static_cast<double>(loop_count) + param.param1;
            break;
        case FormulaKind::ContinuousPiecewiseLinear: {
            const double loop = static_cast<double>(loop_count);
            const double breakpoint = param.param3;
            value = param.param0 * std::min(loop, breakpoint) + param.param1 +
                    param.param2 * std::max(loop - breakpoint, 0.0);
        } break;
        case FormulaKind::LinearRows:
            value = param.param0 * static_cast<double>(loop_count) + param.param1 * static_cast<double>(outer_iter) +
                    param.param2;
            break;
        case FormulaKind::LinearTail:
            value = param.param0 * static_cast<double>(loop_count) + param.param1 * static_cast<double>(tail_count) +
                    param.param2;
            break;
        case FormulaKind::NestedLoop: {
            const double inner = static_cast<double>(inner_iter);
            value =
                (param.param0 * inner + param.param1 + param.param3 * inner * inner) * static_cast<double>(outer_iter) +
                param.param2;
        } break;
        case FormulaKind::NestedLoopInnerBias: {
            const double inner = static_cast<double>(inner_iter);
            value = (param.param0 * inner + param.param1) * static_cast<double>(outer_iter) + param.param2 * inner +
                    param.param3;
        } break;
        case FormulaKind::DualLoopMax: {
            const double flat_path = param.param0 * static_cast<double>(loop_count) + param.param1;
            const double row_path =
                (param.param2 + static_cast<double>(row_inner_iter)) * static_cast<double>(outer_iter) + param.param3;
            value = std::max(flat_path, row_path);
        } break;
        case FormulaKind::DualLoopMaxScaled: {
            const double flat_path = param.param0 * static_cast<double>(loop_count) + param.param1;
            const double row_path =
                (param.param2 + param.param3 * static_cast<double>(row_inner_iter)) * static_cast<double>(outer_iter) +
                kRowLoopStartupCycles;
            value = std::max(flat_path, row_path);
        } break;
        case FormulaKind::LinearFloor:
            value = std::max(param.param0 * static_cast<double>(loop_count) + param.param1, param.param2);
            break;
        case FormulaKind::AffineGrid:
            value = param.param0 + param.param1 * static_cast<double>(outer_iter) +
                    param.param2 * static_cast<double>(row_inner_iter) +
                    param.param3 * static_cast<double>(outer_iter * row_inner_iter);
            break;
        default:
            return 0;
    }
    return RoundToCycles(value);
}

} // namespace pto::mocker::lightweight::a5::fit

#endif
