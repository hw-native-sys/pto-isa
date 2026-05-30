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

#include <cmath>
#include <cstdint>
#include <string_view>

#include <pto/costmodel/a5/formula_costmodel/formula_params_generated.hpp>

namespace pto::mocker::lightweight::a5::fit {

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

inline bool TryLookupVfFormulaParam(const VfCurveKey &key, const VfFormulaParam *&param)
{
    for (const auto &entry : kVfFormulaParamTable) {
        if (entry.op == key.op && entry.src_dtype == key.src_dtype && entry.dst_dtype == key.dst_dtype &&
            entry.shape_path == key.shape_path && entry.vf_impl_kind == key.vf_impl_kind &&
            entry.tail_kind == key.tail_kind && entry.op_params == key.op_params) {
            param = &entry;
            return true;
        }
    }
    return false;
}

inline uint64_t EvalVfFormula(const VfFormulaParam &param, uint64_t loop_count, uint64_t outer_iter,
                              uint64_t inner_iter, uint64_t tail_count)
{
    double value = 0.0;
    switch (param.formula_kind) {
        case FormulaKind::Linear:
            value = param.param0 * static_cast<double>(loop_count) + param.param1;
            break;
        case FormulaKind::LinearRows:
            value = param.param0 * static_cast<double>(loop_count) + param.param1 * static_cast<double>(outer_iter) +
                    param.param2;
            break;
        case FormulaKind::LinearTail:
            value = param.param0 * static_cast<double>(loop_count) + param.param1 * static_cast<double>(tail_count) +
                    param.param2;
            break;
        case FormulaKind::NestedLoop:
            value = (param.param0 * static_cast<double>(inner_iter) + param.param1) * static_cast<double>(outer_iter) +
                    param.param2;
            break;
        default:
            return 0;
    }
    return RoundToCycles(value);
}

} // namespace pto::mocker::lightweight::a5::fit

#endif
