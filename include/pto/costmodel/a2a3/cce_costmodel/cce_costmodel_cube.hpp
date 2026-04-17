/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#pragma once

#include <type_traits>
#include <pto/costmodel/a2a3/cce_costmodel/cce_costmodel_core.hpp>

namespace {
constexpr uint64_t kDefaultHeadCycles = 6;
}

template <typename CType, typename AType, typename BType>
inline void mad(CType c, AType a, BType b, auto m, auto k, auto n, auto phase, auto kDirectionAlign, auto cmatrixSource,
                auto cmatrixInitVal)
{
    using dtype_a = std::remove_pointer_t<AType>;
    int cycle_per_repeat = 1;
    if (std::is_same_v<dtype_a, float>) {
        cycle_per_repeat = 2;
    }
    const uint64_t mTiles = CeilDiv(m, 16);
    const uint64_t kTiles = CeilDiv(k, 32 / sizeof(dtype_a));
    const uint64_t nTiles = CeilDiv(n, 16);
    const uint64_t cycles = EstimateLinearCycles(::pto::mocker::evaluator::PipeKey::CUBE, mTiles * kTiles * nTiles,
                                                 kDefaultHeadCycles, cycle_per_repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::CUBE, "mad", cycles, c, a, b, m, k, n, phase,
                                 kDirectionAlign, cmatrixSource, cmatrixInitVal);
}