/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COSTMODEL_ST_A5_FIT_A5_COST_CHECK_HPP
#define PTO_COSTMODEL_ST_A5_FIT_A5_COST_CHECK_HPP

#include <cmath>
#include <iostream>
#include <type_traits>

#include <gtest/gtest.h>

#include <pto/common/type.hpp>
#include <pto/costmodel/lightweight_costmodel.hpp>

namespace pto::test {

template <typename T>
constexpr ::pto::mocker::lightweight::DType ToA5CostModelDType()
{
    if constexpr (std::is_same_v<T, float>) {
        return ::pto::mocker::lightweight::DType::Float;
    } else if constexpr (std::is_same_v<T, half>) {
        return ::pto::mocker::lightweight::DType::Half;
    } else {
        return ::pto::mocker::lightweight::DType::Float;
    }
}

template <typename T, ::pto::mocker::lightweight::PtoOpcode Op, int Rows, int Cols, int ValidRows, int ValidCols,
          ::pto::VFImplKind VfImplKindValue = ::pto::VFImplKind::VFIMPL_DEFAULT>
::pto::mocker::lightweight::CostModelInput MakeA5VfInput()
{
    return ::pto::mocker::lightweight::CostModelInput{
        .op = Op,
        .dtype = ToA5CostModelDType<T>(),
        .rows = Rows,
        .cols = Cols,
        .arch = ::pto::mocker::lightweight::CostModelArch::A5,
        .valid_rows = ValidRows,
        .valid_cols = ValidCols,
        .vf_impl_kind = VfImplKindValue,
    };
}

inline void ExpectA5CycleNear(const char *case_name, const ::pto::mocker::lightweight::CostModelInput &input,
                              double expected_cycles, double min_precision)
{
    ::pto::mocker::lightweight::CostModelResult result{};
    ASSERT_TRUE(::pto::mocker::lightweight::EstimateCycles(input, result)) << case_name;

    const double actual_cycles = result.cycles;
    const double precision = expected_cycles == 0.0 ?
                                 (actual_cycles == 0.0 ? 1.0 : 0.0) :
                                 std::max(0.0, 1.0 - std::fabs(expected_cycles - actual_cycles) / expected_cycles);
    std::cout << "[A5_CYCLE] " << case_name << " actual=" << actual_cycles << " expected=" << expected_cycles
              << " precision=" << precision << " min_precision=" << min_precision << std::endl;
    constexpr double kPrecisionTolerance = 0.05;
    EXPECT_GE(precision + kPrecisionTolerance, min_precision) << case_name;
}

} // namespace pto::test

#endif // PTO_COSTMODEL_ST_A5_FIT_A5_COST_CHECK_HPP
