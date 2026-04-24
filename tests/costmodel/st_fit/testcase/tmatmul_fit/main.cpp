/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>

#include <pto/costmodel/lightweight_costmodel.hpp>

TEST(TMatmulFit, cycles_on_dtype_half)
{
    ::pto::mocker::lightweight::CostModelInput half_input{
        .op = ::pto::mocker::lightweight::PtoOpcode::TMATMUL,
        .dtype = ::pto::mocker::lightweight::DType::Half,
        .rows = 31,
        .cols = 17,
        .k = 33,
        .dtype2 = ::pto::mocker::lightweight::DType::Float,
        .dst_dtype = ::pto::mocker::lightweight::DType::Float,
    };

    ::pto::mocker::lightweight::CostModelResult half_result{};
    ASSERT_TRUE(::pto::mocker::lightweight::EstimateCycles(half_input, half_result));
    EXPECT_DOUBLE_EQ(half_result.cycles, 18.0); // head(6) + repeats(ceil(31/16)*ceil(33/16)*ceil(17/16))*1
}

TEST(TMatmulFit, cycles_on_dtype_float)
{
    ::pto::mocker::lightweight::CostModelInput float_input{
        .op = ::pto::mocker::lightweight::PtoOpcode::TMATMUL,
        .dtype = ::pto::mocker::lightweight::DType::Float,
        .rows = 31,
        .cols = 17,
        .k = 33,
        .dtype2 = ::pto::mocker::lightweight::DType::Half,
        .dst_dtype = ::pto::mocker::lightweight::DType::Float,
    };

    ::pto::mocker::lightweight::CostModelResult float_result{};
    ASSERT_TRUE(::pto::mocker::lightweight::EstimateCycles(float_input, float_result));
    EXPECT_DOUBLE_EQ(float_result.cycles, 46.0); // head(6) + repeats(ceil(31/16)*ceil(33/8)*ceil(17/16))*2
}
