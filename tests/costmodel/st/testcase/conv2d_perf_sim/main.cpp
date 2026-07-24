/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/costmodel/perf_sim/launch.hpp>
#include <gtest/gtest.h>

#include "conv2d_forward_kernel.cpp"

using namespace pto;

// Small Conv2D: batch=1, cin=16, hin=8, win=16, c0=16, hk=3, wk=3
// hout=8, wout=16 (stride=1, pad=1, dilation=1)
// mLoop=1, nLoop=1, kLoop=3
void runConv2d_Small()
{
    RunConv2dForward<
        half, half, half,
        /*blockDim=*/1,
        /*m=*/128, /*k=*/144, /*n=*/256,
        /*singleCoreM=*/128, /*singleCoreK=*/144, /*singleCoreN=*/256,
        /*baseM=*/128, /*baseK=*/48, /*baseN=*/256,
        /*stepM=*/1, /*stepKa=*/3, /*stepKb=*/3, /*stepN=*/1,
        /*batch=*/1, /*cin=*/16, /*hin=*/8, /*win=*/16, /*c0=*/16,
        /*hk=*/3, /*wk=*/3,
        /*hout=*/8, /*wout=*/16,
        /*strideH=*/1, /*strideW=*/1,
        /*dilationH=*/1, /*dilationW=*/1,
        /*padTop=*/1, /*padBottom=*/1, /*padLeft=*/1, /*padRight=*/1>(nullptr, nullptr, nullptr);
}

TEST(Conv2dPerfSim, RunConv2d_Small)
{
    LAUNCH_KERNEL(runConv2d_Small, , (1, nullptr, nullptr));

    auto& instrs = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    EXPECT_GT(instrs.size(), 0u) << "Expected recorded Conv2D instructions";

    bool has_matrix = false, has_mte2 = false, has_fixp = false;
    for (auto& rec : instrs) {
        if (rec.stage == ::pto::perf_sim::PipeStage::Matrix)
            has_matrix = true;
        if (rec.stage == ::pto::perf_sim::PipeStage::MTE2_AIC)
            has_mte2 = true;
        if (rec.stage == ::pto::perf_sim::PipeStage::Fixpipe)
            has_fixp = true;
    }
    EXPECT_TRUE(has_matrix) << "Expected Matrix (TMATMUL) instructions";
    EXPECT_TRUE(has_mte2) << "Expected MTE2_AIC (TLOAD) instructions";
    EXPECT_TRUE(has_fixp) << "Expected Fixpipe (TSTORE) instructions";
}
