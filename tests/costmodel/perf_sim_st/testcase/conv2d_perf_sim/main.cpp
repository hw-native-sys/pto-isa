/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/costmodel/perf_sim/launch.hpp>
#include <gtest/gtest.h>

#include "conv2d_forward_kernel.cpp"

using namespace pto;

// ── Small: cin=16, hin=8, win=16, hk=wk=3, c0=16, stride=1, pad=1 ──
// hout=8, wout=16 → m=128, k=144(3*3*16), n=256
void runConv2d_Small()
{
    RunConv2dForward<half, half, half,
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

TEST(Conv2dPerfSim, Small_16cin_8x16)
{
    LAUNCH_KERNEL(runConv2d_Small, , (1, nullptr, nullptr));

    auto &instrs = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    EXPECT_GT(instrs.size(), 0u) << "Expected recorded Conv2D instructions";

    bool has_matrix = false, has_mte2 = false, has_fixp = false;
    for (auto &rec : instrs) {
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

// ── Medium: cin=32, hin=16, win=16, hk=wk=3, c0=16 ──
// hout=16, wout=16 → m=256, k=288(3*3*32), n=256
void runConv2d_Medium()
{
    RunConv2dForward<half, half, half,
                     /*blockDim=*/1,
                     /*m=*/256, /*k=*/288, /*n=*/256,
                     /*singleCoreM=*/256, /*singleCoreK=*/288, /*singleCoreN=*/256,
                     /*baseM=*/128, /*baseK=*/48, /*baseN=*/256,
                     /*stepM=*/1, /*stepKa=*/3, /*stepKb=*/3, /*stepN=*/1,
                     /*batch=*/1, /*cin=*/32, /*hin=*/16, /*win=*/16, /*c0=*/16,
                     /*hk=*/3, /*wk=*/3,
                     /*hout=*/16, /*wout=*/16,
                     /*strideH=*/1, /*strideW=*/1,
                     /*dilationH=*/1, /*dilationW=*/1,
                     /*padTop=*/1, /*padBottom=*/1, /*padLeft=*/1, /*padRight=*/1>(nullptr, nullptr, nullptr);
}

TEST(Conv2dPerfSim, Medium_32cin_16x16)
{
    LAUNCH_KERNEL(runConv2d_Medium, , (1, nullptr, nullptr));

    auto &instrs = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    EXPECT_GT(instrs.size(), 0u) << "Expected recorded Conv2D instructions";

    bool has_matrix = false, has_mte2 = false, has_fixp = false;
    for (auto &rec : instrs) {
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

// ── Large: cin=64, hin=32, win=32, hk=wk=3, c0=16 ──
// hout=32, wout=32 → m=1024, k=576(3*3*64), n=256
void runConv2d_Large()
{
    RunConv2dForward<half, half, half,
                     /*blockDim=*/1,
                     /*m=*/1024, /*k=*/576, /*n=*/256,
                     /*singleCoreM=*/1024, /*singleCoreK=*/576, /*singleCoreN=*/256,
                     /*baseM=*/128, /*baseK=*/96, /*baseN=*/256,
                     /*stepM=*/1, /*stepKa=*/3, /*stepKb=*/3, /*stepN=*/1,
                     /*batch=*/1, /*cin=*/64, /*hin=*/32, /*win=*/32, /*c0=*/16,
                     /*hk=*/3, /*wk=*/3,
                     /*hout=*/32, /*wout=*/32,
                     /*strideH=*/1, /*strideW=*/1,
                     /*dilationH=*/1, /*dilationW=*/1,
                     /*padTop=*/1, /*padBottom=*/1, /*padLeft=*/1, /*padRight=*/1>(nullptr, nullptr, nullptr);
}

TEST(Conv2dPerfSim, Large_64cin_32x32)
{
    LAUNCH_KERNEL(runConv2d_Large, , (1, nullptr, nullptr));

    auto &instrs = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    EXPECT_GT(instrs.size(), 0u) << "Expected recorded Conv2D instructions";

    bool has_matrix = false, has_mte2 = false, has_fixp = false;
    for (auto &rec : instrs) {
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
