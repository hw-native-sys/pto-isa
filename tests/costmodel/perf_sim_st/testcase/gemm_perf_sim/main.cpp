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

#include "gemm_performance_kernel.cpp"

using namespace pto;

// ── Small: single tile 128x64x256 ──
void runGemm_128x64x256()
{
    RunGemmE2E<float, half, half, float, /*blockDim=*/1, 128, 64, 256, // m, k, n
               128, 64, 256,                                           // validM, validK, validN
               128, 64, 256,                                           // singleCoreM, singleCoreK, singleCoreN
               128, 64, 256,                                           // baseM, baseK, baseN
               1, 1, 1, 1>(                                            // stepM, stepKa, stepKb, stepN
        nullptr, nullptr, nullptr);
}

TEST(GemmPerfSim, Small_128x64x256)
{
    LAUNCH_KERNEL(runGemm_128x64x256, , (1, nullptr, nullptr));

    auto &instrs = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    EXPECT_GT(instrs.size(), 0u) << "Expected recorded GEMM instructions";

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

// ── Medium: multi K-tile 1536x256x1024, stepKa/Kb=4 (double-buffer) ──
void runGemm_1536x256x1024()
{
    RunGemmE2E<float, half, half, float, /*blockDim=*/1, 1536, 256, 1024, // m, k, n
               1536, 256, 1024,                                           // validM, validK, validN
               1536, 256, 1024,                                           // singleCoreM, singleCoreK, singleCoreN
               128, 64, 256,                                              // baseM, baseK, baseN
               1, 4, 4, 1>(                                               // stepM, stepKa, stepKb, stepN
        nullptr, nullptr, nullptr);
}

TEST(GemmPerfSim, Medium_1536x256x1024)
{
    LAUNCH_KERNEL(runGemm_1536x256x1024, , (1, nullptr, nullptr));

    auto &instrs = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    EXPECT_GT(instrs.size(), 0u) << "Expected recorded GEMM instructions";

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

// ── LargeK: K-heavy 256x4096x256, stepKa/Kb=4 ──
void runGemm_256x4096x256()
{
    RunGemmE2E<float, half, half, float, /*blockDim=*/1, 256, 4096, 256, // m, k, n
               256, 4096, 256,                                           // validM, validK, validN
               256, 4096, 256,                                           // singleCoreM, singleCoreK, singleCoreN
               128, 64, 256,                                             // baseM, baseK, baseN
               1, 4, 4, 1>(                                              // stepM, stepKa, stepKb, stepN
        nullptr, nullptr, nullptr);
}

TEST(GemmPerfSim, LargeK_256x4096x256)
{
    LAUNCH_KERNEL(runGemm_256x4096x256, , (1, nullptr, nullptr));

    auto &instrs = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    EXPECT_GT(instrs.size(), 0u) << "Expected recorded GEMM instructions";

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
