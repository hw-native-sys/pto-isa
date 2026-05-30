/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// ── Real FA kernel costmodel test ──
// Compiles the actual flash_atten kernel under __COSTMODEL and runs perf-sim.

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/costmodel/perf_sim/launch.hpp>
#include <algorithm>
#include <gtest/gtest.h>

// Pull in the real FA kernel (runTFA + helpers; LaunchTFA is #ifndef __COSTMODEL guarded)
#include "fa_performance_kernel.cpp"

using namespace pto;

// Wrapper matching LAUNCH_KERNEL signature (no args)
void runTFA_128x128x1024()
{
    // S0=128, HEAD_SIZE=128, S1=1024, CUBE_S0=128, CUBE_S1=128, TILE_S1=256,
    // QK_PRELOAD=4, CV_FIFO_SIZE=8, INTERMEDIATE_CHECK=false, CAUSAL_MASK=false,
    // CV_FIFO_CONS_SYNC_PERIOD=4
    runTFA<128, 128, 1024, 128, 128, 256, 4, 8, false, false, 4>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                                 nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                                 nullptr, nullptr);
}

TEST(FAPerfSim, RunTFA_128x128x1024)
{
    LAUNCH_KERNEL(runTFA_128x128x1024, , (1, nullptr, nullptr));

    // Core 0 has Cube+VecCore0 records (logical cores 0,1 for block_dim=1)
    auto &instrs = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    EXPECT_GT(instrs.size(), 0u) << "Expected recorded instructions from real FA kernel";

    bool has_matrix = false, has_vector = false, has_gm_read = false, has_gm_write = false;
    for (auto &rec : instrs) {
        if (rec.stage == ::pto::perf_sim::PipeStage::Matrix)
            has_matrix = true;
        if (rec.stage == ::pto::perf_sim::PipeStage::Vector)
            has_vector = true;
        if (rec.stage == ::pto::perf_sim::PipeStage::MTE2_AIV || rec.stage == ::pto::perf_sim::PipeStage::MTE2_AIC)
            has_gm_read = true;
        if (rec.stage == ::pto::perf_sim::PipeStage::MTE3 || rec.stage == ::pto::perf_sim::PipeStage::Fixpipe)
            has_gm_write = true;
    }
    EXPECT_TRUE(has_matrix) << "Expected Matrix (TMATMUL) instructions";
    EXPECT_TRUE(has_vector) << "Expected Vector (TEXP/TADD) instructions";
    EXPECT_TRUE(has_gm_read) << "Expected GM read (TLOAD) instructions";
    EXPECT_TRUE(has_gm_write) << "Expected GM write (TSTORE) instructions";
}

// Multi-core test: 4 cores running the FA kernel
TEST(FAPerfSim, RunTFA_MultiCore_4x128x128x1024)
{
    LAUNCH_KERNEL(runTFA_128x128x1024, , (4, nullptr, nullptr));

    constexpr uint32_t kNumCores = 4;
    constexpr uint32_t kLogicalPerCore = ::pto::perf_sim::VEC_CORES_PER_AIC;
    for (uint32_t c = 0; c < kNumCores; ++c) {
        auto &instrs = ::pto::perf_sim::PtoRecorder::GetForCore(c * kLogicalPerCore);
        EXPECT_GT(instrs.size(), 0u) << "Core " << c << " should have instructions";
        for (auto &r : instrs) {
            EXPECT_EQ(r.core_id / kLogicalPerCore, c) << "core_id mismatch on core " << c;
        }
    }
}
