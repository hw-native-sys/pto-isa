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
#include <algorithm>
#include <gtest/gtest.h>

#include "fa_performance_kernel.cpp"

using namespace pto;

// ── Standard: S0=128, HEAD_SIZE=128, S1=1024, single core ──
void runTFA_128x128x1024()
{
    runTFA<128, 128, 1024, 128, 128, 256, 4, 8, false, false, 4>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                                 nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                                 nullptr, nullptr);
}

TEST(FAPerfSim, Standard_128x128x1024)
{
    LAUNCH_KERNEL(runTFA_128x128x1024, , (1, nullptr, nullptr));

    auto &instrs = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    EXPECT_GT(instrs.size(), 0u) << "Expected recorded instructions from FA kernel";

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

// ── MultiCore: 4 cores ──
TEST(FAPerfSim, MultiCore_4x128x128x1024)
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

// ── Small: S0=64, HEAD_SIZE=64, S1=512 ──
void runTFA_64x64x512()
{
    runTFA<64, 64, 512, 64, 64, 256, 4, 8, false, false, 4>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                            nullptr, nullptr);
}

TEST(FAPerfSim, Small_64x64x512)
{
    LAUNCH_KERNEL(runTFA_64x64x512, , (1, nullptr, nullptr));

    auto &instrs = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    EXPECT_GT(instrs.size(), 0u) << "Expected recorded instructions from FA kernel";

    bool has_matrix = false, has_vector = false;
    for (auto &rec : instrs) {
        if (rec.stage == ::pto::perf_sim::PipeStage::Matrix)
            has_matrix = true;
        if (rec.stage == ::pto::perf_sim::PipeStage::Vector)
            has_vector = true;
    }
    EXPECT_TRUE(has_matrix) << "Expected Matrix (TMATMUL) instructions";
    EXPECT_TRUE(has_vector) << "Expected Vector instructions";
}

// ── CAModelCompare: S0=128, HEAD_SIZE=64, S1=512, CUBE_S0=64, 2 cores ──
void runTFA_64x128x512_2core()
{
    runTFA<128, 64, 512, 64, 128, 256, 4, 8, false, false, 4>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                              nullptr, nullptr);
}

TEST(FAPerfSim, CAModelCompare_64x128x512_2core)
{
    LAUNCH_KERNEL(runTFA_64x128x512_2core, , (2, nullptr, nullptr));
}

// ── LongSeq: S0=256, HEAD_SIZE=64, S1=2048, CUBE_S0=128, CUBE_S1=128 ──
void runTFA_256x64x2048()
{
    runTFA<256, 64, 2048, 128, 128, 256, 4, 8, false, false, 4>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                                nullptr, nullptr);
}

TEST(FAPerfSim, LongSeq_256x64x2048)
{
    LAUNCH_KERNEL(runTFA_256x64x2048, , (1, nullptr, nullptr));

    auto &instrs = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    EXPECT_GT(instrs.size(), 0u) << "Expected recorded instructions from FA kernel";

    bool has_matrix = false, has_vector = false;
    for (auto &rec : instrs) {
        if (rec.stage == ::pto::perf_sim::PipeStage::Matrix)
            has_matrix = true;
        if (rec.stage == ::pto::perf_sim::PipeStage::Vector)
            has_vector = true;
    }
    EXPECT_TRUE(has_matrix) << "Expected Matrix (TMATMUL) instructions";
    EXPECT_TRUE(has_vector) << "Expected Vector instructions";
}
