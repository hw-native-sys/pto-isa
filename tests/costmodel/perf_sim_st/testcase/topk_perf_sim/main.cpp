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

#include "topk_kernel.cpp"

using namespace pto;

// ── Small: 128 rows x 1024 cols, topk=128 ──
void runTopk_128x1024_k128()
{
    runTOPK<float, 1, 1, 1, 128, 1024, 1, 1, 1, 128, 1024, 128, 1>(nullptr, nullptr, nullptr, nullptr);
}

TEST(TopkPerfSim, Small_128x1024_k128)
{
    LAUNCH_KERNEL(runTopk_128x1024_k128, , (1, nullptr, nullptr));

    auto &instrs = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    EXPECT_GT(instrs.size(), 0u) << "Expected recorded TopK instructions";

    bool has_vector = false, has_mte2 = false, has_mte3 = false;
    for (auto &rec : instrs) {
        if (rec.stage == ::pto::perf_sim::PipeStage::Vector)
            has_vector = true;
        if (rec.stage == ::pto::perf_sim::PipeStage::MTE2_AIV)
            has_mte2 = true;
        if (rec.stage == ::pto::perf_sim::PipeStage::MTE3)
            has_mte3 = true;
    }
    EXPECT_TRUE(has_vector) << "Expected Vector (TSORT/TMRGSORT) instructions";
    EXPECT_TRUE(has_mte2) << "Expected MTE2_AIV (TLOAD) instructions";
    EXPECT_TRUE(has_mte3) << "Expected MTE3 (TSTORE) instructions";
}

// ── Medium: 256 rows x 2048 cols, topk=64 ──
void runTopk_256x2048_k64()
{
    runTOPK<float, 1, 1, 1, 256, 2048, 1, 1, 1, 256, 2048, 64, 1>(nullptr, nullptr, nullptr, nullptr);
}

TEST(TopkPerfSim, Medium_256x2048_k64)
{
    LAUNCH_KERNEL(runTopk_256x2048_k64, , (1, nullptr, nullptr));

    auto &instrs = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    EXPECT_GT(instrs.size(), 0u) << "Expected recorded TopK instructions";

    bool has_vector = false, has_mte2 = false, has_mte3 = false;
    for (auto &rec : instrs) {
        if (rec.stage == ::pto::perf_sim::PipeStage::Vector)
            has_vector = true;
        if (rec.stage == ::pto::perf_sim::PipeStage::MTE2_AIV)
            has_mte2 = true;
        if (rec.stage == ::pto::perf_sim::PipeStage::MTE3)
            has_mte3 = true;
    }
    EXPECT_TRUE(has_vector) << "Expected Vector (TSORT/TMRGSORT) instructions";
    EXPECT_TRUE(has_mte2) << "Expected MTE2_AIV (TLOAD) instructions";
    EXPECT_TRUE(has_mte3) << "Expected MTE3 (TSTORE) instructions";
}

// ── Narrow: 128 rows x 512 cols, topk=32 ──
void runTopk_128x512_k32()
{
    runTOPK<float, 1, 1, 1, 128, 512, 1, 1, 1, 128, 512, 32, 1>(nullptr, nullptr, nullptr, nullptr);
}

TEST(TopkPerfSim, Narrow_128x512_k32)
{
    LAUNCH_KERNEL(runTopk_128x512_k32, , (1, nullptr, nullptr));

    auto &instrs = ::pto::perf_sim::PtoRecorder::GetForCore(0);
    EXPECT_GT(instrs.size(), 0u) << "Expected recorded TopK instructions";

    bool has_vector = false, has_mte2 = false, has_mte3 = false;
    for (auto &rec : instrs) {
        if (rec.stage == ::pto::perf_sim::PipeStage::Vector)
            has_vector = true;
        if (rec.stage == ::pto::perf_sim::PipeStage::MTE2_AIV)
            has_mte2 = true;
        if (rec.stage == ::pto::perf_sim::PipeStage::MTE3)
            has_mte3 = true;
    }
    EXPECT_TRUE(has_vector) << "Expected Vector (TSORT/TMRGSORT) instructions";
    EXPECT_TRUE(has_mte2) << "Expected MTE2_AIV (TLOAD) instructions";
    EXPECT_TRUE(has_mte3) << "Expected MTE3 (TSTORE) instructions";
}
