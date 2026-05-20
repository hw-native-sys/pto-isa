/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "cpu_tile_test_utils.h"

#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

using CpuTileTestUtils::AssignTileStorage;
using CpuTileTestUtils::FillLinear;

void LaunchTraceKernel(float *out, float *src0, float *src1, void *stream);

namespace {

class TTraceTest : public testing::Test {
protected:
    void SetUp() override
    {
        pto::cpu_sim::ResetInstructionTrace();
        pto::cpu_sim::reset_execution_context();
        pto::NPU_MEMORY_CLEAR();
        pto::cpu_sim::set_execution_context(7, 0, 1);
    }

    void TearDown() override
    {
        pto::cpu_sim::ResetInstructionTrace();
        pto::cpu_sim::reset_execution_context();
        pto::NPU_MEMORY_CLEAR();
    }
};

TEST_F(TTraceTest, CapturesZeroOperandAndBasicTileInstructions)
{
    if (!pto::cpu_sim::kInstructionTraceEnabled) {
        GTEST_SKIP() << "PTO_CPU_SIM_TRACE_MODE is disabled for this build.";
    }

    using TileData = pto::Tile<pto::TileType::Vec, float, 2, 32>;

    TileData dst;
    TileData src0;
    TileData src1;
    std::size_t addr = 0;
    AssignTileStorage(addr, dst, src0, src1);
    FillLinear(src0, 1.0f);
    FillLinear(src1, 2.0f);
    pto::cpu_sim::ResetInstructionTrace();

    pto::TSYNC<pto::Op::TADD>();
    pto::TADD(dst, src0, src1);
    pto::TDIV(dst, src0, src1);

    const auto &trace = pto::cpu_sim::GetInstructionTrace();
    ASSERT_EQ(trace.records.size(), 3u);

    EXPECT_EQ(trace.records[0].opcode, "TSYNC");
    EXPECT_EQ(trace.records[0].block_idx, 7u);
    EXPECT_EQ(trace.records[0].sequence_id, 0u);
    EXPECT_TRUE(trace.records[0].input_tiles.empty());
    EXPECT_TRUE(trace.records[0].scalar_inputs.empty());
    EXPECT_TRUE(trace.records[0].output_tiles.empty());

    EXPECT_EQ(trace.records[1].opcode, "TADD");
    EXPECT_EQ(trace.records[1].sequence_id, 1u);
    ASSERT_EQ(trace.records[1].input_tiles.size(), 2u);
    ASSERT_EQ(trace.records[1].output_tiles.size(), 1u);
    EXPECT_EQ(trace.records[1].input_tiles[0].address, src0.GetAssignedAddress());
    EXPECT_EQ(trace.records[1].input_tiles[1].address, src1.GetAssignedAddress());
    EXPECT_EQ(trace.records[1].output_tiles[0].address, dst.GetAssignedAddress());
    EXPECT_EQ(trace.records[1].output_tiles[0].shape, (std::vector<int64_t>{2, 32}));

    EXPECT_EQ(trace.records[2].opcode, "TDIV");
    EXPECT_EQ(trace.records[2].sequence_id, 2u);
    ASSERT_EQ(trace.records[2].input_tiles.size(), 2u);
    ASSERT_EQ(trace.records[2].output_tiles.size(), 1u);

    std::ostringstream json;
    pto::cpu_sim::DumpInstructionTraceJson(json);
    EXPECT_NE(json.str().find("\"opcode\":\"TADD\""), std::string::npos);
    EXPECT_NE(json.str().find("\"opcode\":\"TDIV\""), std::string::npos);
}

TEST_F(TTraceTest, CapturesInterleavedAndMultiOutputOperands)
{
    if (!pto::cpu_sim::kInstructionTraceEnabled) {
        GTEST_SKIP() << "PTO_CPU_SIM_TRACE_MODE is disabled for this build.";
    }

    using SrcTile = pto::Tile<pto::TileType::Vec, float, 4, 32>;
    using ValTile = pto::Tile<pto::TileType::Vec, float, 8, 1, pto::BLayout::ColMajor, 4, 1>;
    using IdxTile = pto::Tile<pto::TileType::Vec, uint32_t, 8, 1, pto::BLayout::ColMajor, 4, 1>;
    using TmpTile = pto::Tile<pto::TileType::Vec, uint32_t, 4, 32>;

    SrcTile src;
    ValTile dstVal;
    IdxTile dstIdx;
    TmpTile tmp;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, dstVal, dstIdx, tmp);
    FillLinear(src, 1.0f);
    pto::cpu_sim::ResetInstructionTrace();

    pto::TROWARGMAX(dstVal, dstIdx, src, tmp);

    const auto &trace = pto::cpu_sim::GetInstructionTrace();
    ASSERT_EQ(trace.records.size(), 1u);
    EXPECT_EQ(trace.records[0].opcode, "TROWARGMAX");
    ASSERT_EQ(trace.records[0].output_tiles.size(), 2u);
    ASSERT_EQ(trace.records[0].input_tiles.size(), 2u);
    EXPECT_EQ(trace.records[0].output_tiles[0].address, dstVal.GetAssignedAddress());
    EXPECT_EQ(trace.records[0].output_tiles[1].address, dstIdx.GetAssignedAddress());
    EXPECT_EQ(trace.records[0].input_tiles[0].address, src.GetAssignedAddress());
    EXPECT_EQ(trace.records[0].input_tiles[1].address, tmp.GetAssignedAddress());
}

TEST_F(TTraceTest, CapturesPointerOutputsForMxQuant)
{
    if (!pto::cpu_sim::kInstructionTraceEnabled) {
        GTEST_SKIP() << "PTO_CPU_SIM_TRACE_MODE is disabled for this build.";
    }

    using SrcTile = pto::Tile<pto::TileType::Vec, float, 16, 64>;
    using Fp8Tile = pto::Tile<pto::TileType::Vec, int8_t, 16, 64>;
    using ExpTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, 32, pto::BLayout::RowMajor, 1, 32>;
    using MaxTile = pto::Tile<pto::TileType::Vec, float, 1, 32, pto::BLayout::RowMajor, 1, 32>;

    SrcTile src;
    Fp8Tile dst;
    ExpTile exp;
    ExpTile expZz;
    MaxTile max;
    SrcTile scaling;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, dst, exp, expZz, max, scaling);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            src.data()[pto::GetTileElementOffset<SrcTile>(r, c)] = static_cast<float>((r + c) % 11) * 0.25f - 1.0f;
        }
    }
    pto::cpu_sim::ResetInstructionTrace();

    pto::TQUANT<pto::QuantType::MXFP8>(dst, src, &exp, &max, &scaling);
    pto::TQUANT<pto::QuantType::MXFP8, pto::VecStoreMode::NZ>(dst, src, &exp, &max, &scaling, &expZz);

    const auto &trace = pto::cpu_sim::GetInstructionTrace();
    ASSERT_EQ(trace.records.size(), 2u);

    EXPECT_EQ(trace.records[0].opcode, "TQUANT");
    ASSERT_EQ(trace.records[0].input_tiles.size(), 1u);
    ASSERT_EQ(trace.records[0].output_tiles.size(), 4u);
    EXPECT_EQ(trace.records[0].input_tiles[0].address, src.GetAssignedAddress());
    EXPECT_EQ(trace.records[0].output_tiles[0].address, dst.GetAssignedAddress());
    EXPECT_EQ(trace.records[0].output_tiles[1].address, exp.GetAssignedAddress());
    EXPECT_EQ(trace.records[0].output_tiles[2].address, max.GetAssignedAddress());
    EXPECT_EQ(trace.records[0].output_tiles[3].address, scaling.GetAssignedAddress());

    EXPECT_EQ(trace.records[1].opcode, "TQUANT");
    ASSERT_EQ(trace.records[1].input_tiles.size(), 1u);
    ASSERT_EQ(trace.records[1].output_tiles.size(), 5u);
    EXPECT_EQ(trace.records[1].output_tiles[4].address, expZz.GetAssignedAddress());
}

TEST_F(TTraceTest, WritesKernelTraceFile)
{
    if (!pto::cpu_sim::kInstructionTraceEnabled) {
        GTEST_SKIP() << "PTO_CPU_SIM_TRACE_MODE is disabled for this build.";
    }

    constexpr std::size_t kNumel = 4 * 32;
    std::vector<float> out(kNumel, 0.0f);
    std::vector<float> src0(kNumel, 0.0f);
    std::vector<float> src1(kNumel, 0.0f);
    for (std::size_t i = 0; i < kNumel; ++i) {
        src0[i] = static_cast<float>(i);
        src1[i] = static_cast<float>(2 * i);
    }

    pto::cpu_sim::ResetInstructionTrace();
    LaunchTraceKernel(out.data(), src0.data(), src1.data(), nullptr);

    const auto tracePath = std::filesystem::temp_directory_path() / "pto_cpu_sim_ttrace_kernel_trace.jsonl";
    std::ofstream outFile(tracePath, std::ios::trunc);
    ASSERT_TRUE(outFile.is_open());
    pto::cpu_sim::DumpInstructionTraceJson(outFile);
    outFile.close();

    std::ifstream inFile(tracePath);
    ASSERT_TRUE(inFile.is_open());
    std::stringstream contents;
    contents << inFile.rdbuf();
    const std::string text = contents.str();
    EXPECT_NE(text.find("\"opcode\":\"TLOAD\""), std::string::npos);
    EXPECT_NE(text.find("\"opcode\":\"TADD\""), std::string::npos);
    EXPECT_NE(text.find("\"opcode\":\"TSTORE\""), std::string::npos);
}

} // namespace
