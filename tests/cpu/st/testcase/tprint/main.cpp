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

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

#include <pto/pto-inst.hpp>

using namespace pto;

class TPrintTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

template <typename Func>
static std::string CaptureStdout(Func&& func)
{
    std::ostringstream oss;
    auto* oldBuf = std::cout.rdbuf();
    std::cout.rdbuf(oss.rdbuf());
    func();
    std::cout.rdbuf(oldBuf);
    return oss.str();
}

TEST_F(TPrintTest, FloatFormats)
{
    using TileT = Tile<TileType::Vec, float, 1, 8>;
    TileT tile;
    TASSIGN(tile, 0);
    tile.data()[0] = 1.234567f;
    tile.data()[1] = -3.456789f;
    const std::string outputDefault = CaptureStdout([&]() { TPRINT(tile); });
    const std::string outputPrecision2 = CaptureStdout([&]() { TPRINT<PrintFormat::Width8_Precision2>(tile); });
    const std::string outputPrecision6 = CaptureStdout([&]() { TPRINT<PrintFormat::Width10_Precision6>(tile); });

    EXPECT_NE(outputDefault.find("1.2346"), std::string::npos);
    EXPECT_NE(outputDefault.find("-3.4568"), std::string::npos);
    EXPECT_NE(outputPrecision2.find("1.23"), std::string::npos);
    EXPECT_NE(outputPrecision2.find("-3.46"), std::string::npos);
    EXPECT_NE(outputPrecision6.find("1.234567"), std::string::npos);
    EXPECT_NE(outputPrecision6.find("-3.456789"), std::string::npos);
}

TEST_F(TPrintTest, SignedIntFormats)
{
    using TileT = Tile<TileType::Vec, int32_t, 1, 8>;
    TileT tile;
    TASSIGN(tile, 0);
    tile.data()[0] = 42;
    tile.data()[1] = -17;
    tile.data()[2] = 1024;
    tile.data()[3] = -9999;
    const std::string outputWidth8 = CaptureStdout([&]() { TPRINT(tile); });
    const std::string outputWidth10 = CaptureStdout([&]() { TPRINT<PrintFormat::Width10_Precision6>(tile); });

    EXPECT_NE(outputWidth8.find("42"), std::string::npos);
    EXPECT_NE(outputWidth8.find("-17"), std::string::npos);
    EXPECT_NE(outputWidth8.find("1024"), std::string::npos);
    EXPECT_NE(outputWidth8.find("-9999"), std::string::npos);
    EXPECT_NE(outputWidth10.find("42"), std::string::npos);
    EXPECT_NE(outputWidth10.find("-17"), std::string::npos);
    EXPECT_NE(outputWidth10.find("1024"), std::string::npos);
    EXPECT_NE(outputWidth10.find("-9999"), std::string::npos);
}

TEST_F(TPrintTest, UnsignedIntFormats)
{
    using TileT = Tile<TileType::Vec, uint32_t, 1, 8>;
    TileT tile;
    TASSIGN(tile, 0);
    tile.data()[0] = 0;
    tile.data()[1] = 17;
    tile.data()[2] = 65535;
    tile.data()[3] = 123456;
    const std::string output = CaptureStdout([&]() { TPRINT(tile); });

    EXPECT_NE(output.find("0"), std::string::npos);
    EXPECT_NE(output.find("17"), std::string::npos);
    EXPECT_NE(output.find("65535"), std::string::npos);
    EXPECT_NE(output.find("123456"), std::string::npos);
}

TEST_F(TPrintTest, Int8AndUint8PrintedAsNumbers)
{
    using Int8Tile = Tile<TileType::Vec, int8_t, 1, 32>;
    using UInt8Tile = Tile<TileType::Vec, uint8_t, 1, 32>;
    Int8Tile int8Tile;
    UInt8Tile uint8Tile;
    TASSIGN(int8Tile, 0);
    TASSIGN(uint8Tile, Int8Tile::Numel * sizeof(int8_t));
    int8Tile.data()[0] = static_cast<int8_t>(-12);
    int8Tile.data()[1] = static_cast<int8_t>(65);
    uint8Tile.data()[0] = static_cast<uint8_t>(255);
    uint8Tile.data()[1] = static_cast<uint8_t>(127);
    const std::string int8Output = CaptureStdout([&]() { TPRINT(int8Tile); });
    const std::string uint8Output = CaptureStdout([&]() { TPRINT(uint8Tile); });

    EXPECT_NE(int8Output.find("-12"), std::string::npos);
    EXPECT_NE(int8Output.find("65"), std::string::npos);
    EXPECT_NE(uint8Output.find("255"), std::string::npos);
    EXPECT_NE(uint8Output.find("127"), std::string::npos);
}

TEST_F(TPrintTest, TileShapeHeaderIsPrinted)
{
    using TileT = Tile<TileType::Vec, float, 2, 8>;
    TileT tile;
    TASSIGN(tile, 0);
    const std::string output = CaptureStdout([&]() { TPRINT(tile); });

    EXPECT_NE(output.find("TPRINT 2x8"), std::string::npos);
}

TEST_F(TPrintTest, OverloadWithTempBuf)
{
    using TileT = Tile<TileType::Vec, float, 1, 8>;
    TileT tile;
    TileT tmp;
    TASSIGN(tile, 0);
    TASSIGN(tmp, TileT::Numel * sizeof(float));
    tile.data()[0] = 3.141592f;
    const std::string output = CaptureStdout([&]() { TPRINT<PrintFormat::Width10_Precision6>(tile, tmp); });

    EXPECT_NE(output.find("3.141592"), std::string::npos);
}
