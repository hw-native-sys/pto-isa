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
#include <gtest/gtest.h>
#include <pto/common/fifo.hpp>
#include <vector>
#include "test_common.h"

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

template <typename T, int rows, int cols, TileType srcLoc>
void testPushPop()
{
    using PPDataFIFO = DataFIFO<T, FIFOType::GM_FIFO, 8, 1>;
    using PPSync = TFIFOSync<0, PPDataFIFO, TSyncOpType::TSTORE_C2GM, TSyncOpType::TLOAD>;
    using PPTile = Tile<srcLoc, T, rows, cols>;
    std::vector<T> fifoStorage(PPTile::Numel * PPDataFIFO::fifoDepth, static_cast<T>(0));
    PPDataFIFO gmFIFO(fifoStorage.data());
    PPTile src;
    PPTile dst;

    std::vector<T> expected(PPTile::Numel, static_cast<T>(0));

    for (int i = 0; i < src.Numel; i++) {
        src.data()[i] = static_cast<T>((i % 17) + 1);
        expected[i] = src.data()[i];
    }
    for (int i = 0; i < dst.Numel; i++) {
        dst.data()[i] = static_cast<T>(0);
    }

    typename PPSync::Producer prod;
    typename PPSync::Consumer cons;

    TPUSH(prod, src, gmFIFO);
    TPOP(cons, dst, gmFIFO);

    EXPECT_TRUE(ResultCmp(expected, dst.data(), 0));
}

class TPushPopTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

#define TPUSHPOP_TEST(T, rows, cols, srcLoc)             \
    TEST_F(TPushPopTest, T##_##rows##_##cols##_##srcLoc) \
    {                                                    \
        testPushPop<T, rows, cols, TileType::srcLoc>();  \
    }

TPUSHPOP_TEST(float, 64, 128, Vec)
TPUSHPOP_TEST(float, 128, 128, Vec)
TPUSHPOP_TEST(float, 64, 128, Mat)
TPUSHPOP_TEST(float, 128, 128, Mat)
TPUSHPOP_TEST(uint32_t, 64, 128, Vec)
TPUSHPOP_TEST(uint32_t, 128, 128, Vec)
TPUSHPOP_TEST(uint32_t, 64, 128, Mat)
TPUSHPOP_TEST(uint32_t, 128, 128, Mat)
