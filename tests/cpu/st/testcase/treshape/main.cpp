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
#include <pto/pto-inst.hpp>

using namespace pto;

class TReshapeTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

TEST_F(TReshapeTest, AliasesBackingStorageInCpuSim)
{
    using SrcTile = Tile<TileType::Vec, float, 2, 16>;
    using DstTile = Tile<TileType::Vec, float, 1, 32>;

    SrcTile src;
    DstTile dst;

    TASSIGN(src, 0);
    TASSIGN(dst, 2 * 16 * sizeof(float));

    for (int i = 0; i < SrcTile::Numel; ++i) {
        src.data()[i] = static_cast<float>(i + 1);
    }

    TRESHAPE(dst, src);

    ASSERT_EQ(dst.data(), src.data());

    src.data()[17] = 123.0f;
    EXPECT_FLOAT_EQ(dst.data()[17], 123.0f);

    dst.data()[3] = -5.0f;
    EXPECT_FLOAT_EQ(src.data()[3], -5.0f);
}
