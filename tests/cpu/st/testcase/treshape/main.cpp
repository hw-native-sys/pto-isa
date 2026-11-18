/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace pto;

class TReshapeTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TReshapeTest, AliasesBackingStorageInCpuSim)
{
    using SrcTile = Tile<TileType::Vec, float, 2, 16>;
    using DstTile = Tile<TileType::Vec, float, 1, 32>;

    SrcTile src;
    DstTile dst;
    TASSIGN(src, 0);
    TASSIGN(dst, SrcTile::GetSizeInBytes());

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

TEST_F(TReshapeTest, FloatViewsShareByteStorage)
{
    using Bytes = Tile<TileType::Vec, unsigned char, 2, 32>;
    using Flat = Tile<TileType::Vec, float, 1, 16>;
    using Matrix = Tile<TileType::Vec, float, 2, 8>;

    Bytes bytes;
    Flat flat;
    Matrix matrix;
    TASSIGN(bytes, 0);
    TASSIGN(flat, Bytes::StorageBytes);
    TASSIGN(matrix, Bytes::StorageBytes + Flat::StorageBytes);

    std::array<float, Flat::Numel> expected;
    for (int i = 0; i < Flat::Numel; ++i) {
        expected[i] = static_cast<float>(i) + 0.5f;
    }
    std::memcpy(bytes.data(), expected.data(), Bytes::StorageBytes);

    TRESHAPE(flat, bytes);
    TRESHAPE(matrix, bytes);
    ASSERT_EQ(static_cast<void*>(flat.data()), static_cast<void*>(bytes.data()));
    ASSERT_EQ(matrix.data(), flat.data());
    EXPECT_EQ(std::memcmp(flat.data(), expected.data(), Bytes::StorageBytes), 0);

    matrix.data()[Matrix::Numel - 1] = -0.0f;
    expected.back() = -0.0f;
    EXPECT_EQ(std::memcmp(bytes.data(), expected.data(), Bytes::StorageBytes), 0);
}

TEST_F(TReshapeTest, PackedFp4AliasesItsPhysicalBytes)
{
    using Packed = Tile<TileType::Vec, float4_e2m1x2_t, 1, 64>;
    using Bytes = Tile<TileType::Vec, unsigned char, 1, 32>;

    Packed packed;
    Bytes bytes;
    TASSIGN(packed, 0);
    TASSIGN(bytes, Packed::StorageBytes);

    TRESHAPE(bytes, packed);
    ASSERT_EQ(static_cast<void*>(bytes.data()), static_cast<void*>(packed.data()));
    packed.data()[Packed::GetSizeInUnits() - 1] = float4_e2m1x2_t::FromRaw(0xa5);
    EXPECT_EQ(bytes.data()[Bytes::Numel - 1], 0xa5);
}
