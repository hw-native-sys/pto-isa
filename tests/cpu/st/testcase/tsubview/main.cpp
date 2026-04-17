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
#include "test_common.h"
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace pto;
using namespace CpuTileTestUtils;
using namespace PtoTestCommon;

namespace {

TEST(TSubViewAliasTest, AliasesRowMajorSourceStorageAtOffset)
{
    using SrcTile = Tile<TileType::Vec, float, 4, 8>;
    using DstTile = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, 3, 6>;

    SrcTile src;
    DstTile dst;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, dst);

    FillLinear(src, 1.0f);
    TSUBVIEW(dst, src, 1, 2);

    ASSERT_EQ(dst.data(), src.data() + SrcTile::RowStride + 2 * SrcTile::ColStride);
    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            ExpectValueEquals(GetValue(dst, r, c), GetValue(src, r + 1, c + 2));
        }
    }

    SetValue(dst, 2, 5, 99.0f);
    ExpectValueEquals(GetValue(src, 3, 7), 99.0f);
}

TEST(TSubViewAliasTest, AliasesColMajorSourceStorageAtOffset)
{
    using SrcTile = Tile<TileType::Vec, float, 8, 8, BLayout::ColMajor>;
    using DstTile = Tile<TileType::Vec, float, 8, 8, BLayout::ColMajor, 2, 3>;

    SrcTile src;
    DstTile dst;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, dst);

    FillLinear(src, 10.0f);
    TSUBVIEW(dst, src, 1, 4);

    ASSERT_EQ(dst.data(), src.data() + SrcTile::RowStride + 4 * SrcTile::ColStride);
    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            ExpectValueEquals(GetValue(dst, r, c), GetValue(src, r + 1, c + 4));
        }
    }
}

} // namespace

class TSUBVIEWTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kSubRows_, int kSubCols_>
void LaunchTSubView(T *out, T *src, void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kSubRows_, int kSubCols_>
void test_tsubview()
{
    size_t srcFileSize = kGRows_ * kGCols_ * sizeof(T);
    size_t dstFileSize = kSubRows_ * kSubCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize));

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTSubView<T, kGRows_, kGCols_, kTRows_, kTCols_, kSubRows_, kSubCols_>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize);
    std::vector<T> devFinal(dstFileSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TSUBVIEWTest, case_float_64x64_64x64_32x32)
{
    test_tsubview<float, 64, 64, 64, 64, 32, 32>();
}

TEST_F(TSUBVIEWTest, case_int32_64x64_64x64_32x32)
{
    test_tsubview<int32_t, 64, 64, 64, 64, 32, 32>();
}

TEST_F(TSUBVIEWTest, case_int16_64x64_64x64_32x32)
{
    test_tsubview<int16_t, 64, 64, 64, 64, 32, 32>();
}

TEST_F(TSUBVIEWTest, case_half_16x256_16x256_8x128)
{
    test_tsubview<aclFloat16, 16, 256, 16, 256, 8, 128>();
}

#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TSUBVIEWTest, case_bf16_16x256_16x256_8x128)
{
    test_tsubview<bfloat16_t, 16, 256, 16, 256, 8, 128>();
}
#endif
