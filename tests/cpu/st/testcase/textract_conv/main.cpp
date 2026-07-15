/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include "test_common.h"
#include "pto/pto-inst.hpp"
#include <gtest/gtest.h>

using namespace std;
using namespace pto;
using namespace PtoTestCommon;
using namespace pto;

template <
    typename T, size_t c1hw, size_t n1, size_t n0, size_t c0, size_t dst_row, size_t dst_col, size_t idxR, size_t idxC>
AICORE inline void runTEXTRACT_4D(__gm__ T* out, __gm__ T* src)
{
    static_assert(c0 == 32 / sizeof(T));

    constexpr size_t srcElemNum = c1hw * n1 * n0 * c0;
    constexpr size_t srcBufferSize = srcElemNum * sizeof(T);

    constexpr size_t dstElemNum = dst_row * dst_col;
    constexpr size_t dstBufferSize = dstElemNum * sizeof(T);

    using SrcShapeDim5 = Shape<1, 1, 1, 1, srcElemNum>;
    using SrcStrideDim5 = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;
    SrcTileData src0Tile;
    TASSIGN(src0Tile, 0x0);
    DstTileData dst0Tile;
    TASSIGN(dst0Tile, 0x0 + srcBufferSize);

    using SrcConvTile = ConvTile<TileType::Mat, T, srcBufferSize, Layout::FRACTAL_Z, ConvTileShape<c1hw, n1, n0, c0>>;
    using DstTileFractal =
        Tile<TileType::Right, T, dst_row, dst_col, BLayout::RowMajor, dst_row, dst_col, SLayout::ColMajor>;

    SrcConvTile srcTile;
    static_assert(srcTile.totalDimCount == 4);
    srcTile.data() = src0Tile.data();

    DstTileFractal dstTile;
    dstTile.data() = dst0Tile.data();

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TEXTRACT(dstTile, srcTile, idxR, idxC);
    TSTORE(dstGlobal, dst0Tile);
}

class TEXTRACTTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <
    typename T, size_t c1hw, size_t n1, size_t n0, size_t c0, size_t dst_row, size_t dst_col, size_t idxR, size_t idxC>
void textract_test_4d()
{
    size_t srcFileSize = c1hw * n1 * n0 * c0 * sizeof(T);
    size_t dstFileSize = dst_row * dst_col * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcHost;
    uint8_t *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t inputSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", inputSize, srcHost, srcFileSize));

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    runTEXTRACT_4D<T, c1hw, n1, n0, c0, dst_row, dst_col, idxR, idxC>((T*)dstDevice, (T*)srcDevice);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    size_t dstElem = dstFileSize / sizeof(T);
    std::vector<T> golden(dstElem);
    std::vector<T> devFinal(dstElem);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    EXPECT_TRUE(ret);
}

TEST_F(TEXTRACTTest, case_0) { textract_test_4d<aclFloat16, 4, 3, 16, 16, 3 * 16, 2 * 16, 16, 16>(); }

TEST_F(TEXTRACTTest, case_1) { textract_test_4d<uint16_t, 4, 3, 16, 16, 3 * 16, 2 * 16, 16, 16>(); }

TEST_F(TEXTRACTTest, case_2) { textract_test_4d<float, 4, 3, 16, 8, 3 * 8, 2 * 16, 8, 16>(); }

TEST_F(TEXTRACTTest, case_3) { textract_test_4d<int32_t, 4, 3, 16, 8, 3 * 8, 2 * 16, 8, 16>(); }
