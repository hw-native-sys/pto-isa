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
#include "test_common.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TCONCATTest : public testing::Test {
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

struct TilesSize {
    int dstH;
    int dstW;
    int src0H;
    int src0W;
    int src1H;
    int src1W;
    int vRows;
    int vCols0;
    int vCols1;
};

template <typename T, TilesSize sizes>
void LaunchTConcat(T *out, T *src0, T *src1, void *stream);

template <typename T, typename TIdx, TilesSize sizes, TilesSize idxSizes>
void LaunchTConcat(T *out, T *src0, T *src1, TIdx *src0Idx, TIdx *src1Idx, void *stream);

template <typename T, typename TIdx, TilesSize sizes, TilesSize idxSizes = sizes>
void LaunchTConcat(T *out, T *src0, T *src1, TIdx *outIdx, TIdx *src0Idx, TIdx *src1Idx, void *stream);

constexpr int WITHOUT_IDX_TILES = 0;
constexpr int USE_SRC_IDX = 1;
constexpr int USE_DST_IDX = 2;

template <typename T, TilesSize sizes, typename TIdx = int, int useIdx = WITHOUT_IDX_TILES, TilesSize idxSizes = sizes>
void test_tconcat()
{
    size_t fileSizeDst = sizes.dstH * sizes.dstW * sizeof(T);
    size_t fileSizeSrc0 = sizes.src0H * sizes.src0W * sizeof(T);
    size_t fileSizeSrc1 = sizes.src1H * sizes.src1W * sizeof(T);
    size_t fileSizeDstIdx = idxSizes.dstH * idxSizes.dstW * sizeof(TIdx);
    size_t fileSizeSrc0Idx = idxSizes.src0H * idxSizes.src0W * sizeof(TIdx);
    size_t fileSizeSrc1Idx = idxSizes.src1H * idxSizes.src1W * sizeof(TIdx);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;
    TIdx *dstIdxHost, *src0IdxHost, *src1IdxHost;
    TIdx *dstIdxDevice, *src0IdxDevice, *src1IdxDevice;

    aclrtMallocHost((void **)(&dstHost), fileSizeDst);
    aclrtMallocHost((void **)(&src0Host), fileSizeSrc0);
    aclrtMallocHost((void **)(&src1Host), fileSizeSrc1);

    if constexpr (useIdx == USE_DST_IDX) {
        aclrtMallocHost((void **)(&dstIdxHost), fileSizeDstIdx);
    }
    if constexpr (useIdx >= USE_SRC_IDX) {
        aclrtMallocHost((void **)(&src0IdxHost), fileSizeSrc0Idx);
        aclrtMallocHost((void **)(&src1IdxHost), fileSizeSrc1Idx);
    }

    std::fill(dstHost, dstHost + sizes.dstH * sizes.dstW, 0u);

    if constexpr (useIdx == USE_DST_IDX) {
        std::fill(dstIdxHost, dstIdxHost + idxSizes.dstH * idxSizes.dstW, 0u);
    }

    aclrtMalloc((void **)&dstDevice, fileSizeDst, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSizeSrc0, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, fileSizeSrc1, ACL_MEM_MALLOC_HUGE_FIRST);

    if constexpr (useIdx == USE_DST_IDX) {
        aclrtMalloc((void **)&dstIdxDevice, fileSizeDstIdx, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if constexpr (useIdx >= USE_SRC_IDX) {
        aclrtMalloc((void **)&src0IdxDevice, fileSizeSrc0Idx, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&src1IdxDevice, fileSizeSrc1Idx, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", fileSizeSrc0, src0Host, fileSizeSrc0));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2.bin", fileSizeSrc1, src1Host, fileSizeSrc1));
    if constexpr (useIdx >= USE_SRC_IDX) {
        CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1_idx.bin", fileSizeSrc0Idx, src0IdxHost, fileSizeSrc0Idx));
        CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2_idx.bin", fileSizeSrc1Idx, src1IdxHost, fileSizeSrc1Idx));
    }

    aclrtMemcpy(src0Device, fileSizeSrc0, src0Host, fileSizeSrc0, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSizeSrc1, src1Host, fileSizeSrc1, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dstDevice, fileSizeDst, dstHost, fileSizeDst, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (useIdx >= USE_SRC_IDX) {
        aclrtMemcpy(src0IdxDevice, fileSizeSrc0Idx, src0IdxHost, fileSizeSrc0Idx, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src1IdxDevice, fileSizeSrc1Idx, src1IdxHost, fileSizeSrc1Idx, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if constexpr (useIdx == USE_DST_IDX) {
        aclrtMemcpy(dstIdxDevice, fileSizeDstIdx, dstIdxHost, fileSizeDstIdx, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    if constexpr (useIdx == USE_DST_IDX) {
        LaunchTConcat<T, TIdx, sizes, idxSizes>(dstDevice, src0Device, src1Device, dstIdxDevice, src0IdxDevice,
                                                src1IdxDevice, stream);
    } else if constexpr (useIdx == USE_SRC_IDX) {
        LaunchTConcat<T, TIdx, sizes, idxSizes>(dstDevice, src0Device, src1Device, src0IdxDevice, src1IdxDevice,
                                                stream);
    } else {
        LaunchTConcat<T, sizes>(dstDevice, src0Device, src1Device, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSizeDst, dstDevice, fileSizeDst, ACL_MEMCPY_DEVICE_TO_HOST);
    if constexpr (useIdx == USE_DST_IDX) {
        aclrtMemcpy(dstIdxHost, fileSizeDstIdx, dstIdxDevice, fileSizeDstIdx, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSizeDst);
    if constexpr (useIdx == USE_DST_IDX) {
        WriteFile(GetGoldenDir() + "/output_idx.bin", dstIdxHost, fileSizeDstIdx);
    }

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    if constexpr (useIdx == USE_DST_IDX) {
        aclrtFree(dstIdxDevice);
        aclrtFreeHost(dstIdxHost);
    }
    if constexpr (useIdx >= USE_SRC_IDX) {
        aclrtFree(src0IdxDevice);
        aclrtFree(src1IdxDevice);
        aclrtFreeHost(src0IdxHost);
        aclrtFreeHost(src1IdxHost);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSizeDst);
    std::vector<T> devFinal(fileSizeDst);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSizeDst, golden.data(), fileSizeDst);
    ReadFile(GetGoldenDir() + "/output.bin", fileSizeDst, devFinal.data(), fileSizeDst);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);
    ASSERT_TRUE(ret);

    if constexpr (useIdx == USE_DST_IDX) {
        std::vector<TIdx> goldenIdx(fileSizeDstIdx);
        std::vector<TIdx> devFinalIdx(fileSizeDstIdx);
        ReadFile(GetGoldenDir() + "/golden_idx.bin", fileSizeDstIdx, goldenIdx.data(), fileSizeDstIdx);
        ReadFile(GetGoldenDir() + "/output_idx.bin", fileSizeDstIdx, devFinalIdx.data(), fileSizeDstIdx);

        bool retIdx = ResultCmp<TIdx>(goldenIdx, devFinalIdx, 0.001f);
        ASSERT_TRUE(retIdx);
    }
}

TEST_F(TCONCATTest, case_float_64x128_64x64_64x64_64x64_64x64)
{
    test_tconcat<float, TilesSize{64, 128, 64, 64, 64, 64, 64, 64, 64}>();
}

TEST_F(TCONCATTest, case_int32_64x128_64x64_64x64_64x64_64x64)
{
    test_tconcat<int32_t, TilesSize{64, 128, 64, 64, 64, 64, 64, 64, 64}>();
}

TEST_F(TCONCATTest, case_half_16x256_16x128_16x128_16x128_16x128)
{
    test_tconcat<aclFloat16, TilesSize{16, 256, 16, 128, 16, 128, 16, 128, 128}>();
}

TEST_F(TCONCATTest, case_float_16x64_16x32_16x32_16x32_16x32)
{
    test_tconcat<float, TilesSize{16, 64, 16, 32, 16, 32, 16, 32, 32}>();
}

TEST_F(TCONCATTest, case_int16_32x256_32x128_32x128_32x128_32x128)
{
    test_tconcat<int16_t, TilesSize{32, 256, 32, 128, 32, 128, 32, 128, 128}>();
}

TEST_F(TCONCATTest, case_half_16x128_16x64_16x64_16x63_16x64)
{
    test_tconcat<aclFloat16, TilesSize{16, 128, 16, 64, 16, 64, 16, 63, 64}>();
}

TEST_F(TCONCATTest, case_float_16x64_16x32_16x32_16x31_16x32)
{
    test_tconcat<float, TilesSize{16, 64, 16, 32, 16, 32, 16, 31, 32}>();
}

TEST_F(TCONCATTest, case_int16_32x256_32x128_32x128_32x127_32x128)
{
    test_tconcat<int16_t, TilesSize{32, 256, 32, 128, 32, 128, 32, 127, 128}>();
}

// idx

TEST_F(TCONCATTest, case_float_int32_src_64x128_64x64_64x64_64x64_64x64_64x128_64x64_64x64)
{
    test_tconcat<float, TilesSize{64, 128, 64, 64, 64, 64, 64, 64, 64}, int32_t, USE_SRC_IDX>();
}

TEST_F(TCONCATTest, case_int32_int32_src_64x128_64x64_64x64_64x64_64x64_64x128_64x64_64x64)
{
    test_tconcat<int32_t, TilesSize{64, 128, 64, 64, 64, 64, 64, 64, 64}, int32_t, USE_SRC_IDX>();
}

TEST_F(TCONCATTest, case_half_int32_src_16x256_16x128_16x128_16x128_16x128_16x256_16x128_16x128)
{
    test_tconcat<aclFloat16, TilesSize{16, 256, 16, 128, 16, 128, 16, 128, 128}, int32_t, USE_SRC_IDX>();
}

TEST_F(TCONCATTest, case_float_int32_src_16x64_16x32_16x32_16x32_16x32_16x64_16x32_16x32)
{
    test_tconcat<float, TilesSize{16, 64, 16, 32, 16, 32, 16, 32, 32}, int32_t, USE_SRC_IDX>();
}

TEST_F(TCONCATTest, case_int16_int32_src_32x256_32x128_32x128_32x128_32x128_32x256_32x128_32x128)
{
    test_tconcat<int16_t, TilesSize{32, 256, 32, 128, 32, 128, 32, 128, 128}, int32_t, USE_SRC_IDX>();
}

TEST_F(TCONCATTest, case_half_int32_src_16x128_16x64_16x64_16x63_16x64_16x128_16x64_16x64)
{
    test_tconcat<aclFloat16, TilesSize{16, 128, 16, 64, 16, 64, 16, 63, 64}, int32_t, USE_SRC_IDX>();
}

TEST_F(TCONCATTest, case_float_int32_src_16x64_16x32_16x32_16x31_16x32_16x64_16x32_16x32)
{
    test_tconcat<float, TilesSize{16, 64, 16, 32, 16, 32, 16, 31, 32}, int32_t, USE_SRC_IDX>();
}

TEST_F(TCONCATTest, case_int16_int32_src_32x256_32x128_32x128_32x127_32x128_32x256_32x128_32x128)
{
    test_tconcat<int16_t, TilesSize{32, 256, 32, 128, 32, 128, 32, 127, 128}, int32_t, USE_SRC_IDX>();
}

// dstidx

TEST_F(TCONCATTest, case_float_int32_dst_64x128_64x64_64x64_64x64_64x64_64x128_64x64_64x64)
{
    test_tconcat<float, TilesSize{64, 128, 64, 64, 64, 64, 64, 64, 64}, int32_t, USE_DST_IDX>();
}

TEST_F(TCONCATTest, case_int32_int32_dst_64x128_64x64_64x64_64x64_64x64_64x128_64x64_64x64)
{
    test_tconcat<int32_t, TilesSize{64, 128, 64, 64, 64, 64, 64, 64, 64}, int32_t, USE_DST_IDX>();
}

TEST_F(TCONCATTest, case_half_int32_dst_16x256_16x128_16x128_16x128_16x128_16x256_16x128_16x128)
{
    test_tconcat<aclFloat16, TilesSize{16, 256, 16, 128, 16, 128, 16, 128, 128}, int32_t, USE_DST_IDX>();
}

TEST_F(TCONCATTest, case_float_int32_dst_16x64_16x32_16x32_16x32_16x32_16x64_16x32_16x32)
{
    test_tconcat<float, TilesSize{16, 64, 16, 32, 16, 32, 16, 32, 32}, int32_t, USE_DST_IDX>();
}

TEST_F(TCONCATTest, case_int16_int32_dst_32x256_32x128_32x128_32x128_32x128_32x256_32x128_32x128)
{
    test_tconcat<int16_t, TilesSize{32, 256, 32, 128, 32, 128, 32, 128, 128}, int32_t, USE_DST_IDX>();
}

TEST_F(TCONCATTest, case_half_int32_dst_16x128_16x64_16x64_16x63_16x64_16x128_16x64_16x64)
{
    test_tconcat<aclFloat16, TilesSize{16, 128, 16, 64, 16, 64, 16, 63, 64}, int32_t, USE_DST_IDX>();
}

TEST_F(TCONCATTest, case_float_int32_dst_16x64_16x32_16x32_16x31_16x32_16x64_16x32_16x32)
{
    test_tconcat<float, TilesSize{16, 64, 16, 32, 16, 32, 16, 31, 32}, int32_t, USE_DST_IDX>();
}

TEST_F(TCONCATTest, case_int16_int32_dst_32x256_32x128_32x128_32x127_32x128_32x256_32x128_32x128)
{
    test_tconcat<int16_t, TilesSize{32, 256, 32, 128, 32, 128, 32, 127, 128}, int32_t, USE_DST_IDX>();
}
