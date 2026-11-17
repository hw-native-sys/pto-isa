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
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void launchTRems_demo(uint8_t* out, uint8_t* src, void* stream);

class TREMSTest : public testing::Test {
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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTRems(T* out, T* src, T* scalar, void* stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_trems()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);
    size_t scalarSize = sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost, *scalarHost;
    T *dstDevice, *srcDevice, *scalarDevice;
    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&srcHost), fileSize);
    aclrtMallocHost((void**)(&scalarHost), scalarSize);

    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&scalarDevice, scalarSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(dstDevice, fileSize, 0, fileSize);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", fileSize, srcHost, fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/scalar.bin", scalarSize, scalarHost, scalarSize));
    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(scalarDevice, scalarSize, scalarHost, scalarSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTRems<T, kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, srcDevice, scalarDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(scalarDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(scalarHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize / sizeof(T));
    std::vector<T> devFinal(fileSize / sizeof(T));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TREMSTest, case_float_64x64_64x64) { test_trems<float, NUM_64, NUM_64, NUM_64, NUM_64>(); }
TEST_F(TREMSTest, case_int32_64x64_64x64) { test_trems<int32_t, NUM_64, NUM_64, NUM_64, NUM_64>(); }
TEST_F(TREMSTest, case_int64_64x64_64x64) { test_trems<int64_t, NUM_64, NUM_64, NUM_64, NUM_64>(); }
TEST_F(TREMSTest, case_uint64_64x64_64x64) { test_trems<uint64_t, NUM_64, NUM_64, NUM_64, NUM_64>(); }
TEST_F(TREMSTest, case_int16_64x64_64x64) { test_trems<int16_t, NUM_64, NUM_64, NUM_64, NUM_64>(); }
TEST_F(TREMSTest, case_half_16x256_16x256) { test_trems<aclFloat16, NUM_16, NUM_256, NUM_16, NUM_256>(); }

TEST_F(TREMSTest, case_float_64x512_64x64) { test_trems<float, NUM_64, NUM_512, NUM_64, NUM_64>(); }
TEST_F(TREMSTest, case_int32_64x512_64x64) { test_trems<int32_t, NUM_64, NUM_512, NUM_64, NUM_64>(); }
TEST_F(TREMSTest, case_int16_64x512_64x64) { test_trems<int16_t, NUM_64, NUM_512, NUM_64, NUM_64>(); }
TEST_F(TREMSTest, case_half_32x512_16x256) { test_trems<aclFloat16, NUM_32, NUM_512, NUM_16, NUM_256>(); }
#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TREMSTest, case_bf16_16x256_16x256) { test_trems<bfloat16_t, NUM_16, NUM_256, NUM_16, NUM_256>(); }
TEST_F(TREMSTest, case_bf16_32x256_16x256) { test_trems<bfloat16_t, NUM_32, NUM_256, NUM_16, NUM_256>(); }
#endif

template <typename T, int ScratchCols>
void TestIndependentScratch()
{
    using DataTile = Tile<TileType::Vec, T, 1, 64>;
    using ScratchTile = Tile<TileType::Vec, T, 1, ScratchCols>;
    using GlobalData = GlobalTensor<T, Shape<1, 1, 1, 1, 64>, Stride<64, 64, 64, 64, 1>>;
    const T values[8] = {-7, -6, -1, 0, 1, 6, 7, 17};
    const T positive[8] = {2, 0, 2, 0, 1, 0, 1, 2};
    const T negative[8] = {-1, 0, -1, 0, -2, 0, -2, -1};
    T input[64], output[64] = {}, preserved[64] = {};
    for (int i = 0; i < 64; ++i) {
        input[i] = values[i % 8];
    }
    DataTile src, dst;
    ScratchTile tmp;
    TASSIGN(src, 0);
    TASSIGN(dst, 256);
    TASSIGN(tmp, 512);
    GlobalData source(input), destination(output), sourceAfter(preserved);
    TLOAD(src, source);
    for (T scalar : {T{3}, T{-3}}) {
        TREMS(dst, src, scalar, tmp);
        TSTORE(destination, dst);
        TSTORE(sourceAfter, src);
        for (int i = 0; i < 64; ++i) {
            EXPECT_EQ(output[i], scalar > 0 ? positive[i % 8] : negative[i % 8]);
            EXPECT_EQ(preserved[i], input[i]);
        }
    }
}

TEST_F(TREMSTest, IndependentInt32Scratch) { TestIndependentScratch<int32_t, 144>(); }
TEST_F(TREMSTest, IndependentFloatScratch) { TestIndependentScratch<float, 80>(); }
