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
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int kRows, int kCols>
void launchTMOV_nd2nz(uint8_t* out, uint8_t* src, void* stream);

// fp4 kind selected by int tag (defined in the kernel TU): 0 = e1m2, 1 = e2m1.
// The host driver never names a fp4 C++ type (it lives in a __CPU_SIM-only header).
constexpr int kFp4E1M2 = 0;
constexpr int kFp4E2M1 = 1;
template <int kFp4Kind, int kRows, int kCols>
void launchTMOV_nd2nz_f4(uint8_t* out, uint8_t* src, void* stream);

class TMovNd2NzTest : public testing::Test {
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

template <int kRows, int kCols>
void test_tmov_nd2nz()
{
    size_t inputSize = kRows * kCols * sizeof(uint8_t);
    size_t outputSize = kRows * kCols * sizeof(uint8_t); // NZ total == ND total for aligned sizes

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *dstDevice;
    uint8_t *srcHost, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), outputSize);
    aclrtMallocHost((void**)(&srcHost), inputSize);
    aclrtMalloc((void**)(&dstDevice), outputSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)(&srcDevice), inputSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", inputSize, srcHost, inputSize);
    aclrtMemset(dstDevice, outputSize, 0, outputSize);

    aclrtMemcpy(srcDevice, inputSize, srcHost, inputSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOV_nd2nz<kRows, kCols>(dstDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputSize, dstDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, outputSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<uint8_t> golden(outputSize);
    std::vector<uint8_t> devFinal(outputSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputSize, golden.data(), outputSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", outputSize, devFinal.data(), outputSize);

    bool ret = ResultCmp<uint8_t>(golden, devFinal, 0.0f);
    EXPECT_TRUE(ret);
}

template <int kFp4Kind, int kRows, int kCols>
void test_tmov_nd2nz_f4()
{
    constexpr int c0 = 64;                                  // f4 nibbles per NZ panel row (32 bytes)
    constexpr int alignedCols = (kCols + c0 - 1) / c0 * c0; // padded to full NZ panels
    size_t inputSize = kRows * kCols / 2;                   // dense GM feed, 2 nibbles/byte
    size_t outputSize = kRows * alignedCols / 2;            // NZ over the padded width

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *dstDevice;
    uint8_t *srcHost, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), outputSize);
    aclrtMallocHost((void**)(&srcHost), inputSize);
    aclrtMalloc((void**)(&dstDevice), outputSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)(&srcDevice), inputSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", inputSize, srcHost, inputSize);
    aclrtMemset(dstDevice, outputSize, 0, outputSize);

    aclrtMemcpy(srcDevice, inputSize, srcHost, inputSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOV_nd2nz_f4<kFp4Kind, kRows, kCols>(dstDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputSize, dstDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, outputSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<uint8_t> golden(outputSize);
    std::vector<uint8_t> devFinal(outputSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputSize, golden.data(), outputSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", outputSize, devFinal.data(), outputSize);

    bool ret = ResultCmp<uint8_t>(golden, devFinal, 0.0f);
    EXPECT_TRUE(ret);
}

TEST_F(TMovNd2NzTest, case_hif8_32x32) { test_tmov_nd2nz<32, 32>(); }

TEST_F(TMovNd2NzTest, case_hif8_32x64) { test_tmov_nd2nz<32, 64>(); }

TEST_F(TMovNd2NzTest, case_hif8_64x64) { test_tmov_nd2nz<64, 64>(); }

TEST_F(TMovNd2NzTest, case_f4e1m2_16x8160) { test_tmov_nd2nz_f4<kFp4E1M2, 16, 8160>(); }

// 12 fp4 regression cases: 6 sizes × {e1m2, e2m1}
TEST_F(TMovNd2NzTest, case_f4e1m2_16x32) { test_tmov_nd2nz_f4<kFp4E1M2, 16, 32>(); }
TEST_F(TMovNd2NzTest, case_f4e1m2_16x8160b) { test_tmov_nd2nz_f4<kFp4E1M2, 16, 8160>(); }
TEST_F(TMovNd2NzTest, case_f4e1m2_4080x32) { test_tmov_nd2nz_f4<kFp4E1M2, 4080, 32>(); }
TEST_F(TMovNd2NzTest, case_f4e1m2_32x32) { test_tmov_nd2nz_f4<kFp4E1M2, 32, 32>(); }
TEST_F(TMovNd2NzTest, case_f4e1m2_32x64) { test_tmov_nd2nz_f4<kFp4E1M2, 32, 64>(); }
TEST_F(TMovNd2NzTest, case_f4e1m2_64x64) { test_tmov_nd2nz_f4<kFp4E1M2, 64, 64>(); }
TEST_F(TMovNd2NzTest, case_f4e2m1_16x32) { test_tmov_nd2nz_f4<kFp4E2M1, 16, 32>(); }
TEST_F(TMovNd2NzTest, case_f4e2m1_16x8160) { test_tmov_nd2nz_f4<kFp4E2M1, 16, 8160>(); }
TEST_F(TMovNd2NzTest, case_f4e2m1_4080x32) { test_tmov_nd2nz_f4<kFp4E2M1, 4080, 32>(); }
TEST_F(TMovNd2NzTest, case_f4e2m1_32x32) { test_tmov_nd2nz_f4<kFp4E2M1, 32, 32>(); }
TEST_F(TMovNd2NzTest, case_f4e2m1_32x64) { test_tmov_nd2nz_f4<kFp4E2M1, 32, 64>(); }
TEST_F(TMovNd2NzTest, case_f4e2m1_64x64) { test_tmov_nd2nz_f4<kFp4E2M1, 64, 64>(); }
