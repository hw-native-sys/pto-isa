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

template <int32_t tilingKey>
void LaunchTMATMUL(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

template <int32_t tilingKey>
void LaunchTMATMULBIAS(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);

class TMATMULTest : public testing::Test {
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

template <typename T, typename U, typename S, int32_t key>
void tmatmul_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(U);
    size_t bFileSize = K * N * sizeof(S);
    size_t cFileSize = M * N * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTMATMUL<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(cFileSize);
    std::vector<T> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.0001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMATMULTest, case1) { tmatmul_test<float, uint16_t, uint16_t, 1>(40, 50, 60); }

TEST_F(TMATMULTest, case2) { tmatmul_test<int32_t, int8_t, int8_t, 2>(6, 7, 8); }

TEST_F(TMATMULTest, case3)
{
    uint32_t M = 127;
    uint32_t N = 61;
    uint32_t K = 128;

    tmatmul_test<float, uint16_t, uint16_t, 3>(M, K, N);
}

TEST_F(TMATMULTest, case4) { tmatmul_test<float, float, float, 4>(120, 110, 50); }

TEST_F(TMATMULTest, case5) { tmatmul_test<float, uint16_t, uint16_t, 5>(144, 80, 48); }

TEST_F(TMATMULTest, case6) { tmatmul_test<float, uint8_t, uint8_t, 6>(32, 64, 96); }

TEST_F(TMATMULTest, case7) { tmatmul_test<float, uint8_t, uint8_t, 7>(128, 96, 64); }

TEST_F(TMATMULTest, case8) { tmatmul_test<float, uint8_t, uint8_t, 8>(145, 115, 85); }

TEST_F(TMATMULTest, case9) { tmatmul_test<float, uint8_t, uint8_t, 9>(120, 90, 160); }

TEST_F(TMATMULTest, case10) { tmatmul_test<float, uint8_t, uint8_t, 10>(30, 90, 60); }

TEST_F(TMATMULTest, case11) { tmatmul_test<float, uint16_t, uint16_t, 11>(1, 300, 60); }

TEST_F(TMATMULTest, case12) { tmatmul_test<float, float, float, 12>(16, 32, 64); }

TEST_F(TMATMULTest, case13) { tmatmul_test<float, float, float, 13>(128, 96, 64); }

template <typename T, typename U, typename S, typename B, int32_t key>
void tmatmul_bias_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(U);
    size_t bFileSize = K * N * sizeof(S);
    size_t cFileSize = M * N * sizeof(T);
    size_t biasFileSize = 1 * N * sizeof(B);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);
    aclrtMallocHost((void**)(&src2Host), biasFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src2Device, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/bias_gm.bin", biasFileSize, src2Host, biasFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, biasFileSize, src2Host, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTMATMULBIAS<key>(dstDevice, src0Device, src1Device, src2Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(src2Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(cFileSize);
    std::vector<T> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMATMULTest, case_bias_1) { tmatmul_bias_test<int32_t, int8_t, int8_t, int32_t, 1>(8, 7, 6); }

TEST_F(TMATMULTest, case_bias_2) { tmatmul_bias_test<float, uint16_t, uint16_t, uint16_t, 2>(16, 15, 16); }

TEST_F(TMATMULTest, case_bias_3) { tmatmul_bias_test<float, uint16_t, uint16_t, uint16_t, 3>(112, 127, 80); }

TEST_F(TMATMULTest, case_bias_4) { tmatmul_bias_test<float, uint16_t, uint16_t, uint16_t, 4>(80, 112, 63); }

TEST_F(TMATMULTest, case_bias_5)
{
    uint32_t M = 127;
    uint32_t N = 63;
    uint32_t K = 128;

    tmatmul_bias_test<float, float, float, float, 5>(M, K, N);
}

TEST_F(TMATMULTest, case_bias_6) { tmatmul_bias_test<float, uint8_t, uint8_t, float, 6>(120, 90, 160); }

TEST_F(TMATMULTest, case_bias_7) { tmatmul_bias_test<float, uint8_t, uint8_t, float, 7>(32, 64, 96); }

TEST_F(TMATMULTest, case_bias_8) { tmatmul_bias_test<float, uint8_t, uint8_t, float, 8>(128, 96, 64); }

TEST_F(TMATMULTest, case_bias_9) { tmatmul_bias_test<float, uint8_t, uint8_t, float, 9>(30, 90, 60); }

TEST_F(TMATMULTest, case_bias_10) { tmatmul_bias_test<float, uint8_t, uint8_t, float, 10>(145, 115, 85); }

TEST_F(TMATMULTest, case_bias_11) { tmatmul_bias_test<float, uint16_t, uint16_t, float, 11>(1, 512, 85); }
