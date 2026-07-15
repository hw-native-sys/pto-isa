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

template <int32_t tilingKey, int32_t format>
void LaunchTMOV_MX(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, void* stream);

class TMOVMXTest : public testing::Test {
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

template <typename T>
const T CeilDiv(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2;
}

template <typename T>
const T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename T, typename U, typename S, bool isFp4, int32_t key, int format>
void TmovMXTest(
    uint32_t validM, uint32_t validK, uint32_t validN, uint16_t indexM, uint16_t indexK, uint16_t indexN,
    uint16_t baseM = 0, uint16_t baseK = 0, uint16_t baseN = 0)
{
    uint32_t kAlign = CeilAlign<uint32_t>(validK, 64);
    size_t aFileSize = isFp4 ? CeilDiv<uint32_t>(validM * validK, 2) : validM * validK * sizeof(U);
    size_t bFileSize = isFp4 ? CeilDiv<uint32_t>(validK * validN, 2) : validK * validN * sizeof(S);
    size_t aScaleFileSize = validM * CeilDiv<uint32_t>(kAlign, 32);
    size_t bScaleFileSize = validN * CeilDiv<uint32_t>(kAlign, 32);

    if (baseM != 0) { // compact cases
        aFileSize = isFp4 ? CeilDiv<uint32_t>(baseM * baseK, 2) : baseM * baseK * sizeof(U);
        bFileSize = isFp4 ? CeilDiv<uint32_t>(baseK * baseN, 2) : baseK * baseN * sizeof(S);
        aScaleFileSize = baseM * CeilDiv<uint32_t>(baseK, 32);
        bScaleFileSize = baseN * CeilDiv<uint32_t>(baseK, 32);
    }
    size_t cFileSize = (validM - indexM) * (validN - indexN) * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host, *src3Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device, *src3Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);
    aclrtMallocHost((void**)(&src2Host), aScaleFileSize);
    aclrtMallocHost((void**)(&src3Host), bScaleFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src2Device, aScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src3Device, bScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/x1_mx_gm.bin", aScaleFileSize, src2Host, aScaleFileSize);
    ReadFile(GetGoldenDir() + "/x2_mx_gm.bin", bScaleFileSize, src3Host, bScaleFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, aScaleFileSize, src2Host, aScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src3Device, bScaleFileSize, src3Host, bScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTMOV_MX<key, format>(dstDevice, src0Device, src1Device, src2Device, src3Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(src2Device);
    aclrtFree(src3Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtFreeHost(src3Host);

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

TEST_F(TMOVMXTest, case1)
{
    uint32_t M = 128;
    uint32_t K = 64;
    uint32_t N = 64;

    TmovMXTest<float, uint8_t, uint8_t, false, 1, 0>(M, K, N, 0, 0, 0);
}

TEST_F(TMOVMXTest, case2)
{
    uint32_t M = 32;
    uint32_t K = 128;
    uint32_t N = 64;

    TmovMXTest<float, uint8_t, uint8_t, true, 2, 0>(M, K, N, 0, 0, 0);
}

TEST_F(TMOVMXTest, case3)
{
    uint32_t M = 64;
    uint32_t K = 128;
    uint32_t N = 80;

    TmovMXTest<float, uint8_t, uint8_t, false, 3, 0>(M, K, N, 0, 0, 0);
}

TEST_F(TMOVMXTest, case4)
{
    uint32_t M = 115;
    uint32_t K = 64;
    uint32_t N = 30;

    TmovMXTest<float, uint8_t, uint8_t, false, 4, 1>(M, K, N, 0, 0, 0);
}

TEST_F(TMOVMXTest, case5)
{
    uint32_t M = 64;
    uint32_t K = 120;
    uint32_t N = 64;

    TmovMXTest<float, uint8_t, uint8_t, false, 5, 1>(M, K, N, 0, 0, 0);
}

TEST_F(TMOVMXTest, case6)
{
    uint32_t M = 48;
    uint32_t K = 192;
    uint32_t N = 96;

    TmovMXTest<float, uint8_t, uint8_t, true, 6, 1>(M, K, N, 0, 0, 0);
}

TEST_F(TMOVMXTest, case7)
{
    uint32_t M = 128;
    uint32_t K = 64;
    uint32_t N = 64;

    TmovMXTest<float, uint8_t, uint8_t, false, 7, 2>(M, K, N, 0, 0, 0);
}

TEST_F(TMOVMXTest, case8)
{
    uint32_t M = 95;
    uint32_t K = 12;
    uint32_t N = 90;

    TmovMXTest<float, uint8_t, uint8_t, true, 8, 2>(M, K, N, 0, 0, 0);
}

TEST_F(TMOVMXTest, case9)
{
    uint32_t M = 4;
    uint32_t K = 30;
    uint32_t N = 8;

    TmovMXTest<float, uint8_t, uint8_t, false, 9, 2>(M, K, N, 0, 0, 0);
}

TEST_F(TMOVMXTest, case10)
{
    uint32_t M = 128;
    uint32_t K = 32;
    uint32_t N = 64;

    TmovMXTest<float, uint8_t, uint8_t, false, 10, 0>(M, K, N, 64, 0, 32);
}

TEST_F(TMOVMXTest, case11)
{
    uint32_t M = 128;
    uint32_t K = 98;
    uint32_t N = 64;

    TmovMXTest<float, uint8_t, uint8_t, true, 11, 0>(M, K, N, 32, 64, 0);
}

TEST_F(TMOVMXTest, case12)
{
    uint32_t M = 128;
    uint32_t K = 60;
    uint32_t N = 254;

    TmovMXTest<float, uint8_t, uint8_t, true, 12, 1>(M, K, N, 16, 0, 64);
}

TEST_F(TMOVMXTest, case13)
{
    uint32_t M = 48;
    uint32_t K = 180;
    uint32_t N = 96;

    TmovMXTest<float, uint8_t, uint8_t, false, 13, 1>(M, K, N, 16, 64, 32);
}

TEST_F(TMOVMXTest, case14)
{
    uint32_t M = 95;
    uint32_t K = 120;
    uint32_t N = 89;

    TmovMXTest<float, uint8_t, uint8_t, false, 14, 2>(M, K, N, 16, 64, 32);
}

TEST_F(TMOVMXTest, case15)
{
    uint32_t M = 48;
    uint32_t K = 190;
    uint32_t N = 98;

    TmovMXTest<float, uint8_t, uint8_t, true, 15, 2>(M, K, N, 16, 0, 64);
}

TEST_F(TMOVMXTest, case16) { TmovMXTest<float, uint8_t, uint8_t, false, 16, 0>(46, 66, 45, 0, 0, 0, 128, 256, 128); }

TEST_F(TMOVMXTest, case17)
{
    TmovMXTest<float, uint8_t, uint8_t, false, 17, 0>(68, 130, 80, 16, 64, 32, 128, 256, 128);
}

TEST_F(TMOVMXTest, case18)
{
    TmovMXTest<float, uint8_t, uint8_t, true, 18, 0>(127, 126, 130, 32, 64, 64, 256, 128, 256);
}

TEST_F(TMOVMXTest, case19) { TmovMXTest<float, uint8_t, uint8_t, false, 19, 1>(80, 96, 192, 48, 0, 64, 128, 256, 256); }

TEST_F(TMOVMXTest, case20)
{
    TmovMXTest<float, uint8_t, uint8_t, false, 20, 1>(98, 126, 108, 32, 64, 32, 128, 256, 128);
}

TEST_F(TMOVMXTest, case21) { TmovMXTest<float, uint8_t, uint8_t, true, 21, 1>(68, 196, 80, 0, 64, 64, 128, 256, 128); }

TEST_F(TMOVMXTest, case22) { TmovMXTest<float, uint8_t, uint8_t, false, 22, 2>(32, 64, 108, 16, 0, 32, 128, 256, 128); }

TEST_F(TMOVMXTest, case23)
{
    TmovMXTest<float, uint8_t, uint8_t, false, 23, 2>(196, 146, 96, 64, 64, 32, 256, 256, 128);
}

TEST_F(TMOVMXTest, case24) { TmovMXTest<float, uint8_t, uint8_t, true, 24, 2>(97, 96, 122, 32, 0, 64, 128, 256, 128); }
