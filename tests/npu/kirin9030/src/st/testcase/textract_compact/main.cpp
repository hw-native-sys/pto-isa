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
void launchTEXTRACT(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

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

template <int32_t key, typename T, typename U, typename S>
void textract_test(
    uint32_t M, uint32_t K, uint32_t N, uint16_t indexM, uint16_t indexK, uint16_t indexN, uint16_t baseM,
    uint16_t baseK, uint16_t baseN)
{
    uint32_t mValid = M - indexM;
    uint32_t nValid = N - indexN;
    size_t aFileSize = baseM * baseK * sizeof(U);
    size_t bFileSize = baseK * baseN * sizeof(U);
    size_t cFileSize = mValid * nValid * sizeof(T);

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
    launchTEXTRACT<key>(dstDevice, src0Device, src1Device, stream);

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

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TEXTRACTTest, case11) { textract_test<11, uint16_t, uint16_t, uint16_t>(63, 48, 66, 0, 0, 0, 128, 64, 256); }
TEST_F(TEXTRACTTest, case12) { textract_test<12, uint16_t, uint16_t, uint16_t>(68, 93, 97, 0, 0, 0, 128, 128, 128); }
TEST_F(TEXTRACTTest, case14) { textract_test<14, uint16_t, uint16_t, uint16_t>(59, 232, 61, 16, 16, 16, 64, 256, 64); }
TEST_F(TEXTRACTTest, case21) { textract_test<21, int32_t, int8_t, int8_t>(97, 231, 83, 0, 0, 0, 128, 256, 128); }
TEST_F(TEXTRACTTest, case22) { textract_test<22, int32_t, int8_t, int8_t>(71, 188, 82, 0, 0, 0, 128, 256, 128); }
TEST_F(TEXTRACTTest, case23) { textract_test<23, int32_t, int8_t, int8_t>(63, 112, 98, 32, 32, 32, 64, 128, 128); }
TEST_F(TEXTRACTTest, case24) { textract_test<24, int32_t, int8_t, int8_t>(106, 125, 60, 32, 32, 32, 128, 128, 64); }
