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
void textract_test(uint32_t M, uint32_t K, uint32_t N, uint16_t indexM, uint16_t indexK, uint16_t indexN)
{
    uint32_t mValid = M - indexM;
    uint32_t nValid = N - indexN;
    size_t aFileSize = M * K * sizeof(U);
    size_t bFileSize = K * N * sizeof(U);
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
    aclrtMemset(dstHost, cFileSize, 0, cFileSize);

    aclrtMemcpy(dstDevice, cFileSize, dstHost, cFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
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

TEST_F(TEXTRACTTest, case1) { textract_test<1, uint16_t, uint16_t, uint16_t>(32, 96, 64, 0, 0, 0); }

TEST_F(TEXTRACTTest, case2) { textract_test<2, int32_t, int8_t, int8_t>(128, 128, 64, 0, 0, 0); }

TEST_F(TEXTRACTTest, case3) { textract_test<3, uint16_t, uint16_t, uint16_t>(64, 96, 64, 32, 16, 16); }

TEST_F(TEXTRACTTest, case4) { textract_test<4, int32_t, int8_t, int8_t>(128, 128, 64, 32, 64, 32); }

TEST_F(TEXTRACTTest, case5) { textract_test<5, uint16_t, uint16_t, uint16_t>(64, 128, 64, 0, 64, 0); }

TEST_F(TEXTRACTTest, case6) { textract_test<6, int32_t, int8_t, int8_t>(128, 64, 128, 32, 0, 0); }

TEST_F(TEXTRACTTest, case7) { textract_test<7, int32_t, int8_t, int8_t>(64, 96, 32, 32, 0, 0); }

TEST_F(TEXTRACTTest, case8) { textract_test<8, uint16_t, uint16_t, uint16_t>(64, 48, 96, 16, 16, 0); }