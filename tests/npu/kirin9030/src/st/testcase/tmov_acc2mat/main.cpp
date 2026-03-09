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
void LaunchTMOVAcc2MatNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2MatNZ2NZ(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

class TMOVTest : public testing::Test {
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

template <int32_t funcKey, typename CType, typename AType, typename BType, int32_t key, uint32_t IdxRow = 0,
          uint32_t IdxCol = 0, bool isInsert = false, uint32_t DstRow = 0, uint32_t DstCol = 0>
void tmov_acc2mat_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(BType);
    size_t cFileSize = (M - IdxRow) * (N - IdxCol) * sizeof(CType);
    if (isInsert) {
        cFileSize = DstRow * DstCol * sizeof(CType);
    }

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);
    aclrtMallocHost((void **)(&src2Host), cFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src2Device, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    if (isInsert) {
        ReadFile(GetGoldenDir() + "/dst.bin", cFileSize, src2Host, cFileSize);
    }

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (isInsert) {
        aclrtMemcpy(src2Device, cFileSize, src2Host, cFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if constexpr (funcKey == 1) {
        LaunchTMOVAcc2MatNZ2NZ<key>(dstDevice, src0Device, src1Device, src2Device, stream);
    } else if constexpr (funcKey == 2) {
        LaunchTMOVAcc2MatNZ2ND<key>(dstDevice, src0Device, src1Device, stream);
    }

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

    std::vector<CType> golden(cFileSize);
    std::vector<CType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMOVTest, case_nz2nz_1)
{
    tmov_acc2mat_test<1, uint16_t, uint16_t, uint16_t, 1>(96, 80, 112);
}

TEST_F(TMOVTest, case_nz2nz_2)
{
    tmov_acc2mat_test<1, uint16_t, uint16_t, uint16_t, 2>(128, 64, 128);
}

TEST_F(TMOVTest, case_nz2nz_3)
{
    tmov_acc2mat_test<1, uint16_t, uint16_t, uint16_t, 3>(16, 16, 16);
}

TEST_F(TMOVTest, case_nz2nz_4)
{
    tmov_acc2mat_test<1, uint16_t, uint16_t, uint16_t, 4>(32, 128, 64);
}

TEST_F(TMOVTest, case_nz2nd_1)
{
    tmov_acc2mat_test<2, uint16_t, uint16_t, uint16_t, 1>(65, 40, 80);
}

TEST_F(TMOVTest, case_nz2nd_2)
{
    tmov_acc2mat_test<2, uint16_t, uint16_t, uint16_t, 2>(111, 48, 88);
}

TEST_F(TMOVTest, case_nz2nd_3)
{
    tmov_acc2mat_test<2, uint16_t, uint16_t, uint16_t, 3>(80, 128, 112);
}

TEST_F(TMOVTest, case_nz2nd_4)
{
    tmov_acc2mat_test<2, uint16_t, uint16_t, uint16_t, 4>(6, 7, 8);
}

TEST_F(TMOVTest, case_nz2nz_extract)
{
    tmov_acc2mat_test<1, uint16_t, uint16_t, uint16_t, 5, 16, 16>(64, 64, 64);
}

TEST_F(TMOVTest, case_nz2nz_insert)
{
    tmov_acc2mat_test<1, uint16_t, uint16_t, uint16_t, 6, 32, 32, true, 128, 128>(32, 32, 32);
}
