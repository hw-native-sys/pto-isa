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

template <
    int format, typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void LaunchTStore(T* out, T* src, void* stream);

class TStoreMat2GMTest : public testing::Test {
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
    int format, typename DataType, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void TestTStore()
{
    size_t dataSize = gWholeShape0 * gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4 * sizeof(DataType);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    DataType *dstHost, *srcHost;
    DataType *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dataSize);
    aclrtMallocHost((void**)(&srcHost), dataSize);

    aclrtMalloc((void**)&dstDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", dataSize, srcHost, dataSize);

    aclrtMemcpy(srcDevice, dataSize, srcHost, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTStore<
        format, DataType, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1, gWholeShape2,
        gWholeShape3, gWholeShape4>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dataSize, dstDevice, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dataSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<DataType> golden(dataSize);
    std::vector<DataType> devFinal(dataSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dataSize, golden.data(), dataSize);
    ReadFile(GetGoldenDir() + "/output.bin", dataSize, devFinal.data(), dataSize);

    bool ret = ResultCmp<DataType>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TStoreMat2GMTest, case_nd1) { TestTStore<0, int64_t, 1, 2, 1, 11, 32, 1, 3, 2, 93, 32>(); }

TEST_F(TStoreMat2GMTest, case_nd2) { TestTStore<0, float, 1, 1, 1, 3, 128, 3, 3, 3, 32, 128>(); }

TEST_F(TStoreMat2GMTest, case_nd3) { TestTStore<0, int16_t, 2, 2, 1, 2, 32, 3, 3, 3, 111, 64>(); }

TEST_F(TStoreMat2GMTest, case_nd4) { TestTStore<0, int8_t, 1, 2, 1, 11, 32, 1, 3, 2, 93, 32>(); }

TEST_F(TStoreMat2GMTest, case_nd5) { TestTStore<0, uint16_t, 1, 1, 1, 128, 128, 1, 1, 1, 256, 256>(); }

TEST_F(TStoreMat2GMTest, case_dn1) { TestTStore<1, int64_t, 2, 2, 1, 32, 2, 3, 3, 3, 64, 111>(); }

TEST_F(TStoreMat2GMTest, case_dn2) { TestTStore<1, float, 1, 1, 1, 128, 3, 3, 3, 3, 128, 32>(); }

TEST_F(TStoreMat2GMTest, case_dn3) { TestTStore<1, int16_t, 2, 2, 1, 32, 2, 3, 3, 3, 64, 111>(); }

TEST_F(TStoreMat2GMTest, case_dn4) { TestTStore<1, int8_t, 1, 2, 1, 32, 11, 1, 3, 2, 32, 93>(); }

TEST_F(TStoreMat2GMTest, case_dn5) { TestTStore<1, uint16_t, 1, 2, 2, 128, 311, 4, 3, 3, 256, 400>(); }

TEST_F(TStoreMat2GMTest, case_nz1) { TestTStore<2, float, 1, 5, 21, 16, 8, 1, 5, 21, 16, 8>(); }

TEST_F(TStoreMat2GMTest, case_nz2) { TestTStore<2, int16_t, 2, 15, 11, 16, 16, 3, 23, 13, 16, 16>(); }

TEST_F(TStoreMat2GMTest, case_nz3) { TestTStore<2, int8_t, 1, 16, 32, 16, 32, 1, 32, 32, 16, 32>(); }

TEST_F(TStoreMat2GMTest, case_nz4) { TestTStore<2, uint16_t, 2, 4, 5, 16, 16, 7, 7, 7, 16, 16>(); }