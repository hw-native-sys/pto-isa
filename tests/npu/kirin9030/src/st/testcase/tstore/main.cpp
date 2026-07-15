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
void LaunchTStore(T* out, T* src0, void* stream);
class TStoreTest : public testing::Test {
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
void test_tstore()
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
    aclrtMemset(dstHost, dataSize, 0, dataSize);

    aclrtMemcpy(dstDevice, dataSize, dstHost, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
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

TEST_F(TStoreTest, case1) { test_tstore<2, float, 1, 1, 1, 16, 8, 1, 1, 2, 16, 8>(); }

TEST_F(TStoreTest, case2) { test_tstore<2, uint8_t, 1, 2, 1, 16, 32, 2, 4, 2, 16, 32>(); }

TEST_F(TStoreTest, case3) { test_tstore<2, int16_t, 2, 2, 2, 16, 16, 5, 3, 3, 16, 16>(); }

TEST_F(TStoreTest, case4) { test_tstore<0, float, 2, 1, 1, 39, 47, 3, 2, 1, 43, 61>(); }

TEST_F(TStoreTest, case5) { test_tstore<0, int16_t, 1, 2, 1, 23, 121, 3, 2, 2, 35, 128>(); }

TEST_F(TStoreTest, case6) { test_tstore<0, int8_t, 2, 2, 3, 23, 47, 3, 3, 4, 32, 50>(); }

TEST_F(TStoreTest, case7) { test_tstore<1, float, 1, 1, 1, 4, 21, 1, 1, 1, 8, 32>(); }

TEST_F(TStoreTest, case8) { test_tstore<1, uint16_t, 3, 1, 1, 1, 124, 5, 1, 1, 2, 128>(); }

TEST_F(TStoreTest, case9) { test_tstore<1, int8_t, 2, 3, 7, 47, 13, 2, 3, 7, 55, 29>(); }

TEST_F(TStoreTest, case10) { test_tstore<0, int64_t, 1, 1, 2, 16, 16, 2, 2, 2, 16, 16>(); }

TEST_F(TStoreTest, case11) { test_tstore<1, uint64_t, 1, 1, 2, 16, 64, 2, 2, 2, 16, 64>(); }

TEST_F(TStoreTest, case12) { test_tstore<0, int64_t, 1, 1, 2, 39, 47, 2, 2, 2, 43, 50>(); }

TEST_F(TStoreTest, case13) { test_tstore<0, int32_t, 1, 1, 1, 8, 16, 1, 1, 1, 16, 16>(); }

TEST_F(TStoreTest, case14) { test_tstore<0, int16_t, 1, 2, 3, 16, 32, 2, 3, 4, 32, 32>(); }

TEST_F(TStoreTest, case15) { test_tstore<0, int8_t, 2, 3, 1, 16, 32, 3, 4, 2, 32, 32>(); }

TEST_F(TStoreTest, case16) { test_tstore<0, uint32_t, 2, 1, 2, 8, 16, 3, 2, 3, 16, 16>(); }

TEST_F(TStoreTest, case17) { test_tstore<0, int32_t, 3, 4, 1, 16, 8, 4, 5, 2, 32, 16>(); }

TEST_F(TStoreTest, case18) { test_tstore<0, uint16_t, 2, 1, 1, 16, 32, 3, 2, 2, 32, 32>(); }

TEST_F(TStoreTest, case19) { test_tstore<0, int16_t, 1, 1, 1, 32, 16, 2, 2, 2, 32, 32>(); }

TEST_F(TStoreTest, case20) { test_tstore<0, uint16_t, 1, 1, 1, 111, 88, 1, 1, 1, 112, 96>(); }

TEST_F(TStoreTest, case21) { test_tstore<0, uint16_t, 1, 1, 1, 97, 65, 1, 1, 1, 112, 80>(); }

TEST_F(TStoreTest, case22) { test_tstore<0, uint16_t, 1, 1, 1, 112, 88, 1, 1, 1, 112, 96>(); }

TEST_F(TStoreTest, case23) { test_tstore<0, uint16_t, 1, 1, 1, 112, 65, 1, 1, 1, 112, 80>(); }

TEST_F(TStoreTest, case24) { test_tstore<0, uint16_t, 1, 1, 1, 97, 97, 1, 1, 1, 112, 112>(); }

TEST_F(TStoreTest, case25) { test_tstore<0, uint16_t, 1, 1, 1, 64, 65, 1, 1, 1, 64, 80>(); }

TEST_F(TStoreTest, case26) { test_tstore<0, uint16_t, 1, 1, 1, 48, 65, 1, 1, 1, 48, 80>(); }

TEST_F(TStoreTest, case27) { test_tstore<0, uint16_t, 1, 1, 1, 80, 49, 1, 1, 1, 80, 64>(); }

TEST_F(TStoreTest, case28) { test_tstore<0, uint16_t, 1, 1, 1, 64, 81, 1, 1, 1, 64, 96>(); }

TEST_F(TStoreTest, case29) { test_tstore<0, uint16_t, 1, 1, 1, 48, 97, 1, 1, 1, 48, 112>(); }

TEST_F(TStoreTest, case30) { test_tstore<0, uint16_t, 1, 1, 1, 80, 83, 1, 1, 1, 80, 96>(); }

TEST_F(TStoreTest, case31) { test_tstore<0, uint16_t, 1, 1, 1, 112, 73, 1, 1, 1, 112, 80>(); }

TEST_F(TStoreTest, case32) { test_tstore<0, uint16_t, 1, 1, 1, 80, 79, 1, 1, 1, 80, 80>(); }
