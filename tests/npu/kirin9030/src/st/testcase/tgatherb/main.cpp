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

class TGATHERBTest : public testing::Test {
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
    typename T, uint64_t dstS1, uint64_t dstS0, uint64_t offsetS1, uint64_t offsetS0, uint64_t srcS1, uint64_t srcS0>
void LaunchTGatherB(T* out, T* src, uint32_t* offset, void* stream);

template <
    typename T, uint64_t dstS1, uint64_t dstS0, uint64_t offsetS1, uint64_t offsetS0, uint64_t srcS1, uint64_t srcS0>
void test_tgatherb()
{
    size_t dstFileSize = dstS1 * dstS0 * sizeof(T);
    size_t srcFileSize = srcS1 * srcS0 * sizeof(T);
    size_t offsetFileSize = offsetS1 * offsetS0 * 4;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost, *dstDevice, *srcDevice;
    uint32_t *offsetHost, *offsetDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);
    aclrtMallocHost((void**)(&offsetHost), offsetFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&offsetDevice, offsetFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x.bin", srcFileSize, srcHost, srcFileSize);
    ReadFile(GetGoldenDir() + "/offset.bin", offsetFileSize, offsetHost, offsetFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(offsetDevice, offsetFileSize, offsetHost, offsetFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTGatherB<T, dstS1, dstS0, offsetS1, offsetS0, srcS1, srcS0>(dstDevice, srcDevice, offsetDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(offsetDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(offsetHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize);
    std::vector<T> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TGATHERBTest, case_float_2x128_2x16_2x128) { test_tgatherb<float, 2, 128, 2, 16, 2, 128>(); }
TEST_F(TGATHERBTest, case_int32_2x128_2x16_2x128) { test_tgatherb<int32_t, 2, 128, 2, 16, 2, 128>(); }
TEST_F(TGATHERBTest, case_uint32_2x128_2x16_2x128) { test_tgatherb<uint32_t, 2, 128, 2, 16, 2, 128>(); }
TEST_F(TGATHERBTest, case_int16_1x16384_1x2048_1x16384) { test_tgatherb<int16_t, 1, 16384, 1, 2048, 1, 16384>(); }
TEST_F(TGATHERBTest, case_uint16_257x128_257x8_257x128) { test_tgatherb<uint16_t, 129, 128, 129, 8, 129, 128>(); }
TEST_F(TGATHERBTest, case_int8_2x256_2x8_2x256) { test_tgatherb<int8_t, 2, 256, 2, 8, 2, 256>(); }