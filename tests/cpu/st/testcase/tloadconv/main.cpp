/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <pto/pto-inst.hpp>
#include "test_common.h"
#include <gtest/gtest.h>
#include <fstream>

using namespace std;
using namespace PtoTestCommon;

template <int32_t testKey>
void launchTLOAD(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);

class TLoadConvTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

inline size_t GetFileSize(const std::string &filename)
{
    std::ifstream in(filename, std::ios::binary | std::ios::ate);

    return in.is_open() ? static_cast<size_t>(in.tellg()) : 0;
}

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

#define LOGSIZE 128
#define PRINTLOG 4
#define MAXBLOCK 64

template <int32_t testKey, typename T>
void tload_test()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    string path = GetGoldenDir();
    size_t in_byteSize = GetFileSize(path + "/input.bin");
    size_t gold_byteSize = GetFileSize(path + "/golden.bin");

    T *srcHost, *goldHost, *dstHost;
    void *srcDevice, *dstDevice, *logDevice = nullptr;

    aclrtMallocHost((void **)&srcHost, in_byteSize);
    aclrtMallocHost((void **)&goldHost, gold_byteSize);
    aclrtMallocHost((void **)&dstHost, gold_byteSize);
    aclrtMalloc((void **)&srcDevice, in_byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDevice, gold_byteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(path + "/input.bin", in_byteSize, srcHost, in_byteSize);
    ReadFile(path + "/golden.bin", gold_byteSize, goldHost, gold_byteSize);
    std::fill(dstHost, dstHost + (gold_byteSize / sizeof(*dstHost)), 0);

    aclrtMemcpy(srcDevice, in_byteSize, srcHost, in_byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dstDevice, gold_byteSize, dstHost, gold_byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

#ifdef DEBUGLOG
    aclrtMalloc((void **)&logDevice, MAXBLOCK * LOGSIZE * 8, ACL_MEM_MALLOC_HUGE_FIRST);
#endif

    launchTLOAD<testKey>((uint8_t *)dstDevice, (uint8_t *)srcDevice, (uint64_t *)logDevice, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(dstHost, gold_byteSize, dstDevice, gold_byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    int elements = gold_byteSize / sizeof(T);
    bool ret = ResultCmp(vector<T>(goldHost, goldHost + elements), vector<T>(dstHost, dstHost + elements), 0);

    aclrtFreeHost(srcHost);
    aclrtFreeHost(goldHost);
    aclrtFreeHost(dstHost);
    aclrtFree(srcDevice);
    aclrtFree(dstDevice);
    if (logDevice)
        aclrtFree(logDevice);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    EXPECT_TRUE(ret);
}

TEST_F(TLoadConvTest, case_5HD_fused_fp16)
{
    tload_test<1, half>();
}

TEST_F(TLoadConvTest, case_5HD_cropped_fp32)
{
    tload_test<2, float>();
}

TEST_F(TLoadConvTest, case_FracZ_4D_fp16)
{
    tload_test<3, half>();
}

TEST_F(TLoadConvTest, case_FracZ_5D_small_int8)
{
    tload_test<4, int8_t>();
}

#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TLoadConvTest, case_5HD_fused_bf16)
{
    tload_test<5, bfloat16_t>();
}
#endif
