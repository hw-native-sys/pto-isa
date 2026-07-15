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

template <int32_t testKey>
void launchTSORT32(uint64_t* out, uint64_t* src, uint32_t* idx, uint64_t* tmp, void* stream);

class TSort32Test : public testing::Test {
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

template <int32_t testKey, typename dType>
void tsort32_test(int32_t rows, int32_t cols, int32_t colsAlign)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int shape[2] = {rows, cols};
    int typeSize = sizeof(float);
    size_t srcByteSize = shape[0] * shape[1] * typeSize;
    size_t idxByteSize = shape[0] * shape[1] * sizeof(uint32_t);
    size_t dstByteSize = 2 * shape[0] * shape[1] * typeSize;
    size_t tmpByteSize = 1 * colsAlign * typeSize;
    uint64_t *dstHost, *srcHost, *tmpHost;
    uint64_t *dstDevice, *srcDevice, *tmpDevice;
    uint32_t *idxHost, *idxDevice;

    aclrtMallocHost((void**)(&dstHost), dstByteSize);
    aclrtMallocHost((void**)(&srcHost), srcByteSize);
    aclrtMallocHost((void**)(&idxHost), idxByteSize);
    aclrtMallocHost((void**)(&tmpHost), tmpByteSize);

    aclrtMalloc((void**)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&idxDevice, idxByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&tmpDevice, tmpByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", srcByteSize, srcHost, srcByteSize);
    ReadFile(GetGoldenDir() + "/input_idx.bin", idxByteSize, idxHost, idxByteSize);
    ReadFile(GetGoldenDir() + "/input_tmp.bin", tmpByteSize, tmpHost, tmpByteSize);

    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDevice, idxByteSize, idxHost, idxByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tmpDevice, tmpByteSize, tmpHost, tmpByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTSORT32<testKey>(dstDevice, srcDevice, idxDevice, tmpDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(idxDevice);
    aclrtFree(tmpDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(idxHost);
    aclrtFreeHost(tmpHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<dType> golden(dstByteSize);
    std::vector<dType> devFinal(dstByteSize);
    ReadFile(GetGoldenDir() + "/golden_output.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TSort32Test, case1) { tsort32_test<1, float>(2, 32, 32); }

TEST_F(TSort32Test, case2) { tsort32_test<2, uint16_t>(4, 64, 64); }

TEST_F(TSort32Test, case3) { tsort32_test<3, float>(1, 32 * 256, 32 * 256); }

TEST_F(TSort32Test, case4) { tsort32_test<4, float>(2, 13, 16); }