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
    uint32_t tRows, uint32_t tCols, uint32_t vRows, uint32_t vCols, uint32_t alignedCols = (vCols + 31 - 1) / 32 * 32>
void launchTSORT32Half(uint64_t* out, uint64_t* src, uint32_t* idx, uint64_t* tmp, void* stream);

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

template <
    typename T, bool isHalf, uint32_t tRows, uint32_t tCols, uint32_t vRows, uint32_t vCols,
    uint32_t colsAlign = (vCols + 31 - 1) / 32 * 32>
void tsort32_test()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int typeSize = sizeof(uint32_t);
    size_t srcByteSize = vRows * vCols * typeSize;
    size_t idxByteSize = vRows * vCols * sizeof(uint32_t);
    size_t dstByteSize = 2 * vRows * vCols * typeSize;
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
    aclrtMemset(dstHost, dstByteSize, 0, dstByteSize);

    aclrtMemcpy(dstDevice, dstByteSize, dstHost, dstByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDevice, idxByteSize, idxHost, idxByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tmpDevice, tmpByteSize, tmpHost, tmpByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTSORT32Half<tRows, tCols, vRows, vCols, colsAlign>(dstDevice, srcDevice, idxDevice, tmpDevice, stream);

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

    std::vector<T> golden(dstByteSize);
    std::vector<T> devFinal(dstByteSize);
    ReadFile(GetGoldenDir() + "/golden_output.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);

    bool ret = ResultCmp(golden, devFinal, 0.01f);
    EXPECT_TRUE(ret);
}

TEST_F(TSort32Test, case1) { tsort32_test<aclFloat16, true, 2, 32, 2, 32>(); }

TEST_F(TSort32Test, case2) { tsort32_test<aclFloat16, true, 4, 64, 4, 64>(); }

TEST_F(TSort32Test, case3) { tsort32_test<aclFloat16, true, 1, 32 * 256, 1, 32 * 256>(); }
