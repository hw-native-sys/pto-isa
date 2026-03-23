/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace std;
using namespace PtoTestCommon;

class TMINSTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, float profiling, float accuracy>
void LaunchTMins(T *out, T *src0, T *src1, void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, float profiling, float accuracy>
void test_tmins()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);
    size_t scalarFileSize = sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&src0Host), fileSize);
    aclrtMallocHost((void **)(&src1Host), scalarFileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, scalarFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, scalarFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTMins<T, kGRows_, kGCols_, kTRows_, kTCols_, profiling, accuracy>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return;
}

TEST_F(TMINSTest, case_float_64x64_64x64_64x64)
{
    test_tmins<float, 64, 64, 64, 64, 78.0f, 1.0f>();
}
TEST_F(TMINSTest, case_int32_64x64_64x64_64x64)
{
    test_tmins<int32_t, 64, 64, 64, 64, 78.0f, 1.0f>();
}
TEST_F(TMINSTest, case_int16_64x64_64x64_64x64)
{
    test_tmins<int16_t, 64, 64, 64, 64, 78.0f, 1.0f>();
}
TEST_F(TMINSTest, case_half_64x64_64x64_64x64)
{
    test_tmins<aclFloat16, 64, 64, 64, 64, 78.0f, 1.0f>();
}
TEST_F(TMINSTest, case_half_16x256_16x256_16x256)
{
    test_tmins<aclFloat16, 16, 256, 16, 256, 46.0f, 1.0f>();
}