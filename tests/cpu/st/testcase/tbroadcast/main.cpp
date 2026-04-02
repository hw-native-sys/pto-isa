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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

class TBroadCastTest : public testing::Test {
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

template <typename T, int D0, int D1, int D2, int D3, int D4, int KN>
void LaunchTBroadcast(T *out, T *src, void *stream);

/**
 * @tparam T Data type (float, int32_t, etc.)
 * @tparam D0, D1, D2, D3, D4 The 5D shape dimensions
 * @tparam KN The duplication count (N / rank)
 */
template <typename T, int D0, int D1, int D2, int D3, int D4, int KN>
void test_tbroadcast_5d()
{
    const size_t sizePerCopy = D0 * D1 * D2 * D3 * D4;
    const size_t totalSize = sizePerCopy * KN;

    const size_t tileBytesSrc = sizePerCopy * sizeof(T);
    const size_t tileBytesDst = totalSize * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), tileBytesDst);
    aclrtMallocHost((void **)(&srcHost), tileBytesSrc);
    aclrtMalloc((void **)&dstDevice, tileBytesDst, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, tileBytesSrc, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t readSizeSrc = tileBytesSrc;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", readSizeSrc, srcHost, tileBytesSrc));

    aclrtMemcpy(srcDevice, tileBytesSrc, srcHost, tileBytesSrc, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTBroadcast<T, D0, D1, D2, D3, D4, KN>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);

    aclrtMemcpy(dstHost, tileBytesDst, dstDevice, tileBytesDst, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<T> golden(totalSize);
    size_t readSizeGolden = tileBytesDst;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", readSizeGolden, golden.data(), tileBytesDst));

    std::vector<T> devFinal(totalSize);
    std::copy(dstHost, dstHost + (tileBytesDst / sizeof(T)), devFinal.begin());

    EXPECT_TRUE(ResultCmp<T>(golden, devFinal, 0.001f));

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TBroadCastTest, case_float_1)
{
    test_tbroadcast_5d<float, 1, 2, 4, 64, 64, 5>();
}

TEST_F(TBroadCastTest, case_int32_2)
{
    test_tbroadcast_5d<int32_t, 1, 2, 4, 64, 64, 3>();
}

TEST_F(TBroadCastTest, case_int16_3)
{
    test_tbroadcast_5d<int16_t, 2, 2, 3, 64, 64, 2>();
}

TEST_F(TBroadCastTest, case_half_4)
{
    test_tbroadcast_5d<aclFloat16, 1, 2, 1, 16, 256, 1>();
}
#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TBroadCastTest, case_bf16_5)
{
    test_tbroadcast_5d<bfloat16_t, 1, 2, 1, 16, 256, 1>();
}
#endif
