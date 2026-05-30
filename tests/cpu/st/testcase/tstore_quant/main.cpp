/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

using namespace std;
using namespace PtoTestCommon;

template <int format, typename DstT, typename SrcT, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
          int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, bool is_v_quant,
          bool saturate_inf, bool apply_relu>
void LaunchTStoreQuant(DstT *out, SrcT *src, uint64_t *fbQuant, void *stream);

class TStoreQuantTest : public testing::Test {
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

template <int format, typename SrcDataType, typename DstDataType, int gShape0, int gShape1, int gShape2, int gShape3,
          int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4,
          bool is_v_quant, bool saturate_inf, bool apply_relu>
void test_tstore_quant()
{
    size_t srcDataSize = gWholeShape0 * gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4 * sizeof(SrcDataType);
    size_t dstDataSize = gWholeShape0 * gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4 * sizeof(DstDataType);
    size_t vectorSize = (is_v_quant ? (format == 0 ? gWholeShape4 : gWholeShape3) : 1) * sizeof(uint64_t);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    DstDataType *dstHost;
    SrcDataType *srcHost;
    uint64_t *quantHost;
    DstDataType *dstDevice;
    SrcDataType *srcDevice;
    uint64_t *quantDevice;

    aclrtMallocHost((void **)(&dstHost), dstDataSize);
    aclrtMallocHost((void **)(&srcHost), srcDataSize);
    aclrtMallocHost((void **)(&quantHost), vectorSize);

    aclrtMalloc((void **)&dstDevice, dstDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&quantDevice, vectorSize, ACL_MEM_MALLOC_HUGE_FIRST);

    std::fill(dstDevice, dstDevice + (dstDataSize / sizeof(DstDataType)), 0);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", srcDataSize, srcHost, srcDataSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/quant.bin", vectorSize, quantHost, vectorSize));

    aclrtMemcpy(srcDevice, srcDataSize, srcHost, srcDataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(quantDevice, vectorSize, quantHost, vectorSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTStoreQuant<format, DstDataType, SrcDataType, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0,
                      gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4, is_v_quant, saturate_inf, apply_relu>(
        dstDevice, srcDevice, quantDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstDataSize, dstDevice, dstDataSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstDataSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<DstDataType> golden(dstDataSize);
    std::vector<DstDataType> devFinal(dstDataSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", dstDataSize, golden.data(), dstDataSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", dstDataSize, devFinal.data(), dstDataSize));

    bool ret = ResultCmp<DstDataType>(golden, devFinal, 0);
    EXPECT_TRUE(ret);
}

TEST_F(TStoreQuantTest, ND_1)
{
    test_tstore_quant<0, float, int8_t, 1, 1, 1, 2, 128, 1, 1, 1, 2, 128, true, true, true>();
}

TEST_F(TStoreQuantTest, ND_2)
{
    test_tstore_quant<0, int32_t, int16_t, 1, 2, 1, 23, 121, 3, 2, 2, 35, 125, true, true, false>();
}

TEST_F(TStoreQuantTest, ND_3)
{
    test_tstore_quant<0, int32_t, int8_t, 2, 2, 3, 23, 47, 3, 3, 4, 32, 50, true, false, true>();
}

TEST_F(TStoreQuantTest, DN_4)
{
    test_tstore_quant<1, float, aclFloat16, 1, 1, 1, 4, 21, 1, 1, 1, 8, 32, false, true, true>();
}

TEST_F(TStoreQuantTest, DN_5)
{
    test_tstore_quant<1, float, aclFloat16, 3, 1, 1, 1, 124, 5, 1, 1, 2, 128, true, false, false>();
}

TEST_F(TStoreQuantTest, DN_6)
{
    test_tstore_quant<1, int32_t, int8_t, 2, 1, 2, 32, 32, 3, 4, 3, 64, 35, false, true, false>();
}

TEST_F(TStoreQuantTest, DN_7)
{
    test_tstore_quant<1, float, aclFloat16, 1, 1, 1, 16, 8, 1, 1, 2, 16, 8, false, false, true>();
}

TEST_F(TStoreQuantTest, DN_8)
{
    test_tstore_quant<1, int32_t, int16_t, 2, 2, 2, 16, 16, 5, 3, 3, 16, 16, false, false, false>();
}

TEST_F(TStoreQuantTest, DN_9)
{
    test_tstore_quant<1, int32_t, int8_t, 1, 2, 1, 16, 32, 2, 4, 2, 16, 32, true, true, true>();
}
