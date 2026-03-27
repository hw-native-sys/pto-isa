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
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <typename TDst, typename TSrc, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols>
void LaunchTRowArgMin(TDst *out, TSrc *src, void *stream);
template <typename TDst, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols>
void LaunchTRowArgMinHalf(TDst *out, aclFloat16 *src, void *stream);

class TROWARGMINTest : public testing::Test {
private:
    aclrtStream stream;
    void *dstHost;
    void *srcHost;
    void *dstDevice;
    void *srcDevice;
    size_t dstFileSize;
    size_t srcFileSize;

protected:
    std::string GetGoldenDir()
    {
        const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
        const std::string caseName = testInfo->name();
        std::string suiteName = testInfo->test_suite_name();
        std::string fullPath = "../" + suiteName + "." + caseName;
        return fullPath;
    }
    void SetUp() override
    {
        aclInit(nullptr);
        aclrtSetDevice(0);
        aclrtCreateStream(&this->stream);
    }
    void TearDown() override
    {
        aclrtDestroyStream(this->stream);
        aclrtResetDevice(0);
        aclFinalize();
    }
    template <typename TDst, typename TSrc, int dstTileH, int dstTileW, int srcTileH, int srcTileW>
    void BeforeLaunch()
    {
        this->dstFileSize = sizeof(TDst) * dstTileH * dstTileW;
        this->srcFileSize = sizeof(TSrc) * srcTileH * srcTileW;

        aclrtMallocHost(&this->dstHost, this->dstFileSize);
        aclrtMallocHost(&this->srcHost, this->srcFileSize);
        memset(this->dstHost, 0, this->dstFileSize);
        ReadFile(GetGoldenDir() + "/input.bin", this->srcFileSize, this->srcHost, this->srcFileSize);

        aclrtMalloc(&this->dstDevice, this->dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&this->srcDevice, this->srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dstDevice, dstFileSize, dstHost, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    template <typename TDst>
    bool AfterLaunch()
    {
        aclrtSynchronizeStream(this->stream);
        std::vector<TDst> golden(this->dstFileSize);
        std::vector<TDst> devFinal(this->dstFileSize);
        aclrtMemcpy(devFinal.data(), this->dstFileSize, this->dstDevice, this->dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

        aclrtFree(this->dstDevice);
        aclrtFree(this->srcDevice);
        aclrtFreeHost(this->srcHost);
        aclrtFreeHost(this->dstHost);

        ReadFile(GetGoldenDir() + "/golden.bin", this->dstFileSize, golden.data(), this->dstFileSize);
        bool res = ResultCmp<TDst>(golden, devFinal, 0.0001f);
        if (!res) {
            WriteFile(GetGoldenDir() + "/output.bin", devFinal.data(), this->dstFileSize);
        }
        return res;
    }

    template <typename TDst, typename TSrc, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows,
              int vCols, bool isHalf = false>
    void Launch()
    {
        this->BeforeLaunch<TDst, TSrc, dstTileH, dstTileW, srcTileH, srcTileW>();
        if constexpr (isHalf) {
            LaunchTRowArgMinHalf<TDst, dstTileH, dstTileW, srcTileH, srcTileW, vRows, vCols>(
                (TDst *)this->dstDevice, (TSrc *)this->srcDevice, this->stream);
        } else {
            LaunchTRowArgMin<TDst, TSrc, dstTileH, dstTileW, srcTileH, srcTileW, vRows, vCols>(
                (TDst *)this->dstDevice, (TSrc *)this->srcDevice, this->stream);
        }
        bool res = this->AfterLaunch<TDst>();
        EXPECT_TRUE(res);
    }
};

TEST_F(TROWARGMINTest, case_uint32_float_8x1_8x8_8x8)
{
    this->Launch<uint32_t, float, 8, 1, 8, 8, 8, 8>();
}
TEST_F(TROWARGMINTest, case_uint32_float_1024x1_1024x8_1024x8)
{
    this->Launch<uint32_t, float, 1024, 1, 1024, 8, 1024, 8>();
}
TEST_F(TROWARGMINTest, case_uint32_float_16x1_13x16_13x13)
{
    this->Launch<uint32_t, float, 16, 1, 13, 16, 13, 13>();
}
TEST_F(TROWARGMINTest, case_uint32_float_1024x1_1023x24_1023x17)
{
    this->Launch<uint32_t, float, 1024, 1, 1023, 24, 1023, 17>();
}
TEST_F(TROWARGMINTest, case_uint32_float_8x1_8x64_8x64)
{
    this->Launch<uint32_t, float, 8, 1, 8, 64, 8, 64>();
}
TEST_F(TROWARGMINTest, case_uint32_float_264x1_260x64_260x64)
{
    this->Launch<uint32_t, float, 264, 1, 260, 64, 260, 64>();
}
TEST_F(TROWARGMINTest, case_uint32_float_8x1_1x128_1x128)
{
    this->Launch<uint32_t, float, 8, 1, 1, 128, 1, 128>();
}
TEST_F(TROWARGMINTest, case_uint32_float_64x1_32x128_32x128)
{
    this->Launch<uint32_t, float, 64, 1, 32, 128, 32, 128>();
}
TEST_F(TROWARGMINTest, case_uint32_float_8x1_3x4096_3x4095)
{
    this->Launch<uint32_t, float, 8, 1, 3, 4096, 3, 4095>();
}
TEST_F(TROWARGMINTest, case_uint32_float_8x1_2x16384_2x16381)
{
    this->Launch<uint32_t, float, 8, 1, 2, 16384, 2, 16381>();
}
TEST_F(TROWARGMINTest, case_uint32_half_16x1_2x16_2x16)
{
    this->Launch<uint32_t, aclFloat16, 16, 1, 2, 16, 2, 16, true>();
}
TEST_F(TROWARGMINTest, case_uint32_half_16x1_13x16_13x13)
{
    this->Launch<uint32_t, aclFloat16, 16, 1, 13, 16, 13, 13, true>();
}
TEST_F(TROWARGMINTest, case_uint32_half_272x1_260x64_260x64)
{
    this->Launch<uint32_t, aclFloat16, 272, 1, 260, 64, 260, 64, true>();
}
TEST_F(TROWARGMINTest, case_uint32_half_16x1_3x8192_3x8191)
{
    this->Launch<uint32_t, aclFloat16, 16, 1, 3, 8192, 3, 8191, true>();
}
TEST_F(TROWARGMINTest, case_uint32_half_16x1_1x16384_1x16381)
{
    this->Launch<uint32_t, aclFloat16, 16, 1, 1, 16384, 1, 16381, true>();
}
TEST_F(TROWARGMINTest, case_uint32_half_16x1_1x32768_1x32761)
{
    this->Launch<uint32_t, aclFloat16, 16, 1, 1, 32768, 1, 32761, true>();
}
TEST_F(TROWARGMINTest, case_int32_float_16x1_13x16_13x13)
{
    this->Launch<int32_t, float, 16, 1, 13, 16, 13, 13>();
}
TEST_F(TROWARGMINTest, case_int32_half_16x1_13x16_13x13)
{
    this->Launch<int32_t, aclFloat16, 16, 1, 13, 16, 13, 13, true>();
}
TEST_F(TROWARGMINTest, case_uint32_float_3x8_3x3480_3x3473)
{
    this->Launch<uint32_t, float, 3, 8, 3, 3480, 3, 3473>();
}
TEST_F(TROWARGMINTest, case_uint32_float_260x8_260x64_260x64)
{
    this->Launch<uint32_t, float, 260, 8, 260, 64, 260, 64>();
}
TEST_F(TROWARGMINTest, case_uint32_float_1023x8_1023x24_1023x17)
{
    this->Launch<uint32_t, float, 1023, 8, 1023, 24, 1023, 17>();
}
TEST_F(TROWARGMINTest, case_uint32_half_3x16_3x3488_3x3473)
{
    this->Launch<uint32_t, aclFloat16, 3, 16, 3, 3488, 3, 3473, true>();
}
TEST_F(TROWARGMINTest, case_uint32_half_260x16_260x64_260x64)
{
    this->Launch<uint32_t, aclFloat16, 260, 16, 260, 64, 260, 64, true>();
}
TEST_F(TROWARGMINTest, case_uint32_half_1023x16_1023x32_1023x17)
{
    this->Launch<uint32_t, aclFloat16, 1023, 16, 1023, 32, 1023, 17, true>();
}
