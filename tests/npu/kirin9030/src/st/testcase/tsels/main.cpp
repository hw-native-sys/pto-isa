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

template <typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH,
          int srcTileW, int vRows, int vCols>
void LaunchTSels(T *out, TMask *mask, T *src, T scalar, void *stream);
template <typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH, int srcTileW,
          int vRows, int vCols>
void LaunchTSelsHalf(aclFloat16 *out, TMask *mask, aclFloat16 *src, aclFloat16 scalar, void *stream);

class TSELSTest : public testing::Test {
private:
    aclrtStream stream;
    void *dstHost;
    void *srcHost;
    void *maskHost;
    void *dstDevice;
    void *srcDevice;
    void *maskDevice;
    size_t dstFileSize;
    size_t maskFileSize;
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
    template <typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH,
              int srcTileW>
    void BeforeLaunch(T &scalar)
    {
        this->dstFileSize = sizeof(T) * dstTileH * dstTileW;
        this->maskFileSize = sizeof(TMask) * maskTileH * maskTileW;
        this->srcFileSize = sizeof(T) * srcTileH * srcTileW;
        size_t scalarFileSize = sizeof(T);

        aclrtMallocHost(&this->dstHost, this->dstFileSize);
        aclrtMallocHost(&this->maskHost, this->maskFileSize);
        aclrtMallocHost(&this->srcHost, this->srcFileSize);
        memset(this->dstHost, 0, this->dstFileSize);
        ReadFile(GetGoldenDir() + "/mask.bin", this->maskFileSize, this->maskHost, this->maskFileSize);
        ReadFile(GetGoldenDir() + "/input1.bin", this->srcFileSize, this->srcHost, this->srcFileSize);
        ReadFile(GetGoldenDir() + "/input2.bin", scalarFileSize, &scalar, scalarFileSize);

        aclrtMalloc(&this->dstDevice, this->dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&this->maskDevice, this->maskFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&this->srcDevice, this->srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dstDevice, dstFileSize, dstHost, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(maskDevice, maskFileSize, maskHost, maskFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    template <typename T>
    bool AfterLaunch()
    {
        aclrtSynchronizeStream(this->stream);
        std::vector<T> golden(this->dstFileSize);
        std::vector<T> devFinal(this->dstFileSize);
        aclrtMemcpy(devFinal.data(), this->dstFileSize, this->dstDevice, this->dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

        aclrtFree(this->dstDevice);
        aclrtFree(this->maskDevice);
        aclrtFree(this->srcDevice);
        aclrtFreeHost(this->maskHost);
        aclrtFreeHost(this->srcHost);
        aclrtFreeHost(this->dstHost);

        ReadFile(GetGoldenDir() + "/golden.bin", this->dstFileSize, golden.data(), this->dstFileSize);
        bool res = ResultCmp<T>(golden, devFinal, 0.0001f);
        if (!res) {
            WriteFile(GetGoldenDir() + "/output.bin", devFinal.data(), this->dstFileSize);
        }
        return res;
    }

    template <typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH,
              int srcTileW, int vRows, int vCols, bool isHalf = false>
    void Launch()
    {
        T scalar;
        this->BeforeLaunch<T, TMask, dstTileH, dstTileW, maskTileH, maskTileW, srcTileH, srcTileW>(scalar);
        if constexpr (isHalf) {
            LaunchTSelsHalf<TMask, dstTileH, dstTileW, maskTileH, maskTileW, srcTileH, srcTileW, vRows, vCols>(
                (T *)this->dstDevice, (TMask *)this->maskDevice, (T *)this->srcDevice, scalar, this->stream);
        } else {
            LaunchTSels<T, TMask, dstTileH, dstTileW, maskTileH, maskTileW, srcTileH, srcTileW, vRows, vCols>(
                (T *)this->dstDevice, (TMask *)this->maskDevice, (T *)this->srcDevice, scalar, this->stream);
        }
        bool res = this->AfterLaunch<T>();
        EXPECT_TRUE(res);
    }
};

TEST_F(TSELSTest, case_uint8_uint8_2x32_2x32_2x32_2x32)
{
    this->Launch<uint8_t, uint8_t, 2, 32, 2, 32, 2, 32, 2, 32>();
}
TEST_F(TSELSTest, case_uint8_uint16_2x32_2x16_2x32_2x32)
{
    this->Launch<uint8_t, uint16_t, 2, 32, 2, 16, 2, 32, 2, 32>();
}
TEST_F(TSELSTest, case_uint8_uint32_2x32_2x8_2x32_2x32)
{
    this->Launch<uint8_t, uint32_t, 2, 32, 2, 8, 2, 32, 2, 32>();
}
TEST_F(TSELSTest, case_uint16_uint8_2x16_2x32_2x16_2x16)
{
    this->Launch<uint16_t, uint8_t, 2, 16, 2, 32, 2, 16, 2, 16>();
}
TEST_F(TSELSTest, case_uint16_uint16_2x16_2x16_2x16_2x16)
{
    this->Launch<uint16_t, uint16_t, 2, 16, 2, 16, 2, 16, 2, 16>();
}
TEST_F(TSELSTest, case_uint16_uint32_2x16_2x8_2x16_2x16)
{
    this->Launch<uint16_t, uint32_t, 2, 16, 2, 8, 2, 16, 2, 16>();
}
TEST_F(TSELSTest, case_uint32_uint8_2x8_2x32_2x8_2x8)
{
    this->Launch<uint32_t, uint8_t, 2, 8, 2, 32, 2, 8, 2, 8>();
}
TEST_F(TSELSTest, case_uint32_uint16_2x8_2x16_2x8_2x8)
{
    this->Launch<uint32_t, uint16_t, 2, 8, 2, 16, 2, 8, 2, 8>();
}
TEST_F(TSELSTest, case_uint32_uint32_2x8_2x8_2x8_2x8)
{
    this->Launch<uint32_t, uint32_t, 2, 8, 2, 8, 2, 8, 2, 8>();
}
TEST_F(TSELSTest, case_half_uint8_2x16_2x32_2x16_2x16)
{
    this->Launch<aclFloat16, uint8_t, 2, 16, 2, 32, 2, 16, 2, 16, true>();
}
TEST_F(TSELSTest, case_half_uint16_2x16_2x16_2x16_2x16)
{
    this->Launch<aclFloat16, uint16_t, 2, 16, 2, 16, 2, 16, 2, 16, true>();
}
TEST_F(TSELSTest, case_half_uint32_2x16_2x8_2x16_2x16)
{
    this->Launch<aclFloat16, uint32_t, 2, 16, 2, 8, 2, 16, 2, 16, true>();
}
TEST_F(TSELSTest, case_float_uint8_2x8_2x32_2x8_2x8)
{
    this->Launch<float, uint8_t, 2, 8, 2, 32, 2, 8, 2, 8>();
}
TEST_F(TSELSTest, case_float_uint16_2x8_2x16_2x8_2x8)
{
    this->Launch<float, uint16_t, 2, 8, 2, 16, 2, 8, 2, 8>();
}
TEST_F(TSELSTest, case_float_uint32_2x8_2x8_2x8_2x8)
{
    this->Launch<float, uint32_t, 2, 8, 2, 8, 2, 8, 2, 8>();
}
TEST_F(TSELSTest, case_uint8_uint8_2x32_2x64_2x128_2x31)
{
    this->Launch<uint8_t, uint8_t, 2, 32, 2, 64, 2, 128, 2, 31>();
}
TEST_F(TSELSTest, case_uint16_uint8_2x32_2x64_2x128_2x31)
{
    this->Launch<uint16_t, uint8_t, 2, 32, 2, 64, 2, 128, 2, 31>();
}
TEST_F(TSELSTest, case_float_uint8_2x32_2x64_2x128_2x31)
{
    this->Launch<float, uint8_t, 2, 32, 2, 64, 2, 128, 2, 31>();
}
TEST_F(TSELSTest, case_uint8_uint8_32x672_32x96_32x672_32x666)
{
    this->Launch<uint8_t, uint8_t, 32, 672, 32, 96, 32, 672, 32, 666>();
}
TEST_F(TSELSTest, case_half_uint8_32x672_32x96_32x672_32x666)
{
    this->Launch<aclFloat16, uint8_t, 32, 672, 32, 96, 32, 672, 32, 666, true>();
}
TEST_F(TSELSTest, case_float_uint8_32x672_32x96_32x672_32x666)
{
    this->Launch<float, uint8_t, 32, 672, 32, 96, 32, 672, 32, 666>();
}
TEST_F(TSELSTest, case_float_uint8_1x8192_1x4096_1x8192_1x8192)
{
    this->Launch<float, uint8_t, 1, 8192, 1, 4096, 1, 8192, 1, 8192>();
}
