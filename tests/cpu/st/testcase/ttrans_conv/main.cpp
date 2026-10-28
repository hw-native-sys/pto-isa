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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <
    typename T, int format, int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4, int dstShape0,
    int dstShape1, int dstShape2, int dstShape3, int dstShape4, int dstShape5, int groupN>
void LaunchTTRANSConv(T* out, T* src, void* stream);

template <
    typename MXType, int format, int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4,
    int dstShape0, int dstShape1, int dstShape2, int dstShape3, int dstShape4, int dstShape5, int groupN>
void LaunchTTRANSConvMX(uint8_t* out, uint8_t* src, void* stream);

class TTRANSConvTest : public testing::Test {
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
    typename T, int format, int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4, int dstShape0,
    int dstShape1, int dstShape2, int dstShape3, int dstShape4, int dstShape5 = 1, int groupN = 1>
void test_ttrans()
{
    size_t srcFileSize = srcShape0 * srcShape1 * srcShape2 * srcShape3 * srcShape4 * groupN * sizeof(T);
    size_t dstFileSize = dstShape0 * dstShape1 * dstShape2 * dstShape3 * dstShape4 * dstShape5 * groupN * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTTRANSConv<
        T, format, srcShape0, srcShape1, srcShape2, srcShape3, srcShape4, dstShape0, dstShape1, dstShape2, dstShape3,
        dstShape4, dstShape5, groupN>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize / sizeof(T), 0);
    std::vector<T> result(dstFileSize / sizeof(T), 0);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp<T>(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

template <
    typename MXType, int format, int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4,
    int dstShape0, int dstShape1, int dstShape2, int dstShape3, int dstShape4, int dstShape5 = 1, int groupN = 1>
void test_ttrans_MX()
{
    size_t srcFileSize = srcShape0 * srcShape1 * srcShape2 * srcShape3 * srcShape4 * groupN * sizeof(uint8_t);
    size_t dstFileSize =
        dstShape0 * dstShape1 * dstShape2 * dstShape3 * dstShape4 * dstShape5 * groupN * sizeof(uint8_t);
    if (IsTwinType<MXType>()) {
        srcFileSize /= 2;
        dstFileSize /= 2;
    }

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcHost;
    uint8_t *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize * 2);
    aclrtMallocHost((void**)(&srcHost), srcFileSize * 2);

    aclrtMalloc((void**)&dstDevice, dstFileSize * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize * 2, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize * 2, srcHost, srcFileSize * 2, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTTRANSConvMX<
        MXType, format, srcShape0, srcShape1, srcShape2, srcShape3, srcShape4, dstShape0, dstShape1, dstShape2,
        dstShape3, dstShape4, dstShape5, groupN>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize * 2, dstDevice, dstFileSize * 2, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<uint8_t> golden(dstFileSize / sizeof(uint8_t), 0);
    std::vector<uint8_t> result(dstFileSize / sizeof(uint8_t), 0);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp<uint8_t>(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TTRANSConvTest, NCHW2NC1HWC0_1) { test_ttrans<float, 0, 5, 4, 3, 8, 1, 5, 1, 3, 8, 8>(); }

TEST_F(TTRANSConvTest, NCHW2NC1HWC0_2) { test_ttrans<int32_t, 0, 5, 14, 13, 16, 1, 5, 2, 13, 16, 8>(); }

TEST_F(TTRANSConvTest, NCHW2NC1HWC0_3) { test_ttrans<uint16_t, 0, 1, 11, 13, 16, 1, 1, 1, 13, 16, 16>(); }

TEST_F(TTRANSConvTest, NCHW2NC1HWC0_4) { test_ttrans<int32_t, 0, 4, 32, 3, 7, 1, 4, 4, 3, 7, 8>(); }

TEST_F(TTRANSConvTest, NCHW2NC1HWC0_5) { test_ttrans<int8_t, 0, 4, 32, 3, 7, 1, 4, 1, 3, 7, 32>(); }

TEST_F(TTRANSConvTest, NCHW2NC1HWC0_MX_e8m0) { test_ttrans_MX<float8_e8m0_t, 0, 4, 32, 3, 7, 1, 4, 1, 3, 7, 32>(); }

/*------------------------------------------------*/

TEST_F(TTRANSConvTest, NC1HWC02NCHW_1) { test_ttrans<float, 1, 5, 3, 3, 4, 8, 5, 24, 3, 4, 1>(); }

TEST_F(TTRANSConvTest, NC1HWC02NCHW_2) { test_ttrans<int32_t, 1, 5, 2, 4, 5, 8, 5, 16, 4, 5, 1>(); }

/*------------------------------------------------*/

TEST_F(TTRANSConvTest, NC1HWC02C1HWN1N0C0_1) { test_ttrans<float, 2, 25, 4, 3, 8, 8, 4, 3, 8, 2, 16, 8>(); }

TEST_F(TTRANSConvTest, NC1HWC02C1HWN1N0C0_2) { test_ttrans<int32_t, 2, 15, 2, 3, 16, 8, 2, 3, 16, 2, 8, 8>(); }

TEST_F(TTRANSConvTest, NC1HWC02C1HWN1N0C0_3) { test_ttrans<uint16_t, 2, 11, 3, 2, 16, 16, 3, 2, 16, 2, 8, 16>(); }

TEST_F(TTRANSConvTest, NC1HWC02C1HWN1N0C0_4) { test_ttrans<int32_t, 2, 4, 32, 3, 7, 8, 32, 3, 7, 1, 4, 8>(); }

TEST_F(TTRANSConvTest, NC1HWC02C1HWN1N0C0_5) { test_ttrans<int8_t, 2, 4, 2, 3, 7, 32, 2, 3, 7, 1, 8, 32>(); }

TEST_F(TTRANSConvTest, NC1HWC02C1HWN1N0C0_MX_e4m3)
{
    test_ttrans_MX<float8_e4m3_t, 2, 4, 2, 3, 7, 32, 2, 3, 7, 1, 8, 32>();
}

/*------------------------------------------------*/

TEST_F(TTRANSConvTest, GNCHW2GNC1HWC0_1) { test_ttrans<float, 3, 5, 4, 3, 8, 1, 5, 1, 3, 8, 8, 1, 4>(); }

TEST_F(TTRANSConvTest, GNCHW2GNC1HWC0_2) { test_ttrans<int32_t, 3, 5, 14, 13, 8, 1, 5, 2, 13, 8, 8, 1, 2>(); }

TEST_F(TTRANSConvTest, GNCHW2GNC1HWC0_3) { test_ttrans<uint16_t, 3, 1, 11, 13, 16, 1, 1, 1, 13, 16, 16, 1, 3>(); }

TEST_F(TTRANSConvTest, GNCHW2GNC1HWC0_4) { test_ttrans<int32_t, 3, 4, 32, 3, 7, 1, 4, 4, 3, 7, 8, 1, 1>(); }

TEST_F(TTRANSConvTest, GNCHW2GNC1HWC0_5) { test_ttrans<int8_t, 3, 4, 32, 3, 7, 1, 4, 1, 3, 7, 32, 1, 3>(); }

// TEST_F(TTRANSConvTest, GNCHW2GNC1HWC0_MX_e2m1)
// {
//     test_ttrans_MX<float4_e1m2x2_t, 3, 4, 64, 3, 7, 1, 4, 1, 3, 7, 64, 1, 3>();
// }

TEST_F(TTRANSConvTest, GNCHW2GNC1HWC0_MXFP4_e2m1)
{
    test_ttrans_MX<float4_e2m1x2_t, 3, 4, 64, 3, 14, 1, 4, 1, 3, 14, 64, 1, 1>();
}

/*------------------------------------------*/

TEST_F(TTRANSConvTest, GNC1HWC02C1HWN1N0C0_1) { test_ttrans<float, 4, 25, 4, 3, 4, 8, 4, 3, 4, 2, 16, 8, 2>(); }

TEST_F(TTRANSConvTest, GNC1HWC02C1HWN1N0C0_2) { test_ttrans<int32_t, 4, 15, 2, 3, 4, 8, 2, 3, 4, 2, 8, 8, 3>(); }

TEST_F(TTRANSConvTest, GNC1HWC02C1HWN1N0C0_3) { test_ttrans<uint16_t, 4, 11, 3, 2, 16, 16, 3, 2, 16, 2, 8, 16, 2>(); }

TEST_F(TTRANSConvTest, GNC1HWC02C1HWN1N0C0_4) { test_ttrans<int32_t, 4, 4, 8, 3, 7, 8, 8, 3, 7, 1, 4, 8, 3>(); }

TEST_F(TTRANSConvTest, GNC1HWC02C1HWN1N0C0_5) { test_ttrans<int8_t, 4, 4, 2, 3, 7, 32, 2, 3, 7, 1, 8, 32, 1>(); }

/*---------------------------------------------------------------*/

TEST_F(TTRANSConvTest, GNC1HWC02GNCHW_1) { test_ttrans<float, 8, 1, 1, 2, 4, 8, 1, 8, 2, 4, 1, 1, 1>(); }

/*---------------------------------------------------------------*/

TEST_F(TTRANSConvTest, NCDHW2C1DHWN1N0C0_1) { test_ttrans<float, 5, 5, 3, 3, 4, 8, 96, 1, 16, 8, 1>(); }

TEST_F(TTRANSConvTest, NCDHW2C1DHWN1N0C0_2) { test_ttrans<int32_t, 5, 18, 2, 4, 5, 3, 60, 2, 16, 8, 1>(); }

/*---------------------------------------------------------------*/

TEST_F(TTRANSConvTest, NCHW2C1HWN1N0C0_1) { test_ttrans<float, 6, 15, 4, 3, 8, 1, 24, 1, 16, 8, 1>(); }

TEST_F(TTRANSConvTest, NCHW2C1HWN1N0C0_2) { test_ttrans<int32_t, 6, 11, 14, 3, 16, 1, 96, 2, 8, 8, 1>(); }

TEST_F(TTRANSConvTest, NCHW2C1HWN1N0C0_3) { test_ttrans<int8_t, 6, 4, 8, 3, 8, 1, 24, 1, 32, 32, 1>(); }

/*---------------------------------------------------------------*/

TEST_F(TTRANSConvTest, NCHW2C1DHWN1N0C0_1) { test_ttrans<float, 7, 15, 4, 3, 8, 1, 24, 1, 16, 8, 1>(); }

TEST_F(TTRANSConvTest, NCHW2C1DHWN1N0C0_2) { test_ttrans<int32_t, 7, 11, 14, 3, 16, 1, 96, 2, 8, 8, 1>(); }

/*=========================================================================================*/
/* Valid shape 用例                                                                        */
/*   raw shape 只表达每维 stride 与容量；转换域完全由两侧 valid shape 决定。                 */
/*   golden 中：valid 之外 = 哨兵 -12345（必须原样保留），valid 盒内的格式补零 = 0。        */
/*=========================================================================================*/

void LaunchTTRANSConvValid_NCHW2NC1HWC0(
    float* out, float* src, void* stream, const int64_t* srcValid, const int64_t* dstValid);
void LaunchTTRANSConvValid_NC1HWC02NCHW(
    float* out, float* src, void* stream, const int64_t* srcValid, const int64_t* dstValid);
void LaunchTTRANSConvValid_NC1HWC02C1HWN1N0C0(
    float* out, float* src, void* stream, const int64_t* srcValid, const int64_t* dstValid);
void LaunchTTRANSConvValid_GNCHW2GNC1HWC0(
    float* out, float* src, void* stream, const int64_t* srcValid, const int64_t* dstValid);
void LaunchTTRANSConvValid_GNC1HWC02C1HWN1N0C0(
    float* out, float* src, void* stream, const int64_t* srcValid, const int64_t* dstValid);
void LaunchTTRANSConvValid_NCDHW2C1DHWN1N0C0(
    float* out, float* src, void* stream, const int64_t* srcValid, const int64_t* dstValid);
void LaunchTTRANSConvValid_NCHW2C1HWN1N0C0(
    float* out, float* src, void* stream, const int64_t* srcValid, const int64_t* dstValid);
void LaunchTTRANSConvValid_NCHW2C1DHWN1N0C0(
    float* out, float* src, void* stream, const int64_t* srcValid, const int64_t* dstValid);

// 分配 -> 启动 -> 回读 -> 与 golden 逐元素比对
template <typename T, typename LaunchFn>
void test_ttrans_valid_impl(size_t srcFileSize, size_t dstFileSize, LaunchFn launch)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launch(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize / sizeof(T), 0);
    std::vector<T> result(dstFileSize / sizeof(T), 0);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp<T>(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

// 场景 NCHW2NC1HWC0：src raw (1, 16, 8, 16) valid (1, 16, 4, 16)；dst raw (2, 4, 8, 32, 16) valid (1, 2, 4, 16, 8)
TEST_F(TTRANSConvTest, valid_NCHW2NC1HWC0_dst_all_shrunk)
{
    constexpr size_t srcFileSize = (size_t)1 * 16 * 8 * 16 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)2 * 4 * 8 * 32 * 16 * sizeof(float);
    const int64_t srcValid[4] = {1, 16, 4, 16};
    const int64_t dstValid[5] = {1, 2, 4, 16, 8};
    test_ttrans_valid_impl<float>(srcFileSize, dstFileSize, [&](float* dst, float* src, void* stream) {
        LaunchTTRANSConvValid_NCHW2NC1HWC0(dst, src, stream, srcValid, dstValid);
    });
}

// 场景 NC1HWC02NCHW：src raw (3, 5, 7, 9, 16) valid (2, 3, 4, 5, 8)；dst raw (3, 32, 7, 9) valid (2, 24, 4, 5)
TEST_F(TTRANSConvTest, valid_NC1HWC02NCHW_all_dims_differ)
{
    constexpr size_t srcFileSize = (size_t)3 * 5 * 7 * 9 * 16 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)3 * 32 * 7 * 9 * sizeof(float);
    const int64_t srcValid[5] = {2, 3, 4, 5, 8};
    const int64_t dstValid[4] = {2, 24, 4, 5};
    test_ttrans_valid_impl<float>(srcFileSize, dstFileSize, [&](float* dst, float* src, void* stream) {
        LaunchTTRANSConvValid_NC1HWC02NCHW(dst, src, stream, srcValid, dstValid);
    });
}

// 场景 NC1HWC02C1HWN1N0C0：src raw (6, 3, 4, 5, 16) valid (5, 2, 2, 3, 8)；dst raw (16, 2, 24, 16) valid (12, 1, 16, 8)
TEST_F(TTRANSConvTest, valid_NC1HWC02C1HWN1N0C0_all_dims_differ)
{
    constexpr size_t srcFileSize = (size_t)6 * 3 * 4 * 5 * 16 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)16 * 2 * 24 * 16 * sizeof(float);
    const int64_t srcValid[5] = {5, 2, 2, 3, 8};
    const int64_t dstValid[4] = {12, 1, 16, 8};
    test_ttrans_valid_impl<float>(srcFileSize, dstFileSize, [&](float* dst, float* src, void* stream) {
        LaunchTTRANSConvValid_NC1HWC02C1HWN1N0C0(dst, src, stream, srcValid, dstValid);
    });
}

// 场景 GNCHW2GNC1HWC0：src raw (3, 4, 10, 5, 6) valid (2, 3, 8, 4, 5)；dst raw (3, 4, 3, 5, 6, 16) valid (2, 3, 1, 4,
// 5, 8)
TEST_F(TTRANSConvTest, valid_GNCHW2GNC1HWC0_all_dims_differ)
{
    constexpr size_t srcFileSize = (size_t)3 * 4 * 10 * 5 * 6 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)3 * 4 * 3 * 5 * 6 * 16 * sizeof(float);
    const int64_t srcValid[5] = {2, 3, 8, 4, 5};
    const int64_t dstValid[6] = {2, 3, 1, 4, 5, 8};
    test_ttrans_valid_impl<float>(srcFileSize, dstFileSize, [&](float* dst, float* src, void* stream) {
        LaunchTTRANSConvValid_GNCHW2GNC1HWC0(dst, src, stream, srcValid, dstValid);
    });
}

// 场景 GNC1HWC02C1HWN1N0C0：src raw (3, 4, 3, 4, 5, 16) valid (2, 3, 1, 3, 4, 8)；dst raw (28, 2, 24, 16) valid (24, 1,
// 16, 8)
TEST_F(TTRANSConvTest, valid_GNC1HWC02C1HWN1N0C0_all_dims_differ)
{
    constexpr size_t srcFileSize = (size_t)3 * 4 * 3 * 4 * 5 * 16 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)28 * 2 * 24 * 16 * sizeof(float);
    const int64_t srcValid[6] = {2, 3, 1, 3, 4, 8};
    const int64_t dstValid[4] = {24, 1, 16, 8};
    test_ttrans_valid_impl<float>(srcFileSize, dstFileSize, [&](float* dst, float* src, void* stream) {
        LaunchTTRANSConvValid_GNC1HWC02C1HWN1N0C0(dst, src, stream, srcValid, dstValid);
    });
}

// 场景 NCDHW2C1DHWN1N0C0：src raw (3, 10, 3, 4, 5) valid (2, 8, 2, 3, 4)；dst raw (32, 2, 24, 16) valid (24, 1, 16, 8)
TEST_F(TTRANSConvTest, valid_NCDHW2C1DHWN1N0C0_all_dims_differ)
{
    constexpr size_t srcFileSize = (size_t)3 * 10 * 3 * 4 * 5 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)32 * 2 * 24 * 16 * sizeof(float);
    const int64_t srcValid[5] = {2, 8, 2, 3, 4};
    const int64_t dstValid[4] = {24, 1, 16, 8};
    test_ttrans_valid_impl<float>(srcFileSize, dstFileSize, [&](float* dst, float* src, void* stream) {
        LaunchTTRANSConvValid_NCDHW2C1DHWN1N0C0(dst, src, stream, srcValid, dstValid);
    });
}

// 场景 NCHW2C1HWN1N0C0：src raw (8, 10, 4, 6) valid (6, 8, 3, 4)；dst raw (16, 2, 24, 16) valid (12, 1, 16, 8)
TEST_F(TTRANSConvTest, valid_NCHW2C1HWN1N0C0_all_dims_differ)
{
    constexpr size_t srcFileSize = (size_t)8 * 10 * 4 * 6 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)16 * 2 * 24 * 16 * sizeof(float);
    const int64_t srcValid[4] = {6, 8, 3, 4};
    const int64_t dstValid[4] = {12, 1, 16, 8};
    test_ttrans_valid_impl<float>(srcFileSize, dstFileSize, [&](float* dst, float* src, void* stream) {
        LaunchTTRANSConvValid_NCHW2C1HWN1N0C0(dst, src, stream, srcValid, dstValid);
    });
}

// 场景 NCHW2C1DHWN1N0C0：src raw (6, 18, 6, 5) valid (5, 16, 4, 3)；dst raw (32, 2, 24, 16) valid (24, 1, 16, 8)
TEST_F(TTRANSConvTest, valid_NCHW2C1DHWN1N0C0_all_dims_differ)
{
    constexpr size_t srcFileSize = (size_t)6 * 18 * 6 * 5 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)32 * 2 * 24 * 16 * sizeof(float);
    const int64_t srcValid[4] = {5, 16, 4, 3};
    const int64_t dstValid[4] = {24, 1, 16, 8};
    test_ttrans_valid_impl<float>(srcFileSize, dstFileSize, [&](float* dst, float* src, void* stream) {
        LaunchTTRANSConvValid_NCHW2C1DHWN1N0C0(dst, src, stream, srcValid, dstValid);
    });
}
