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
#include "acl_env.h"
#include <gtest/gtest.h>
#include <acl/acl.h>

using namespace std;
using namespace PtoTestCommon;

template <
    typename T, int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
    bool highPrecision = false>
void LaunchTDivS(T* out, T* src, T scalar, void* stream);

template <
    int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
    bool highPrecision = false>
void LaunchTDivSHalf(aclFloat16* out, aclFloat16* src, aclFloat16 scalar, void* stream);

template <typename T, int tileRow, int tileCol, int validRow, int validCol>
void LaunchTDivSInplace(T* out, T* src, T scalar, void* stream);

class TDIVSTest : public testing::Test {
public:
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
    typename T, int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
    bool isHalf = false, bool highPrecision = false>
void TDivSTestFramework()
{
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dstByteSize = dstTileRow * dstTileCol * sizeof(T);
    size_t srcByteSize = srcTileRow * srcTileCol * sizeof(T);
    size_t scalarByteSize = sizeof(T);
    T* dstHost;
    T* srcHost;
    T* dstDevice;
    T* srcDevice;
    T scalar;

    aclrtMallocHost((void**)(&dstHost), dstByteSize);
    aclrtMallocHost((void**)(&srcHost), srcByteSize);

    aclrtMalloc((void**)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(dstDevice, dstByteSize, 0, dstByteSize);

    ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
    ReadFile(GetGoldenDir() + "/divider.bin", scalarByteSize, (void*)&scalar, sizeof(T));
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (isHalf) {
        LaunchTDivSHalf<dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, highPrecision>(
            dstDevice, srcDevice, scalar, stream);
    } else {
        LaunchTDivS<T, dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, highPrecision>(
            dstDevice, srcDevice, scalar, stream);
    }
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);

    std::vector<T> golden(dstByteSize / sizeof(T));
    std::vector<T> devFinal(dstByteSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);

    constexpr auto resPrecision = highPrecision ? 0.0000001f : 0.001f;
    bool ret;
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        ret = ResultCmpExact(golden, devFinal.data());
    } else {
        ret = ResultCmp<T>(golden, devFinal, resPrecision);
    }
    EXPECT_TRUE(ret);
}

template <typename T, int tileRow, int tileCol, int validRow, int validCol>
void TDivSInplaceTestFramework()
{
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t byteSize = tileRow * tileCol * sizeof(T);
    T* dstHost;
    T* dstDevice;
    T scalar;

    aclrtMallocHost((void**)(&dstHost), byteSize);
    aclrtMalloc((void**)&dstDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", byteSize, dstHost, byteSize);
    size_t scalarSize = sizeof(T);
    ReadFile(GetGoldenDir() + "/divider.bin", scalarSize, &scalar, sizeof(T));

    aclrtMemcpy(dstDevice, byteSize, dstHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTDivSInplace<T, tileRow, tileCol, validRow, validCol>(dstDevice, dstDevice, scalar, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, byteSize, dstDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, byteSize);

    aclrtFree(dstDevice);
    aclrtFreeHost(dstHost);

    aclrtDestroyStream(stream);

    std::vector<T> golden(byteSize / sizeof(T));
    std::vector<T> devFinal(byteSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", byteSize, golden.data(), byteSize);
    ReadFile(GetGoldenDir() + "/output.bin", byteSize, devFinal.data(), byteSize);

    bool ret;
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        ret = ResultCmpExact(golden, devFinal.data());
    } else {
        ret = ResultCmp<T>(golden, devFinal, 0.001f);
    }
    EXPECT_TRUE(ret);
}

TEST_F(TDIVSTest, case1) { TDivSTestFramework<float, 32, 128, 32, 64, 32, 64>(); }
TEST_F(TDIVSTest, case2) { TDivSTestFramework<aclFloat16, 63, 128, 63, 64, 63, 64, true>(); }
TEST_F(TDIVSTest, case4) { TDivSTestFramework<int16_t, 15, 192, 15, 192, 15, 192>(); }
TEST_F(TDIVSTest, case5) { TDivSTestFramework<float, 7, 512, 7, 448, 7, 448>(); }
TEST_F(TDIVSTest, case6) { TDivSTestFramework<float, 256, 32, 256, 16, 256, 16>(); }
TEST_F(TDIVSTest, case7) { TDivSTestFramework<float, 1, 32, 1, 16, 1, 16>(); }
TEST_F(TDIVSTest, caseHP1) { TDivSTestFramework<float, 2, 16, 2, 16, 2, 16, false, true>(); }
TEST_F(TDIVSTest, caseHP2) { TDivSTestFramework<aclFloat16, 2, 32, 2, 32, 2, 32, true, true>(); }
TEST_F(TDIVSTest, case_int64_4x16) { TDivSTestFramework<int64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TDIVSTest, case_uint64_4x16) { TDivSTestFramework<uint64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TDIVSTest, case_int64_4x64) { TDivSTestFramework<int64_t, 4, 64, 4, 64, 4, 64>(); }
TEST_F(TDIVSTest, case_uint64_4x64) { TDivSTestFramework<uint64_t, 4, 64, 4, 64, 4, 64>(); }
TEST_F(TDIVSTest, case_int64_1x16364) { TDivSTestFramework<int64_t, 1, 16364, 1, 16364, 1, 16364>(); }
TEST_F(TDIVSTest, case_uint64_1x16364) { TDivSTestFramework<uint64_t, 1, 16364, 1, 16364, 1, 16364>(); }
TEST_F(TDIVSTest, case_int64_1x32732) { TDivSTestFramework<int64_t, 1, 32732, 1, 32732, 1, 32732>(); }
TEST_F(TDIVSTest, case_int64_4091x4) { TDivSTestFramework<int64_t, 4091, 4, 4091, 4, 4091, 4>(); }
TEST_F(TDIVSTest, case_uint64_4091x4) { TDivSTestFramework<uint64_t, 4091, 4, 4091, 4, 4091, 4>(); }
TEST_F(TDIVSTest, case_int64_4x32_inplace) { TDivSInplaceTestFramework<int64_t, 4, 32, 4, 32>(); }
TEST_F(TDIVSTest, case_uint64_4x32_inplace) { TDivSInplaceTestFramework<uint64_t, 4, 32, 4, 32>(); }
TEST_F(TDIVSTest, case_int64_1x1024_inplace) { TDivSInplaceTestFramework<int64_t, 1, 1024, 1, 1024>(); }
TEST_F(TDIVSTest, case_int64_4x64_40_inplace) { TDivSInplaceTestFramework<int64_t, 4, 64, 4, 40>(); }
TEST_F(TDIVSTest, case_int64_1x2048_2045_inplace) { TDivSInplaceTestFramework<int64_t, 1, 2048, 1, 2045>(); }

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new AclEnv());
    return RUN_ALL_TESTS();
}
