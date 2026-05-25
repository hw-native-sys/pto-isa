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
#include <gtest/gtest.h>
#include <acl/acl.h>

using namespace std;
using namespace PtoTestCommon;

// =============================================================================
// Pure index dispatcher (3-arg TCOLARGMAX)
// =============================================================================
template <uint32_t caseId>
void launchTCOLCMAXTestCase(void *out, void *src, aclrtStream stream);

// =============================================================================
// Value + index dispatcher (4-arg TCOLARGMAX)
// =============================================================================
template <uint32_t caseId>
void launchTCOLIDXVALMAXCase(void *outVal, void *outIdx, void *src, aclrtStream stream);

// =============================================================================
std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

// =============================================================================
// Test fixture: supports both pure index (3-arg) and value+index (4-arg)
// =============================================================================
class TCOLCMAXTest : public testing::Test {
public:
    aclrtStream stream;
    void *dstHost;
    void *srcHost;
    void *dstDevice;
    void *srcDevice;

    void *dstHostVal;
    void *dstHostIdx;
    void *dstDeviceVal;
    void *dstDeviceIdx;

protected:
    void SetUp() override
    {
        aclInit(nullptr);
        aclrtSetDevice(0);
        aclrtCreateStream(&stream);
    }

    void TearDown() override
    {
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    }

    bool CompareGolden(size_t dstByteSize, bool printAllEn = false)
    {
        std::vector<uint32_t> golden(dstByteSize);
        std::vector<uint32_t> result(dstByteSize);
        float eps = 0.001f;
        ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
        ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, result.data(), dstByteSize);
        if (printAllEn) {
            return ResultCmp(golden, result, eps, 0, 1000, true);
        }
        return ResultCmp(golden, result, eps, 0, 1000, false, true);
    }

    template <typename TVal, typename TIdx>
    bool CompareGoldenValIdx(size_t dstByteSize, bool printAllEn = false)
    {
        std::vector<TIdx> goldenIdx(dstByteSize);
        std::vector<TIdx> resultIdx(dstByteSize);
        std::vector<TVal> goldenVal(dstByteSize);
        std::vector<TVal> resultVal(dstByteSize);

        float eps = 0.001f;
        ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, goldenVal.data(), dstByteSize);
        ReadFile(GetGoldenDir() + "/output_val.bin", dstByteSize, resultVal.data(), dstByteSize);
        ReadFile(GetGoldenDir() + "/idx.bin", dstByteSize, goldenIdx.data(), dstByteSize);
        ReadFile(GetGoldenDir() + "/output_idx.bin", dstByteSize, resultIdx.data(), dstByteSize);
        if (printAllEn) {
            return ResultCmp(goldenVal, resultVal, eps, 0, 1000, true) &&
                   ResultCmp(goldenIdx, resultIdx, eps, 0, 1000, true);
        }
        return ResultCmp(goldenVal, resultVal, eps, 0, 1000, false, true) &&
               ResultCmp(goldenIdx, resultIdx, eps, 0, 1000, false, true);
    }

    // -------------------------------------------------------------------------
    // Pure index framework (3-arg)
    // -------------------------------------------------------------------------
    template <uint32_t caseId, typename T, int srcRow, int srcValidRow, int dstRow, int col, int validCol>
    bool TCOLCMAXTestFramework()
    {
        constexpr int dstCol = (validCol + 7) / 8 * 8;
        size_t dstByteSize = dstRow * dstCol * sizeof(uint32_t);
        size_t srcByteSize = srcRow * col * sizeof(T);
        aclrtMallocHost(&dstHost, dstByteSize);
        aclrtMallocHost(&srcHost, srcByteSize);
        aclrtMalloc(&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
        aclrtMemset(dstHost, dstByteSize, 0, dstByteSize);

        aclrtMemcpy(dstDevice, dstByteSize, dstHost, dstByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTCOLCMAXTestCase<caseId>(dstDevice, srcDevice, stream);
        aclrtSynchronizeStream(stream);

        aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

        aclrtFree(dstDevice);
        aclrtFree(srcDevice);
        aclrtFreeHost(dstHost);
        aclrtFreeHost(srcHost);

        return CompareGolden(dstByteSize);
    }

    // -------------------------------------------------------------------------
    // Value + index framework (4-arg)
    // -------------------------------------------------------------------------
    template <uint32_t caseId, typename TVal, typename TIdx, int srcRow, int srcValidRow, int dstRow, int col,
              int validCol>
    bool TCOLCMAXTestFramework()
    {
        constexpr int dstCol = (validCol + 7) / 8 * 8;
        size_t dstIdxByteSize = dstRow * dstCol * sizeof(TIdx);
        size_t dstValByteSize = dstRow * dstCol * sizeof(TVal);
        size_t srcByteSize = srcRow * col * sizeof(TVal);
        aclrtMallocHost(&dstHostIdx, dstIdxByteSize);
        aclrtMallocHost(&dstHostVal, dstValByteSize);
        aclrtMallocHost(&srcHost, srcByteSize);
        aclrtMalloc(&dstDeviceIdx, dstIdxByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dstDeviceVal, dstValByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
        aclrtMemset(dstHostIdx, dstIdxByteSize, 0, dstIdxByteSize);
        aclrtMemset(dstHostVal, dstValByteSize, 0, dstValByteSize);

        aclrtMemcpy(dstDeviceIdx, dstIdxByteSize, dstHostIdx, dstIdxByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(dstDeviceVal, dstValByteSize, dstHostVal, dstValByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTCOLIDXVALMAXCase<caseId>(dstDeviceVal, dstDeviceIdx, srcDevice, stream);
        aclrtSynchronizeStream(stream);

        aclrtMemcpy(dstHostIdx, dstIdxByteSize, dstDeviceIdx, dstIdxByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(dstHostVal, dstValByteSize, dstDeviceVal, dstValByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

        WriteFile(GetGoldenDir() + "/output_val.bin", dstHostVal, dstValByteSize);
        WriteFile(GetGoldenDir() + "/output_idx.bin", dstHostIdx, dstIdxByteSize);

        aclrtFree(dstDeviceIdx);
        aclrtFree(dstDeviceVal);
        aclrtFree(srcDevice);
        aclrtFreeHost(dstHostVal);
        aclrtFreeHost(dstHostIdx);
        aclrtFreeHost(srcHost);

        return CompareGoldenValIdx<TVal, TIdx>(dstValByteSize);
    }
};

// =============================================================================
// Pure index TEST_F macros (35 cases)
// =============================================================================
TEST_F(TCOLCMAXTest, case01)
{
    bool ret = TCOLCMAXTestFramework<1, float, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case02)
{
    bool ret = TCOLCMAXTestFramework<2, float, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case03)
{
    bool ret = TCOLCMAXTestFramework<3, float, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMAXTest, case11)
{
    bool ret = TCOLCMAXTestFramework<11, aclFloat16, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case12)
{
    bool ret = TCOLCMAXTestFramework<12, aclFloat16, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case13)
{
    bool ret = TCOLCMAXTestFramework<13, aclFloat16, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMAXTest, case21)
{
    bool ret = TCOLCMAXTestFramework<21, int8_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case22)
{
    bool ret = TCOLCMAXTestFramework<22, int8_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case23)
{
    bool ret = TCOLCMAXTestFramework<23, int8_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMAXTest, case31)
{
    bool ret = TCOLCMAXTestFramework<31, uint8_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case32)
{
    bool ret = TCOLCMAXTestFramework<32, uint8_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case33)
{
    bool ret = TCOLCMAXTestFramework<33, uint8_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMAXTest, case41)
{
    bool ret = TCOLCMAXTestFramework<41, int16_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case42)
{
    bool ret = TCOLCMAXTestFramework<42, int16_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case43)
{
    bool ret = TCOLCMAXTestFramework<43, int16_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMAXTest, case51)
{
    bool ret = TCOLCMAXTestFramework<51, uint16_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case52)
{
    bool ret = TCOLCMAXTestFramework<52, uint16_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case53)
{
    bool ret = TCOLCMAXTestFramework<53, uint16_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMAXTest, case61)
{
    bool ret = TCOLCMAXTestFramework<61, int32_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case62)
{
    bool ret = TCOLCMAXTestFramework<62, int32_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case63)
{
    bool ret = TCOLCMAXTestFramework<63, int32_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMAXTest, case71)
{
    bool ret = TCOLCMAXTestFramework<71, uint32_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case72)
{
    bool ret = TCOLCMAXTestFramework<72, uint32_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case73)
{
    bool ret = TCOLCMAXTestFramework<73, uint32_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMAXTest, case81)
{
    bool ret = TCOLCMAXTestFramework<81, aclFloat16, 16, 16, 1, 32, 32>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case82)
{
    bool ret = TCOLCMAXTestFramework<82, uint16_t, 16, 16, 1, 32, 32>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case83)
{
    bool ret = TCOLCMAXTestFramework<83, uint32_t, 16, 16, 1, 32, 31>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case84)
{
    bool ret = TCOLCMAXTestFramework<84, float, 16, 16, 1, 32, 31>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case85)
{
    bool ret = TCOLCMAXTestFramework<85, int8_t, 16, 16, 1, 32, 31>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case86)
{
    bool ret = TCOLCMAXTestFramework<86, uint8_t, 16, 16, 1, 32, 31>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case87)
{
    bool ret = TCOLCMAXTestFramework<87, int16_t, 16, 16, 1, 32, 31>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case88)
{
    bool ret = TCOLCMAXTestFramework<88, int32_t, 16, 16, 1, 32, 31>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMAXTest, case91)
{
    bool ret = TCOLCMAXTestFramework<91, uint16_t, 16, 16, 1, 128, 120>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case92)
{
    bool ret = TCOLCMAXTestFramework<92, aclFloat16, 16, 16, 1, 96, 88>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case93)
{
    bool ret = TCOLCMAXTestFramework<93, uint16_t, 4, 4, 1, 48, 34>();
    EXPECT_TRUE(ret);
}

// =============================================================================
// Value + index TEST_F macros (27 cases)
// =============================================================================
TEST_F(TCOLCMAXTest, case001)
{
    bool ret = TCOLCMAXTestFramework<1, float, int32_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case002)
{
    bool ret = TCOLCMAXTestFramework<2, float, int32_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case003)
{
    bool ret = TCOLCMAXTestFramework<3, float, int32_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMAXTest, case011)
{
    bool ret = TCOLCMAXTestFramework<11, aclFloat16, int16_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case012)
{
    bool ret = TCOLCMAXTestFramework<12, aclFloat16, int16_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case013)
{
    bool ret = TCOLCMAXTestFramework<13, aclFloat16, int16_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMAXTest, case041)
{
    bool ret = TCOLCMAXTestFramework<41, int16_t, int16_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case042)
{
    bool ret = TCOLCMAXTestFramework<42, int16_t, int16_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case043)
{
    bool ret = TCOLCMAXTestFramework<43, int16_t, int16_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMAXTest, case051)
{
    bool ret = TCOLCMAXTestFramework<51, uint16_t, int16_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case052)
{
    bool ret = TCOLCMAXTestFramework<52, uint16_t, int16_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case053)
{
    bool ret = TCOLCMAXTestFramework<53, uint16_t, int16_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMAXTest, case061)
{
    bool ret = TCOLCMAXTestFramework<61, int32_t, int32_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case062)
{
    bool ret = TCOLCMAXTestFramework<62, int32_t, int32_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case063)
{
    bool ret = TCOLCMAXTestFramework<63, int32_t, int32_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMAXTest, case071)
{
    bool ret = TCOLCMAXTestFramework<71, uint32_t, int32_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case072)
{
    bool ret = TCOLCMAXTestFramework<72, uint32_t, int32_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case073)
{
    bool ret = TCOLCMAXTestFramework<73, uint32_t, int32_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMAXTest, case081)
{
    bool ret = TCOLCMAXTestFramework<81, aclFloat16, int16_t, 16, 16, 1, 32, 32>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case082)
{
    bool ret = TCOLCMAXTestFramework<82, uint16_t, int16_t, 16, 16, 1, 32, 32>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case083)
{
    bool ret = TCOLCMAXTestFramework<83, uint32_t, int32_t, 16, 16, 1, 32, 31>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case084)
{
    bool ret = TCOLCMAXTestFramework<84, float, int32_t, 16, 16, 1, 32, 31>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case085)
{
    bool ret = TCOLCMAXTestFramework<85, int16_t, int16_t, 16, 16, 1, 32, 31>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case086)
{
    bool ret = TCOLCMAXTestFramework<86, int32_t, int32_t, 16, 16, 1, 32, 31>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMAXTest, case091)
{
    bool ret = TCOLCMAXTestFramework<91, uint16_t, int16_t, 16, 16, 1, 128, 120>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case092)
{
    bool ret = TCOLCMAXTestFramework<92, aclFloat16, int16_t, 16, 16, 1, 96, 88>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case093)
{
    bool ret = TCOLCMAXTestFramework<93, uint16_t, int16_t, 4, 4, 1, 48, 34>();
    EXPECT_TRUE(ret);
}
