/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include <vector>
#include <pto/common/type.hpp>
#include "acl/acl.h"
#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

namespace TloadHif4A6 {
template <int validM, int validK, int validN>
void Launch(uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
} // namespace TloadHif4A6

class TLOAD_HIF4_A6_TEST : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

// Minimal TLOAD-only harness: load the four GM inputs into L1, sync, and return.
// No output buffer is read back — the test passes if the kernel launches and
// aclrtSynchronizeStream returns ACL_SUCCESS (meaning no DMA faults occurred).
// The sim log's l1.wr_log.dump is the actual verification artifact.
template <int validM, int validK, int validN>
void RunTloadHif4Case(const std::string& goldenDir)
{
    constexpr int totalA = validM * validK;
    constexpr int totalB = validK * validN;

    constexpr size_t aDataBytes = totalA / 2; // hifloat4x2_t: 2 nibbles per byte
    constexpr size_t bDataBytes = totalB / 2;
    constexpr size_t aScaleBytes = (totalA / 64) * 4; // HIF4: 4B per 64-element group
    constexpr size_t bScaleBytes = (totalB / 64) * 4;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    std::vector<uint8_t> aDataHost(aDataBytes);
    std::vector<uint8_t> aScaleHost(aScaleBytes);
    std::vector<uint8_t> bDataHost(bDataBytes);
    std::vector<uint8_t> bScaleHost(bScaleBytes);
    size_t aDataRead = aDataBytes;
    size_t aScaleRead = aScaleBytes;
    size_t bDataRead = bDataBytes;
    size_t bScaleRead = bScaleBytes;
    ReadFile(goldenDir + "/a_data.bin", aDataRead, aDataHost.data(), aDataBytes);
    ReadFile(goldenDir + "/a_scale.bin", aScaleRead, aScaleHost.data(), aScaleBytes);
    ReadFile(goldenDir + "/b_data.bin", bDataRead, bDataHost.data(), bDataBytes);
    ReadFile(goldenDir + "/b_scale.bin", bScaleRead, bScaleHost.data(), bScaleBytes);

    uint8_t *aDataDev = nullptr, *aScaleDev = nullptr;
    uint8_t *bDataDev = nullptr, *bScaleDev = nullptr;
    aclrtMalloc((void**)&aDataDev, aDataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&aScaleDev, aScaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&bDataDev, bDataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&bScaleDev, bScaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(aDataDev, aDataBytes, aDataHost.data(), aDataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(aScaleDev, aScaleBytes, aScaleHost.data(), aScaleBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(bDataDev, bDataBytes, bDataHost.data(), bDataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(bScaleDev, bScaleBytes, bScaleHost.data(), bScaleBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    TloadHif4A6::Launch<validM, validK, validN>(aDataDev, aScaleDev, bDataDev, bScaleDev, stream);

    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed (ret=" << syncRet
                                    << "): " << aclGetRecentErrMsg();

    aclrtFree(bScaleDev);
    aclrtFree(bDataDev);
    aclrtFree(aScaleDev);
    aclrtFree(aDataDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TLOAD_HIF4_A6_TEST, case_hif4_128x128x128_nd) { RunTloadHif4Case<128, 128, 128>(GetGoldenDir()); }
