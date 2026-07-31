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
#include <securec.h>
#include <string>
#include <vector>
#include <pto/common/type.hpp>
#include "acl/acl.h"
#include "test_common.h"

using namespace PtoTestCommon;

namespace TmatmulMxE1m2 {
template <typename OutT, int validM, int validK, int validN>
void Launch(uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
} // namespace TmatmulMxE1m2

class TMATMUL_MX_E1M2_TEST : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

namespace {

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

// BF16 -> FP32 bitcast (uint16 << 16). fixpipe casts L0C FP32 -> BF16 on store,
// so the GM output is BF16; ResultCmp needs float for relative-error semantics.
std::vector<float> Bf16BytesToFloat(const uint8_t* raw, int n)
{
    std::vector<float> v(n);
    const auto* u16 = reinterpret_cast<const uint16_t*>(raw);
    for (int i = 0; i < n; i++) {
        uint32_t bits = static_cast<uint32_t>(u16[i]) << 16;
        if (memcpy_s(&v[i], sizeof(float), &bits, sizeof(bits)) != EOK) {
            return {};
        }
    }
    return v;
}

// Upload one host buffer to device memory.
uint8_t* UploadToDevice(const std::vector<uint8_t>& host)
{
    uint8_t* dev = nullptr;
    aclrtMalloc((void**)&dev, host.size(), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dev, host.size(), host.data(), host.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    return dev;
}

// Load an input bin into a host buffer.
std::vector<uint8_t> ReadInput(const std::string& path, size_t bytes)
{
    std::vector<uint8_t> host(bytes);
    size_t read = bytes;
    ReadFile(path, read, host.data(), bytes);
    return host;
}

} // namespace

template <int validM, int validK, int validN>
void RunE1m2MxMatmulCase(const std::string& goldenDir)
{
    constexpr int totalA = validM * validK;
    constexpr int totalB = validK * validN;
    constexpr int totalOut = validM * validN;
    // MX: per-32-element e8m0 scale. A[M,K] = M*(K/32) bytes; B[K,N] = (K/32)*N bytes.
    constexpr size_t aScaleBytes = totalA / 32;
    constexpr size_t bScaleBytes = totalB / 32;
    constexpr size_t outBytes = totalOut * sizeof(uint16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    auto aDataHost = ReadInput(goldenDir + "/a_data.bin", totalA / 2);
    auto aScaleHost = ReadInput(goldenDir + "/a_scale.bin", aScaleBytes);
    auto bDataHost = ReadInput(goldenDir + "/b_data.bin", totalB / 2);
    auto bScaleHost = ReadInput(goldenDir + "/b_scale.bin", bScaleBytes);

    uint8_t* aDataDev = UploadToDevice(aDataHost);
    uint8_t* aScaleDev = UploadToDevice(aScaleHost);
    uint8_t* bDataDev = UploadToDevice(bDataHost);
    uint8_t* bScaleDev = UploadToDevice(bScaleHost);
    uint8_t* outDev = nullptr;
    aclrtMalloc((void**)&outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    TmatmulMxE1m2::Launch<uint16_t, validM, validK, validN>(outDev, aDataDev, aScaleDev, bDataDev, bScaleDev, stream);

    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed (ret=" << syncRet
                                    << "): " << aclGetRecentErrMsg();

    std::vector<uint8_t> outHost(outBytes);
    std::vector<uint8_t> goldenHost(outBytes);
    aclrtMemcpy(outHost.data(), outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(goldenDir + "/output.bin", outHost.data(), outBytes);
    size_t goldenRead = outBytes;
    ReadFile(goldenDir + "/golden_out.bin", goldenRead, goldenHost.data(), outBytes);

    auto outVals = Bf16BytesToFloat(outHost.data(), totalOut);
    auto goldenVals = Bf16BytesToFloat(goldenHost.data(), totalOut);
    // e1m2 MX is bit-exact (no quantization hierarchy noise). Use tight tolerance.
    EXPECT_TRUE(ResultCmp<float>(goldenVals, outVals, 0.01f)) << "e1m2 MX matmul output mismatch";

    aclrtFree(outDev);
    aclrtFree(bScaleDev);
    aclrtFree(bDataDev);
    aclrtFree(aScaleDev);
    aclrtFree(aDataDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TMATMUL_MX_E1M2_TEST, case_e1m2_128x128x128_nd) { RunE1m2MxMatmulCase<128, 128, 128>(GetGoldenDir()); }
