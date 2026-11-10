/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <securec.h>

using namespace std;
using namespace PtoTestCommon;

namespace TmatmulMxA6 {
template <int caseId>
void Launch(uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
} // namespace TmatmulMxA6

class TMATMUL_MX_A6_TEST : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

// A/B dtype kinds. Byte sizes:
//   fp4 (e1m2/e2m1/hif4): 0.5 B/elem (packed).  fp8: 1 B/elem.  fp16/bf16: 2 B/elem.
// A-scale: e8m0 (non-hif4 A) = M*K/32.  hif4 A = (M*K/64)*4.
// B-scale: e8m0 (e2m1/e1m2 B) = K*N/32.  hif4 B = (K*N/64)*4.
enum class AKind : uint8_t { E1M2, E2M1, E4M3, F16, BF16, HIF4, HIF4_ND };
enum class BKind : uint8_t { E1M2, E2M1, HIF4, HIF4_DN };

struct CaseConfig {
    int m, k, n;
    AKind aKind;
    BKind bKind;
};

template <int caseId>
constexpr CaseConfig GetCaseConfig();

#define CFG(ID, M, K, N, AK, BK)                \
    template <>                                 \
    constexpr CaseConfig GetCaseConfig<ID>()    \
    {                                           \
        return {M, K, N, AKind::AK, BKind::BK}; \
    }

CFG(1, 128, 128, 128, E1M2, E1M2)
CFG(2, 64, 64, 64, E1M2, E1M2)
CFG(3, 128, 128, 128, E2M1, E2M1)
CFG(4, 128, 128, 128, E1M2, E2M1)
CFG(5, 128, 128, 128, E2M1, E1M2)
CFG(6, 128, 128, 128, E4M3, E2M1)
CFG(7, 64, 128, 64, E4M3, E2M1)
CFG(8, 128, 128, 128, F16, E2M1)
CFG(9, 64, 128, 64, F16, E2M1)
CFG(10, 128, 128, 128, BF16, E2M1)
CFG(11, 64, 128, 64, BF16, E2M1)
CFG(12, 128, 128, 128, E4M3, HIF4)
CFG(13, 64, 128, 64, E4M3, HIF4)
CFG(14, 128, 128, 128, F16, HIF4)
CFG(15, 64, 128, 64, F16, HIF4)
CFG(16, 128, 128, 128, BF16, HIF4)
CFG(17, 64, 128, 64, BF16, HIF4)
CFG(18, 128, 128, 128, HIF4, HIF4)
CFG(19, 128, 256, 128, HIF4, HIF4)
CFG(20, 256, 128, 128, HIF4, HIF4)
CFG(21, 64, 64, 64, HIF4, HIF4)
CFG(22, 256, 256, 256, HIF4, HIF4)
CFG(23, 128, 512, 128, HIF4, HIF4)
CFG(24, 512, 128, 512, HIF4, HIF4)
CFG(25, 128, 128, 256, HIF4, HIF4)
CFG(26, 256, 128, 512, HIF4, HIF4)

CFG(27, 1, 256, 64, E4M3, E2M1)
CFG(28, 1, 256, 64, F16, E2M1)
CFG(29, 1, 256, 64, BF16, HIF4)
CFG(30, 64, 128, 64, E2M1, E2M1)
CFG(31, 64, 64, 64, E1M2, E2M1)
CFG(32, 64, 64, 64, E2M1, E1M2)
CFG(33, 128, 256, 128, E4M3, E2M1)
CFG(34, 128, 256, 128, F16, HIF4)
CFG(35, 128, 128, 256, E4M3, HIF4)
CFG(36, 128, 128, 128, HIF4_ND, HIF4_DN)
CFG(37, 128, 256, 128, HIF4_ND, HIF4_DN)
CFG(38, 256, 128, 128, HIF4_ND, HIF4_DN)
CFG(39, 128, 128, 256, HIF4_ND, HIF4_DN)

#undef CFG

static constexpr size_t elemBytes(AKind k)
{
    return (k == AKind::E4M3) ? 1 : (k == AKind::F16 || k == AKind::BF16) ? 2 : 0; // 0 => fp4 packed (0.5 B)
}

static size_t aDataBytes(AKind k, int m, int kk)
{
    size_t elems = static_cast<size_t>(m) * kk;
    size_t eb = elemBytes(k);
    return eb ? elems * eb : elems / 2;
}

static size_t bDataBytes(BKind k, int kk, int n)
{
    size_t elems = static_cast<size_t>(kk) * n;
    return elems / 2; // B is always fp4-family (e1m2/e2m1/hif4): 0.5 B/elem packed
}

// Scale tile byte sizes. The MX_A_ZZ (A) / MX_B_NN (B) / HIF4 fractal layouts
// pad the M-axis (A) and N-axis (B) up to 16 rows, so the on-GM scale buffer
// is larger than the raw M*K/32 element count for small M/N. These must match
// what gen_data.py writes (convert_x1/x2_scale_format pads to block_size=16).
static size_t aScaleBytes(AKind k, int m, int kk)
{
    size_t mPadded = static_cast<size_t>((m + 15) / 16) * 16;
    return (k == AKind::HIF4 || k == AKind::HIF4_ND) ? (mPadded * kk / 64) * 4 : mPadded * kk / 32;
}

static size_t bScaleBytes(BKind k, int kk, int n)
{
    size_t nPadded = static_cast<size_t>((n + 15) / 16) * 16;
    return (k == BKind::HIF4 || k == BKind::HIF4_DN) ? (kk * nPadded / 64) * 4 : kk * nPadded / 32;
}

std::vector<float> Bf16BytesToFloat(const uint8_t* raw, int n)
{
    std::vector<float> v(n);
    const auto* u16 = reinterpret_cast<const uint16_t*>(raw);
    for (int i = 0; i < n; i++) {
        uint32_t bits = static_cast<uint32_t>(u16[i]) << 16;
        memcpy_s(&v[i], sizeof(float), &bits, sizeof(bits));
    }
    return v;
}

struct CaseBuffers {
    uint8_t *aDataHost, *aScaleHost, *bDataHost, *bScaleHost;
    uint8_t *aDataDev, *aScaleDev, *bDataDev, *bScaleDev, *outDev;
    aclrtStream stream;
};

void UploadInputs(
    CaseBuffers& buf, const std::string& goldenDir, size_t aBytes, size_t bBytes, size_t asBytes, size_t bsBytes,
    size_t outBytes)
{
    aclrtMallocHost((void**)(&buf.aDataHost), aBytes);
    aclrtMallocHost((void**)(&buf.aScaleHost), asBytes);
    aclrtMallocHost((void**)(&buf.bDataHost), bBytes);
    aclrtMallocHost((void**)(&buf.bScaleHost), bsBytes);
    aclrtMalloc((void**)&buf.aDataDev, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&buf.aScaleDev, asBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&buf.bDataDev, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&buf.bScaleDev, bsBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&buf.outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t rd = aBytes;
    CHECK_RESULT_GTEST(ReadFile(goldenDir + "/a_data.bin", rd, buf.aDataHost, aBytes));
    rd = asBytes;
    CHECK_RESULT_GTEST(ReadFile(goldenDir + "/a_scale.bin", rd, buf.aScaleHost, asBytes));
    rd = bBytes;
    CHECK_RESULT_GTEST(ReadFile(goldenDir + "/b_data.bin", rd, buf.bDataHost, bBytes));
    rd = bsBytes;
    CHECK_RESULT_GTEST(ReadFile(goldenDir + "/b_scale.bin", rd, buf.bScaleHost, bsBytes));
    aclrtMemcpy(buf.aDataDev, aBytes, buf.aDataHost, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(buf.aScaleDev, asBytes, buf.aScaleHost, asBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(buf.bDataDev, bBytes, buf.bDataHost, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(buf.bScaleDev, bsBytes, buf.bScaleHost, bsBytes, ACL_MEMCPY_HOST_TO_DEVICE);
}

void FreeBuffers(CaseBuffers& buf)
{
    aclrtFree(buf.outDev);
    aclrtFree(buf.bScaleDev);
    aclrtFree(buf.bDataDev);
    aclrtFree(buf.aScaleDev);
    aclrtFree(buf.aDataDev);
    aclrtFreeHost(buf.bScaleHost);
    aclrtFreeHost(buf.bDataHost);
    aclrtFreeHost(buf.aScaleHost);
    aclrtFreeHost(buf.aDataHost);
}

template <int caseId>
void RunCase(const std::string& goldenDir)
{
    constexpr auto cfg = GetCaseConfig<caseId>();
    const int m = cfg.m, k = cfg.k, n = cfg.n;
    const int totalOutElems = m * n;
    const size_t aBytes = aDataBytes(cfg.aKind, m, k);
    const size_t bBytes = bDataBytes(cfg.bKind, k, n);
    const size_t asBytes = aScaleBytes(cfg.aKind, m, k);
    const size_t bsBytes = bScaleBytes(cfg.bKind, k, n);
    const size_t outBytes = static_cast<size_t>(totalOutElems) * sizeof(uint16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    CaseBuffers buf;
    aclrtCreateStream(&buf.stream);
    UploadInputs(buf, goldenDir, aBytes, bBytes, asBytes, bsBytes, outBytes);

    TmatmulMxA6::Launch<caseId>(buf.outDev, buf.aDataDev, buf.aScaleDev, buf.bDataDev, buf.bScaleDev, buf.stream);

    aclError syncRet = aclrtSynchronizeStream(buf.stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed (ret=" << syncRet
                                    << "): " << aclGetRecentErrMsg();

    std::vector<uint8_t> outHost(outBytes);
    aclrtMemcpy(outHost.data(), outBytes, buf.outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(goldenDir + "/output.bin", outHost.data(), outBytes);

    auto outVals = Bf16BytesToFloat(outHost.data(), totalOutElems);
    std::vector<uint8_t> goldenHost(outBytes);
    size_t goldenRead = outBytes;
    CHECK_RESULT_GTEST(ReadFile(goldenDir + "/golden_out.bin", goldenRead, goldenHost.data(), outBytes));
    auto goldenVals = Bf16BytesToFloat(goldenHost.data(), totalOutElems);
    EXPECT_TRUE(ResultCmp<float>(goldenVals, outVals, 0.03f)) << "TMATMUL_MX case mismatch";

    FreeBuffers(buf);
    aclrtDestroyStream(buf.stream);
    aclrtResetDevice(0);
    aclFinalize();
}

#define CASE(ID, NAME) \
    TEST_F(TMATMUL_MX_A6_TEST, NAME) { RunCase<ID>(GetGoldenDir()); }

CASE(1, case_mmad_mx_e1m2e1m2_128x128x128)
CASE(2, case_mmad_mx_e1m2e1m2_64x64x64)
CASE(3, case_mmad_mx_e2m1e2m1_128x128x128)
CASE(4, case_mmad_mx_e1m2e2m1_128x128x128)
CASE(5, case_mmad_mx_e2m1e1m2_128x128x128)
CASE(6, case_mmad_mx_e4m3e2m1_128x128x128)
CASE(7, case_mmad_mx_e4m3e2m1_64x128x64)
CASE(8, case_mmad_mx_fp16e2m1_128x128x128)
CASE(9, case_mmad_mx_fp16e2m1_64x128x64)
CASE(10, case_mmad_mx_bf16e2m1_128x128x128)
CASE(11, case_mmad_mx_bf16e2m1_64x128x64)
CASE(12, case_mmad_mx_e4m3hi4_128x128x128)
CASE(13, case_mmad_mx_e4m3hi4_64x128x64)
CASE(14, case_mmad_mx_fp16hi4_128x128x128)
CASE(15, case_mmad_mx_fp16hi4_64x128x64)
CASE(16, case_mmad_mx_bf16hi4_128x128x128)
CASE(17, case_mmad_mx_bf16hi4_64x128x64)
CASE(18, case_mmad_mx_hif4hif4_128x128x128)
CASE(19, case_mmad_mx_hif4hif4_128x256x128)
CASE(20, case_mmad_mx_hif4hif4_256x128x128)
CASE(21, case_mmad_mx_hif4hif4_64x64x64)
CASE(22, case_mmad_mx_hif4hif4_256x256x256)
CASE(23, case_mmad_mx_hif4hif4_128x512x128)
CASE(24, case_mmad_mx_hif4hif4_512x128x512)
CASE(25, case_mmad_mx_hif4hif4_128x128x256)
CASE(26, case_mmad_mx_hif4hif4_256x128x512)

CASE(27, case_mmad_mx_e4m3e2m1_1x256x64_gemv)
CASE(28, case_mmad_mx_fp16e2m1_1x256x64_gemv)
CASE(29, case_mmad_mx_bf16hi4_1x256x64_gemv)
CASE(30, case_mmad_mx_e2m1e2m1_64x128x64)
CASE(31, case_mmad_mx_e1m2e2m1_64x64x64)
CASE(32, case_mmad_mx_e2m1e1m2_64x64x64)
CASE(33, case_mmad_mx_e4m3e2m1_128x256x128)
CASE(34, case_mmad_mx_fp16hi4_128x256x128)
CASE(35, case_mmad_mx_e4m3hi4_128x128x256)
CASE(36, case_mmad_mx_hif4hif4_nd2zz_dn2nn_128x128x128)
CASE(37, case_mmad_mx_hif4hif4_nd2zz_dn2nn_128x256x128)
CASE(38, case_mmad_mx_hif4hif4_nd2zz_dn2nn_256x128x128)
CASE(39, case_mmad_mx_hif4hif4_nd2zz_dn2nn_128x128x256)

#undef CASE
