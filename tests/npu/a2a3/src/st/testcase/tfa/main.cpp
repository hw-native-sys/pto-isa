/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/*
 * gtest driver for TFA
 */

#include <gtest/gtest.h>
#include <acl/acl.h>

#include "test_common.h"
#include "runtime/rt.h"

using namespace std;
using namespace PtoTestCommon;

template <int S0, int HEAD_SIZE, int S1, int CUBE_S1 = 128, bool INTERMEDIATE_CHECK = false>
void LaunchTFA(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    aclrtStream stream);

class TFATest : public testing::Test {
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

/*
 * Template usage:
 * - The template parameter `INTERMEDIATE_CHECK` (default false) enables
 *   extra, more-detailed intermediate-value checks. When enabled, the
 *   host will compare the device softmax/intermediate tensor outputs
 *   (e.g. `p_out` / xexp) against golden files. On the device side the
 *   kernel should perform the necessary TSTORE operations to expose
 *   these intermediate buffers for host readback.
 *
 * Example:
 *   run_tfa<float, 64, 128, 256, true>(); // enable intermediate checks
 */
template <typename T, int S0, int HEAD_SIZE, int S1, bool INTERMEDIATE_CHECK = false>
void run_tfa()
{
    size_t fullSize = S0 * S1 * sizeof(T); // Keep output as float
    size_t qSize = S0 * HEAD_SIZE * sizeof(aclFloat16);
    size_t kSize = HEAD_SIZE * S1 * sizeof(aclFloat16);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    T* outHost;
    aclFloat16 *qHost, *kHost;
    aclFloat16* xexpHost;
    float* tmpFloatExpHost;
    aclFloat16* vHost;
    T* outDevice; // qk_out
    aclFloat16* xexpDevice;
    T* midDevice = nullptr; // not used by this test but kept for symmetry
    aclFloat16 *qDevice, *kDevice;
    aclFloat16* vDevice;
    T* out2Device; // pv_out
    T* out2Host;

    aclrtMallocHost((void**)(&outHost), fullSize); // Allocate output buffer
    aclrtMallocHost((void**)(&qHost), qSize);
    aclrtMallocHost((void**)(&kHost), kSize);

    aclrtMalloc((void**)&outDevice, fullSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&qDevice, qSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&kDevice, kSize, ACL_MEM_MALLOC_HUGE_FIRST);
    size_t halfSize = S0 * S1 * sizeof(aclFloat16);
    size_t floatSize = S0 * S1 * sizeof(float);
    aclrtMalloc((void**)&xexpDevice, halfSize, ACL_MEM_MALLOC_HUGE_FIRST); // p_out (half)
    void* pOutFp32Device = nullptr;
    aclrtMalloc((void**)&pOutFp32Device, floatSize, ACL_MEM_MALLOC_HUGE_FIRST); // p_out_fp32 (float)
    // allocate v and out2 buffers
    size_t vSize = S1 * HEAD_SIZE * sizeof(aclFloat16);
    size_t pvPartSize = S0 * HEAD_SIZE * sizeof(T);
    int num_tiles = S1 / 128;
    size_t out2TotalSize = pvPartSize * num_tiles;
    aclrtMalloc((void**)&vDevice, vSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&out2Device, out2TotalSize, ACL_MEM_MALLOC_HUGE_FIRST);
    // allocate global_sum buffer (per-tile S0 floats)
    size_t gsumTotalElems = static_cast<size_t>(S0) * static_cast<size_t>(num_tiles);
    size_t gsumSize = gsumTotalElems * sizeof(float);
    float* gSumDevice = nullptr;
    aclrtMalloc((void**)&gSumDevice, gsumSize, ACL_MEM_MALLOC_HUGE_FIRST);
    // allocate per-tile exp_max buffer (per-tile S0 floats)
    float* expMaxDevice = nullptr;
    aclrtMalloc((void**)&expMaxDevice, gsumSize, ACL_MEM_MALLOC_HUGE_FIRST);
    // allocate running output o (S0 x HEAD_SIZE)
    T* oDevice = nullptr;
    size_t oSize = pvPartSize; // S0 * HEAD_SIZE * sizeof(T)
    aclrtMalloc((void**)&oDevice, oSize, ACL_MEM_MALLOC_HUGE_FIRST);
    // allocate per-iteration running output snapshots (num_tiles * S0 * HEAD_SIZE)
    T* oPartsDevice = nullptr;
    size_t oPartsTotalSize = pvPartSize * num_tiles;
    aclrtMalloc((void**)&oPartsDevice, oPartsTotalSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/q.bin", qSize, qHost, qSize); // Read q data
    ReadFile(GetGoldenDir() + "/kt.bin", kSize, kHost, kSize);
    // read v
    aclrtMallocHost((void**)(&vHost), S1 * HEAD_SIZE * sizeof(aclFloat16));
    ReadFile(GetGoldenDir() + "/v.bin", vSize, vHost, vSize);

    aclrtMemcpy(qDevice, qSize, qHost, qSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(kDevice, kSize, kHost, kSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(vDevice, vSize, vHost, vSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // Debug logging setup (preserve original tqksv behavior)
    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    // logging disabled

    if constexpr (INTERMEDIATE_CHECK) {
        std::cout << "[INFO] Intermediate checking is ENABLED for this run" << std::endl;
    } else {
        std::cout << "[INFO] Intermediate checking is disabled" << std::endl;
    }

    // Launch kernel, pass ffts ctrl addr and device-side log buffer, and xexp/tmp_float_exp device ptrs
    LaunchTFA<S0, HEAD_SIZE, S1, 128, INTERMEDIATE_CHECK>(
        (uint16_t*)ffts, (aclFloat16*)qDevice, (aclFloat16*)kDevice, (aclFloat16*)vDevice, (aclFloat16*)xexpDevice,
        (float*)pOutFp32Device, (float*)gSumDevice, (float*)expMaxDevice, (float*)oDevice, (float*)oPartsDevice,
        (float*)outDevice, (float*)out2Device, stream);

    aclrtSynchronizeStream(stream);

    // copy outputs back
    aclrtMemcpy(outHost, fullSize, outDevice, fullSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMallocHost((void**)(&xexpHost), halfSize);
    aclrtMallocHost((void**)(&tmpFloatExpHost), floatSize);
    aclrtMemcpy(xexpHost, halfSize, xexpDevice, halfSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(tmpFloatExpHost, floatSize, pOutFp32Device, floatSize, ACL_MEMCPY_DEVICE_TO_HOST);
    // copy second matmul partial outputs (concatenated per-tile)
    aclrtMallocHost((void**)(&out2Host), out2TotalSize);
    aclrtMemcpy(out2Host, out2TotalSize, out2Device, out2TotalSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // copy global_sum back
    float* gSumHost = nullptr;
    aclrtMallocHost((void**)(&gSumHost), gsumSize);
    aclrtMemcpy(gSumHost, gsumSize, gSumDevice, gsumSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // copy exp_max back
    float* expMaxHost = nullptr;
    aclrtMallocHost((void**)(&expMaxHost), gsumSize);
    aclrtMemcpy(expMaxHost, gsumSize, expMaxDevice, gsumSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // copy running output o back
    T* oHost = nullptr;
    aclrtMallocHost((void**)(&oHost), oSize);
    aclrtMemcpy(oHost, oSize, oDevice, oSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // copy per-iteration o parts back
    T* oPartsHost = nullptr;
    aclrtMallocHost((void**)(&oPartsHost), oPartsTotalSize);
    aclrtMemcpy(oPartsHost, oPartsTotalSize, oPartsDevice, oPartsTotalSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", outHost, fullSize);
    WriteFile(GetGoldenDir() + "/p_out.bin", xexpHost, halfSize);
    WriteFile(GetGoldenDir() + "/p_out_fp32.bin", tmpFloatExpHost, floatSize);
    WriteFile(GetGoldenDir() + "/out2.bin", out2Host, out2TotalSize);
    // write per-tile global_sum parts
    for (int ti = 0; ti < num_tiles; ++ti) {
        size_t partOffset = static_cast<size_t>(ti) * static_cast<size_t>(S0);
        WriteFile(
            GetGoldenDir() + "/global_sum_part" + std::to_string(ti) + "_out.bin", gSumHost + partOffset,
            S0 * sizeof(float));
    }
    // write per-tile exp_max parts
    for (int ti = 0; ti < num_tiles; ++ti) {
        size_t partOffset = static_cast<size_t>(ti) * static_cast<size_t>(S0);
        WriteFile(
            GetGoldenDir() + "/exp_max_part" + std::to_string(ti) + "_out.bin", expMaxHost + partOffset,
            S0 * sizeof(float));
    }
    // write running output
    WriteFile(GetGoldenDir() + "/o_out.bin", oHost, oSize);
    // write per-iteration running output snapshots
    for (int ti = 0; ti < num_tiles; ++ti) {
        size_t byteOffset = static_cast<size_t>(ti) * pvPartSize;
        WriteFile(
            GetGoldenDir() + "/o_part" + std::to_string(ti) + "_out.bin", ((uint8_t*)oPartsHost) + byteOffset,
            pvPartSize);
    }

    aclrtFree(outDevice);
    aclrtFree(oDevice);
    aclrtFree(oPartsDevice);
    aclrtFree(qDevice);
    aclrtFree(kDevice);
    aclrtFree(xexpDevice);
    aclrtFree(pOutFp32Device);
    aclrtFree(vDevice);
    aclrtFree(out2Device);
    aclrtFree(gSumDevice);
    aclrtFree(expMaxDevice);

    // compare
    bool qk_ok = true;
    if constexpr (INTERMEDIATE_CHECK) {
        std::vector<T> golden(fullSize / sizeof(T));
        std::vector<T> devFinal(fullSize / sizeof(T));
        ReadFile(GetGoldenDir() + "/golden.bin", fullSize, golden.data(), fullSize);
        ReadFile(GetGoldenDir() + "/output.bin", fullSize, devFinal.data(), fullSize);

        std::cout << "[CHECK] QK compare" << std::endl;
        qk_ok = ResultCmp<T>(golden, devFinal, 0.001f);
    }

    // compare xexp (half) - enabled only when INTERMEDIATE_CHECK is true
    if constexpr (INTERMEDIATE_CHECK) {
        std::vector<aclFloat16> golden_xexp(S0 * S1);
        std::vector<aclFloat16> dev_xexp(S0 * S1);
        ReadFile(GetGoldenDir() + "/p.bin", halfSize, golden_xexp.data(), halfSize);
        ReadFile(GetGoldenDir() + "/p_out.bin", halfSize, dev_xexp.data(), halfSize);
        bool ok2 = ResultCmp<aclFloat16>(golden_xexp, dev_xexp, 0.001f);
        std::cout << "softmax ok: " << (ok2 ? "true" : "false") << std::endl;
        EXPECT_TRUE(ok2);
    }

    // compare tmp_float_exp (fp32)
    bool p_fp32_ok = true;
    if constexpr (INTERMEDIATE_CHECK) {
        std::vector<float> golden_tmp((S0 * S1));
        std::vector<float> dev_tmp((S0 * S1));
        ReadFile(GetGoldenDir() + "/p_fp32.bin", floatSize, golden_tmp.data(), floatSize);
        ReadFile(GetGoldenDir() + "/p_out_fp32.bin", floatSize, dev_tmp.data(), floatSize);
        std::cout << "[CHECK] P_fp32 compare" << std::endl;
        p_fp32_ok = ResultCmp<float>(golden_tmp, dev_tmp, 0.001f);
    }

    // compare per-tile PV partials and also combined sum
    int numTiles = S1 / 128;
    // prepare status variables and buffers used by either branch
    bool pv_parts_ok = true;
    std::vector<bool> pv_part_status(numTiles, false);
    std::vector<T> golden_part(S0 * HEAD_SIZE);
    std::vector<T> dev_part(S0 * HEAD_SIZE);
    std::vector<T> dev_sum(S0 * HEAD_SIZE);
    std::fill(dev_sum.begin(), dev_sum.end(), 0);

    bool pv_sum_ok = true;

    // compare per-tile global_sum parts
    bool gsum_parts_ok = true;
    std::vector<bool> gsum_part_status(numTiles, false);
    std::vector<float> golden_gsum(S0);
    std::vector<float> dev_gsum(S0);
    size_t gSum_size = S0 * sizeof(float);

    // compare per-tile exp_max parts
    bool exp_parts_ok = true;
    std::vector<bool> exp_part_status(numTiles, false);
    std::vector<float> golden_exp(S0);
    std::vector<float> dev_exp(S0);

    // compare per-iteration running output snapshots
    bool o_parts_ok = true;
    std::vector<bool> o_part_status(numTiles, false);
    std::vector<float> golden_opart(S0 * HEAD_SIZE);
    std::vector<float> dev_opart(S0 * HEAD_SIZE);

    std::vector<float> golden_o(S0 * HEAD_SIZE);
    std::vector<float> dev_o(S0 * HEAD_SIZE);
    bool o_ok = true;

    if constexpr (INTERMEDIATE_CHECK) {
        for (int ti = 0; ti < numTiles; ++ti) {
            std::string partName = GetGoldenDir() + "/pv_part" + std::to_string(ti) + ".bin";
            ReadFile(partName, pvPartSize, golden_part.data(), pvPartSize);
            // copy corresponding chunk from out2Host
            memcpy(dev_part.data(), ((uint8_t*)out2Host) + ti * pvPartSize, pvPartSize);
            std::cout << "[CHECK] PV part " << ti << " compare" << std::endl;
            bool partOk = ResultCmp<T>(golden_part, dev_part, 0.001f);
            pv_part_status[ti] = partOk;
            if (!partOk)
                pv_parts_ok = false;
            // accumulate into dev_sum
            for (size_t i = 0; i < dev_sum.size(); ++i)
                dev_sum[i] += dev_part[i];
        }
        EXPECT_TRUE(pv_parts_ok);
        // compare accumulated sum to pv.bin
        std::vector<T> golden_pv(S0 * HEAD_SIZE);
        ReadFile(GetGoldenDir() + "/pv.bin", pvPartSize, golden_pv.data(), pvPartSize);
        std::cout << "[CHECK] PV sum compare" << std::endl;
        pv_sum_ok = ResultCmp<T>(golden_pv, dev_sum, 0.001f);
        EXPECT_TRUE(pv_sum_ok);

        for (int ti = 0; ti < numTiles; ++ti) {
            std::cout << "[CHECK] Global sum " + std::to_string(ti) << std::endl;
            std::string gname = GetGoldenDir() + "/global_sum_part" + std::to_string(ti) + ".bin";
            ReadFile(gname, gSum_size, golden_gsum.data(), gSum_size);
            memcpy(dev_gsum.data(), ((uint8_t*)gSumHost) + ti * gSum_size, gSum_size);
            bool okg = ResultCmp<float>(golden_gsum, dev_gsum, 0.001f);
            gsum_part_status[ti] = okg;
            if (!okg)
                gsum_parts_ok = false;
        }
        EXPECT_TRUE(gsum_parts_ok);

        // ti=0 no meaning for exp_max
        for (int ti = 1; ti < numTiles; ++ti) {
            std::cout << "[CHECK] expmax part " + std::to_string(ti) << std::endl;
            std::string ename = GetGoldenDir() + "/exp_max_part" + std::to_string(ti) + ".bin";
            ReadFile(ename, gSum_size, golden_exp.data(), gSum_size);
            memcpy(dev_exp.data(), ((uint8_t*)expMaxHost) + ti * gSum_size, gSum_size);
            bool oke = ResultCmp<float>(golden_exp, dev_exp, 0.001f);
            exp_part_status[ti] = oke;
            if (!oke)
                exp_parts_ok = false;
        }
        EXPECT_TRUE(exp_parts_ok);

        for (int ti = 0; ti < numTiles; ++ti) {
            std::cout << "[CHECK] O part " + std::to_string(ti) << std::endl;
            std::string oname = GetGoldenDir() + "/o_part" + std::to_string(ti) + ".bin";
            ReadFile(oname, pvPartSize, golden_opart.data(), pvPartSize);
            memcpy(dev_opart.data(), ((uint8_t*)oPartsHost) + ti * pvPartSize, pvPartSize);
            bool ok = ResultCmp<float>(golden_opart, dev_opart, 0.001f);
            o_part_status[ti] = ok;
            if (!ok)
                o_parts_ok = false;
        }
        EXPECT_TRUE(o_parts_ok);

        // Final running output compare is performed unconditionally after
        // the INTERMEDIATE_CHECK block so we don't do it here.
    } else {
        // Intermediate checking disabled: mark intermediate checks as passed
        pv_parts_ok = true;
        std::fill(pv_part_status.begin(), pv_part_status.end(), true);
        pv_sum_ok = true;
        gsum_parts_ok = true;
        std::fill(gsum_part_status.begin(), gsum_part_status.end(), true);
        exp_parts_ok = true;
        std::fill(exp_part_status.begin(), exp_part_status.end(), true);
        o_parts_ok = true;
        std::fill(o_part_status.begin(), o_part_status.end(), true);
        // leave o_ok to be computed unconditionally below
    }

    // Final running output compare (always check correctness)
    ReadFile(GetGoldenDir() + "/o.bin", oSize, golden_o.data(), oSize);
    ReadFile(GetGoldenDir() + "/o_out.bin", oSize, dev_o.data(), oSize);
    std::cout << "[CHECK] O running output compare" << std::endl;
    o_ok = ResultCmp<float>(golden_o, dev_o, 0.001f);
    EXPECT_TRUE(o_ok);

    // Summary printout (single line)
    constexpr bool intermediateEnabled = INTERMEDIATE_CHECK;
    auto summary_status = [](bool enabled, bool ok) -> const char* { return enabled ? (ok ? "OK" : "FAIL") : "SKIP"; };

    std::cout << "Summary: "
              << "QK=" << summary_status(intermediateEnabled, qk_ok) << ", "
              << "P_fp32=" << summary_status(intermediateEnabled, p_fp32_ok) << ", "
              << "PV_parts=" << summary_status(intermediateEnabled, pv_parts_ok) << " [";
    for (int ti = 0; ti < numTiles; ++ti) {
        std::cout << (intermediateEnabled ? (pv_part_status[ti] ? "OK" : "FAIL") : "SKIP");
        if (ti != numTiles - 1)
            std::cout << ",";
    }
    std::cout << "]"
              << ", "
              << "PV_sum=" << summary_status(intermediateEnabled, pv_sum_ok) << ", "
              << "GSum_parts=" << summary_status(intermediateEnabled, gsum_parts_ok) << " [";
    for (int ti = 0; ti < numTiles; ++ti) {
        std::cout << (intermediateEnabled ? (gsum_part_status[ti] ? "OK" : "FAIL") : "SKIP");
        if (ti != numTiles - 1)
            std::cout << ",";
    }
    std::cout << "]"
              << ", "
              << "Exp_parts=" << summary_status(intermediateEnabled, exp_parts_ok) << " [";
    // ti=0 no meaning for exp_max
    for (int ti = 1; ti < numTiles; ++ti) {
        std::cout << (intermediateEnabled ? (exp_part_status[ti] ? "OK" : "FAIL") : "SKIP");
        if (ti != numTiles - 1)
            std::cout << ",";
    }
    std::cout << "]"
              << ", "
              << "O_parts=" << summary_status(intermediateEnabled, o_parts_ok) << " [";
    for (int ti = 0; ti < numTiles; ++ti) {
        std::cout << (intermediateEnabled ? (o_part_status[ti] ? "OK" : "FAIL") : "SKIP");
        if (ti != numTiles - 1)
            std::cout << ",";
    }
    std::cout << "]"
              << ", "
              << "O=" << (o_ok ? "OK" : "FAIL") << std::endl;

    aclrtFreeHost(outHost); // Free host memory
    aclrtFreeHost(qHost);
    aclrtFreeHost(kHost);
    aclrtFreeHost(xexpHost);
    aclrtFreeHost(tmpFloatExpHost);
    aclrtFreeHost(vHost);
    aclrtFreeHost(out2Host);
    aclrtFreeHost(oHost);
    aclrtFreeHost(oPartsHost);
    aclrtFreeHost(gSumHost);
    aclrtFreeHost(expMaxHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TFATest, case_float_H_128_S0_64_S1_256) { run_tfa<float, 64, 128, 256>(); }

TEST_F(TFATest, case_float_H_128_S0_64_S1_128) { run_tfa<float, 64, 128, 128>(); }

TEST_F(TFATest, case_float_H_128_S0_64_S1_512) { run_tfa<float, 64, 128, 512>(); }

TEST_F(TFATest, case_float_H_128_S0_128_S1_512) { run_tfa<float, 128, 128, 512>(); }

TEST_F(TFATest, case_float_H_128_S0_128_S1_2048) { run_tfa<float, 128, 128, 2048>(); }

TEST_F(TFATest, case_float_H_128_S0_128_S1_8192) { run_tfa<float, 128, 128, 8192>(); }

// Debug test that enables intermediate-value checks (expects device to
// expose intermediate buffers via TSTORE so host can read and compare).
// NOTE: test name intentionally uses "64x512_precision_debug" while
// the run uses S1=256 — this mirrors the requested debug naming.
TEST_F(TFATest, case_float_H_128_S0_64_S1_128_precision_debug) { run_tfa<float, 64, 128, 128, true>(); }

TEST_F(TFATest, case_float_H_128_S0_64_S1_256_precision_debug) { run_tfa<float, 64, 128, 256, true>(); }

TEST_F(TFATest, case_float_H_128_S0_64_S1_512_precision_debug) { run_tfa<float, 64, 128, 512, true>(); }

TEST_F(TFATest, case_float_H_128_S0_128_S1_512_precision_debug) { run_tfa<float, 128, 128, 512, true>(); }

TEST_F(TFATest, case_float_H_128_S0_128_S1_2048_precision_debug) { run_tfa<float, 128, 128, 2048, true>(); }

TEST_F(TFATest, case_float_H_128_S0_128_S1_8192_precision_debug) { run_tfa<float, 128, 128, 8192, true>(); }
