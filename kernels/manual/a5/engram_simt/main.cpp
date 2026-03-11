/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <acl/acl.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

#include "test_common.h"
#include "generated_cases.h"

using namespace PtoTestCommon;

template <int kEmbDim, int kBlockSize>
void LaunchEngramBaseline(float *out, float *table, int32_t *indices, float *hid, float *gw, int tableRows,
                          void *stream);

template <int kEmbDim, int kBlockSize>
void LaunchEngramFused(float *out, float *table, int32_t *indices, float *hid, float *gw, int tableRows, void *stream);

enum class Variant
{
    Baseline,
    Fused
};

static int g_chip_id = 0;

static bool FullResultCmp(const std::vector<float> &expected, const std::vector<float> &actual, float eps = 0.001f)
{
    if (expected.size() != actual.size()) {
        printf("[FullResultCmp] size mismatch: expected %zu, actual %zu\n", expected.size(), actual.size());
        return false;
    }
    const size_t n = expected.size();
    int errCount = 0;
    float maxDiff = 0.0f;
    size_t maxIdx = 0;
    size_t firstErrIdx = n;
    for (size_t i = 0; i < n; ++i) {
        float diff = std::fabs(expected[i] - actual[i]);
        float relDenom = std::max(std::fabs(expected[i]), 1e-8f);
        float rel = diff / relDenom;
        if (diff > eps && rel > eps) {
            if (errCount < 5) {
                printf("  ERR[%zu]: exp=%.6e act=%.6e diff=%.6e\n", i, expected[i], actual[i], diff);
            }
            if (firstErrIdx == n)
                firstErrIdx = i;
            ++errCount;
        }
        if (diff > maxDiff) {
            maxDiff = diff;
            maxIdx = i;
        }
    }
    int threshold = static_cast<int>(n * eps);
    printf(
        "[FullResultCmp] elems=%zu, err_count=%d, threshold=%d, max_diff=%.6e at [%zu] (exp=%.6e act=%.6e) "
        "first_err=%zu\n",
        n, errCount, threshold, maxDiff, maxIdx, maxIdx < n ? expected[maxIdx] : 0.0f,
        maxIdx < n ? actual[maxIdx] : 0.0f, firstErrIdx);
    return errCount <= threshold;
}

template <int kEmbDim, int kBlockSize, Variant V>
static bool RunOneCase(const std::string &caseName, int tableRows)
{
    static_assert(kEmbDim == 128 || kEmbDim == 256 || kEmbDim == 512 || kEmbDim == 1024,
                  "EmbDim must be 128, 256, 512, or 1024");
    static_assert(kBlockSize >= 1 && kBlockSize <= 64 && (kBlockSize & (kBlockSize - 1)) == 0,
                  "BlockSize must be power-of-2 in [1, 64]");
    static_assert(std::is_same_v<float, float>, "only float dtype supported");

    constexpr int H = 8;
    constexpr int D = kEmbDim;
    constexpr int B = kBlockSize;

    const std::string gDir = std::string("../data/") + caseName;
    const size_t tElems = (size_t)tableRows * D;
    const size_t tBytes = tElems * sizeof(float);
    constexpr size_t idxBytes = (size_t)B * H * sizeof(int32_t);
    constexpr size_t hidBytes = (size_t)B * D * sizeof(float);
    constexpr size_t outBytes = hidBytes;

    aclInit(nullptr);
    aclrtSetDevice(g_chip_id);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *hTable, *hHidden, *hGateW, *hOutput;
    int32_t *hIdx;
    aclrtMallocHost((void **)&hTable, tBytes);
    aclrtMallocHost((void **)&hIdx, idxBytes);
    aclrtMallocHost((void **)&hHidden, hidBytes);
    aclrtMallocHost((void **)&hGateW, hidBytes);
    aclrtMallocHost((void **)&hOutput, outBytes);

    float *dTable, *dHidden, *dGateW, *dOutput;
    int32_t *dIdx;
    aclrtMalloc((void **)&dTable, tBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dIdx, idxBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dHidden, hidBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dGateW, hidBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dOutput, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    for (size_t r = 0; r < (size_t)tableRows; ++r)
        for (size_t c = 0; c < (size_t)D; ++c)
            hTable[r * D + c] = (float)(((r + c) % 8) + 1) * 0.0625f;

    size_t fSz = 0;
    ReadFile(gDir + "/indices.bin", fSz, hIdx, idxBytes);
    ReadFile(gDir + "/hidden.bin", fSz, hHidden, hidBytes);
    ReadFile(gDir + "/gate_weight.bin", fSz, hGateW, hidBytes);

    aclrtMemcpy(dTable, tBytes, hTable, tBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dIdx, idxBytes, hIdx, idxBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dHidden, hidBytes, hHidden, hidBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dGateW, hidBytes, hGateW, hidBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    if constexpr (V == Variant::Baseline) {
        LaunchEngramBaseline<D, B>(dOutput, dTable, dIdx, dHidden, dGateW, tableRows, stream);
    } else {
        LaunchEngramFused<D, B>(dOutput, dTable, dIdx, dHidden, dGateW, tableRows, stream);
    }
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(hOutput, outBytes, dOutput, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> golden(B * D);
    std::vector<float> actual(B * D);
    ReadFile(gDir + "/golden.bin", fSz, golden.data(), outBytes);
    memcpy(actual.data(), hOutput, outBytes);

    bool pass = FullResultCmp(golden, actual, 0.001f);

    aclrtFree(dTable);
    aclrtFree(dIdx);
    aclrtFree(dHidden);
    aclrtFree(dGateW);
    aclrtFree(dOutput);
    aclrtFreeHost(hTable);
    aclrtFreeHost(hIdx);
    aclrtFreeHost(hHidden);
    aclrtFreeHost(hGateW);
    aclrtFreeHost(hOutput);
    aclrtDestroyStream(stream);
    aclrtResetDevice(g_chip_id);
    aclFinalize();

    return pass;
}

int main(int argc, char **argv)
{
    struct CaseEntry {
        std::string name;
        std::function<bool()> run;
    };

    std::vector<CaseEntry> cases = {
#define ENGRAM_CASE_ENTRY(D, B, T, TAG)                                                                  \
    {"ENGRAMSIMTTest.baseline_E" #D "_B" #B "_" #TAG,                                                    \
     []() -> bool {                                                                                      \
         return RunOneCase<D, B, Variant::Baseline>("ENGRAMSIMTTest.baseline_E" #D "_B" #B "_" #TAG, T); \
     }},                                                                                                 \
        {"ENGRAMSIMTTest.fused_E" #D "_B" #B "_" #TAG,                                                   \
         []() -> bool { return RunOneCase<D, B, Variant::Fused>("ENGRAMSIMTTest.fused_E" #D "_B" #B "_" #TAG, T); }},
        ENGRAM_FOR_EACH_CASE(ENGRAM_CASE_ENTRY)
#undef ENGRAM_CASE_ENTRY
    };

    std::vector<std::string> filters;
    std::string filter_arg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--case=", 0) == 0) {
            filter_arg = arg.substr(7);
            continue;
        }
        if (arg.rfind("--cases=", 0) == 0) {
            filter_arg = arg.substr(8);
            continue;
        }
        if (arg.rfind("--npu=", 0) == 0) {
            g_chip_id = std::stoi(arg.substr(6));
            continue;
        }
        if ((arg == "--case" || arg == "--cases") && (i + 1) < argc) {
            filter_arg = argv[++i];
            continue;
        }
        if ((arg == "--npu" || arg == "-n") && (i + 1) < argc) {
            g_chip_id = std::stoi(argv[++i]);
            continue;
        }
    }

    if (!filter_arg.empty()) {
        size_t start = 0;
        while (start < filter_arg.size()) {
            size_t end = filter_arg.find(';', start);
            if (end == std::string::npos)
                end = filter_arg.size();
            std::string tok = filter_arg.substr(start, end - start);
            if (!tok.empty())
                filters.push_back(tok);
            start = end + 1;
        }
    }

    auto should_run = [&](const std::string &name) {
        if (filters.empty())
            return true;
        for (const auto &f : filters) {
            if (name.find(f) != std::string::npos)
                return true;
        }
        return false;
    };

    printf("[engram-simt] Available cases (%zu):\n", cases.size());
    for (const auto &c : cases) {
        printf("  %s\n", c.name.c_str());
    }

    std::vector<const CaseEntry *> to_run;
    for (const auto &c : cases) {
        if (should_run(c.name))
            to_run.push_back(&c);
    }

    if (to_run.empty()) {
        printf("[engram-simt] No matching cases.\n");
        return 1;
    }

    printf("[engram-simt] Running %zu case(s)...\n", to_run.size());
    int passed = 0, failed = 0;
    for (const auto *c : to_run) {
        printf("\n=== %s ===\n", c->name.c_str());
        bool ok = c->run();
        if (ok) {
            printf("[PASS] %s\n", c->name.c_str());
            ++passed;
        } else {
            printf("[FAIL] %s\n", c->name.c_str());
            ++failed;
        }
    }
    printf("\n[engram-simt] Results: %d passed, %d failed, %zu total\n", passed, failed, to_run.size());
    return failed > 0 ? 1 : 0;
}
