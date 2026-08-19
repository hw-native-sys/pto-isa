/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Host driver for the GridPipe reduce <-> unicast CHANNEL RELAY smoke kernel.
//
// Three cells on one device, backed by per-cell GM windows + a fake
// CommDeviceContext (the same mock as the other GridPipe demos).  FOUR LAUNCHES
// over the SAME windows.  The first three run one stage each -- reduce, unicast,
// reduce -- which is what a caller that never marks a last round needs, since its
// collective's channel tenancy then ends at the next InitGridPipeFromWindow.  The
// FOURTH runs all three stages inside ONE launch, with `isLastRound` on the final
// round of each reduce; that is the same handover with the caller, rather than the
// launch boundary, ending the tenancy.
//
// The windows are zeroed ONCE, before the first launch: the pipe record surviving
// between launches is exactly what carries the channel's tenant kind and counters
// from one phase to the next.
//
//   contribution(stage s, round r, cell c) = (c + 1) + 100*s + r
//   unicast tile t from cell c             = (c + 1) + t
//
//   expected[0/2/4/6] = sum over r, c of contribution(s, r, c)  (reduce stages 0..3)
//   expected[1/5]     = sum over t < TILES of (1 + t)           (the drained flow)
//   expected[3/7]     = sum over t < SIDE_TILES of (2 + t)      (the side flow)
//
// Verified in-process; no data files.

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "runtime/rt.h"
#ifdef RT_STREAM_PRIORITY_DEFAULT
#undef RT_STREAM_PRIORITY_DEFAULT
#endif

#ifdef AICORE
#undef AICORE
#endif
#define AICORE

#ifndef __gm__
#define __gm__
#endif

// Skip SdmaWorkspaceManager pull-in from common.hpp (needs CCE attrs on host).
#define PTO_COMM_ST_SKIP_SDMA_WORKSPACE_MANAGER
#include "common.hpp"

#include "relay_smoke_config.hpp"
#include "relay_smoke_launch.hpp"

static bool ParseDeviceIdValue(const char* value, int& deviceId)
{
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    deviceId = static_cast<int>(parsed);
    return true;
}

static int GetDeviceId(int argc, char** argv)
{
    int deviceId = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device-id") == 0 || std::strcmp(argv[i], "-d") == 0) {
            if (i + 1 >= argc || !ParseDeviceIdValue(argv[i + 1], deviceId)) {
                std::cerr << "[ERROR] invalid --device-id value" << std::endl;
                std::exit(1);
            }
            return deviceId;
        }
        constexpr const char* kPrefix = "--device-id=";
        constexpr size_t kPrefixLen = 12;
        if (std::strncmp(argv[i], kPrefix, kPrefixLen) == 0) {
            if (!ParseDeviceIdValue(argv[i] + kPrefixLen, deviceId)) {
                std::cerr << "[ERROR] invalid --device-id value" << std::endl;
                std::exit(1);
            }
            return deviceId;
        }
    }
    if (const char* env = std::getenv("ASCEND_DEVICE_ID")) {
        (void)ParseDeviceIdValue(env, deviceId);
    }
    return deviceId;
}

static bool InitAcl(int deviceId)
{
    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] aclInit failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }
    aRet = aclrtSetDevice(deviceId);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSetDevice(" << deviceId << ") failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }
    return true;
}

struct Resources {
    aclrtStream stream = nullptr;
    void* windows_dev = nullptr;
    void* in_dev = nullptr;
    void* out_dev = nullptr;
    void* hccl_ctx_dev = nullptr;
    uint64_t ffts = 0;
    uint32_t fftsLen = 0;

    size_t cells = static_cast<size_t>(RELAY_CELLS);
    size_t windowsBytes = 0;
};

// The contribution of cell `c` in round `r` of reduce STAGE `s` (0..3: phase 0,
// phase 2, and the two inside the single-launch phase).  Kept in one place so the
// fill and the golden cannot drift.
static float ReduceContribution(int s, int r, int c) { return static_cast<float>(c + 1 + s * RELAY_STAGE_STAMP + r); }

static bool BuildFakeHcclCtx(Resources& r)
{
    CommDeviceContext hostCtx{};
    hostCtx.rankId = 0;
    hostCtx.rankNum = static_cast<uint32_t>(r.cells);
    hostCtx.winSize = static_cast<uint64_t>(RELAY_WINDOW_BYTES);
    uint64_t base = reinterpret_cast<uint64_t>(r.windows_dev);
    for (size_t i = 0; i < r.cells && i < HCCL_MAX_RANK_NUM; ++i) {
        hostCtx.windowsIn[i] = base + i * static_cast<size_t>(RELAY_WINDOW_BYTES);
        hostCtx.windowsOut[i] = hostCtx.windowsIn[i];
    }
    if (aclrtMalloc(&r.hccl_ctx_dev, sizeof(CommDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc(hccl_ctx) failed" << std::endl;
        return false;
    }
    if (aclrtMemcpy(
            r.hccl_ctx_dev, sizeof(CommDeviceContext), &hostCtx, sizeof(CommDeviceContext),
            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy(hccl_ctx) failed" << std::endl;
        return false;
    }
    return true;
}

static bool Allocate(Resources& r)
{
    if (aclrtCreateStream(&r.stream) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtCreateStream failed" << std::endl;
        return false;
    }
    r.windowsBytes = r.cells * static_cast<size_t>(RELAY_WINDOW_BYTES);

    aclrtMalloc(&r.windows_dev, r.windowsBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.in_dev, static_cast<size_t>(RELAY_IN_BYTES), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.out_dev, static_cast<size_t>(RELAY_OUT_BYTES), ACL_MEM_MALLOC_HUGE_FIRST);
    if (!r.windows_dev || !r.in_dev || !r.out_dev) {
        std::cerr << "[ERROR] aclrtMalloc failed" << std::endl;
        return false;
    }
    // Zeroed ONCE.  Every launch after this adopts the record the previous one left.
    aclrtMemset(r.windows_dev, r.windowsBytes, 0, r.windowsBytes);
    aclrtMemset(r.out_dev, static_cast<size_t>(RELAY_OUT_BYTES), 0, static_cast<size_t>(RELAY_OUT_BYTES));

    // Zero everywhere, so the contribution ARENA (which the kernel fills round by
    // round) starts empty: a fold that ran ahead of a member's store reads zeros.
    std::vector<float> hostIn(static_cast<size_t>(RELAY_IN_TILES) * static_cast<size_t>(RELAY_TILE_ELEMS), 0.0f);
    for (int s = 0; s < RELAY_REDUCE_STAGES; ++s) {
        for (int round = 0; round < RELAY_ROUNDS; ++round) {
            for (int c = 0; c < RELAY_CELLS; ++c) {
                const size_t tile = static_cast<size_t>((s * RELAY_ROUNDS + round) * RELAY_CELLS + c);
                float* dst = hostIn.data() + tile * static_cast<size_t>(RELAY_TILE_ELEMS);
                const float v = ReduceContribution(s, round, c);
                for (int e = 0; e < RELAY_TILE_ELEMS; ++e) {
                    dst[e] = v;
                }
            }
        }
    }
    for (int c = 0; c < RELAY_CELLS; ++c) {
        float* dst =
            hostIn.data() + static_cast<size_t>(RELAY_UNICAST_IN_BASE + c) * static_cast<size_t>(RELAY_TILE_ELEMS);
        for (int e = 0; e < RELAY_TILE_ELEMS; ++e) {
            dst[e] = static_cast<float>(c + 1);
        }
    }
    if (aclrtMemcpy(
            r.in_dev, static_cast<size_t>(RELAY_IN_BYTES), hostIn.data(), static_cast<size_t>(RELAY_IN_BYTES),
            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy(in_dev) failed" << std::endl;
        return false;
    }
    if (!BuildFakeHcclCtx(r)) {
        return false;
    }
    rtGetC2cCtrlAddr(&r.ffts, &r.fftsLen);
    if (r.ffts == 0) {
        std::cerr << "[ERROR] rtGetC2cCtrlAddr returned null FFTS address" << std::endl;
        return false;
    }
    return true;
}

static bool Verify(Resources& r)
{
    std::vector<float> outHost(static_cast<size_t>(RELAY_OUT_TILES) * static_cast<size_t>(RELAY_TILE_ELEMS), 0.0f);
    if (aclrtMemcpy(
            outHost.data(), static_cast<size_t>(RELAY_OUT_BYTES), r.out_dev, static_cast<size_t>(RELAY_OUT_BYTES),
            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        std::cerr << "[ERROR] out D2H memcpy failed" << std::endl;
        return false;
    }

    double expected[RELAY_OUT_TILES] = {0.0};
    static const int kReduceOut[RELAY_REDUCE_STAGES] = {
        RELAY_OUT_REDUCE0, RELAY_OUT_REDUCE1, RELAY_OUT_REDUCE2, RELAY_OUT_REDUCE3};
    for (int s = 0; s < RELAY_REDUCE_STAGES; ++s) {
        for (int round = 0; round < RELAY_ROUNDS; ++round) {
            for (int c = 0; c < RELAY_CELLS; ++c) {
                expected[kReduceOut[s]] += static_cast<double>(ReduceContribution(s, round, c));
            }
        }
    }
    for (int t = 0; t < RELAY_TILES; ++t) {
        expected[RELAY_OUT_UNICAST0] += static_cast<double>(RELAY_CELL_PROD + 1 + t);
        expected[RELAY_OUT_UNICAST1] += static_cast<double>(RELAY_CELL_PROD + 1 + t);
    }
    for (int t = 0; t < RELAY_SIDE_TILES; ++t) {
        expected[RELAY_OUT_SIDE0] += static_cast<double>(RELAY_CELL_MID + 1 + t);
        expected[RELAY_OUT_SIDE1] += static_cast<double>(RELAY_CELL_MID + 1 + t);
    }

    static const char* kPhaseName[RELAY_OUT_TILES] = {
        "launch 0   reduce   (fresh channel)",
        "launch 1   unicast  (reduce -> unicast: the ZERO rule)",
        "launch 2   reduce   (unicast -> reduce: DRAIN + baseline&round)",
        "launch 1   side flow (cell 1 -> cell 0, leaves a DIFFERENT credit leftover)",
        "in-launch  reduce A (isLastRound releases the channel mid-launch)",
        "in-launch  unicast  (takes it over with NO launch boundary: the ZERO rule)",
        "in-launch  reduce C (takes it back mid-launch: DRAIN + baseline&round)",
        "in-launch  side flow"};

    double maxDiff = 0.0;
    bool ok = true;
    for (int p = 0; p < RELAY_OUT_TILES; ++p) {
        if ((p == RELAY_OUT_SIDE0 || p == RELAY_OUT_SIDE1) && RELAY_SIDE_TILES == 0) {
            continue;
        }
        const float* tile = outHost.data() + static_cast<size_t>(p) * static_cast<size_t>(RELAY_TILE_ELEMS);
        double worst = 0.0;
        for (int e = 0; e < RELAY_TILE_ELEMS; ++e) {
            const double d = std::abs(static_cast<double>(tile[e]) - expected[p]);
            if (d > worst) {
                worst = d;
            }
        }
        if (worst > maxDiff) {
            maxDiff = worst;
        }
        std::cout << "[INFO] " << kPhaseName[p] << ": expected=" << expected[p] << " got=" << tile[0]
                  << " max diff=" << worst << std::endl;
        if (worst != 0.0) {
            ok = false;
        }
    }
    std::cout << "[INFO] channel relay smoke: rounds/reduce=" << RELAY_ROUNDS << " unicast tiles=" << RELAY_TILES
              << " side tiles=" << RELAY_SIDE_TILES << " slots=" << RELAY_SLOT_COUNT << " shared-pool channels=1"
              << " max diff=" << maxDiff << std::endl;
    if (!ok) {
        std::cerr << "[ERROR] mismatch (max diff " << maxDiff << ")" << std::endl;
    }
    return ok;
}

static bool CheckFaults(Resources& r, int phase)
{
    constexpr size_t kFlagWords = static_cast<size_t>(RELAY_GRID_FLAGS_BYTES) / sizeof(uint32_t);
    constexpr size_t kScbLineWords = 64 / sizeof(uint32_t); // grid_mock::kScbLineStrideU32
    constexpr size_t kFaultWordInLine = 10;                 // grid_mock::kFaultFlagWordOffset
    constexpr size_t kScbLines = kFlagWords / kScbLineWords;
    std::vector<uint32_t> flags(r.cells * kFlagWords, 0);
    for (size_t cell = 0; cell < r.cells; ++cell) {
        auto* src = reinterpret_cast<uint8_t*>(r.windows_dev) + cell * RELAY_WINDOW_BYTES;
        auto* dst = flags.data() + cell * kFlagWords;
        if (aclrtMemcpy(
                dst, kFlagWords * sizeof(uint32_t), src, kFlagWords * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST) !=
            ACL_SUCCESS) {
            std::cerr << "[ERROR] flag D2H memcpy failed for cell " << cell << std::endl;
            return false;
        }
    }
    bool ok = true;
    for (size_t cell = 0; cell < r.cells; ++cell) {
        const uint32_t* cf = flags.data() + cell * kFlagWords;
        for (size_t line = 0; line < kScbLines; ++line) {
            const size_t i = line * kScbLineWords + kFaultWordInLine;
            if (cf[i] >= 0x100U) {
                std::cerr << "[ERROR] GridPipe fault phase=" << phase << " cell=" << cell << " flagWord=" << i
                          << " code=0x" << std::hex << cf[i] << std::dec << std::endl;
                ok = false;
            }
        }
    }
    return ok;
}

static void Cleanup(Resources& r)
{
    if (r.hccl_ctx_dev) {
        aclrtFree(r.hccl_ctx_dev);
    }
    if (r.windows_dev) {
        aclrtFree(r.windows_dev);
    }
    if (r.in_dev) {
        aclrtFree(r.in_dev);
    }
    if (r.out_dev) {
        aclrtFree(r.out_dev);
    }
    if (r.stream) {
        aclrtDestroyStream(r.stream);
    }
}

static bool Run()
{
    Resources r;
    if (!Allocate(r)) {
        Cleanup(r);
        return false;
    }

    static const char* kPhaseLabel[RELAY_LAUNCHES] = {
        "reduce            ", "unicast           ", "reduce            ", "reduce+unicast+reduce, ONE launch"};
    for (int phase = 0; phase < RELAY_LAUNCHES; ++phase) {
        auto t0 = std::chrono::high_resolution_clock::now();
        launchRelaySmokeKernel(
            reinterpret_cast<uint8_t*>(r.ffts), reinterpret_cast<uint8_t*>(r.windows_dev),
            reinterpret_cast<uint8_t*>(r.in_dev), reinterpret_cast<uint8_t*>(r.out_dev),
            reinterpret_cast<uint8_t*>(r.hccl_ctx_dev), phase, r.stream);
        aclError syncRet = aclrtSynchronizeStream(r.stream);
        auto t1 = std::chrono::high_resolution_clock::now();
        const bool faultsOk = CheckFaults(r, phase);
        std::cout << "[INFO] phase " << phase << " (" << kPhaseLabel[phase] << ") "
                  << std::chrono::duration<double, std::micro>(t1 - t0).count()
                  << " us (rc=" << static_cast<int>(syncRet) << " faults_ok=" << (faultsOk ? 1 : 0) << ")" << std::endl;
        if (syncRet != ACL_SUCCESS || !faultsOk) {
            Cleanup(r);
            return false;
        }
    }

    const bool ok = Verify(r);
    Cleanup(r);
    return ok;
}

int main(int argc, char** argv)
{
    int deviceId = GetDeviceId(argc, argv);
    std::cout << "[INFO] using device " << deviceId << std::endl;
    if (!InitAcl(deviceId)) {
        return 1;
    }

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  GridPipe reduce <-> unicast CHANNEL RELAY smoke test" << std::endl;
    std::cout << "  one shared-pool channel carries: reduce -> unicast -> reduce" << std::endl;
    std::cout << "  cells 0,1,2 fan in to cell 2; cell 0 -> cell 2 x" << RELAY_TILES << " in between" << std::endl;
    std::cout << "  run twice: a stage per launch, then all three INSIDE one launch" << std::endl;
    std::cout << "================================================================" << std::endl;

    bool ok = Run();
    std::cout << (ok ? "[SUCCESS] GridPipe reduce/unicast channel relay smoke PASS." :
                       "[FAILED] GridPipe reduce/unicast channel relay smoke FAILED.")
              << std::endl;
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ok ? 0 : 1;
}
