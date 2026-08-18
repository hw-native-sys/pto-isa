/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Host driver for the GridPipe unicast HANDOVER smoke kernel.
//
// Three cells on one device, backed by per-cell GM windows + a fake
// CommDeviceContext (the same mock as the other GridPipe demos).  Cell c's input
// tile is stamped with (c + 1), and producer c pushes tiles stamped + 0, + 1, ...
// so the consumer's accumulator is a SUM that does not depend on which producer's
// tile arrived when -- only on all of them arriving exactly once.
//
//   expected(cell 1) = sum over p in {A, B}, t in [0, TILES) of (p + 1 + t)
//   expected(cell 0) = expected(cell 2) = 0        (producers store nothing)
//
// Verified in-process; no data files.

#include <chrono>
#include <climits>
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

#include "unicast_smoke_config.hpp"
#include "unicast_smoke_launch.hpp"

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

    size_t cells = static_cast<size_t>(UNICAST_CELLS);
    size_t windowsBytes = 0;
    size_t bufBytes = 0;
};

static bool BuildFakeHcclCtx(Resources& r)
{
    CommDeviceContext hostCtx{};
    hostCtx.rankId = 0;
    hostCtx.rankNum = static_cast<uint32_t>(r.cells);
    hostCtx.winSize = static_cast<uint64_t>(UNICAST_WINDOW_BYTES);
    uint64_t base = reinterpret_cast<uint64_t>(r.windows_dev);
    for (size_t i = 0; i < r.cells && i < HCCL_MAX_RANK_NUM; ++i) {
        hostCtx.windowsIn[i] = base + i * static_cast<size_t>(UNICAST_WINDOW_BYTES);
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
    r.windowsBytes = r.cells * static_cast<size_t>(UNICAST_WINDOW_BYTES);
    r.bufBytes = r.cells * static_cast<size_t>(UNICAST_TILE_BYTES);

    aclrtMalloc(&r.windows_dev, r.windowsBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.in_dev, r.bufBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.out_dev, r.bufBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!r.windows_dev || !r.in_dev || !r.out_dev) {
        std::cerr << "[ERROR] aclrtMalloc failed" << std::endl;
        return false;
    }
    aclrtMemset(r.windows_dev, r.windowsBytes, 0, r.windowsBytes);
    aclrtMemset(r.out_dev, r.bufBytes, 0, r.bufBytes);

    std::vector<float> hostIn(r.cells * static_cast<size_t>(UNICAST_TILE_ELEMS));
    for (size_t cell = 0; cell < r.cells; ++cell) {
        float stamp = static_cast<float>(cell + 1);
        float* dst = hostIn.data() + cell * static_cast<size_t>(UNICAST_TILE_ELEMS);
        for (int e = 0; e < UNICAST_TILE_ELEMS; ++e) {
            dst[e] = stamp;
        }
    }
    if (aclrtMemcpy(r.in_dev, r.bufBytes, hostIn.data(), r.bufBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
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
    std::vector<float> outHost(r.cells * static_cast<size_t>(UNICAST_TILE_ELEMS), 0.0f);
    if (aclrtMemcpy(outHost.data(), r.bufBytes, r.out_dev, r.bufBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        std::cerr << "[ERROR] out D2H memcpy failed" << std::endl;
        return false;
    }

    // The consumer holds every tile both producers pushed; the producers store
    // nothing.  A sum, so it is independent of the arrival order -- what it pins
    // down is that each of the 2*TILES tiles arrived exactly once, which is what a
    // handover that dropped or duplicated the retiring producer's tail would break.
    double expectedC = 0.0;
    const int producers[2] = {UNICAST_CELL_A, UNICAST_CELL_B};
    for (int p : producers) {
        for (int t = 0; t < UNICAST_TILES; ++t) {
            expectedC += static_cast<double>(p + 1 + t);
        }
    }

    double maxDiff = 0.0;
    size_t firstBadCell = SIZE_MAX;
    for (size_t cell = 0; cell < r.cells; ++cell) {
        const float expected = (cell == static_cast<size_t>(UNICAST_CELL_C)) ? static_cast<float>(expectedC) : 0.0f;
        const float* tile = outHost.data() + cell * static_cast<size_t>(UNICAST_TILE_ELEMS);
        for (int e = 0; e < UNICAST_TILE_ELEMS; ++e) {
            double d = std::abs(static_cast<double>(tile[e]) - static_cast<double>(expected));
            if (d > maxDiff) {
                maxDiff = d;
                if (d > 0.0 && firstBadCell == SIZE_MAX) {
                    firstBadCell = cell;
                }
            }
        }
        std::cout << "[INFO] cell " << cell << " ("
                  << (cell == static_cast<size_t>(UNICAST_CELL_A) ? "producer A" :
                      cell == static_cast<size_t>(UNICAST_CELL_C) ? "consumer  " :
                                                                    "producer B")
                  << ") expected=" << expected << " got=" << tile[0] << std::endl;
    }

    std::cout << "[INFO] unicast handover smoke: tiles/turn=" << UNICAST_TILES << " slots=" << UNICAST_SLOT_COUNT
              << (UNICAST_SLOT_COUNT < 2 * UNICAST_TILES ? " (ring wraps: the new owner waits on the relayed credit)" :
                                                           " (turns land in disjoint slots)")
              << " max diff=" << maxDiff << std::endl;
    if (maxDiff != 0.0) {
        std::cerr << "[ERROR] mismatch (max diff " << maxDiff << ", first bad cell " << firstBadCell << ")"
                  << std::endl;
        return false;
    }
    return true;
}

static bool CheckFaults(Resources& r)
{
    constexpr size_t kFlagWords = static_cast<size_t>(UNICAST_GRID_FLAGS_BYTES) / sizeof(uint32_t);
    constexpr size_t kScbLineWords = 64 / sizeof(uint32_t); // grid_mock::kScbLineStrideU32
    constexpr size_t kFaultWordInLine = 10;                 // grid_mock::kFaultFlagWordOffset
    constexpr size_t kScbLines = kFlagWords / kScbLineWords;
    std::vector<uint32_t> flags(r.cells * kFlagWords, 0);
    for (size_t cell = 0; cell < r.cells; ++cell) {
        auto* src = reinterpret_cast<uint8_t*>(r.windows_dev) + cell * UNICAST_WINDOW_BYTES;
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
                std::cerr << "[ERROR] GridPipe fault cell=" << cell << " flagWord=" << i << " code=0x" << std::hex
                          << cf[i] << std::dec << std::endl;
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

    auto t0 = std::chrono::high_resolution_clock::now();
    launchUnicastSmokeKernel(
        reinterpret_cast<uint8_t*>(r.ffts), reinterpret_cast<uint8_t*>(r.windows_dev),
        reinterpret_cast<uint8_t*>(r.in_dev), reinterpret_cast<uint8_t*>(r.out_dev),
        reinterpret_cast<uint8_t*>(r.hccl_ctx_dev), r.stream);
    aclError syncRet = aclrtSynchronizeStream(r.stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    const bool faultsOk = CheckFaults(r);
    std::cout << "[INFO] launch+sync " << std::chrono::duration<double, std::micro>(t1 - t0).count()
              << " us (rc=" << static_cast<int>(syncRet) << " faults_ok=" << (faultsOk ? 1 : 0) << ")" << std::endl;
    if (syncRet != ACL_SUCCESS || !faultsOk) {
        Cleanup(r);
        return false;
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
    std::cout << "  GridPipe unicast HANDOVER smoke test" << std::endl;
    std::cout << "  cell0 (A) -> cell1 (C) x" << UNICAST_TILES << " + CLOSE, baton -> cell2 (B),"
              << " B -> C x" << UNICAST_TILES << " on the SAME channel" << std::endl;
    std::cout << "  the channel changes owner while A's tiles are still undrained" << std::endl;
    std::cout << "================================================================" << std::endl;

    bool ok = Run();
    std::cout << (ok ? "[SUCCESS] GridPipe unicast handover smoke PASS." :
                       "[FAILED] GridPipe unicast handover smoke FAILED.")
              << std::endl;
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ok ? 0 : 1;
}
