/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the software repository for the full text of the License.
*/

// Pure 1D N-cut 32-cell WSE FFN emulation host driver -- TPUSH AllGather variant.
//
// Verifies the GridPipe TPUSH/TPOP unicast primitives on a real distributed-FFN
// AllGather: identical compute and topology to the TBROADCAST AllGather variant,
// but the two gather phases are nearest-neighbor TPUSH/TPOP relays (a fan-in-1 DAG)
// instead of the MPSC TBROADCAST collective.  The output is compared bit-exact
// against the same SwiGLU golden.
//
// Emulates a real 4x8 = 32-cell wafer topology on 24 physical AICores by running
// the FFN as 4 phase-dispatched launches with wave-by-communication-phase +
// aclrtSynchronizeStream barriers.  Phase B (row relay) waves by whole rows; Phase C
// (col relay) waves by whole cols, so every active relay group is wholly present in
// its wave (no spin-wait deadlock).  Shapes are the real DeepSeek-v4 Pro FFN
// (M=8, H=7168, I=3072).

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
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

#include "common.hpp"

#ifdef DT_UNDEFINED
#define DT_UNDEFINED_SAVED DT_UNDEFINED
#undef DT_UNDEFINED
#endif
#include "test_common.h"
#ifdef DT_UNDEFINED_SAVED
#define DT_UNDEFINED DT_UNDEFINED_SAVED
#undef DT_UNDEFINED_SAVED
#endif

#include "ffn_config.hpp"
#include "kernel_launch.hpp"
#include "batcher.hpp"

struct DeviceResources {
    aclrtStream stream = nullptr;
    // Two unicast GridPipe arenas (P1 row-relay hidden shard, P2 col-relay row
    // block), each with its own CommDeviceContext resolving cross-cell addresses.
    void *p1_windows_dev = nullptr;
    void *p2_windows_dev = nullptr;
    void *p1_hccl_ctx_dev = nullptr;
    void *p2_hccl_ctx_dev = nullptr;
    size_t p1WindowsBytes = 0;
    size_t p2WindowsBytes = 0;
    // Per-cell GM intermediates that persist across the 4 launches.
    void *hidden_shard_dev = nullptr;  // [cells, T, I_shard]   half (Phase A -> B)
    void *row_block_dev = nullptr;     // [cells, T, row_block] half (Phase B -> C)
    void *hidden_full_dev = nullptr;   // [cells, T, I]         half (Phase C -> D)
    void *gate_partial_dev = nullptr;  // [cells, T, I_shard]   fp32 (cube->vec C2V, Phase A)
    void *up_partial_dev = nullptr;

    uint64_t ffts = 0;
    uint32_t fftsLen = 0;

    int rows = FFN_NCUT_ROWS;
    int cols = FFN_NCUT_COLS;
    int cells = FFN_NCUT_ROWS * FFN_NCUT_COLS;
    int physCores = 24; // physical cube AICores (910B1 default; runtime-queried)

    std::string dataDir = "./out";

    FfnBatcher batcher;
};

// ---- arg/env device-id parsing (shared shape with the other variants) ----

static bool ParseIntValue(const char *value, int &out)
{
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

static bool ParseIntEnv(const char *name, int &out)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    if (!ParseIntValue(value, out)) {
        std::cerr << "[WARN] ignoring invalid " << name << "=" << value << std::endl;
        return false;
    }
    return true;
}

static int GetDeviceId(int argc, char **argv)
{
    int deviceId = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device-id") == 0 || std::strcmp(argv[i], "-d") == 0) {
            if (i + 1 >= argc || !ParseIntValue(argv[i + 1], deviceId)) {
                std::cerr << "[ERROR] invalid --device-id value" << std::endl;
                std::exit(1);
            }
            return deviceId;
        }
        constexpr const char *kPrefix = "--device-id=";
        if (std::strncmp(argv[i], kPrefix, 12) == 0) {
            if (!ParseIntValue(argv[i] + 12, deviceId)) {
                std::cerr << "[ERROR] invalid --device-id value" << std::endl;
                std::exit(1);
            }
            return deviceId;
        }
    }
    if (ParseIntEnv("FFN_GRID_DEVICE_ID", deviceId) || ParseIntEnv("ASCEND_DEVICE_ID", deviceId) ||
        ParseIntEnv("DEVICE_ID", deviceId)) {
        return deviceId;
    }
    return 0;
}

// Physical cube AICore count: --phys-cores arg > FFN_GRID_PHYS_CORES env >
// rtGetAiCoreCount() > 910B1 default 24 (mirrors syncall_cfg::GetCoreConfig).
static int GetPhysCores(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "--phys-cores") == 0) && i + 1 < argc) {
            int v = 0;
            if (ParseIntValue(argv[i + 1], v) && v > 0) {
                return v;
            }
        }
    }
    if (const char *env = std::getenv("FFN_GRID_PHYS_CORES")) {
        int v = 0;
        if (ParseIntValue(env, v) && v > 0) {
            return v;
        }
    }
    uint32_t aiCoreCnt = 0;
    if (rtGetAiCoreCount(&aiCoreCnt) == RT_ERROR_NONE && aiCoreCnt > 0) {
        return static_cast<int>(aiCoreCnt);
    }
    return 24;
}

static bool ShouldUseRtSetDevice()
{
    const char *value = std::getenv("FFN_GRID_USE_RT_SET_DEVICE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool InitAcl(int device_id)
{
    constexpr int kAclRepeatInit = 100002;
    std::cout << "[INFO] aclInit begin" << std::endl;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] aclInit failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }
    std::cout << "[INFO] aclInit done rc=" << static_cast<int>(aRet) << std::endl;

    if (ShouldUseRtSetDevice()) {
        rtError_t rtRet = rtSetDevice(device_id);
        if (rtRet != RT_ERROR_NONE) {
            std::cerr << "[ERROR] rtSetDevice(" << device_id << ") failed: " << static_cast<int>(rtRet) << std::endl;
            return false;
        }
    }
    std::cout << "[INFO] aclrtSetDevice(" << device_id << ") begin" << std::endl;
    aRet = aclrtSetDevice(device_id);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSetDevice(" << device_id << ") failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }
    std::cout << "[INFO] aclrtSetDevice(" << device_id << ") done" << std::endl;
    return true;
}

// Build one CommDeviceContext whose windowsIn[i] = arena + i*winBytes (one window
// per cell, contiguous).
static bool InitGridPipeContext(void *arenaDev, int winBytes, void *&ctxDev)
{
    CommDeviceContext hostCtx{};
    hostCtx.rankId = 0;
    hostCtx.rankNum = static_cast<uint32_t>(FFN_NCUT_CELLS);
    hostCtx.winSize = static_cast<uint64_t>(winBytes);
    uint64_t base = reinterpret_cast<uint64_t>(arenaDev);
    for (int i = 0; i < FFN_NCUT_CELLS && i < HCCL_MAX_RANK_NUM; ++i) {
        hostCtx.windowsIn[i] = base + static_cast<uint64_t>(i) * winBytes;
        hostCtx.windowsOut[i] = hostCtx.windowsIn[i];
    }
    if (aclrtMalloc(&ctxDev, sizeof(CommDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc(hccl_ctx) failed" << std::endl;
        return false;
    }
    if (aclrtMemcpy(ctxDev, sizeof(CommDeviceContext), &hostCtx, sizeof(CommDeviceContext),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy(hccl_ctx) failed" << std::endl;
        return false;
    }
    return true;
}

static void *MallocBufOrDie(size_t bytes, const char *what)
{
    void *p = nullptr;
    if (aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || p == nullptr) {
        std::cerr << "[ERROR] aclrtMalloc(" << what << ", " << bytes << ") failed" << std::endl;
        return nullptr;
    }
    aclrtMemset(p, bytes, 0, bytes);
    return p;
}

static bool AllocateResources(DeviceResources &r, int physCores)
{
    r.physCores = physCores;
    if (const char *env = std::getenv("FFN_GRID_DATA_DIR")) {
        r.dataDir = env;
    }
    if (aclrtCreateStream(&r.stream) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtCreateStream failed" << std::endl;
        return false;
    }

    // Two unicast GridPipe arenas (one window per cell, memset so scoreboards start at 0).
    r.p1WindowsBytes = static_cast<size_t>(FFN_NCUT_CELLS) * FFN_NCUT_TPUSH_WIN_P1;
    r.p2WindowsBytes = static_cast<size_t>(FFN_NCUT_CELLS) * FFN_NCUT_TPUSH_WIN_P2;
    r.p1_windows_dev = MallocBufOrDie(r.p1WindowsBytes, "p1_windows");
    r.p2_windows_dev = MallocBufOrDie(r.p2WindowsBytes, "p2_windows");
    if (!r.p1_windows_dev || !r.p2_windows_dev) {
        return false;
    }

    // Per-cell GM intermediates + cube/vec C2V working buffers.
    r.hidden_shard_dev = MallocBufOrDie(static_cast<size_t>(FFN_NCUT_CELLS) * FFN_NCUT_HIDDEN_SHARD_BYTES, "hidden_shard");
    r.row_block_dev = MallocBufOrDie(static_cast<size_t>(FFN_NCUT_CELLS) * FFN_NCUT_ROW_BLOCK_BYTES, "row_block");
    r.hidden_full_dev = MallocBufOrDie(static_cast<size_t>(FFN_NCUT_CELLS) * FFN_NCUT_HIDDEN_FULL_BYTES, "hidden_full");
    r.gate_partial_dev = MallocBufOrDie(static_cast<size_t>(FFN_NCUT_CELLS) * FFN_NCUT_GATE_PARTIAL_BYTES, "gate_partial");
    r.up_partial_dev = MallocBufOrDie(static_cast<size_t>(FFN_NCUT_CELLS) * FFN_NCUT_GATE_PARTIAL_BYTES, "up_partial");
    if (!r.hidden_shard_dev || !r.row_block_dev || !r.hidden_full_dev || !r.gate_partial_dev || !r.up_partial_dev) {
        return false;
    }

    if (!r.batcher.AllocateNcut(r.rows, r.cols)) {
        return false;
    }
    if (!InitGridPipeContext(r.p1_windows_dev, FFN_NCUT_TPUSH_WIN_P1, r.p1_hccl_ctx_dev) ||
        !InitGridPipeContext(r.p2_windows_dev, FFN_NCUT_TPUSH_WIN_P2, r.p2_hccl_ctx_dev)) {
        return false;
    }

    rtGetC2cCtrlAddr(&r.ffts, &r.fftsLen);
    if (r.ffts == 0) {
        std::cerr << "[ERROR] rtGetC2cCtrlAddr returned null FFTS address" << std::endl;
        return false;
    }

    std::cout << "[INFO] grid=" << r.rows << "x" << r.cols << " cells=" << r.cells << " physCores=" << r.physCores
              << " dataDir=" << r.dataDir << " p1Win=" << r.p1WindowsBytes << "B p2Win=" << r.p2WindowsBytes << "B"
              << std::endl;
    return true;
}

static bool LoadBatcherData(DeviceResources &r)
{
    return r.batcher.LoadAndDistributeNcut(r.dataDir);
}

static const char *GridPipeFaultName(uint32_t code)
{
    switch (code) {
        case 0x101:
            return "push north boundary";
        case 0x102:
            return "push east boundary";
        case 0x103:
            return "push west boundary";
        case 0x104:
            return "push south boundary";
        case 0x105:
            return "push source boundary";
        case 0x201:
            return "pop north boundary";
        case 0x202:
            return "pop east boundary";
        case 0x203:
            return "pop west boundary";
        case 0x204:
            return "pop south boundary";
        case 0x205:
            return "pop non-local segment";
        case 0x301:
            return "wait ready timeout";
        case 0x302:
            return "wait free timeout";
        default:
            return "unknown";
    }
}

// Scan one arena's per-cell flag header (first 128 B = unicast ready/free scbs +
// reserved) for fault sentinels (any word >= 0x100).
static bool CheckArenaFaults(void *arenaDev, int winBytes, int cells, const char *arenaName)
{
    constexpr size_t kFlagWords = static_cast<size_t>(FFN_NCUT_GRID_FLAGS_BYTES) / sizeof(uint32_t);
    std::vector<uint32_t> flags(static_cast<size_t>(cells) * kFlagWords, 0);
    for (int cell = 0; cell < cells; ++cell) {
        auto *src = reinterpret_cast<uint8_t *>(arenaDev) + static_cast<size_t>(cell) * winBytes;
        auto *dst = flags.data() + static_cast<size_t>(cell) * kFlagWords;
        if (aclrtMemcpy(dst, kFlagWords * sizeof(uint32_t), src, kFlagWords * sizeof(uint32_t),
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            std::cerr << "[ERROR] GridPipe flag D2H failed (" << arenaName << " cell " << cell << ")" << std::endl;
            return false;
        }
    }
    bool ok = true;
    for (int cell = 0; cell < cells; ++cell) {
        int row = cell / FFN_NCUT_COLS;
        int col = cell - row * FFN_NCUT_COLS;
        const uint32_t *cellFlags = flags.data() + static_cast<size_t>(cell) * kFlagWords;
        for (size_t i = 0; i < kFlagWords; ++i) {
            uint32_t value = cellFlags[i];
            if (value >= 0x100U) {
                std::cerr << "[ERROR] GridPipe fault " << arenaName << " cell=" << cell << " row=" << row << " col=" << col
                          << " word=" << i << " code=0x" << std::hex << value << std::dec << " ("
                          << GridPipeFaultName(value) << ")" << std::endl;
                ok = false;
            }
        }
    }
    return ok;
}

static bool CheckGridPipeFaults(DeviceResources &r)
{
    return CheckArenaFaults(r.p1_windows_dev, FFN_NCUT_TPUSH_WIN_P1, r.cells, "P1") &&
           CheckArenaFaults(r.p2_windows_dev, FFN_NCUT_TPUSH_WIN_P2, r.cells, "P2");
}

static bool VerifyOutput(DeviceResources &r)
{
    const size_t outputElems = static_cast<size_t>(FFN_NCUT_T) * FFN_NCUT_H; // [8,7168]
    const size_t outputBytes = outputElems * sizeof(float);
    std::vector<float> outHost(outputElems);
    if (aclrtMemcpy(outHost.data(), outputBytes, r.batcher.NcutYFull(), outputBytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        std::cerr << "[ERROR] y_output D2H memcpy failed" << std::endl;
        return false;
    }
    std::string goldenPath = r.dataDir + "/golden.bin";
    std::vector<float> golden(outputElems);
    size_t fileSize = 0;
    if (!PtoTestCommon::ReadFile(goldenPath, fileSize, golden.data(), outputBytes) || fileSize != outputBytes) {
        std::cerr << "[ERROR] golden.bin mismatch: " << goldenPath << " (got " << fileSize << " bytes, expected "
                  << outputBytes << ")" << std::endl;
        return false;
    }
    std::cout << "[INFO] ResultCmp 32-cell N-cut TPUSH AllGather output vs golden:" << std::endl;
    return PtoTestCommon::ResultCmp(golden, outHost.data(), 0.001f);
}

static void Cleanup(DeviceResources &r)
{
    auto freeBuf = [](void *&p) {
        if (p) {
            aclrtFree(p);
            p = nullptr;
        }
    };
    freeBuf(r.p1_hccl_ctx_dev);
    freeBuf(r.p2_hccl_ctx_dev);
    freeBuf(r.p1_windows_dev);
    freeBuf(r.p2_windows_dev);
    freeBuf(r.hidden_shard_dev);
    freeBuf(r.row_block_dev);
    freeBuf(r.hidden_full_dev);
    freeBuf(r.gate_partial_dev);
    freeBuf(r.up_partial_dev);
    if (r.stream) {
        aclrtDestroyStream(r.stream);
        r.stream = nullptr;
    }
    // r.batcher frees its own GM arena in its destructor.
}

// Launch one wave (one phase, one wave's block count) and synchronize.  Stream
// ordering serializes waves; the explicit aclrtSynchronizeStream is the host
// barrier, and lets us fault-check between waves.  The TPUSH relay gather is a
// fan-in-1 DAG, so the vec-only gather waves need no cube keep-alive (unlike the
// MPSC TBROADCAST variant) -- the idle cube half simply returns and the vec relay
// completes under stream ordering.
static bool LaunchWave(DeviceResources &r, int phase, int rowStart, int colStart, int waveCols, int blockCount,
                       const char *tag)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    launchDistributedFfnGridTpushAllGatherMixedKernel(
        reinterpret_cast<uint8_t *>(r.ffts), reinterpret_cast<uint8_t *>(r.p1_windows_dev),
        reinterpret_cast<uint8_t *>(r.p2_windows_dev), r.batcher.NcutXFull(), r.batcher.NcutWGateShards(),
        r.batcher.NcutWUpShards(), r.batcher.NcutWDownShards(), reinterpret_cast<uint8_t *>(r.hidden_shard_dev),
        reinterpret_cast<uint8_t *>(r.row_block_dev), reinterpret_cast<uint8_t *>(r.hidden_full_dev),
        reinterpret_cast<uint8_t *>(r.gate_partial_dev), reinterpret_cast<uint8_t *>(r.up_partial_dev),
        r.batcher.NcutYFull(), reinterpret_cast<uint8_t *>(r.p1_hccl_ctx_dev),
        reinterpret_cast<uint8_t *>(r.p2_hccl_ctx_dev), phase, rowStart,
        colStart, waveCols, FFN_NCUT_ROWS, FFN_NCUT_COLS, blockCount, r.stream);
    aclError rc = aclrtSynchronizeStream(r.stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    bool faultsOk = (rc == ACL_SUCCESS) && CheckGridPipeFaults(r);
    std::cout << "[INFO] phase " << tag << "  blocks=" << blockCount << "  " << us << " us  (rc="
              << static_cast<int>(rc) << " faults_ok=" << (faultsOk ? 1 : 0) << ")" << std::endl;
    return faultsOk;
}

static bool RunSingleDevice(int physCores)
{
    DeviceResources r;
    if (!AllocateResources(r, physCores) || !LoadBatcherData(r)) {
        Cleanup(r);
        return false;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // Phase A: gate + up + SwiGLU -> hidden_shard.  No comm: launch all 32 cells.
    if (!LaunchWave(r, /*phase=*/0, 0, 0, FFN_NCUT_COLS, FFN_NCUT_CELLS, "A gate+up+SwiGLU  (no comm)")) {
        Cleanup(r);
        return false;
    }

    // Phase B: row relay gather (EAST fwd + WEST bwd, 8-way).  Wave by WHOLE ROWS
    // so every row relay (8 cells) is wholly present in its wave.  The relay is a
    // fan-in-1 DAG (deadlock-free under any scheduling), so the only bound on wave
    // size is physCores (no oversubscription) -- not the old ~8-poller thrash cap
    // (kCommWaveCap, removed).
    const int rowsPerWave = std::max(1, r.physCores / FFN_NCUT_COLS);
    for (int rs = 0; rs < FFN_NCUT_ROWS; rs += rowsPerWave) {
        int waveRows = std::min(rowsPerWave, FFN_NCUT_ROWS - rs);
        if (!LaunchWave(r, /*phase=*/1, rs, 0, FFN_NCUT_COLS, waveRows * FFN_NCUT_COLS, "B row relay gather (TPUSH)")) {
            Cleanup(r);
            return false;
        }
    }

    // Phase C: col relay gather (SOUTH fwd + NORTH bwd, 4-way).  Wave by WHOLE COLS
    // so every col relay (4 cells) is wholly present in its wave.  Waves are still
    // required here (not one 32-block launch): a col relay spans all 4 rows, so a
    // single 32-block launch oversubscribes (32>24) and first-batch cells would spin
    // on second-batch (row-3) doorbells that never get a core -> deadlock.  Wave by
    // whole cols keeps each col relay within a <= physCores wave.
    const int colsPerWave = std::max(1, r.physCores / FFN_NCUT_ROWS);
    for (int cs = 0; cs < FFN_NCUT_COLS; cs += colsPerWave) {
        int waveCols = std::min(colsPerWave, FFN_NCUT_COLS - cs);
        if (!LaunchWave(r, /*phase=*/2, 0, cs, waveCols, FFN_NCUT_ROWS * waveCols, "C col relay gather (TPUSH)")) {
            Cleanup(r);
            return false;
        }
    }

    // Phase D: down GEMM -> y_shard.  No comm: launch all 32 cells.
    if (!LaunchWave(r, /*phase=*/3, 0, 0, FFN_NCUT_COLS, FFN_NCUT_CELLS, "D down GEMM      (no comm)")) {
        Cleanup(r);
        return false;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    int bWaves = (FFN_NCUT_ROWS + rowsPerWave - 1) / rowsPerWave;
    int cWaves = (FFN_NCUT_COLS + colsPerWave - 1) / colsPerWave;
    const int totalLaunches = 1 /*A*/ + bWaves /*B*/ + cWaves /*C*/ + 1 /*D*/;
    std::cout << "[INFO] emulated 32-cell N-cut TPUSH AllGather on " << r.physCores << " phys AICores, " << totalLaunches
              << " launches (B=" << bWaves << " row-waves, C=" << cWaves << " col-waves), total " << us << " us"
              << std::endl;

    bool verifyOk = VerifyOutput(r);
    Cleanup(r);
    return verifyOk;
}

int main(int argc, char **argv)
{
    int deviceId = GetDeviceId(argc, argv);
    int physCores = GetPhysCores(argc, argv);
    std::cout << "[INFO] using device " << deviceId << "  physCores=" << physCores << std::endl;
    if (!InitAcl(deviceId)) {
        return 1;
    }

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Pure 1D N-cut 32-cell WSE FFN emulation -- TPUSH AllGather" << std::endl;
    std::cout << "  grid=" << FFN_NCUT_ROWS << "x" << FFN_NCUT_COLS << " (" << FFN_NCUT_CELLS << " cells)"
              << "  T=" << FFN_NCUT_T << "  H=" << FFN_NCUT_H << "  I=" << FFN_NCUT_I << std::endl;
    std::cout << "  per-cell: I_shard=" << FFN_NCUT_I_SHARD << "  H_shard=" << FFN_NCUT_H_SHARD
              << "  row_block=" << FFN_NCUT_ROW_BLOCK << std::endl;
    std::cout << "  AG = TPUSH row relay(8-way) + TPUSH col relay(4-way); emulated on " << physCores << " phys AICores"
              << std::endl;
    std::cout << "================================================================" << std::endl;

    bool ok = RunSingleDevice(physCores);
    std::cout << (ok ? "[SUCCESS] 32-cell N-cut FFN GridPipe TPUSH AllGather PASS." :
                       "[FAILED] 32-cell N-cut FFN GridPipe TPUSH AllGather FAILED.")
              << std::endl;
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ok ? 0 : 1;
}
