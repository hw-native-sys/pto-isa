/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the software repository for the full text of the License.
*/

// Pure 1D N-cut 32-cell WSE FFN emulation host driver -- ReduceSum variant (Option B).
//
// 32 cells split I, broadcast x, each computes a full-H down partial, and the 32
// partials are reduced EAST 8-way (row) then SOUTH 4-way (col).  3 phase-dispatched
// launches (A compute, B EAST-reduce waves, C SOUTH-reduce wave) with host barriers;
// 32 > 24 phys cores so the reduce is waved by whole rows / a single col.  Real
// DeepSeek-v4 Pro shapes M=8, H=7168, I=3072.

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
    void *reduce_windows_dev = nullptr;
    void *hccl_ctx_dev = nullptr;
    size_t reduceWindowsBytes = 0;
    // Per-cell GM intermediates that persist across the launches.
    void *partial_dev = nullptr;      // [cells, T, H] fp32 (Phase A -> B)
    void *row_partial_dev = nullptr;  // [rows,  T, H] fp32 (Phase B -> C, sink col 7 per row)
    void *gate_partial_dev = nullptr; // [cells, T, I_shard] fp32 (cube->vec C2V, Phase A)
    void *up_partial_dev = nullptr;
    void *hidden_dev = nullptr;       // [cells, T, I_shard] half (vec->cube V2C, Phase A)

    uint64_t ffts = 0;
    uint32_t fftsLen = 0;

    int rows = FFN_NCUT_ROWS;
    int cols = FFN_NCUT_COLS;
    int cells = FFN_NCUT_ROWS * FFN_NCUT_COLS;
    int physCores = 24;

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

static bool InitGridPipeContext(DeviceResources &r)
{
    r.reduceWindowsBytes = static_cast<size_t>(r.cells) * FFN_RS_REDUCE_WIN;
    if (aclrtMalloc(&r.reduce_windows_dev, r.reduceWindowsBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc(reduce_windows) failed" << std::endl;
        return false;
    }
    aclrtMemset(r.reduce_windows_dev, r.reduceWindowsBytes, 0, r.reduceWindowsBytes);

    CommDeviceContext hostCtx{};
    hostCtx.rankId = 0;
    hostCtx.rankNum = static_cast<uint32_t>(r.cells);
    hostCtx.winSize = static_cast<uint64_t>(FFN_RS_REDUCE_WIN);
    uint64_t base = reinterpret_cast<uint64_t>(r.reduce_windows_dev);
    for (int i = 0; i < r.cells && i < HCCL_MAX_RANK_NUM; ++i) {
        hostCtx.windowsIn[i] = base + static_cast<uint64_t>(i) * FFN_RS_REDUCE_WIN;
        hostCtx.windowsOut[i] = hostCtx.windowsIn[i];
    }
    if (aclrtMalloc(&r.hccl_ctx_dev, sizeof(CommDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc(hccl_ctx) failed" << std::endl;
        return false;
    }
    if (aclrtMemcpy(r.hccl_ctx_dev, sizeof(CommDeviceContext), &hostCtx, sizeof(CommDeviceContext),
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

    r.partial_dev = MallocBufOrDie(static_cast<size_t>(r.cells) * FFN_RS_PARTIAL_BYTES, "partial");
    r.row_partial_dev = MallocBufOrDie(static_cast<size_t>(r.rows) * FFN_RS_ROW_PARTIAL_BYTES, "row_partial");
    r.gate_partial_dev = MallocBufOrDie(static_cast<size_t>(r.cells) * FFN_NCUT_GATE_PARTIAL_BYTES, "gate_partial");
    r.up_partial_dev = MallocBufOrDie(static_cast<size_t>(r.cells) * FFN_NCUT_GATE_PARTIAL_BYTES, "up_partial");
    r.hidden_dev = MallocBufOrDie(static_cast<size_t>(r.cells) * FFN_NCUT_HIDDEN_SHARD_BYTES, "hidden");
    if (!r.partial_dev || !r.row_partial_dev || !r.gate_partial_dev || !r.up_partial_dev || !r.hidden_dev) {
        return false;
    }

    if (!r.batcher.AllocateNcutReduce(r.rows, r.cols)) {
        return false;
    }
    if (!InitGridPipeContext(r)) {
        return false;
    }

    rtGetC2cCtrlAddr(&r.ffts, &r.fftsLen);
    if (r.ffts == 0) {
        std::cerr << "[ERROR] rtGetC2cCtrlAddr returned null FFTS address" << std::endl;
        return false;
    }

    std::cout << "[INFO] grid=" << r.rows << "x" << r.cols << " cells=" << r.cells << " physCores=" << r.physCores
              << " dataDir=" << r.dataDir << " reduceWin=" << r.reduceWindowsBytes << "B" << std::endl;
    return true;
}

static bool LoadBatcherData(DeviceResources &r)
{
    return r.batcher.LoadAndDistributeNcutReduce(r.dataDir);
}

static const char *GridPipeFaultName(uint32_t code)
{
    switch (code) {
        case 0x101: return "push north boundary";
        case 0x102: return "push east boundary";
        case 0x103: return "push west boundary";
        case 0x104: return "push south boundary";
        case 0x105: return "push source boundary";
        case 0x201: return "pop north boundary";
        case 0x202: return "pop east boundary";
        case 0x203: return "pop west boundary";
        case 0x204: return "pop south boundary";
        case 0x205: return "pop non-local segment";
        case 0x301: return "wait ready timeout";
        case 0x302: return "wait free timeout";
        default: return "unknown";
    }
}

static bool CheckGridPipeFaults(DeviceResources &r)
{
    constexpr size_t kFlagWords = static_cast<size_t>(FFN_NCUT_GRID_FLAGS_BYTES) / sizeof(uint32_t);
    std::vector<uint32_t> flags(static_cast<size_t>(r.cells) * kFlagWords, 0);
    for (int cell = 0; cell < r.cells; ++cell) {
        auto *src = reinterpret_cast<uint8_t *>(r.reduce_windows_dev) + static_cast<size_t>(cell) * FFN_RS_REDUCE_WIN;
        auto *dst = flags.data() + static_cast<size_t>(cell) * kFlagWords;
        if (aclrtMemcpy(dst, kFlagWords * sizeof(uint32_t), src, kFlagWords * sizeof(uint32_t),
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            std::cerr << "[ERROR] GridPipe flag D2H failed (cell " << cell << ")" << std::endl;
            return false;
        }
    }
    bool ok = true;
    for (int cell = 0; cell < r.cells; ++cell) {
        int row = cell / r.cols;
        int col = cell - row * r.cols;
        const uint32_t *cellFlags = flags.data() + static_cast<size_t>(cell) * kFlagWords;
        for (size_t i = 0; i < kFlagWords; ++i) {
            uint32_t value = cellFlags[i];
            if (value >= 0x100U) {
                std::cerr << "[ERROR] GridPipe fault cell=" << cell << " row=" << row << " col=" << col << " word=" << i
                          << " code=0x" << std::hex << value << std::dec << " (" << GridPipeFaultName(value) << ")"
                          << std::endl;
                ok = false;
            }
        }
    }
    return ok;
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
        std::cerr << "[ERROR] golden.bin mismatch: " << goldenPath << " (got " << fileSize << " expected " << outputBytes
                  << ")" << std::endl;
        return false;
    }
    std::cout << "[INFO] ResultCmp 32-cell N-cut ReduceSum output vs golden:" << std::endl;
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
    freeBuf(r.hccl_ctx_dev);
    freeBuf(r.reduce_windows_dev);
    freeBuf(r.partial_dev);
    freeBuf(r.row_partial_dev);
    freeBuf(r.gate_partial_dev);
    freeBuf(r.up_partial_dev);
    freeBuf(r.hidden_dev);
    if (r.stream) {
        aclrtDestroyStream(r.stream);
        r.stream = nullptr;
    }
}

static bool LaunchWave(DeviceResources &r, int phase, int rowStart, int colStart, int waveCols, int blockCount,
                       const char *tag)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    launchDistributedFfnGridTreduceReduceSumMixedKernel(
        reinterpret_cast<uint8_t *>(r.ffts), reinterpret_cast<uint8_t *>(r.reduce_windows_dev), r.batcher.NcutXFull(),
        r.batcher.NcutWGateShards(), r.batcher.NcutWUpShards(), r.batcher.NcutReduceWDownShards(),
        reinterpret_cast<uint8_t *>(r.partial_dev), reinterpret_cast<uint8_t *>(r.row_partial_dev), r.batcher.NcutYFull(),
        reinterpret_cast<uint8_t *>(r.gate_partial_dev), reinterpret_cast<uint8_t *>(r.up_partial_dev),
        reinterpret_cast<uint8_t *>(r.hidden_dev), reinterpret_cast<uint8_t *>(r.hccl_ctx_dev), phase, rowStart, colStart,
        waveCols, FFN_NCUT_ROWS, FFN_NCUT_COLS, blockCount, r.stream);
    aclError rc = aclrtSynchronizeStream(r.stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    bool faultsOk = (rc == ACL_SUCCESS) && CheckGridPipeFaults(r);
    std::cout << "[INFO] phase " << tag << "  blocks=" << blockCount << "  " << us << " us  (rc=" << static_cast<int>(rc)
              << " faults_ok=" << (faultsOk ? 1 : 0) << ")" << std::endl;
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

    // Phase A: gate+up+SwiGLU+down -> partial_c.  No comm: launch all 32 cells.
    if (!LaunchWave(r, /*phase=*/0, 0, 0, FFN_NCUT_COLS, FFN_NCUT_CELLS, "A gate+up+SwiGLU+down (no comm)")) {
        Cleanup(r);
        return false;
    }

    // Phase B: EAST 8-way reduce (row).  Wave by WHOLE ROWS so every EAST chain is
    // wholly present in its wave (cap 8 cells/wave = 1 row).
    constexpr int kCommWaveCap = 8;
    const int rowsPerWave = std::max(1, std::min(r.physCores, kCommWaveCap) / FFN_NCUT_COLS);
    for (int rs = 0; rs < FFN_NCUT_ROWS; rs += rowsPerWave) {
        int waveRows = std::min(rowsPerWave, FFN_NCUT_ROWS - rs);
        if (!LaunchWave(r, /*phase=*/1, rs, 0, FFN_NCUT_COLS, waveRows * FFN_NCUT_COLS, "B EAST row reduce")) {
            Cleanup(r);
            return false;
        }
    }

    // Phase C: SOUTH 4-way reduce at col 7 (the 4 row-sink cells).  One wave of 4.
    if (!LaunchWave(r, /*phase=*/2, 0, FFN_NCUT_COLS - 1, 1, FFN_NCUT_ROWS, "C SOUTH col reduce")) {
        Cleanup(r);
        return false;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    int bWaves = 1 + (FFN_NCUT_ROWS + rowsPerWave - 1) / rowsPerWave;
    std::cout << "[INFO] emulated 32-cell N-cut TREDUCE ReduceSum on " << r.physCores
              << " phys AICores, 3 launches (B=" << bWaves << " EAST waves + 1 SOUTH wave), total " << us
              << " us" << std::endl;

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
    std::cout << "  Pure 1D N-cut 32-cell WSE FFN emulation -- TREDUCE ReduceSum (Option B)" << std::endl;
    std::cout << "  grid=" << FFN_NCUT_ROWS << "x" << FFN_NCUT_COLS << " (" << FFN_NCUT_CELLS << " cells)"
              << "  T=" << FFN_NCUT_T << "  H=" << FFN_NCUT_H << "  I=" << FFN_NCUT_I << std::endl;
    std::cout << "  per-cell I_shard=" << FFN_NCUT_I_SHARD << "  reduce EAST(8-way)+SOUTH(4-way), H-chunk="
              << FFN_RS_REDUCE_H_BASE << std::endl;
    std::cout << "================================================================" << std::endl;

    bool ok = RunSingleDevice(physCores);
    std::cout << (ok ? "[SUCCESS] 32-cell N-cut FFN GridPipe TREDUCE ReduceSum PASS." :
                       "[FAILED] 32-cell N-cut FFN GridPipe TREDUCE ReduceSum FAILED.")
              << std::endl;
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ok ? 0 : 1;
}
