/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Single-device multi-block FFN GridPipe AllGather driver.
//
// gridRows*gridCols blocks form a logical grid on one NPU.  Each cell computes
// one model-parallel FFN shard, then cells in the same row AllGather fp16
// hidden shards before down projection.  Down is sharded by output H columns.

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
    void *gate_partial_dev = nullptr;
    void *up_partial_dev = nullptr;
    void *hidden_dev = nullptr;
    void *down_partial_dev = nullptr;
    void *gather_pipe_windows_dev = nullptr;
    void *fake_hccl_ctx_dev = nullptr;
    uint64_t ffts = 0;
    uint32_t fftsLen = 0;

    size_t rows = static_cast<size_t>(FFN_GRID_ROWS);
    size_t cols = static_cast<size_t>(FFN_GRID_COLS);
    size_t cells = static_cast<size_t>(FFN_GRID_ROWS) * static_cast<size_t>(FFN_GRID_COLS);
    size_t gatePartialBytes = 0;
    size_t upPartialBytes = 0;
    size_t hiddenBytes = 0;
    size_t downPartialBytes = 0;
    size_t gatherPipeBytes = 0;

    std::string dataDir = "./out";

    // GM-simulated Batcher: owns the full input + resident weights + per-col
    // shards + the output collection region.  Self-frees on destruction.
    FfnBatcher batcher;
};

static bool ParseDeviceIdValue(const char *value, int &deviceId)
{
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    deviceId = static_cast<int>(parsed);
    return true;
}

static bool ParseDeviceIdEnv(const char *name, int &deviceId)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    if (!ParseDeviceIdValue(value, deviceId)) {
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
            if (i + 1 >= argc || !ParseDeviceIdValue(argv[i + 1], deviceId)) {
                std::cerr << "[ERROR] invalid --device-id value" << std::endl;
                std::exit(1);
            }
            return deviceId;
        }
        constexpr const char *kPrefix = "--device-id=";
        constexpr size_t kPrefixLen = 12;
        if (std::strncmp(argv[i], kPrefix, kPrefixLen) == 0) {
            if (!ParseDeviceIdValue(argv[i] + kPrefixLen, deviceId)) {
                std::cerr << "[ERROR] invalid --device-id value" << std::endl;
                std::exit(1);
            }
            return deviceId;
        }
    }

    if (ParseDeviceIdEnv("FFN_GRID_DEVICE_ID", deviceId) || ParseDeviceIdEnv("ASCEND_DEVICE_ID", deviceId) ||
        ParseDeviceIdEnv("DEVICE_ID", deviceId)) {
        return deviceId;
    }
    return 0;
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
        std::cout << "[INFO] rtSetDevice(" << device_id << ") begin" << std::endl;
        rtError_t rtRet = rtSetDevice(device_id);
        if (rtRet != RT_ERROR_NONE) {
            std::cerr << "[ERROR] rtSetDevice(" << device_id << ") failed: " << static_cast<int>(rtRet) << std::endl;
            return false;
        }
        std::cout << "[INFO] rtSetDevice(" << device_id << ") done" << std::endl;
    } else {
        std::cout << "[INFO] rtSetDevice skipped; set FFN_GRID_USE_RT_SET_DEVICE=1 to enable" << std::endl;
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

static bool InitLocalGridPipeContext(DeviceResources &r)
{
    r.gatherPipeBytes = r.cells * static_cast<size_t>(FFN_GRID_WINDOW_BYTES);
    if (aclrtMalloc(&r.gather_pipe_windows_dev, r.gatherPipeBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc(gather_pipe_windows) failed" << std::endl;
        return false;
    }
    aclrtMemset(r.gather_pipe_windows_dev, r.gatherPipeBytes, 0, r.gatherPipeBytes);

    CommDeviceContext hostCtx{};
    hostCtx.rankId = 0;
    hostCtx.rankNum = static_cast<uint32_t>(r.cells);
    hostCtx.winSize = static_cast<uint64_t>(FFN_GRID_WINDOW_BYTES);
    uint64_t base = reinterpret_cast<uint64_t>(r.gather_pipe_windows_dev);
    for (size_t i = 0; i < r.cells && i < HCCL_MAX_RANK_NUM; ++i) {
        hostCtx.windowsIn[i] = base + i * static_cast<size_t>(FFN_GRID_WINDOW_BYTES);
        hostCtx.windowsOut[i] = hostCtx.windowsIn[i];
    }

    if (aclrtMalloc(&r.fake_hccl_ctx_dev, sizeof(CommDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc(fake_hccl_ctx) failed" << std::endl;
        return false;
    }
    if (aclrtMemcpy(r.fake_hccl_ctx_dev, sizeof(CommDeviceContext), &hostCtx, sizeof(CommDeviceContext),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy(fake_hccl_ctx) failed" << std::endl;
        return false;
    }
    return true;
}

static bool AllocateResources(DeviceResources &r)
{
    if (r.cells == 0 || r.cells > HCCL_MAX_RANK_NUM) {
        std::cerr << "[ERROR] invalid cell count " << r.cells << "; supported range is 1.." << HCCL_MAX_RANK_NUM
                  << std::endl;
        return false;
    }
    if (FFN_MODEL_TILE % FFN_GRID_COLS != 0) {
        std::cerr << "[ERROR] AllGather split requires MODEL_TILE divisible by GRID_COLS" << std::endl;
        return false;
    }
    if (const char *env = std::getenv("FFN_GRID_DATA_DIR")) {
        r.dataDir = env;
    }

    if (aclrtCreateStream(&r.stream) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtCreateStream failed" << std::endl;
        return false;
    }

    // Per-cell C2V/V2C working buffers only (x / weights / y now live in the
    // Batcher GM arena -- full resident weights, per-col shards, per-row x, and
    // the collected output).
    r.gatePartialBytes = r.cells * static_cast<size_t>(FFN_GATE_PARTIAL_BYTES);
    r.upPartialBytes = r.cells * static_cast<size_t>(FFN_UP_PARTIAL_BYTES);
    r.hiddenBytes = r.cells * static_cast<size_t>(FFN_HIDDEN_FULL_BYTES);
    r.downPartialBytes = r.cells * static_cast<size_t>(FFN_DOWN_PARTIAL_BYTES);

    aclrtMalloc(&r.gate_partial_dev, r.gatePartialBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.up_partial_dev, r.upPartialBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.hidden_dev, r.hiddenBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.down_partial_dev, r.downPartialBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!r.gate_partial_dev || !r.up_partial_dev || !r.hidden_dev || !r.down_partial_dev) {
        std::cerr << "[ERROR] aclrtMalloc(working buffers) failed" << std::endl;
        return false;
    }

    aclrtMemset(r.gate_partial_dev, r.gatePartialBytes, 0, r.gatePartialBytes);
    aclrtMemset(r.up_partial_dev, r.upPartialBytes, 0, r.upPartialBytes);
    aclrtMemset(r.hidden_dev, r.hiddenBytes, 0, r.hiddenBytes);
    aclrtMemset(r.down_partial_dev, r.downPartialBytes, 0, r.downPartialBytes);

    if (!r.batcher.Allocate(r.rows, r.cols, /*splitMode=*/"allgather")) {
        return false;
    }

    if (!InitLocalGridPipeContext(r)) {
        return false;
    }

    rtGetC2cCtrlAddr(&r.ffts, &r.fftsLen);
    if (r.ffts == 0) {
        std::cerr << "[ERROR] rtGetC2cCtrlAddr returned null FFTS address" << std::endl;
        return false;
    }

    std::cout << "[INFO] grid=" << r.rows << "x" << r.cols << " cells=" << r.cells << " dataDir=" << r.dataDir
              << " gatherPipeBytes=" << r.gatherPipeBytes << std::endl;
    return true;
}

// Drive the GM-simulated Batcher: load resident full weights into GM, split them
// into per-column shards, and broadcast the full input.  Replaces the legacy
// per-cell LoadInputs/LoadWeights (the kernel now reads x per-row, weight shards
// per-col, and writes output into the Batcher y region).
static bool LoadBatcherData(DeviceResources &r)
{
    if (!r.batcher.LoadResidentWeights(r.dataDir) || !r.batcher.DistributeWeights() ||
        !r.batcher.BroadcastInput(r.dataDir)) {
        return false;
    }
    return true;
}

static bool VerifyOutput(DeviceResources &r)
{
    const size_t outputElems = r.rows * static_cast<size_t>(FFN_TILE_ELEMS);
    const size_t outputBytes = r.rows * static_cast<size_t>(FFN_Y_OUTPUT_BYTES);

    std::vector<float> outHost(outputElems);
    if (aclrtMemcpy(outHost.data(), outputBytes, r.batcher.YFull(), outputBytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
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

    std::cout << "[INFO] ResultCmp single-device GridPipe AllGather output shards vs golden:" << std::endl;
    return PtoTestCommon::ResultCmp(golden, outHost.data(), 0.001f);
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

static bool CheckGridPipeFaults(DeviceResources &r)
{
    constexpr size_t kFlagWordsPerWindow = static_cast<size_t>(FFN_GRID_FLAGS_BYTES) / sizeof(uint32_t);
    std::vector<uint32_t> flags(r.cells * kFlagWordsPerWindow, 0);
    for (size_t cell = 0; cell < r.cells; ++cell) {
        auto *src = reinterpret_cast<uint8_t *>(r.gather_pipe_windows_dev) + cell * FFN_GRID_WINDOW_BYTES;
        auto *dst = flags.data() + cell * kFlagWordsPerWindow;
        if (aclrtMemcpy(dst, kFlagWordsPerWindow * sizeof(uint32_t), src, kFlagWordsPerWindow * sizeof(uint32_t),
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            std::cerr << "[ERROR] GridPipe flag D2H memcpy failed for cell " << cell << std::endl;
            return false;
        }
    }

    bool ok = true;
    for (size_t cell = 0; cell < r.cells; ++cell) {
        size_t row = cell / r.cols;
        size_t col = cell - row * r.cols;
        const uint32_t *cellFlags = flags.data() + cell * kFlagWordsPerWindow;
        for (size_t i = 0; i < kFlagWordsPerWindow; ++i) {
            uint32_t value = cellFlags[i];
            if (value >= 0x100U) {
                std::cerr << "[ERROR] GridPipe fault cell=" << cell << " row=" << row << " col=" << col
                          << " flagWord=" << i << " code=0x" << std::hex << value << std::dec << " ("
                          << GridPipeFaultName(value) << ")" << std::endl;
                ok = false;
            }
        }
    }
    return ok;
}

static void Cleanup(DeviceResources &r)
{
    if (r.fake_hccl_ctx_dev) {
        aclrtFree(r.fake_hccl_ctx_dev);
        r.fake_hccl_ctx_dev = nullptr;
    }
    if (r.gather_pipe_windows_dev) {
        aclrtFree(r.gather_pipe_windows_dev);
        r.gather_pipe_windows_dev = nullptr;
    }
    if (r.gate_partial_dev) {
        aclrtFree(r.gate_partial_dev);
        r.gate_partial_dev = nullptr;
    }
    if (r.up_partial_dev) {
        aclrtFree(r.up_partial_dev);
        r.up_partial_dev = nullptr;
    }
    if (r.hidden_dev) {
        aclrtFree(r.hidden_dev);
        r.hidden_dev = nullptr;
    }
    if (r.down_partial_dev) {
        aclrtFree(r.down_partial_dev);
        r.down_partial_dev = nullptr;
    }
    if (r.stream) {
        aclrtDestroyStream(r.stream);
        r.stream = nullptr;
    }
    // r.batcher frees its own GM arena in its destructor.
}

static bool RunSingleDevice()
{
    DeviceResources r;
    if (!AllocateResources(r) || !LoadBatcherData(r)) {
        Cleanup(r);
        return false;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    launchDistributedFfnGridAllGatherMixedKernel(
        reinterpret_cast<uint8_t *>(r.ffts), reinterpret_cast<uint8_t *>(r.gather_pipe_windows_dev),
        r.batcher.XFull(), r.batcher.WGateShards(), r.batcher.WUpShards(), r.batcher.WDownShards(),
        reinterpret_cast<uint8_t *>(r.gate_partial_dev), reinterpret_cast<uint8_t *>(r.up_partial_dev),
        reinterpret_cast<uint8_t *>(r.hidden_dev), reinterpret_cast<uint8_t *>(r.down_partial_dev),
        r.batcher.YFull(), reinterpret_cast<uint8_t *>(r.fake_hccl_ctx_dev), FFN_GRID_ROWS, FFN_GRID_COLS, r.stream);
    aclError mixedRet = aclrtSynchronizeStream(r.stream);
    bool gridPipeOk = (mixedRet == ACL_SUCCESS) && CheckGridPipeFaults(r);

    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::cout << "[INFO] launch+sync " << us << " us  (mixed rc=" << static_cast<int>(mixedRet)
              << " gridpipe_ok=" << (gridPipeOk ? 1 : 0);
    std::cout << ")" << std::endl;

    bool syncOk = (mixedRet == ACL_SUCCESS) && gridPipeOk;
    bool verifyOk = syncOk && VerifyOutput(r);
    Cleanup(r);
    return syncOk && verifyOk;
}

int main(int argc, char **argv)
{
    int deviceId = GetDeviceId(argc, argv);
    std::cout << "[INFO] using device " << deviceId << std::endl;
    if (!InitAcl(deviceId)) {
        return 1;
    }

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Single-device multi-block FFN GridPipe AllGather demo" << std::endl;
    std::cout << "  grid=" << FFN_GRID_ROWS << "x" << FFN_GRID_COLS << " cells=" << (FFN_GRID_ROWS * FFN_GRID_COLS)
              << " tile=" << FFN_TOKEN_TILE << "x" << FFN_MODEL_TILE << " ffnTile=" << FFN_FFN_TILE << std::endl;
    std::cout << "  Mode: hidden AllGather across cols, down split by output H columns" << std::endl;
    std::cout << "================================================================" << std::endl;

    bool ok = RunSingleDevice();
    std::cout << (ok ? "[SUCCESS] Single-device multi-block FFN GridPipe AllGather PASS." :
                       "[FAILED] Single-device multi-block FFN GridPipe AllGather FAILED.")
              << std::endl;
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ok ? 0 : 1;
}
