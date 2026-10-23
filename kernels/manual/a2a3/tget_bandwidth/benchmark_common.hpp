/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef BENCHMARK_COMMON_HPP_
#define BENCHMARK_COMMON_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <pto/pto-inst.hpp>

#include "common.hpp"
#include "pto/comm/async/sdma/sdma_types.hpp"

namespace benchmark {

constexpr size_t kBytesPerKiB = 1024;
constexpr uint32_t kMaxDeviceBaselinePosts = 256;
constexpr double kDeviceCycleSeconds = 20.0e-9;

inline uint64_t ReadEnvUint64(const char* name, uint64_t defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    return end != value && *end == '\0' ? static_cast<uint64_t>(parsed) : defaultValue;
}

template <typename T>
T DeviceBaselinePattern(int rankId, int outerIndex, bool measured, uint32_t post, size_t elemIndex)
{
    constexpr uint64_t kPatternSpan = 7000000;
    constexpr uint64_t kWarmupBase = 1;
    constexpr uint64_t kMeasuredBase = 8000001;
    const uint64_t key = static_cast<uint64_t>(rankId) * 524287 + static_cast<uint64_t>(outerIndex) * 131071 +
                         static_cast<uint64_t>(post) * 4099 + elemIndex;
    return static_cast<T>((measured ? kMeasuredBase : kWarmupBase) + key % kPatternSpan);
}

inline AICORE uint64_t GetSyscnt()
{
    uint64_t syscnt;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

inline bool CheckAclCall(aclError ret, const char* op)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] " << op << " failed: " << static_cast<int>(ret) << std::endl;
        return false;
    }
    return true;
}

struct DeviceBaselineConfig {
    uint64_t transferBytesValue;
    uint64_t blockDivisorValue;
    uint64_t queueNumValue;
    uint64_t postCountValue;
    uint64_t outerWarmupValue;
    uint64_t outerItersValue;
    uint64_t innerWarmupValue;
    uint64_t innerItersValue;
    bool waitEachEvent;
    size_t transferBytes;
    size_t totalBytes;
    size_t elemCount;
    uint32_t blockBytes;
    uint32_t queueNum;
    uint32_t postCount;
    int outerWarmup;
    int outerIters;
    int innerWarmup;
    int innerIters;
};

struct DeviceBaselineEnvNames {
    const char* transferBytes;
    const char* blockDivisor;
    const char* queueNum;
    const char* postCount;
    const char* outerWarmup;
    const char* outerIters;
    const char* innerWarmup;
    const char* innerIters;
    const char* waitEachEvent;
};

template <typename T>
bool IsTransferConfigValid(const DeviceBaselineConfig& config)
{
    return config.transferBytesValue > 0 && config.transferBytesValue <= UINT32_MAX &&
           config.transferBytesValue % sizeof(T) == 0 && config.blockDivisorValue > 0 &&
           config.transferBytesValue % config.blockDivisorValue == 0 &&
           config.transferBytesValue / config.blockDivisorValue >= pto::comm::sdma::kSdmaMinTransferBytes;
}

inline bool IsSubmissionConfigValid(const DeviceBaselineConfig& config)
{
    return config.queueNumValue > 0 && config.queueNumValue <= pto::comm::sdma::kSdmaMaxChannel &&
           config.postCountValue > 0 && config.postCountValue <= kMaxDeviceBaselinePosts;
}

inline bool IsIterationConfigValid(const DeviceBaselineConfig& config)
{
    return config.outerItersValue > 0 && config.outerItersValue <= INT32_MAX &&
           config.outerWarmupValue <= INT32_MAX - config.outerItersValue && config.innerItersValue > 0 &&
           config.innerItersValue <= INT32_MAX && config.innerWarmupValue <= INT32_MAX;
}

inline bool IsDeviceAllocationSizeValid(const DeviceBaselineConfig& config)
{
    return config.transferBytesValue <= SIZE_MAX / config.postCountValue;
}

template <typename T>
bool FinalizeDeviceBaselineConfig(DeviceBaselineConfig& config)
{
    if (!IsTransferConfigValid<T>(config) || !IsSubmissionConfigValid(config) || !IsIterationConfigValid(config) ||
        !IsDeviceAllocationSizeValid(config)) {
        return false;
    }
    config.transferBytes = static_cast<size_t>(config.transferBytesValue);
    config.totalBytes = config.transferBytes * static_cast<size_t>(config.postCountValue);
    config.elemCount = config.transferBytes / sizeof(T);
    config.blockBytes = static_cast<uint32_t>(config.transferBytesValue / config.blockDivisorValue);
    config.queueNum = static_cast<uint32_t>(config.queueNumValue);
    config.postCount = static_cast<uint32_t>(config.postCountValue);
    config.outerWarmup = static_cast<int>(config.outerWarmupValue);
    config.outerIters = static_cast<int>(config.outerItersValue);
    config.innerWarmup = static_cast<int>(config.innerWarmupValue);
    config.innerIters = static_cast<int>(config.innerItersValue);
    return true;
}

template <typename T>
bool LoadDeviceBaselineConfig(const DeviceBaselineEnvNames& names, DeviceBaselineConfig& config)
{
    config.transferBytesValue = ReadEnvUint64(names.transferBytes, 128 * kBytesPerKiB);
    config.blockDivisorValue = ReadEnvUint64(names.blockDivisor, 1);
    config.queueNumValue = ReadEnvUint64(names.queueNum, 1);
    config.postCountValue = ReadEnvUint64(names.postCount, 1);
    config.outerWarmupValue = ReadEnvUint64(names.outerWarmup, 1);
    config.outerItersValue = ReadEnvUint64(names.outerIters, 5);
    config.innerWarmupValue = ReadEnvUint64(names.innerWarmup, 1);
    config.innerItersValue = ReadEnvUint64(names.innerIters, 10);
    config.waitEachEvent = ReadEnvUint64(names.waitEachEvent, 0) != 0;
    return FinalizeDeviceBaselineConfig<T>(config);
}

inline void PrintDeviceBaselineConfig(const char* instr, const DeviceBaselineConfig& config)
{
    std::cout << "\n================ " << instr << " Device Baseline ================" << std::endl;
    std::cout << "bytes=" << config.transferBytes << " block_bytes=" << config.blockBytes
              << " queue_num=" << config.queueNum << " post_count=" << config.postCount
              << " outer_warmup=" << config.outerWarmup << " outer_iters=" << config.outerIters
              << " inner_warmup=" << config.innerWarmup << " inner_iters=" << config.innerIters
              << " wait_each_event=" << config.waitEachEvent << " verify_each_outer=1" << std::endl;
}

inline void PrintDeviceBaselineResult(const char* instr, const DeviceBaselineConfig& config, uint64_t aggregateCycles)
{
    const double kernelTotalCycles =
        static_cast<double>(aggregateCycles) / static_cast<double>(config.outerIters * config.innerIters);
    const double bandwidthGBps =
        static_cast<double>(config.totalBytes) / (kernelTotalCycles * kDeviceCycleSeconds) / 1.0e9;
    std::cout << "[VERIFY] instr=" << instr << " status=PASS checked_posts=" << config.postCount << std::endl;
    std::cout << std::fixed << std::setprecision(4) << "[DEVICE_BASELINE] instr=" << instr
              << " bytes=" << config.transferBytes << " block_divisor=" << config.blockDivisorValue
              << " block_bytes=" << config.blockBytes << " queue_num=" << config.queueNum
              << " sqe_num_per_post=" << config.blockDivisorValue
              << " sqe_num_total=" << config.blockDivisorValue * config.postCountValue
              << " post_count=" << config.postCount << " outer_warmup=" << config.outerWarmup
              << " outer_iters=" << config.outerIters << " inner_warmup=" << config.innerWarmup
              << " inner_iters=" << config.innerIters << " wait_each_event=" << config.waitEachEvent
              << " verify_each_outer=1 kernel_total_cycles=" << kernelTotalCycles
              << " device_bandwidth_GBps=" << bandwidthGBps << std::endl;
}

template <typename T>
struct DeviceBaselineResources {
    T* inputHost = nullptr;
    T* verifyHost = nullptr;
    uint64_t* profileBufDev = nullptr;
    uint64_t* profileBufHost = nullptr;
    T* shmem = nullptr;
    T* sendShmem = nullptr;
    T* recvShmem = nullptr;
    SdmaWorkspaceManager sdmaMgr;

    bool Allocate(size_t totalBytes)
    {
        constexpr size_t kProfileCount = 2;
        return CheckAclCall(
                   aclrtMallocHost(reinterpret_cast<void**>(&inputHost), totalBytes),
                   "aclrtMallocHost(device baseline input)") &&
               CheckAclCall(
                   aclrtMallocHost(reinterpret_cast<void**>(&verifyHost), totalBytes),
                   "aclrtMallocHost(device baseline verify)") &&
               CheckAclCall(
                   aclrtMalloc(
                       reinterpret_cast<void**>(&profileBufDev), kProfileCount * sizeof(uint64_t),
                       ACL_MEM_MALLOC_HUGE_FIRST),
                   "aclrtMalloc(device baseline profile)") &&
               CheckAclCall(
                   aclrtMallocHost(reinterpret_cast<void**>(&profileBufHost), kProfileCount * sizeof(uint64_t)),
                   "aclrtMallocHost(device baseline profile)");
    }

    void MapWindow(TestContext& ctx, int rankId, size_t totalBytes, size_t requiredWindowBytes)
    {
        uint64_t localWinBase = ctx.hostCtx.windowsIn[rankId];
        size_t winOffset = 0;
        shmem = reinterpret_cast<T*>(WindowAlloc(localWinBase, winOffset, requiredWindowBytes));
        auto* shmemBytes = reinterpret_cast<uint8_t*>(shmem);
        sendShmem = reinterpret_cast<T*>(shmemBytes + 64 * sizeof(int32_t));
        recvShmem = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(sendShmem) + totalBytes);
    }

    void Release(TestContext& ctx, bool finalizeSdma)
    {
        if (inputHost != nullptr) {
            ctx.aclStatus |= aclrtFreeHost(inputHost);
        }
        if (verifyHost != nullptr) {
            ctx.aclStatus |= aclrtFreeHost(verifyHost);
        }
        if (profileBufDev != nullptr) {
            ctx.aclStatus |= aclrtFree(profileBufDev);
        }
        if (profileBufHost != nullptr) {
            ctx.aclStatus |= aclrtFreeHost(profileBufHost);
        }
        if (finalizeSdma) {
            sdmaMgr.Finalize();
        }
    }
};

template <typename T>
void FillDeviceBaselinePattern(
    T* buffer, int rankId, int patternOuter, bool measured, uint32_t postCount, size_t elemCount);

template <typename T>
bool PrepareDeviceBaselineSource(
    int rankId, int sourceRank, int patternOuter, bool measured, const DeviceBaselineConfig& config,
    DeviceBaselineResources<T>& resources)
{
    bool sourceOk = true;
    if (rankId == sourceRank) {
        FillDeviceBaselinePattern(
            resources.inputHost, rankId, patternOuter, measured, config.postCount, config.elemCount);
        sourceOk = CheckAclCall(
            aclrtMemcpy(
                resources.sendShmem, config.totalBytes, resources.inputHost, config.totalBytes,
                ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy(device baseline source)");
    }
    char sourceStatus = sourceOk ? 1 : 0;
    CommMpiBcast(&sourceStatus, 1, COMM_MPI_CHAR, sourceRank);
    return sourceStatus != 0;
}

template <typename T>
void FillDeviceBaselinePattern(
    T* buffer, int rankId, int patternOuter, bool measured, uint32_t postCount, size_t elemCount)
{
    for (uint32_t post = 0; post < postCount; ++post) {
        for (size_t i = 0; i < elemCount; ++i) {
            const size_t index = static_cast<size_t>(post) * elemCount + i;
            buffer[index] = DeviceBaselinePattern<T>(rankId, patternOuter, measured, post, i);
        }
    }
}

template <typename T>
bool VerifyDeviceBaselinePattern(
    const T* buffer, int expectedRank, int patternOuter, bool measured, int outer, uint32_t postCount, size_t elemCount)
{
    for (uint32_t post = 0; post < postCount; ++post) {
        for (size_t i = 0; i < elemCount; ++i) {
            const size_t index = static_cast<size_t>(post) * elemCount + i;
            const T expected = DeviceBaselinePattern<T>(expectedRank, patternOuter, measured, post, i);
            if (buffer[index] != expected) {
                std::cerr << "[VERIFY] status=FAIL outer=" << outer << " post=" << post << " index=" << i
                          << " expected=" << static_cast<float>(expected)
                          << " actual=" << static_cast<float>(buffer[index]) << std::endl;
                return false;
            }
        }
    }
    return true;
}

} // namespace benchmark

#endif // BENCHMARK_COMMON_HPP_
