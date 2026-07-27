/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>
#include <filesystem>

using namespace PtoTestCommon;

class SYNCALLTest : public testing::Test {
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
    std::filesystem::create_directories(fullPath);
    return fullPath;
}

void LaunchSoftSyncAll(int32_t* out, int32_t* flags, int32_t* syncWorkspace, int32_t totalBlocks, void* stream);
void LaunchSoftSyncAllPartial(
    int32_t* out, int32_t* flags, int32_t* syncWorkspace, int32_t launchBlocks, int32_t syncBlocks, void* stream);
void LaunchSoftSyncAllAIC(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream);
void LaunchSoftSyncAllAICPartial(
    int32_t* out, int32_t* flags, int32_t* syncWorkspace, int32_t launchBlocks, int32_t syncBlocks, void* stream);
void LaunchHardSyncAll(int32_t* out, int32_t* flags, int32_t totalBlocks, void* stream);
void LaunchSoftSyncAllMix11(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream);
void LaunchSoftSyncAllMix12(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream);
void LaunchHardSyncAllMix12(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream);
void LaunchHardSyncAllAIC(int32_t* out, void* stream);

#define EXPECT_ACL_OK(expr)                                             \
    do {                                                                \
        const auto ret = (expr);                                        \
        ASSERT_EQ(ret, ACL_SUCCESS) << #expr << " failed, ret=" << ret; \
    } while (0)

// Partial-participation soft SYNCALL: every core of the launch runs, but only the
// first syncBlocks of them call SYNCALL. Participants must still see every other
// participant's flags on both sides of the barriers (out == 1); the idle cores only
// record that they ran without joining (out == kIdleCoreMark), which distinguishes
// them from a core that never started (out == 0). int32PerCacheLine is the per-core
// slot stride and must match the kernel that launchFn drives.
template <typename LaunchFn>
void RunSoftPartialCase(
    size_t launchBlocks, size_t syncBlocks, size_t int32PerCacheLine, LaunchFn launchFn, const char* label)
{
    constexpr int32_t kIdleCoreMark = 2; // Must match kIdleCoreMark in the kernels.
    constexpr size_t kSyncWsBytes = 64;  // One A5 cache line for the shared atomic counter.
    const size_t elementCount = launchBlocks * int32PerCacheLine;
    const size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t* outHost = nullptr;
    int32_t* flagsHost = nullptr;
    int32_t* outDevice = nullptr;
    int32_t* flagsDevice = nullptr;
    int32_t* syncWorkspaceDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&syncWorkspaceDevice), kSyncWsBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemset(syncWorkspaceDevice, kSyncWsBytes, 0, kSyncWsBytes));

    launchFn(outDevice, flagsDevice, syncWorkspaceDevice, stream);
    const int32_t syncRet = static_cast<int32_t>(aclrtSynchronizeStream(stream));
    EXPECT_EQ(syncRet, ACL_SUCCESS) << label << " barrier faulted";

    if (syncRet == ACL_SUCCESS) {
        EXPECT_ACL_OK(aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        EXPECT_ACL_OK(aclrtMemcpy(flagsHost, byteSize, flagsDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, byteSize));

        std::vector<int32_t> golden(launchBlocks);
        std::vector<int32_t> devFinal(launchBlocks);
        for (size_t i = 0; i < launchBlocks; ++i) {
            golden[i] = (i < syncBlocks) ? 1 : kIdleCoreMark;
            devFinal[i] = outHost[i * int32PerCacheLine];
        }

        const bool ret = ResultCmp<int32_t>(golden, devFinal, 0.0f);
        if (!ret) {
            std::printf("%s launch=%zu sync=%zu out:", label, launchBlocks, syncBlocks);
            for (size_t i = 0; i < launchBlocks; ++i) {
                std::printf(" %d", outHost[i * int32PerCacheLine]);
            }
            std::printf("\n");
        }
        EXPECT_TRUE(ret);
    }

    (void)aclrtFree(outDevice);
    (void)aclrtFree(flagsDevice);
    (void)aclrtFree(syncWorkspaceDevice);
    (void)aclrtFreeHost(outHost);
    (void)aclrtFreeHost(flagsHost);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
}

TEST_F(SYNCALLTest, case_soft_aiv_only_all_blocks)
{
    constexpr int32_t blockCount = 18;
    constexpr size_t int32PerCacheLine = 8;
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t* outHost = nullptr;
    int32_t* flagsHost = nullptr;
    int32_t* outDevice = nullptr;
    int32_t* flagsDevice = nullptr;
    int32_t* syncWorkspaceDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&syncWorkspaceDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(syncWorkspaceDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchSoftSyncAll(outDevice, flagsDevice, syncWorkspaceDevice, blockCount, stream);
    EXPECT_ACL_OK(aclrtSynchronizeStream(stream));
    EXPECT_ACL_OK(aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    EXPECT_ACL_OK(aclrtMemcpy(flagsHost, byteSize, flagsDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, byteSize));

    std::vector<int32_t> golden(blockCount);
    std::vector<int32_t> devFinal(blockCount);
    for (size_t i = 0; i < blockCount; ++i) {
        golden[i] = 1;
        devFinal[i] = outHost[i * int32PerCacheLine];
    }

    EXPECT_TRUE(ResultCmp<int32_t>(golden, devFinal, 0.0f));

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtFree(flagsDevice));
    EXPECT_ACL_OK(aclrtFree(syncWorkspaceDevice));
    EXPECT_ACL_OK(aclrtFreeHost(outHost));
    EXPECT_ACL_OK(aclrtFreeHost(flagsHost));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}

TEST_F(SYNCALLTest, case_soft_aiv_only_partial_blocks)
{
    constexpr size_t launchBlocks = 18;
    constexpr size_t syncBlocks = launchBlocks / 2;
    constexpr size_t int32PerCacheLine = 8; // Must match kInt32PerCacheLine in syncall_soft_kernel.cpp.
    RunSoftPartialCase(
        launchBlocks, syncBlocks, int32PerCacheLine,
        [](int32_t* out, int32_t* flags, int32_t* ws, void* stream) {
            LaunchSoftSyncAllPartial(
                out, flags, ws, static_cast<int32_t>(launchBlocks), static_cast<int32_t>(syncBlocks), stream);
        },
        "soft_aiv_partial");
}

// AIC-only soft SYNCALL: all cube cores publish a flag via scalar GM store, run
// the AIC-only atomic-counter barrier, then scalar-read every flag. out[idx]==1
// proves this cube core saw all peers' round-1 and round-2 writes across barriers.
TEST_F(SYNCALLTest, case_soft_aic_only_all_blocks)
{
    constexpr int32_t blockCount = 18;
    // 16 int32 = one 64-byte A5 cache line per core slot; must match
    // kAicSoftCacheLine in syncall_aic_soft_kernel.cpp (avoids cube dcci false sharing).
    constexpr size_t int32PerCacheLine = 16;
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t* outHost = nullptr;
    int32_t* flagsHost = nullptr;
    int32_t* outDevice = nullptr;
    int32_t* flagsDevice = nullptr;
    int32_t* syncWorkspaceDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&syncWorkspaceDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemset(syncWorkspaceDevice, byteSize, 0, byteSize));

    LaunchSoftSyncAllAIC(outDevice, flagsDevice, syncWorkspaceDevice, stream);
    const int32_t syncRet = static_cast<int32_t>(aclrtSynchronizeStream(stream));
    EXPECT_EQ(syncRet, ACL_SUCCESS) << "AIC-only soft barrier faulted";

    if (syncRet == ACL_SUCCESS) {
        EXPECT_ACL_OK(aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        EXPECT_ACL_OK(aclrtMemcpy(flagsHost, byteSize, flagsDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, byteSize));

        std::vector<int32_t> golden(blockCount);
        std::vector<int32_t> devFinal(blockCount);
        for (size_t i = 0; i < blockCount; ++i) {
            golden[i] = 1;
            devFinal[i] = outHost[i * int32PerCacheLine];
        }
        EXPECT_TRUE(ResultCmp<int32_t>(golden, devFinal, 0.0f));
    }

    (void)aclrtFree(outDevice);
    (void)aclrtFree(flagsDevice);
    (void)aclrtFree(syncWorkspaceDevice);
    (void)aclrtFreeHost(outHost);
    (void)aclrtFreeHost(flagsHost);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
}

TEST_F(SYNCALLTest, case_soft_aic_only_partial_blocks)
{
    constexpr size_t launchBlocks = 18;
    constexpr size_t syncBlocks = launchBlocks / 2;
    constexpr size_t int32PerCacheLine = 16; // Must match kAicSoftCacheLine in syncall_aic_soft_kernel.cpp.
    RunSoftPartialCase(
        launchBlocks, syncBlocks, int32PerCacheLine,
        [](int32_t* out, int32_t* flags, int32_t* ws, void* stream) {
            LaunchSoftSyncAllAICPartial(
                out, flags, ws, static_cast<int32_t>(launchBlocks), static_cast<int32_t>(syncBlocks), stream);
        },
        "soft_aic_partial");
}

TEST_F(SYNCALLTest, case_hard_aiv_only_all_blocks)
{
    constexpr int32_t blockCount = 18;
    constexpr size_t int32PerCacheLine = 8;
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t* outHost = nullptr;
    int32_t* flagsHost = nullptr;
    int32_t* outDevice = nullptr;
    int32_t* flagsDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchHardSyncAll(outDevice, flagsDevice, blockCount, stream);
    EXPECT_ACL_OK(aclrtSynchronizeStream(stream));
    EXPECT_ACL_OK(aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, byteSize));

    std::vector<int32_t> golden(blockCount);
    std::vector<int32_t> devFinal(blockCount);
    for (size_t i = 0; i < blockCount; ++i) {
        golden[i] = 1;
        devFinal[i] = outHost[i * int32PerCacheLine];
    }

    EXPECT_TRUE(ResultCmp<int32_t>(golden, devFinal, 0.0f));

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtFree(flagsDevice));
    EXPECT_ACL_OK(aclrtFreeHost(outHost));
    EXPECT_ACL_OK(aclrtFreeHost(flagsHost));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}

TEST_F(SYNCALLTest, case_soft_mix_1_2_all_blocks)
{
    constexpr int32_t blockCount = 54;
    constexpr size_t int32PerCacheLine = 16; // Must match kInt32PerCacheLine in syncall_mix_common.hpp.
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);
    // The soft barrier is a single shared atomic counter; give it its own cache line.
    constexpr size_t syncWsBytes = int32PerCacheLine * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t* outHost = nullptr;
    int32_t* flagsHost = nullptr;
    int32_t* outDevice = nullptr;
    int32_t* flagsDevice = nullptr;
    int32_t* syncWorkspaceDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&syncWorkspaceDevice), syncWsBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemset(syncWorkspaceDevice, syncWsBytes, 0, syncWsBytes));

    LaunchSoftSyncAllMix12(outDevice, flagsDevice, syncWorkspaceDevice, stream);
    const int32_t syncRet = static_cast<int32_t>(aclrtSynchronizeStream(stream));
    EXPECT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed";

    if (syncRet == ACL_SUCCESS) {
        EXPECT_ACL_OK(aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        EXPECT_ACL_OK(aclrtMemcpy(flagsHost, byteSize, flagsDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, byteSize));

        std::vector<int32_t> golden(blockCount);
        std::vector<int32_t> devFinal(blockCount);
        for (size_t i = 0; i < blockCount; ++i) {
            golden[i] = 1;
            devFinal[i] = outHost[i * int32PerCacheLine];
        }
        EXPECT_TRUE(ResultCmp<int32_t>(golden, devFinal, 0.0f));
    }

    (void)aclrtFree(outDevice);
    (void)aclrtFree(flagsDevice);
    (void)aclrtFree(syncWorkspaceDevice);
    (void)aclrtFreeHost(outHost);
    (void)aclrtFreeHost(flagsHost);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
}

TEST_F(SYNCALLTest, case_soft_mix_1_1_all_blocks)
{
    constexpr int32_t blockCount = 36;       // 18 cube + 18 vector (1:1), dual-stream chevron.
    constexpr size_t int32PerCacheLine = 16; // Must match kInt32PerCacheLine in syncall_mix_common.hpp.
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);
    // The soft barrier is a single shared atomic counter; give it its own cache line.
    constexpr size_t syncWsBytes = int32PerCacheLine * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t* outHost = nullptr;
    int32_t* flagsHost = nullptr;
    int32_t* outDevice = nullptr;
    int32_t* flagsDevice = nullptr;
    int32_t* syncWorkspaceDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&syncWorkspaceDevice), syncWsBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemset(syncWorkspaceDevice, syncWsBytes, 0, syncWsBytes));

    LaunchSoftSyncAllMix11(outDevice, flagsDevice, syncWorkspaceDevice, stream);
    EXPECT_ACL_OK(aclrtSynchronizeStream(stream));
    EXPECT_ACL_OK(aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    EXPECT_ACL_OK(aclrtMemcpy(flagsHost, byteSize, flagsDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, byteSize));

    std::vector<int32_t> golden(blockCount);
    std::vector<int32_t> devFinal(blockCount);
    for (size_t i = 0; i < blockCount; ++i) {
        golden[i] = 1;
        devFinal[i] = outHost[i * int32PerCacheLine];
    }

    EXPECT_TRUE(ResultCmp<int32_t>(golden, devFinal, 0.0f));

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtFree(flagsDevice));
    EXPECT_ACL_OK(aclrtFree(syncWorkspaceDevice));
    EXPECT_ACL_OK(aclrtFreeHost(outHost));
    EXPECT_ACL_OK(aclrtFreeHost(flagsHost));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}

TEST_F(SYNCALLTest, case_hard_mix_1_2_all_blocks)
{
    constexpr int32_t blockCount = 54;       // 18 cube + 36 vector (1:2), auto-split chevron.
    constexpr size_t int32PerCacheLine = 16; // Must match kInt32PerCacheLine in syncall_mix_common.hpp.
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t* outHost = nullptr;
    int32_t* flagsHost = nullptr;
    int32_t* outDevice = nullptr;
    int32_t* flagsDevice = nullptr;
    int32_t* syncWorkspaceDevice = nullptr; // unused by hard barrier, kept for the shared body signature.

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&syncWorkspaceDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchHardSyncAllMix12(outDevice, flagsDevice, syncWorkspaceDevice, stream);
    const int32_t syncRet = static_cast<int32_t>(aclrtSynchronizeStream(stream));
    EXPECT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed";

    if (syncRet == ACL_SUCCESS) {
        EXPECT_ACL_OK(aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        EXPECT_ACL_OK(aclrtMemcpy(flagsHost, byteSize, flagsDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, byteSize));

        std::vector<int32_t> golden(blockCount);
        std::vector<int32_t> devFinal(blockCount);
        for (size_t i = 0; i < blockCount; ++i) {
            golden[i] = 1;
            devFinal[i] = outHost[i * int32PerCacheLine];
        }
        EXPECT_TRUE(ResultCmp<int32_t>(golden, devFinal, 0.0f));
    }

    (void)aclrtFree(outDevice);
    (void)aclrtFree(flagsDevice);
    (void)aclrtFree(syncWorkspaceDevice);
    (void)aclrtFreeHost(outHost);
    (void)aclrtFreeHost(flagsHost);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
}

TEST_F(SYNCALLTest, case_hard_aic_only_all_blocks)
{
    constexpr int32_t blockCount = 18;
    constexpr size_t byteSize = blockCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t* outDevice = nullptr;
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    LaunchHardSyncAllAIC(outDevice, stream);
    EXPECT_ACL_OK(aclrtSynchronizeStream(stream));

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}
