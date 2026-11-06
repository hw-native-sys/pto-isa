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

void LaunchHardSyncAll(int32_t* out, int32_t* flags, void* stream);
void LaunchHardSyncAllAIC(int32_t* out, int32_t* flags, void* stream);
void LaunchHardSyncAllMix(int32_t* out, int32_t* flags, void* stream);
void LaunchSoftSyncAll(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream);
void LaunchSoftSyncAllPartial(
    int32_t* out, int32_t* flags, int32_t* syncWorkspace, int32_t launchBlocks, int32_t syncBlocks, void* stream);
void LaunchSoftSyncAllAIC(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream);
void LaunchSoftSyncAllAICPartial(
    int32_t* out, int32_t* flags, int32_t* syncWorkspace, int32_t launchBlocks, int32_t syncBlocks, void* stream);
void LaunchSoftSyncAllMix(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream);

#define EXPECT_ACL_OK(expr)                                             \
    do {                                                                \
        const auto ret = (expr);                                        \
        ASSERT_EQ(ret, ACL_SUCCESS) << #expr << " failed, ret=" << ret; \
    } while (0)

// Must match kInt32PerCacheLine in syncall_hard_common.hpp.
constexpr size_t kInt32PerCacheLine = 16;

// Drives one hard SYNCALL kernel and checks that every participant reported having
// seen all peers on both sides of the barriers (out slot == 1). A participant that
// never ran leaves its slot at 0, which distinguishes a launch problem from a
// visibility problem.
template <typename LaunchFn>
void RunHardSyncAllCase(size_t participants, LaunchFn launchFn, const char* label)
{
    const size_t elementCount = participants * kInt32PerCacheLine;
    const size_t byteSize = elementCount * sizeof(int32_t);

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

    launchFn(outDevice, flagsDevice, stream);
    const int32_t syncRet = static_cast<int32_t>(aclrtSynchronizeStream(stream));
    EXPECT_EQ(syncRet, ACL_SUCCESS) << label << " barrier faulted";

    if (syncRet == ACL_SUCCESS) {
        EXPECT_ACL_OK(aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        EXPECT_ACL_OK(aclrtMemcpy(flagsHost, byteSize, flagsDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, byteSize));

        std::vector<int32_t> golden(participants, 1);
        std::vector<int32_t> devFinal(participants);
        for (size_t i = 0; i < participants; ++i) {
            devFinal[i] = outHost[i * kInt32PerCacheLine];
        }

        const bool ret = ResultCmp<int32_t>(golden, devFinal, 0.0f);
        if (!ret) {
            std::printf("%s participants=%zu out:", label, participants);
            for (size_t i = 0; i < participants; ++i) {
                std::printf(" %d", outHost[i * kInt32PerCacheLine]);
            }
            std::printf("\n");
        }
        EXPECT_TRUE(ret);
    }

    (void)aclrtFree(outDevice);
    (void)aclrtFree(flagsDevice);
    (void)aclrtFreeHost(outHost);
    (void)aclrtFreeHost(flagsHost);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
}

TEST_F(SYNCALLTest, case_hard_aiv_only_all_blocks)
{
    constexpr size_t participants = 18; // Must match kAivHardBlockCount in syncall_hard_kernel.cpp.
    RunHardSyncAllCase(participants, LaunchHardSyncAll, "hard_aiv_only");
}

TEST_F(SYNCALLTest, case_hard_aic_only_all_blocks)
{
    constexpr size_t participants = 18; // Must match kAicHardBlockCount in syncall_aic_hard_kernel.cpp.
    RunHardSyncAllCase(participants, LaunchHardSyncAllAIC, "hard_aic_only");
}

// A6 pairs one AIC with one AIV, so an 18-block chevron yields 18 cube + 18 vector.
// A hang here (aclrtSynchronizeStream never returning) is the 1:2 regression; a
// returned stream with non-1 slots means a barrier let a core through too early.
TEST_F(SYNCALLTest, case_hard_mix_1_1_all_blocks)
{
    constexpr size_t participants = 36; // Must match kMixParticipants in syncall_hard_mix_kernel.cpp.
    RunHardSyncAllCase(participants, LaunchHardSyncAllMix, "hard_mix_1_1");
}

// One cache line for the shared atomic counter. Must stay 64 bytes: dcci writes
// the whole line back, and the counter must not share that line with the flags.
constexpr size_t kSoftSyncWsBytes = 64;
// Must match kIdleCoreMark in syncall_soft_common.hpp.
constexpr int32_t kIdleCoreMark = 2;

// Same visibility check as the hard cases, plus the GM workspace the soft barrier
// uses as its shared counter. Every participant must report out slot == 1.
template <typename LaunchFn>
void RunSoftSyncAllCase(size_t participants, LaunchFn launchFn, const char* label)
{
    const size_t elementCount = participants * kInt32PerCacheLine;
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
    EXPECT_ACL_OK(
        aclrtMalloc(reinterpret_cast<void**>(&syncWorkspaceDevice), kSoftSyncWsBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemset(syncWorkspaceDevice, kSoftSyncWsBytes, 0, kSoftSyncWsBytes));

    launchFn(outDevice, flagsDevice, syncWorkspaceDevice, stream);
    const int32_t syncRet = static_cast<int32_t>(aclrtSynchronizeStream(stream));
    EXPECT_EQ(syncRet, ACL_SUCCESS) << label << " barrier faulted";

    if (syncRet == ACL_SUCCESS) {
        EXPECT_ACL_OK(aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        EXPECT_ACL_OK(aclrtMemcpy(flagsHost, byteSize, flagsDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, byteSize));

        std::vector<int32_t> golden(participants, 1);
        std::vector<int32_t> devFinal(participants);
        for (size_t i = 0; i < participants; ++i) {
            devFinal[i] = outHost[i * kInt32PerCacheLine];
        }

        const bool ret = ResultCmp<int32_t>(golden, devFinal, 0.0f);
        if (!ret) {
            std::printf("%s participants=%zu out:", label, participants);
            for (size_t i = 0; i < participants; ++i) {
                std::printf(" %d", outHost[i * kInt32PerCacheLine]);
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

// Every launched core runs, but only the first syncBlocks join SYNCALL. Participants
// report 1; idle cores report kIdleCoreMark. A slot left at 0 means that core never ran.
template <typename LaunchFn>
void RunSoftPartialCase(size_t launchBlocks, size_t syncBlocks, LaunchFn launchFn, const char* label)
{
    const size_t elementCount = launchBlocks * kInt32PerCacheLine;
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
    EXPECT_ACL_OK(
        aclrtMalloc(reinterpret_cast<void**>(&syncWorkspaceDevice), kSoftSyncWsBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemset(syncWorkspaceDevice, kSoftSyncWsBytes, 0, kSoftSyncWsBytes));

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
            devFinal[i] = outHost[i * kInt32PerCacheLine];
        }

        const bool ret = ResultCmp<int32_t>(golden, devFinal, 0.0f);
        if (!ret) {
            std::printf("%s launch=%zu sync=%zu out:", label, launchBlocks, syncBlocks);
            for (size_t i = 0; i < launchBlocks; ++i) {
                std::printf(" %d", outHost[i * kInt32PerCacheLine]);
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
    constexpr size_t participants = 18; // Must match kAivSoftBlockCount in syncall_soft_kernel.cpp.
    RunSoftSyncAllCase(participants, LaunchSoftSyncAll, "soft_aiv_only");
}

TEST_F(SYNCALLTest, case_soft_aiv_only_partial_blocks)
{
    constexpr size_t launchBlocks = 18;
    constexpr size_t syncBlocks = launchBlocks / 2;
    RunSoftPartialCase(
        launchBlocks, syncBlocks,
        [](int32_t* out, int32_t* flags, int32_t* ws, void* stream) {
            LaunchSoftSyncAllPartial(
                out, flags, ws, static_cast<int32_t>(launchBlocks), static_cast<int32_t>(syncBlocks), stream);
        },
        "soft_aiv_partial");
}

TEST_F(SYNCALLTest, case_soft_aic_only_all_blocks)
{
    constexpr size_t participants = 18; // Must match kAicSoftBlockCount in syncall_aic_soft_kernel.cpp.
    RunSoftSyncAllCase(participants, LaunchSoftSyncAllAIC, "soft_aic_only");
}

TEST_F(SYNCALLTest, case_soft_aic_only_partial_blocks)
{
    constexpr size_t launchBlocks = 18;
    constexpr size_t syncBlocks = launchBlocks / 2;
    RunSoftPartialCase(
        launchBlocks, syncBlocks,
        [](int32_t* out, int32_t* flags, int32_t* ws, void* stream) {
            LaunchSoftSyncAllAICPartial(
                out, flags, ws, static_cast<int32_t>(launchBlocks), static_cast<int32_t>(syncBlocks), stream);
        },
        "soft_aic_partial");
}

// A6 auto-split is 1:1, so this is the software counterpart of case_hard_mix_1_1.
// A5's 1:2 mix cases do not apply here.
TEST_F(SYNCALLTest, case_soft_mix_1_1_all_blocks)
{
    constexpr size_t participants = 36; // Must match kSoftMixParticipants in syncall_soft_mix_kernel.cpp.
    RunSoftSyncAllCase(participants, LaunchSoftSyncAllMix, "soft_mix_1_1");
}
