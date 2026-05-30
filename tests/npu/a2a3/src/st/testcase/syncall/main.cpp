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
#include "runtime/rt.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>
#include <filesystem>

using namespace PtoTestCommon;

class SYNCALLTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    std::filesystem::create_directories(fullPath);
    return fullPath;
}

void LaunchSyncAll(uint8_t *ffts, int32_t *out, int32_t *flags, void *stream);
void LaunchSoftSyncAll(int32_t *out, int32_t *flags, int32_t *syncWorkspace, void *stream);
void LaunchHardSyncAllAIC(uint8_t *ffts, int32_t *out, int32_t *flags, void *stream);
void LaunchSoftSyncAllAIC(int32_t *out, int32_t *flags, int32_t *syncWorkspace, void *stream);
void LaunchSyncAllMix11(uint8_t *ffts, int32_t *out, int32_t *flags, void *stream);
void LaunchSyncAllMix12(uint8_t *ffts, int32_t *out, int32_t *flags, void *stream);
void LaunchSoftSyncAllMix11(uint8_t *ffts, int32_t *out, int32_t *flags, int32_t *syncWorkspace, void *stream);
void LaunchSoftSyncAllMix12(uint8_t *ffts, int32_t *out, int32_t *flags, int32_t *syncWorkspace, void *stream);

#define EXPECT_ACL_OK(expr)                                             \
    do {                                                                \
        const auto ret = (expr);                                        \
        ASSERT_EQ(ret, ACL_SUCCESS) << #expr << " failed, ret=" << ret; \
    } while (0)

#define EXPECT_RT_OK(expr)                                    \
    do {                                                      \
        const auto ret = (expr);                              \
        ASSERT_EQ(ret, 0) << #expr << " failed, ret=" << ret; \
    } while (0)

template <size_t blockCount, bool withWorkspace, typename LaunchFn>
void RunMixCase(LaunchFn launchFn, const char *label)
{
    constexpr size_t int32PerCacheLine = 8;
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t *outHost = nullptr;
    int32_t *flagsHost = nullptr;
    int32_t *outDevice = nullptr;
    int32_t *flagsDevice = nullptr;
    int32_t *syncWorkspaceDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if constexpr (withWorkspace) {
        EXPECT_ACL_OK(
            aclrtMalloc(reinterpret_cast<void **>(&syncWorkspaceDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    if constexpr (withWorkspace) {
        EXPECT_ACL_OK(aclrtMemcpy(syncWorkspaceDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    EXPECT_RT_OK(rtGetC2cCtrlAddr(&ffts, &fftsLen));
    ASSERT_NE(ffts, 0UL);

    if constexpr (withWorkspace) {
        launchFn(reinterpret_cast<uint8_t *>(ffts), outDevice, flagsDevice, syncWorkspaceDevice, stream);
    } else {
        launchFn(reinterpret_cast<uint8_t *>(ffts), outDevice, flagsDevice, stream);
    }
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

    bool ret = ResultCmp<int32_t>(golden, devFinal, 0.0f);
    if (!ret) {
        std::printf("%s out[0..7]:", label);
        for (size_t i = 0; i < std::min<size_t>(8, blockCount); ++i) {
            std::printf(" %d", outHost[i * int32PerCacheLine]);
        }
        std::printf("\n%s flags[0..7]:", label);
        for (size_t i = 0; i < std::min<size_t>(8, blockCount); ++i) {
            std::printf(" %d", flagsHost[i * int32PerCacheLine]);
        }
        std::printf("\n");
    }
    EXPECT_TRUE(ret);

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtFree(flagsDevice));
    if constexpr (withWorkspace) {
        EXPECT_ACL_OK(aclrtFree(syncWorkspaceDevice));
    }
    EXPECT_ACL_OK(aclrtFreeHost(outHost));
    EXPECT_ACL_OK(aclrtFreeHost(flagsHost));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}

TEST_F(SYNCALLTest, case_aiv_only_all_blocks)
{
    constexpr size_t blockCount = 48;
    constexpr size_t int32PerCacheLine = 8;
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t *outHost = nullptr;
    int32_t *flagsHost = nullptr;
    int32_t *outDevice = nullptr;
    int32_t *flagsDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    EXPECT_RT_OK(rtGetC2cCtrlAddr(&ffts, &fftsLen));
    ASSERT_NE(ffts, 0UL);

    LaunchSyncAll(reinterpret_cast<uint8_t *>(ffts), outDevice, flagsDevice, stream);
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

    bool ret = ResultCmp<int32_t>(golden, devFinal, 0.0f);
    if (!ret) {
        std::printf("out[0..7]:");
        for (size_t i = 0; i < 8; ++i) {
            std::printf(" %d", outHost[i * int32PerCacheLine]);
        }
        std::printf("\nflags[0..7]:");
        for (size_t i = 0; i < 8; ++i) {
            std::printf(" %d", flagsHost[i * int32PerCacheLine]);
        }
        std::printf("\n");
    }
    EXPECT_TRUE(ret);

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtFree(flagsDevice));
    EXPECT_ACL_OK(aclrtFreeHost(outHost));
    EXPECT_ACL_OK(aclrtFreeHost(flagsHost));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}

TEST_F(SYNCALLTest, case_soft_aiv_only_all_blocks)
{
    constexpr size_t blockCount = 48;
    constexpr size_t int32PerCacheLine = 8;
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t *outHost = nullptr;
    int32_t *flagsHost = nullptr;
    int32_t *outDevice = nullptr;
    int32_t *flagsDevice = nullptr;
    int32_t *syncWorkspaceDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&syncWorkspaceDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(syncWorkspaceDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchSoftSyncAll(outDevice, flagsDevice, syncWorkspaceDevice, stream);
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

    bool ret = ResultCmp<int32_t>(golden, devFinal, 0.0f);
    if (!ret) {
        std::printf("soft out[0..7]:");
        for (size_t i = 0; i < 8; ++i) {
            std::printf(" %d", outHost[i * int32PerCacheLine]);
        }
        std::printf("\nsoft flags[0..7]:");
        for (size_t i = 0; i < 8; ++i) {
            std::printf(" %d", flagsHost[i * int32PerCacheLine]);
        }
        std::printf("\n");
    }
    EXPECT_TRUE(ret);

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtFree(flagsDevice));
    EXPECT_ACL_OK(aclrtFree(syncWorkspaceDevice));
    EXPECT_ACL_OK(aclrtFreeHost(outHost));
    EXPECT_ACL_OK(aclrtFreeHost(flagsHost));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}

TEST_F(SYNCALLTest, case_mix_1_1_all_blocks)
{
    RunMixCase<48, false>(LaunchSyncAllMix11, "mix_1_1");
}

TEST_F(SYNCALLTest, case_mix_1_2_all_blocks)
{
    RunMixCase<72, false>(LaunchSyncAllMix12, "mix_1_2");
}

TEST_F(SYNCALLTest, case_soft_mix_1_1_all_blocks)
{
    RunMixCase<48, true>(LaunchSoftSyncAllMix11, "soft_mix_1_1");
}

TEST_F(SYNCALLTest, case_soft_mix_1_2_all_blocks)
{
    RunMixCase<72, true>(LaunchSoftSyncAllMix12, "soft_mix_1_2");
}

TEST_F(SYNCALLTest, case_hard_aic_only_all_blocks)
{
    constexpr size_t blockCount = 24;
    constexpr size_t int32PerCacheLine = 8;
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t *outHost = nullptr;
    int32_t *flagsHost = nullptr;
    int32_t *outDevice = nullptr;
    int32_t *flagsDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    EXPECT_RT_OK(rtGetC2cCtrlAddr(&ffts, &fftsLen));
    ASSERT_NE(ffts, 0UL);

    LaunchHardSyncAllAIC(reinterpret_cast<uint8_t *>(ffts), outDevice, flagsDevice, stream);
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

    bool ret = ResultCmp<int32_t>(golden, devFinal, 0.0f);
    if (!ret) {
        std::printf("hard_aic out[0..7]:");
        for (size_t i = 0; i < 8; ++i) {
            std::printf(" %d", outHost[i * int32PerCacheLine]);
        }
        std::printf("\nhard_aic flags[0..7]:");
        for (size_t i = 0; i < 8; ++i) {
            std::printf(" %d", flagsHost[i * int32PerCacheLine]);
        }
        std::printf("\n");
    }
    EXPECT_TRUE(ret);

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtFree(flagsDevice));
    EXPECT_ACL_OK(aclrtFreeHost(outHost));
    EXPECT_ACL_OK(aclrtFreeHost(flagsHost));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}

TEST_F(SYNCALLTest, case_soft_aic_only_all_blocks)
{
    constexpr size_t blockCount = 24;
    constexpr size_t int32PerCacheLine = 8;
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t *outHost = nullptr;
    int32_t *flagsHost = nullptr;
    int32_t *outDevice = nullptr;
    int32_t *flagsDevice = nullptr;
    int32_t *syncWorkspaceDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&syncWorkspaceDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(syncWorkspaceDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchSoftSyncAllAIC(outDevice, flagsDevice, syncWorkspaceDevice, stream);
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

    bool ret = ResultCmp<int32_t>(golden, devFinal, 0.0f);
    if (!ret) {
        std::printf("soft_aic out[0..7]:");
        for (size_t i = 0; i < 8; ++i) {
            std::printf(" %d", outHost[i * int32PerCacheLine]);
        }
        std::printf("\nsoft_aic flags[0..7]:");
        for (size_t i = 0; i < 8; ++i) {
            std::printf(" %d", flagsHost[i * int32PerCacheLine]);
        }
        std::printf("\n");
    }
    EXPECT_TRUE(ret);

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtFree(flagsDevice));
    EXPECT_ACL_OK(aclrtFree(syncWorkspaceDevice));
    EXPECT_ACL_OK(aclrtFreeHost(outHost));
    EXPECT_ACL_OK(aclrtFreeHost(flagsHost));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}
