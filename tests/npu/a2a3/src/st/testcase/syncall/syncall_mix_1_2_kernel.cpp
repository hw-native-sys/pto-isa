/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "syncall_mix_common.hpp"

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
#include "runtime/rt.h"

#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <fstream>
#include <vector>
#endif

constexpr int32_t kMix12HardParticipants = 72;
constexpr int32_t kMix12SoftParticipants = 72;
constexpr uint64_t kMix12HardTilingKey = 1201;

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
extern "C" __global__ AICORE void RunSoftSyncAllMix12_1202_mix_aiv(__gm__ uint64_t __in__ *fftsAddr,
                                                                   __gm__ int32_t __out__ *out,
                                                                   __gm__ int32_t __out__ *flags,
                                                                   __gm__ int32_t __out__ *syncWorkspace);
#endif

#if defined(SYNCALL_MIX_BUILD_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunSyncAllMix12_1201_mix_aic, 1, 2);
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunSoftSyncAllMix12_1202_mix_aic, 1, 2);

extern "C" __global__ AICORE void RunSyncAllMix12_1201_mix_aic(__gm__ uint64_t __in__ *fftsAddr,
                                                               __gm__ int32_t __out__ *out,
                                                               __gm__ int32_t __out__ *flags)
{
    RunMixSyncAllBody<kMix12HardParticipants, false>(fftsAddr, out, flags, nullptr);
}

extern "C" __global__ AICORE void RunSoftSyncAllMix12_1202_mix_aic(__gm__ uint64_t __in__ *fftsAddr,
                                                                   __gm__ int32_t __out__ *out,
                                                                   __gm__ int32_t __out__ *flags,
                                                                   __gm__ int32_t __out__ *syncWorkspace)
{
    RunMixSyncAllBody<kMix12SoftParticipants, true>(fftsAddr, out, flags, syncWorkspace);
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunSyncAllMix12_1201_mix_aiv, 1, 2);
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunSoftSyncAllMix12_1202_mix_aiv, 1, 2);

extern "C" __global__ AICORE void RunSyncAllMix12_1201_mix_aiv(__gm__ uint64_t __in__ *fftsAddr,
                                                               __gm__ int32_t __out__ *out,
                                                               __gm__ int32_t __out__ *flags)
{
    RunMixSyncAllBody<kMix12HardParticipants, false>(fftsAddr, out, flags, nullptr);
}

extern "C" __global__ AICORE void RunSoftSyncAllMix12_1202_mix_aiv(__gm__ uint64_t __in__ *fftsAddr,
                                                                   __gm__ int32_t __out__ *out,
                                                                   __gm__ int32_t __out__ *flags,
                                                                   __gm__ int32_t __out__ *syncWorkspace)
{
    RunMixSyncAllBody<kMix12SoftParticipants, true>(fftsAddr, out, flags, syncWorkspace);
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
namespace {
const char *GetCurrentSharedObjectPath(const void *anchor)
{
#if defined(SYNCALL_MIX_REGISTER_OBJECT_PATH)
    (void)anchor;
    return SYNCALL_MIX_REGISTER_OBJECT_PATH;
#else
    Dl_info info{};
    if (dladdr(anchor, &info) == 0 || info.dli_fname == nullptr) {
        std::fprintf(stderr, "dladdr failed for SYNCALL mix 1:2 kernel\n");
        std::abort();
    }
    return info.dli_fname;
#endif
}

std::vector<char> ReadCurrentSharedObject(const char *path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::fprintf(stderr, "failed to open SYNCALL mix 1:2 kernel binary: %s\n", path);
        std::abort();
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "invalid SYNCALL mix 1:2 kernel binary size: %s\n", path);
        std::abort();
    }

    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), size)) {
        std::fprintf(stderr, "failed to read SYNCALL mix 1:2 kernel binary: %s\n", path);
        std::abort();
    }
    return data;
}

void LaunchHardMixKernel(const void *anchor, uint64_t tilingKey, uint8_t *ffts, int32_t *out, int32_t *flags,
                         void *stream)
{
    const char *path = GetCurrentSharedObjectPath(anchor);
    static const std::vector<char> kernelBinary = ReadCurrentSharedObject(path);
    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, kernelBinary.data(), kernelBinary.size()};
    void *handle = nullptr;
    rtError_t ret = rtRegisterAllKernel(&binary, &handle);
    if (ret != RT_ERROR_NONE || handle == nullptr) {
        ret = rtBinaryLoadWithoutTilingKey(kernelBinary.data(), kernelBinary.size(), &handle);
        if (ret != RT_ERROR_NONE || handle == nullptr) {
            std::fprintf(stderr, "register SYNCALL mix 1:2 kernel failed, path=%s, size=%zu, ret=%d\n", path,
                         kernelBinary.size(), ret);
            std::abort();
        }
    }

    void *args[] = {reinterpret_cast<uint64_t *>(ffts), out, flags};
    rtArgsEx_t argsInfo{};
    argsInfo.args = args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo{};
    ret = rtKernelLaunchWithHandleV2(handle, tilingKey, 24, &argsInfo, nullptr, stream, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for SYNCALL mix 1:2, ret=%d\n", ret);
        std::abort();
    }
}
} // namespace

void LaunchSyncAllMix12(uint8_t *ffts, int32_t *out, int32_t *flags, void *stream)
{
    LaunchHardMixKernel(reinterpret_cast<const void *>(&LaunchSyncAllMix12), kMix12HardTilingKey, ffts, out, flags,
                        stream);
}

void LaunchSoftSyncAllMix12(uint8_t *ffts, int32_t *out, int32_t *flags, int32_t *syncWorkspace, void *stream)
{
    aclrtStream aivStream;
    (void)aclrtCreateStream(&aivStream);
    RunSoftSyncAllMix12_1202_mix_aic<<<24, nullptr, stream>>>(reinterpret_cast<uint64_t *>(ffts), out, flags,
                                                              syncWorkspace);
    RunSoftSyncAllMix12_1202_mix_aiv<<<48, nullptr, aivStream>>>(reinterpret_cast<uint64_t *>(ffts), out, flags,
                                                                 syncWorkspace);
    (void)aclrtSynchronizeStream(aivStream);
    (void)aclrtDestroyStream(aivStream);
}
#endif
