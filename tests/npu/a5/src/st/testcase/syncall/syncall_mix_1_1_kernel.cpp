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

constexpr int32_t kMix11SoftParticipants = 36;
constexpr uint64_t kMix11SoftTilingKey = 1101;

#if defined(SYNCALL_MIX_BUILD_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunSoftSyncAllMix11_1101_mix_aic, 1, 1);

extern "C" __global__ AICORE void RunSoftSyncAllMix11_1101_mix_aic(__gm__ int32_t __out__ *out,
                                                                   __gm__ int32_t __out__ *flags,
                                                                   __gm__ int32_t __out__ *syncWorkspace)
{
    RunMixSyncAllBody<kMix11SoftParticipants>(out, flags, syncWorkspace);
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunSoftSyncAllMix11_1101_mix_aiv, 1, 1);

extern "C" __global__ AICORE void RunSoftSyncAllMix11_1101_mix_aiv(__gm__ int32_t __out__ *out,
                                                                   __gm__ int32_t __out__ *flags,
                                                                   __gm__ int32_t __out__ *syncWorkspace)
{
    RunMixSyncAllBody<kMix11SoftParticipants>(out, flags, syncWorkspace);
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
        std::fprintf(stderr, "dladdr failed for SYNCALL mix 1:1 kernel\n");
        std::abort();
    }
    return info.dli_fname;
#endif
}

std::vector<char> ReadCurrentSharedObject(const char *path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::fprintf(stderr, "failed to open SYNCALL mix 1:1 kernel binary: %s\n", path);
        std::abort();
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "invalid SYNCALL mix 1:1 kernel binary size: %s\n", path);
        std::abort();
    }

    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), size)) {
        std::fprintf(stderr, "failed to read SYNCALL mix 1:1 kernel binary: %s\n", path);
        std::abort();
    }
    return data;
}

void LaunchSoftMixKernel(const void *anchor, uint64_t tilingKey, int32_t *out, int32_t *flags, int32_t *syncWorkspace,
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
            std::fprintf(stderr, "register SYNCALL mix 1:1 kernel failed, path=%s, size=%zu, ret=%d\n", path,
                         kernelBinary.size(), ret);
            std::abort();
        }
    }

    void *args[] = {out, flags, syncWorkspace};
    rtArgsEx_t argsInfo{};
    argsInfo.args = args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo{};
    ret = rtKernelLaunchWithHandleV2(handle, tilingKey, 18, &argsInfo, nullptr, stream, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for SYNCALL soft mix 1:1, ret=%d\n", ret);
        std::abort();
    }
}
} // namespace

void LaunchSoftSyncAllMix11(int32_t *out, int32_t *flags, int32_t *syncWorkspace, void *stream)
{
    LaunchSoftMixKernel(reinterpret_cast<const void *>(&LaunchSoftSyncAllMix11), kMix11SoftTilingKey, out, flags,
                        syncWorkspace, stream);
}
#endif
