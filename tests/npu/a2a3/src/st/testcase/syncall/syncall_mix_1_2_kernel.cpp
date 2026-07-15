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

extern "C" __global__ AICORE void RunSyncAllMix12_1201(
    __gm__ uint64_t __in__* fftsAddr, __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, int32_t aicBlocks,
    int32_t totalParticipants)
{
    RunMixSyncAllBody<kMix12HardParticipants, false>(fftsAddr, out, flags, nullptr);
}

extern "C" __global__ AICORE void RunSoftSyncAllMix12_1202(
    __gm__ uint64_t __in__* fftsAddr, __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags,
    __gm__ int32_t __out__* syncWorkspace, int32_t aicBlocks, int32_t totalParticipants)
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

void LaunchSyncAllMix12(uint8_t* ffts, int32_t* out, int32_t* flags, int32_t aicBlocks, void* stream)
{
    const int32_t totalParticipants = aicBlocks * 3; // 1 cube + 2 vectors per cube (1:2)
    RunSyncAllMix12_1201<<<aicBlocks, nullptr, stream>>>(
        reinterpret_cast<uint64_t*>(ffts), out, flags, aicBlocks, totalParticipants);
}

void LaunchSoftSyncAllMix12(
    uint8_t* ffts, int32_t* out, int32_t* flags, int32_t* syncWorkspace, int32_t aicBlocks, int32_t totalParticipants,
    void* stream)
{
    RunSoftSyncAllMix12_1202<<<aicBlocks, nullptr, stream>>>(
        reinterpret_cast<uint64_t*>(ffts), out, flags, syncWorkspace, aicBlocks, totalParticipants);
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
