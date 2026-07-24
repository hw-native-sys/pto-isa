/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
#include "runtime/rt.h"

#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <fstream>
#include <vector>
#endif

using namespace pto;

constexpr int32_t kAicHardBlockCount = 24;
constexpr int32_t kAicHardInt32PerLine = 8;
constexpr uint64_t kAicHardFlagL1 = 0x0;
constexpr uint64_t kAicHardOutL1 = 0x1000;
constexpr uint64_t kAicHardTilingKey = 2001;

#if defined(__DAV_CUBE__)
PTO_INTERNAL void StoreInt32LineL1(__gm__ int32_t* dst, int32_t value, uint64_t l1Addr)
{
    __cbuf__ int32_t* l1 = reinterpret_cast<__cbuf__ int32_t*>(l1Addr);
    constexpr int64_t repeatConfig = (static_cast<int64_t>(1) << 16) | 1;
    create_cbuf_matrix(l1, repeatConfig, static_cast<uint32_t>(value));
    pipe_barrier(PIPE_ALL);
    copy_cbuf_to_gm(static_cast<__gm__ void*>(dst), static_cast<__cbuf__ void*>(l1), 0, 1, 1, 0, 0);
    pipe_barrier(PIPE_ALL);
    dcci(static_cast<__gm__ void*>(dst), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
}

PTO_INTERNAL void InvalidateGmLines(__gm__ int32_t* addr, int32_t lines)
{
    for (int32_t i = 0; i < lines; ++i) {
        __asm__ __volatile__("");
        dcci(static_cast<__gm__ void*>(addr + i * kInt32PerCacheLine), SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
    dsb(DSB_DDR);
}

extern "C" __global__ AICORE void RunHardSyncAllAIC_2001(
    __gm__ uint64_t __in__* fftsAddr, __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, int32_t totalBlocks)
{
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
    const int32_t idx = block_idx;

    AicStoreL1(flags + idx * kAicHardInt32PerLine, idx + 1, kAicHardFlagL1);
    SYNCALL<SyncCoreType::AICOnly>();

    AicInvalidateLines(flags, kAicHardBlockCount);
    int32_t allVisible = 1;
    for (int32_t i = 0; i < totalBlocks; ++i) {
        __gm__ int32_t* flag = flags + i * kInt32PerCacheLine;
        dcci(static_cast<__gm__ void*>(flag), SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        if (flag[0] != i + 1) {
            allVisible = 0;
        }
    }
    AicStoreL1(out + idx * kAicHardInt32PerLine, allVisible, kAicHardOutL1);
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunHardSyncAllAIC_2001_mix_aiv, 1, 0);

extern "C" __global__ AICORE void RunHardSyncAllAIC_2001_mix_aiv(__gm__ uint64_t __in__ *fftsAddr,
                                                                 __gm__ int32_t __out__ *out,
                                                                 __gm__ int32_t __out__ *flags)
{
    (void)fftsAddr;
    (void)out;
    (void)flags;
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
namespace {
const char *GetRegisterElfPath(const void *anchor)
{
#if defined(SYNCALL_MIX_REGISTER_OBJECT_PATH)
    (void)anchor;
    return SYNCALL_MIX_REGISTER_OBJECT_PATH;
#else
    Dl_info info{};
    if (dladdr(anchor, &info) == 0 || info.dli_fname == nullptr) {
        std::fprintf(stderr, "dladdr failed for SYNCALL AIC-only hard kernel\n");
        std::abort();
    }
    return info.dli_fname;
#endif
}

void LaunchHardSyncAllAIC(uint8_t* ffts, int32_t* out, int32_t* flags, int32_t launchBlocks, void* stream)
{
    RunHardSyncAllAIC_2001<<<launchBlocks, nullptr, stream>>>(
        reinterpret_cast<uint64_t*>(ffts), out, flags, launchBlocks);
}
} // namespace

void LaunchHardSyncAllAIC(uint8_t *ffts, int32_t *out, int32_t *flags, void *stream)
{
    const char *path = GetRegisterElfPath(reinterpret_cast<const void *>(&LaunchHardSyncAllAIC));
    static const std::vector<char> kernelBin = ReadBinaryFile(path);

    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, kernelBin.data(), kernelBin.size()};
    void *handle = nullptr;
    rtError_t ret = rtRegisterAllKernel(&binary, &handle);
    if (ret != RT_ERROR_NONE || handle == nullptr) {
        ret = rtBinaryLoadWithoutTilingKey(kernelBin.data(), kernelBin.size(), &handle);
        if (ret != RT_ERROR_NONE || handle == nullptr) {
            std::fprintf(stderr, "register AIC-only hard kernel failed, path=%s, size=%zu, ret=%d\n", path,
                         kernelBin.size(), ret);
            std::abort();
        }
    }

    void *args[] = {reinterpret_cast<uint64_t *>(ffts), out, flags};
    rtArgsEx_t argsInfo{};
    argsInfo.args = args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo{};
    ret = rtKernelLaunchWithHandleV2(handle, kAicHardTilingKey, 24, &argsInfo, nullptr, stream, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for AIC-only hard, ret=%d\n", ret);
        std::abort();
    }
}
#endif
