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

constexpr int32_t kAicHardBlockCount = 18;
constexpr uint64_t kAicHardTilingKey = 3001;

#if defined(SYNCALL_MIX_BUILD_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunHardSyncAllAIC_3001_mix_aic, 1, 0);

extern "C" __global__ AICORE void RunHardSyncAllAIC_3001_mix_aic(__gm__ int32_t __out__* out)
{
    (void)out;
    SYNCALL<SyncCoreType::AICOnly>();
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(RunHardSyncAllAIC_3001_mix_aiv, 1, 0);

extern "C" __global__ AICORE void RunHardSyncAllAIC_3001_mix_aiv(__gm__ int32_t __out__* out) { (void)out; }
#endif

#if defined(SYNCALL_MIX_BUILD_AIC) && !defined(SYNCALL_MIX_REGISTER_BUILD)
namespace {
const char* GetRegisterElfPath(const void* anchor)
{
#if defined(SYNCALL_MIX_REGISTER_OBJECT_PATH)
    (void)anchor;
    return SYNCALL_MIX_REGISTER_OBJECT_PATH;
#else
    Dl_info info{};
    if (dladdr(anchor, &info) == 0 || info.dli_fname == nullptr) {
        std::fprintf(stderr, "dladdr failed for A5 SYNCALL AIC-only hard kernel\n");
        std::abort();
    }
    return info.dli_fname;
#endif
}

std::vector<char> ReadBinaryFile(const char* path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::fprintf(stderr, "failed to open A5 AIC-only hard register ELF: %s\n", path);
        std::abort();
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "invalid A5 AIC-only hard register ELF size: %s\n", path);
        std::abort();
    }
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), size)) {
        std::fprintf(stderr, "failed to read A5 AIC-only hard register ELF: %s\n", path);
        std::abort();
    }
    return data;
}
} // namespace

void LaunchHardSyncAllAIC(int32_t* out, void* stream)
{
    const char* path = GetRegisterElfPath(reinterpret_cast<const void*>(&LaunchHardSyncAllAIC));
    static const std::vector<char> kernelBin = ReadBinaryFile(path);

    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, kernelBin.data(), kernelBin.size()};
    void* handle = nullptr;
    rtError_t ret = rtRegisterAllKernel(&binary, &handle);
    if (ret != RT_ERROR_NONE || handle == nullptr) {
        ret = rtBinaryLoadWithoutTilingKey(kernelBin.data(), kernelBin.size(), &handle);
        if (ret != RT_ERROR_NONE || handle == nullptr) {
            std::fprintf(
                stderr, "register A5 AIC-only hard kernel failed, path=%s, size=%zu, ret=%d\n", path, kernelBin.size(),
                ret);
            std::abort();
        }
    }

    void* args[] = {out};
    rtArgsEx_t argsInfo{};
    argsInfo.args = args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo{};
    ret =
        rtKernelLaunchWithHandleV2(handle, kAicHardTilingKey, kAicHardBlockCount, &argsInfo, nullptr, stream, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::fprintf(stderr, "rtKernelLaunchWithHandleV2 failed for A5 AIC-only hard, ret=%d\n", ret);
        std::abort();
    }
}
#endif
