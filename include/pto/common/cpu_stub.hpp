/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#ifndef PTO_CPUSTUB_HPP
#define PTO_CPUSTUB_HPP

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <exception>
#include <chrono>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <type_traits>
#include <cstdio>
#include <dlfcn.h>
#include <string>
#include "type.hpp"

#define __global__
#define AICORE
#define __aicore__
#define __gm__
#define __out__
#define __in__
#define __ubuf__
#define __cbuf__
#define __ca__
#define __cb__
#define __cc__
#define __fbuf__
#define __biasbuf__
#define __tf__

typedef void *aclrtStream;
inline uint32_t get_block_idx();
typedef int pipe_t;
using event_t = int;
const pipe_t PIPE_S = 0;
const pipe_t PIPE_V = 1;
const pipe_t PIPE_MTE1 = 2;
const pipe_t PIPE_MTE2 = 3;
const pipe_t PIPE_MTE3 = 4;
const pipe_t PIPE_M = 5;
const pipe_t PIPE_ALL = 6;
const pipe_t PIPE_FIX = 7;
inline void pipe_barrier(pipe_t pipe)
{
    (void)pipe;
}

// CCEC provides cache maintenance intrinsics and barrier constants for real
// AICore builds. Host CPU sim kernels still see generated dcci/dsb calls, so
// model them as full fences to preserve cross-thread ordering.
#ifndef dcci
template <typename... Args>
inline void dcci(Args &&...)
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
}
#endif
#ifndef mem_dsb_t
typedef int mem_dsb_t;
#endif
#ifndef dsb
#define dsb(kind)                                            \
    do {                                                     \
        (void)(kind);                                        \
        std::atomic_thread_fence(std::memory_order_seq_cst); \
    } while (0)
#endif
#ifndef ENTIRE_DATA_CACHE
#define ENTIRE_DATA_CACHE 0
#endif
#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif
#ifndef CACHELINE_OUT
#define CACHELINE_OUT 0
#endif
#ifndef DSB_DDR
#define DSB_DDR 0
#endif
#ifndef DSB_ALL
#define DSB_ALL 0
#endif
#ifndef DSB_UB
#define DSB_UB 0
#endif

constexpr pipe_t opPipeList[] = {};

#define aclFloat16ToFloat(x) (float)(x)
#define aclInit(x)
#define aclrtSetDevice(x)

enum
{
    ACL_MEMCPY_HOST_TO_DEVICE = 0,
    ACL_MEMCPY_DEVICE_TO_HOST = 1,
    ACL_MEMCPY_DEVICE_TO_DEVICE = 2,
};

#define aclFloat16ToFloat(x) ((float)(x))

static inline int aclrtMallocHost(void **p, size_t sz)
{
    assert(sz != 0 && "[PTO][CA] Constraint violated. Condition: %s. Hint: see docs/coding/debug.md\n");
    *p = malloc(sz);
    return 0;
}

#define set_flag(a, b, c)
#define wait_flag(a, b, c)
#define __cce_get_tile_ptr(x) x
#define set_mask_norm(...)
#define set_vector_mask(...)
inline uint64_t get_ctrl()
{
    return 0;
}
inline void set_ctrl(uint64_t)
{}
inline uint64_t sbitset1(uint64_t value, int)
{
    return value;
}
inline uint64_t sbitset0(uint64_t value, int)
{
    return value;
}

#include <pto/cpu/trace.hpp>

/* <Hccl> */
#define HcclHostBarrier(x, y)
#define CommMpiInit(x, y) (true)
#define CommMpiFinalize()
#define SKIP_IF_RANKS_LT(n)
static constexpr uint32_t HCCL_MAX_RANK_NUM = 64;

struct HcclRootInfo {};

struct HcclRootInfo {};

struct CommDeviceContext {
    uint64_t workSpace;
    uint64_t workSpaceSize;

    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[HCCL_MAX_RANK_NUM];
    uint64_t windowsOut[HCCL_MAX_RANK_NUM];
};
/* </Hccl> */

#define EVENT_ID0 0
#define EVENT_ID1 1
#define EVENT_ID2 2
#define EVENT_ID3 3

#define F16_MAX 65504.0f

namespace pto::cpu_sim {
struct RuntimeConfig {
    bool initialized = false;
    uint32_t device_id = 0;
    uint32_t num_cores = 1;
    bool trace_enabled = kInstructionTraceEnabled;
    std::filesystem::path trace_root = "cpu_sim_traces";
    uint64_t next_stream_id = 1;
    uint64_t next_launch_id = 0;
    std::mutex mutex;
};

struct StreamState {
    uint64_t id = 0;
};

inline const auto g_time_origin = std::chrono::steady_clock::now();

using SetExecutionContextHookFn = void (*)(uint32_t block_idx, uint32_t subblock_id, uint32_t subblock_dim);
using GetExecutionContextHookFn = void (*)(uint32_t *block_idx, uint32_t *subblock_id, uint32_t *subblock_dim);
using GetSharedStorageHookFn = void *(*)(std::string key, size_t size);
using GetTaskCookieHookFn = uint64_t (*)();
using GetSubblockIdInjectedHookFn = uint32_t (*)();
using GetPipeSharedStateInjectedHookFn = void *(*)(uint64_t pipe_key, size_t size);

inline void set_execution_context(uint32_t block_idx, uint32_t subblock_id, uint32_t subblock_dim);
inline void reset_execution_context();

inline GetSubblockIdInjectedHookFn injected_subblock_id_hook = nullptr;
inline GetPipeSharedStateInjectedHookFn injected_pipe_shared_state_hook = nullptr;

inline RuntimeConfig &runtime_config()
{
    static RuntimeConfig config;
    return config;
}

inline uint32_t ReadEnvU32(const char *name, uint32_t fallback)
{
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }
    return parsed == 0 ? fallback : static_cast<uint32_t>(parsed);
}

inline bool ReadEnvBool(const char *name, bool fallback)
{
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0) {
        return false;
    }
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0) {
        return true;
    }
    return fallback;
}

inline void InitializeRuntime()
{
    auto &config = runtime_config();
    std::scoped_lock lock(config.mutex);
    config.device_id = 0;
    config.num_cores = ReadEnvU32("PTO_CPU_SIM_NUM_CORES", 4);
    config.trace_enabled = kInstructionTraceEnabled && ReadEnvBool("PTO_CPU_SIM_TRACE_ENABLE", true);
    if (const char *trace_dir = std::getenv("PTO_CPU_SIM_TRACE_DIR"); trace_dir != nullptr && *trace_dir != '\0') {
        config.trace_root = trace_dir;
    } else {
        config.trace_root = "cpu_sim_traces";
    }
    config.next_stream_id = 1;
    if (config.trace_enabled) {
        std::filesystem::create_directories(config.trace_root);
    }
    config.initialized = true;
}

inline void ShutdownRuntime()
{
    auto &config = runtime_config();
    std::scoped_lock lock(config.mutex);
    config.initialized = false;
    config.next_stream_id = 1;
}

inline void EnsureRuntimeInitialized()
{
    if (!runtime_config().initialized) {
        InitializeRuntime();
    }
}

inline uint32_t GetConfiguredCoreCount()
{
    EnsureRuntimeInitialized();
    return runtime_config().num_cores;
}

inline bool IsTraceEnabled()
{
    EnsureRuntimeInitialized();
    return runtime_config().trace_enabled;
}

inline std::filesystem::path GetTraceRoot()
{
    EnsureRuntimeInitialized();
    return runtime_config().trace_root;
}

inline uint64_t NextLaunchId()
{
    auto &config = runtime_config();
    std::scoped_lock lock(config.mutex);
    return config.next_launch_id++;
}

inline std::filesystem::path CreateKernelTraceDir(const std::string &kernel_name)
{
    EnsureRuntimeInitialized();
    const auto dir = GetTraceRoot() / kernel_name / ("launch_" + std::to_string(NextLaunchId()));
    if (IsTraceEnabled()) {
        std::filesystem::create_directories(dir);
    }
    return dir;
}

struct KernelLaunchOptions {
    std::string kernel_name;
    uint32_t requested_cores = 0;
    uint32_t total_work_items = 0;
    uint32_t work_quantum = 1;
    uint32_t subblocks_per_block = 1;
    uint32_t explicit_block_count = 0;
    bool write_trace_files = true;
};

inline uint32_t ResolveActiveCoreCount(uint32_t requested_cores, uint32_t total_work_items = 0,
                                       uint32_t work_quantum = 1)
{
    const uint32_t configured = requested_cores == 0 ? GetConfiguredCoreCount() : requested_cores;
    uint32_t active = std::max<uint32_t>(1, configured);
    if (total_work_items == 0) {
        return active;
    }

    active = std::max<uint32_t>(1, std::min(active, total_work_items));
    const uint32_t quantum = std::max<uint32_t>(1, work_quantum);
    while (active > 1) {
        const uint32_t chunk = (total_work_items + active - 1) / active;
        if (chunk % quantum == 0) {
            break;
        }
        --active;
    }
    return active;
}

template <typename KernelFn>
inline void LaunchKernelMultiCore(const KernelLaunchOptions &options, aclrtStream stream, KernelFn &&kernel)
{
    (void)stream;
    EnsureRuntimeInitialized();

    const uint32_t subblocks_per_block = std::max<uint32_t>(1, options.subblocks_per_block);
    const uint32_t active_blocks =
        options.explicit_block_count != 0 ?
            options.explicit_block_count :
            ResolveActiveCoreCount(options.requested_cores, options.total_work_items, options.work_quantum);
    const uint32_t active_cores = active_blocks * subblocks_per_block;
    const bool trace_enabled = IsTraceEnabled() && options.write_trace_files;
    const std::filesystem::path trace_dir =
        trace_enabled ? CreateKernelTraceDir(options.kernel_name) : std::filesystem::path{};

    std::vector<std::thread> workers;
    workers.reserve(active_cores);
    std::vector<std::string> trace_chunks(active_cores);
    std::exception_ptr first_error;
    std::mutex error_mutex;

    for (uint32_t core_id = 0; core_id < active_cores; ++core_id) {
        workers.emplace_back([&, core_id]() {
            try {
                const uint32_t block_idx = core_id / subblocks_per_block;
                const uint32_t subblock_id = core_id % subblocks_per_block;
                ResetInstructionTrace();
                reset_execution_context();
                set_execution_context(block_idx, subblock_id, subblocks_per_block);
                kernel();
                if (trace_enabled) {
                    std::ostringstream out;
                    DumpInstructionTraceJson(out);
                    trace_chunks[core_id] = out.str();
                }
                reset_execution_context();
            } catch (...) {
                std::scoped_lock lock(error_mutex);
                if (first_error == nullptr) {
                    first_error = std::current_exception();
                }
            }
        });
    }

    for (auto &worker : workers) {
        worker.join();
    }

    if (first_error != nullptr) {
        std::rethrow_exception(first_error);
    }

    if (trace_enabled) {
        std::ofstream combined(trace_dir / "trace.jsonl", std::ios::trunc);
        for (const auto &chunk : trace_chunks) {
            combined << chunk;
        }
    }
}

inline StreamState *ToStreamState(aclrtStream stream)
{
    return reinterpret_cast<StreamState *>(stream);
}

inline SetExecutionContextHookFn ResolveSetExecutionContextHook()
{
    static auto hook =
        reinterpret_cast<SetExecutionContextHookFn>(dlsym(RTLD_DEFAULT, "pto_cpu_sim_set_execution_context"));
    return hook;
}

inline GetExecutionContextHookFn ResolveExecutionContextHook()
{
    static auto hook =
        reinterpret_cast<GetExecutionContextHookFn>(dlsym(RTLD_DEFAULT, "pto_cpu_sim_get_execution_context"));
    return hook;
}

inline GetSharedStorageHookFn ResolveSharedStorageHook()
{
    static auto hook = reinterpret_cast<GetSharedStorageHookFn>(dlsym(RTLD_DEFAULT, "pto_cpu_sim_get_shared_storage"));
    return hook;
}

inline GetTaskCookieHookFn ResolveTaskCookieHook()
{
    static auto hook = reinterpret_cast<GetTaskCookieHookFn>(dlsym(RTLD_DEFAULT, "pto_cpu_sim_get_task_cookie"));
    return hook;
}

inline GetSubblockIdInjectedHookFn ResolveSubblockIdHook()
{
    static auto hook = reinterpret_cast<GetSubblockIdInjectedHookFn>(dlsym(RTLD_DEFAULT, "pto_sim_get_subblock_id"));
    return hook;
}

inline GetPipeSharedStateInjectedHookFn ResolvePipeSharedStateHook()
{
    static auto hook =
        reinterpret_cast<GetPipeSharedStateInjectedHookFn>(dlsym(RTLD_DEFAULT, "pto_sim_get_pipe_shared_state"));
    return hook;
}

struct ExecutionContext {
    uint32_t block_idx = 0;
    uint32_t subblock_id = 0;
    uint32_t subblock_dim = 1;
    uint64_t task_cookie = 0;
};

inline thread_local ExecutionContext execution_context{};

inline void register_hooks(void *get_subblock_id, void *get_pipe_shared_state)
{
    injected_subblock_id_hook = reinterpret_cast<GetSubblockIdInjectedHookFn>(get_subblock_id);
    injected_pipe_shared_state_hook = reinterpret_cast<GetPipeSharedStateInjectedHookFn>(get_pipe_shared_state);
}

inline void set_execution_context(uint32_t block_idx, uint32_t subblock_id, uint32_t subblock_dim = 1)
{
    execution_context.block_idx = block_idx;
    execution_context.subblock_id = subblock_id;
    execution_context.subblock_dim = (subblock_dim == 0) ? 1 : subblock_dim;
    if (auto hook = ResolveSetExecutionContextHook(); hook != nullptr) {
        hook(execution_context.block_idx, execution_context.subblock_id, execution_context.subblock_dim);
    }
}

inline void reset_execution_context()
{
    execution_context = {};
}

inline void set_task_cookie(uint64_t task_cookie)
{
    execution_context.task_cookie = task_cookie;
}

class ScopedExecutionContext {
public:
    ScopedExecutionContext(uint32_t block_idx, uint32_t subblock_id, uint32_t subblock_dim = 1)
        : saved_(execution_context)
    {
        set_execution_context(block_idx, subblock_id, subblock_dim);
    }

    ~ScopedExecutionContext()
    {
        execution_context = saved_;
    }

private:
    ExecutionContext saved_{};
};
} // namespace pto::cpu_sim

namespace cce {
template <typename... Args>
inline int printf(const char *fmt, Args... args)
{
    return std::printf(fmt, args...);
}
} // namespace cce

inline int aclInit(const char *)
{
    pto::cpu_sim::InitializeRuntime();
    return 0;
}

inline int aclrtSetDevice(int device_id)
{
    auto &config = pto::cpu_sim::runtime_config();
    std::scoped_lock lock(config.mutex);
    config.device_id = static_cast<uint32_t>(std::max(0, device_id));
    return 0;
}

inline int aclrtCreateStream(aclrtStream *stream)
{
    pto::cpu_sim::EnsureRuntimeInitialized();
    auto &config = pto::cpu_sim::runtime_config();
    auto *state = new pto::cpu_sim::StreamState();
    {
        std::scoped_lock lock(config.mutex);
        state->id = config.next_stream_id++;
    }
    *stream = reinterpret_cast<aclrtStream>(state);
    return 0;
}

inline int aclrtMalloc(void **p, size_t sz, int)
{
    return aclrtMallocHost(p, sz);
}

inline int aclrtMemcpy(void *dst, size_t sz_dst, const void *src, size_t sz_src, int)
{
    std::memcpy(dst, src, std::min(sz_dst, sz_src));
    return 0;
}

inline int aclrtMemset(void *dst, size_t dstSize, int value, size_t count)
{
    constexpr int ACL_SUCCESS = 0;
    constexpr int ACL_ERROR_GE_PARAM_INVALID = 145000;

    if (count == 0) {
        return ACL_SUCCESS;
    }
    if (dst == nullptr || count > dstSize) {
        return ACL_ERROR_GE_PARAM_INVALID;
    }
    std::fill_n(reinterpret_cast<uint8_t *>(dst), count, static_cast<uint8_t>(value));
    return ACL_SUCCESS;
}

inline int aclrtSynchronizeStream(aclrtStream)
{
    return 0;
}

inline int aclrtFree(void *p)
{
    free(p);
    return 0;
}

inline int aclrtFreeHost(void *p)
{
    free(p);
    return 0;
}

inline int aclrtDestroyStream(aclrtStream stream)
{
    delete pto::cpu_sim::ToStreamState(stream);
    return 0;
}

inline int aclrtResetDevice(int)
{
    return 0;
}

inline int aclFinalize()
{
    pto::cpu_sim::ShutdownRuntime();
    return 0;
}

inline uint32_t get_block_idx()
{
    if (auto hook = pto::cpu_sim::ResolveExecutionContextHook(); hook != nullptr) {
        uint32_t block_idx = 0;
        uint32_t subblock_id = 0;
        uint32_t subblock_dim = 1;
        hook(&block_idx, &subblock_id, &subblock_dim);
        return block_idx;
    }
    return pto::cpu_sim::execution_context.block_idx;
}

inline uint32_t get_subblockid()
{
    if (pto::cpu_sim::injected_subblock_id_hook != nullptr) {
        return pto::cpu_sim::injected_subblock_id_hook();
    }
    if (auto hook = pto::cpu_sim::ResolveSubblockIdHook(); hook != nullptr) {
        return hook();
    }
    if (auto hook = pto::cpu_sim::ResolveExecutionContextHook(); hook != nullptr) {
        uint32_t block_idx = 0;
        uint32_t subblock_id = 0;
        uint32_t subblock_dim = 1;
        hook(&block_idx, &subblock_id, &subblock_dim);
        return subblock_id;
    }
    return pto::cpu_sim::execution_context.subblock_id;
}

inline uint32_t get_subblockdim()
{
    if (auto hook = pto::cpu_sim::ResolveExecutionContextHook(); hook != nullptr) {
        uint32_t block_idx = 0;
        uint32_t subblock_id = 0;
        uint32_t subblock_dim = 1;
        hook(&block_idx, &subblock_id, &subblock_dim);
        return subblock_dim;
    }
    return pto::cpu_sim::execution_context.subblock_dim;
}

inline uint32_t get_block_num()
{
    return pto::cpu_sim::GetConfiguredCoreCount();
}

inline uint32_t get_coreid()
{
    return get_block_idx();
}

inline uint64_t get_task_cookie()
{
    if (auto hook = pto::cpu_sim::ResolveTaskCookieHook(); hook != nullptr) {
        return hook();
    }
    return pto::cpu_sim::execution_context.task_cookie;
}

template <typename T>
struct is_event : std::false_type {};

template <typename... Ts>
inline constexpr bool all_events_v = (is_event<Ts>::value && ...);

namespace pto {
template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
inline void SYNCALL_IMPL()
{
    (void)CoreType;
}

template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
inline void SYNCALL_SOFT_IMPL(int32_t *gmWorkspace, int32_t *ubWorkspace, int32_t usedCores)
{
    (void)CoreType;
    (void)gmWorkspace;
    (void)ubWorkspace;
    (void)usedCores;
}

inline void SYNCALL_SOFT_AIC_IMPL(int32_t *gmWorkspace, int32_t *l1Workspace, int32_t usedCores)
{
    (void)gmWorkspace;
    (void)l1Workspace;
    (void)usedCores;
}

template <SyncCoreType CoreType = SyncCoreType::Mix>
inline void SYNCALL_SOFT_MIX_IMPL(int32_t *gmWorkspace, int32_t *ubWorkspace, int32_t *l1Workspace, int32_t usedCores)
{
    (void)CoreType;
    (void)gmWorkspace;
    (void)ubWorkspace;
    (void)l1Workspace;
    (void)usedCores;
}
} // namespace pto

#endif
