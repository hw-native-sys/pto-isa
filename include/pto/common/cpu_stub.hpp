/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPUSTUB_HPP
#define PTO_CPUSTUB_HPP

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <type_traits>

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
#define __tf__

typedef void *aclrtStream;
typedef int pipe_t;
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

constexpr pipe_t opPipeList[] = {};

#define aclFloat16ToFloat(x) ((float)(x)
#define aclInit(x)
#define aclrtSetDevice(x)

#define aclrtCreateStream(x)

static inline void aclrtMallocHost(void **p, size_t sz)
{
    assert(sz != 0 && "[PTO][CA] Constraint violated. Condition: %s. Hint: see docs/coding/debug.md\n");
    *p = malloc(sz);
}

#define aclrtMalloc(a, b, c) aclrtMallocHost(a, b)

#define aclrtMemcpy(dst, sz_dst, src, sz_src, type)                              \
    {                                                                            \
        for (size_t i = 0; i < sz_src && i < sz_dst; i++)                        \
            reinterpret_cast<char *>(dst)[i] = reinterpret_cast<char *>(src)[i]; \
    }

#define aclrtSynchronizeStream(x)
#define aclrtFree(x) free(x)
#define aclrtFreeHost(x) free(x)
#define aclrtDestroyStream(x)
#define aclrtResetDevice(x)
#define aclFinalize(x)
#define set_flag(a, b, c)
#define wait_flag(a, b, c)
#define __cce_get_tile_ptr(x) x
#define set_mask_norm(...)
#define set_vector_mask(...)

typedef int event_t;
#define EVENT_ID0 0

namespace pto::cpu_sim {
using GetSubblockIdHookFn = uint32_t (*)();
using GetPipeSharedStateHookFn = void *(*)(uint64_t pipe_key, size_t size);

inline GetSubblockIdHookFn &get_subblock_id_hook()
{
    static GetSubblockIdHookFn fn = nullptr;
    return fn;
}

inline GetPipeSharedStateHookFn &get_pipe_shared_state_hook()
{
    static GetPipeSharedStateHookFn fn = nullptr;
    return fn;
}

struct ExecutionContext {
    uint32_t block_idx = 0;
    uint32_t subblock_id = 0;
    uint32_t subblock_dim = 1;
};

inline thread_local ExecutionContext execution_context{};

inline void set_execution_context(uint32_t block_idx, uint32_t subblock_id, uint32_t subblock_dim = 1)
{
    execution_context.block_idx = block_idx;
    execution_context.subblock_id = subblock_id;
    execution_context.subblock_dim = (subblock_dim == 0) ? 1 : subblock_dim;
}

inline void reset_execution_context()
{
    execution_context = {};
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

/**
 * Register simulation hook function pointers for this SO.
 *
 * Called by the sim platform (DeviceRunner) after dlopen'ing each kernel SO.
 * Each SO has its own copy of the inline static function pointers, so every
 * SO that uses TPUSH/TPOP must be registered.
 *
 * Pass nullptr to clear hooks (fallback to thread_local execution context).
 */
extern "C" __attribute__((visibility("default"), used)) inline void pto_sim_register_hooks(void *get_subblock_id,
                                                                                           void *get_pipe_shared_state)
{
    pto::cpu_sim::get_subblock_id_hook() = reinterpret_cast<pto::cpu_sim::GetSubblockIdHookFn>(get_subblock_id);
    pto::cpu_sim::get_pipe_shared_state_hook() =
        reinterpret_cast<pto::cpu_sim::GetPipeSharedStateHookFn>(get_pipe_shared_state);
}

/**
 * Set the execution context (block_idx, subblock_id, subblock_dim) for this SO.
 *
 * Each dlopen'd kernel SO has its own thread_local execution_context. The sim
 * platform must dlsym this symbol after dlopen and call it before each kernel
 * dispatch so that get_block_idx() / get_subblockdim() return correct values.
 */
extern "C" __attribute__((visibility("default"), used)) inline void pto_sim_set_execution_context(uint32_t block_idx,
                                                                                                  uint32_t subblock_id,
                                                                                                  uint32_t subblock_dim)
{
    pto::cpu_sim::set_execution_context(block_idx, subblock_id, subblock_dim);
}

inline uint32_t get_block_idx()
{
    return pto::cpu_sim::execution_context.block_idx;
}

inline uint32_t get_subblockid()
{
    if (auto hook = pto::cpu_sim::get_subblock_id_hook(); hook != nullptr) {
        return hook();
    }
    return pto::cpu_sim::execution_context.subblock_id;
}

inline uint32_t get_subblockdim()
{
    return pto::cpu_sim::execution_context.subblock_dim;
}

namespace pto {
template <typename T>
struct is_event : std::false_type {};

template <typename... Ts>
inline constexpr bool all_events_v = (is_event<Ts>::value && ...);
} // namespace pto
#endif
