/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_TRACE_HPP
#define PTO_MOCKER_TRACE_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <pto/costmodel/arch_config.hpp>

namespace pto::mocker {

using Arg = uint64_t;

struct CcePipeTraceState {
    std::vector<std::size_t> queue;
    uint64_t last_cce_tail = 0;
    bool has_pending_tail = false;
};

inline constexpr std::size_t kPipeKeyCount = static_cast<std::size_t>(evaluator::PipeKey::COUNT);

struct CceCallRecord {
    std::string name;
    std::vector<Arg> args;
    uint64_t cycles = 0;
};

struct PtoInstrRecord {
    std::string name;
    std::vector<CceCallRecord> cce_calls;
    uint64_t total_cycles = 0;
};

struct TraceState {
    std::vector<PtoInstrRecord> executed_pto;
    std::vector<std::size_t> active_pto_stack;
    std::array<CcePipeTraceState, kPipeKeyCount> cce_pipe_traces;
};

inline thread_local TraceState g_trace_state;

inline void ResetTrace()
{
    g_trace_state = {};
}

inline TraceState &GetMutableTrace()
{
    return g_trace_state;
}

inline const TraceState &GetTrace()
{
    return g_trace_state;
}

inline uint64_t GetLastPtoInstrCycles()
{
    const auto &trace = g_trace_state;
    return trace.executed_pto.empty() ? 0 : trace.executed_pto.back().total_cycles;
}

inline constexpr std::size_t ToPipeIndex(evaluator::PipeKey pipe)
{
    return static_cast<std::size_t>(pipe);
}

inline CcePipeTraceState &GetPipeTrace(TraceState &trace, evaluator::PipeKey pipe)
{
    return trace.cce_pipe_traces[ToPipeIndex(pipe)];
}

inline const CcePipeTraceState &GetPipeTrace(const TraceState &trace, evaluator::PipeKey pipe)
{
    return trace.cce_pipe_traces[ToPipeIndex(pipe)];
}

template <typename T>
inline constexpr bool kUnsupportedTraceType = false;

template <typename T>
inline uint64_t ToTraceValue(T value)
{
    using Decayed = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_pointer_v<Decayed>) {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(value));
    } else if constexpr (std::is_enum_v<Decayed>) {
        using Underlying = std::underlying_type_t<Decayed>;
        return static_cast<uint64_t>(static_cast<Underlying>(value));
    } else if constexpr (std::is_integral_v<Decayed>) {
        return static_cast<uint64_t>(value);
    } else if constexpr (std::is_floating_point_v<Decayed> && sizeof(Decayed) == sizeof(uint32_t)) {
        return static_cast<uint64_t>(std::bit_cast<uint32_t>(value));
    } else if constexpr (std::is_floating_point_v<Decayed> && sizeof(Decayed) == sizeof(uint64_t)) {
        return std::bit_cast<uint64_t>(value);
    } else if constexpr (sizeof(Decayed) == sizeof(uint16_t) && !std::is_integral_v<Decayed> &&
                         !std::is_enum_v<Decayed> && !std::is_pointer_v<Decayed>) {
        // Handles _Float16 / __fp16 / half which may not satisfy std::is_floating_point_v
        return static_cast<uint64_t>(std::bit_cast<uint16_t>(value));
    } else {
        static_assert(kUnsupportedTraceType<Decayed>, "Unsupported trace argument type.");
        return 0;
    }
}

inline bool IsPipeQueueEmpty(evaluator::PipeKey pipe)
{
    const auto &trace = g_trace_state;
    if (trace.active_pto_stack.empty()) {
        return true;
    }
    return GetPipeTrace(trace, pipe).queue.empty();
}

inline void SetLastCceTail(evaluator::PipeKey pipe, uint64_t tail)
{
    auto &trace = g_trace_state;
    if (trace.active_pto_stack.empty()) {
        return;
    }
    auto &pipe_trace = GetPipeTrace(trace, pipe);
    pipe_trace.last_cce_tail = tail;
    pipe_trace.has_pending_tail = (tail != 0);
}

inline void FlushPendingTail(evaluator::PipeKey pipe)
{
    auto &trace = g_trace_state;
    if (trace.active_pto_stack.empty()) {
        return;
    }

    auto &pipe_trace = GetPipeTrace(trace, pipe);
    if (pipe_trace.has_pending_tail) {
        trace.executed_pto[trace.active_pto_stack.back()].total_cycles += pipe_trace.last_cce_tail;
        pipe_trace.last_cce_tail = 0;
        pipe_trace.has_pending_tail = false;
    }
    pipe_trace.queue.clear();
}

inline void FlushAllPendingTails()
{
    for (std::size_t i = 0; i < kPipeKeyCount; ++i) {
        FlushPendingTail(static_cast<evaluator::PipeKey>(i));
    }
}

inline void BeginPtoInstr(std::string_view name)
{
    auto &trace = g_trace_state;
    if (trace.active_pto_stack.empty()) {
        trace.executed_pto.push_back(PtoInstrRecord{std::string(name), {}, 0});
        trace.cce_pipe_traces = {};
        trace.active_pto_stack.push_back(trace.executed_pto.size() - 1);
    } else {
        // Collapse nested PTO helper calls into the current top-level PTO record.
        trace.active_pto_stack.push_back(trace.active_pto_stack.back());
    }
}

inline void EndPtoInstr()
{
    auto &stack = g_trace_state.active_pto_stack;
    if (!stack.empty()) {
        if (stack.size() == 1) {
            FlushAllPendingTails();
        }
        stack.pop_back();
    }
}

inline constexpr std::size_t kInvalidCceCallIndex = static_cast<std::size_t>(-1);

template <typename... Args>
inline std::size_t AppendCceCall(std::string_view name, uint64_t cycles, Args &&... args)
{
    auto &trace = g_trace_state;
    if (trace.active_pto_stack.empty()) {
        return kInvalidCceCallIndex;
    }

    CceCallRecord call;
    call.name = std::string(name);
    call.cycles = cycles;
    call.args.reserve(sizeof...(Args));
    if constexpr (sizeof...(Args) > 0) {
        (call.args.push_back(ToTraceValue(std::forward<Args>(args))), ...);
    }

    auto &pto = trace.executed_pto[trace.active_pto_stack.back()];
    pto.total_cycles += cycles;
    pto.cce_calls.push_back(std::move(call));
    return pto.cce_calls.size() - 1;
}

template <typename... Args>
inline void RecordCceCall(std::string_view name, uint64_t cycles, Args &&... args)
{
    (void)AppendCceCall(name, cycles, std::forward<Args>(args)...);
}

template <typename... Args>
inline void RecordCceCall(evaluator::PipeKey pipe, std::string_view name, uint64_t cycles, Args &&... args)
{
    auto &trace = g_trace_state;
    const std::size_t call_index = AppendCceCall(name, cycles, std::forward<Args>(args)...);
    if (call_index == kInvalidCceCallIndex) {
        return;
    }
    GetPipeTrace(trace, pipe).queue.push_back(call_index);
}

class PtoInstrScope {
public:
    explicit PtoInstrScope(std::string_view name)
    {
        BeginPtoInstr(name);
    }

    ~PtoInstrScope()
    {
        EndPtoInstr();
    }

    PtoInstrScope(const PtoInstrScope &) = delete;
    PtoInstrScope &operator=(const PtoInstrScope &) = delete;
};

} // namespace pto::mocker

#endif
