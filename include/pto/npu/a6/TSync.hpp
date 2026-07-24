/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSYNC_HPP
#define TSYNC_HPP
#include <pto/common/type.hpp>
#include <pto/common/event.hpp>

namespace pto {
template <Op OpCode>
PTO_INTERNAL static constexpr pipe_t GetPipeByOpForA6()
{
    return OpPipeEntry<OpCode>::pipe;
}

// A6 compiler restricts set_flag/wait_flag pipe parameters to values {0, 2, 3, 4, 5, 10}.
// Pipes outside this range (e.g. PIPE_V=1, PIPE_FIX=7) must use pipe_barrier instead.
template <pipe_t P>
PTO_INTERNAL static constexpr bool IsValidFlagPipe()
{
    return (P == PIPE_S) || (P == PIPE_MTE1) || (P == PIPE_MTE2) || (P == PIPE_MTE3) || (P == PIPE_FIX);
}

// single pipeline wait
template <Op OpCode>
PTO_INTERNAL void TSYNC_IMPL()
{
#ifndef __PTO_AUTO__
    constexpr pipe_t pipe = GetPipeByOpForA6<OpCode>();
    PTO_STATIC_ASSERT(
        (pipe == PIPE_MTE2) || (pipe == PIPE_MTE3) || (pipe == PIPE_ALL),
        "Single Op TSYNC only supports MTE2 / MTE3 / ALL pipeline.");
    pipe_barrier((pipe_t)pipe);
#endif
}

template <Op SrcOp, Op DstOp, bool AutoToken = true, event_t EventID = EVENT_ID0>
struct Event : EventBase<Event<SrcOp, DstOp, AutoToken, EventID>, SrcOp, DstOp, AutoToken, EventID> {
    using Base = EventBase<Event, SrcOp, DstOp, AutoToken, EventID>;
    using Base::operator=;

    template <Op op>
    PTO_INTERNAL static constexpr pipe_t GetPipeByOp()
    {
        return GetPipeByOpForA6<op>();
    }
#ifndef __PTO_AUTO__
    // Whether both srcPipe and dstPipe are valid for set_flag/wait_flag on A6.
    static constexpr bool isValidFlagPair = IsValidFlagPipe<Base::srcPipe>() && IsValidFlagPipe<Base::dstPipe>();

    PTO_STATIC_ASSERT(Base::srcPipe != PIPE_ALL, "SrcOp are invalid.");
    PTO_STATIC_ASSERT(Base::dstPipe != PIPE_ALL, "DstOp are invalid.");
#endif

    PTO_INTERNAL Event& InitAddrImpl(uint64_t fftsAddr) { return *this; }

    template <uint8_t CrossCoreId = 0xff>
    PTO_INTERNAL Event& WaitImpl()
    {
#ifndef __PTO_AUTO__
        if constexpr (Base::isSamePipe) {
            pipe_barrier((pipe_t)Base::srcPipe);
        } else if constexpr (isValidFlagPair) {
            wait_flag((pipe_t)Base::srcPipe, (pipe_t)Base::dstPipe, Base::token);
        } else {
            // Pipes not supported by set_flag/wait_flag (e.g. PIPE_V) — use full barrier.
            pipe_barrier(PIPE_ALL);
        }
#endif
        return *this;
    }

    PTO_INTERNAL Event() = default;
    PTO_INTERNAL Event(RecordEvent re) : Base(re) {}

    template <uint8_t CrossCoreId = 0xff>
    PTO_INTERNAL Event& InitImpl()
    {
#ifndef __PTO_AUTO__
        if constexpr (!Base::isSamePipe) {
            if constexpr (isValidFlagPair) {
                set_flag((pipe_t)Base::srcPipe, (pipe_t)Base::dstPipe, Base::token);
            }
            // else: no-op — synchronization handled via pipe_barrier in WaitImpl.
        }
#endif
        return *this;
    }
};

} // namespace pto
#endif
