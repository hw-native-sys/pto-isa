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
PTO_INTERNAL static constexpr pipe_t GetPipeByOpForKirin9030()
{
    return OpPipeEntry<OpCode>::pipe;
}

// single pipeline wait
template <Op OpCode>
PTO_INTERNAL void TSYNC_IMPL()
{
#ifndef __PTO_AUTO__
    constexpr pipe_t pipe = GetPipeByOpForKirin9030<OpCode>();
    PTO_STATIC_ASSERT(
        (pipe == PIPE_M) || (pipe == PIPE_MTE1) || (pipe == PIPE_MTE2) || (pipe == PIPE_MTE3) || (pipe == PIPE_ALL) ||
            (pipe == PIPE_FIX),
        "Single Op TSYNC only supports M / MTE1 / MTE2 / MTE3 / ALL / FIX pipeline.");
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
        return GetPipeByOpForKirin9030<op>();
    }
#ifndef __PTO_AUTO__
    static constexpr bool isValidBarrierPipe =
        ((Base::dstPipe == PIPE_M) || (Base::dstPipe == PIPE_MTE1) || (Base::dstPipe == PIPE_MTE2) ||
         (Base::dstPipe == PIPE_MTE3) || (Base::dstPipe == PIPE_FIX));

    PTO_STATIC_ASSERT(Base::srcPipe != PIPE_ALL, "SrcOp are invalid.");
    PTO_STATIC_ASSERT(Base::dstPipe != PIPE_ALL, "DstOp are invalid.");
#endif

    PTO_INTERNAL Event& InitAddrImpl(uint64_t fftsAddr) { return *this; }

    template <uint8_t CrossCoreId = 0xff>
    PTO_INTERNAL Event& WaitImpl()
    {
#ifndef __PTO_AUTO__
        if constexpr (Base::isSamePipe) {
            if constexpr (isValidBarrierPipe) {
                pipe_barrier((pipe_t)Base::srcPipe);
            }
        } else {
            wait_flag((pipe_t)Base::srcPipe, (pipe_t)Base::dstPipe, Base::token);
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
            set_flag((pipe_t)Base::srcPipe, (pipe_t)Base::dstPipe, Base::token);
        }
#endif
        return *this;
    }
};

} // namespace pto
#endif
